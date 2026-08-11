#!/usr/bin/env python3
"""Per-chunk-position ("temporal") action error diagnostic for the early-action-weighted-loss
experiment (`combined65_early_weight_v1`).

Motivation
----------
The seed-sweep diagnostic (`scripts/sweep_grid35_first_action_seed.py`) only keeps chunk index 0
(the action actually delivered by `select_action()`), discarding the rest of the model's predicted
50-step chunk. This script reuses the exact same infrastructure (same reference Shadow observation,
same checkpoint-loading path, same 20 fixed seeds, same nearest-training-demo GT lookup) but keeps
the **entire** predicted chunk `policy.predict_action_chunk()` already computes - no extra
inference cost, since euler-integrating the flow-matching chunk produces all `chunk_size` steps in
one call regardless of how many are kept afterward.

Ground truth for chunk position `k` is the nearest-training-demo trajectory's own action at
`min(matched_frame + k, episode_end - 1)` - the exact same `chunk[k] == raw_action[min(t+k,
episode_end-1)]` alignment law established in
`scripts/verify_smolvla_training_target_alignment.py`, applied to the nearest real demo instead of
a live rollout (there is no live closed-loop trajectory here, by design - no real hardware writes).

Buckets reported (matching the experiment's own hypothesis - "early chunk positions get pulled
toward the chunk's large mean/future-movement magnitude"):
  - step 0      (what `select_action()` actually delivers)
  - steps 1-2   (the "1~2" early band the loss upweights in this experiment)
  - steps 3-49  ("the rest of the chunk")

Read-only: no dataset/checkpoint writes, no Safety Gate threshold changes, no real-robot writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_grid35_first_action_pipeline import JOINTS, safety_threshold_check  # noqa: E402
from scripts.sweep_grid35_first_action_seed import (  # noqa: E402
    build_batch,
    load_images,
    load_policy_bundle,
    load_shadow_observation,
)

KEY_JOINTS = ["shoulder_lift", "elbow_flex"]
DEFAULT_SHADOW_REPORT = PROJECT_ROOT / "reports" / "grid35_v2_shadow_T01" / "shadow_20260808_211555.json"
DEFAULT_SAFETY_GATE_YAML = PROJECT_ROOT / "configs" / "safety_gate.yaml"
BUCKETS = {"step0": [0], "step1_2": [1, 2], "step3_plus": None}  # None -> filled with range(3, chunk_size)


def find_nearest_demo_full_chunk(dataset_root: Path, shadow_state: dict[str, float], chunk_size: int) -> dict[str, Any]:
    """Nearest-training-demo lookup (same L2-over-state-space search as
    `nearest_frame_trajectory_analysis`), but returns the matched trajectory's full `chunk_size`-
    step action array (absolute, same units as the dataset) instead of only a handful of horizons.
    """
    files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    shadow_arr = np.array([shadow_state[j] for j in JOINTS])

    best = None  # (dist, episode_index, frame_index, action_chunk (chunk_size, 6))
    for fp in files:
        df = pd.read_parquet(fp, columns=["action", "observation.state", "frame_index", "episode_index"])
        for ep, ep_df in df.groupby("episode_index"):
            ep_df = ep_df.sort_values("frame_index")
            state = np.stack(ep_df["observation.state"].to_numpy())
            action = np.stack(ep_df["action"].to_numpy())
            dist = np.linalg.norm(state - shadow_arr[None, :], axis=1)
            idx = int(np.argmin(dist))
            d = float(dist[idx])
            if best is None or d < best[0]:
                k_idx = np.arange(chunk_size)
                frame_idx = np.minimum(idx + k_idx, len(action) - 1)
                gt_chunk = action[frame_idx]  # (chunk_size, 6), absolute units
                best = (d, int(ep), idx, gt_chunk)

    dist, ep, idx, gt_chunk = best
    return {"episode": ep, "frame": idx, "l2_dist_deg": dist, "gt_chunk": gt_chunk}


def run_full_chunk_sweep(bundle: dict[str, Any], state: dict, images: dict, task: str, seeds: list[int], chunk_size: int) -> list[np.ndarray]:
    """Same per-seed loop as `run_seed_sweep_with_bundle`, but returns the FULL predicted chunk
    (chunk_size, action_dim) per seed instead of discarding everything past index 0."""
    torch = bundle["torch"]
    policy = bundle["policy"]
    device = bundle["device"]
    preprocessor = bundle["preprocessor"]
    postprocessor = bundle["postprocessor"]

    batch = build_batch(state, images, task, device, torch)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for seed in seeds:
            processed = preprocessor(batch)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            raw_chunk = policy.predict_action_chunk(processed)  # (1, chunk_size, action_dim)
            postproc_chunk = postprocessor(raw_chunk)
            chunk = postproc_chunk.detach().to("cpu").numpy()[0][:chunk_size]  # (chunk_size, 6)
            chunks.append(chunk)
    return chunks


def bucket_indices(chunk_size: int) -> dict[str, list[int]]:
    b = dict(BUCKETS)
    b["step3_plus"] = list(range(3, chunk_size))
    return {k: [i for i in v if i < chunk_size] for k, v in b.items()}


def compute_temporal_error(chunks: list[np.ndarray], gt_chunk: np.ndarray) -> dict[str, Any]:
    chunk_size = gt_chunk.shape[0]
    buckets = bucket_indices(chunk_size)
    chunks_arr = np.stack(chunks)  # (n_seeds, chunk_size, 6)
    abs_err = np.abs(chunks_arr - gt_chunk[None, :, :])  # (n_seeds, chunk_size, 6)

    result: dict[str, Any] = {"n_seeds": len(chunks), "chunk_size": chunk_size, "buckets": {}}
    for bucket_name, idxs in buckets.items():
        if not idxs:
            continue
        sub = abs_err[:, idxs, :]  # (n_seeds, len(idxs), 6)
        per_joint = {j: float(sub[:, :, i].mean()) for i, j in enumerate(JOINTS)}
        result["buckets"][bucket_name] = {
            "chunk_indices": idxs,
            "mae_overall": float(sub.mean()),
            "mae_per_joint": per_joint,
            "mae_key_joints_mean": float(np.mean([per_joint[j] for j in KEY_JOINTS])),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shadow-report", type=Path, default=DEFAULT_SHADOW_REPORT)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=19)
    parser.add_argument("--label", default=None, help="free-text label stored in the JSON (e.g. checkpoint step / experiment name)")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_end + 1))

    print(f"[temporal] loading reference Shadow observation from {args.shadow_report}")
    obs = load_shadow_observation(args.shadow_report.resolve())
    images = load_images(obs["camera_frame_paths"])

    print(f"[temporal] loading checkpoint {args.checkpoint}")
    bundle = load_policy_bundle(args.checkpoint.resolve())
    chunk_size = bundle["policy"].config.chunk_size

    print(f"[temporal] finding nearest training demo + its full {chunk_size}-step GT chunk from {args.dataset_root}")
    nearest = find_nearest_demo_full_chunk(args.dataset_root.resolve(), obs["state"], chunk_size)
    print(f"[temporal]   matched episode={nearest['episode']} frame={nearest['frame']} l2_dist={nearest['l2_dist_deg']:.3f}deg")

    print(f"[temporal] running {len(seeds)} seeded full-chunk inferences (seeds {seeds[0]}..{seeds[-1]})")
    chunks = run_full_chunk_sweep(bundle, obs["state"], images, args.task, seeds, chunk_size)

    result = compute_temporal_error(chunks, nearest["gt_chunk"])
    thresholds = safety_threshold_check(DEFAULT_SAFETY_GATE_YAML)["would_clamp_threshold_deg"]

    payload = {
        "checkpoint": str(args.checkpoint),
        "label": args.label,
        "dataset_root": str(args.dataset_root),
        "shadow_report": str(args.shadow_report),
        "task": args.task,
        "seeds": seeds,
        "nearest_demo_match": {"episode": nearest["episode"], "frame": nearest["frame"], "l2_dist_deg": nearest["l2_dist_deg"]},
        "would_clamp_threshold_deg": thresholds,
        **result,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "temporal_chunk_error.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f"[temporal] wrote {json_path}")

    print()
    for bucket_name, b in result["buckets"].items():
        print(f"[temporal] {bucket_name} (idx {b['chunk_indices'][0]}..{b['chunk_indices'][-1]}): "
              f"overall MAE={b['mae_overall']:.3f}deg, key-joint MAE={b['mae_key_joints_mean']:.3f}deg "
              f"(shoulder_lift={b['mae_per_joint']['shoulder_lift']:.3f}, elbow_flex={b['mae_per_joint']['elbow_flex']:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
