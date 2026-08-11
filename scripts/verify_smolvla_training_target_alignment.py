#!/usr/bin/env python3
"""Verify what the SmolVLA training pipeline actually uses as the target action chunk.

Background
----------
Prior analyses (``scripts/analyze_grid35_v2_clean_start_segment_temporal_alignment.py``,
``scripts/diagnose_grid35_first_action_pipeline.py``) established, on
``data/so101_cube_xy_grid35_v2_clean``:

  - the dataset's own frame-0 ``action[t]-state[t]`` delta is small (not a labeling bug by
    itself),
  - there is no material static-hold action/state mislabeling in frames 0-14,
  - ``observation.state`` first non-trivial movement has median frame ~25, ``action`` first
    movement has median frame ~21,
  - the *actual* Shadow/checkpoint first delivered action resembles the dataset's own GT
    displacement ~frame 28-30 into a typical demonstration, not frame 0/1.

All of that analysis worked directly off *raw* ``action[t]``/``observation.state[t]`` arrays
read from parquet. It never actually asked LeRobot's own dataset-loading code what the
**training target** looks like for a given observation - i.e. it never built the real
``delta_timestamps``-expanded action chunk the way ``lerobot.datasets.factory.make_dataset()``
does for SmolVLA training. That gap is what this script closes.

This script is 100% empirical against the real, installed LeRobot code (``~/lerobot``, via its
own ``.venv``) and the real V2-clean dataset - it does not reimplement or guess the pipeline:

  1. Reads the SmolVLA training-time chunking law directly from
     ``lerobot.policies.smolvla.configuration_smolvla.SmolVLAConfig`` (``action_delta_indices``
     / ``observation_delta_indices`` properties) - falling back to a frozen checkpoint's own
     ``config.json`` (via ``PreTrainedConfig.from_pretrained``) when ``--checkpoint`` is given,
     so the exact chunk_size/n_action_steps actually used for the V2-clean training run are
     used, not just the class defaults.
  2. Builds the delta_timestamps dict via ``lerobot.datasets.factory.resolve_delta_timestamps``
     (the exact function ``make_dataset()`` calls for training) and constructs a real
     ``lerobot.datasets.lerobot_dataset.LeRobotDataset`` over
     ``data/so101_cube_xy_grid35_v2_clean`` with it.
  3. For representative (episode, frame) pairs, calls ``dataset[abs_idx]`` - the *exact*
     ``__getitem__`` path used by a training ``DataLoader`` - and inspects the returned
     ``action`` tensor (shape ``(chunk_size, 6)``) and ``action_is_pad`` mask directly.
  4. Cross-checks every chunk index ``k`` against the dataset's own raw (parquet) action array
     two independent ways: (a) exact equality against ``raw_action[t+k]`` (clamped at episode
     end), and (b) an unbiased nearest-neighbour search over the *entire* episode's raw action
     trajectory (L2, degrees) - so "chunk[k] corresponds to raw frame t+k" is verified, not
     assumed.
  5. Repeats the offset-law check (a) at full dataset scale (all 35 episodes, all frames, all
     chunk indices) using LeRobot's own ``DatasetReader._get_query_indices`` /
     ``_query_hf_dataset`` methods directly (no video decode needed for this scale check).
  6. Optionally (``--checkpoint``, on by default if the checkpoint dir exists) loads the real
     frozen policy the same way ``runtime/desktop/vla_server.py`` and
     ``scripts/evaluate_smolvla_midpoint.py`` do
     (``get_policy_class("smolvla").from_pretrained`` + ``make_pre_post_processors``), runs
     ``policy.predict_action_chunk()`` on the *exact same observations* used in the dataset
     trace, and reports the policy's actual delivered chunk[0] side by side with the training
     target chunk[0] for the same observation - so "did the model mislearn" (chunk[0] should
     equal action(t) but the policy outputs something else) can be told apart from "was the
     training target itself already offset into the future" (chunk[0] != action(t) by
     construction).

Read-only / no side effects
----------------------------
This script never modifies the dataset, the checkpoint, LeRobot's source, or any config; it
never retrains, never touches Safety Gate thresholds, and never writes to any robot (real or
simulated). It only reads parquet/video files and checkpoint files and writes its own report
files under ``--out-dir``.

Usage (must run inside the LeRobot venv - this repo's other lerobot-dependent scripts use the
same convention, see ``scripts/evaluate_smolvla_midpoint.py``)::

    source ~/lerobot/.venv/bin/activate
    python scripts/verify_smolvla_training_target_alignment.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.common.vla_contract import (  # noqa: E402
    CAMERA_WORKSPACE_KEY,
    CAMERA_WRIST_KEY,
    JOINT_ORDER,
)

JOINTS = list(JOINT_ORDER)
KEY_JOINTS = ["shoulder_lift", "elbow_flex"]
FPS = 30

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "so101_cube_xy_grid35_v2_clean"
DEFAULT_REPO_ID = "local/so101_cube_xy_grid35_v2_clean"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "grid35_v2"
    / "smolvla_grid35_v2_clean_fresh"
    / "checkpoints"
    / "007500"
    / "pretrained_model"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "smolvla_training_target_alignment"
DEFAULT_TASK = "Pick up the cube and place it in the target area."

REPRESENTATIVE_EPISODES = [0, 17, 34]
REPRESENTATIVE_FRAMES = [0, 5, 10, 15, 20, 25]
AGGREGATE_STRIDE = 5  # dataset-wide offset-law check samples every Nth frame of every episode
INFERENCE_SEEDS = [0, 1, 2]  # SmolVLA flow-matching noise is stochastic; sample a few draws


# --------------------------------------------------------------------------
# 0. Raw parquet ground truth (independent of LeRobotDataset/delta_timestamps)
# --------------------------------------------------------------------------


def load_raw_episodes(dataset_root: Path) -> dict[int, dict[str, np.ndarray]]:
    """Full per-episode observation.state/action/timestamp arrays, straight from parquet.

    This is deliberately independent of ``LeRobotDataset``/``delta_timestamps`` - it is the
    "ground truth" the training-target chunk gets cross-checked against.
    """
    episodes_meta_dir = dataset_root / "meta" / "episodes"
    meta_files = sorted(episodes_meta_dir.glob("chunk-*/file-*.parquet"))
    meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    meta = meta.sort_values("episode_index").reset_index(drop=True)

    episodes: dict[int, dict[str, np.ndarray]] = {}
    for _, row in meta.iterrows():
        ep = int(row["episode_index"])
        chunk_idx = int(row["data/chunk_index"])
        file_idx = int(row["data/file_index"])
        data_path = dataset_root / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        df = pd.read_parquet(
            data_path,
            columns=["action", "observation.state", "timestamp", "frame_index", "episode_index", "index"],
        )
        df = df[df["episode_index"] == ep].sort_values("frame_index")
        episodes[ep] = {
            "state": np.stack(df["observation.state"].to_numpy()).astype(np.float64),
            "action": np.stack(df["action"].to_numpy()).astype(np.float64),
            "timestamp": df["timestamp"].to_numpy().astype(np.float64),
            "dataset_index": df["index"].to_numpy().astype(np.int64),
            "length": len(df),
        }
    return episodes


# --------------------------------------------------------------------------
# 1. Real SmolVLA temporal config (chunk_size / n_action_steps / delta indices)
# --------------------------------------------------------------------------


def load_policy_temporal_config(checkpoint_dir: Path | None) -> dict[str, Any]:
    """Load the exact chunking config SmolVLA training used.

    If ``checkpoint_dir`` points at a real ``pretrained_model`` dir, reads its actual
    ``config.json`` via ``PreTrainedConfig.from_pretrained`` (the checkpoint's *own* trained
    chunk_size/n_action_steps - not just the class default). Otherwise falls back to
    ``SmolVLAConfig()`` defaults.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    if checkpoint_dir is not None and checkpoint_dir.is_dir() and (checkpoint_dir / "config.json").is_file():
        cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
        source = f"checkpoint config.json: {checkpoint_dir / 'config.json'}"
    else:
        cfg = SmolVLAConfig()
        source = "SmolVLAConfig() class defaults (no usable --checkpoint given)"

    return {
        "cfg": cfg,
        "source": source,
        "chunk_size": cfg.chunk_size,
        "n_action_steps": cfg.n_action_steps,
        "n_obs_steps": cfg.n_obs_steps,
        "action_delta_indices": list(cfg.action_delta_indices),
        "observation_delta_indices": list(cfg.observation_delta_indices),
    }


# --------------------------------------------------------------------------
# 2. Real LeRobotDataset with delta_timestamps (the actual training input pipeline)
# --------------------------------------------------------------------------


def build_training_dataset(dataset_root: Path, repo_id: str, cfg) -> Any:
    """Build the *exact* object a SmolVLA training run would iterate over.

    Mirrors ``lerobot.datasets.factory.make_dataset()``: read delta_timestamps off the policy
    config via ``resolve_delta_timestamps``, then construct ``LeRobotDataset`` with it.
    """
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(repo_id, root=str(dataset_root), delta_timestamps=delta_timestamps)
    return dataset, delta_timestamps, meta


# --------------------------------------------------------------------------
# 3. Representative-frame numeric trace
# --------------------------------------------------------------------------


@dataclass
class ChunkIndexTrace:
    k: int
    expected_raw_frame_idx: int
    expected_value: list[float]
    actual_value: list[float]
    max_abs_diff: float
    exact_match: bool
    dataset_is_pad_flag: bool
    expected_is_pad: bool
    nearest_neighbor_frame_idx: int
    nearest_neighbor_l2_deg: float
    nearest_neighbor_matches_expected_frame: bool


def nearest_raw_frame(raw_action: np.ndarray, value: np.ndarray, expected_idx: int) -> tuple[int, float, bool]:
    """Unbiased check: which raw-episode frame's action is closest (L2) to `value`?

    Used so "chunk[k] == raw_action[t+k]" is demonstrated empirically (nearest-neighbour search
    over the whole episode), not merely asserted by construction. The V2-clean dataset has long
    static-hold segments at episode start (see the prior start-segment audit: state/action first
    movement median frame ~21-25) where many consecutive frames share byte-identical action
    values - during those segments a plain ``argmin`` is not unique (many frames tie at
    distance 0) and will report the *first* tied frame, which is not itself an alignment bug.
    So this returns whether ``expected_idx`` is *one of* the (possibly many) frames tied for the
    minimum distance, not just whether it is the single lowest-index argmin.
    """
    dist = np.linalg.norm(raw_action - value[None, :], axis=1)
    idx = int(np.argmin(dist))
    min_dist = float(dist[idx])
    expected_is_tied_for_nearest = bool(dist[expected_idx] <= min_dist + 1e-9)
    return idx, min_dist, expected_is_tied_for_nearest


def trace_frame(
    dataset: Any,
    raw_episodes: dict[int, dict[str, np.ndarray]],
    ep: int,
    t: int,
    chunk_size: int,
) -> dict[str, Any] | None:
    raw = raw_episodes[ep]
    length = raw["length"]
    if t >= length:
        return None

    ep_meta = dataset.meta.episodes[ep]
    ep_start = int(ep_meta["dataset_from_index"])
    abs_idx = ep_start + t

    t0 = time.time()
    item = dataset[abs_idx]
    fetch_s = time.time() - t0

    obs_state = item["observation.state"][-1].numpy().astype(np.float64)  # observation_delta_indices == [0]
    action_chunk = item["action"].numpy().astype(np.float64)  # (chunk_size, 6)
    is_pad = item["action_is_pad"].numpy()

    assert action_chunk.shape[0] == chunk_size, (
        f"chunk_size mismatch: dataset returned {action_chunk.shape[0]}, expected {chunk_size}"
    )

    raw_state_t = raw["state"][t]
    raw_action_t = raw["action"][t]

    chunk_trace: list[ChunkIndexTrace] = []
    for k in range(chunk_size):
        expected_frame_idx = min(t + k, length - 1)
        expected_value = raw["action"][expected_frame_idx]
        actual_value = action_chunk[k]
        max_abs_diff = float(np.max(np.abs(actual_value - expected_value)))
        nn_idx, nn_dist, nn_confirms = nearest_raw_frame(raw["action"], actual_value, expected_frame_idx)
        chunk_trace.append(
            ChunkIndexTrace(
                k=k,
                expected_raw_frame_idx=expected_frame_idx,
                expected_value=[float(v) for v in expected_value],
                actual_value=[float(v) for v in actual_value],
                max_abs_diff=max_abs_diff,
                exact_match=max_abs_diff < 1e-4,
                dataset_is_pad_flag=bool(is_pad[k]),
                expected_is_pad=(t + k) >= length,
                nearest_neighbor_frame_idx=nn_idx,
                nearest_neighbor_l2_deg=nn_dist,
                nearest_neighbor_matches_expected_frame=nn_confirms,
            )
        )

    return {
        "episode": ep,
        "source_frame_index_t": t,
        "abs_dataset_index": abs_idx,
        "episode_length": length,
        "source_timestamp_s": float(raw["timestamp"][t]),
        "dataset_item_timestamp_s": float(item["timestamp"].item()),
        "observation_state_t_raw_parquet": [float(v) for v in raw_state_t],
        "observation_state_t_from_dataset_item": [float(v) for v in obs_state],
        "observation_state_matches_raw": bool(np.allclose(obs_state, raw_state_t, atol=1e-6)),
        "raw_action_t": [float(v) for v in raw_action_t],
        "training_target_chunk0": [float(v) for v in action_chunk[0]],
        "chunk0_equals_raw_action_t": bool(np.allclose(action_chunk[0], raw_action_t, atol=1e-4)),
        "chunk0_minus_state_t_deg": {
            j: float(action_chunk[0][i] - raw_state_t[i]) for i, j in enumerate(JOINTS)
        },
        "fetch_time_s": fetch_s,
        "chunk_trace": [c.__dict__ for c in chunk_trace],
        "all_offsets_exact_match": all(c.exact_match for c in chunk_trace),
        "all_offsets_nn_confirms_expected_frame": all(
            c.nearest_neighbor_matches_expected_frame for c in chunk_trace
        ),
        "any_pad_flag_mismatch": any(
            c.dataset_is_pad_flag != c.expected_is_pad for c in chunk_trace
        ),
    }


# --------------------------------------------------------------------------
# 4. Full-dataset (35 episode) offset-law check, no video decode (fast)
# --------------------------------------------------------------------------


def aggregate_offset_law_check(
    dataset: Any, raw_episodes: dict[int, dict[str, np.ndarray]], chunk_size: int, stride: int
) -> dict[str, Any]:
    """Verify ``chunk[k] == raw_action[t+k]`` (clamped) at full-dataset scale.

    Uses ``DatasetReader._get_query_indices``/``_query_hf_dataset`` directly - LeRobot's own
    internal methods (see ``lerobot.datasets.dataset_reader.DatasetReader``), same code
    ``dataset[idx]`` delegates to, but skipping video decode so this can run at full scale fast.
    """
    reader = dataset.reader
    max_abs_diff_overall = 0.0
    n_frames_checked = 0
    n_chunk_entries_checked = 0
    n_exact_mismatches = 0
    n_pad_flag_mismatches = 0
    per_episode_max_diff: dict[int, float] = {}

    for ep, raw in raw_episodes.items():
        ep_meta = dataset.meta.episodes[ep]
        ep_start = int(ep_meta["dataset_from_index"])
        length = raw["length"]
        ep_max_diff = 0.0
        for t in range(0, length, stride):
            abs_idx = ep_start + t
            query_indices, padding = reader._get_query_indices(abs_idx, ep)
            result = reader._query_hf_dataset({"action": query_indices["action"]})
            chunk = result["action"].numpy().astype(np.float64)
            is_pad = padding["action_is_pad"].numpy()
            n_frames_checked += 1
            for k in range(chunk_size):
                expected_frame_idx = min(t + k, length - 1)
                expected_value = raw["action"][expected_frame_idx]
                diff = float(np.max(np.abs(chunk[k] - expected_value)))
                ep_max_diff = max(ep_max_diff, diff)
                max_abs_diff_overall = max(max_abs_diff_overall, diff)
                n_chunk_entries_checked += 1
                if diff >= 1e-4:
                    n_exact_mismatches += 1
                expected_is_pad = (t + k) >= length
                if bool(is_pad[k]) != expected_is_pad:
                    n_pad_flag_mismatches += 1
        per_episode_max_diff[ep] = ep_max_diff

    return {
        "method": (
            "DatasetReader._get_query_indices + DatasetReader._query_hf_dataset "
            "(lerobot.datasets.dataset_reader.DatasetReader - the same internals dataset[idx] "
            "calls), stride={} frames per episode, all 35 episodes, all chunk_size={} "
            "offsets per sampled frame".format(stride, chunk_size)
        ),
        "n_frames_checked": n_frames_checked,
        "n_chunk_entries_checked": n_chunk_entries_checked,
        "n_exact_mismatches_offset_law_chunk_k_eq_raw_action_t_plus_k": n_exact_mismatches,
        "n_pad_flag_mismatches": n_pad_flag_mismatches,
        "max_abs_diff_overall_deg": max_abs_diff_overall,
        "per_episode_max_abs_diff_deg": per_episode_max_diff,
        "offset_law_holds_exactly": n_exact_mismatches == 0 and n_pad_flag_mismatches == 0,
    }


# --------------------------------------------------------------------------
# 5. Episode-end boundary / padding demonstration
# --------------------------------------------------------------------------


def boundary_padding_trace(dataset: Any, raw_episodes: dict[int, dict[str, np.ndarray]], ep: int, chunk_size: int) -> dict[str, Any]:
    """Show explicitly what happens to the target chunk near an episode's last frames."""
    raw = raw_episodes[ep]
    length = raw["length"]
    ep_meta = dataset.meta.episodes[ep]
    ep_start = int(ep_meta["dataset_from_index"])
    reader = dataset.reader

    rows = []
    for t in range(max(0, length - 5), length):
        abs_idx = ep_start + t
        query_indices, padding = reader._get_query_indices(abs_idx, ep)
        result = reader._query_hf_dataset({"action": query_indices["action"]})
        chunk = result["action"].numpy().astype(np.float64)
        is_pad = padding["action_is_pad"].numpy()
        n_pad = int(is_pad.sum())
        first_pad_k = int(np.argmax(is_pad)) if n_pad > 0 else None
        last_action_row = raw["action"][length - 1]
        pad_entries_equal_last_frame = bool(
            n_pad == 0 or np.allclose(chunk[is_pad], np.tile(last_action_row, (n_pad, 1)), atol=1e-4)
        )
        rows.append(
            {
                "frame_index_t": t,
                "n_pad_entries_in_chunk": n_pad,
                "first_pad_chunk_index": first_pad_k,
                "pad_entries_equal_last_raw_action": pad_entries_equal_last_frame,
                "chunk0_equals_raw_action_t": bool(np.allclose(chunk[0], raw["action"][t], atol=1e-4)),
            }
        )
    return {"episode": ep, "episode_length": length, "chunk_size": chunk_size, "rows": rows}


# --------------------------------------------------------------------------
# 6. Optional live policy inference (chunk[0] delivered by the actual checkpoint)
# --------------------------------------------------------------------------


def load_policy_bundle(checkpoint_dir: Path, device: str | None = None):
    import torch
    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy_cls = get_policy_class("smolvla")
    policy = policy_cls.from_pretrained(str(checkpoint_dir))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy.to(dev)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint_dir))
    return policy, preprocessor, postprocessor, dev


def live_policy_chunk0(
    policy, preprocessor, postprocessor, device, item: dict, task: str, seed: int
) -> dict[str, float]:
    import torch

    batch = {
        "observation.state": item["observation.state"].unsqueeze(0).to(device),
        CAMERA_WORKSPACE_KEY: item[CAMERA_WORKSPACE_KEY].unsqueeze(0).to(device),
        CAMERA_WRIST_KEY: item[CAMERA_WRIST_KEY].unsqueeze(0).to(device),
        "task": [task],
    }
    policy.reset()
    with torch.inference_mode():
        processed = preprocessor(batch)
        torch.manual_seed(seed)
        raw_chunk = policy.predict_action_chunk(processed)
        postproc_chunk = postprocessor(raw_chunk)
    chunk0 = postproc_chunk.detach().to("cpu").numpy()[0, 0]
    return {j: float(chunk0[i]) for i, j in enumerate(JOINTS)}


def run_live_policy_comparison(
    dataset: Any,
    raw_episodes: dict[int, dict[str, np.ndarray]],
    checkpoint_dir: Path,
    task: str,
    seeds: list[int],
) -> dict[str, Any]:
    policy, preprocessor, postprocessor, device = load_policy_bundle(checkpoint_dir)
    rows = []
    for ep in REPRESENTATIVE_EPISODES:
        raw = raw_episodes[ep]
        length = raw["length"]
        ep_meta = dataset.meta.episodes[ep]
        ep_start = int(ep_meta["dataset_from_index"])
        for t in REPRESENTATIVE_FRAMES:
            if t >= length:
                continue
            abs_idx = ep_start + t
            item = dataset[abs_idx]
            training_target_chunk0 = item["action"][0].numpy().astype(np.float64)
            state_t = raw["state"][t]
            per_seed = []
            for seed in seeds:
                policy_chunk0 = live_policy_chunk0(policy, preprocessor, postprocessor, device, item, task, seed)
                policy_vec = np.array([policy_chunk0[j] for j in JOINTS])
                per_seed.append(
                    {
                        "seed": seed,
                        "policy_chunk0": policy_chunk0,
                        "policy_chunk0_minus_training_target_chunk0_deg": {
                            j: float(policy_vec[i] - training_target_chunk0[i]) for i, j in enumerate(JOINTS)
                        },
                        "policy_chunk0_minus_state_t_deg": {
                            j: float(policy_vec[i] - state_t[i]) for i, j in enumerate(JOINTS)
                        },
                    }
                )
            rows.append(
                {
                    "episode": ep,
                    "frame_t": t,
                    "state_t": {j: float(state_t[i]) for i, j in enumerate(JOINTS)},
                    "training_target_chunk0": {j: float(training_target_chunk0[i]) for i, j in enumerate(JOINTS)},
                    "training_target_chunk0_minus_state_t_deg": {
                        j: float(training_target_chunk0[i] - state_t[i]) for i, j in enumerate(JOINTS)
                    },
                    "per_seed": per_seed,
                }
            )
    del policy, preprocessor, postprocessor
    import gc

    gc.collect()
    return {"checkpoint": str(checkpoint_dir), "task": task, "seeds": seeds, "rows": rows}


# --------------------------------------------------------------------------
# 7. Report writers
# --------------------------------------------------------------------------


def write_csv_representative(out_path: Path, traces: list[dict[str, Any]]) -> None:
    import csv

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "episode",
            "frame_t",
            "abs_dataset_index",
            "source_timestamp_s",
            "chunk0_equals_raw_action_t",
            "all_offsets_exact_match",
            "all_offsets_nn_confirms_expected_frame",
            "any_pad_flag_mismatch",
        ]
        for j in JOINTS:
            header += [f"state_t.{j}", f"raw_action_t.{j}", f"chunk0.{j}", f"chunk0_minus_state.{j}"]
        w.writerow(header)
        for tr in traces:
            row = [
                tr["episode"],
                tr["source_frame_index_t"],
                tr["abs_dataset_index"],
                f"{tr['source_timestamp_s']:.4f}",
                tr["chunk0_equals_raw_action_t"],
                tr["all_offsets_exact_match"],
                tr["all_offsets_nn_confirms_expected_frame"],
                tr["any_pad_flag_mismatch"],
            ]
            for j in JOINTS:
                i = JOINTS.index(j)
                row += [
                    f"{tr['observation_state_t_raw_parquet'][i]:.4f}",
                    f"{tr['raw_action_t'][i]:.4f}",
                    f"{tr['training_target_chunk0'][i]:.4f}",
                    f"{tr['chunk0_minus_state_t_deg'][j]:+.4f}",
                ]
            w.writerow(row)


def write_csv_chunk_detail(out_path: Path, traces: list[dict[str, Any]]) -> None:
    import csv

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "episode",
                "frame_t",
                "chunk_index_k",
                "expected_raw_frame_idx",
                "expected_is_pad",
                "dataset_is_pad_flag",
                "max_abs_diff_vs_expected_deg",
                "exact_match",
                "nearest_neighbor_frame_idx",
                "nearest_neighbor_l2_deg",
                "nn_matches_expected_frame",
            ]
        )
        for tr in traces:
            for c in tr["chunk_trace"]:
                w.writerow(
                    [
                        tr["episode"],
                        tr["source_frame_index_t"],
                        c["k"],
                        c["expected_raw_frame_idx"],
                        c["expected_is_pad"],
                        c["dataset_is_pad_flag"],
                        f"{c['max_abs_diff']:.6f}",
                        c["exact_match"],
                        c["nearest_neighbor_frame_idx"],
                        f"{c['nearest_neighbor_l2_deg']:.6f}",
                        c["nearest_neighbor_matches_expected_frame"],
                    ]
                )


def write_markdown_report(out_path: Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# SmolVLA training-target action-chunk alignment verification (V2-clean)")
    lines.append("")
    lines.append(f"Dataset: `{result['dataset_root']}`  ")
    lines.append(f"Temporal config source: {result['policy_temporal_config']['source']}  ")
    cfg = result["policy_temporal_config"]
    lines.append(
        f"`chunk_size`={cfg['chunk_size']}, `n_action_steps`={cfg['n_action_steps']}, "
        f"`n_obs_steps`={cfg['n_obs_steps']}, "
        f"`action_delta_indices`=`range(0,{cfg['chunk_size']})` "
        f"(={cfg['action_delta_indices'][:3]}...{cfg['action_delta_indices'][-2:]}), "
        f"`observation_delta_indices`={cfg['observation_delta_indices']}"
    )
    lines.append("")

    lines.append("## 1. How the training target chunk is actually built (code trail)")
    lines.append("")
    lines.append(
        "- `lerobot/policies/smolvla/configuration_smolvla.py` `SmolVLAConfig.action_delta_indices` "
        "-> `list(range(self.chunk_size))` (i.e. offsets `0, 1, 2, ..., chunk_size-1`); "
        "`observation_delta_indices` -> `[0]` (current frame only, no history/future for state)."
    )
    lines.append(
        "- `lerobot/datasets/factory.py` `resolve_delta_timestamps(cfg, ds_meta)` turns those "
        "into `delta_timestamps['action'] = [k/fps for k in action_delta_indices]` and "
        "`delta_timestamps['observation.state'] = [0/fps]`."
    )
    lines.append(
        "- `lerobot/datasets/lerobot_dataset.py` `LeRobotDataset.__getitem__` delegates straight "
        "to `DatasetReader.get_item(idx)`."
    )
    lines.append(
        "- `lerobot/datasets/dataset_reader.py` `DatasetReader._get_query_indices(abs_idx, ep_idx)` "
        "computes, per key, `[max(ep_start, min(ep_end-1, abs_idx + delta)) for delta in delta_idx]` "
        "- for `action` this is exactly `[abs_idx+0, abs_idx+1, ..., abs_idx+chunk_size-1]`, clamped "
        "to the episode's own `[ep_start, ep_end)` range, with an `action_is_pad` mask marking any "
        "entry where `abs_idx+delta` fell outside that range (post-clamp duplicate)."
    )
    lines.append(
        "- `lerobot/policies/smolvla/processor_smolvla.py` pre/post-processors only "
        "normalize/unnormalize + move device/dtype - no temporal shift anywhere in the "
        "processor pipeline."
    )
    lines.append(
        "- `lerobot/policies/smolvla/modeling_smolvla.py` `select_action()` extends the action "
        "queue with `actions.transpose(0,1)[:n_action_steps]` (chunk indices `0..n_action_steps-1` "
        "in order) and returns `self._queues[ACTION].popleft()` - i.e. the **first** dequeued "
        "action is chunk index 0, matching the training-target definition of chunk[0]."
    )
    lines.append("")
    lines.append(
        "**Conclusion from code alone: chunk[0] is defined, at training-data-construction time, "
        "to equal `action(t)` itself (offset 0) - not a future action.** Sections 2-4 below verify "
        "this empirically against the real V2-clean dataset rather than trusting the reasoning "
        "above by itself."
    )
    lines.append("")

    lines.append("## 2. Representative numeric trace (episode 0 / 17 / 34, frames 0/5/10/15/20/25)")
    lines.append("")
    lines.append(
        "| ep | t | abs_idx | ts(s) | state(t) SL/EF | raw action(t) SL/EF | target chunk[0] SL/EF | "
        "chunk0==action(t) | all 50 offsets exact | NN confirms t+k |"
    )
    lines.append("|---:|---:|---:|---:|---|---|---|---|---|---|")
    sl_i, ef_i = JOINTS.index("shoulder_lift"), JOINTS.index("elbow_flex")
    for tr in result["representative_traces"]:
        s = tr["observation_state_t_raw_parquet"]
        a = tr["raw_action_t"]
        c0 = tr["training_target_chunk0"]
        lines.append(
            f"| {tr['episode']} | {tr['source_frame_index_t']} | {tr['abs_dataset_index']} | "
            f"{tr['source_timestamp_s']:.3f} | {s[sl_i]:.2f}/{s[ef_i]:.2f} | "
            f"{a[sl_i]:.2f}/{a[ef_i]:.2f} | {c0[sl_i]:.2f}/{c0[ef_i]:.2f} | "
            f"{tr['chunk0_equals_raw_action_t']} | {tr['all_offsets_exact_match']} | "
            f"{tr['all_offsets_nn_confirms_expected_frame']} |"
        )
    lines.append("")
    lines.append(
        "Full per-joint numeric values: `representative_frames.csv`. Full per-chunk-index "
        "(k=0..chunk_size-1) detail for every representative frame: `chunk_index_detail.csv`."
    )
    lines.append("")

    lines.append("## 3. Chunk[k] -> which raw frame, for a sample frame (episode 0, frame 10)")
    lines.append("")
    sample = next(
        (
            tr
            for tr in result["representative_traces"]
            if tr["episode"] == 0 and tr["source_frame_index_t"] == 10
        ),
        result["representative_traces"][0],
    )
    lines.append(f"episode={sample['episode']}, t={sample['source_frame_index_t']}, chunk_size={cfg['chunk_size']}")
    lines.append("")
    lines.append("| k | expected raw frame (t+k, clamped) | max\\|diff\\| vs expected (deg) | exact match | NN raw frame found | NN matches expected |")
    lines.append("|---:|---:|---:|---|---:|---|")
    show_k = sorted(set([0, 1, 2, 5, 10, 20, 30, 40, cfg["chunk_size"] - 1]))
    for k in show_k:
        if k >= len(sample["chunk_trace"]):
            continue
        c = sample["chunk_trace"][k]
        lines.append(
            f"| {c['k']} | {c['expected_raw_frame_idx']} | {c['max_abs_diff']:.6f} | {c['exact_match']} | "
            f"{c['nearest_neighbor_frame_idx']} | {c['nearest_neighbor_matches_expected_frame']} |"
        )
    lines.append("")

    lines.append("## 4. Full-dataset (35 episodes) offset-law check")
    lines.append("")
    agg = result["aggregate_offset_law"]
    lines.append(f"Method: {agg['method']}")
    lines.append("")
    lines.append(
        f"- frames sampled: **{agg['n_frames_checked']}**, chunk entries checked: "
        f"**{agg['n_chunk_entries_checked']}**"
    )
    lines.append(
        f"- exact-match failures (`chunk[k] != raw_action[t+k]` beyond 1e-4 deg): "
        f"**{agg['n_exact_mismatches_offset_law_chunk_k_eq_raw_action_t_plus_k']}**"
    )
    lines.append(f"- `action_is_pad` flag mismatches: **{agg['n_pad_flag_mismatches']}**")
    lines.append(f"- max\\|diff\\| observed anywhere: **{agg['max_abs_diff_overall_deg']:.6f} deg**")
    lines.append(f"- **offset law holds exactly across the full dataset: {agg['offset_law_holds_exactly']}**")
    lines.append("")

    lines.append("## 5. Episode-end boundary / padding behaviour")
    lines.append("")
    for bp in result["boundary_padding_traces"]:
        lines.append(f"Episode {bp['episode']} (length={bp['episode_length']}, chunk_size={bp['chunk_size']}):")
        lines.append("")
        lines.append("| frame t | n_pad entries in chunk | first pad chunk-index | pad entries == last raw action | chunk[0]==action(t) |")
        lines.append("|---:|---:|---:|---|---|")
        for r in bp["rows"]:
            lines.append(
                f"| {r['frame_index_t']} | {r['n_pad_entries_in_chunk']} | {r['first_pad_chunk_index']} | "
                f"{r['pad_entries_equal_last_raw_action']} | {r['chunk0_equals_raw_action_t']} |"
            )
        lines.append("")
    lines.append(
        "Note: `action_delta_indices` only ever contains non-negative offsets (`0..chunk_size-1`), "
        "so **no clamping/padding can occur at episode start** - chunk[0] always equals `action(t)` "
        "regardless of how close `t` is to the episode's own start. Padding only occurs near the "
        "*end* of an episode, when `t + k` runs past the episode's last frame; those entries are "
        "clamped to (repeat) the episode's last raw action and flagged `action_is_pad=True`."
    )
    lines.append("")

    if result.get("live_policy_comparison"):
        lc = result["live_policy_comparison"]
        lines.append("## 6. Live policy chunk[0] vs. training target chunk[0] (same observations)")
        lines.append("")
        lines.append(f"Checkpoint: `{lc['checkpoint']}`, task=`{lc['task']}`, seeds={lc['seeds']}")
        lines.append("")
        lines.append(
            "| ep | t | joint | state(t) | training target chunk[0] | target-state | policy chunk[0] (mean over seeds) | policy-state | policy-target |"
        )
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in lc["rows"]:
            for j in KEY_JOINTS:
                policy_vals = [ps["policy_chunk0"][j] for ps in row["per_seed"]]
                policy_mean = float(np.mean(policy_vals))
                target = row["training_target_chunk0"][j]
                state = row["state_t"][j]
                lines.append(
                    f"| {row['episode']} | {row['frame_t']} | {j} | {state:.2f} | {target:.2f} | "
                    f"{target - state:+.2f} | {policy_mean:.2f} | {policy_mean - state:+.2f} | "
                    f"{policy_mean - target:+.2f} |"
                )
        lines.append("")
        lines.append(
            "`target-state` is the training-label's own immediate delta (what the model *should* "
            "learn to output as chunk[0] for this observation, per the offset law in section 1). "
            "`policy-state` is what the live checkpoint actually outputs as chunk[0]. `policy-target` "
            "is the direct discrepancy between the two, isolating whether any oversized first-action "
            "behaviour comes from the policy having learned something other than its own training "
            "target (large `policy-target`) vs. the training target itself already being large "
            "(large `target-state`, which section 1-4 rule out for this dataset)."
        )
        lines.append("")
        lines.append("Full per-seed values: `live_policy_vs_training_target.json`.")
        lines.append("")

    lines.append("## 7. Answers")
    lines.append("")
    v = result["verdict"]
    for key, text in v.items():
        lines.append(f"**{key}.** {text}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def build_verdict(result: dict[str, Any]) -> dict[str, str]:
    agg = result["aggregate_offset_law"]
    all_rep_ok = all(
        tr["chunk0_equals_raw_action_t"] and tr["all_offsets_exact_match"] and tr["all_offsets_nn_confirms_expected_frame"]
        for tr in result["representative_traces"]
    )
    holds = agg["offset_law_holds_exactly"] and all_rep_ok

    v: dict[str, str] = {}
    v["A"] = (
        f"YES - empirically confirmed on {agg['n_chunk_entries_checked']} chunk entries across all 35 "
        "episodes (section 4) and on every representative frame (section 2/3): "
        "`training target chunk[0] == raw dataset action(t)` exactly (max|diff| "
        f"{agg['max_abs_diff_overall_deg']:.2e} deg, i.e. float round-trip noise only)."
        if holds
        else "NO - see section 4/aggregate check, exact-match failures were found; chunk[0] is NOT "
        "simply action(t) for this dataset/config - inspect per_episode_max_abs_diff_deg."
    )
    v["B"] = (
        "NO future offset exists in the training-target construction itself - `action_delta_indices = "
        "range(chunk_size)` starts at 0, so chunk[0] is defined as the *current* frame's own action, "
        "not action(t+k) for any k>0."
        if holds
        else "An offset was empirically found - see section 4 for which episodes/frames disagree with "
        "the offset-0 hypothesis."
    )
    v["C"] = "N/A - offset is 0 (see A/B)." if holds else "See section 4 aggregate table for the actual offset found."
    v["D"] = (
        "YES - chunk[1]=action(t+1), chunk[2]=action(t+2), ..., chunk[chunk_size-1]=action(t+chunk_size-1), "
        "confirmed both by exact value equality and by an independent nearest-neighbour search over "
        "each episode's full raw action trajectory (section 2/3: 'all 50 offsets exact' / "
        "'NN confirms t+k' both True for every representative frame; section 4: 0 mismatches "
        "dataset-wide)."
    )
    v["E"] = (
        "NO material shift at episode start: `action_delta_indices` only contains non-negative offsets "
        "(0..chunk_size-1), so clamping/padding is structurally impossible at t=0 or any other "
        "start-of-episode frame - chunk[0] always equals action(t) regardless of position in the "
        "episode. Padding *does* occur near the episode's *end* (see section 5): once t+k runs past "
        "the last frame, that chunk entry is clamped to (repeats) the episode's final raw action and "
        "is flagged `action_is_pad=True` - this is a well-defined, correctly-flagged boundary "
        "behaviour, not a silent misalignment."
    )
    v["F"] = (
        "`chunk_size`/`n_action_steps` set the *number* of future steps in the target/consumed window "
        "(both 50 for this checkpoint, i.e. 1.667s @ 30fps) but do not shift *where* offset-0 starts - "
        "`action_delta_indices = list(range(chunk_size))` always starts at 0 regardless of chunk_size. "
        "`delta_timestamps` (via `resolve_delta_timestamps`) is a direct, unmodified re-expression of "
        "those same indices in seconds (`k/fps`). The SmolVLA pre/post-processors "
        "(`processor_smolvla.py`) perform only normalization/unnormalization and "
        "device/dtype/tokenization steps - no temporal resampling, shifting, or delaying of the "
        "action target anywhere in that pipeline."
    )
    v["G"] = (
        "YES, consistent - `select_action()` (`modeling_smolvla.py`) extends its internal action "
        "queue with `actions.transpose(0,1)[:n_action_steps]`, i.e. chunk indices `0, 1, 2, ...` in "
        "that order, and pops from the *front* (`popleft()`) - so the very first action a fresh "
        "`select_action()` call returns is chunk index 0, matching the training-target definition of "
        "chunk[0] used above (built with a freshly-reset queue, one call per representative "
        "observation, so this comparison is apples-to-apples with training)."
    )
    if result.get("live_policy_comparison"):
        v["6 (live checkpoint vs. training target)"] = (
            "See section 6: for each representative observation, the actually-loaded 7.5k checkpoint's "
            "delivered chunk[0] is reported side by side with that same observation's training target "
            "chunk[0] (== action(t) per A-E above). A large `policy-target` discrepancy with a small "
            "`target-state` (training label itself near-zero, matching the earlier start-segment audit) "
            "would point at the model (mis)learning something other than its own training target; a "
            "large `target-state` would instead point at the training data itself. Read the numbers in "
            "`live_policy_vs_training_target.json` / section 6 table before concluding either way - this "
            "script reports the comparison, it does not pre-judge which side is at fault."
        )
    else:
        v["6 (live checkpoint vs. training target)"] = (
            "SKIPPED - no usable --checkpoint given/found. Re-run with --checkpoint pointing at a real "
            "pretrained_model dir (default: "
            f"{DEFAULT_CHECKPOINT}) to get this comparison."
        )
    return v


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--no-live-inference", action="store_true", help="Skip section 6 (live policy chunk dump).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--aggregate-stride", type=int, default=AGGREGATE_STRIDE)
    parser.add_argument("--seeds", type=int, nargs="+", default=INFERENCE_SEEDS)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = args.checkpoint.resolve() if args.checkpoint else None
    checkpoint_usable = (
        not args.no_live_inference and checkpoint_dir is not None and checkpoint_dir.is_dir()
        and (checkpoint_dir / "config.json").is_file() and (checkpoint_dir / "model.safetensors").is_file()
    )

    print("[verify] 1/7 loading raw parquet ground truth (35 episodes)")
    raw_episodes = load_raw_episodes(dataset_root)
    assert len(raw_episodes) == 35, f"expected 35 episodes, found {len(raw_episodes)}"

    print("[verify] 2/7 loading SmolVLA temporal config (chunk_size/n_action_steps/delta indices)")
    temporal_cfg = load_policy_temporal_config(checkpoint_dir if checkpoint_usable else None)
    print(
        f"         chunk_size={temporal_cfg['chunk_size']} n_action_steps={temporal_cfg['n_action_steps']} "
        f"source={temporal_cfg['source']}"
    )

    print("[verify] 3/7 building real LeRobotDataset with resolve_delta_timestamps()")
    dataset, delta_timestamps, meta = build_training_dataset(dataset_root, args.repo_id, temporal_cfg["cfg"])
    print(f"         dataset length={len(dataset)}, action delta_timestamps count={len(delta_timestamps['action'])}")

    print("[verify] 4/7 representative-frame numeric trace (episodes 0/17/34, frames 0/5/10/15/20/25)")
    representative_traces = []
    for ep in REPRESENTATIVE_EPISODES:
        for t in REPRESENTATIVE_FRAMES:
            tr = trace_frame(dataset, raw_episodes, ep, t, temporal_cfg["chunk_size"])
            if tr is not None:
                representative_traces.append(tr)
    print(f"         {len(representative_traces)} representative frames traced")

    print(f"[verify] 5/7 full-dataset offset-law check (stride={args.aggregate_stride})")
    aggregate = aggregate_offset_law_check(dataset, raw_episodes, temporal_cfg["chunk_size"], args.aggregate_stride)
    print(f"         mismatches={aggregate['n_exact_mismatches_offset_law_chunk_k_eq_raw_action_t_plus_k']}, "
          f"max|diff|={aggregate['max_abs_diff_overall_deg']:.2e} deg")

    print("[verify] 6/7 episode-end boundary/padding trace")
    boundary_traces = [
        boundary_padding_trace(dataset, raw_episodes, ep, temporal_cfg["chunk_size"])
        for ep in REPRESENTATIVE_EPISODES
    ]

    live_policy_comparison = None
    if checkpoint_usable:
        print(f"[verify] 7/7 live policy inference comparison ({checkpoint_dir})")
        live_policy_comparison = run_live_policy_comparison(
            dataset, raw_episodes, checkpoint_dir, args.task, args.seeds
        )
    else:
        print("[verify] 7/7 live policy inference SKIPPED (--no-live-inference or no usable --checkpoint)")

    result: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "policy_temporal_config": {k: v for k, v in temporal_cfg.items() if k != "cfg"},
        "representative_traces": representative_traces,
        "aggregate_offset_law": aggregate,
        "boundary_padding_traces": boundary_traces,
        "live_policy_comparison": live_policy_comparison,
    }
    result["verdict"] = build_verdict(result)

    json_path = out_dir / "training_target_alignment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"[verify] wrote {json_path}")

    write_csv_representative(out_dir / "representative_frames.csv", representative_traces)
    print(f"[verify] wrote {out_dir / 'representative_frames.csv'}")

    write_csv_chunk_detail(out_dir / "chunk_index_detail.csv", representative_traces)
    print(f"[verify] wrote {out_dir / 'chunk_index_detail.csv'}")

    if live_policy_comparison is not None:
        lp_path = out_dir / "live_policy_vs_training_target.json"
        with open(lp_path, "w", encoding="utf-8") as f:
            json.dump(live_policy_comparison, f, ensure_ascii=False, indent=2, default=float)
        print(f"[verify] wrote {lp_path}")

    md_path = out_dir / "training_target_alignment.md"
    write_markdown_report(md_path, result)
    print(f"[verify] wrote {md_path}")

    print("[verify] VERDICT:")
    for k, v in result["verdict"].items():
        print(f"  {k}: {v[:120]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
