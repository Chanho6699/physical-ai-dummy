#!/usr/bin/env python3
"""Seed sweep for the Grid35 V2 clean SmolVLA 7.5k first-action stochastic-noise investigation.

Companion to ``scripts/diagnose_grid35_first_action_pipeline.py`` (see that script's "Step 4:
live chunk dump" section and its now-removed hardcoded ``torch.manual_seed(42)``). That earlier
diagnostic established that:

  * ``policy.eval()`` is applied correctly.
  * SmolVLA samples fresh flow-matching noise on every ``predict_action_chunk()`` call unless the
    RNG is seeded immediately beforehand (``--inference-seed``, added to that script).
  * Over 20 unseeded repeats of the same T01 Shadow observation, chunk-index-0 (the action the
    runtime actually sends) varied substantially: shoulder_lift +0.77..+8.44deg (7/20 WOULD_CLAMP),
    elbow_flex -2.73..-11.14deg (13/20 WOULD_CLAMP).
  * Fixing ``torch.manual_seed(42)`` makes repeated inference calls byte-identical - confirming the
    variance's proximate cause is the sampled flow-matching noise, not any other RNG source.

That still leaves seed=42 itself an arbitrary, unvalidated choice. This script does NOT try to
pick a single "good" seed. Instead it sweeps ``--inference-seed`` over a range (default 0..19),
running live chunk-index-0 inference exactly once per seed against one fixed Shadow observation,
and reports how much first-action *quality* (safety-threshold clamp status, and L2 distance to the
nearest-training-demo's immediate ground-truth delta) varies across that seed range. The point is
to quantify seed-to-seed variance, not to bless a value.

Deliberately out of scope (same constraints as the parent diagnostic):
  * No training / fine-tuning.
  * No modification of Safety Gate thresholds (``configs/safety_gate.yaml`` is read-only here).
  * No writes to any robot, real or simulated.
  * No modification of the LeRobot library itself - only this repo's own scripts.

Unlike the parent diagnostic script (which reloads the checkpoint from disk on every invocation),
this script loads the policy/pre/post-processors exactly once and re-seeds+re-runs
``predict_action_chunk`` per seed, since only the RNG state before that call is supposed to change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_grid35_first_action_pipeline import (  # noqa: E402
    JOINTS,
    nearest_frame_trajectory_analysis,
    safety_threshold_check,
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "grid35_v2"
    / "smolvla_grid35_v2_clean_fresh"
    / "checkpoints"
    / "007500"
    / "pretrained_model"
)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "so101_cube_xy_grid35_v2_clean"
DEFAULT_SHADOW_REPORT = PROJECT_ROOT / "reports" / "grid35_v2_shadow_T01" / "shadow_20260808_211555.json"
DEFAULT_SAFETY_GATE_YAML = PROJECT_ROOT / "configs" / "safety_gate.yaml"
DEFAULT_TASK = "Pick up the cube and place it in the target area."
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "grid35_v2_T01_seed_sweep"


def load_shadow_observation(shadow_report: Path) -> dict[str, Any]:
    with open(shadow_report, encoding="utf-8") as f:
        d = json.load(f)
    state = d["observation"]["state"]
    raw_action = d["vla"]["raw_action"]
    camera_paths = d.get("observation", {}).get("camera_frame_paths") or {}
    return {
        "label": d.get("scene_metadata", {}).get("label"),
        "state": state,
        "raw_action_post_postprocessor": raw_action,
        "camera_frame_paths": camera_paths,
    }


def load_images(camera_paths: dict[str, str]) -> dict[str, np.ndarray]:
    from PIL import Image

    images: dict[str, np.ndarray] = {}
    if camera_paths.get("workspace_path") and camera_paths.get("wrist_path"):
        images["observation.images.workspace"] = np.array(
            Image.open(camera_paths["workspace_path"]).convert("RGB"), dtype=np.uint8
        )
        images["observation.images.wrist"] = np.array(
            Image.open(camera_paths["wrist_path"]).convert("RGB"), dtype=np.uint8
        )
    return images


def build_batch(state: dict[str, float], images: dict[str, np.ndarray], task: str, device, torch) -> dict:
    from runtime.common.vla_contract import JOINT_ORDER

    state_arr = np.array([state[j] for j in JOINT_ORDER], dtype=np.float32)
    batch: dict[str, object] = {"observation.state": torch.from_numpy(state_arr).unsqueeze(0).to(device)}
    for key, arr in images.items():
        img = torch.from_numpy(arr)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = img.permute(2, 0, 1).contiguous()
        batch[key] = img.unsqueeze(0).to(device)
    batch["task"] = [task]
    return batch


def load_policy_bundle(checkpoint_dir: Path) -> dict[str, Any]:
    """Load the SmolVLA policy + pre/post-processors exactly once.

    Split out of ``run_seed_sweep`` so multi-scene callers (e.g.
    ``sweep_grid35_multi_scene.py``) can load the checkpoint a single time and reuse it across
    every T01-T10 scene, instead of reloading it from disk once per scene. Single-scene callers
    (this script's own ``main()``) still only ever call this once per process, so behavior is
    unchanged there.
    """
    import torch
    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy_cls = get_policy_class("smolvla")
    policy = policy_cls.from_pretrained(str(checkpoint_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint_dir))
    return {
        "torch": torch,
        "policy": policy,
        "device": device,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
    }


def run_seed_sweep_with_bundle(
    bundle: dict[str, Any],
    state: dict[str, float],
    images: dict[str, np.ndarray],
    task: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Same per-seed inference loop as ``run_seed_sweep``, but against an already-loaded policy
    bundle (see ``load_policy_bundle``). This is the code path both ``run_seed_sweep`` (single
    scene, loads its own bundle) and multi-scene drivers (load one bundle, call this once per
    scene) actually share - no duplicated inference logic between them.
    """
    from runtime.common.vla_contract import JOINT_ORDER

    torch = bundle["torch"]
    policy = bundle["policy"]
    device = bundle["device"]
    preprocessor = bundle["preprocessor"]
    postprocessor = bundle["postprocessor"]

    batch = build_batch(state, images, task, device, torch)

    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        # Preprocessing has no seed-sensitive randomness of its own, but run it fresh per seed
        # anyway so no RNG state leaks between iterations via any incidental preprocessor op.
        for seed in seeds:
            processed = preprocessor(batch)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            raw_chunk = policy.predict_action_chunk(processed)  # (1, chunk_size, action_dim)
            postproc_chunk = postprocessor(raw_chunk)
            chunk = postproc_chunk.detach().to("cpu").numpy()[0]  # (chunk_size, action_dim)
            first_action = {j: float(chunk[0, i]) for i, j in enumerate(JOINT_ORDER)}
            delta = {j: first_action[j] - state[j] for j in JOINT_ORDER}
            results.append({"seed": seed, "first_action": first_action, "delta": delta})
    return results


def run_seed_sweep(
    checkpoint_dir: Path,
    state: dict[str, float],
    images: dict[str, np.ndarray],
    task: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    bundle = load_policy_bundle(checkpoint_dir)
    return run_seed_sweep_with_bundle(bundle, state, images, task, seeds)


def annotate_clamp(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> None:
    for row in rows:
        clamped_joints = [j for j in JOINTS if abs(row["delta"][j]) > thresholds[j]]
        row["clamped_joints"] = clamped_joints
        row["clamp_joint_count"] = len(clamped_joints)
        row["any_clamp"] = len(clamped_joints) > 0


def annotate_gt_l2(rows: list[dict[str, Any]], gt_delta: dict[str, float]) -> None:
    gt_vec = np.array([gt_delta[j] for j in JOINTS])
    for row in rows:
        delta_vec = np.array([row["delta"][j] for j in JOINTS])
        row["l2_error_vs_nearest_demo_gt_deg"] = float(np.linalg.norm(delta_vec - gt_vec))


def summarize(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    per_joint: dict[str, Any] = {}
    for j in JOINTS:
        vals = np.array([row["delta"][j] for row in rows])
        clamp_count = sum(1 for row in rows if j in row["clamped_joints"])
        per_joint[j] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "clamp_count": clamp_count,
            "clamp_rate": clamp_count / len(rows),
            "threshold_deg": thresholds[j],
        }

    clamp_free_seeds = [row["seed"] for row in rows if row["clamp_joint_count"] == 0]
    clamp_count_dist: dict[int, int] = {}
    for row in rows:
        clamp_count_dist[row["clamp_joint_count"]] = clamp_count_dist.get(row["clamp_joint_count"], 0) + 1

    l2_vals = np.array([row["l2_error_vs_nearest_demo_gt_deg"] for row in rows])

    return {
        "n_seeds": len(rows),
        "per_joint": per_joint,
        "clamp_free_seed_count": len(clamp_free_seeds),
        "clamp_free_seeds": clamp_free_seeds,
        "clamp_joint_count_distribution": {str(k): v for k, v in sorted(clamp_count_dist.items())},
        "l2_error_vs_gt": {
            "mean": float(l2_vals.mean()),
            "std": float(l2_vals.std(ddof=0)),
            "min": float(l2_vals.min()),
            "max": float(l2_vals.max()),
        },
    }


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| seed | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper "
        "| clamp joint count | clamped joints | L2 err vs GT (deg) |"
    )
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"
    lines = [header, sep]
    for row in rows:
        d = row["delta"]
        lines.append(
            f"| {row['seed']} | {d['shoulder_pan']:+.2f} | {d['shoulder_lift']:+.2f} | "
            f"{d['elbow_flex']:+.2f} | {d['wrist_flex']:+.2f} | {d['wrist_roll']:+.2f} | "
            f"{d['gripper']:+.2f} | {row['clamp_joint_count']} | "
            f"{', '.join(row['clamped_joints']) if row['clamped_joints'] else '-'} | "
            f"{row['l2_error_vs_nearest_demo_gt_deg']:.2f} |"
        )
    return "\n".join(lines)


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "seed_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f"[sweep] wrote {json_path}")

    csv_path = out_dir / "seed_sweep.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("seed," + ",".join(JOINTS) + ",clamp_joint_count,any_clamp,clamped_joints,l2_error_vs_gt_deg\n")
        for row in payload["rows"]:
            d = row["delta"]
            vals = ",".join(f"{d[j]:.6f}" for j in JOINTS)
            f.write(
                f"{row['seed']},{vals},{row['clamp_joint_count']},{int(row['any_clamp'])},"
                f"\"{';'.join(row['clamped_joints'])}\",{row['l2_error_vs_nearest_demo_gt_deg']:.6f}\n"
            )
    print(f"[sweep] wrote {csv_path}")

    md_path = out_dir / "seed_sweep.md"
    lines: list[str] = []
    lines.append("# Grid35 V2 clean SmolVLA 7.5k - first-action inference-seed sweep")
    lines.append("")
    lines.append(f"Reference Shadow observation: `{payload['reference_observation_label']}` "
                 f"(`{payload['shadow_report']}`)")
    lines.append(f"Checkpoint: `{payload['checkpoint']}`")
    lines.append(f"Task: `{payload['task']}`")
    lines.append(f"Seeds swept: {payload['seeds'][0]}..{payload['seeds'][-1]} "
                 f"({len(payload['seeds'])} seeds, 1 inference call each)")
    lines.append("")
    lines.append("## Safety thresholds used (read-only, not modified)")
    lines.append("")
    lines.append("| joint | WOULD_CLAMP threshold (deg) |")
    lines.append("|---|---:|")
    for j in JOINTS:
        lines.append(f"| {j} | {payload['thresholds'][j]:.2f} |")
    lines.append("")
    lines.append("## Nearest-training-demo immediate GT delta (reference)")
    lines.append("")
    lines.append("| joint | GT immediate delta (deg) |")
    lines.append("|---|---:|")
    for j in JOINTS:
        lines.append(f"| {j} | {payload['gt_immediate_delta'][j]:+.4f} |")
    lines.append("")
    lines.append("## Per-seed chunk[0] delta table")
    lines.append("")
    lines.append(format_markdown_table(payload["rows"]))
    lines.append("")
    lines.append("## Summary statistics (seeds swept)")
    lines.append("")
    lines.append("| joint | mean | std | min | max | clamp count | clamp rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for j in JOINTS:
        s = payload["summary"]["per_joint"][j]
        lines.append(
            f"| {j} | {s['mean']:+.2f} | {s['std']:.2f} | {s['min']:+.2f} | {s['max']:+.2f} | "
            f"{s['clamp_count']}/{payload['summary']['n_seeds']} | {s['clamp_rate']*100:.0f}% |"
        )
    lines.append("")
    lines.append(f"- Seeds with **zero** clamped joints: {payload['summary']['clamp_free_seed_count']}"
                 f"/{payload['summary']['n_seeds']} "
                 f"({payload['summary']['clamp_free_seeds']})")
    lines.append(f"- clamp-joint-count distribution (count -> #seeds): "
                 f"{payload['summary']['clamp_joint_count_distribution']}")
    lines.append(f"- L2 error vs nearest-demo immediate GT delta (deg): "
                 f"mean={payload['summary']['l2_error_vs_gt']['mean']:.2f}, "
                 f"std={payload['summary']['l2_error_vs_gt']['std']:.2f}, "
                 f"min={payload['summary']['l2_error_vs_gt']['min']:.2f}, "
                 f"max={payload['summary']['l2_error_vs_gt']['max']:.2f}")
    lines.append("")
    out_path = md_path
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[sweep] wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--shadow-report", type=Path, default=DEFAULT_SHADOW_REPORT)
    parser.add_argument("--safety-gate-yaml", type=Path, default=DEFAULT_SAFETY_GATE_YAML)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=19, help="Inclusive.")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_end + 1))

    print(f"[sweep] loading reference Shadow observation from {args.shadow_report}")
    obs = load_shadow_observation(args.shadow_report.resolve())
    images = load_images(obs["camera_frame_paths"])
    if not images:
        print("[sweep] WARNING: no camera frames found for this Shadow report; running with no image "
              "inputs (policy will still run, but this does not match the real deployment input).")

    print(f"[sweep] loading safety thresholds from {args.safety_gate_yaml}")
    thresholds = safety_threshold_check(args.safety_gate_yaml.resolve())["would_clamp_threshold_deg"]

    print(f"[sweep] computing nearest-training-demo immediate GT delta from {args.dataset_root}")
    traj = nearest_frame_trajectory_analysis(
        args.dataset_root.resolve(), obs["state"], obs["raw_action_post_postprocessor"]
    )
    gt_immediate_delta = traj["best_match"]["gt_immediate_delta_action_minus_state"]

    print(f"[sweep] loading checkpoint {args.checkpoint} and running {len(seeds)} seeded inference calls "
          f"(seeds {seeds[0]}..{seeds[-1]})")
    rows = run_seed_sweep(args.checkpoint.resolve(), obs["state"], images, args.task, seeds)

    annotate_clamp(rows, thresholds)
    annotate_gt_l2(rows, gt_immediate_delta)
    summary = summarize(rows, thresholds)

    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "shadow_report": str(args.shadow_report),
        "reference_observation_label": obs["label"],
        "reference_observation_state": obs["state"],
        "task": args.task,
        "seeds": seeds,
        "thresholds": thresholds,
        "gt_immediate_delta": gt_immediate_delta,
        "nearest_demo_match": {
            "episode": traj["best_match"]["episode"],
            "frame": traj["best_match"]["frame"],
            "l2_dist_deg": traj["best_match"]["l2_dist_deg"],
        },
        "rows": rows,
        "summary": summary,
    }

    write_report(args.out_dir.resolve(), payload)

    print("")
    print(f"[sweep] {summary['clamp_free_seed_count']}/{summary['n_seeds']} seeds had zero clamped joints")
    print(f"[sweep] clamp-joint-count distribution: {summary['clamp_joint_count_distribution']}")
    print(f"[sweep] L2 error vs GT: mean={summary['l2_error_vs_gt']['mean']:.2f}deg, "
          f"min={summary['l2_error_vs_gt']['min']:.2f}deg, max={summary['l2_error_vs_gt']['max']:.2f}deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
