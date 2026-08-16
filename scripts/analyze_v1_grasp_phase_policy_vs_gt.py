#!/usr/bin/env python3
"""Read-only V1 grasp-phase policy-vs-training-target evaluation.

This script never opens robot hardware and never mutates the source dataset.  It
uses the production ``VLAHttpClient.predict_chunk`` path (including its JPEG-90
RGB transport) and reconstructs LeRobot training targets as action[t:t+50]
with terminal padding, matching the existing initial-semantics analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY, JOINT_ORDER
from runtime.laptop.vla_client import VLAClientConfig, VLAHttpClient


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/so101_blue_cube_place_return_v1"
ONSET_ANALYSIS = ROOT / "reports/v1_initial_target_semantics_analysis/analysis.json"
INITIAL_MANIFEST = ROOT / "reports/v1_initial_target_semantics_analysis/inference_manifest.json"
OUTPUT = ROOT / "reports/v1_7500_grasp_phase_policy_vs_gt_seed20260815"
TASK = 'Pick up the blue cube, place it inside the blue rectangle labeled "BLUE", then return to the starting pose.'
OFFSETS = (-60, -45, -30, -20, -10, 0, 10, 20, 30)
STEPS = (0, 10, 20, 30, 40)


def chunk(actions: np.ndarray, frame: int) -> np.ndarray:
    return np.stack([actions[min(frame + k, len(actions) - 1)] for k in range(50)])


def decode_selected(video: Path, selected: set[int]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    with av.open(str(video)) as container:
        for index, item in enumerate(container.decode(video=0)):
            if index in selected:
                result[index] = item.to_ndarray(format="rgb24")
            if len(result) == len(selected):
                break
    missing = selected - result.keys()
    if missing:
        raise RuntimeError(f"video frames missing in {video}: {sorted(missing)}")
    return result


class FKProxy:
    """MuJoCo CAD FK proxy; calibration origin/sign is not hardware-validated."""

    def __init__(self) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(
            str(ROOT / "simulation/mujoco/assets/robotstudio_so101/scene.xml")
        )
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.joint(j).qposadr for j in JOINT_ORDER]
        self.site = self.model.site("gripperframe").id

    def xyz(self, rows_deg: np.ndarray) -> np.ndarray:
        out = []
        for row in rows_deg:
            for address, degrees in zip(self.qadr, row):
                self.data.qpos[address] = math.radians(float(degrees))
            self.mujoco.mj_forward(self.model, self.data)
            out.append(self.data.site_xpos[self.site].copy())
        return np.asarray(out)


def phase_for(label: str, offset: int | None) -> str:
    if label in ("initial", "early"):
        return label
    assert offset is not None
    if offset <= -20:
        return "pre_grasp"
    if offset < 0:
        return "final_approach"
    return "post_close"


def direction(delta: float, threshold: float) -> int:
    return 1 if delta > threshold else (-1 if delta < -threshold else 0)


def predicted_close_step(pred: np.ndarray, state_gripper: float, threshold: float) -> int | None:
    above = pred[:, 5] >= state_gripper + threshold
    for k in range(48):
        if bool(above[k] and above[k + 1] and above[k + 2]):
            return k
    return None


def summarize(rows: list[dict], phase: str) -> dict:
    subset = [r for r in rows if r["phase"] == phase]
    if not subset:
        return {"n": 0}
    def q(values):
        return {k: float(v) for k, v in zip(("mean", "median", "p95"), (np.mean(values), np.median(values), np.percentile(values, 95)))}
    result = {
        "n": len(subset),
        "mae": q([r["mae"] for r in subset]),
        "joint_mae": {j: q([r["joint_mae"][j] for r in subset]) for j in JOINT_ORDER},
        "direction_match_fraction": {
            j: float(np.mean([r["direction"][j]["match"] for r in subset])) for j in JOINT_ORDER
        },
        "endpoint_error_l2": {str(k): q([r["endpoint_error_l2"][str(k)] for r in subset]) for k in STEPS},
        "fk_xyz_error_m": {str(k): q([r["fk_xyz_error_m"][str(k)] for r in subset]) for k in STEPS},
    }
    timing = [r["gripper_close_timing_error_frames"] for r in subset if r["gripper_close_timing_error_frames"] is not None]
    result["gripper_close_timing_error_frames"] = q(timing) if timing else None
    return result


def plot_episode(out: Path, record: dict, pred: np.ndarray, gt: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    x = np.arange(50)
    for axis, joint, index in zip(axes.flat, JOINT_ORDER, range(6)):
        axis.plot(x, gt[:, index], label="GT", linewidth=2)
        axis.plot(x, pred[:, index], label="V1 predicted", linewidth=1.5)
        axis.set_title(joint)
        axis.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle(f"episode {record['episode']:02d}, frame {record['frame']}, offset {record['offset']}, MAE={record['mae']:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://100.75.147.72:9200")
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--expected-seed", type=int, default=20260815)
    parser.add_argument("--expected-model-substring", default="blue_cube_place_return_v1_fresh/checkpoints/007500")
    args = parser.parse_args()

    health = requests.get(args.server_url.rstrip("/") + "/health", timeout=10).json()
    if health.get("status") != "ok" or not health.get("model_loaded"):
        raise RuntimeError(f"server is not healthy: {health}")
    if health.get("inference_mode") != "deterministic" or health.get("inference_seed") != args.expected_seed:
        raise RuntimeError(f"effective inference mode/seed mismatch: {health}")
    if args.expected_model_substring not in str(health.get("model_id")):
        raise RuntimeError(f"model mismatch: {health.get('model_id')}")

    onset_doc = json.loads(ONSET_ANALYSIS.read_text())
    onset_by_ep = {int(x["episode"]): int(x["onset"]["gripper"]) for x in onset_doc["episode_onsets"]}
    thresholds = np.asarray([onset_doc["noise"][j]["threshold"] for j in JOINT_ORDER], dtype=float)
    old_manifest = json.loads(INITIAL_MANIFEST.read_text())
    initial_frames = {(int(x["episode"]), x["phase"]): int(x["frame"]) for x in old_manifest}
    fk = FKProxy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "chunks"
    plots_dir = args.output_dir / "plots"
    chunks_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    records: list[dict] = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sequence = 0
    with VLAHttpClient(VLAClientConfig(server_url=args.server_url, timeout_s=30, max_retries=2)) as client:
        for ep in range(41):
            frame_data = pd.read_parquet(args.dataset_root / "data/chunk-000" / f"file-{ep:03d}.parquet").sort_values("frame_index")
            actions = np.stack(frame_data["action"]).astype(float)
            states = np.stack(frame_data["observation.state"]).astype(float)
            close = onset_by_ep[ep]
            specs = [("initial", initial_frames[(ep, "initial")], None), ("early", initial_frames[(ep, "early")], None)]
            specs += [("grasp", min(max(close + off, 0), len(actions) - 1), off) for off in OFFSETS]
            selected = {frame for _, frame, _ in specs}
            images = {
                cam: decode_selected(args.dataset_root / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{ep:03d}.mp4", selected)
                for cam in ("workspace", "wrist")
            }
            for label, frame, offset in specs:
                state = {j: float(states[frame, i]) for i, j in enumerate(JOINT_ORDER)}
                response = client.predict_chunk(
                    session_id="v1-grasp-phase-20260815", task=TASK, sequence=sequence, state=state,
                    images={CAMERA_WORKSPACE_KEY: images["workspace"][frame], CAMERA_WRIST_KEY: images["wrist"][frame]},
                )
                if not response.ok or response.chunk is None:
                    raise RuntimeError(f"inference failed ep={ep} frame={frame}: {response.error_kind}: {response.error_message}")
                pred = np.asarray([[step[j] for j in JOINT_ORDER] for step in response.chunk], dtype=float)
                gt = chunk(actions, frame)
                error = np.abs(pred - gt)
                pred_xyz, gt_xyz = fk.xyz(pred), fk.xyz(gt)
                gt_close = max(0, close - frame) if close - frame < 50 else None
                pred_close = predicted_close_step(pred, states[frame, 5], thresholds[5])
                timing_error = pred_close - gt_close if pred_close is not None and gt_close is not None else None
                rec = {
                    "episode": ep, "label": label, "phase": phase_for(label, offset), "offset": offset,
                    "frame": frame, "close_onset": close, "mae": float(error.mean()),
                    "joint_mae": {j: float(error[:, i].mean()) for i, j in enumerate(JOINT_ORDER)},
                    "endpoint_error_l2": {str(k): float(np.linalg.norm(pred[k] - gt[k])) for k in STEPS},
                    "endpoint_joint_error": {str(k): {j: float(pred[k, i] - gt[k, i]) for i, j in enumerate(JOINT_ORDER)} for k in STEPS},
                    "direction": {}, "predicted_gripper_close_step": pred_close, "gt_gripper_close_step": gt_close,
                    "gripper_close_timing_error_frames": timing_error,
                    "lift_progression_error_step0_40": float((pred[40, 1]-pred[0, 1])-(gt[40, 1]-gt[0, 1])),
                    "elbow_progression_error_step0_40": float((pred[40, 2]-pred[0, 2])-(gt[40, 2]-gt[0, 2])),
                    "fk_xyz_error_m": {str(k): float(np.linalg.norm(pred_xyz[k] - gt_xyz[k])) for k in STEPS},
                    "model_id": response.model_id, "inference_latency_ms": response.inference_latency_ms,
                }
                for i, joint in enumerate(JOINT_ORDER):
                    pdirection = direction(pred[40, i] - pred[0, i], thresholds[i])
                    gdirection = direction(gt[40, i] - gt[0, i], thresholds[i])
                    rec["direction"][joint] = {"predicted": pdirection, "gt": gdirection, "match": pdirection == gdirection}
                name = f"ep{ep:02d}_{label}" if offset is None else f"ep{ep:02d}_offset_{offset:+04d}"
                np.savez_compressed(chunks_dir / f"{name}.npz", predicted=pred, gt=gt, state=states[frame])
                arrays[name] = (pred, gt)
                rec["chunk_path"] = str((chunks_dir / f"{name}.npz").resolve())
                records.append(rec)
                sequence += 1
                print(f"[{sequence:03d}/451] ep={ep:02d} frame={frame:03d} {label} {offset} MAE={rec['mae']:.3f}", flush=True)

    phases = ("initial", "early", "pre_grasp", "final_approach", "post_close")
    summary = {phase: summarize(records, phase) for phase in phases}
    grasp_records = [r for r in records if r["label"] == "grasp"]
    episode_mae = {ep: float(np.mean([r["mae"] for r in grasp_records if r["episode"] == ep])) for ep in range(41)}
    ordered = sorted(episode_mae, key=episode_mae.get)
    representatives = {"good": ordered[:3], "bad": ordered[-3:]}
    for category, episodes in representatives.items():
        for ep in episodes:
            candidates = [r for r in records if r["episode"] == ep and r["offset"] == -10]
            rec = candidates[0]
            name = f"ep{ep:02d}_offset_-010"
            plot_episode(plots_dir / f"{category}_ep{ep:02d}_final_approach.png", rec, *arrays[name])

    report = {
        "schema": "v1-grasp-phase-policy-vs-gt-v1", "health": health, "dataset_modified": False,
        "episodes": 41, "evaluations": len(records), "offsets": list(OFFSETS), "steps": list(STEPS),
        "onset_source": str(ONSET_ANALYSIS.resolve()), "target_semantics": "action[t:t+50], terminal-repeat padding",
        "fk_caveat": "MuJoCo CAD proxy, deg->rad sign=+1; physical calibration origin/sign not validated",
        "phase_summary": summary, "representatives": representatives, "records": records,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"phase_summary": summary, "representatives": representatives}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
