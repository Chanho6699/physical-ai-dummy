#!/usr/bin/env python3
"""V2-clean vs V3 start-coverage dataset comparison (read-only analysis, no merge/rename/training).

Background
----------
``scripts/analyze_grid35_v2_clean_dataset_coverage_imbalance.py`` established that
``data/so101_cube_xy_grid35_v2_clean`` (35 episodes) has a small (7.6% of frames), highly
uniform static/low-motion start segment (median 25 frames / ~0.83s), low start-pose diversity
across episodes, and that static-looking observations are nonetheless paired with large-mean
50-step target chunks - plausibly contributing to the observed shoulder_lift/elbow_flex
first-action overshoot.

``data/so101_cube_xy_start_coverage_v3`` (30 episodes) was collected to address this: 15
representative cube positions x2, hold times deliberately varied (~0.2/0.3/0.6/1.0/1.5s),
start poses varied around the old start, smoother first movement, ~11s/episode.

This script is a pure read-only comparison of the two datasets. It does **not** rename any
dataset, does **not** touch ``meta/tasks.parquet`` (task string stays
"Pick up the cube and place it in the target area." for both, even though V3's real behaviour is
a Pick & Drop ~10cm above the bin - this script does not correct that mislabeling), does **not**
merge the two datasets on disk, and does **not** train anything. Real cube x/y target
coordinates are not present in either dataset's metadata and this script never estimates or
fabricates them.

Definitions reused verbatim from the two prior V2-clean audits (kept identical on purpose so V2
numbers reproduce exactly and V3 numbers are comparable):
  - ``scripts/analyze_grid35_v2_clean_dataset_coverage_imbalance.py``: static/moving segmentation,
    frame-window coverage, per-joint action(t)-state(t) delta distributions, Safety Gate
    threshold coverage, whole-chunk mechanism check (chunk_size=50), start-pose pairwise-L2
    diversity.
  - ``scripts/analyze_grid35_v2_clean_start_segment_temporal_alignment.py``: state vs. action
    first-movement-frame detection, each against its **own** frame-0 baseline (state vs
    state[0], action vs action[0]) - this avoids a false "movement" trigger from the constant
    leader/follower static action-state offset that exists even while the arm is genuinely still.

Movement threshold (identical in both prior scripts): cumulative per-joint displacement from the
trajectory's own frame-0 value >= 1.0deg, sustained for >= 3 consecutive frames.

New in this script (V2 vs V3 comparison, sections A/G/H not present in the prior single-dataset
audits):
  A. Dataset integrity (episode/frame counts, fps, length distribution, state/action dims,
     camera features, NaN/Inf, episode-index and frame-index continuity).
  G. Potential conflicting immediate labels: since V3 deliberately varies hold time, near-
     identical start states across episodes could now carry different immediate
     (action(t)-state(t)) labels. Detected via pairwise state-L2 distance among each episode's
     own static-segment frames (state-detector-defined, capped at 90 frames/episode to keep the
     search a genuine "near start" comparison and bound compute), flagging cross-episode pairs
     with small state distance but large action(t)-state(t) difference on shoulder_lift/
     elbow_flex. Image similarity is intentionally NOT computed (no video decoding / no model
     inference) - see summary.md for why, and what would need to be checked next.
  H. Combined (V2+V3, literal concatenation, no new file written) re-run of the static-coverage,
     timing, diversity, and WOULD_CLAMP metrics.

Usage (same venv as the other lerobot-adjacent analysis scripts; no torch/lerobot/GPU needed)::

    source ~/lerobot/.venv/bin/activate
    python scripts/analyze_v2_vs_v3_start_coverage.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_ROOT = PROJECT_ROOT / "data" / "so101_cube_xy_grid35_v2_clean"
DEFAULT_V3_ROOT = PROJECT_ROOT / "data" / "so101_cube_xy_start_coverage_v3"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "grid35_v2_vs_start_coverage_v3"
DEFAULT_SAFETY_GATE_YAML = PROJECT_ROOT / "configs" / "safety_gate.yaml"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
KEY_JOINTS = ["shoulder_lift", "elbow_flex"]
FPS = 30

# Identical to both prior V2-clean audits - see module docstring.
MOVEMENT_THRESHOLD_DEG = 1.0
PERSISTENCE_FRAMES = 3
GROSS_STEP_MULTIPLIER = 5.0
CHUNK_SIZE = 50
FIXED_WINDOWS = {"frames_0_15": (0, 15), "frames_0_30": (0, 30)}
LARGE_ACTION_THRESHOLDS_DEG = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

# --- New for this script: conflicting-label candidate search (section G) -----------------------
# Cap per-episode window searched for conflicts to the episode's own static segment, capped at
# 90 frames (3s) so a rare "detector never found movement" episode can't blow up the search or
# dilute it with genuinely-moving frames far from the start.
CONFLICT_WINDOW_CAP_FRAMES = 90
# "Very close" start-state candidate: L2 across all 6 joints (degrees). sqrt(6)*1.0deg ~= 2.45deg
# would be "each joint individually at the 1deg movement-detector floor" - we round up slightly
# to 3.0deg so genuinely near-duplicate poses are caught without also catching visibly-different
# grid positions.
STATE_CLOSE_L2_THRESHOLD_DEG = 3.0
# Conflicting-action flag: the two frames' action(t)-state(t) deltas differ by >= this much on
# shoulder_lift or elbow_flex specifically (half of each joint's WOULD_CLAMP threshold - large
# enough to not be encoder/read noise, small enough to catch pre-clamp disagreement).
CONFLICT_ACTION_DIFF_DEG = 2.0


# --------------------------------------------------------------------------
# 1. Load dataset (adds frame_index/index continuity fields vs. the prior scripts)
# --------------------------------------------------------------------------


def load_episodes(dataset_root: Path) -> dict[int, dict[str, np.ndarray]]:
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
            columns=["action", "observation.state", "frame_index", "episode_index", "index", "timestamp"],
        )
        df = df[df["episode_index"] == ep].sort_values("frame_index")
        episodes[ep] = {
            "state": np.stack(df["observation.state"].to_numpy()).astype(np.float64),
            "action": np.stack(df["action"].to_numpy()).astype(np.float64),
            "frame_index": df["frame_index"].to_numpy(),
            "global_index": df["index"].to_numpy(),
            "timestamp": df["timestamp"].to_numpy(),
            "length": len(df),
        }
    return episodes


def load_info(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))


def load_tasks(dataset_root: Path) -> list[str]:
    df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    return df.index.tolist()


# --------------------------------------------------------------------------
# A. Dataset integrity
# --------------------------------------------------------------------------


def compute_dataset_integrity(dataset_root: Path, episodes: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    info = load_info(dataset_root)
    tasks = load_tasks(dataset_root)

    lengths = np.array([d["length"] for d in episodes.values()])
    eps_sorted = sorted(episodes.keys())

    # Episode-index continuity: exactly 0..N-1, each once.
    expected = list(range(len(episodes)))
    episode_index_continuous = eps_sorted == expected

    # Per-episode frame_index continuity: exactly 0..length-1, strictly increasing by 1.
    frame_index_issues = []
    for ep in eps_sorted:
        fi = episodes[ep]["frame_index"]
        expected_fi = np.arange(len(fi))
        if not np.array_equal(fi, expected_fi):
            frame_index_issues.append(int(ep))

    # Global `index` column continuity across the whole dataset (concat in episode order).
    global_idx = np.concatenate([episodes[ep]["global_index"] for ep in eps_sorted])
    global_index_continuous = bool(np.array_equal(global_idx, np.arange(len(global_idx))))

    # NaN/Inf check across state+action.
    nan_inf: dict[str, Any] = {"episodes_with_nan": [], "episodes_with_inf": []}
    for ep in eps_sorted:
        d = episodes[ep]
        both = np.concatenate([d["state"], d["action"]], axis=1)
        if np.isnan(both).any():
            nan_inf["episodes_with_nan"].append(int(ep))
        if np.isinf(both).any():
            nan_inf["episodes_with_inf"].append(int(ep))

    # Camera / feature check.
    features = info["features"]
    camera_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    camera_files_ok = {}
    for cam in camera_keys:
        cam_dir = dataset_root / "videos" / cam
        n_files = len(list(cam_dir.glob("chunk-*/file-*.mp4")))
        camera_files_ok[cam] = {"n_mp4_files": n_files}

    state_shape = features["observation.state"]["shape"]
    action_shape = features["action"]["shape"]

    return {
        "dataset_root": str(dataset_root),
        "codebase_version": info.get("codebase_version"),
        "fps_meta": info.get("fps"),
        "fps_used_in_analysis": FPS,
        "task_strings": tasks,
        "n_episodes_meta": info.get("total_episodes"),
        "n_episodes_loaded": len(episodes),
        "n_frames_meta": info.get("total_frames"),
        "n_frames_loaded": int(lengths.sum()),
        "episode_length_distribution": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "std": float(lengths.std()),
            "mean_seconds": float(lengths.mean() / FPS),
            "median_seconds": float(np.median(lengths) / FPS),
        },
        "state_dim": state_shape,
        "action_dim": action_shape,
        "joint_names": features["observation.state"]["names"],
        "camera_features": camera_files_ok,
        "nan_inf_check": nan_inf,
        "episode_index_continuous_0_to_N-1": episode_index_continuous,
        "frame_index_continuity_issues_episodes": frame_index_issues,
        "global_index_continuous": global_index_continuous,
    }


# --------------------------------------------------------------------------
# 2. Movement detection (state[0]/action[0] own-baseline, identical to temporal_alignment script)
# --------------------------------------------------------------------------


def first_movement_frame(traj: np.ndarray, ref: np.ndarray, threshold: float, persistence: int) -> int | None:
    disp = traj - ref[None, :]
    mag = np.max(np.abs(disp), axis=1)
    n = len(mag)
    for t in range(1, n):
        end = min(t + persistence, n)
        if end - t < persistence and t + persistence <= n:
            continue
        window = mag[t:end]
        if len(window) > 0 and np.all(window >= threshold):
            return t
    return None


def first_movement_frame_single_joint(traj_col: np.ndarray, ref_val: float, threshold: float, persistence: int) -> int | None:
    disp = np.abs(traj_col - ref_val)
    n = len(disp)
    for t in range(1, n):
        end = min(t + persistence, n)
        if end - t < persistence and t + persistence <= n:
            continue
        window = disp[t:end]
        if len(window) > 0 and np.all(window >= threshold):
            return t
    return None


def compute_first_movement(episodes: dict[int, dict[str, np.ndarray]]) -> dict[int, dict[str, Any]]:
    out = {}
    for ep, d in episodes.items():
        state, action = d["state"], d["action"]
        state0, action0 = state[0], action[0]
        state_fm = first_movement_frame(state, state0, MOVEMENT_THRESHOLD_DEG, PERSISTENCE_FRAMES)
        action_fm = first_movement_frame(action, action0, MOVEMENT_THRESHOLD_DEG, PERSISTENCE_FRAMES)
        out[ep] = {
            "length": d["length"],
            "state_first_movement_frame": state_fm,
            "state_first_movement_s": state_fm / FPS if state_fm is not None else None,
            "action_first_movement_frame": action_fm,
            "action_first_movement_s": action_fm / FPS if action_fm is not None else None,
        }
    return out


def summarize_timing(values: list[int | None], fps: int) -> dict[str, Any]:
    v = np.array([x for x in values if x is not None], dtype=float)
    n_none = sum(1 for x in values if x is None)
    if len(v) == 0:
        return {"n": 0, "n_never_moved": n_none}
    return {
        "n": int(len(v)),
        "n_never_moved": n_none,
        "mean_frame": float(np.mean(v)),
        "median_frame": float(np.median(v)),
        "p10_frame": float(np.percentile(v, 10)),
        "p90_frame": float(np.percentile(v, 90)),
        "mean_s": float(np.mean(v) / fps),
        "median_s": float(np.median(v) / fps),
        "p10_s": float(np.percentile(v, 10) / fps),
        "p90_s": float(np.percentile(v, 90) / fps),
    }


def compute_timing_summary(first_movement: dict[int, dict[str, Any]]) -> dict[str, Any]:
    state_vals = [v["state_first_movement_frame"] for v in first_movement.values()]
    action_vals = [v["action_first_movement_frame"] for v in first_movement.values()]
    return {
        "state_first_movement": summarize_timing(state_vals, FPS),
        "action_first_movement": summarize_timing(action_vals, FPS),
    }


# --------------------------------------------------------------------------
# 3. Static/moving segmentation + frame coverage (identical definition to coverage_imbalance script:
#    segmentation is state-detector-based)
# --------------------------------------------------------------------------


def compute_segments(episodes: dict[int, dict[str, np.ndarray]], first_movement: dict[int, dict[str, Any]]) -> dict[int, dict[str, int]]:
    out = {}
    for ep, d in episodes.items():
        length = d["length"]
        fm = first_movement[ep]["state_first_movement_frame"]
        static_length = fm if fm is not None else length
        out[ep] = {
            "length": length,
            "first_movement_frame": fm,
            "static_length": static_length,
            "moving_length": length - static_length,
        }
    return out


def compute_frame_coverage(episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]]) -> dict[str, Any]:
    total_frames = sum(d["length"] for d in episodes.values())
    out: dict[str, Any] = {"total_frames": total_frames, "n_episodes": len(episodes)}
    for name, (lo, hi) in FIXED_WINDOWS.items():
        n = sum(max(0, min(hi, d["length"]) - lo) for d in episodes.values())
        out[name] = {"n_frames": n, "fraction_of_total": n / total_frames}
    total_static = sum(s["static_length"] for s in segments.values())
    total_moving = sum(s["moving_length"] for s in segments.values())
    static_lengths = np.array([s["static_length"] for s in segments.values()], dtype=float)
    out["static_segment_data_driven"] = {"n_frames": total_static, "fraction_of_total": total_static / total_frames}
    out["moving_segment_data_driven"] = {"n_frames": total_moving, "fraction_of_total": total_moving / total_frames}
    out["static_length_distribution_frames"] = {
        "mean": float(static_lengths.mean()),
        "median": float(np.median(static_lengths)),
        "p10": float(np.percentile(static_lengths, 10)),
        "p90": float(np.percentile(static_lengths, 90)),
        "min": float(static_lengths.min()),
        "max": float(static_lengths.max()),
    }
    return out


# --------------------------------------------------------------------------
# 4. Immediate action safety (section D) - action(t)-state(t) on static-segment frames
# --------------------------------------------------------------------------


def load_safety_thresholds(safety_gate_yaml: Path) -> dict[str, Any]:
    raw = yaml.safe_load(safety_gate_yaml.read_text(encoding="utf-8"))
    excessive = raw["excessive_step_deg"]
    return {
        "would_clamp_threshold_deg": excessive,
        "reject_threshold_deg": {j: v * GROSS_STEP_MULTIPLIER for j, v in excessive.items()},
        "source_file": str(safety_gate_yaml),
    }


def compute_immediate_action_safety(
    episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]], thresholds: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Returns (pooled summary dict, per-episode-per-joint rows for CSV)."""
    wc = thresholds["would_clamp_threshold_deg"]
    rows = []
    pooled = {j: [] for j in JOINTS}
    for ep in sorted(episodes.keys()):
        d = episodes[ep]
        static_len = segments[ep]["static_length"]
        if static_len <= 0:
            continue
        delta = d["action"][:static_len] - d["state"][:static_len]
        for i, j in enumerate(JOINTS):
            abs_vals = np.abs(delta[:, i])
            pooled[j].append(abs_vals)
            rows.append(
                {
                    "episode": ep,
                    "joint": j,
                    "n_frames": static_len,
                    "mean_abs_delta_deg": float(np.mean(abs_vals)),
                    "median_abs_delta_deg": float(np.median(abs_vals)),
                    "p95_abs_delta_deg": float(np.percentile(abs_vals, 95)) if static_len > 1 else float(abs_vals[0]),
                    "max_abs_delta_deg": float(np.max(abs_vals)),
                    "would_clamp_threshold_deg": wc[j],
                    "frac_would_clamp": float(np.mean(abs_vals > wc[j])),
                }
            )

    summary: dict[str, Any] = {}
    for j in JOINTS:
        if not pooled[j]:
            summary[j] = None
            continue
        abs_vals = np.concatenate(pooled[j])
        summary[j] = {
            "n": int(len(abs_vals)),
            "mean_abs_delta_deg": float(np.mean(abs_vals)),
            "median_abs_delta_deg": float(np.median(abs_vals)),
            "p95_abs_delta_deg": float(np.percentile(abs_vals, 95)),
            "max_abs_delta_deg": float(np.max(abs_vals)),
            "would_clamp_threshold_deg": wc[j],
            "frac_would_clamp": float(np.mean(abs_vals > wc[j])),
        }
    return summary, rows


# --------------------------------------------------------------------------
# 5. Start-pose diversity (section E)
# --------------------------------------------------------------------------


def pairwise_l2(states: np.ndarray) -> dict[str, float]:
    n = states.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(states[i] - states[j])))
    dists = np.array(dists)
    return {
        "n_pairs": int(len(dists)),
        "min": float(np.min(dists)),
        "median": float(np.median(dists)),
        "mean": float(np.mean(dists)),
        "p95": float(np.percentile(dists, 95)),
        "max": float(np.max(dists)),
    }


def compute_start_pose_diversity(episodes: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    eps = sorted(episodes.keys())
    start_states = np.stack([episodes[ep]["state"][0] for ep in eps])

    per_joint_stats = {}
    for i, j in enumerate(JOINTS):
        per_joint_stats[j] = {
            "std": float(np.std(start_states[:, i])),
            "range": float(np.ptp(start_states[:, i])),
            "min": float(np.min(start_states[:, i])),
            "max": float(np.max(start_states[:, i])),
        }

    return {
        "n_episodes": len(eps),
        "per_joint_start_frame0_stats": per_joint_stats,
        "start_frame0_pairwise_l2_deg": pairwise_l2(start_states),
    }


# --------------------------------------------------------------------------
# 6. Chunk future-motion correlation (section F)
# --------------------------------------------------------------------------


def compute_chunk_future_motion(
    episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]], thresholds: dict[str, Any], chunk_size: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wc = thresholds["would_clamp_threshold_deg"]
    pooled_chunk0 = {j: [] for j in KEY_JOINTS}
    pooled_chunkmean = {j: [] for j in KEY_JOINTS}
    pooled_chunkmax = {j: [] for j in KEY_JOINTS}
    n_static_total = 0
    n_chunkmean_exceeds_wc = {j: 0 for j in KEY_JOINTS}

    rows = []
    for ep in sorted(episodes.keys()):
        d = episodes[ep]
        state, action, length = d["state"], d["action"], d["length"]
        static_len = segments[ep]["static_length"]
        if static_len <= 0:
            continue
        ep_chunk0 = {j: [] for j in KEY_JOINTS}
        ep_chunkmean = {j: [] for j in KEY_JOINTS}
        ep_chunkmax = {j: [] for j in KEY_JOINTS}
        ep_exceeds = {j: 0 for j in KEY_JOINTS}
        for t in range(static_len):
            n_static_total += 1
            k_idx = np.arange(chunk_size)
            frame_idx = np.minimum(t + k_idx, length - 1)
            chunk = action[frame_idx]
            delta = chunk - state[t][None, :]
            for j in KEY_JOINTS:
                ji = JOINTS.index(j)
                col = np.abs(delta[:, ji])
                pooled_chunk0[j].append(col[0])
                pooled_chunkmean[j].append(float(np.mean(col)))
                pooled_chunkmax[j].append(float(np.max(col)))
                ep_chunk0[j].append(col[0])
                ep_chunkmean[j].append(float(np.mean(col)))
                ep_chunkmax[j].append(float(np.max(col)))
                if np.mean(col) > wc[j]:
                    n_chunkmean_exceeds_wc[j] += 1
                    ep_exceeds[j] += 1

        for j in KEY_JOINTS:
            rows.append(
                {
                    "episode": ep,
                    "joint": j,
                    "n_static_frames": static_len,
                    "chunk0_abs_delta_mean": float(np.mean(ep_chunk0[j])),
                    "chunk_mean_abs_delta_mean": float(np.mean(ep_chunkmean[j])),
                    "chunk_max_abs_delta_mean": float(np.mean(ep_chunkmax[j])),
                    "frac_static_frames_chunk_mean_exceeds_would_clamp": ep_exceeds[j] / static_len,
                }
            )

    out: dict[str, Any] = {"chunk_size": chunk_size, "n_static_frames_total": n_static_total}
    for j in KEY_JOINTS:
        if not pooled_chunk0[j]:
            out[j] = None
            continue
        out[j] = {
            "chunk0_abs_delta": {
                "mean": float(np.mean(pooled_chunk0[j])),
                "median": float(np.median(pooled_chunk0[j])),
                "p95": float(np.percentile(pooled_chunk0[j], 95)),
            },
            "chunk_mean_abs_delta_over_all_k": {
                "mean": float(np.mean(pooled_chunkmean[j])),
                "median": float(np.median(pooled_chunkmean[j])),
                "p95": float(np.percentile(pooled_chunkmean[j], 95)),
            },
            "chunk_max_abs_delta_over_all_k": {
                "mean": float(np.mean(pooled_chunkmax[j])),
                "median": float(np.median(pooled_chunkmax[j])),
                "p95": float(np.percentile(pooled_chunkmax[j], 95)),
            },
            "fraction_static_frames_where_chunk_mean_exceeds_would_clamp": (
                n_chunkmean_exceeds_wc[j] / n_static_total if n_static_total else None
            ),
        }
    return out, rows


# --------------------------------------------------------------------------
# 7. Conflicting immediate labels (section G)
# --------------------------------------------------------------------------


def compute_conflicting_labels(
    tagged_episodes: dict[tuple[str, int], dict[str, np.ndarray]],
    tagged_segments: dict[tuple[str, int], dict[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pairwise search, restricted to each episode's own static-segment window (capped), for
    cross-episode frame pairs whose state is nearly identical but whose immediate
    action(t)-state(t) differs a lot on shoulder_lift/elbow_flex.
    """
    sl_i = JOINTS.index("shoulder_lift")
    ef_i = JOINTS.index("elbow_flex")

    records = []  # (dataset, episode, frame_t, state_vec, action_delta_vec)
    for key, d in tagged_episodes.items():
        dataset_tag, ep = key
        static_len = min(tagged_segments[key]["static_length"], CONFLICT_WINDOW_CAP_FRAMES)
        if static_len <= 0:
            continue
        state = d["state"][:static_len]
        action_delta = d["action"][:static_len] - d["state"][:static_len]
        for t in range(static_len):
            records.append((dataset_tag, ep, t, state[t], action_delta[t]))

    if len(records) < 2:
        return [], {"total_close_pairs": 0, "total_close_within_v2": 0, "total_close_within_v3": 0, "total_close_cross": 0}

    states = np.stack([r[3] for r in records])
    deltas = np.stack([r[4] for r in records])
    n = len(records)

    candidates = []
    close_counts = {"total_close_pairs": 0, "total_close_within_v2": 0, "total_close_within_v3": 0, "total_close_cross": 0}
    # O(n^2) but bounded by CONFLICT_WINDOW_CAP_FRAMES per episode - fine for tens of episodes.
    for i in range(n):
        for j in range(i + 1, n):
            if records[i][0] == records[j][0] and records[i][1] == records[j][1]:
                continue  # skip same-episode pairs (trivially close/similar by construction)
            state_dist = float(np.linalg.norm(states[i] - states[j]))
            if state_dist > STATE_CLOSE_L2_THRESHOLD_DEG:
                continue
            close_counts["total_close_pairs"] += 1
            if records[i][0] == "v2_clean" and records[j][0] == "v2_clean":
                close_counts["total_close_within_v2"] += 1
            elif records[i][0] == "v3_start_coverage" and records[j][0] == "v3_start_coverage":
                close_counts["total_close_within_v3"] += 1
            else:
                close_counts["total_close_cross"] += 1
            sl_diff = float(abs(deltas[i, sl_i] - deltas[j, sl_i]))
            ef_diff = float(abs(deltas[i, ef_i] - deltas[j, ef_i]))
            if sl_diff < CONFLICT_ACTION_DIFF_DEG and ef_diff < CONFLICT_ACTION_DIFF_DEG:
                continue
            candidates.append(
                {
                    "dataset_a": records[i][0],
                    "episode_a": records[i][1],
                    "frame_a": records[i][2],
                    "dataset_b": records[j][0],
                    "episode_b": records[j][1],
                    "frame_b": records[j][2],
                    "state_l2_dist_deg": state_dist,
                    "shoulder_lift_action_delta_a": float(deltas[i, sl_i]),
                    "shoulder_lift_action_delta_b": float(deltas[j, sl_i]),
                    "shoulder_lift_diff_deg": sl_diff,
                    "elbow_flex_action_delta_a": float(deltas[i, ef_i]),
                    "elbow_flex_action_delta_b": float(deltas[j, ef_i]),
                    "elbow_flex_diff_deg": ef_diff,
                    "max_key_joint_diff_deg": max(sl_diff, ef_diff),
                }
            )

    candidates.sort(key=lambda r: r["max_key_joint_diff_deg"], reverse=True)
    return candidates, close_counts


# --------------------------------------------------------------------------
# Combined (section H): literal in-memory concatenation of V2+V3, no file written
# --------------------------------------------------------------------------


def build_combined(v2_episodes, v3_episodes) -> dict[int, dict[str, np.ndarray]]:
    combined = {}
    for ep, d in v2_episodes.items():
        combined[ep] = d
    offset = 1000  # avoid collision; V3 episode indices become 1000..1029
    for ep, d in v3_episodes.items():
        combined[offset + ep] = d
    return combined


# --------------------------------------------------------------------------
# Per-dataset pipeline
# --------------------------------------------------------------------------


def analyze_dataset(dataset_root: Path | None, episodes: dict[int, dict[str, np.ndarray]], thresholds: dict[str, Any], label: str) -> dict[str, Any]:
    first_movement = compute_first_movement(episodes)
    timing_summary = compute_timing_summary(first_movement)
    segments = compute_segments(episodes, first_movement)
    frame_coverage = compute_frame_coverage(episodes, segments)
    safety_summary, safety_rows = compute_immediate_action_safety(episodes, segments, thresholds)
    diversity = compute_start_pose_diversity(episodes)
    chunk_summary, chunk_rows = compute_chunk_future_motion(episodes, segments, thresholds, CHUNK_SIZE)

    result = {
        "label": label,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "n_episodes": len(episodes),
        "first_movement": first_movement,
        "timing_summary": timing_summary,
        "segments": segments,
        "frame_coverage": frame_coverage,
        "immediate_action_safety_summary": safety_summary,
        "start_pose_diversity": diversity,
        "chunk_future_motion_summary": chunk_summary,
    }
    return result, safety_rows, chunk_rows


# --------------------------------------------------------------------------
# CSV writers
# --------------------------------------------------------------------------


def write_episode_start_timing_csv(out_path: Path, datasets: dict[str, tuple[dict, dict]]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "episode",
                "length",
                "state_first_movement_frame",
                "state_first_movement_s",
                "action_first_movement_frame",
                "action_first_movement_s",
            ]
        )
        for label, (episodes, first_movement) in datasets.items():
            for ep in sorted(episodes.keys()):
                mv = first_movement[ep]
                w.writerow(
                    [
                        label,
                        ep,
                        mv["length"],
                        mv["state_first_movement_frame"],
                        f"{mv['state_first_movement_s']:.4f}" if mv["state_first_movement_s"] is not None else "",
                        mv["action_first_movement_frame"],
                        f"{mv['action_first_movement_s']:.4f}" if mv["action_first_movement_s"] is not None else "",
                    ]
                )


def write_static_coverage_csv(out_path: Path, datasets: dict[str, tuple[dict, dict]]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "episode", "length", "first_movement_frame", "static_length", "moving_length", "static_fraction"])
        for label, (episodes, segments) in datasets.items():
            for ep in sorted(episodes.keys()):
                s = segments[ep]
                w.writerow([label, ep, s["length"], s["first_movement_frame"], s["static_length"], s["moving_length"], f"{s['static_length']/s['length']:.4f}"])


def write_start_pose_diversity_csv(out_path: Path, diversities: dict[str, dict[str, Any]]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "category", "key", "value"])
        for label, div in diversities.items():
            for j in JOINTS:
                s = div["per_joint_start_frame0_stats"][j]
                w.writerow([label, "per_joint_start_std_deg", j, f"{s['std']:.4f}"])
                w.writerow([label, "per_joint_start_range_deg", j, f"{s['range']:.4f}"])
            p = div["start_frame0_pairwise_l2_deg"]
            for stat in ["min", "median", "mean", "p95", "max"]:
                w.writerow([label, "start_frame0_pairwise_l2_deg", stat, f"{p[stat]:.4f}"])
            w.writerow([label, "start_frame0_pairwise_l2_deg", "n_pairs", p["n_pairs"]])


def write_immediate_action_safety_csv(out_path: Path, rows_by_dataset: dict[str, list[dict[str, Any]]]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "episode",
                "joint",
                "n_frames",
                "mean_abs_delta_deg",
                "median_abs_delta_deg",
                "p95_abs_delta_deg",
                "max_abs_delta_deg",
                "would_clamp_threshold_deg",
                "frac_would_clamp",
            ]
        )
        for label, rows in rows_by_dataset.items():
            for r in rows:
                w.writerow(
                    [
                        label,
                        r["episode"],
                        r["joint"],
                        r["n_frames"],
                        f"{r['mean_abs_delta_deg']:.4f}",
                        f"{r['median_abs_delta_deg']:.4f}",
                        f"{r['p95_abs_delta_deg']:.4f}",
                        f"{r['max_abs_delta_deg']:.4f}",
                        r["would_clamp_threshold_deg"],
                        f"{r['frac_would_clamp']:.4f}",
                    ]
                )


def write_chunk_future_motion_csv(out_path: Path, rows_by_dataset: dict[str, list[dict[str, Any]]]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "episode",
                "joint",
                "n_static_frames",
                "chunk0_abs_delta_mean_deg",
                "chunk_mean_abs_delta_mean_deg",
                "chunk_max_abs_delta_mean_deg",
                "frac_static_frames_chunk_mean_exceeds_would_clamp",
            ]
        )
        for label, rows in rows_by_dataset.items():
            for r in rows:
                w.writerow(
                    [
                        label,
                        r["episode"],
                        r["joint"],
                        r["n_static_frames"],
                        f"{r['chunk0_abs_delta_mean']:.4f}",
                        f"{r['chunk_mean_abs_delta_mean']:.4f}",
                        f"{r['chunk_max_abs_delta_mean']:.4f}",
                        f"{r['frac_static_frames_chunk_mean_exceeds_would_clamp']:.4f}",
                    ]
                )


def write_conflicting_labels_csv(out_path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = [
        "dataset_a",
        "episode_a",
        "frame_a",
        "dataset_b",
        "episode_b",
        "frame_b",
        "state_l2_dist_deg",
        "shoulder_lift_action_delta_a",
        "shoulder_lift_action_delta_b",
        "shoulder_lift_diff_deg",
        "elbow_flex_action_delta_a",
        "elbow_flex_action_delta_b",
        "elbow_flex_diff_deg",
        "max_key_joint_diff_deg",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in candidates:
            row = dict(r)
            for k in fields:
                if isinstance(row[k], float):
                    row[k] = f"{row[k]:.4f}"
            w.writerow(row)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--safety-gate-yaml", type=Path, default=DEFAULT_SAFETY_GATE_YAML)
    parser.add_argument("--max-conflict-candidates", type=int, default=200, help="cap rows written to conflicting_label_candidates.csv")
    args = parser.parse_args()

    v2_root = args.v2_root.resolve()
    v3_root = args.v3_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[v2vsv3] loading V2-clean (expect 35 episodes)")
    v2_episodes = load_episodes(v2_root)
    assert len(v2_episodes) == 35, f"expected 35 V2 episodes, found {len(v2_episodes)}"

    print("[v2vsv3] loading V3 start-coverage (expect 30 episodes)")
    v3_episodes = load_episodes(v3_root)
    assert len(v3_episodes) == 30, f"expected 30 V3 episodes, found {len(v3_episodes)}"

    thresholds = load_safety_thresholds(args.safety_gate_yaml)

    print("[v2vsv3] A: dataset integrity")
    integrity_v2 = compute_dataset_integrity(v2_root, v2_episodes)
    integrity_v3 = compute_dataset_integrity(v3_root, v3_episodes)

    print("[v2vsv3] B-F: per-dataset timing / coverage / safety / diversity / chunk analysis")
    result_v2, safety_rows_v2, chunk_rows_v2 = analyze_dataset(v2_root, v2_episodes, thresholds, "v2_clean")
    result_v3, safety_rows_v3, chunk_rows_v3 = analyze_dataset(v3_root, v3_episodes, thresholds, "v3_start_coverage")

    print("[v2vsv3] H: combined V2+V3 (in-memory concat, no file written)")
    combined_episodes = build_combined(v2_episodes, v3_episodes)
    result_combined, safety_rows_combined, chunk_rows_combined = analyze_dataset(None, combined_episodes, thresholds, "combined_v2_plus_v3")

    print("[v2vsv3] G: conflicting immediate-label candidate search")
    tagged_episodes = {}
    tagged_segments = {}
    for ep, d in v2_episodes.items():
        tagged_episodes[("v2_clean", ep)] = d
        tagged_segments[("v2_clean", ep)] = result_v2["segments"][ep]
    for ep, d in v3_episodes.items():
        tagged_episodes[("v3_start_coverage", ep)] = d
        tagged_segments[("v3_start_coverage", ep)] = result_v3["segments"][ep]
    conflict_candidates, close_counts = compute_conflicting_labels(tagged_episodes, tagged_segments)
    print(f"[v2vsv3]   {len(conflict_candidates)} candidate pairs found (state_l2<={STATE_CLOSE_L2_THRESHOLD_DEG}deg, "
          f"key-joint action diff>={CONFLICT_ACTION_DIFF_DEG}deg)")

    # ---- write CSVs ----
    write_episode_start_timing_csv(
        out_dir / "episode_start_timing.csv",
        {"v2_clean": (v2_episodes, result_v2["first_movement"]), "v3_start_coverage": (v3_episodes, result_v3["first_movement"])},
    )
    write_static_coverage_csv(
        out_dir / "static_coverage.csv",
        {"v2_clean": (v2_episodes, result_v2["segments"]), "v3_start_coverage": (v3_episodes, result_v3["segments"])},
    )
    write_start_pose_diversity_csv(
        out_dir / "start_pose_diversity.csv",
        {
            "v2_clean": result_v2["start_pose_diversity"],
            "v3_start_coverage": result_v3["start_pose_diversity"],
            "combined_v2_plus_v3": result_combined["start_pose_diversity"],
        },
    )
    write_immediate_action_safety_csv(
        out_dir / "immediate_action_safety.csv",
        {"v2_clean": safety_rows_v2, "v3_start_coverage": safety_rows_v3},
    )
    write_chunk_future_motion_csv(
        out_dir / "chunk_future_motion.csv",
        {"v2_clean": chunk_rows_v2, "v3_start_coverage": chunk_rows_v3},
    )
    write_conflicting_labels_csv(out_dir / "conflicting_label_candidates.csv", conflict_candidates[: args.max_conflict_candidates])

    for p in [
        "episode_start_timing.csv",
        "static_coverage.csv",
        "start_pose_diversity.csv",
        "immediate_action_safety.csv",
        "chunk_future_motion.csv",
        "conflicting_label_candidates.csv",
    ]:
        print(f"[v2vsv3] wrote {out_dir / p}")

    # split conflict candidates by cross/within-dataset for reporting
    n_within_v3 = sum(1 for c in conflict_candidates if c["dataset_a"] == "v3_start_coverage" and c["dataset_b"] == "v3_start_coverage")
    n_within_v2 = sum(1 for c in conflict_candidates if c["dataset_a"] == "v2_clean" and c["dataset_b"] == "v2_clean")
    n_cross = len(conflict_candidates) - n_within_v3 - n_within_v2

    summary: dict[str, Any] = {
        "dataset_integrity": {"v2_clean": integrity_v2, "v3_start_coverage": integrity_v3},
        "safety_thresholds": thresholds,
        "v2_clean": {k: v for k, v in result_v2.items() if k not in ("first_movement", "segments")},
        "v3_start_coverage": {k: v for k, v in result_v3.items() if k not in ("first_movement", "segments")},
        "combined_v2_plus_v3": {k: v for k, v in result_combined.items() if k not in ("first_movement", "segments")},
        "conflicting_labels": {
            "state_close_l2_threshold_deg": STATE_CLOSE_L2_THRESHOLD_DEG,
            "conflict_action_diff_threshold_deg": CONFLICT_ACTION_DIFF_DEG,
            "conflict_window_cap_frames": CONFLICT_WINDOW_CAP_FRAMES,
            "n_candidates_total": len(conflict_candidates),
            "n_candidates_within_v2": n_within_v2,
            "n_candidates_within_v3": n_within_v3,
            "n_candidates_cross_v2_v3": n_cross,
            "close_state_pair_counts": close_counts,
            "conflict_rate_within_v2": (n_within_v2 / close_counts["total_close_within_v2"]) if close_counts["total_close_within_v2"] else None,
            "conflict_rate_within_v3": (n_within_v3 / close_counts["total_close_within_v3"]) if close_counts["total_close_within_v3"] else None,
            "conflict_rate_cross": (n_cross / close_counts["total_close_cross"]) if close_counts["total_close_cross"] else None,
            "top_5": conflict_candidates[:5],
            "note": (
                "State-distance-only screen (no image embedding / no model inference, by design - "
                "'저비용' constraint). A flagged pair means two near-identical proprioceptive states "
                "(<=3deg L2 across 6 joints) carry substantially different immediate labels "
                "(>=2deg on shoulder_lift or elbow_flex action(t)-state(t)); it does not by itself "
                "confirm the *visual* context was also near-identical - that would need a frame-level "
                "check (see summary.md limitations)."
            ),
        },
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"[v2vsv3] wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
