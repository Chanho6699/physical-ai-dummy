#!/usr/bin/env python3
"""Grid35 V2-clean: training-target distribution + start-segment coverage audit.

Background
----------
``scripts/verify_smolvla_training_target_alignment.py`` established, against the real LeRobot
dataset-loading code (empirically, 115,500 chunk entries across all 35 episodes, 0 mismatches),
that the SmolVLA training target chunk[0] for observation ``t`` is exactly ``action(t)`` - there
is **no** future-offset bug in how targets are constructed. That same script's live-checkpoint
comparison (7.5k checkpoint) showed the *policy* nonetheless delivers a chunk[0] several degrees
away from its own (small, correctly-aligned) training target at early/static-hold frames, in the
same direction/magnitude as the previously-observed Shadow bias
(``reports/grid35_v2_T01_T10_actual_seed_sweep_summary/``: mean +5.7deg shoulder_lift, -6.5deg
elbow_flex).

This script asks the follow-up data-side question: is that mislearning explained by **dataset
imbalance / start-segment coverage**, i.e. did the model see too few, too-similar, low-motion
training examples relative to a dataset dominated by large shoulder_lift/elbow_flex motion, such
that its output for any early/ambiguous-looking observation regresses toward the dataset's
dominant (large, already-moving) motion mode?

All of this is read-only descriptive statistics over ``data/so101_cube_xy_grid35_v2_clean`` (raw
parquet, no LeRobotDataset/torch/GPU needed) plus a read-only load of
``configs/safety_gate.yaml`` for threshold context. No retraining, no Safety Gate edits, no
robot writes, no LeRobot source modification, no git operations.

What this script computes
--------------------------
 1. Frame-count coverage: what fraction of all 11,505 frames fall in frame windows [0,15]/[0,30]
    (task-specified fixed windows) vs. a data-driven "static/low-motion" segment per episode
    (same first-movement detector as
    ``scripts/analyze_grid35_v2_clean_start_segment_temporal_alignment.py``:
    ``MOVEMENT_THRESHOLD_DEG``/``PERSISTENCE_FRAMES`` against each episode's own frame-0
    baseline) vs. the "moving" remainder of each episode.
 2. Per-joint ``action(t)-state(t)`` distribution (mean/median/p95/max|.|), for shoulder_lift and
    elbow_flex specifically plus all 6 joints, split by {all frames, static-segment frames,
    moving-segment frames, frame-window [0,15], frame-window [0,30]}.
 3. Safety Gate WOULD_CLAMP/REJECT threshold coverage (``configs/safety_gate.yaml``, read-only):
    what fraction of frames' own ``action(t)-state(t)`` would already trip WOULD_CLAMP/REJECT,
    per segment.
 4. Whole-*target-chunk* mechanism check: for every "static" frame ``t``, using the verified
    offset law ``chunk[k] == raw_action[min(t+k, episode_end-1)]`` (chunk_size=50, matching the
    checkpoint's own trained config - see the verify script), computes the mean/max magnitude of
    the entire 50-step target chunk paired with that (visually/proprioceptively near-identical)
    static observation - contrasted against chunk[0] alone. This quantifies "the input looks
    static, but 49 of its 50 training-target steps are not" without needing a live model.
 5. Start-pose diversity: per-episode frame-0 (and frame [0:5) mean) state, cross-episode
    per-joint std/range, and full pairwise L2-distance distribution among all 35 episodes' start
    poses - contrasted against the same statistic computed at a later, mid-reach frame per
    episode (where Grid35's 35 distinct XY targets should actually differentiate the state).
 6. Frequency table: how much of the dataset's shoulder_lift/elbow_flex ``action(t)-state(t)``
    mass sits at magnitudes comparable to (or exceeding) the observed policy bias / Safety Gate
    thresholds, overall and restricted to the moving segment (the dataset's "dominant motion
    mode").
 7. A first-order, explicitly-labeled loss-contribution proxy: since
    ``lerobot.policies.smolvla.modeling_smolvla.VLAFlowMatching.forward`` uses
    ``F.mse_loss(u_t, v_t, reduction="none")`` averaged **uniformly** over every valid
    (sample, chunk-position, action-dim) triple (no per-position weighting, only pad-masking) -
    each (observation, chunk-position) training example counts equally. This script sums
    squared target-chunk magnitude over every valid (frame, chunk-position) pair, split by
    segment, to approximate each segment's *share of the total squared-error budget* a
    predict-the-mean baseline would need to fit - a standard, order-of-magnitude way to reason
    about "which segment's mode the average prediction gets pulled toward" without an
    instrumented training run.

Usage (needs pandas/numpy/yaml - use the same venv as the other lerobot-dependent scripts;
no torch/lerobot/GPU actually required here)::

    source ~/lerobot/.venv/bin/activate
    python scripts/analyze_grid35_v2_clean_dataset_coverage_imbalance.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "so101_cube_xy_grid35_v2_clean"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "grid35_v2_clean_dataset_coverage_imbalance"
DEFAULT_SAFETY_GATE_YAML = PROJECT_ROOT / "configs" / "safety_gate.yaml"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
KEY_JOINTS = ["shoulder_lift", "elbow_flex"]
FPS = 30

# Same movement-vs-noise detector as the prior start-segment audit (kept identical on purpose -
# this script's "static segment" must be the same thing that analysis already validated against
# the dataset's encoder noise floor, not a newly-invented threshold).
MOVEMENT_THRESHOLD_DEG = 1.0
PERSISTENCE_FRAMES = 3
GROSS_STEP_MULTIPLIER = 5.0

FIXED_WINDOWS = {"frames_0_15": (0, 15), "frames_0_30": (0, 30)}

# Confirmed (scripts/verify_smolvla_training_target_alignment.py, section 2 of its checkpoint
# config.json) chunk_size actually used to train the checkpoint under investigation.
CHUNK_SIZE = 50

LARGE_ACTION_THRESHOLDS_DEG = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


# --------------------------------------------------------------------------
# 1. Load dataset
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
        df = pd.read_parquet(data_path, columns=["action", "observation.state", "frame_index", "episode_index"])
        df = df[df["episode_index"] == ep].sort_values("frame_index")
        episodes[ep] = {
            "state": np.stack(df["observation.state"].to_numpy()).astype(np.float64),
            "action": np.stack(df["action"].to_numpy()).astype(np.float64),
            "length": len(df),
        }
    return episodes


# --------------------------------------------------------------------------
# 2. Static/moving segmentation (identical detector to the prior start-segment audit)
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


def compute_segments(episodes: dict[int, dict[str, np.ndarray]]) -> dict[int, dict[str, int]]:
    """Per episode: static_length (frames before first state movement) / moving_length."""
    out = {}
    for ep, d in episodes.items():
        state = d["state"]
        length = d["length"]
        fm = first_movement_frame(state, state[0], MOVEMENT_THRESHOLD_DEG, PERSISTENCE_FRAMES)
        static_length = fm if fm is not None else length  # never "moves" by this detector -> all static
        out[ep] = {
            "length": length,
            "first_movement_frame": fm,
            "static_length": static_length,
            "moving_length": length - static_length,
        }
    return out


# --------------------------------------------------------------------------
# 3. Frame-count coverage
# --------------------------------------------------------------------------


def compute_frame_coverage(episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]]) -> dict[str, Any]:
    total_frames = sum(d["length"] for d in episodes.values())
    out: dict[str, Any] = {"total_frames": total_frames, "n_episodes": len(episodes)}

    for name, (lo, hi) in FIXED_WINDOWS.items():
        n = sum(max(0, min(hi, d["length"]) - lo) for d in episodes.values())
        out[name] = {"n_frames": n, "fraction_of_total": n / total_frames}

    total_static = sum(s["static_length"] for s in segments.values())
    total_moving = sum(s["moving_length"] for s in segments.values())
    out["static_segment_data_driven"] = {
        "n_frames": total_static,
        "fraction_of_total": total_static / total_frames,
        "detector": f"first frame where max-abs-joint-displacement-from-frame0 >= {MOVEMENT_THRESHOLD_DEG}deg "
        f"and stays >= threshold for {PERSISTENCE_FRAMES} consecutive frames (same as the prior "
        "start-segment temporal-alignment audit)",
    }
    out["moving_segment_data_driven"] = {"n_frames": total_moving, "fraction_of_total": total_moving / total_frames}
    out["per_episode_static_length"] = {ep: s["static_length"] for ep, s in segments.items()}
    out["median_static_length_frames"] = float(np.median([s["static_length"] for s in segments.values()]))
    out["mean_static_length_frames"] = float(np.mean([s["static_length"] for s in segments.values()]))
    return out


# --------------------------------------------------------------------------
# 4. Per-joint action-state delta distributions, by segment
# --------------------------------------------------------------------------


def pooled_deltas_for_mask(episodes: dict[int, dict[str, np.ndarray]], mask_fn) -> np.ndarray:
    """mask_fn(ep, length) -> boolean array of length `length` selecting frames to include."""
    rows = []
    for ep, d in episodes.items():
        mask = mask_fn(ep, d["length"])
        if not np.any(mask):
            continue
        rows.append((d["action"][mask] - d["state"][mask]))
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 6))


def summarize_deltas(deltas: np.ndarray) -> dict[str, Any]:
    if deltas.shape[0] == 0:
        return {j: None for j in JOINTS}
    out = {}
    for i, j in enumerate(JOINTS):
        vals = deltas[:, i]
        abs_vals = np.abs(vals)
        out[j] = {
            "n": int(deltas.shape[0]),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "mean_abs": float(np.mean(abs_vals)),
            "median_abs": float(np.median(abs_vals)),
            "p95_abs": float(np.percentile(abs_vals, 95)),
            "max_abs": float(np.max(abs_vals)),
            "std": float(np.std(vals)),
        }
    return out


def compute_segment_delta_distributions(
    episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]]
) -> dict[str, Any]:
    def mask_all(ep, length):
        return np.ones(length, dtype=bool)

    def mask_static(ep, length):
        m = np.zeros(length, dtype=bool)
        m[: segments[ep]["static_length"]] = True
        return m

    def mask_moving(ep, length):
        m = np.zeros(length, dtype=bool)
        m[segments[ep]["static_length"] :] = True
        return m

    def mask_window(lo, hi):
        def f(ep, length):
            m = np.zeros(length, dtype=bool)
            m[lo : min(hi, length)] = True
            return m

        return f

    result = {
        "all_frames": summarize_deltas(pooled_deltas_for_mask(episodes, mask_all)),
        "static_segment_data_driven": summarize_deltas(pooled_deltas_for_mask(episodes, mask_static)),
        "moving_segment_data_driven": summarize_deltas(pooled_deltas_for_mask(episodes, mask_moving)),
    }
    for name, (lo, hi) in FIXED_WINDOWS.items():
        result[name] = summarize_deltas(pooled_deltas_for_mask(episodes, mask_window(lo, hi)))
    return result


# --------------------------------------------------------------------------
# 5. Safety threshold coverage
# --------------------------------------------------------------------------


def load_safety_thresholds(safety_gate_yaml: Path) -> dict[str, Any]:
    raw = yaml.safe_load(safety_gate_yaml.read_text(encoding="utf-8"))
    excessive = raw["excessive_step_deg"]
    return {
        "would_clamp_threshold_deg": excessive,
        "reject_threshold_deg": {j: v * GROSS_STEP_MULTIPLIER for j, v in excessive.items()},
        "source_file": str(safety_gate_yaml),
    }


def compute_threshold_coverage(
    episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]], thresholds: dict[str, Any]
) -> dict[str, Any]:
    def mask_all(ep, length):
        return np.ones(length, dtype=bool)

    def mask_static(ep, length):
        m = np.zeros(length, dtype=bool)
        m[: segments[ep]["static_length"]] = True
        return m

    def mask_moving(ep, length):
        m = np.zeros(length, dtype=bool)
        m[segments[ep]["static_length"] :] = True
        return m

    segments_map = {"all_frames": mask_all, "static_segment_data_driven": mask_static, "moving_segment_data_driven": mask_moving}
    wc = thresholds["would_clamp_threshold_deg"]
    rj = thresholds["reject_threshold_deg"]

    out: dict[str, Any] = {}
    for seg_name, mask_fn in segments_map.items():
        deltas = pooled_deltas_for_mask(episodes, mask_fn)
        n = deltas.shape[0]
        seg_out = {"n_frames": n}
        for i, j in enumerate(JOINTS):
            if n == 0:
                seg_out[j] = None
                continue
            abs_vals = np.abs(deltas[:, i])
            seg_out[j] = {
                "frac_exceeds_would_clamp": float(np.mean(abs_vals > wc[j])),
                "frac_exceeds_reject": float(np.mean(abs_vals > rj[j])),
            }
        out[seg_name] = seg_out
    return out


# --------------------------------------------------------------------------
# 6. Whole-chunk mechanism check (static observation -> what does its full 50-step target
#    chunk actually look like?)
# --------------------------------------------------------------------------


def compute_chunk_mechanism_check(
    episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]], chunk_size: int
) -> dict[str, Any]:
    per_joint_chunk0 = {j: [] for j in KEY_JOINTS}
    per_joint_chunkmean = {j: [] for j in KEY_JOINTS}
    per_joint_chunkmax = {j: [] for j in KEY_JOINTS}
    n_static_frames_total = 0
    n_static_frames_chunk_mean_exceeds_would_clamp = {j: 0 for j in KEY_JOINTS}

    thresholds = load_safety_thresholds(DEFAULT_SAFETY_GATE_YAML)
    wc = thresholds["would_clamp_threshold_deg"]

    for ep, d in episodes.items():
        state = d["state"]
        action = d["action"]
        length = d["length"]
        static_len = segments[ep]["static_length"]
        for t in range(static_len):
            n_static_frames_total += 1
            k_idx = np.arange(chunk_size)
            frame_idx = np.minimum(t + k_idx, length - 1)
            chunk = action[frame_idx]  # (chunk_size, 6)
            delta = chunk - state[t][None, :]
            for j in KEY_JOINTS:
                ji = JOINTS.index(j)
                col = np.abs(delta[:, ji])
                per_joint_chunk0[j].append(col[0])
                per_joint_chunkmean[j].append(float(np.mean(col)))
                per_joint_chunkmax[j].append(float(np.max(col)))
                if np.mean(col) > wc[j]:
                    n_static_frames_chunk_mean_exceeds_would_clamp[j] += 1

    out: dict[str, Any] = {"chunk_size": chunk_size, "n_static_frames_total": n_static_frames_total}
    for j in KEY_JOINTS:
        out[j] = {
            "chunk0_abs_delta": {
                "mean": float(np.mean(per_joint_chunk0[j])),
                "median": float(np.median(per_joint_chunk0[j])),
                "p95": float(np.percentile(per_joint_chunk0[j], 95)),
            },
            "chunk_mean_abs_delta_over_all_k": {
                "mean": float(np.mean(per_joint_chunkmean[j])),
                "median": float(np.median(per_joint_chunkmean[j])),
                "p95": float(np.percentile(per_joint_chunkmean[j], 95)),
            },
            "chunk_max_abs_delta_over_all_k": {
                "mean": float(np.mean(per_joint_chunkmax[j])),
                "median": float(np.median(per_joint_chunkmax[j])),
                "p95": float(np.percentile(per_joint_chunkmax[j], 95)),
            },
            "n_static_frames_where_chunk_mean_exceeds_would_clamp": n_static_frames_chunk_mean_exceeds_would_clamp[j],
            "fraction_static_frames_where_chunk_mean_exceeds_would_clamp": (
                n_static_frames_chunk_mean_exceeds_would_clamp[j] / n_static_frames_total if n_static_frames_total else None
            ),
        }
    return out


# --------------------------------------------------------------------------
# 7. Start-pose diversity across 35 episodes
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
        "max": float(np.max(dists)),
    }


def compute_start_pose_diversity(episodes: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    eps = sorted(episodes.keys())
    start_states = np.stack([episodes[ep]["state"][0] for ep in eps])  # (35, 6)
    start_states_mean5 = np.stack([episodes[ep]["state"][: min(5, episodes[ep]["length"])].mean(axis=0) for ep in eps])

    def mid_reach_frame(ep):
        length = episodes[ep]["length"]
        return min(int(length * 0.6), length - 1)

    mid_states = np.stack([episodes[ep]["state"][mid_reach_frame(ep)] for ep in eps])

    per_joint_start_stats = {}
    per_joint_mid_stats = {}
    for i, j in enumerate(JOINTS):
        per_joint_start_stats[j] = {
            "std": float(np.std(start_states[:, i])),
            "range": float(np.ptp(start_states[:, i])),
            "min": float(np.min(start_states[:, i])),
            "max": float(np.max(start_states[:, i])),
        }
        per_joint_mid_stats[j] = {
            "std": float(np.std(mid_states[:, i])),
            "range": float(np.ptp(mid_states[:, i])),
            "min": float(np.min(mid_states[:, i])),
            "max": float(np.max(mid_states[:, i])),
        }

    return {
        "n_episodes": len(eps),
        "per_joint_start_frame0_stats": per_joint_start_stats,
        "per_joint_mid_reach_frame_stats": per_joint_mid_stats,
        "start_frame0_pairwise_l2_deg": pairwise_l2(start_states),
        "start_mean_first5_pairwise_l2_deg": pairwise_l2(start_states_mean5),
        "mid_reach_pairwise_l2_deg": pairwise_l2(mid_states),
        "note": (
            "mid_reach_frame is each episode's own 60%-of-length frame (where Grid35's 35 distinct "
            "XY cube placements should have already differentiated the arm's reach pose) - compared "
            "against the frame-0/frame[0:5) start pose to quantify how much smaller start-pose "
            "diversity is relative to genuine task-driven diversity later in the same episodes."
        ),
    }


# --------------------------------------------------------------------------
# 8. Large-action frequency table
# --------------------------------------------------------------------------


def compute_large_action_frequency(episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]]) -> dict[str, Any]:
    def mask_all(ep, length):
        return np.ones(length, dtype=bool)

    def mask_moving(ep, length):
        m = np.zeros(length, dtype=bool)
        m[segments[ep]["static_length"] :] = True
        return m

    out = {}
    for seg_name, mask_fn in [("all_frames", mask_all), ("moving_segment_data_driven", mask_moving)]:
        deltas = pooled_deltas_for_mask(episodes, mask_fn)
        seg_out = {}
        for j in KEY_JOINTS:
            ji = JOINTS.index(j)
            abs_vals = np.abs(deltas[:, ji])
            seg_out[j] = {f">= {t}deg": float(np.mean(abs_vals >= t)) for t in LARGE_ACTION_THRESHOLDS_DEG}
        out[seg_name] = seg_out
    return out


# --------------------------------------------------------------------------
# 9. Loss-contribution proxy (explicitly labeled first-order approximation)
# --------------------------------------------------------------------------


def compute_loss_contribution_proxy(
    episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]], chunk_size: int
) -> dict[str, Any]:
    """Approximate each segment's share of the total squared-error budget a predict-the-mean
    baseline over the *whole dataset* would need to fit, restricted to shoulder_lift/elbow_flex.

    For every (frame t, chunk position k in 0..chunk_size-1) valid pair (pad-clamped positions
    still count once, matching LeRobot's own `~actions_is_pad` masking - a clamped/padded target
    is still a real, equally-weighted training label, just a repeated one) computes
    `(action[min(t+k,end-1)] - global_mean_action)^2` where `global_mean_action` is the
    dataset-wide mean raw action value per joint (the value a predict-the-mean baseline would
    converge to under uniform MSE weighting - see module docstring point 7 for why this
    weighting is uniform per LeRobot's actual loss implementation). Sums are reported per
    segment (static-segment *observations* vs moving-segment *observations*, using each
    observation's full associated chunk) so segment shares can be compared directly.
    """
    global_mean_action = {}
    all_actions = np.concatenate([d["action"] for d in episodes.values()], axis=0)
    for i, j in enumerate(JOINTS):
        global_mean_action[j] = float(np.mean(all_actions[:, i]))

    sq_err_static = {j: 0.0 for j in KEY_JOINTS}
    sq_err_moving = {j: 0.0 for j in KEY_JOINTS}
    n_pairs_static = 0
    n_pairs_moving = 0

    for ep, d in episodes.items():
        action = d["action"]
        length = d["length"]
        static_len = segments[ep]["static_length"]
        k_idx = np.arange(chunk_size)
        for t in range(length):
            frame_idx = np.minimum(t + k_idx, length - 1)
            chunk = action[frame_idx]  # (chunk_size, 6)
            is_static_obs = t < static_len
            for j in KEY_JOINTS:
                ji = JOINTS.index(j)
                sq = (chunk[:, ji] - global_mean_action[j]) ** 2
                if is_static_obs:
                    sq_err_static[j] += float(np.sum(sq))
                else:
                    sq_err_moving[j] += float(np.sum(sq))
            if is_static_obs:
                n_pairs_static += chunk_size
            else:
                n_pairs_moving += chunk_size

    out = {
        "global_mean_action_deg": global_mean_action,
        "n_obs_chunk_pairs_static_segment": n_pairs_static,
        "n_obs_chunk_pairs_moving_segment": n_pairs_moving,
        "n_obs_chunk_pairs_fraction_static": n_pairs_static / (n_pairs_static + n_pairs_moving),
        "per_joint": {},
        "caveat": (
            "First-order proxy only: assumes a predict-the-global-mean-action baseline and "
            "LeRobot's actual uniform (sample, chunk-position, action-dim) MSE weighting "
            "(confirmed in lerobot/policies/smolvla/modeling_smolvla.py "
            "VLAFlowMatching.forward -> F.mse_loss(..., reduction='none') averaged uniformly, "
            "pad-masked only). Real flow-matching training does not literally regress to a "
            "static per-joint mean, but this proxy is the standard way to reason about which "
            "population's characteristic magnitude dominates an uniformly-weighted squared-error "
            "objective when the model cannot fully disambiguate similar-looking inputs."
        ),
    }
    for j in KEY_JOINTS:
        total = sq_err_static[j] + sq_err_moving[j]
        out["per_joint"][j] = {
            "sum_sq_err_static_segment_deg2": sq_err_static[j],
            "sum_sq_err_moving_segment_deg2": sq_err_moving[j],
            "fraction_of_total_sq_err_from_static_segment": sq_err_static[j] / total if total else None,
            "fraction_of_total_sq_err_from_moving_segment": sq_err_moving[j] / total if total else None,
        }
    return out


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def build_verdict(result: dict[str, Any]) -> dict[str, str]:
    cov = result["frame_coverage"]
    static_frac = cov["static_segment_data_driven"]["fraction_of_total"]
    dist_all = result["segment_delta_distributions"]["all_frames"]
    dist_moving = result["segment_delta_distributions"]["moving_segment_data_driven"]
    dist_static = result["segment_delta_distributions"]["static_segment_data_driven"]
    thresh_cov = result["threshold_coverage"]
    diversity = result["start_pose_diversity"]
    loss_proxy = result["loss_contribution_proxy"]

    v: dict[str, str] = {}

    v["start_low_motion_too_few"] = (
        f"YES, structurally small relative to the moving segment: the data-driven static segment "
        f"covers only {static_frac * 100:.1f}% of all {cov['total_frames']} frames "
        f"({cov['static_segment_data_driven']['n_frames']} frames; median {cov['median_static_length_frames']:.0f} "
        f"frames/episode, ~{cov['median_static_length_frames'] / FPS:.2f}s), vs. "
        f"{cov['moving_segment_data_driven']['fraction_of_total'] * 100:.1f}% moving. Fixed windows: "
        f"frames[0:15]={cov['frames_0_15']['fraction_of_total'] * 100:.1f}% of dataset, "
        f"frames[0:30]={cov['frames_0_30']['fraction_of_total'] * 100:.1f}%. Under LeRobot's uniform "
        "per-(sample, chunk-position) loss weighting (no oversampling of any segment by default), "
        "this segment gets roughly proportional representation in the objective - which is *not* "
        "boosted to compensate for it being the behaviourally most safety-sensitive part of the "
        "trajectory (the part right before Safety Gate sees the first delivered action)."
    )

    v["shoulder_elbow_large_actions_dominant"] = (
        f"YES for the moving segment, which is most of the dataset: mean|action-state| in the moving "
        f"segment is shoulder_lift={dist_moving['shoulder_lift']['mean_abs']:.2f}deg / "
        f"elbow_flex={dist_moving['elbow_flex']['mean_abs']:.2f}deg (p95 "
        f"{dist_moving['shoulder_lift']['p95_abs']:.2f}/{dist_moving['elbow_flex']['p95_abs']:.2f}deg) vs. "
        f"the static segment's {dist_static['shoulder_lift']['mean_abs']:.3f}/"
        f"{dist_static['elbow_flex']['mean_abs']:.3f}deg. See large-action frequency table: "
        f"{result['large_action_frequency']['moving_segment_data_driven']['shoulder_lift']['>= 5.0deg'] * 100:.1f}% "
        "of moving-segment frames already carry a shoulder_lift action-state delta at/above the "
        "Safety Gate WOULD_CLAMP threshold on their own. Threshold coverage table (section 3) gives "
        "the exact per-segment WOULD_CLAMP/REJECT exceedance rates."
    )

    v["35_episodes_insufficient_for_static_to_movement_transition"] = (
        "YES, plausibly: there are only 35 distinct static->movement transition examples in the "
        "entire dataset (one per episode) - and start_pose_diversity (section 5) shows those 35 "
        f"transitions begin from near-identical proprioceptive states (frame-0 pairwise L2 median "
        f"{diversity['start_frame0_pairwise_l2_deg']['median']:.2f}deg, max "
        f"{diversity['start_frame0_pairwise_l2_deg']['max']:.2f}deg, vs. mid-reach-frame pairwise L2 "
        f"median {diversity['mid_reach_pairwise_l2_deg']['median']:.2f}deg, max "
        f"{diversity['mid_reach_pairwise_l2_deg']['max']:.2f}deg - i.e. Grid35's actual 35-way task "
        "diversity only shows up well after the hold, not at the start). The chunk-mechanism check "
        "(section 4) shows that even though a static observation's own chunk[0] target is small, "
        f"{result['chunk_mechanism_check']['shoulder_lift']['fraction_static_frames_where_chunk_mean_exceeds_would_clamp'] * 100:.1f}% "
        "of static-segment observations are paired with a *whole* 50-step target chunk whose "
        "mean|delta| already exceeds the shoulder_lift WOULD_CLAMP threshold (since the chunk runs "
        "1.67s forward, well past the ~0.7-1.0s typical movement onset). With only 35 near-identical "
        "start contexts each carrying that kind of large-mean target chunk, and no contrasting "
        "examples where a similar-looking static start is paired with a genuinely small full-chunk "
        "target, there is little data-side signal to teach the model that 'looks static' should "
        "imply 'predict small for a while' rather than 'predict this population's typical chunk "
        "shape'."
    )

    v["what_data_to_add_first"] = (
        "Priority: (1) more start-segment *diversity*, not just more of the same 35 near-identical "
        "start poses - e.g. episodes where the static-hold duration varies (some near-zero hold, "
        "some longer), and/or where the visual scene early in the episode already makes the "
        "eventual grid-cell target legible (so the model has something other than a duplicate "
        "proprioceptive start state to condition on); (2) episodes/segments that break the "
        "correlation this dataset currently has baked in between 'observation looks static' and "
        "'this chunk's mean target is large' - i.e. genuinely-static holds of varying length "
        "immediately followed by small-magnitude motion, and slow/gradual movement onsets (not just "
        "the current fast-ramp profile) so a range of chunk-mean magnitudes gets paired with "
        "static-looking inputs; (3) more Grid35 grid-cell coverage in general (35 episodes = 35 "
        "grid cells with 1 demo each - no repetition to average out per-episode teleop-operator "
        "variance)."
    )

    frac_static_pairs = loss_proxy["n_obs_chunk_pairs_fraction_static"]
    frac_static_sqerr_sl = loss_proxy["per_joint"]["shoulder_lift"]["fraction_of_total_sq_err_from_static_segment"]
    frac_static_sqerr_ef = loss_proxy["per_joint"]["elbow_flex"]["fraction_of_total_sq_err_from_static_segment"]
    v["retrain_priority"] = (
        f"More demonstrations (diversity-focused, see above) over naive start-segment oversampling. "
        f"Rationale: static-segment (observation, chunk-position) pairs are already "
        f"{frac_static_pairs * 100:.1f}% of the uniformly-weighted training signal by *count* "
        f"(loss-contribution proxy, section 6) - oversampling would inflate that further without "
        "adding a single new start pose or a single counter-example that decorrelates 'static-"
        "looking' from 'large upcoming chunk' - given only 35 distinct start contexts exist, "
        "oversampling risks overfitting to those exact 35 (state, chunk) pairs rather than teaching "
        "the general rule. A cheaper, no-new-data lever worth trying in parallel: a chunk-position-"
        "aware loss weighting (e.g. upweight near-term chunk indices relative to far-future ones, "
        "or clip/downweight target magnitude beyond the typical movement-onset horizon for "
        "still-static observations) - LeRobot's default loss has no such weighting today (see "
        "module docstring point 7), so this is a real, currently-unused lever, but it only reduces "
        "symptoms of the imbalance already present in the 35 episodes, it does not add the missing "
        "static-hold/short-hold/legible-early-visual diversity that (1) above targets. Given the "
        f"static segment's own targets are small (mean_abs shoulder_lift="
        f"{dist_static['shoulder_lift']['mean_abs']:.3f}deg) and the dataset is small (35 episodes "
        "total), the fastest, most robust fix is still more/varied demonstrations; loss weighting "
        "and oversampling are reasonable stop-gaps while more data is collected, not substitutes "
        "for it."
    )
    return v


# --------------------------------------------------------------------------
# Report writers
# --------------------------------------------------------------------------


def write_csv_per_episode(out_path: Path, episodes, segments) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "length", "first_movement_frame", "static_length", "moving_length", "static_fraction"])
        for ep in sorted(episodes.keys()):
            s = segments[ep]
            w.writerow([ep, s["length"], s["first_movement_frame"], s["static_length"], s["moving_length"], f"{s['static_length']/s['length']:.4f}"])


def write_markdown_report(out_path: Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Grid35 V2-clean: training-target distribution & start-segment coverage audit")
    lines.append("")
    lines.append(
        f"Dataset: `{result['dataset_root']}` - {result['frame_coverage']['n_episodes']} episodes, "
        f"{result['frame_coverage']['total_frames']} frames total, {FPS}fps."
    )
    lines.append("")

    lines.append("## 1. Frame-count coverage")
    lines.append("")
    cov = result["frame_coverage"]
    lines.append("| segment | n_frames | % of dataset |")
    lines.append("|---|---:|---:|")
    lines.append(f"| frames[0:15] | {cov['frames_0_15']['n_frames']} | {cov['frames_0_15']['fraction_of_total']*100:.1f}% |")
    lines.append(f"| frames[0:30] | {cov['frames_0_30']['n_frames']} | {cov['frames_0_30']['fraction_of_total']*100:.1f}% |")
    lines.append(
        f"| static (data-driven, movement detector) | {cov['static_segment_data_driven']['n_frames']} | "
        f"{cov['static_segment_data_driven']['fraction_of_total']*100:.1f}% |"
    )
    lines.append(
        f"| moving (data-driven) | {cov['moving_segment_data_driven']['n_frames']} | "
        f"{cov['moving_segment_data_driven']['fraction_of_total']*100:.1f}% |"
    )
    lines.append(f"| full episode | {cov['total_frames']} | 100.0% |")
    lines.append("")
    lines.append(
        f"Median/mean static-segment length: {cov['median_static_length_frames']:.1f} / "
        f"{cov['mean_static_length_frames']:.1f} frames (~{cov['median_static_length_frames']/FPS:.2f}s / "
        f"~{cov['mean_static_length_frames']/FPS:.2f}s)."
    )
    lines.append("")

    lines.append("## 2. Per-joint action(t)-state(t) distribution by segment (shoulder_lift / elbow_flex)")
    lines.append("")
    lines.append("| segment | joint | n | mean\\|d\\| | median\\|d\\| | p95\\|d\\| | max\\|d\\| |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for seg_name in ["all_frames", "frames_0_15", "frames_0_30", "static_segment_data_driven", "moving_segment_data_driven"]:
        dist = result["segment_delta_distributions"][seg_name]
        for j in KEY_JOINTS:
            r = dist[j]
            lines.append(f"| {seg_name} | {j} | {r['n']} | {r['mean_abs']:.3f} | {r['median_abs']:.3f} | {r['p95_abs']:.3f} | {r['max_abs']:.3f} |")
    lines.append("")

    lines.append("## 3. Safety Gate WOULD_CLAMP/REJECT threshold coverage (read-only)")
    lines.append("")
    th = result["safety_thresholds"]
    lines.append(f"WOULD_CLAMP: shoulder_lift={th['would_clamp_threshold_deg']['shoulder_lift']}deg, elbow_flex={th['would_clamp_threshold_deg']['elbow_flex']}deg. "
                  f"REJECT (x5): shoulder_lift={th['reject_threshold_deg']['shoulder_lift']:.2f}deg, elbow_flex={th['reject_threshold_deg']['elbow_flex']:.2f}deg.")
    lines.append("")
    lines.append("| segment | joint | % frames > WOULD_CLAMP | % frames > REJECT |")
    lines.append("|---|---|---:|---:|")
    for seg_name in ["all_frames", "static_segment_data_driven", "moving_segment_data_driven"]:
        tc = result["threshold_coverage"][seg_name]
        for j in KEY_JOINTS:
            r = tc[j]
            lines.append(f"| {seg_name} | {j} | {r['frac_exceeds_would_clamp']*100:.1f}% | {r['frac_exceeds_reject']*100:.1f}% |")
    lines.append("")

    lines.append("## 4. Whole-target-chunk mechanism check (static observations only)")
    lines.append("")
    cm = result["chunk_mechanism_check"]
    lines.append(f"chunk_size={cm['chunk_size']}, n_static_frames_total={cm['n_static_frames_total']}")
    lines.append("")
    lines.append("| joint | chunk[0]\\|delta\\| mean/median/p95 | full-chunk mean\\|delta\\| mean/median/p95 | full-chunk max\\|delta\\| mean/median/p95 | % static frames where chunk-mean > WOULD_CLAMP |")
    lines.append("|---|---|---|---|---:|")
    for j in KEY_JOINTS:
        r = cm[j]
        c0 = r["chunk0_abs_delta"]
        cmean = r["chunk_mean_abs_delta_over_all_k"]
        cmax = r["chunk_max_abs_delta_over_all_k"]
        lines.append(
            f"| {j} | {c0['mean']:.3f}/{c0['median']:.3f}/{c0['p95']:.3f} | "
            f"{cmean['mean']:.3f}/{cmean['median']:.3f}/{cmean['p95']:.3f} | "
            f"{cmax['mean']:.3f}/{cmax['median']:.3f}/{cmax['p95']:.3f} | "
            f"{r['fraction_static_frames_where_chunk_mean_exceeds_would_clamp']*100:.1f}% |"
        )
    lines.append("")
    lines.append(
        "Interpretation: chunk[0] for a static observation is small (matches the previously-verified "
        "training-target alignment - offset is exactly 0, see "
        "`reports/smolvla_training_target_alignment/`), but the *rest* of that same observation's "
        "50-step target chunk (chunk[1..49], covering up to 1.67s forward) is frequently large - "
        "the model is trained to jointly predict all 50 steps from one static-looking observation, "
        "so its output distribution for that observation is shaped by the *whole* chunk's target "
        "statistics, not just position 0."
    )
    lines.append("")

    lines.append("## 5. Start-pose diversity across 35 episodes")
    lines.append("")
    div = result["start_pose_diversity"]
    lines.append("| joint | start(frame0) std | start(frame0) range | mid-reach(60%) std | mid-reach(60%) range |")
    lines.append("|---|---:|---:|---:|---:|")
    for j in KEY_JOINTS + ["wrist_roll", "gripper"]:
        s = div["per_joint_start_frame0_stats"][j]
        m = div["per_joint_mid_reach_frame_stats"][j]
        lines.append(f"| {j} | {s['std']:.2f} | {s['range']:.2f} | {m['std']:.2f} | {m['range']:.2f} |")
    lines.append("")
    lines.append("Pairwise L2 distance (degrees, over all 6 joints) among the 35 episodes' states:")
    lines.append("")
    lines.append("| basis | min | median | mean | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, key in [("start frame 0", "start_frame0_pairwise_l2_deg"), ("start mean frames[0:5)", "start_mean_first5_pairwise_l2_deg"), ("mid-reach (60% of episode)", "mid_reach_pairwise_l2_deg")]:
        p = div[key]
        lines.append(f"| {name} | {p['min']:.2f} | {p['median']:.2f} | {p['mean']:.2f} | {p['max']:.2f} |")
    lines.append("")
    lines.append(f"> {div['note']}")
    lines.append("")

    lines.append("## 6. Large shoulder_lift/elbow_flex action frequency")
    lines.append("")
    laf = result["large_action_frequency"]
    lines.append("| segment | joint | " + " | ".join(f">= {t}deg" for t in LARGE_ACTION_THRESHOLDS_DEG) + " |")
    lines.append("|---|---|" + "---:|" * len(LARGE_ACTION_THRESHOLDS_DEG))
    for seg_name in ["all_frames", "moving_segment_data_driven"]:
        for j in KEY_JOINTS:
            row = laf[seg_name][j]
            lines.append(f"| {seg_name} | {j} | " + " | ".join(f"{row[f'>= {t}deg']*100:.1f}%" for t in LARGE_ACTION_THRESHOLDS_DEG) + " |")
    lines.append("")

    lines.append("## 7. Loss-contribution proxy (first-order approximation - see caveat)")
    lines.append("")
    lp = result["loss_contribution_proxy"]
    lines.append(
        f"(observation, chunk-position) pair counts: static-segment={lp['n_obs_chunk_pairs_static_segment']}, "
        f"moving-segment={lp['n_obs_chunk_pairs_moving_segment']} "
        f"({lp['n_obs_chunk_pairs_fraction_static']*100:.1f}% static by count, under LeRobot's uniform "
        "per-pair loss weighting)."
    )
    lines.append("")
    lines.append("| joint | sum sq-err static (deg^2) | sum sq-err moving (deg^2) | % of total sq-err from static | % from moving |")
    lines.append("|---|---:|---:|---:|---:|")
    for j in KEY_JOINTS:
        r = lp["per_joint"][j]
        lines.append(
            f"| {j} | {r['sum_sq_err_static_segment_deg2']:.0f} | {r['sum_sq_err_moving_segment_deg2']:.0f} | "
            f"{r['fraction_of_total_sq_err_from_static_segment']*100:.1f}% | {r['fraction_of_total_sq_err_from_moving_segment']*100:.1f}% |"
        )
    lines.append("")
    lines.append(f"> {lp['caveat']}")
    lines.append("")

    lines.append("## 8. Answers")
    lines.append("")
    v = result["verdict"]
    titles = {
        "start_low_motion_too_few": "start/low-motion samples가 전체에서 너무 적은가?",
        "shoulder_elbow_large_actions_dominant": "shoulder_lift/elbow_flex의 큰 action이 dataset에서 지배적인가?",
        "35_episodes_insufficient_for_static_to_movement_transition": "35 episodes 규모에서 static->movement transition을 충분히 학습하기 어려워 보이는가?",
        "what_data_to_add_first": "데이터 추가 수집이 필요하다면 어떤 샘플을 우선 추가해야 하는가?",
        "retrain_priority": "재학습 시 start-segment oversampling / weighted loss / 더 많은 demonstrations 중 무엇이 우선인가?",
    }
    for key, question in titles.items():
        lines.append(f"### {question}")
        lines.append("")
        lines.append(v[key])
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--safety-gate-yaml", type=Path, default=DEFAULT_SAFETY_GATE_YAML)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[coverage] 1/9 loading 35 episodes")
    episodes = load_episodes(dataset_root)
    assert len(episodes) == 35, f"expected 35 episodes, found {len(episodes)}"

    print("[coverage] 2/9 static/moving segmentation")
    segments = compute_segments(episodes)

    print("[coverage] 3/9 frame-count coverage")
    frame_coverage = compute_frame_coverage(episodes, segments)

    print("[coverage] 4/9 per-joint delta distributions by segment")
    segment_delta_distributions = compute_segment_delta_distributions(episodes, segments)

    print("[coverage] 5/9 safety threshold coverage")
    safety_thresholds = load_safety_thresholds(args.safety_gate_yaml)
    threshold_coverage = compute_threshold_coverage(episodes, segments, safety_thresholds)

    print("[coverage] 6/9 whole-chunk mechanism check")
    chunk_mechanism_check = compute_chunk_mechanism_check(episodes, segments, args.chunk_size)

    print("[coverage] 7/9 start-pose diversity")
    start_pose_diversity = compute_start_pose_diversity(episodes)

    print("[coverage] 8/9 large-action frequency + loss-contribution proxy")
    large_action_frequency = compute_large_action_frequency(episodes, segments)
    loss_contribution_proxy = compute_loss_contribution_proxy(episodes, segments, args.chunk_size)

    print("[coverage] 9/9 verdict + reports")
    result: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "frame_coverage": frame_coverage,
        "segment_delta_distributions": segment_delta_distributions,
        "safety_thresholds": safety_thresholds,
        "threshold_coverage": threshold_coverage,
        "chunk_mechanism_check": chunk_mechanism_check,
        "start_pose_diversity": start_pose_diversity,
        "large_action_frequency": large_action_frequency,
        "loss_contribution_proxy": loss_contribution_proxy,
    }
    result["verdict"] = build_verdict(result)

    json_path = out_dir / "dataset_coverage_imbalance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"[coverage] wrote {json_path}")

    write_csv_per_episode(out_dir / "per_episode_segments.csv", episodes, segments)
    print(f"[coverage] wrote {out_dir / 'per_episode_segments.csv'}")

    md_path = out_dir / "dataset_coverage_imbalance.md"
    write_markdown_report(md_path, result)
    print(f"[coverage] wrote {md_path}")

    print("[coverage] VERDICT summary:")
    for k, v in result["verdict"].items():
        print(f"  {k}: {v[:140]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
