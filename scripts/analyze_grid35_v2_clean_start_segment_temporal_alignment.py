#!/usr/bin/env python3
"""Grid35 V2-clean: episode start-segment (0-0.5s) temporal-alignment audit.

Background
----------
The frozen SmolVLA checkpoint (``outputs/grid35_v2/smolvla_grid35_v2_clean_fresh``)
delivers an oversized *actual Shadow* first action for ``shoulder_lift`` and
``elbow_flex`` at nearly every checkpoint/scene tested so far (T01-T10, see
``reports/grid35_v2_T01_T10_actual_seed_sweep_summary/``: mean first-action
delta ~+5.7deg shoulder_lift / ~-6.5deg elbow_flex, vs. a WOULD_CLAMP
threshold of 5.16/5.73deg -> 50%/72% clamp rate). This script asks whether
that bias is already latent in the **training dataset's own start-of-episode
labels** (a temporal-alignment / action-state mislabeling problem, hypothesis
C) or whether the ground-truth start action is in fact small and the policy
alone is responsible (hypothesis B / D).

What this script does (all read-only, no writes to data/config/robot):

 1. Loads all 35 training episodes of ``data/so101_cube_xy_grid35_v2_clean``
    in full (observation.state, action; 6 joints; degrees).
 2. For frames 0-14 (0-0.5s @ 30fps) of every episode, computes
    ``action[t] - state[t]`` per joint and aggregates mean/median/p95/max|.|
    across the 35 episodes, per frame index, focused on shoulder_lift and
    elbow_flex (full 6-joint table also written).
 3. Estimates, per episode, the first "non-trivial movement" frame -
    overall, and separately for shoulder_lift / elbow_flex - from both the
    recorded ``observation.state`` trajectory and the ``action`` trajectory,
    using an explicit, data-justified noise-vs-movement threshold (see
    ``MOVEMENT_THRESHOLD_DEG`` / ``PERSISTENCE_FRAMES`` below - the exact
    values used are also written into the JSON/MD report).
 4. Computes the dataset-wide (all 35 episodes) mean/median displacement of
    ``state[t]-state[0]`` and ``action[t]-state[0]`` for horizons t=1..30,
    and finds which horizon's mean displacement best matches the *actual*
    Shadow first-action delta reported in
    ``reports/grid35_v2_T01_T10_actual_seed_sweep_summary/t01_t10_summary.json``
    (T01-T10 average ``per_joint_mean_delta``) - i.e. "how many frames into
    a demonstration does the policy's first action actually look like".
 5. Reads (never modifies) ``configs/safety_gate.yaml`` and reports the
    WOULD_CLAMP / REJECT thresholds side by side with the dataset's own
    start-segment delta and the actual-Shadow policy delta.
 6. Emits a final classification across hypotheses A-E (see module-level
    docstring section "Verdict" and the generated Markdown report).

No retraining, no Safety Gate threshold edits, no robot writes, no LeRobot
source modification, no git operations.
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
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "grid35_v2_clean_start_segment_temporal_alignment"
DEFAULT_SAFETY_GATE_YAML = PROJECT_ROOT / "configs" / "safety_gate.yaml"
DEFAULT_ACTUAL_SHADOW_SUMMARY = (
    PROJECT_ROOT / "reports" / "grid35_v2_T01_T10_actual_seed_sweep_summary" / "t01_t10_summary.json"
)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
KEY_JOINTS = ["shoulder_lift", "elbow_flex"]

FPS = 30
START_SEGMENT_FRAMES = 15  # frames 0..14 inclusive == 0.0s..~0.467s @ 30fps (task asks for "~0-0.5s")
HORIZON_MAX_FRAME = 30  # frames 1..30 for the trajectory/horizon-match analysis (task item 5)

# --- Movement-vs-noise threshold (data-justified; NOT tuned to produce a desired answer) -------
# The dataset's own state stream is heavily quantized (many exact-repeat float values during
# holds), consistent with an absolute encoder of resolution ~360/4096 = 0.088deg/count. A single
# encoder step, or even a couple of counts of read jitter, should not be called "movement". We set
# the cumulative-displacement-from-frame-0 threshold an order of magnitude above that floor, and
# additionally require it to persist for PERSISTENCE_FRAMES consecutive frames (so a single noisy
# sample can't flip the verdict), while staying far below the ~5-6deg magnitude of the
# actual-Shadow policy bias under investigation (i.e. good separation margin between "noise",
# "the movement we're trying to detect", and "the anomaly we're comparing against").
MOVEMENT_THRESHOLD_DEG = 1.0
PERSISTENCE_FRAMES = 3
GROSS_STEP_MULTIPLIER = 5.0  # matches scripts/diagnose_grid35_first_action_pipeline.py convention


# --------------------------------------------------------------------------
# 1. Load dataset
# --------------------------------------------------------------------------


def load_episodes(dataset_root: Path) -> dict[int, dict[str, np.ndarray]]:
    """Load full observation.state/action arrays for every episode, keyed by episode_index."""
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
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        action = np.stack(df["action"].to_numpy()).astype(np.float64)
        episodes[ep] = {"state": state, "action": action, "length": len(df)}
    return episodes


# --------------------------------------------------------------------------
# 2. Frame 0-14 delta stats
# --------------------------------------------------------------------------


def compute_start_segment_deltas(episodes: dict[int, dict[str, np.ndarray]], n_frames: int) -> np.ndarray:
    """Return array (n_episodes, n_frames, 6) of action[t]-state[t] for frames 0..n_frames-1."""
    eps = sorted(episodes.keys())
    out = np.full((len(eps), n_frames, 6), np.nan)
    for i, ep in enumerate(eps):
        state = episodes[ep]["state"]
        action = episodes[ep]["action"]
        n = min(n_frames, len(state))
        out[i, :n, :] = action[:n] - state[:n]
    return out


def aggregate_per_frame(deltas: np.ndarray) -> dict[str, Any]:
    """deltas: (n_episodes, n_frames, 6) -> per-frame per-joint mean/median/p95/max|delta|."""
    n_frames = deltas.shape[1]
    result: dict[str, Any] = {}
    for t in range(n_frames):
        frame_slice = deltas[:, t, :]  # (n_episodes, 6)
        abs_slice = np.abs(frame_slice)
        row: dict[str, Any] = {}
        for j_idx, joint in enumerate(JOINTS):
            vals = frame_slice[:, j_idx]
            abs_vals = abs_slice[:, j_idx]
            row[joint] = {
                "mean_delta": float(np.nanmean(vals)),
                "median_delta": float(np.nanmedian(vals)),
                "mean_abs_delta": float(np.nanmean(abs_vals)),
                "median_abs_delta": float(np.nanmedian(abs_vals)),
                "p95_abs_delta": float(np.nanpercentile(abs_vals, 95)),
                "max_abs_delta": float(np.nanmax(abs_vals)),
            }
        result[t] = row
    return result


# --------------------------------------------------------------------------
# 3. First non-trivial movement frame estimation
# --------------------------------------------------------------------------


def first_movement_frame(
    traj: np.ndarray, ref: np.ndarray, threshold: float, persistence: int, joint_idx: int | None = None
) -> int | None:
    """First frame index t (>=1) s.t. |traj[t]-ref| exceeds threshold and stays exceeded for
    `persistence` consecutive frames (t..t+persistence-1, or through episode end if shorter).
    If joint_idx is None, uses max-abs over all 6 joints; else that single joint.
    Returns None if no such frame exists within the trajectory.
    """
    disp = traj - ref[None, :]
    if joint_idx is None:
        mag = np.max(np.abs(disp), axis=1)
    else:
        mag = np.abs(disp[:, joint_idx])
    n = len(mag)
    for t in range(1, n):
        end = min(t + persistence, n)
        if end - t < persistence and t + persistence <= n:
            continue
        window = mag[t:end]
        if len(window) > 0 and np.all(window >= threshold):
            return t
    return None


def compute_first_movement_frames(
    episodes: dict[int, dict[str, np.ndarray]], threshold: float, persistence: int
) -> dict[int, dict[str, Any]]:
    """Per-episode first-movement frame, for both `observation.state` and `action`, each measured
    against its OWN frame-0 value (state vs state[0], action vs action[0]) - not against each
    other's baseline. This matters: action[0]-state[0] is frequently a small but nonzero *constant*
    leader/follower offset (see per_frame_aggregate, e.g. elbow_flex ~1.8-2.3deg, essentially flat
    across frames 0-14) that must NOT itself be counted as "action commanding movement" - only a
    departure from action's *own* start value should. Using state[0] as the action baseline would
    make that constant offset trip the threshold at frame 1 in almost every episode, which is a
    measurement artifact, not evidence of an early command to move.
    """
    out: dict[int, dict[str, Any]] = {}
    sl_idx = JOINTS.index("shoulder_lift")
    ef_idx = JOINTS.index("elbow_flex")
    for ep, d in episodes.items():
        state = d["state"]
        action = d["action"]
        state0 = state[0]
        action0 = action[0]
        out[ep] = {
            "state_first_movement_frame_any_joint": first_movement_frame(state, state0, threshold, persistence),
            "state_first_movement_frame_shoulder_lift": first_movement_frame(
                state, state0, threshold, persistence, sl_idx
            ),
            "state_first_movement_frame_elbow_flex": first_movement_frame(
                state, state0, threshold, persistence, ef_idx
            ),
            "action_first_movement_frame_any_joint": first_movement_frame(action, action0, threshold, persistence),
            "action_first_movement_frame_shoulder_lift": first_movement_frame(
                action, action0, threshold, persistence, sl_idx
            ),
            "action_first_movement_frame_elbow_flex": first_movement_frame(
                action, action0, threshold, persistence, ef_idx
            ),
        }
    return out


def compute_noise_floor_evidence(episodes: dict[int, dict[str, np.ndarray]], up_to_frame: int) -> dict[str, Any]:
    """Frame-to-frame |state[t]-state[t-1]| for t=1..up_to_frame, all episodes, per key joint -
    descriptive stats used to justify MOVEMENT_THRESHOLD_DEG."""
    out: dict[str, Any] = {}
    for joint in JOINTS:
        j_idx = JOINTS.index(joint)
        diffs = []
        for d in episodes.values():
            state = d["state"]
            n = min(up_to_frame + 1, len(state))
            if n < 2:
                continue
            diffs.append(np.abs(np.diff(state[:n, j_idx])))
        all_diffs = np.concatenate(diffs) if diffs else np.array([])
        out[joint] = {
            "n_samples": int(all_diffs.size),
            "frac_exactly_zero": float(np.mean(all_diffs == 0.0)) if all_diffs.size else None,
            "p50": float(np.percentile(all_diffs, 50)) if all_diffs.size else None,
            "p90": float(np.percentile(all_diffs, 90)) if all_diffs.size else None,
            "p99": float(np.percentile(all_diffs, 99)) if all_diffs.size else None,
            "max": float(np.max(all_diffs)) if all_diffs.size else None,
        }
    return out


# --------------------------------------------------------------------------
# 4. Horizon / trajectory-match analysis (task item 5)
# --------------------------------------------------------------------------


def compute_horizon_trajectories(
    episodes: dict[int, dict[str, np.ndarray]], max_horizon: int
) -> dict[str, dict[int, dict[str, float]]]:
    """Dataset-wide mean/median of state[t]-state[0] and action[t]-state[0] for t=1..max_horizon."""
    state_disp: dict[int, list[np.ndarray]] = {t: [] for t in range(1, max_horizon + 1)}
    action_disp: dict[int, list[np.ndarray]] = {t: [] for t in range(1, max_horizon + 1)}
    for d in episodes.values():
        state = d["state"]
        action = d["action"]
        state0 = state[0]
        n = len(state)
        for t in range(1, max_horizon + 1):
            if t < n:
                state_disp[t].append(state[t] - state0)
                action_disp[t].append(action[t] - state0)

    def summarize(disp_dict: dict[int, list[np.ndarray]]) -> dict[int, dict[str, Any]]:
        out = {}
        for t, arrs in disp_dict.items():
            if not arrs:
                continue
            stacked = np.stack(arrs)  # (n_episodes, 6)
            out[t] = {
                "mean": {j: float(np.mean(stacked[:, i])) for i, j in enumerate(JOINTS)},
                "median": {j: float(np.median(stacked[:, i])) for i, j in enumerate(JOINTS)},
                "n_episodes": int(stacked.shape[0]),
            }
        return out

    return {"state_minus_state0": summarize(state_disp), "action_minus_state0": summarize(action_disp)}


def estimate_matching_horizon(
    horizon_traj: dict[int, dict[str, Any]], target_delta: dict[str, float], joints: list[str]
) -> dict[str, Any]:
    """For each joint, find the horizon k whose mean displacement best matches target_delta[joint]
    (linear interpolation between bracketing horizons; falls back to nearest single horizon)."""
    ks = sorted(horizon_traj.keys())
    result: dict[str, Any] = {}
    for j in joints:
        vals = [horizon_traj[k]["mean"][j] for k in ks]
        target = target_delta[j]
        best_k = None
        for i in range(len(ks) - 1):
            v0, v1 = vals[i], vals[i + 1]
            if (v0 - target) * (v1 - target) <= 0 and v0 != v1:
                frac = (target - v0) / (v1 - v0)
                best_k = ks[i] + frac * (ks[i + 1] - ks[i])
                break
        if best_k is None:
            closest_idx = int(np.argmin([abs(v - target) for v in vals]))
            best_k = ks[closest_idx]
        result[j] = {
            "target_delta_deg": target,
            "estimated_matching_horizon_frames": best_k,
            "estimated_matching_horizon_seconds": best_k / FPS,
        }
    return result


# --------------------------------------------------------------------------
# 5. Safety threshold (read-only)
# --------------------------------------------------------------------------


def load_safety_thresholds(safety_gate_yaml: Path) -> dict[str, Any]:
    raw = yaml.safe_load(safety_gate_yaml.read_text(encoding="utf-8"))
    excessive = raw["excessive_step_deg"]
    return {
        "would_clamp_threshold_deg": excessive,
        "reject_threshold_deg": {j: v * GROSS_STEP_MULTIPLIER for j, v in excessive.items()},
        "source_file": str(safety_gate_yaml),
        "note": "Read-only. Not modified by this script.",
    }


# --------------------------------------------------------------------------
# 6. Actual-Shadow policy first-action stats (T01-T10, for cross-reference)
# --------------------------------------------------------------------------


def load_actual_shadow_policy_stats(summary_json: Path) -> dict[str, Any]:
    with open(summary_json, encoding="utf-8") as f:
        d = json.load(f)
    scenes = d["scenes"]
    per_joint_means: dict[str, list[float]] = {j: [] for j in JOINTS}
    per_scene_rows = []
    for s in scenes:
        pj_mean = s["per_joint_mean_delta"]
        pj_std = s["per_joint_std_delta"]
        for j in JOINTS:
            per_joint_means[j].append(pj_mean[j])
        per_scene_rows.append(
            {
                "scene": s["scene"],
                "per_joint_mean_delta": pj_mean,
                "per_joint_std_delta": pj_std,
                "per_joint_clamp_rate": s.get("per_joint_clamp_rate"),
                "nearest_demo_match": s.get("nearest_demo_match"),
            }
        )
    avg_across_scenes = {j: float(np.mean(vals)) for j, vals in per_joint_means.items()}
    return {
        "source_file": str(summary_json),
        "per_scene": per_scene_rows,
        "t01_t10_average_per_joint_mean_delta_deg": avg_across_scenes,
    }


# --------------------------------------------------------------------------
# 7. Verdict (A-E)
# --------------------------------------------------------------------------


def classify_verdict(
    per_frame_agg: dict[int, Any],
    first_movement: dict[int, dict[str, Any]],
    actual_shadow: dict[str, Any],
    safety: dict[str, Any],
    matching_horizon_state: dict[str, Any],
    matching_horizon_action: dict[str, Any],
) -> dict[str, Any]:
    # A: is the GT start action itself large?
    frame0 = per_frame_agg[0]
    gt_frame0_sl = frame0["shoulder_lift"]["mean_abs_delta"]
    gt_frame0_ef = frame0["elbow_flex"]["mean_abs_delta"]
    frame14 = per_frame_agg[START_SEGMENT_FRAMES - 1]
    gt_frame14_sl = frame14["shoulder_lift"]["mean_abs_delta"]
    gt_frame14_ef = frame14["elbow_flex"]["mean_abs_delta"]
    wc_sl = safety["would_clamp_threshold_deg"]["shoulder_lift"]
    wc_ef = safety["would_clamp_threshold_deg"]["elbow_flex"]

    a_gt_start_large = (gt_frame0_sl > wc_sl) or (gt_frame0_ef > wc_ef)

    # B: policy delta vs GT delta magnitude ratio
    policy_sl = actual_shadow["t01_t10_average_per_joint_mean_delta_deg"]["shoulder_lift"]
    policy_ef = actual_shadow["t01_t10_average_per_joint_mean_delta_deg"]["elbow_flex"]
    ratio_sl = abs(policy_sl) / max(gt_frame0_sl, 1e-6)
    ratio_ef = abs(policy_ef) / max(gt_frame0_ef, 1e-6)
    b_policy_much_larger_than_gt = ratio_sl > 3.0 or ratio_ef > 3.0

    # C: static-hold action-label misalignment - does action within the frame 0-14 hold segment
    # already show a nontrivial, monotonically growing offset from state that anticipates the
    # eventual movement (vs. just a small constant leader/follower bias)?
    n_eps = 35
    early_lead_frames = []
    for ep, mv in first_movement.items():
        a_f = mv["action_first_movement_frame_any_joint"]
        s_f = mv["state_first_movement_frame_any_joint"]
        if a_f is not None and s_f is not None:
            early_lead_frames.append(s_f - a_f)
    mean_lead = float(np.mean(early_lead_frames)) if early_lead_frames else None
    c_misalignment_material = (
        mean_lead is not None and mean_lead >= 5 and (gt_frame14_sl > wc_sl or gt_frame14_ef > wc_ef)
    )

    # D: does the policy first-action resemble a future-GT-frame delta well within the 0-30 frame
    # window used at training/inference time (chunk horizon)?
    d_frames = [
        matching_horizon_state.get("shoulder_lift", {}).get("estimated_matching_horizon_frames"),
        matching_horizon_state.get("elbow_flex", {}).get("estimated_matching_horizon_frames"),
    ]
    d_frames = [f for f in d_frames if f is not None]
    d_future_frame_estimate = float(np.mean(d_frames)) if d_frames else None

    # E: root cause lean
    if a_gt_start_large:
        lean = "A: GT start action itself already exceeds WOULD_CLAMP threshold - dataset labeling issue, not purely a policy issue."
    elif c_misalignment_material:
        lean = (
            "C leaning: action consistently anticipates state movement by "
            f"~{mean_lead:.1f} frames on average even within the 0-0.5s hold window, AND the "
            "start-segment GT delta itself grows close to/over threshold by frame 14 - temporal "
            "alignment is a plausible material contributor."
        )
    else:
        lean = (
            "E leaning towards D/training-coverage: GT start-segment delta (frames 0-14) stays "
            "well under the WOULD_CLAMP threshold and does not grow enough within the 0.5s window "
            "to explain the actual-Shadow policy bias; the policy's delivered first-action delta "
            "magnitude matches dataset GT displacement several hundred ms to ~1s further into a "
            "typical demonstration than frame 0 (see estimated_matching_horizon). This is "
            "consistent with an undertrained/undifferentiated flow-matching policy (D) rather than "
            "a training-time state/action mislabeling bug (C) - the dataset's own frame-0 labels "
            "are not the source of the oversized first action."
        )

    return {
        "A_gt_start_action_itself_large": {
            "verdict": a_gt_start_large,
            "gt_frame0_mean_abs_delta_deg": {"shoulder_lift": gt_frame0_sl, "elbow_flex": gt_frame0_ef},
            "would_clamp_threshold_deg": {"shoulder_lift": wc_sl, "elbow_flex": wc_ef},
        },
        "B_gt_normal_policy_large": {
            "verdict": b_policy_much_larger_than_gt,
            "policy_vs_gt_frame0_ratio": {"shoulder_lift": ratio_sl, "elbow_flex": ratio_ef},
        },
        "C_static_hold_action_label_misalignment": {
            "verdict": c_misalignment_material,
            "mean_action_leads_state_first_movement_by_frames": mean_lead,
            "n_episodes_with_both_defined": len(early_lead_frames),
            "gt_frame14_mean_abs_delta_deg": {"shoulder_lift": gt_frame14_sl, "elbow_flex": gt_frame14_ef},
        },
        "D_policy_firstaction_resembles_future_frame": {
            "estimated_future_frame_state_based": matching_horizon_state,
            "estimated_future_frame_action_based": matching_horizon_action,
            "approx_mean_matching_frame": d_future_frame_estimate,
            "approx_mean_matching_seconds": d_future_frame_estimate / FPS if d_future_frame_estimate else None,
        },
        "E_root_cause_lean": lean,
    }


# --------------------------------------------------------------------------
# Report writers
# --------------------------------------------------------------------------


def write_csv_per_episode(out_path: Path, episodes: dict[int, dict[str, np.ndarray]], first_movement: dict[int, dict[str, Any]], deltas: np.ndarray) -> None:
    eps = sorted(episodes.keys())
    sl_idx = JOINTS.index("shoulder_lift")
    ef_idx = JOINTS.index("elbow_flex")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "episode_index",
                "length_frames",
                "frame0_abs_delta_shoulder_lift",
                "frame0_abs_delta_elbow_flex",
                "frame14_abs_delta_shoulder_lift",
                "frame14_abs_delta_elbow_flex",
                "max_abs_delta_frames0_14_shoulder_lift",
                "max_abs_delta_frames0_14_elbow_flex",
                "state_first_movement_frame_any_joint",
                "state_first_movement_frame_shoulder_lift",
                "state_first_movement_frame_elbow_flex",
                "action_first_movement_frame_any_joint",
                "action_first_movement_frame_shoulder_lift",
                "action_first_movement_frame_elbow_flex",
            ]
        )
        for i, ep in enumerate(eps):
            mv = first_movement[ep]
            w.writerow(
                [
                    ep,
                    episodes[ep]["length"],
                    f"{abs(deltas[i, 0, sl_idx]):.4f}",
                    f"{abs(deltas[i, 0, ef_idx]):.4f}",
                    f"{abs(deltas[i, START_SEGMENT_FRAMES - 1, sl_idx]):.4f}",
                    f"{abs(deltas[i, START_SEGMENT_FRAMES - 1, ef_idx]):.4f}",
                    f"{np.nanmax(np.abs(deltas[i, :, sl_idx])):.4f}",
                    f"{np.nanmax(np.abs(deltas[i, :, ef_idx])):.4f}",
                    mv["state_first_movement_frame_any_joint"],
                    mv["state_first_movement_frame_shoulder_lift"],
                    mv["state_first_movement_frame_elbow_flex"],
                    mv["action_first_movement_frame_any_joint"],
                    mv["action_first_movement_frame_shoulder_lift"],
                    mv["action_first_movement_frame_elbow_flex"],
                ]
            )


def write_csv_per_frame(out_path: Path, per_frame_agg: dict[int, Any]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "time_s", "joint", "mean_delta", "median_delta", "mean_abs_delta", "median_abs_delta", "p95_abs_delta", "max_abs_delta"])
        for t in sorted(per_frame_agg.keys()):
            for j in JOINTS:
                r = per_frame_agg[t][j]
                w.writerow(
                    [
                        t,
                        f"{t / FPS:.4f}",
                        j,
                        f"{r['mean_delta']:.4f}",
                        f"{r['median_delta']:.4f}",
                        f"{r['mean_abs_delta']:.4f}",
                        f"{r['median_abs_delta']:.4f}",
                        f"{r['p95_abs_delta']:.4f}",
                        f"{r['max_abs_delta']:.4f}",
                    ]
                )


def write_markdown_report(out_path: Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Grid35 V2-clean: episode start-segment (0-0.5s) temporal-alignment audit")
    lines.append("")
    lines.append(
        f"Dataset: `{result['dataset_root']}` - 35 training episodes, {FPS}fps, "
        f"start segment = frames 0-{START_SEGMENT_FRAMES - 1} (0.0s-{(START_SEGMENT_FRAMES - 1) / FPS:.3f}s)."
    )
    lines.append("")
    lines.append(
        f"Movement-vs-noise threshold used: cumulative displacement from frame-0 state >= "
        f"**{MOVEMENT_THRESHOLD_DEG} deg**, sustained for **{PERSISTENCE_FRAMES} consecutive frames** "
        "(see script docstring for the data-based justification; noise-floor evidence in section 2 below)."
    )
    lines.append("")

    lines.append("## 1. Frame 0-14 GT action-state delta (shoulder_lift / elbow_flex)")
    lines.append("")
    lines.append("Aggregated across all 35 episodes, per frame index (0=episode start).")
    lines.append("")
    lines.append("| frame | time(s) | joint | mean\\|delta\\| | median\\|delta\\| | p95\\|delta\\| | max\\|delta\\| |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|")
    for t in sorted(result["per_frame_aggregate"].keys()):
        for j in KEY_JOINTS:
            r = result["per_frame_aggregate"][t][j]
            lines.append(
                f"| {t} | {t / FPS:.3f} | {j} | {r['mean_abs_delta']:.3f} | {r['median_abs_delta']:.3f} | "
                f"{r['p95_abs_delta']:.3f} | {r['max_abs_delta']:.3f} |"
            )
    lines.append("")
    wc = result["safety_thresholds"]["would_clamp_threshold_deg"]
    lines.append(
        f"For reference, Safety Gate WOULD_CLAMP thresholds (read-only, `configs/safety_gate.yaml`): "
        f"shoulder_lift={wc['shoulder_lift']}deg, elbow_flex={wc['elbow_flex']}deg."
    )
    lines.append("")

    lines.append("## 2. Noise-floor evidence (frame-to-frame |state[t]-state[t-1]|, frames 1-14, all episodes)")
    lines.append("")
    lines.append("| joint | n samples | frac exactly 0 | p50 | p90 | p99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for j in JOINTS:
        r = result["noise_floor_evidence"][j]
        lines.append(
            f"| {j} | {r['n_samples']} | {r['frac_exactly_zero']:.3f} | {r['p50']:.4f} | {r['p90']:.4f} | "
            f"{r['p99']:.4f} | {r['max']:.4f} |"
        )
    lines.append("")

    lines.append("## 3. 35-episode aggregate table: first movement frames + start-segment delta")
    lines.append("")
    lines.append(
        "| episode | length | \\|delta\\|@f0 SL | \\|delta\\|@f0 EF | \\|delta\\|@f14 SL | \\|delta\\|@f14 EF | "
        "state 1st-move (any) | state 1st-move SL | state 1st-move EF | action 1st-move (any) | "
        "action 1st-move SL | action 1st-move EF |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["per_episode_table"]:
        lines.append(
            f"| {row['episode']} | {row['length']} | {row['frame0_abs_delta_sl']:.3f} | "
            f"{row['frame0_abs_delta_ef']:.3f} | {row['frame14_abs_delta_sl']:.3f} | {row['frame14_abs_delta_ef']:.3f} | "
            f"{row['state_first_movement_any']} | {row['state_first_movement_sl']} | {row['state_first_movement_ef']} | "
            f"{row['action_first_movement_any']} | {row['action_first_movement_sl']} | {row['action_first_movement_ef']} |"
        )
    lines.append("")
    fm_summary = result["first_movement_summary"]
    lines.append(
        f"Median state first-movement frame (any joint): **{fm_summary['median_state_first_movement_any']}** "
        f"(~{fm_summary['median_state_first_movement_any_s']:.3f}s). "
        f"Median action first-movement frame (any joint): **{fm_summary['median_action_first_movement_any']}** "
        f"(~{fm_summary['median_action_first_movement_any_s']:.3f}s). "
        f"Median (state-action) lead = **{fm_summary['median_lead_frames']}** frames."
    )
    lines.append("")

    lines.append("## 4. Dataset-wide horizon trajectory (frames 1-30) vs. actual-Shadow policy first-action")
    lines.append("")
    lines.append("Dataset-wide mean `state[t]-state[0]` and `action[t]-state[0]` displacement (degrees):")
    lines.append("")
    ks = sorted(result["horizon_trajectories"]["state_minus_state0"].keys())
    header = "| joint | series | " + " | ".join(f"t={k}({k/FPS:.2f}s)" for k in ks) + " |"
    lines.append(header)
    lines.append("|---|---|" + "---:|" * len(ks))
    for j in KEY_JOINTS:
        row_state = [result["horizon_trajectories"]["state_minus_state0"][k]["mean"][j] for k in ks]
        row_action = [result["horizon_trajectories"]["action_minus_state0"][k]["mean"][j] for k in ks]
        lines.append(f"| {j} | state[t]-state[0] | " + " | ".join(f"{v:+.2f}" for v in row_state) + " |")
        lines.append(f"| {j} | action[t]-state[0] | " + " | ".join(f"{v:+.2f}" for v in row_action) + " |")
    lines.append("")

    lines.append(
        "Actual-Shadow policy first-action mean delta, averaged over T01-T10 "
        f"(`{result['actual_shadow_policy']['source_file']}`):"
    )
    lines.append("")
    lines.append("| joint | T01-T10 avg policy delta (deg) |")
    lines.append("|---|---:|")
    for j in KEY_JOINTS:
        v = result["actual_shadow_policy"]["t01_t10_average_per_joint_mean_delta_deg"][j]
        lines.append(f"| {j} | {v:+.2f} |")
    lines.append("")

    lines.append("Estimated dataset horizon (frames/seconds) whose GT displacement best matches that policy delta:")
    lines.append("")
    lines.append("| joint | basis | target delta (deg) | matching horizon (frames) | matching horizon (s) |")
    lines.append("|---|---|---:|---:|---:|")
    for j in KEY_JOINTS:
        for basis, key in [("state[t]-state[0]", "matching_horizon_state"), ("action[t]-state[0]", "matching_horizon_action")]:
            v = result[key][j]
            lines.append(
                f"| {j} | {basis} | {v['target_delta_deg']:+.2f} | {v['estimated_matching_horizon_frames']:.1f} | "
                f"{v['estimated_matching_horizon_seconds']:.3f} |"
            )
    lines.append("")

    lines.append(
        "Cross-reference: the earlier live-pipeline trace "
        "(`reports/grid35_v2_first_action_diagnostic_T01/`) found the *nearest-neighbor* GT state to "
        "the actual Shadow T01 observation at episode 33, frame 25 (L2~2.2deg) - i.e. **not** an "
        "episode-0/start-of-episode frame, but ~0.83s into a demonstration. Per-scene nearest-demo "
        "matches (T01-T10), read from the actual-Shadow summary JSON:"
    )
    lines.append("")
    lines.append("| scene | nearest episode | nearest frame | L2 dist (deg) |")
    lines.append("|---|---:|---:|---:|")
    for row in result["actual_shadow_policy"]["per_scene"]:
        ndm = row.get("nearest_demo_match") or {}
        lines.append(f"| {row['scene']} | {ndm.get('episode')} | {ndm.get('frame')} | {ndm.get('l2_dist_deg', float('nan')):.2f} |")
    lines.append("")

    lines.append("## 5. Safety threshold comparison (read-only)")
    lines.append("")
    lines.append("| joint | WOULD_CLAMP (deg) | REJECT (deg, x5) | GT mean\\|delta\\|@frame0 | GT mean\\|delta\\|@frame14 | actual policy mean delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for j in KEY_JOINTS:
        wc_j = result["safety_thresholds"]["would_clamp_threshold_deg"][j]
        rj_j = result["safety_thresholds"]["reject_threshold_deg"][j]
        gt0 = result["per_frame_aggregate"][0][j]["mean_abs_delta"]
        gt14 = result["per_frame_aggregate"][START_SEGMENT_FRAMES - 1][j]["mean_abs_delta"]
        pol = result["actual_shadow_policy"]["t01_t10_average_per_joint_mean_delta_deg"][j]
        lines.append(f"| {j} | {wc_j:.2f} | {rj_j:.2f} | {gt0:.3f} | {gt14:.3f} | {pol:+.2f} |")
    lines.append("")

    lines.append("## 6. Verdict")
    lines.append("")
    v = result["verdict"]
    lines.append("### A. GT start action itself large?")
    lines.append(f"**{v['A_gt_start_action_itself_large']['verdict']}** - "
                  f"frame-0 mean|delta|: shoulder_lift={v['A_gt_start_action_itself_large']['gt_frame0_mean_abs_delta_deg']['shoulder_lift']:.3f}deg, "
                  f"elbow_flex={v['A_gt_start_action_itself_large']['gt_frame0_mean_abs_delta_deg']['elbow_flex']:.3f}deg, "
                  f"vs. WOULD_CLAMP thresholds {v['A_gt_start_action_itself_large']['would_clamp_threshold_deg']}.")
    lines.append("")
    lines.append("### B. GT normal, policy alone large?")
    r_sl = v["B_gt_normal_policy_large"]["policy_vs_gt_frame0_ratio"]["shoulder_lift"]
    r_ef = v["B_gt_normal_policy_large"]["policy_vs_gt_frame0_ratio"]["elbow_flex"]
    lines.append(f"**{v['B_gt_normal_policy_large']['verdict']}** - policy delta / GT frame-0 delta ratio: "
                  f"shoulder_lift={r_sl:.1f}x, elbow_flex={r_ef:.1f}x.")
    lines.append("")
    lines.append("### C. Static-hold action-label misalignment?")
    c = v["C_static_hold_action_label_misalignment"]
    lines.append(f"**{c['verdict']}** - mean (state_first_move - action_first_move) lead = "
                  f"{c['mean_action_leads_state_first_movement_by_frames']:.2f} frames "
                  f"(n={c['n_episodes_with_both_defined']} episodes with both defined); "
                  f"frame-14 GT mean|delta|: shoulder_lift={c['gt_frame14_mean_abs_delta_deg']['shoulder_lift']:.3f}deg, "
                  f"elbow_flex={c['gt_frame14_mean_abs_delta_deg']['elbow_flex']:.3f}deg.")
    lines.append("")
    lines.append("### D. Policy first-action resembles which future GT frame?")
    d = v["D_policy_firstaction_resembles_future_frame"]
    lines.append(f"Approximate mean matching frame across shoulder_lift/elbow_flex (state-based): "
                  f"**{d['approx_mean_matching_frame']:.1f} frames** (~{d['approx_mean_matching_seconds']:.2f}s) "
                  "into a typical demonstration.")
    lines.append("")
    lines.append("### E. Root-cause lean")
    lines.append(v["E_root_cause_lean"])
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
    parser.add_argument("--actual-shadow-summary", type=Path, default=DEFAULT_ACTUAL_SHADOW_SUMMARY)
    parser.add_argument("--movement-threshold-deg", type=float, default=MOVEMENT_THRESHOLD_DEG)
    parser.add_argument("--persistence-frames", type=int, default=PERSISTENCE_FRAMES)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[analyze] 1/7 loading 35 episodes (full length)")
    episodes = load_episodes(dataset_root)
    assert len(episodes) == 35, f"expected 35 episodes, found {len(episodes)}"

    print("[analyze] 2/7 start-segment (frames 0-14) delta stats")
    deltas = compute_start_segment_deltas(episodes, START_SEGMENT_FRAMES)
    per_frame_agg = aggregate_per_frame(deltas)

    print("[analyze] 3/7 noise-floor evidence")
    noise_floor = compute_noise_floor_evidence(episodes, START_SEGMENT_FRAMES - 1)

    print("[analyze] 4/7 first-movement-frame estimation")
    first_movement = compute_first_movement_frames(episodes, args.movement_threshold_deg, args.persistence_frames)

    print("[analyze] 5/7 horizon trajectory (frames 1-30) + actual-Shadow policy cross-reference")
    horizon_traj = compute_horizon_trajectories(episodes, HORIZON_MAX_FRAME)
    actual_shadow = load_actual_shadow_policy_stats(args.actual_shadow_summary)
    target = actual_shadow["t01_t10_average_per_joint_mean_delta_deg"]
    matching_horizon_state = estimate_matching_horizon(horizon_traj["state_minus_state0"], target, KEY_JOINTS)
    matching_horizon_action = estimate_matching_horizon(horizon_traj["action_minus_state0"], target, KEY_JOINTS)

    print("[analyze] 6/7 safety threshold (read-only)")
    safety = load_safety_thresholds(args.safety_gate_yaml)

    print("[analyze] 7/7 verdict + reports")
    verdict = classify_verdict(
        per_frame_agg, first_movement, actual_shadow, safety, matching_horizon_state, matching_horizon_action
    )

    eps = sorted(episodes.keys())
    sl_idx = JOINTS.index("shoulder_lift")
    ef_idx = JOINTS.index("elbow_flex")
    per_episode_table = []
    state_first_any = []
    action_first_any = []
    leads = []
    for i, ep in enumerate(eps):
        mv = first_movement[ep]
        per_episode_table.append(
            {
                "episode": ep,
                "length": episodes[ep]["length"],
                "frame0_abs_delta_sl": abs(deltas[i, 0, sl_idx]),
                "frame0_abs_delta_ef": abs(deltas[i, 0, ef_idx]),
                "frame14_abs_delta_sl": abs(deltas[i, START_SEGMENT_FRAMES - 1, sl_idx]),
                "frame14_abs_delta_ef": abs(deltas[i, START_SEGMENT_FRAMES - 1, ef_idx]),
                "state_first_movement_any": mv["state_first_movement_frame_any_joint"],
                "state_first_movement_sl": mv["state_first_movement_frame_shoulder_lift"],
                "state_first_movement_ef": mv["state_first_movement_frame_elbow_flex"],
                "action_first_movement_any": mv["action_first_movement_frame_any_joint"],
                "action_first_movement_sl": mv["action_first_movement_frame_shoulder_lift"],
                "action_first_movement_ef": mv["action_first_movement_frame_elbow_flex"],
            }
        )
        if mv["state_first_movement_frame_any_joint"] is not None:
            state_first_any.append(mv["state_first_movement_frame_any_joint"])
        if mv["action_first_movement_frame_any_joint"] is not None:
            action_first_any.append(mv["action_first_movement_frame_any_joint"])
        if mv["state_first_movement_frame_any_joint"] is not None and mv["action_first_movement_frame_any_joint"] is not None:
            leads.append(mv["state_first_movement_frame_any_joint"] - mv["action_first_movement_frame_any_joint"])

    first_movement_summary = {
        "median_state_first_movement_any": float(np.median(state_first_any)) if state_first_any else None,
        "median_state_first_movement_any_s": float(np.median(state_first_any)) / FPS if state_first_any else None,
        "median_action_first_movement_any": float(np.median(action_first_any)) if action_first_any else None,
        "median_action_first_movement_any_s": float(np.median(action_first_any)) / FPS if action_first_any else None,
        "median_lead_frames": float(np.median(leads)) if leads else None,
        "n_episodes_state_movement_found": len(state_first_any),
        "n_episodes_action_movement_found": len(action_first_any),
    }

    result = {
        "dataset_root": str(dataset_root),
        "n_episodes": len(episodes),
        "start_segment_frames": START_SEGMENT_FRAMES,
        "horizon_max_frame": HORIZON_MAX_FRAME,
        "movement_threshold_deg": args.movement_threshold_deg,
        "persistence_frames": args.persistence_frames,
        "per_frame_aggregate": per_frame_agg,
        "noise_floor_evidence": noise_floor,
        "per_episode_table": per_episode_table,
        "first_movement_summary": first_movement_summary,
        "horizon_trajectories": horizon_traj,
        "actual_shadow_policy": actual_shadow,
        "matching_horizon_state": matching_horizon_state,
        "matching_horizon_action": matching_horizon_action,
        "safety_thresholds": safety,
        "verdict": verdict,
    }

    json_path = out_dir / "start_segment_temporal_alignment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"[analyze] wrote {json_path}")

    csv_episode_path = out_dir / "per_episode_start_segment.csv"
    write_csv_per_episode(csv_episode_path, episodes, first_movement, deltas)
    print(f"[analyze] wrote {csv_episode_path}")

    csv_frame_path = out_dir / "per_frame_aggregate.csv"
    write_csv_per_frame(csv_frame_path, per_frame_agg)
    print(f"[analyze] wrote {csv_frame_path}")

    md_path = out_dir / "start_segment_temporal_alignment.md"
    write_markdown_report(md_path, result)
    print(f"[analyze] wrote {md_path}")

    print("[analyze] VERDICT summary:")
    print(f"  A (GT start large): {verdict['A_gt_start_action_itself_large']['verdict']}")
    print(f"  B (policy >> GT): {verdict['B_gt_normal_policy_large']['verdict']}")
    print(f"  C (static-hold misalignment material): {verdict['C_static_hold_action_label_misalignment']['verdict']}")
    print(f"  D (matches future frame ~): {verdict['D_policy_firstaction_resembles_future_frame']['approx_mean_matching_frame']}")
    print(f"  E: {verdict['E_root_cause_lean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
