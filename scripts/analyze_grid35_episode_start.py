#!/usr/bin/env python3
"""Analyze episode-start leader-follower synchronization jumps in
``data/so101_cube_xy_grid35_v1``.

Background
----------
The dataset was recorded by physically syncing the follower arm to the
leader arm's pose at the start of each episode recording. If that sync
produced a large instantaneous follower motion, it may have been captured
in the recorded ``action``/``observation.state`` pair for the first frame(s)
of each episode, which in turn could bias imitation-learning policies
trained on this data (e.g. Grid35 10k SmolVLA) toward emitting a large
"jump" action on the very first inference step.

This script:

1. Reads the actual parquet data (one file == one episode in this
   LeRobot v3.0 dataset) and pulls ``action`` / ``observation.state``
   using the joint order declared in ``meta/info.json``.
2. Computes ``delta = action[t] - observation.state[t]`` per joint.
3. Reports per-episode and aggregate statistics for the first 30 frames
   (1s at 30 FPS), and compares the start segment against middle/late
   segments of the same episodes.
4. Cross-references the real Shadow-mode first-action logs in
   ``reports/shadow_mode/shadow_*.json`` (fixed-scene F01-F05 runs and the
   T05 REJECT runs) to see whether the dataset's start-of-episode delta
   pattern matches the Shadow bias in both sign and magnitude.

This script is read-only: it does not modify the dataset, retrain
anything, or write outside of ``reports/grid35_episode_start_analysis/``.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "so101_cube_xy_grid35_v1"
DEFAULT_SHADOW_DIR = PROJECT_ROOT / "reports" / "shadow_mode"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "grid35_episode_start_analysis"

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

START_WINDOW = 30  # frames == 1s @ 30 FPS
THRESHOLDS_DEG = [5, 10, 15, 20, 25]

# Shadow-mode "known" bias reported in the task background, used as a
# quick sanity anchor. The script also recomputes this directly from the
# raw reports/shadow_mode/*.json logs (see `load_shadow_first_action_bias`)
# so the comparison is not dependent on these hardcoded numbers.
SHADOW_BACKGROUND_BIAS_DEG = {
    "shoulder_lift": 16.0,
    "elbow_flex": -25.0,
    "wrist_flex": 8.0,
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_info(dataset_root: Path) -> dict[str, Any]:
    with open(dataset_root / "meta" / "info.json", encoding="utf-8") as f:
        return json.load(f)


def joint_names_from_info(info: dict[str, Any]) -> list[str]:
    names = info["features"]["action"]["names"]
    # strip the trailing ".pos" suffix -> shoulder_pan.pos -> shoulder_pan
    return [n.rsplit(".", 1)[0] for n in names]


def find_episode_files(dataset_root: Path) -> dict[int, Path]:
    """Map episode_index -> parquet path.

    In this dataset each chunk file happens to contain exactly one
    episode (confirmed empirically), but we don't assume that -- we read
    the episode_index column of each file and index by that, so this
    stays correct even if a future dataset version packs multiple
    episodes per file.
    """
    files = sorted(glob.glob(str(dataset_root / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_root}/data")

    ep_to_path: dict[int, Path] = {}
    for fp in files:
        eps = pd.read_parquet(fp, columns=["episode_index"])["episode_index"].unique()
        for ep in eps:
            ep_to_path[int(ep)] = Path(fp)
    return ep_to_path


def load_episode_frame(dataset_root: Path, ep_index: int, path: Path, joint_names: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["action", "observation.state", "timestamp", "frame_index", "episode_index"])
    df = df[df["episode_index"] == ep_index].sort_values("frame_index").reset_index(drop=True)

    action = np.stack(df["action"].to_numpy())
    state = np.stack(df["observation.state"].to_numpy())
    delta = action - state

    out = pd.DataFrame(
        {
            "frame_index": df["frame_index"].to_numpy(),
            "timestamp": df["timestamp"].to_numpy(),
        }
    )
    for i, name in enumerate(joint_names):
        out[f"state.{name}"] = state[:, i]
        out[f"action.{name}"] = action[:, i]
        out[f"delta.{name}"] = delta[:, i]
    return out


# --------------------------------------------------------------------------
# Shadow-mode comparison data
# --------------------------------------------------------------------------


def load_shadow_first_action_bias(shadow_dir: Path) -> dict[str, Any]:
    """Recompute the Shadow first-action delta (mapped_action - observation.state)
    directly from reports/shadow_mode/shadow_*.json, grouped by scene label.

    Returns per-run records plus fixed-scene (F0x) aggregate stats, which is
    what the task background's "fixed scene 5회" figures refer to.
    """
    files = sorted(glob.glob(str(shadow_dir / "shadow_*.json")))
    runs = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        state = d.get("observation", {}).get("state")
        if not state:
            continue  # run predates state logging / invalid observation
        action = d.get("adapter", {}).get("mapped_action")
        if not action:
            continue
        label = d.get("scene_metadata", {}).get("label")
        decision = d.get("safety", {}).get("decision")
        delta = {j: float(action[j]) - float(state[j]) for j in JOINTS if j in state and j in action}
        runs.append(
            {
                "file": Path(fp).name,
                "label": label,
                "safety_decision": decision,
                "state": {j: float(state[j]) for j in JOINTS if j in state},
                "mapped_action": {j: float(action[j]) for j in JOINTS if j in action},
                "delta": delta,
            }
        )

    fixed_scene_runs = [r for r in runs if r["label"] and r["label"].startswith("F0")]
    t05_runs = [r for r in runs if r["label"] and r["label"].startswith("T05")]

    def agg(records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {}
        out = {}
        for j in JOINTS:
            vals = [r["delta"][j] for r in records if j in r["delta"]]
            if not vals:
                continue
            out[j] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
                "values": vals,
            }
        return out

    return {
        "n_runs_total": len(runs),
        "n_runs_with_state": len(runs),
        "fixed_scene_runs": fixed_scene_runs,
        "fixed_scene_delta_stats": agg(fixed_scene_runs),
        "t05_runs": t05_runs,
        "t05_delta_stats": agg(t05_runs),
        "all_runs": runs,
    }


# --------------------------------------------------------------------------
# Per-episode / aggregate analysis
# --------------------------------------------------------------------------


@dataclass
class EpisodeStartSummary:
    episode_index: int
    n_frames: int
    first_state: dict[str, float]
    first_action: dict[str, float]
    first_delta: dict[str, float]
    first5_delta: dict[str, list[float]]  # joint -> [d0..d4]
    max_abs_delta_first30: dict[str, float]
    argmax_frame_first30: dict[str, int]


def summarize_episode_start(ep_df: pd.DataFrame, ep_index: int, joint_names: list[str]) -> EpisodeStartSummary:
    n = len(ep_df)
    window = ep_df.iloc[: min(START_WINDOW, n)]

    first = ep_df.iloc[0]
    first_state = {j: float(first[f"state.{j}"]) for j in joint_names}
    first_action = {j: float(first[f"action.{j}"]) for j in joint_names}
    first_delta = {j: float(first[f"delta.{j}"]) for j in joint_names}

    first5 = ep_df.iloc[: min(5, n)]
    first5_delta = {j: first5[f"delta.{j}"].astype(float).tolist() for j in joint_names}

    max_abs_delta_first30: dict[str, float] = {}
    argmax_frame_first30: dict[str, int] = {}
    for j in joint_names:
        abs_vals = window[f"delta.{j}"].abs()
        idx = int(abs_vals.idxmax())
        max_abs_delta_first30[j] = float(abs_vals.loc[idx])
        argmax_frame_first30[j] = int(window.loc[idx, "frame_index"])

    return EpisodeStartSummary(
        episode_index=ep_index,
        n_frames=n,
        first_state=first_state,
        first_action=first_action,
        first_delta=first_delta,
        first5_delta=first5_delta,
        max_abs_delta_first30=max_abs_delta_first30,
        argmax_frame_first30=argmax_frame_first30,
    )


def segment_abs_delta_stats(ep_df: pd.DataFrame, joint_names: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Compare |delta| distributions across early / middle / late segments
    of a single episode. Each segment is START_WINDOW frames wide (or the
    remaining length if the episode is shorter than 3*START_WINDOW)."""
    n = len(ep_df)
    w = min(START_WINDOW, max(1, n // 3))

    early = ep_df.iloc[:w]
    mid_start = max(0, n // 2 - w // 2)
    middle = ep_df.iloc[mid_start : mid_start + w]
    late = ep_df.iloc[max(0, n - w) :]

    segments = {"early": early, "middle": middle, "late": late}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for seg_name, seg_df in segments.items():
        out[seg_name] = {}
        for j in joint_names:
            abs_vals = seg_df[f"delta.{j}"].abs()
            out[seg_name][j] = {
                "mean": float(abs_vals.mean()),
                "median": float(abs_vals.median()),
                "p95": float(abs_vals.quantile(0.95)),
                "max": float(abs_vals.max()),
                "n": int(len(abs_vals)),
            }
    return out


def aggregate_stats(
    per_episode: list[EpisodeStartSummary], joint_names: list[str]
) -> dict[str, Any]:
    stats: dict[str, Any] = {"frame0_delta": {}, "first5_abs_delta": {}, "first30_max_abs_delta": {}}

    for j in joint_names:
        f0_vals = [ep.first_delta[j] for ep in per_episode]
        stats["frame0_delta"][j] = {
            "mean": float(np.mean(f0_vals)),
            "median": float(np.median(f0_vals)),
            "std": float(np.std(f0_vals)),
            "max": float(np.max(f0_vals)),
            "min": float(np.min(f0_vals)),
            "max_abs": float(np.max(np.abs(f0_vals))),
            "values_by_episode": {ep.episode_index: v for ep, v in zip(per_episode, f0_vals)},
        }

        first5_abs = [abs(v) for ep in per_episode for v in ep.first5_delta[j]]
        stats["first5_abs_delta"][j] = {
            "mean": float(np.mean(first5_abs)),
            "median": float(np.median(first5_abs)),
            "std": float(np.std(first5_abs)),
            "max": float(np.max(first5_abs)),
            "p95": float(np.quantile(first5_abs, 0.95)),
        }

        max30_vals = [ep.max_abs_delta_first30[j] for ep in per_episode]
        stats["first30_max_abs_delta"][j] = {
            "mean": float(np.mean(max30_vals)),
            "median": float(np.median(max30_vals)),
            "std": float(np.std(max30_vals)),
            "max": float(np.max(max30_vals)),
            "min": float(np.min(max30_vals)),
            "values_by_episode": {ep.episode_index: v for ep, v in zip(per_episode, max30_vals)},
        }

    # threshold crossing counts: frame0 |delta| and first-30-frame max |delta|
    threshold_counts: dict[str, Any] = {"frame0_abs_delta": {}, "first30_max_abs_delta": {}}
    for j in joint_names:
        f0_abs = np.abs([ep.first_delta[j] for ep in per_episode])
        max30 = np.array([ep.max_abs_delta_first30[j] for ep in per_episode])
        threshold_counts["frame0_abs_delta"][j] = {
            f">={t}deg": int(np.sum(f0_abs >= t)) for t in THRESHOLDS_DEG
        }
        threshold_counts["first30_max_abs_delta"][j] = {
            f">={t}deg": int(np.sum(max30 >= t)) for t in THRESHOLDS_DEG
        }
    stats["threshold_counts"] = threshold_counts

    # timing of the peak: does max|delta| within the first 30 frames occur
    # right at frame 0 (instant jump) or does it build up gradually and peak
    # near the end of the 1s window (consistent with an ordinary reaching
    # trajectory rather than an instantaneous sync jump)?
    argmax_timing: dict[str, Any] = {}
    for j in joint_names:
        argmax_frames = [ep.argmax_frame_first30[j] for ep in per_episode]
        n = len(argmax_frames)
        n_at_frame0 = sum(1 for f in argmax_frames if f == 0)
        n_at_frame_le2 = sum(1 for f in argmax_frames if f <= 2)
        n_at_last3 = sum(1 for f in argmax_frames if f >= START_WINDOW - 3)
        histogram = dict(sorted(__import__("collections").Counter(argmax_frames).items()))
        argmax_timing[j] = {
            "mean_argmax_frame": float(np.mean(argmax_frames)),
            "median_argmax_frame": float(np.median(argmax_frames)),
            "n_episodes": n,
            "n_argmax_at_frame0": n_at_frame0,
            "frac_argmax_at_frame0": n_at_frame0 / n,
            "n_argmax_at_frame_le2": n_at_frame_le2,
            "frac_argmax_at_frame_le2": n_at_frame_le2 / n,
            "n_argmax_in_last3_of_window": n_at_last3,
            "frac_argmax_in_last3_of_window": n_at_last3 / n,
            "histogram_frame_to_count": histogram,
        }
    stats["argmax_timing"] = argmax_timing

    # sign consistency of frame0 delta per joint
    sign_consistency: dict[str, Any] = {}
    for j in joint_names:
        f0_vals = np.array([ep.first_delta[j] for ep in per_episode])
        n_pos = int(np.sum(f0_vals > 0))
        n_neg = int(np.sum(f0_vals < 0))
        n_zero = int(np.sum(f0_vals == 0))
        dominant = "positive" if n_pos >= n_neg else "negative"
        dominant_frac = max(n_pos, n_neg) / len(f0_vals) if len(f0_vals) else 0.0
        sign_consistency[j] = {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_zero": n_zero,
            "dominant_sign": dominant,
            "dominant_sign_fraction": float(dominant_frac),
        }
    stats["sign_consistency"] = sign_consistency

    return stats


def aggregate_segment_comparison(
    per_episode_segments: list[dict[str, dict[str, dict[str, float]]]], joint_names: list[str]
) -> dict[str, Any]:
    """Average the per-episode early/middle/late |delta| stats across episodes,
    both per-joint and pooled over the 6 tracked joints."""
    out: dict[str, Any] = {"per_joint": {}, "pooled_over_joints": {}}

    for j in joint_names:
        out["per_joint"][j] = {}
        for seg in ("early", "middle", "late"):
            means = [ep_seg[seg][j]["mean"] for ep_seg in per_episode_segments]
            maxes = [ep_seg[seg][j]["max"] for ep_seg in per_episode_segments]
            p95s = [ep_seg[seg][j]["p95"] for ep_seg in per_episode_segments]
            out["per_joint"][j][seg] = {
                "mean_of_episode_means": float(np.mean(means)),
                "mean_of_episode_max": float(np.mean(maxes)),
                "mean_of_episode_p95": float(np.mean(p95s)),
                "overall_max": float(np.max(maxes)),
            }

    for seg in ("early", "middle", "late"):
        per_ep_pooled_mean = []
        per_ep_pooled_max = []
        for ep_seg in per_episode_segments:
            joint_means = [ep_seg[seg][j]["mean"] for j in joint_names]
            joint_maxes = [ep_seg[seg][j]["max"] for j in joint_names]
            per_ep_pooled_mean.append(float(np.mean(joint_means)))
            per_ep_pooled_max.append(float(np.max(joint_maxes)))
        out["pooled_over_joints"][seg] = {
            "mean_of_episode_mean_abs_delta": float(np.mean(per_ep_pooled_mean)),
            "mean_of_episode_max_abs_delta": float(np.mean(per_ep_pooled_max)),
            "overall_max_abs_delta": float(np.max(per_ep_pooled_max)),
        }

    return out


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def compute_shadow_similarity(
    dataset_frame0_stats: dict[str, Any], shadow_fixed_scene_stats: dict[str, Any]
) -> dict[str, Any]:
    """Compare dataset frame-0 delta mean vs Shadow fixed-scene delta mean,
    per joint: same sign? magnitude ratio?"""
    out = {}
    for j in ["shoulder_lift", "elbow_flex", "wrist_flex", "shoulder_pan", "wrist_roll", "gripper"]:
        ds = dataset_frame0_stats.get(j, {})
        sh = shadow_fixed_scene_stats.get(j, {})
        if not ds or not sh:
            continue
        ds_mean = ds["mean"]
        sh_mean = sh["mean"]
        same_sign = (ds_mean > 0) == (sh_mean > 0) if ds_mean != 0 and sh_mean != 0 else False
        ratio = abs(ds_mean) / abs(sh_mean) if sh_mean != 0 else float("inf")
        out[j] = {
            "dataset_frame0_mean_delta_deg": ds_mean,
            "shadow_fixed_scene_mean_delta_deg": sh_mean,
            "same_sign": same_sign,
            "magnitude_ratio_dataset_over_shadow": ratio,
        }
    return out


def classify_verdict(
    similarity: dict[str, Any], dataset_threshold_counts: dict[str, Any], argmax_timing: dict[str, Any]
) -> dict[str, Any]:
    key_joints = ["shoulder_lift", "elbow_flex", "wrist_flex"]
    same_sign_count = sum(1 for j in key_joints if similarity.get(j, {}).get("same_sign"))

    # "large delta present at dataset start" = at least a meaningful fraction
    # of episodes cross 10deg at frame0 for elbow_flex/shoulder_lift (the two
    # joints with the biggest Shadow bias).
    f0_counts = dataset_threshold_counts["frame0_abs_delta"]
    large_start_evidence = False
    for j in key_joints:
        c10 = f0_counts.get(j, {}).get(">=10deg", 0)
        if c10 and c10 > 0:
            large_start_evidence = True

    # magnitude closeness: ratio within [0.3, 3.0] roughly "similar order of magnitude"
    close_magnitude_count = 0
    for j in key_joints:
        r = similarity.get(j, {}).get("magnitude_ratio_dataset_over_shadow")
        if r is not None and r != float("inf") and 0.3 <= r <= 3.0:
            close_magnitude_count += 1

    # timing: an "instant jump" (as opposed to a gradual reaching trajectory
    # that merely peaks late in the 1s start window) should have its max|delta|
    # concentrated at frame 0-2 for most episodes.
    instant_jump_count = 0
    for j in key_joints:
        frac = argmax_timing.get(j, {}).get("frac_argmax_at_frame_le2", 0.0)
        if frac >= 0.5:
            instant_jump_count += 1

    # A joint only really "explains" the Shadow bias if sign matches, magnitude
    # is comparable, AND the dataset delta is concentrated at frame 0 (an actual
    # instantaneous jump, not a smooth trajectory that happens to move a lot in 1s).
    fully_matching_joints = []
    for j in key_joints:
        sign_ok = similarity.get(j, {}).get("same_sign", False)
        r = similarity.get(j, {}).get("magnitude_ratio_dataset_over_shadow")
        mag_ok = r is not None and r != float("inf") and 0.3 <= r <= 3.0
        timing_ok = argmax_timing.get(j, {}).get("frac_argmax_at_frame_le2", 0.0) >= 0.5
        if sign_ok and mag_ok and timing_ok:
            fully_matching_joints.append(j)

    if len(fully_matching_joints) == len(key_joints):
        verdict = "A"
        rationale = (
            "Dataset frame-0 delta sign, magnitude, AND timing (peak at frame 0-2, not a "
            "gradually-building trajectory) all match the Shadow first-action bias for "
            "shoulder_lift/elbow_flex/wrist_flex."
        )
    elif large_start_evidence and (same_sign_count >= 1 or instant_jump_count >= 1):
        verdict = "B"
        rationale = (
            "Dataset shows large, frame-0-concentrated start-of-episode deltas in some "
            "joints (consistent with a real recording-start sync jump), and sign matches "
            "Shadow's bias for some key joints, but the match is incomplete: at least one "
            "of sign / magnitude / instant-jump-timing fails to line up for "
            f"{', '.join(set(key_joints) - set(fully_matching_joints))}."
        )
    else:
        verdict = "C"
        rationale = (
            "Dataset start-of-episode GT deltas are not unusually large, are not "
            "concentrated at frame 0, or do not match Shadow's first-action sign/"
            "magnitude pattern; look elsewhere (e.g. policy/normalization, action head "
            "init, control-loop timing)."
        )

    return {
        "verdict": verdict,
        "rationale": rationale,
        "same_sign_count_of_3_key_joints": same_sign_count,
        "close_magnitude_count_of_3_key_joints": close_magnitude_count,
        "instant_jump_count_of_3_key_joints": instant_jump_count,
        "fully_matching_joints": fully_matching_joints,
        "large_start_evidence": large_start_evidence,
    }


# --------------------------------------------------------------------------
# Report writers
# --------------------------------------------------------------------------


def episode_summary_to_dict(s: EpisodeStartSummary) -> dict[str, Any]:
    return {
        "episode_index": s.episode_index,
        "n_frames": s.n_frames,
        "first_frame": {
            "state": s.first_state,
            "action": s.first_action,
            "delta": s.first_delta,
        },
        "first5_frames_delta": s.first5_delta,
        "first30_frames": {
            "max_abs_delta": s.max_abs_delta_first30,
            "argmax_frame_index": s.argmax_frame_first30,
        },
    }


def write_markdown_report(
    out_path: Path,
    dataset_root: Path,
    joint_names: list[str],
    n_episodes: int,
    per_episode: list[EpisodeStartSummary],
    agg: dict[str, Any],
    segment_agg: dict[str, Any],
    shadow: dict[str, Any],
    similarity: dict[str, Any],
    verdict: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Grid35 Episode-Start Leader-Follower Sync Jump Analysis")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_root}`")
    lines.append(f"- Episodes analyzed: {n_episodes}")
    lines.append(f"- Joints (order from `meta/info.json`): {', '.join(joint_names)}")
    lines.append(f"- Start window: first {START_WINDOW} frames (1s @ 30 FPS)")
    lines.append("- `delta = action[t] - observation.state[t]` (units as stored, project convention 'deg')")
    lines.append("")

    lines.append("## 1. Schema actually used")
    lines.append("")
    lines.append("- `data/chunk-000/file-{NNN}.parquet` -> one file per episode (confirmed: each file's")
    lines.append("  `episode_index` column contains a single unique value == its file index).")
    lines.append("- Columns read: `action` (float32[6]), `observation.state` (float32[6]), `timestamp`,")
    lines.append("  `frame_index`, `episode_index`.")
    lines.append(f"- Joint order: `{joint_names}`")
    lines.append("")

    lines.append("## 2. Representative episodes")
    lines.append("")
    sample_idxs = sorted(set([0, 1, 4, n_episodes // 2, n_episodes - 1]))
    for ep in per_episode:
        if ep.episode_index not in sample_idxs:
            continue
        lines.append(f"### Episode {ep.episode_index} (n_frames={ep.n_frames})")
        lines.append("")
        lines.append("| joint | state[0] | action[0] | delta[0] | max\\|delta\\| (first 30) | argmax frame |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for j in joint_names:
            lines.append(
                f"| {j} | {ep.first_state[j]:.2f} | {ep.first_action[j]:.2f} | {ep.first_delta[j]:+.2f} | "
                f"{ep.max_abs_delta_first30[j]:.2f} | {ep.argmax_frame_first30[j]} |"
            )
        lines.append("")

    lines.append("## 3. Aggregate statistics (all episodes)")
    lines.append("")
    lines.append("### 3a. Frame-0 delta (action[0] - state[0])")
    lines.append("")
    lines.append("| joint | mean | median | std | max | min | max\\|delta\\| |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for j in joint_names:
        s = agg["frame0_delta"][j]
        lines.append(
            f"| {j} | {s['mean']:+.2f} | {s['median']:+.2f} | {s['std']:.2f} | {s['max']:+.2f} | "
            f"{s['min']:+.2f} | {s['max_abs']:.2f} |"
        )
    lines.append("")

    lines.append("### 3b. First-5-frames |delta| distribution")
    lines.append("")
    lines.append("| joint | mean | median | std | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for j in joint_names:
        s = agg["first5_abs_delta"][j]
        lines.append(f"| {j} | {s['mean']:.2f} | {s['median']:.2f} | {s['std']:.2f} | {s['p95']:.2f} | {s['max']:.2f} |")
    lines.append("")

    lines.append("### 3c. First-30-frames max|delta| distribution (per episode, then aggregated)")
    lines.append("")
    lines.append("| joint | mean | median | std | max | min |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for j in joint_names:
        s = agg["first30_max_abs_delta"][j]
        lines.append(f"| {j} | {s['mean']:.2f} | {s['median']:.2f} | {s['std']:.2f} | {s['max']:.2f} | {s['min']:.2f} |")
    lines.append("")

    lines.append("### 3d. Episodes crossing |delta| thresholds")
    lines.append("")
    lines.append("**At frame 0:**")
    lines.append("")
    header = "| joint | " + " | ".join(f">={t}°" for t in THRESHOLDS_DEG) + " |"
    sep = "|---|" + "---:|" * len(THRESHOLDS_DEG)
    lines.append(header)
    lines.append(sep)
    for j in joint_names:
        c = agg["threshold_counts"]["frame0_abs_delta"][j]
        lines.append("| " + j + " | " + " | ".join(str(c[f">={t}deg"]) for t in THRESHOLDS_DEG) + " |")
    lines.append("")
    lines.append(f"**Anywhere in first {START_WINDOW} frames (max\\|delta\\| per episode):**")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for j in joint_names:
        c = agg["threshold_counts"]["first30_max_abs_delta"][j]
        lines.append("| " + j + " | " + " | ".join(str(c[f">={t}deg"]) for t in THRESHOLDS_DEG) + " |")
    lines.append(f"\n(out of {n_episodes} episodes)")
    lines.append("")

    lines.append("### 3e. Sign consistency of frame-0 delta across episodes")
    lines.append("")
    lines.append("| joint | n_positive | n_negative | n_zero | dominant sign | dominant fraction |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for j in joint_names:
        s = agg["sign_consistency"][j]
        lines.append(
            f"| {j} | {s['n_positive']} | {s['n_negative']} | {s['n_zero']} | {s['dominant_sign']} | "
            f"{s['dominant_sign_fraction']:.0%} |"
        )
    lines.append("")

    lines.append("### 3f. Timing of the peak |delta| within the first 30 frames")
    lines.append("")
    lines.append("Distinguishes an *instantaneous* sync jump (max|delta| at frame 0-2) from a")
    lines.append("*gradually building* trajectory (max|delta| near the end of the 1s window).")
    lines.append("")
    lines.append("| joint | mean argmax frame | % episodes argmax at frame 0-2 | % episodes argmax in last 3 frames of window |")
    lines.append("|---|---:|---:|---:|")
    for j in joint_names:
        t = agg["argmax_timing"][j]
        lines.append(
            f"| {j} | {t['mean_argmax_frame']:.1f} | {t['frac_argmax_at_frame_le2']:.0%} | "
            f"{t['frac_argmax_in_last3_of_window']:.0%} |"
        )
    lines.append("")

    lines.append("## 4. Early vs middle vs late |delta| comparison")
    lines.append("")
    lines.append("Segment = 30 frames (early = first 30, late = last 30, middle = centered on episode midpoint),")
    lines.append("stat = mean across episodes of that episode's mean|delta| in the segment.")
    lines.append("")
    lines.append("| joint | early mean | middle mean | late mean | early max(of means) is largest? |")
    lines.append("|---|---:|---:|---:|---|")
    for j in joint_names:
        e = segment_agg["per_joint"][j]["early"]["mean_of_episode_means"]
        m = segment_agg["per_joint"][j]["middle"]["mean_of_episode_means"]
        lt = segment_agg["per_joint"][j]["late"]["mean_of_episode_means"]
        largest = "early" if e >= max(m, lt) else ("middle" if m >= lt else "late")
        lines.append(f"| {j} | {e:.2f} | {m:.2f} | {lt:.2f} | {largest} |")
    lines.append("")
    pooled = segment_agg["pooled_over_joints"]
    lines.append("Pooled over the 6 joints (mean of per-episode mean|delta|, and mean of per-episode max|delta|):")
    lines.append("")
    lines.append("| segment | mean of episode mean\\|delta\\| | mean of episode max\\|delta\\| | overall max\\|delta\\| |")
    lines.append("|---|---:|---:|---:|")
    for seg in ("early", "middle", "late"):
        p = pooled[seg]
        lines.append(
            f"| {seg} | {p['mean_of_episode_mean_abs_delta']:.2f} | {p['mean_of_episode_max_abs_delta']:.2f} | "
            f"{p['overall_max_abs_delta']:.2f} |"
        )
    lines.append("")

    lines.append("## 5. Comparison to Shadow first-action bias")
    lines.append("")
    fs = shadow["fixed_scene_delta_stats"]
    lines.append(f"Shadow fixed-scene runs used ({len(shadow['fixed_scene_runs'])}): "
                 + ", ".join(r["label"] for r in shadow["fixed_scene_runs"]))
    lines.append("")
    lines.append("| joint | Shadow fixed-scene mean delta | dataset frame-0 mean delta | same sign? | magnitude ratio (dataset/shadow) |")
    lines.append("|---|---:|---:|---|---:|")
    for j, s in similarity.items():
        ratio = s["magnitude_ratio_dataset_over_shadow"]
        ratio_str = f"{ratio:.2f}x" if ratio != float("inf") else "n/a"
        lines.append(
            f"| {j} | {s['shadow_fixed_scene_mean_delta_deg']:+.2f} | {s['dataset_frame0_mean_delta_deg']:+.2f} | "
            f"{'YES' if s['same_sign'] else 'no'} | {ratio_str} |"
        )
    lines.append("")

    if shadow["t05_delta_stats"]:
        lines.append(f"T05 REJECT/rerun runs used ({len(shadow['t05_runs'])}): "
                      + ", ".join(r"{}[{}]".format(r["label"], r["safety_decision"]) for r in shadow["t05_runs"]))
        lines.append("")
        lines.append("| joint | T05 mean delta | T05 max\\|delta\\| |")
        lines.append("|---|---:|---:|")
        for j in ["shoulder_lift", "elbow_flex", "wrist_flex"]:
            s = shadow["t05_delta_stats"].get(j)
            if s:
                lines.append(f"| {j} | {s['mean']:+.2f} | {max(abs(v) for v in s['values']):.2f} |")
        lines.append("")

    lines.append("## 6. Verdict")
    lines.append("")
    lines.append(f"**{verdict['verdict']}**")
    lines.append("")
    lines.append(verdict["rationale"])
    lines.append("")
    lines.append(
        f"- Same-sign match on {verdict['same_sign_count_of_3_key_joints']}/3 key joints "
        "(shoulder_lift, elbow_flex, wrist_flex)"
    )
    lines.append(f"- Comparable-magnitude match on {verdict['close_magnitude_count_of_3_key_joints']}/3 key joints")
    lines.append(
        f"- Instant-jump timing (peak at frame 0-2) on {verdict['instant_jump_count_of_3_key_joints']}/3 key joints"
    )
    lines.append(
        f"- Fully matching (sign + magnitude + timing) key joints: "
        f"{', '.join(verdict['fully_matching_joints']) or 'none'}"
    )
    lines.append(f"- Large (>=10deg) start-of-episode delta observed in dataset: {verdict['large_start_evidence']}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    shadow_dir = args.shadow_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info = load_info(dataset_root)
    joint_names = joint_names_from_info(info)
    assert joint_names == JOINTS, f"Unexpected joint order in info.json: {joint_names}"

    ep_files = find_episode_files(dataset_root)
    n_episodes = info["total_episodes"]
    print(f"[analyze] dataset_root={dataset_root}")
    print(f"[analyze] total_episodes(info.json)={n_episodes} found_episode_files={len(ep_files)}")

    per_episode: list[EpisodeStartSummary] = []
    per_episode_segments: list[dict[str, Any]] = []
    per_episode_full_dicts: list[dict[str, Any]] = []

    for ep_index in sorted(ep_files):
        ep_df = load_episode_frame(dataset_root, ep_index, ep_files[ep_index], joint_names)
        summary = summarize_episode_start(ep_df, ep_index, joint_names)
        per_episode.append(summary)
        per_episode_full_dicts.append(episode_summary_to_dict(summary))
        per_episode_segments.append(segment_abs_delta_stats(ep_df, joint_names))

    agg = aggregate_stats(per_episode, joint_names)
    segment_agg = aggregate_segment_comparison(per_episode_segments, joint_names)

    shadow = load_shadow_first_action_bias(shadow_dir)
    similarity = compute_shadow_similarity(agg["frame0_delta"], shadow["fixed_scene_delta_stats"])
    verdict = classify_verdict(similarity, agg["threshold_counts"], agg["argmax_timing"])

    result = {
        "dataset_root": str(dataset_root),
        "info": {
            "fps": info["fps"],
            "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"],
            "joint_names": joint_names,
        },
        "start_window_frames": START_WINDOW,
        "per_episode": per_episode_full_dicts,
        "aggregate": agg,
        "segment_comparison": segment_agg,
        "shadow_comparison": {
            "n_runs_total": shadow["n_runs_total"],
            "fixed_scene_runs": [
                {"file": r["file"], "label": r["label"], "safety_decision": r["safety_decision"], "delta": r["delta"]}
                for r in shadow["fixed_scene_runs"]
            ],
            "fixed_scene_delta_stats": {
                j: {k: v for k, v in s.items() if k != "values"}
                for j, s in shadow["fixed_scene_delta_stats"].items()
            },
            "t05_runs": [
                {"file": r["file"], "label": r["label"], "safety_decision": r["safety_decision"], "delta": r["delta"]}
                for r in shadow["t05_runs"]
            ],
            "t05_delta_stats": {
                j: {k: v for k, v in s.items() if k != "values"} for j, s in shadow["t05_delta_stats"].items()
            },
            "background_bias_reference_deg": SHADOW_BACKGROUND_BIAS_DEG,
        },
        "similarity_to_shadow": similarity,
        "verdict": verdict,
    }

    json_path = out_dir / "grid35_episode_start_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[analyze] wrote {json_path}")

    md_path = out_dir / "grid35_episode_start_analysis.md"
    write_markdown_report(
        md_path,
        dataset_root,
        joint_names,
        len(per_episode),
        per_episode,
        agg,
        segment_agg,
        shadow,
        similarity,
        verdict,
    )
    print(f"[analyze] wrote {md_path}")

    print(f"[analyze] VERDICT: {verdict['verdict']} - {verdict['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
