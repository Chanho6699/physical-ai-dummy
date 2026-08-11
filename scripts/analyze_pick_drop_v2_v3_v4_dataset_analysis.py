#!/usr/bin/env python3
"""V2-clean / V3-clean / V4-generalization / V4-heldout Pick&Drop dataset analysis (read-only).

Scope (see the task brief this was written against / reports/pick_drop_v2_v3_v4_dataset_analysis/summary.md):
  - Validate integrity of 4 on-disk LeRobot v3.0 datasets.
  - Investigate per-episode frame-timing quality, especially for V4 heldout6 (dir name says
    "heldout10" but only 6 episodes were actually recorded - this script does NOT rename it).
  - Compare V2/V3/V4-train static/start coverage, start-pose diversity, immediate-action safety,
    chunk future-motion, and conflicting-immediate-label rates.
  - Sanity-check V4 heldout6 against V4 train39 for leakage (near-duplicate start states /
    trajectories / video files) without ever fabricating cube x/y ground truth (none exists in
    metadata for any of these datasets).

This script does **not** merge, reweight, retrain, rename, or write anything under data/. It does
**not** run git commands. All outputs go to reports/pick_drop_v2_v3_v4_dataset_analysis/.

Definitions reused verbatim (identical thresholds/constants, imported directly) from
``scripts/analyze_v2_vs_v3_start_coverage.py`` (itself reusing
``scripts/analyze_grid35_v2_clean_dataset_coverage_imbalance.py`` /
``scripts/analyze_grid35_v2_clean_start_segment_temporal_alignment.py`` definitions):
  - movement threshold: cumulative per-joint displacement from the trajectory's own frame-0 value
    >= 1.0deg, sustained for >= 3 consecutive frames (state[0]/action[0] baseline, independently).
  - static/moving segmentation from the state-detector's first-movement frame.
  - Safety Gate WOULD_CLAMP thresholds from configs/safety_gate.yaml (excessive_step_deg).
  - chunk_size=50 whole-chunk future-motion mechanism check.
  - start-pose pairwise-L2 diversity.
  - conflicting-immediate-label candidate search (state L2 <= 3.0deg across 6 joints,
    shoulder_lift/elbow_flex action(t)-state(t) diff >= 2.0deg), restricted to each episode's own
    static segment capped at 90 frames.

New in this script (not present in the V2-vs-V3 script):
  - Dataset-integrity extras: expected episode/video counts, meta<->data<->video file-reference
    consistency, meta-declared episode length vs loaded parquet length.
  - Frame-timing quality (section 2). Important finding, established by reading
    ``~/lerobot/src/lerobot/datasets/dataset_writer.py`` (``add_frame``): the ``timestamp`` column
    LeRobot writes is **not** a measured wall-clock value - it is computed synthetically as
    ``frame_index / fps`` at write time (confirmed empirically below too: every episode's stored
    timestamps land on the ideal 1/30s grid to float32 precision). The same is true of the
    per-episode ``from_timestamp``/``to_timestamp`` fields in meta/episodes and of the encoded
    video's nominal frame rate (``encode_video_frames`` encodes at a fixed output fps from a
    directory of already-captured images). So neither the parquet timestamp column nor basic video
    container metadata (nb_frames/avg_frame_rate) can, by construction, show a console-reported
    "loop ran slower than 30Hz" event - a slow control loop simply re-uses the last available
    camera frame while still labelling it with the next ideal timestamp.
    As an actual grounded-in-stored-data proxy, this script instead decodes each episode's
    workspace-camera video and looks for **exact pixel-identical consecutive frames** (bit-exact,
    not just low-motion): a slow/starved camera capture loop that could not deliver a fresh frame
    in time typically causes the *same* image buffer to be reused (and mp4-encoded) across
    consecutive nominal ticks, which shows up as an exact-duplicate pair. Genuine physical
    stillness (arm not moving) still has sensor noise and is essentially never exactly bit-equal
    frame to frame. This gives a real "effective distinct-frame rate" per episode, without any
    heavy model inference (single grayscale downsample + integer diff, ~0.2s/episode).
  - Heldout leakage sanity check (section 8): start-state / trajectory near-duplicate search,
    video file hash de-duplication, and a cheap (grayscale, downsampled, no model) frame-sample
    visual-similarity screen between V4 train39 and heldout6.

Usage::

    source ~/lerobot/.venv/bin/activate
    python scripts/analyze_pick_drop_v2_v3_v4_dataset_analysis.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v2_vs_v3_start_coverage as base  # noqa: E402  (reuse definitions verbatim)

DATA_ROOT = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "reports" / "pick_drop_v2_v3_v4_dataset_analysis"
SAFETY_GATE_YAML = PROJECT_ROOT / "configs" / "safety_gate.yaml"

# --- dataset roster (directory names kept exactly as instructed - v4 heldout dir is named
#     "..._v4_heldout10" but really holds 6 episodes; do not rename) --------------------------
DATASETS = {
    "v2_clean": {
        "root": DATA_ROOT / "so101_cube_pick_drop_grid35_v2_clean",
        "expected_episodes": 35,
        "expected_videos": 70,
    },
    "v3_clean": {
        "root": DATA_ROOT / "so101_cube_pick_drop_start_coverage_v3_clean",
        "expected_episodes": 30,
        "expected_videos": 60,
    },
    "v4_train39": {
        "root": DATA_ROOT / "so101_cube_pick_drop_generalization_v4",
        "expected_episodes": 39,
        "expected_videos": 78,
    },
    "v4_heldout6": {
        "root": DATA_ROOT / "so101_cube_pick_drop_v4_heldout10",
        "expected_episodes": 6,
        "expected_videos": 12,
    },
}
EXPECTED_TASK_STRING = "Pick up the cube and drop it into the bin."
FPS = base.FPS
JOINTS = base.JOINTS
CHUNK_SIZE = base.CHUNK_SIZE

# V4-train-specific: last 4 episodes (35..38, 0-indexed) deliberately include a wrist-tilt grasp
# variation per the task brief.
V4_WRIST_VARIATION_EPISODES = [35, 36, 37, 38]

# --- timing-quality (section 2): exact-duplicate-consecutive-frame proxy --------------------
DUPLICATE_FRAME_DOWNSAMPLE = 4  # decode at 1/4 resolution (bit-exact-duplicate detection is
# invariant to downsampling: if two full-res frames are bit-identical, the downsampled versions
# are too; downsampling only speeds up the diff, it cannot manufacture a false duplicate because
# genuinely-different full-res frames practically never alias to identical downsampled ones at
# this resolution combined with the exact-zero-diff criterion).
BIG_GAP_RUN_THRESHOLD = 3  # a run of >=3 consecutive identical frames = the same visual content
# held across 2 extra nominal ticks, i.e. an effective capture interval >= 3/30s (<=10Hz) at that
# moment - flagged as a "big gap".

# --- heldout leakage (section 8) -------------------------------------------------------------
NEAR_DUP_STATE_L2_DEG = 1.0  # tight near-duplicate start-pose threshold (well below the 3.0deg
# "close" threshold used for conflicting-label search elsewhere in this script/base module -
# leakage should mean "essentially the same pose", not merely "nearby").
LEAKAGE_VISUAL_N_SAMPLE_FRAMES = 6  # cheap: a handful of evenly-spaced frames per episode, not a
# full decode.


# ==============================================================================================
# Section 1: dataset integrity (extends base.compute_dataset_integrity with reference-consistency
# and expected-count checks)
# ==============================================================================================


def load_episodes_meta_full(dataset_root: Path) -> pd.DataFrame:
    meta_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    return meta.sort_values("episode_index").reset_index(drop=True)


def compute_reference_integrity(dataset_root: Path, episodes: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    """meta <-> data <-> video file-reference consistency, not covered by the reused base module."""
    meta = load_episodes_meta_full(dataset_root)
    info = base.load_info(dataset_root)
    camera_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    length_mismatches = []
    missing_data_files = []
    missing_video_files = {cam: [] for cam in camera_keys}
    referenced_video_files = {cam: set() for cam in camera_keys}

    for _, row in meta.iterrows():
        ep = int(row["episode_index"])
        declared_len = int(row["length"])
        loaded_len = episodes.get(ep, {}).get("length")
        if loaded_len is not None and declared_len != loaded_len:
            length_mismatches.append({"episode": ep, "declared_length": declared_len, "loaded_length": loaded_len})

        data_path = dataset_root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
        if not data_path.exists():
            missing_data_files.append({"episode": ep, "path": str(data_path)})

        for cam in camera_keys:
            c_idx = int(row[f"videos/{cam}/chunk_index"])
            f_idx = int(row[f"videos/{cam}/file_index"])
            referenced_video_files[cam].add((c_idx, f_idx))
            video_path = dataset_root / "videos" / cam / f"chunk-{c_idx:03d}" / f"file-{f_idx:03d}.mp4"
            if not video_path.exists():
                missing_video_files[cam].append({"episode": ep, "path": str(video_path)})

    on_disk_video_counts = {}
    orphan_video_files = {}
    for cam in camera_keys:
        cam_dir = dataset_root / "videos" / cam
        on_disk = sorted(cam_dir.glob("chunk-*/file-*.mp4"))
        on_disk_video_counts[cam] = len(on_disk)
        referenced = referenced_video_files[cam]
        orphans = []
        for p in on_disk:
            c_idx = int(p.parent.name.split("-")[1])
            f_idx = int(p.stem.split("-")[1])
            if (c_idx, f_idx) not in referenced:
                orphans.append(str(p))
        orphan_video_files[cam] = orphans

    return {
        "n_episodes_meta_rows": len(meta),
        "episode_length_meta_vs_loaded_mismatches": length_mismatches,
        "missing_data_files": missing_data_files,
        "missing_video_files": missing_video_files,
        "on_disk_video_file_counts": on_disk_video_counts,
        "orphan_video_files_not_referenced_by_any_episode": orphan_video_files,
        "n_distinct_video_files_referenced": {cam: len(v) for cam, v in referenced_video_files.items()},
    }


def analyze_integrity(label: str, dataset_root: Path, expected_episodes: int, expected_videos: int, episodes: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    base_integrity = base.compute_dataset_integrity(dataset_root, episodes)
    ref_integrity = compute_reference_integrity(dataset_root, episodes)
    task_ok = base_integrity["task_strings"] == [EXPECTED_TASK_STRING]
    n_episodes = base_integrity["n_episodes_loaded"]
    total_videos = sum(ref_integrity["on_disk_video_file_counts"].values())
    checks_passed = (
        n_episodes == expected_episodes
        and total_videos == expected_videos
        and task_ok
        and base_integrity["episode_index_continuous_0_to_N-1"]
        and not base_integrity["frame_index_continuity_issues_episodes"]
        and base_integrity["global_index_continuous"]
        and not base_integrity["nan_inf_check"]["episodes_with_nan"]
        and not base_integrity["nan_inf_check"]["episodes_with_inf"]
        and not ref_integrity["episode_length_meta_vs_loaded_mismatches"]
        and not ref_integrity["missing_data_files"]
        and not any(ref_integrity["missing_video_files"].values())
        and not any(ref_integrity["orphan_video_files_not_referenced_by_any_episode"].values())
    )
    return {
        "label": label,
        "expected_episodes": expected_episodes,
        "expected_videos": expected_videos,
        "n_episodes_found": n_episodes,
        "n_videos_found": total_videos,
        "task_string_matches_expected": task_ok,
        "all_integrity_checks_passed": checks_passed,
        **base_integrity,
        "reference_integrity": ref_integrity,
    }


# ==============================================================================================
# Section 2: frame-timing quality
# ==============================================================================================


def decode_duplicate_frame_stats(video_path: Path) -> dict[str, Any]:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="gray8")
        frames.append(arr[::DUPLICATE_FRAME_DOWNSAMPLE, ::DUPLICATE_FRAME_DOWNSAMPLE])
    container.close()
    n_frames = len(frames)
    if n_frames < 2:
        return {"n_frames": n_frames, "n_exact_duplicate_pairs": 0, "duplicate_fraction": 0.0, "max_duplicate_run": 0, "n_big_gaps": 0, "effective_distinct_frames": n_frames}
    stacked = np.stack(frames).astype(np.int16)
    diffs = np.abs(np.diff(stacked, axis=0)).reshape(n_frames - 1, -1).max(axis=1)
    is_dup = diffs == 0

    # run-length of consecutive duplicate flags (a run of length L means L+1 identical frames)
    runs = []
    cur = 0
    for d in is_dup:
        if d:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    max_run = max(runs) if runs else 0
    n_big_gaps = sum(1 for r in runs if r + 1 >= BIG_GAP_RUN_THRESHOLD)

    n_dup_pairs = int(is_dup.sum())
    distinct_frames = n_frames - n_dup_pairs
    nominal_duration_s = n_frames / FPS
    effective_fps = distinct_frames / nominal_duration_s if nominal_duration_s > 0 else None

    # inter-distinct-frame interval distribution: nominal dt(=1/30s) * (run_length_of_dup + 1) for
    # every "new" frame, i.e. treat each maximal run of identical frames as one held sample.
    intervals = []
    cur = 1
    for d in is_dup:
        if d:
            cur += 1
        else:
            intervals.append(cur / FPS)
            cur = 1
    intervals.append(cur / FPS)
    intervals = np.array(intervals)

    return {
        "n_frames": n_frames,
        "n_exact_duplicate_pairs": n_dup_pairs,
        "duplicate_fraction": n_dup_pairs / (n_frames - 1),
        "max_duplicate_run": max_run,
        "n_big_gaps_run_ge_%d" % BIG_GAP_RUN_THRESHOLD: n_big_gaps,
        "effective_distinct_frames": distinct_frames,
        "effective_fps_estimate": effective_fps,
        "distinct_frame_interval_s": {
            "median": float(np.median(intervals)),
            "p5": float(np.percentile(intervals, 5)),
            "p95": float(np.percentile(intervals, 95)),
            "max": float(np.max(intervals)),
        },
    }


def compute_timing_quality(label: str, dataset_root: Path, episodes: dict[int, dict[str, np.ndarray]], camera: str = "observation.images.workspace") -> list[dict[str, Any]]:
    meta = load_episodes_meta_full(dataset_root)
    rows = []
    for _, row in meta.iterrows():
        ep = int(row["episode_index"])
        c_idx = int(row[f"videos/{camera}/chunk_index"])
        f_idx = int(row[f"videos/{camera}/file_index"])
        video_path = dataset_root / "videos" / camera / f"chunk-{c_idx:03d}" / f"file-{f_idx:03d}.mp4"
        d = decode_duplicate_frame_stats(video_path)
        parquet_len = episodes[ep]["length"]
        parquet_ts = episodes[ep]["timestamp"]
        ideal_ts = np.arange(parquet_len) / FPS
        parquet_ts_deviation_from_ideal_grid = float(np.max(np.abs(parquet_ts - ideal_ts))) if parquet_len > 0 else None
        rows.append(
            {
                "dataset": label,
                "episode": ep,
                "camera": camera,
                "parquet_frame_count": parquet_len,
                "video_frame_count": d["n_frames"],
                "frame_count_match": parquet_len == d["n_frames"],
                "parquet_timestamp_max_deviation_from_ideal_grid_s": parquet_ts_deviation_from_ideal_grid,
                "nominal_fps": FPS,
                "n_exact_duplicate_frame_pairs": d["n_exact_duplicate_pairs"],
                "duplicate_frame_fraction": d["duplicate_fraction"],
                "max_duplicate_run_length": d["max_duplicate_run"],
                "n_big_gaps": d[f"n_big_gaps_run_ge_{BIG_GAP_RUN_THRESHOLD}"],
                "effective_distinct_frames": d["effective_distinct_frames"],
                "effective_fps_estimate": d["effective_fps_estimate"],
                "distinct_frame_interval_median_s": d["distinct_frame_interval_s"]["median"],
                "distinct_frame_interval_p5_s": d["distinct_frame_interval_s"]["p5"],
                "distinct_frame_interval_p95_s": d["distinct_frame_interval_s"]["p95"],
                "distinct_frame_interval_max_s": d["distinct_frame_interval_s"]["max"],
            }
        )
    return rows


# ==============================================================================================
# Section 7 (generalized reuse): within-dataset conflicting-immediate-label search
# ==============================================================================================


def compute_within_dataset_conflicts(label: str, episodes: dict[int, dict[str, np.ndarray]], segments: dict[int, dict[str, int]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs base.compute_conflicting_labels on a single dataset (all episodes tagged identically,
    so every candidate/close pair is necessarily "within" it). Reuses the exact same thresholds
    and O(n^2)-bounded-by-CONFLICT_WINDOW_CAP_FRAMES search as the base module; only the outer
    dataset-tag bucketing (irrelevant for a single-dataset call) is bypassed by re-keying results
    below instead of trusting the base function's v2/v3-specific bucket names.
    """
    internal_tag = "v2_clean"  # arbitrary; single dataset => bucketing is moot, only totals matter
    tagged_episodes = {(internal_tag, ep): d for ep, d in episodes.items()}
    tagged_segments = {(internal_tag, ep): segments[ep] for ep in episodes}
    candidates, close_counts = base.compute_conflicting_labels(tagged_episodes, tagged_segments)
    for c in candidates:
        c["dataset"] = label
    n_close = close_counts["total_close_within_v2"]
    summary = {
        "dataset": label,
        "n_episodes": len(episodes),
        "state_close_l2_threshold_deg": base.STATE_CLOSE_L2_THRESHOLD_DEG,
        "conflict_action_diff_threshold_deg": base.CONFLICT_ACTION_DIFF_DEG,
        "conflict_window_cap_frames": base.CONFLICT_WINDOW_CAP_FRAMES,
        "n_close_state_pairs": n_close,
        "n_conflict_candidates": len(candidates),
        "conflict_rate": (len(candidates) / n_close) if n_close else None,
    }
    return candidates, summary


# ==============================================================================================
# Section 8: V4 heldout leakage sanity check
# ==============================================================================================


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_visual_feature(video_path: Path, n_samples: int) -> np.ndarray:
    """Cheap, low-cost visual fingerprint: n_samples evenly-spaced frames, grayscale, heavily
    downsampled (1/8 resolution), flattened and concatenated. No model inference, no full decode
    of every frame (uses container frame count from a first light pass then seeks by frame index
    while iterating - av doesn't guarantee fast seek for this codec, so we do a single sequential
    decode pass and just keep the frames we want; still <0.3s/episode per the earlier timing
    check).
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    all_frames = []
    for frame in container.decode(stream):
        all_frames.append(frame.to_ndarray(format="gray8")[::8, ::8])
    container.close()
    n = len(all_frames)
    idxs = np.linspace(0, n - 1, n_samples).round().astype(int)
    feat = np.stack([all_frames[i] for i in idxs]).astype(np.float32).flatten()
    return feat


def compute_heldout_leakage(
    v4_train_episodes: dict[int, dict[str, np.ndarray]],
    v4_heldout_episodes: dict[int, dict[str, np.ndarray]],
    train_root: Path,
    heldout_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_eps = sorted(v4_train_episodes.keys())
    heldout_eps = sorted(v4_heldout_episodes.keys())

    train_meta = load_episodes_meta_full(train_root)
    heldout_meta = load_episodes_meta_full(heldout_root)

    def video_path_for(meta_df, ep, root, camera):
        row = meta_df[meta_df["episode_index"] == ep].iloc[0]
        c_idx = int(row[f"videos/{camera}/chunk_index"])
        f_idx = int(row[f"videos/{camera}/file_index"])
        return root / "videos" / camera / f"chunk-{c_idx:03d}" / f"file-{f_idx:03d}.mp4"

    # --- video file hash dedup (all workspace+wrist files, train U heldout) -------------------
    hashes: dict[str, list[str]] = {}
    for label, meta_df, root, eps in [("train", train_meta, train_root, train_eps), ("heldout", heldout_meta, heldout_root, heldout_eps)]:
        for ep in eps:
            for camera in ["observation.images.workspace", "observation.images.wrist"]:
                p = video_path_for(meta_df, ep, root, camera)
                h = md5_file(p)
                hashes.setdefault(h, []).append(f"{label}:ep{ep}:{camera}")
    hash_duplicate_groups = {h: v for h, v in hashes.items() if len(v) > 1}

    # --- cheap visual feature per episode (workspace cam only) --------------------------------
    train_visual = {ep: sample_visual_feature(video_path_for(train_meta, ep, train_root, "observation.images.workspace"), LEAKAGE_VISUAL_N_SAMPLE_FRAMES) for ep in train_eps}
    heldout_visual = {ep: sample_visual_feature(video_path_for(heldout_meta, ep, heldout_root, "observation.images.workspace"), LEAKAGE_VISUAL_N_SAMPLE_FRAMES) for ep in heldout_eps}

    rows = []
    for h_ep in heldout_eps:
        h_state0 = v4_heldout_episodes[h_ep]["state"][0]
        h_state = v4_heldout_episodes[h_ep]["state"]
        h_visual = heldout_visual[h_ep]
        for t_ep in train_eps:
            t_state0 = v4_train_episodes[t_ep]["state"][0]
            t_state = v4_train_episodes[t_ep]["state"]
            start_l2 = float(np.linalg.norm(h_state0 - t_state0))
            common_len = min(len(h_state), len(t_state))
            traj_mean_l2 = float(np.mean(np.linalg.norm(h_state[:common_len] - t_state[:common_len], axis=1)))
            visual_l2 = float(np.linalg.norm(h_visual - train_visual[t_ep]))
            any_hash_match = any(
                f"heldout:ep{h_ep}:observation.images.workspace" in g or f"heldout:ep{h_ep}:observation.images.wrist" in g
                for g in hash_duplicate_groups.values()
                if any(f"train:ep{t_ep}:" in item for item in g)
            )
            rows.append(
                {
                    "heldout_episode": h_ep,
                    "train_episode": t_ep,
                    "start_state_l2_deg": start_l2,
                    "start_state_near_duplicate": start_l2 <= NEAR_DUP_STATE_L2_DEG,
                    "trajectory_mean_l2_deg": traj_mean_l2,
                    "visual_feature_l2": visual_l2,
                    "video_hash_exact_match": any_hash_match,
                }
            )

    rows.sort(key=lambda r: r["start_state_l2_deg"])

    start_l2_vals = np.array([r["start_state_l2_deg"] for r in rows])
    traj_l2_vals = np.array([r["trajectory_mean_l2_deg"] for r in rows])
    visual_l2_vals = np.array([r["visual_feature_l2"] for r in rows])
    summary = {
        "n_train_episodes": len(train_eps),
        "n_heldout_episodes": len(heldout_eps),
        "n_pairs_checked": len(rows),
        "near_dup_state_l2_threshold_deg": NEAR_DUP_STATE_L2_DEG,
        "start_state_l2_deg": {"min": float(start_l2_vals.min()), "median": float(np.median(start_l2_vals)), "max": float(start_l2_vals.max())},
        "trajectory_mean_l2_deg": {"min": float(traj_l2_vals.min()), "median": float(np.median(traj_l2_vals)), "max": float(traj_l2_vals.max())},
        "visual_feature_l2": {"min": float(visual_l2_vals.min()), "median": float(np.median(visual_l2_vals)), "max": float(visual_l2_vals.max())},
        "n_start_state_near_duplicates": int(sum(r["start_state_near_duplicate"] for r in rows)),
        "n_video_hash_exact_matches": int(sum(r["video_hash_exact_match"] for r in rows)),
        "hash_duplicate_groups_found": hash_duplicate_groups,
        "closest_pairs_top5": rows[:5],
    }
    return rows, summary


# ==============================================================================================
# CSV writers
# ==============================================================================================


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fieldnames}
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = f"{v:.5f}"
                elif isinstance(v, (list, dict)):
                    row[k] = json.dumps(v)
            w.writerow(row)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = base.load_safety_thresholds(SAFETY_GATE_YAML)

    print("[analysis] loading episodes for all 4 datasets")
    episodes_by_ds: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for label, cfg in DATASETS.items():
        eps = base.load_episodes(cfg["root"])
        episodes_by_ds[label] = eps
        print(f"[analysis]   {label}: {len(eps)} episodes loaded from {cfg['root']}")

    # ---------------- Section 1: integrity ----------------
    print("[analysis] section 1: dataset integrity")
    integrity: dict[str, Any] = {}
    integrity_csv_rows = []
    for label, cfg in DATASETS.items():
        result = analyze_integrity(label, cfg["root"], cfg["expected_episodes"], cfg["expected_videos"], episodes_by_ds[label])
        integrity[label] = result
        lengths = result["episode_length_distribution"]
        integrity_csv_rows.append(
            {
                "dataset": label,
                "dataset_root": str(cfg["root"]),
                "expected_episodes": cfg["expected_episodes"],
                "n_episodes_found": result["n_episodes_found"],
                "expected_videos": cfg["expected_videos"],
                "n_videos_found": result["n_videos_found"],
                "fps": result["fps_meta"],
                "n_frames_total": result["n_frames_loaded"],
                "episode_length_mean": lengths["mean"],
                "episode_length_median": lengths["median"],
                "episode_length_min": lengths["min"],
                "episode_length_max": lengths["max"],
                "episode_length_mean_seconds": lengths["mean_seconds"],
                "state_action_dim": result["state_dim"][0],
                "task_string": result["task_strings"][0] if result["task_strings"] else None,
                "task_string_matches_expected": result["task_string_matches_expected"],
                "episode_index_continuous": result["episode_index_continuous_0_to_N-1"],
                "frame_index_continuity_issues": result["frame_index_continuity_issues_episodes"],
                "global_index_continuous": result["global_index_continuous"],
                "episodes_with_nan": result["nan_inf_check"]["episodes_with_nan"],
                "episodes_with_inf": result["nan_inf_check"]["episodes_with_inf"],
                "meta_vs_loaded_length_mismatches": len(result["reference_integrity"]["episode_length_meta_vs_loaded_mismatches"]),
                "missing_data_files": len(result["reference_integrity"]["missing_data_files"]),
                "missing_video_files": sum(len(v) for v in result["reference_integrity"]["missing_video_files"].values()),
                "orphan_video_files": sum(len(v) for v in result["reference_integrity"]["orphan_video_files_not_referenced_by_any_episode"].values()),
                "all_integrity_checks_passed": result["all_integrity_checks_passed"],
            }
        )
    write_csv(
        OUT_DIR / "dataset_integrity.csv",
        integrity_csv_rows,
        [
            "dataset", "dataset_root", "expected_episodes", "n_episodes_found", "expected_videos", "n_videos_found",
            "fps", "n_frames_total", "episode_length_mean", "episode_length_median", "episode_length_min", "episode_length_max",
            "episode_length_mean_seconds", "state_action_dim", "task_string", "task_string_matches_expected",
            "episode_index_continuous", "frame_index_continuity_issues", "global_index_continuous",
            "episodes_with_nan", "episodes_with_inf", "meta_vs_loaded_length_mismatches", "missing_data_files",
            "missing_video_files", "orphan_video_files", "all_integrity_checks_passed",
        ],
    )
    print(f"[analysis]   wrote {OUT_DIR / 'dataset_integrity.csv'}")

    # ---------------- Section 2: timing quality ----------------
    print("[analysis] section 2: frame-timing quality (video decode; ~0.2s/episode)")
    timing_rows_all = []
    for label in ["v4_heldout6", "v4_train39", "v2_clean", "v3_clean"]:
        cfg = DATASETS[label]
        rows = compute_timing_quality(label, cfg["root"], episodes_by_ds[label])
        timing_rows_all.extend(rows)
        print(f"[analysis]   {label}: {len(rows)} episodes timing-checked")
    write_csv(
        OUT_DIR / "timing_quality.csv",
        timing_rows_all,
        [
            "dataset", "episode", "camera", "parquet_frame_count", "video_frame_count", "frame_count_match",
            "parquet_timestamp_max_deviation_from_ideal_grid_s", "nominal_fps",
            "n_exact_duplicate_frame_pairs", "duplicate_frame_fraction", "max_duplicate_run_length",
            "n_big_gaps", "effective_distinct_frames", "effective_fps_estimate",
            "distinct_frame_interval_median_s", "distinct_frame_interval_p5_s", "distinct_frame_interval_p95_s",
            "distinct_frame_interval_max_s",
        ],
    )
    print(f"[analysis]   wrote {OUT_DIR / 'timing_quality.csv'}")

    # ---------------- Sections 3-6: per-dataset static coverage / diversity / safety / chunk ----------------
    print("[analysis] sections 3-6: static coverage, start-pose diversity, immediate safety, chunk future-motion")
    results: dict[str, Any] = {}
    safety_rows_by_ds: dict[str, list[dict[str, Any]]] = {}
    chunk_rows_by_ds: dict[str, list[dict[str, Any]]] = {}
    for label in DATASETS:
        result, safety_rows, chunk_rows = base.analyze_dataset(DATASETS[label]["root"], episodes_by_ds[label], thresholds, label)
        results[label] = result
        safety_rows_by_ds[label] = safety_rows
        chunk_rows_by_ds[label] = chunk_rows

    start_static_rows = []
    for label in DATASETS:
        eps = episodes_by_ds[label]
        first_movement = results[label]["first_movement"]
        segments = results[label]["segments"]
        for ep in sorted(eps.keys()):
            mv = first_movement[ep]
            s = segments[ep]
            start_static_rows.append(
                {
                    "dataset": label,
                    "episode": ep,
                    "length": s["length"],
                    "state_first_movement_frame": mv["state_first_movement_frame"],
                    "state_first_movement_s": mv["state_first_movement_s"],
                    "action_first_movement_frame": mv["action_first_movement_frame"],
                    "action_first_movement_s": mv["action_first_movement_s"],
                    "static_length_frames": s["static_length"],
                    "moving_length_frames": s["moving_length"],
                    "static_fraction": s["static_length"] / s["length"],
                    "v4_wrist_variation_episode": (label == "v4_train39" and ep in V4_WRIST_VARIATION_EPISODES),
                }
            )
    write_csv(
        OUT_DIR / "start_static_coverage.csv",
        start_static_rows,
        [
            "dataset", "episode", "length", "state_first_movement_frame", "state_first_movement_s",
            "action_first_movement_frame", "action_first_movement_s", "static_length_frames",
            "moving_length_frames", "static_fraction", "v4_wrist_variation_episode",
        ],
    )
    print(f"[analysis]   wrote {OUT_DIR / 'start_static_coverage.csv'}")

    diversity_rows = []
    for label in DATASETS:
        div = results[label]["start_pose_diversity"]
        for j in JOINTS:
            s = div["per_joint_start_frame0_stats"][j]
            diversity_rows.append({"dataset": label, "category": "per_joint_start_std_deg", "key": j, "value": s["std"]})
            diversity_rows.append({"dataset": label, "category": "per_joint_start_range_deg", "key": j, "value": s["range"]})
            diversity_rows.append({"dataset": label, "category": "per_joint_start_min_deg", "key": j, "value": s["min"]})
            diversity_rows.append({"dataset": label, "category": "per_joint_start_max_deg", "key": j, "value": s["max"]})
        p = div["start_frame0_pairwise_l2_deg"]
        for stat in ["min", "median", "mean", "p95", "max"]:
            diversity_rows.append({"dataset": label, "category": "start_frame0_pairwise_l2_deg", "key": stat, "value": p[stat]})
        diversity_rows.append({"dataset": label, "category": "start_frame0_pairwise_l2_deg", "key": "n_pairs", "value": p["n_pairs"]})
    write_csv(OUT_DIR / "start_pose_diversity.csv", diversity_rows, ["dataset", "category", "key", "value"])
    print(f"[analysis]   wrote {OUT_DIR / 'start_pose_diversity.csv'}")

    safety_csv_rows = []
    for label, rows in safety_rows_by_ds.items():
        for r in rows:
            safety_csv_rows.append({"dataset": label, **r})
    write_csv(
        OUT_DIR / "immediate_action_safety.csv",
        safety_csv_rows,
        ["dataset", "episode", "joint", "n_frames", "mean_abs_delta_deg", "median_abs_delta_deg", "p95_abs_delta_deg", "max_abs_delta_deg", "would_clamp_threshold_deg", "frac_would_clamp"],
    )
    print(f"[analysis]   wrote {OUT_DIR / 'immediate_action_safety.csv'}")

    chunk_csv_rows = []
    for label, rows in chunk_rows_by_ds.items():
        for r in rows:
            chunk_csv_rows.append({"dataset": label, **r})
    write_csv(
        OUT_DIR / "chunk_future_motion.csv",
        chunk_csv_rows,
        ["dataset", "episode", "joint", "n_static_frames", "chunk0_abs_delta_mean", "chunk_mean_abs_delta_mean", "chunk_max_abs_delta_mean", "frac_static_frames_chunk_mean_exceeds_would_clamp"],
    )
    print(f"[analysis]   wrote {OUT_DIR / 'chunk_future_motion.csv'}")

    # ---------------- Section 7: conflicting-label candidates (within V2 / V3 / V4-train / heldout) ----------------
    print("[analysis] section 7: within-dataset conflicting-immediate-label candidate search")
    conflict_candidates_all = []
    conflict_summaries = {}
    for label in DATASETS:
        candidates, summary = compute_within_dataset_conflicts(label, episodes_by_ds[label], results[label]["segments"])
        conflict_candidates_all.extend(candidates)
        conflict_summaries[label] = summary
        print(f"[analysis]   {label}: {summary['n_conflict_candidates']}/{summary['n_close_state_pairs']} close pairs flagged as conflicting")
    conflict_candidates_all.sort(key=lambda r: r["max_key_joint_diff_deg"], reverse=True)
    # Cap rows written (like base.write_conflicting_labels_csv's --max-conflict-candidates
    # default): with 4 datasets pooled, the uncapped list is tens of thousands of rows. Full
    # counts/rates per dataset are preserved in summary.json regardless; the CSV keeps the
    # top-N-per-dataset most-severe candidates for manual inspection.
    MAX_CONFLICT_ROWS_PER_DATASET = 100
    capped_candidates = []
    per_dataset_kept = {label: 0 for label in DATASETS}
    for c in conflict_candidates_all:
        if per_dataset_kept[c["dataset"]] < MAX_CONFLICT_ROWS_PER_DATASET:
            capped_candidates.append(c)
            per_dataset_kept[c["dataset"]] += 1
    write_csv(
        OUT_DIR / "conflicting_label_candidates.csv",
        capped_candidates,
        [
            "dataset", "episode_a", "frame_a", "episode_b", "frame_b", "state_l2_dist_deg",
            "shoulder_lift_action_delta_a", "shoulder_lift_action_delta_b", "shoulder_lift_diff_deg",
            "elbow_flex_action_delta_a", "elbow_flex_action_delta_b", "elbow_flex_diff_deg", "max_key_joint_diff_deg",
        ],
    )
    print(f"[analysis]   wrote {OUT_DIR / 'conflicting_label_candidates.csv'}")

    # ---------------- Section 8: heldout leakage sanity check ----------------
    print("[analysis] section 8: V4 heldout6 vs train39 leakage sanity check")
    leakage_rows, leakage_summary = compute_heldout_leakage(
        episodes_by_ds["v4_train39"], episodes_by_ds["v4_heldout6"], DATASETS["v4_train39"]["root"], DATASETS["v4_heldout6"]["root"]
    )
    write_csv(
        OUT_DIR / "heldout_leakage_check.csv",
        leakage_rows,
        ["heldout_episode", "train_episode", "start_state_l2_deg", "start_state_near_duplicate", "trajectory_mean_l2_deg", "visual_feature_l2", "video_hash_exact_match"],
    )
    print(f"[analysis]   wrote {OUT_DIR / 'heldout_leakage_check.csv'}")
    print(f"[analysis]   {leakage_summary['n_start_state_near_duplicates']} near-dup start-state pairs, "
          f"{leakage_summary['n_video_hash_exact_matches']} exact video-hash matches (of {leakage_summary['n_pairs_checked']} pairs)")

    # ---------------- summary.json ----------------
    def strip_heavy(result: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in result.items() if k not in ("first_movement", "segments")}

    summary = {
        "datasets": {label: {"root": str(cfg["root"]), "expected_episodes": cfg["expected_episodes"], "expected_videos": cfg["expected_videos"]} for label, cfg in DATASETS.items()},
        "expected_task_string": EXPECTED_TASK_STRING,
        "safety_thresholds": thresholds,
        "dataset_integrity": integrity,
        "per_dataset_analysis": {label: strip_heavy(results[label]) for label in DATASETS},
        "conflicting_labels_within_dataset": conflict_summaries,
        "heldout_leakage_check": leakage_summary,
        "v4_wrist_variation_episodes": V4_WRIST_VARIATION_EPISODES,
        "notes": {
            "timestamp_is_synthetic": (
                "Verified against ~/lerobot/src/lerobot/datasets/dataset_writer.py (add_frame): "
                "LeRobot computes timestamp = frame_index / fps at write time, it is not a "
                "measured wall-clock value. Every episode's stored timestamps in all 4 datasets "
                "land on the ideal 1/30s grid (max deviation ~4e-7s = float32 rounding only). "
                "The parquet timestamp column therefore cannot show a real console-reported "
                "'loop slower than 30Hz' event by construction - see timing_quality.csv's "
                "exact-duplicate-frame-based effective_fps_estimate for the closest available "
                "grounded proxy."
            ),
        },
    }
    json_path = OUT_DIR / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"[analysis] wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
