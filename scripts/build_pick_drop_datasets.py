#!/usr/bin/env python3
"""Build canonical "Pick & Drop" datasets from V2-clean and V3 start-coverage, plus a combined
65-episode dataset - using LeRobot v0.6.2's own dataset-editing API, not manual parquet edits and
not a naive folder ``mv``.

Background
----------
``data/so101_cube_xy_grid35_v2_clean`` (35 episodes) and ``data/so101_cube_xy_start_coverage_v3``
(30 episodes) both currently store the task string
``"Pick up the cube and place it in the target area."``, but the actual recorded behaviour in both
is: grasp cube -> lift -> move toward bin -> open gripper ~10cm above the bin -> drop. This is
being split out as a distinct "Pick & Drop" canonical task (separate from a future, not-yet-built,
gentler "Pick & Place" and a "Handover" task) via the new task string
``"Pick up the cube and drop it into the bin."``.

Why not a plain ``mv``/rename
------------------------------
LeRobot v0.6.2 does not store ``repo_id`` inside any metadata file (checked: absent from
``meta/info.json``) - ``repo_id`` is purely a client-side identifier passed to ``LeRobotDataset()``,
so a bare directory rename would not, by itself, corrupt anything LeRobot reads from disk. The
actual risk this script avoids is different: (1) hand-editing ``task_index``/``tasks`` columns
directly in parquet without going through ``lerobot.datasets.dataset_tools.modify_tasks`` risks
missing one of the four places the task string is denormalized (``meta/tasks.parquet``,
every ``data/**/*.parquet`` row's ``task_index``, every ``meta/episodes/**/*.parquet`` row's
``tasks`` list, and ``meta/info.json``'s ``total_tasks``); (2) merging two datasets by hand risks
getting episode_index/frame index/chunk-file bookkeeping wrong. Both are exactly what LeRobot's own
``lerobot.datasets.dataset_tools`` module (``modify_tasks``, ``merge_datasets`` ->
``lerobot.datasets.aggregate.aggregate_datasets``) is built to do correctly - this script calls
those functions directly (the same functions the ``lerobot-edit-dataset`` CLI wraps) instead of
reimplementing them.

Pipeline
--------
1. Byte-copy (``shutil.copytree``) each source dataset to a new directory. The two original
   dataset directories are never opened in write mode by this script.
2. On each **copy**, call ``lerobot.datasets.modify_tasks(dataset, new_task=...)`` (the official,
   in-place task-editing function) to set the single new task string. Since it operates in-place,
   pointing it at the copy (never the original) is what keeps the source read-only.
3. Verify every file in each source directory is still byte-identical to its pre-run checksum
   (proves the copy step never mutated the source), and that every non-task column in the copies'
   data/episodes parquet files is still exactly equal to the source (proves ``modify_tasks`` only
   touched what its own docstring says it touches).
4. Assert the two retasked datasets have an identical feature schema
   (``lerobot.datasets.feature_utils.features_equal_for_merge``, the same check
   ``aggregate_datasets`` uses internally) before merging.
5. Call ``lerobot.datasets.merge_datasets`` (official merge/aggregate path) to build the combined
   65-episode dataset.
6. Validate the combined dataset's integrity (episode/frame counts, index continuity, feature
   schema, NaN/Inf, on-disk file references).
7. Re-run the static/low-motion coverage + start-pose-diversity + immediate-WOULD_CLAMP metrics
   from ``scripts/analyze_v2_vs_v3_start_coverage.py`` directly against the new combined dataset,
   as a sanity check against ``reports/grid35_v2_vs_start_coverage_v3/summary.json``'s
   ``combined_v2_plus_v3`` numbers (computed by literal in-memory concatenation there; this
   confirms the on-disk merge didn't change the underlying state/action data).
8. Write ``reports/pick_drop_combined65_dataset_build/summary.{md,json}``.

This script does **not** train, does **not** create checkpoints, does **not** modify or delete the
original datasets, does **not** reinterpret/edit any action values, does **not** invent cube x/y
metadata, does **not** build Pick & Place / Handover datasets, and does **not** run git commands.

Usage::

    source ~/lerobot/.venv/bin/activate
    python scripts/build_pick_drop_datasets.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[build] %(message)s")
log = logging.getLogger("build_pick_drop_datasets")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v2_vs_v3_start_coverage as cov  # noqa: E402  (reuse V2-vs-V3 coverage definitions)

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, merge_datasets, modify_tasks  # noqa: E402
from lerobot.datasets.feature_utils import features_equal_for_merge  # noqa: E402

SRC_V2 = PROJECT_ROOT / "data" / "so101_cube_xy_grid35_v2_clean"
SRC_V3 = PROJECT_ROOT / "data" / "so101_cube_xy_start_coverage_v3"
DST_V2 = PROJECT_ROOT / "data" / "so101_cube_pick_drop_grid35_v2_clean"
DST_V3 = PROJECT_ROOT / "data" / "so101_cube_pick_drop_start_coverage_v3_clean"
DST_COMBINED = PROJECT_ROOT / "data" / "so101_cube_pick_drop_combined65_v1"
REPORT_DIR = PROJECT_ROOT / "reports" / "pick_drop_combined65_dataset_build"
PRIOR_COVERAGE_SUMMARY = PROJECT_ROOT / "reports" / "grid35_v2_vs_start_coverage_v3" / "summary.json"

OLD_TASK = "Pick up the cube and place it in the target area."
NEW_TASK = "Pick up the cube and drop it into the bin."

EXPECTED = {
    "v2": {"episodes": 35, "frames": 11505},
    "v3": {"episodes": 30, "frames": 9822},
}
JOINTS = cov.JOINTS


# --------------------------------------------------------------------------
# 0. Checksums (prove the originals are never mutated)
# --------------------------------------------------------------------------


def snapshot_dir(root: Path) -> dict[str, tuple[int, str]]:
    """relative_path -> (size_bytes, sha256) for every file under root."""
    snap: dict[str, tuple[int, str]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            snap[str(p.relative_to(root))] = (len(data), hashlib.sha256(data).hexdigest())
    return snap


def diff_snapshots(before: dict[str, tuple[int, str]], after: dict[str, tuple[int, str]]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(k for k in (set(before) & set(after)) if before[k] != after[k])
    return {
        "unchanged": len(before) == len(after) and not missing and not added and not changed,
        "n_files_before": len(before),
        "n_files_after": len(after),
        "missing_files": missing,
        "added_files": added,
        "changed_files": changed,
    }


# --------------------------------------------------------------------------
# 1. Copy + retask
# --------------------------------------------------------------------------


def copy_dataset(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        if not force:
            raise FileExistsError(f"{dst} already exists - pass force=True to overwrite (source is never touched)")
        log.info(f"removing pre-existing {dst} (force=True)")
        shutil.rmtree(dst)
    log.info(f"copying {src} -> {dst}")
    shutil.copytree(src, dst)


def retask_dataset(root: Path, repo_id: str, new_task: str) -> dict[str, Any]:
    dataset = LeRobotDataset(repo_id, root=root)
    before_tasks = list(dataset.meta.tasks.index)
    modified = modify_tasks(dataset, new_task=new_task)
    after_tasks = list(modified.meta.tasks.index)
    return {"repo_id": repo_id, "root": str(root), "tasks_before": before_tasks, "tasks_after": after_tasks}


# --------------------------------------------------------------------------
# 2. Per-episode array-equality check (everything except task columns)
# --------------------------------------------------------------------------


def _deep_equal(a: Any, b: Any) -> bool:
    """Structural equality that tolerates ragged/nested numpy object arrays (e.g. per-episode
    image-stat cells shaped like array([array([array([0.])]), ...])), which np.array_equal chokes
    on when comparing whole columns at once."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        if a_arr.shape != b_arr.shape:
            return False
        if a_arr.dtype == object or b_arr.dtype == object:
            return all(_deep_equal(x, y) for x, y in zip(a_arr.flatten(), b_arr.flatten(), strict=True))
        if np.issubdtype(a_arr.dtype, np.floating):
            return bool(np.array_equal(a_arr, b_arr, equal_nan=True))
        return bool(np.array_equal(a_arr, b_arr))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        a_list, b_list = list(a), list(b)
        if len(a_list) != len(b_list):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a_list, b_list, strict=True))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return bool(a == b)


def compare_non_task_columns(orig_root: Path, new_root: Path) -> dict[str, Any]:
    """Compare every data/*.parquet and meta/episodes/*.parquet file pair, column by column,
    excluding the columns modify_tasks is documented to touch (task_index / tasks)."""
    mismatches: list[str] = []
    n_data_files_compared = 0
    n_episode_meta_files_compared = 0

    orig_data_files = sorted((orig_root / "data").glob("chunk-*/file-*.parquet"))
    for orig_path in orig_data_files:
        rel = orig_path.relative_to(orig_root)
        new_path = new_root / rel
        if not new_path.exists():
            mismatches.append(f"data file missing in new dataset: {rel}")
            continue
        df_orig = pd.read_parquet(orig_path)
        df_new = pd.read_parquet(new_path)
        cols_to_check = [c for c in df_orig.columns if c != "task_index"]
        if list(df_orig.columns) != list(df_new.columns):
            mismatches.append(f"{rel}: column set/order differs ({list(df_orig.columns)} vs {list(df_new.columns)})")
            continue
        for c in cols_to_check:
            a = np.stack(df_orig[c].to_numpy()) if df_orig[c].dtype == object else df_orig[c].to_numpy()
            b = np.stack(df_new[c].to_numpy()) if df_new[c].dtype == object else df_new[c].to_numpy()
            if not np.array_equal(a, b):
                mismatches.append(f"{rel}: column '{c}' differs")
        n_data_files_compared += 1

    orig_ep_files = sorted((orig_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for orig_path in orig_ep_files:
        rel = orig_path.relative_to(orig_root)
        new_path = new_root / rel
        if not new_path.exists():
            mismatches.append(f"episode meta file missing in new dataset: {rel}")
            continue
        df_orig = pd.read_parquet(orig_path).sort_values("episode_index").reset_index(drop=True)
        df_new = pd.read_parquet(new_path).sort_values("episode_index").reset_index(drop=True)
        cols_to_check = [c for c in df_orig.columns if c != "tasks"]
        for c in cols_to_check:
            col_a, col_b = df_orig[c].to_numpy(), df_new[c].to_numpy()
            if len(col_a) != len(col_b) or not all(
                _deep_equal(x, y) for x, y in zip(col_a, col_b, strict=True)
            ):
                mismatches.append(f"{rel}: episode-meta column '{c}' differs")
        n_episode_meta_files_compared += 1

    return {
        "n_data_files_compared": n_data_files_compared,
        "n_episode_meta_files_compared": n_episode_meta_files_compared,
        "mismatches": mismatches,
        "all_non_task_columns_identical": len(mismatches) == 0,
    }


def compare_other_files_byte_identical(orig_root: Path, new_root: Path) -> dict[str, Any]:
    """Every file that is NOT one of the task-touching files must be byte-identical."""
    task_touching = {"meta/tasks.parquet", "meta/info.json"}

    def is_task_touching(rel: str) -> bool:
        return rel in task_touching or rel.startswith("data/") or rel.startswith("meta/episodes/")

    orig_snap = snapshot_dir(orig_root)
    new_snap = snapshot_dir(new_root)
    diffs = []
    for rel, (size, sha) in orig_snap.items():
        if is_task_touching(rel):
            continue
        if rel not in new_snap:
            diffs.append(f"missing in new dataset: {rel}")
        elif new_snap[rel] != (size, sha):
            diffs.append(f"byte-differs (should be identical, e.g. videos/stats.json): {rel}")
    return {"n_checked": sum(1 for r in orig_snap if not is_task_touching(r)), "diffs": diffs, "all_identical": len(diffs) == 0}


# --------------------------------------------------------------------------
# 3. Single-dataset validation
# --------------------------------------------------------------------------


def validate_single_dataset(root: Path, expected_episodes: int, expected_frames: int, expected_task: str) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    episodes = cov.load_episodes(root)

    lengths = [d["length"] for d in episodes.values()]
    total_frames = sum(lengths)

    ep_meta_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(f) for f in ep_meta_files], ignore_index=True)
    all_tasks_correct = all(list(t) == [expected_task] for t in ep_meta.sort_values("episode_index")["tasks"])

    features = info["features"]
    camera_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    camera_file_counts = {}
    for cam in camera_keys:
        camera_file_counts[cam] = len(list((root / "videos" / cam).glob("chunk-*/file-*.mp4")))

    nan_found = any(np.isnan(np.concatenate([d["state"], d["action"]], axis=1)).any() for d in episodes.values())
    inf_found = any(np.isinf(np.concatenate([d["state"], d["action"]], axis=1)).any() for d in episodes.values())

    checks = {
        "n_episodes_meta": info["total_episodes"],
        "n_episodes_loaded": len(episodes),
        "n_episodes_ok": len(episodes) == expected_episodes == info["total_episodes"],
        "n_frames_meta": info["total_frames"],
        "n_frames_loaded": total_frames,
        "n_frames_ok": total_frames == expected_frames == info["total_frames"],
        "fps": info["fps"],
        "fps_ok": info["fps"] == 30,
        "state_dim": features["observation.state"]["shape"],
        "action_dim": features["action"]["shape"],
        "dims_ok": features["observation.state"]["shape"] == [6] and features["action"]["shape"] == [6],
        "camera_file_counts": camera_file_counts,
        "cameras_ok": camera_file_counts.get("observation.images.workspace") == expected_episodes
        and camera_file_counts.get("observation.images.wrist") == expected_episodes,
        "task_count": info["total_tasks"],
        "task_count_ok": info["total_tasks"] == 1 and len(tasks_df) == 1,
        "task_string": list(tasks_df.index)[0] if len(tasks_df) else None,
        "task_string_ok": len(tasks_df) == 1 and list(tasks_df.index)[0] == expected_task,
        "all_episode_tasks_correct": all_tasks_correct,
        "nan_found": nan_found,
        "inf_found": inf_found,
        "no_nan_inf": not nan_found and not inf_found,
    }
    checks["all_ok"] = all(
        checks[k]
        for k in [
            "n_episodes_ok",
            "n_frames_ok",
            "fps_ok",
            "dims_ok",
            "cameras_ok",
            "task_count_ok",
            "task_string_ok",
            "all_episode_tasks_correct",
            "no_nan_inf",
        ]
    )
    return checks


# --------------------------------------------------------------------------
# 4. Merge
# --------------------------------------------------------------------------


def assert_feature_schema_equal(root_a: Path, repo_a: str, root_b: Path, repo_b: str) -> dict[str, Any]:
    meta_a = LeRobotDatasetMetadata(repo_a, root=root_a)
    meta_b = LeRobotDatasetMetadata(repo_b, root=root_b)
    equal = features_equal_for_merge(meta_a.features, meta_b.features)
    fps_equal = meta_a.fps == meta_b.fps
    robot_type_equal = meta_a.robot_type == meta_b.robot_type
    return {
        "features_equal_for_merge": equal,
        "fps_a": meta_a.fps,
        "fps_b": meta_b.fps,
        "fps_equal": fps_equal,
        "robot_type_a": meta_a.robot_type,
        "robot_type_b": meta_b.robot_type,
        "robot_type_equal": robot_type_equal,
        "feature_keys_a": sorted(meta_a.features.keys()),
        "feature_keys_b": sorted(meta_b.features.keys()),
        "safe_to_merge": equal and fps_equal and robot_type_equal,
    }


def merge_pick_drop_datasets(v2_root: Path, v2_repo: str, v3_root: Path, v3_repo: str, out_root: Path, out_repo: str, force: bool) -> dict[str, Any]:
    if out_root.exists():
        if not force:
            raise FileExistsError(f"{out_root} already exists - pass force=True to rebuild")
        log.info(f"removing pre-existing {out_root} (force=True)")
        shutil.rmtree(out_root)

    ds_v2 = LeRobotDataset(v2_repo, root=v2_root)
    ds_v3 = LeRobotDataset(v3_repo, root=v3_root)

    merged = merge_datasets([ds_v2, ds_v3], output_repo_id=out_repo, output_dir=out_root)
    return {
        "repo_id": merged.repo_id,
        "root": str(merged.root),
        "total_episodes": merged.meta.total_episodes,
        "total_frames": merged.meta.total_frames,
        "tasks": list(merged.meta.tasks.index),
    }


# --------------------------------------------------------------------------
# 5. Combined-dataset integrity validation
# --------------------------------------------------------------------------


def validate_combined_dataset(root: Path, expected_v2_frames: int, expected_v3_frames: int) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")

    ep_meta_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(f) for f in ep_meta_files], ignore_index=True).sort_values("episode_index").reset_index(drop=True)

    n_episodes = len(ep_meta)
    episode_indices = ep_meta["episode_index"].to_numpy()
    episode_index_continuous = bool(np.array_equal(episode_indices, np.arange(n_episodes)))

    # dataset_from_index / dataset_to_index must chain: ep[i].to == ep[i+1].from, ep[0].from==0,
    # ep[-1].to == total_frames.
    from_idx = ep_meta["dataset_from_index"].to_numpy()
    to_idx = ep_meta["dataset_to_index"].to_numpy()
    boundary_ok = bool(from_idx[0] == 0 and np.array_equal(to_idx[:-1], from_idx[1:]))
    lengths_consistent = bool(np.array_equal(to_idx - from_idx, ep_meta["length"].to_numpy()))

    total_frames_meta = info["total_frames"]
    total_frames_from_lengths = int(ep_meta["length"].sum())
    total_frames_expected = expected_v2_frames + expected_v3_frames

    all_tasks_correct = all(list(t) == [NEW_TASK] for t in ep_meta["tasks"])

    features = info["features"]
    camera_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    # File-reference integrity: every (chunk,file) referenced by an episode must exist on disk and
    # actually contain that episode's rows / duration.
    data_ref_issues = []
    video_ref_issues = {cam: [] for cam in camera_keys}
    data_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for _, row in ep_meta.iterrows():
        ep = int(row["episode_index"])
        c, f = int(row["data/chunk_index"]), int(row["data/file_index"])
        data_path = root / "data" / f"chunk-{c:03d}" / f"file-{f:03d}.parquet"
        if not data_path.exists():
            data_ref_issues.append(f"ep{ep}: missing data file chunk-{c:03d}/file-{f:03d}.parquet")
            continue
        if (c, f) not in data_cache:
            data_cache[(c, f)] = pd.read_parquet(data_path, columns=["episode_index"])
        n_rows = int((data_cache[(c, f)]["episode_index"] == ep).sum())
        if n_rows != int(row["length"]):
            data_ref_issues.append(f"ep{ep}: data file has {n_rows} rows, expected length {row['length']}")

        for cam in camera_keys:
            vc, vf = int(row[f"videos/{cam}/chunk_index"]), int(row[f"videos/{cam}/file_index"])
            video_path = root / "videos" / cam / f"chunk-{vc:03d}" / f"file-{vf:03d}.mp4"
            if not video_path.exists():
                video_ref_issues[cam].append(f"ep{ep}: missing video file chunk-{vc:03d}/file-{vf:03d}.mp4")

    # NaN/Inf across all data files actually referenced.
    nan_inf_issue = False
    for (c, f) in sorted(set(zip(ep_meta["data/chunk_index"], ep_meta["data/file_index"], strict=True))):
        data_path = root / "data" / f"chunk-{int(c):03d}" / f"file-{int(f):03d}.parquet"
        df = pd.read_parquet(data_path, columns=["action", "observation.state"])
        state = np.stack(df["observation.state"].to_numpy())
        action = np.stack(df["action"].to_numpy())
        if np.isnan(state).any() or np.isnan(action).any() or np.isinf(state).any() or np.isinf(action).any():
            nan_inf_issue = True

    checks = {
        "n_episodes": n_episodes,
        "n_episodes_ok": n_episodes == 65 == info["total_episodes"],
        "episode_index_continuous_0_to_64": episode_index_continuous,
        "dataset_from_to_index_boundary_ok": boundary_ok,
        "episode_lengths_consistent_with_from_to": lengths_consistent,
        "total_frames_meta": total_frames_meta,
        "total_frames_from_episode_lengths": total_frames_from_lengths,
        "total_frames_expected": total_frames_expected,
        "total_frames_ok": total_frames_meta == total_frames_from_lengths == total_frames_expected,
        "fps": info["fps"],
        "fps_ok": info["fps"] == 30,
        "state_dim": features["observation.state"]["shape"],
        "action_dim": features["action"]["shape"],
        "dims_ok": features["observation.state"]["shape"] == [6] and features["action"]["shape"] == [6],
        "camera_keys": camera_keys,
        "task_count": info["total_tasks"],
        "task_count_ok": info["total_tasks"] == 1 and len(tasks_df) == 1,
        "task_string": list(tasks_df.index)[0] if len(tasks_df) else None,
        "task_string_ok": len(tasks_df) == 1 and list(tasks_df.index)[0] == NEW_TASK,
        "all_episode_tasks_correct": bool(all_tasks_correct),
        "no_nan_inf": not nan_inf_issue,
        "data_reference_issues": data_ref_issues,
        "video_reference_issues": video_ref_issues,
        "data_and_video_references_ok": len(data_ref_issues) == 0 and all(len(v) == 0 for v in video_ref_issues.values()),
    }
    checks["all_ok"] = all(
        checks[k]
        for k in [
            "n_episodes_ok",
            "episode_index_continuous_0_to_64",
            "dataset_from_to_index_boundary_ok",
            "episode_lengths_consistent_with_from_to",
            "total_frames_ok",
            "fps_ok",
            "dims_ok",
            "task_count_ok",
            "task_string_ok",
            "all_episode_tasks_correct",
            "no_nan_inf",
            "data_and_video_references_ok",
        ]
    )
    return checks


# --------------------------------------------------------------------------
# 6. Coverage sanity check (reuses analyze_v2_vs_v3_start_coverage.py definitions verbatim)
# --------------------------------------------------------------------------


def coverage_sanity_check(combined_root: Path) -> dict[str, Any]:
    episodes = cov.load_episodes(combined_root)
    thresholds = cov.load_safety_thresholds(cov.DEFAULT_SAFETY_GATE_YAML)
    result, safety_rows, chunk_rows = cov.analyze_dataset(combined_root, episodes, thresholds, "combined65_on_disk")

    static_frac = result["frame_coverage"]["static_segment_data_driven"]["fraction_of_total"]
    start_l2_median = result["start_pose_diversity"]["start_frame0_pairwise_l2_deg"]["median"]
    sl_would_clamp = result["immediate_action_safety_summary"]["shoulder_lift"]["frac_would_clamp"]
    ef_would_clamp = result["immediate_action_safety_summary"]["elbow_flex"]["frac_would_clamp"]

    prior = json.loads(PRIOR_COVERAGE_SUMMARY.read_text(encoding="utf-8"))["combined_v2_plus_v3"]
    prior_static_frac = prior["frame_coverage"]["static_segment_data_driven"]["fraction_of_total"]
    prior_start_l2_median = prior["start_pose_diversity"]["start_frame0_pairwise_l2_deg"]["median"]
    prior_sl_would_clamp = prior["immediate_action_safety_summary"]["shoulder_lift"]["frac_would_clamp"]
    prior_ef_would_clamp = prior["immediate_action_safety_summary"]["elbow_flex"]["frac_would_clamp"]

    return {
        "on_disk_combined65": {
            "static_low_motion_fraction": static_frac,
            "start_pose_pairwise_l2_median_deg": start_l2_median,
            "shoulder_lift_would_clamp_frac": sl_would_clamp,
            "elbow_flex_would_clamp_frac": ef_would_clamp,
        },
        "prior_report_in_memory_concat": {
            "static_low_motion_fraction": prior_static_frac,
            "start_pose_pairwise_l2_median_deg": prior_start_l2_median,
            "shoulder_lift_would_clamp_frac": prior_sl_would_clamp,
            "elbow_flex_would_clamp_frac": prior_ef_would_clamp,
        },
        "matches": {
            "static_low_motion_fraction": abs(static_frac - prior_static_frac) < 1e-9,
            "start_pose_pairwise_l2_median_deg": abs(start_l2_median - prior_start_l2_median) < 1e-6,
            "shoulder_lift_would_clamp_frac": abs(sl_would_clamp - prior_sl_would_clamp) < 1e-9,
            "elbow_flex_would_clamp_frac": abs(ef_would_clamp - prior_ef_would_clamp) < 1e-9,
        },
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(force: bool = False) -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"new_task": NEW_TASK, "old_task": OLD_TASK}

    log.info("=== 0. snapshot originals (pre) ===")
    src_v2_before = snapshot_dir(SRC_V2)
    src_v3_before = snapshot_dir(SRC_V3)
    log.info(f"  V2 source: {len(src_v2_before)} files, V3 source: {len(src_v3_before)} files")

    log.info("=== 1. copy datasets (originals opened read-only) ===")
    copy_dataset(SRC_V2, DST_V2, force=force)
    copy_dataset(SRC_V3, DST_V3, force=force)

    log.info("=== 2. retask copies via lerobot.datasets.modify_tasks (official, in-place-on-copy) ===")
    v2_repo_id = DST_V2.name
    v3_repo_id = DST_V3.name
    retask_v2 = retask_dataset(DST_V2, v2_repo_id, NEW_TASK)
    retask_v3 = retask_dataset(DST_V3, v3_repo_id, NEW_TASK)
    report["retask"] = {"v2": retask_v2, "v3": retask_v3}
    log.info(f"  V2 tasks: {retask_v2['tasks_before']} -> {retask_v2['tasks_after']}")
    log.info(f"  V3 tasks: {retask_v3['tasks_before']} -> {retask_v3['tasks_after']}")

    log.info("=== 3. prove originals untouched + non-task columns identical ===")
    src_v2_after = snapshot_dir(SRC_V2)
    src_v3_after = snapshot_dir(SRC_V3)
    v2_untouched = diff_snapshots(src_v2_before, src_v2_after)
    v3_untouched = diff_snapshots(src_v3_before, src_v3_after)
    report["originals_untouched"] = {"v2": v2_untouched, "v3": v3_untouched}
    assert v2_untouched["unchanged"], f"V2 source mutated! {v2_untouched}"
    assert v3_untouched["unchanged"], f"V3 source mutated! {v3_untouched}"
    log.info(f"  V2 source unchanged: {v2_untouched['unchanged']}, V3 source unchanged: {v3_untouched['unchanged']}")

    v2_col_compare = compare_non_task_columns(SRC_V2, DST_V2)
    v3_col_compare = compare_non_task_columns(SRC_V3, DST_V3)
    v2_other_files = compare_other_files_byte_identical(SRC_V2, DST_V2)
    v3_other_files = compare_other_files_byte_identical(SRC_V3, DST_V3)
    report["non_task_data_identical"] = {
        "v2": {**v2_col_compare, "other_files": v2_other_files},
        "v3": {**v3_col_compare, "other_files": v3_other_files},
    }
    log.info(f"  V2 non-task columns identical: {v2_col_compare['all_non_task_columns_identical']}, other files identical: {v2_other_files['all_identical']}")
    log.info(f"  V3 non-task columns identical: {v3_col_compare['all_non_task_columns_identical']}, other files identical: {v3_other_files['all_identical']}")

    log.info("=== 4. validate individual Pick&Drop datasets ===")
    v2_validation = validate_single_dataset(DST_V2, EXPECTED["v2"]["episodes"], EXPECTED["v2"]["frames"], NEW_TASK)
    v3_validation = validate_single_dataset(DST_V3, EXPECTED["v3"]["episodes"], EXPECTED["v3"]["frames"], NEW_TASK)
    report["single_dataset_validation"] = {"v2": v2_validation, "v3": v3_validation}
    log.info(f"  V2 all_ok={v2_validation['all_ok']}, V3 all_ok={v3_validation['all_ok']}")
    assert v2_validation["all_ok"], f"V2 Pick&Drop dataset failed validation: {v2_validation}"
    assert v3_validation["all_ok"], f"V3 Pick&Drop dataset failed validation: {v3_validation}"

    log.info("=== 5. assert feature schema equality before merge ===")
    schema_check = assert_feature_schema_equal(DST_V2, v2_repo_id, DST_V3, v3_repo_id)
    report["schema_equality_check"] = schema_check
    log.info(f"  safe_to_merge={schema_check['safe_to_merge']} (features_equal={schema_check['features_equal_for_merge']}, fps_equal={schema_check['fps_equal']}, robot_type_equal={schema_check['robot_type_equal']})")
    assert schema_check["safe_to_merge"], f"V2/V3 Pick&Drop schemas differ, refusing to merge: {schema_check}"

    log.info("=== 6. merge into combined65 via lerobot.datasets.merge_datasets ===")
    combined_repo_id = DST_COMBINED.name
    merge_result = merge_pick_drop_datasets(DST_V2, v2_repo_id, DST_V3, v3_repo_id, DST_COMBINED, combined_repo_id, force=force)
    report["merge_result"] = merge_result
    log.info(f"  merged: {merge_result['total_episodes']} episodes, {merge_result['total_frames']} frames, tasks={merge_result['tasks']}")

    log.info("=== 7. validate combined65 integrity ===")
    combined_validation = validate_combined_dataset(DST_COMBINED, EXPECTED["v2"]["frames"], EXPECTED["v3"]["frames"])
    report["combined_validation"] = combined_validation
    log.info(f"  combined65 all_ok={combined_validation['all_ok']}")
    assert combined_validation["all_ok"], f"combined65 failed validation: {combined_validation}"

    log.info("=== 8. re-check originals untouched after merge too ===")
    src_v2_final = snapshot_dir(SRC_V2)
    src_v3_final = snapshot_dir(SRC_V3)
    v2_untouched_final = diff_snapshots(src_v2_before, src_v2_final)
    v3_untouched_final = diff_snapshots(src_v3_before, src_v3_final)
    report["originals_untouched_final"] = {"v2": v2_untouched_final, "v3": v3_untouched_final}
    assert v2_untouched_final["unchanged"] and v3_untouched_final["unchanged"], "originals mutated during merge step!"
    log.info("  originals still untouched after merge")

    log.info("=== 9. coverage sanity check vs prior V2-vs-V3 report ===")
    sanity = coverage_sanity_check(DST_COMBINED)
    report["coverage_sanity_check"] = sanity
    log.info(f"  matches: {sanity['matches']}")

    json_path = REPORT_DIR / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    log.info(f"wrote {json_path}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite pre-existing destination directories (never touches source datasets)")
    args = parser.parse_args()
    raise SystemExit(main(force=args.force))
