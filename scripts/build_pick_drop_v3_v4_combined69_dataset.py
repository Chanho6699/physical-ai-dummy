#!/usr/bin/env python3
"""Build `data/so101_cube_pick_drop_v3_v4_combined69_v1` (69 episodes) from V3-clean (30ep) and
V4-generalization (39ep) - using LeRobot's own official dataset-editing API
(`lerobot.datasets.merge_datasets` -> `lerobot.datasets.aggregate.aggregate_datasets`), the same
function `build_pick_drop_datasets.py` used to build `combined65_v1`, not a naive folder `mv` or
manual parquet edit.

Unlike the V2+V3 -> combined65 build, **no retasking is needed here**: both
`so101_cube_pick_drop_start_coverage_v3_clean` and `so101_cube_pick_drop_generalization_v4`
already store the canonical task string `"Pick up the cube and drop it into the bin."` (verified
below before merging). So this script merges the two ORIGINAL dataset directories directly -
never copies/retasks them first - and proves via before/after checksum snapshots that
`merge_datasets` never opened either source in write mode.

Does NOT touch V2 (not used in this ablation). Does NOT touch the historical heldout10
(`so101_cube_xy_midpoint_test10_v2_clean`) or V4's own heldout6
(`so101_cube_pick_drop_v4_heldout10`, despite the "10" in its name it holds only 6 held-out
episodes) - neither is read or written by this script. Does not train, does not create
checkpoints, does not run git commands.

Usage::

    source ~/lerobot/.venv/bin/activate
    python scripts/build_pick_drop_v3_v4_combined69_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[build] %(message)s")
log = logging.getLogger("build_pick_drop_v3_v4_combined69")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v2_vs_v3_start_coverage as cov  # noqa: E402  (reuse coverage/NaN-Inf loader helpers)

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, merge_datasets  # noqa: E402
from lerobot.datasets.feature_utils import features_equal_for_merge  # noqa: E402

SRC_V3 = PROJECT_ROOT / "data" / "so101_cube_pick_drop_start_coverage_v3_clean"
SRC_V4 = PROJECT_ROOT / "data" / "so101_cube_pick_drop_generalization_v4"
DST_COMBINED = PROJECT_ROOT / "data" / "so101_cube_pick_drop_v3_v4_combined69_v1"
REPORT_DIR = PROJECT_ROOT / "reports" / "pick_drop_v3_v4_combined69_dataset_build"

CANONICAL_TASK = "Pick up the cube and drop it into the bin."

EXPECTED = {
    "v3": {"episodes": 30, "frames": 9822},
    "v4": {"episodes": 39, "frames": 12795},
}
EXPECTED_TOTAL_EPISODES = EXPECTED["v3"]["episodes"] + EXPECTED["v4"]["episodes"]  # 69
EXPECTED_TOTAL_FRAMES = EXPECTED["v3"]["frames"] + EXPECTED["v4"]["frames"]


# --------------------------------------------------------------------------
# 0. Checksums (prove the originals are never mutated)
# --------------------------------------------------------------------------


def snapshot_dir(root: Path) -> dict[str, tuple[int, str]]:
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
# 1. Pre-merge validation of each source (canonical task already present, no retask)
# --------------------------------------------------------------------------


def validate_source_dataset(root: Path, expected_episodes: int, expected_frames: int) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    episodes = cov.load_episodes(root)
    lengths = [d["length"] for d in episodes.values()]
    total_frames = sum(lengths)
    features = info["features"]

    nan_found = any(np.isnan(np.concatenate([d["state"], d["action"]], axis=1)).any() for d in episodes.values())
    inf_found = any(np.isinf(np.concatenate([d["state"], d["action"]], axis=1)).any() for d in episodes.values())

    checks = {
        "root": str(root),
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
        "task_count": info["total_tasks"],
        "task_string": list(tasks_df.index)[0] if len(tasks_df) else None,
        "task_string_ok": len(tasks_df) == 1 and list(tasks_df.index)[0] == CANONICAL_TASK,
        "no_nan_inf": not nan_found and not inf_found,
    }
    checks["all_ok"] = all(checks[k] for k in ["n_episodes_ok", "n_frames_ok", "fps_ok", "dims_ok", "task_string_ok", "no_nan_inf"])
    return checks


def assert_feature_schema_equal(root_a: Path, repo_a: str, root_b: Path, repo_b: str) -> dict[str, Any]:
    meta_a = LeRobotDatasetMetadata(repo_a, root=root_a)
    meta_b = LeRobotDatasetMetadata(repo_b, root=root_b)
    equal = features_equal_for_merge(meta_a.features, meta_b.features)
    fps_equal = meta_a.fps == meta_b.fps
    robot_type_equal = meta_a.robot_type == meta_b.robot_type
    return {
        "features_equal_for_merge": equal,
        "fps_a": meta_a.fps, "fps_b": meta_b.fps, "fps_equal": fps_equal,
        "robot_type_a": meta_a.robot_type, "robot_type_b": meta_b.robot_type, "robot_type_equal": robot_type_equal,
        "feature_keys_a": sorted(meta_a.features.keys()), "feature_keys_b": sorted(meta_b.features.keys()),
        "safe_to_merge": equal and fps_equal and robot_type_equal,
    }


# --------------------------------------------------------------------------
# 2. Merge
# --------------------------------------------------------------------------


def merge_v3_v4(v3_root: Path, v3_repo: str, v4_root: Path, v4_repo: str, out_root: Path, out_repo: str, force: bool) -> dict[str, Any]:
    if out_root.exists():
        if not force:
            raise FileExistsError(f"{out_root} already exists - pass force=True to rebuild")
        log.info(f"removing pre-existing {out_root} (force=True)")
        shutil.rmtree(out_root)

    ds_v3 = LeRobotDataset(v3_repo, root=v3_root)
    ds_v4 = LeRobotDataset(v4_repo, root=v4_root)

    merged = merge_datasets([ds_v3, ds_v4], output_repo_id=out_repo, output_dir=out_root)
    return {
        "repo_id": merged.repo_id, "root": str(merged.root),
        "total_episodes": merged.meta.total_episodes, "total_frames": merged.meta.total_frames,
        "tasks": list(merged.meta.tasks.index),
    }


# --------------------------------------------------------------------------
# 3. Combined-dataset integrity validation
# --------------------------------------------------------------------------


def validate_combined_dataset(root: Path) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")

    ep_meta_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(f) for f in ep_meta_files], ignore_index=True).sort_values("episode_index").reset_index(drop=True)

    n_episodes = len(ep_meta)
    episode_indices = ep_meta["episode_index"].to_numpy()
    episode_index_continuous = bool(np.array_equal(episode_indices, np.arange(n_episodes)))

    from_idx = ep_meta["dataset_from_index"].to_numpy()
    to_idx = ep_meta["dataset_to_index"].to_numpy()
    boundary_ok = bool(from_idx[0] == 0 and np.array_equal(to_idx[:-1], from_idx[1:]))
    lengths_consistent = bool(np.array_equal(to_idx - from_idx, ep_meta["length"].to_numpy()))

    total_frames_meta = info["total_frames"]
    total_frames_from_lengths = int(ep_meta["length"].sum())

    all_tasks_correct = all(list(t) == [CANONICAL_TASK] for t in ep_meta["tasks"])

    features = info["features"]
    camera_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

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
        "n_episodes_ok": n_episodes == EXPECTED_TOTAL_EPISODES == info["total_episodes"],
        "episode_index_continuous": episode_index_continuous,
        "dataset_from_to_index_boundary_ok": boundary_ok,
        "episode_lengths_consistent_with_from_to": lengths_consistent,
        "total_frames_meta": total_frames_meta,
        "total_frames_from_episode_lengths": total_frames_from_lengths,
        "total_frames_expected": EXPECTED_TOTAL_FRAMES,
        "total_frames_ok": total_frames_meta == total_frames_from_lengths == EXPECTED_TOTAL_FRAMES,
        "fps": info["fps"], "fps_ok": info["fps"] == 30,
        "state_dim": features["observation.state"]["shape"], "action_dim": features["action"]["shape"],
        "dims_ok": features["observation.state"]["shape"] == [6] and features["action"]["shape"] == [6],
        "camera_keys": camera_keys,
        "cameras_ok": "observation.images.workspace" in camera_keys and "observation.images.wrist" in camera_keys,
        "task_count": info["total_tasks"],
        "task_count_ok": info["total_tasks"] == 1 and len(tasks_df) == 1,
        "task_string": list(tasks_df.index)[0] if len(tasks_df) else None,
        "task_string_ok": len(tasks_df) == 1 and list(tasks_df.index)[0] == CANONICAL_TASK,
        "all_episode_tasks_correct": bool(all_tasks_correct),
        "no_nan_inf": not nan_inf_issue,
        "data_reference_issues": data_ref_issues,
        "video_reference_issues": video_ref_issues,
        "data_and_video_references_ok": len(data_ref_issues) == 0 and all(len(v) == 0 for v in video_ref_issues.values()),
    }
    checks["all_ok"] = all(
        checks[k]
        for k in [
            "n_episodes_ok", "episode_index_continuous", "dataset_from_to_index_boundary_ok",
            "episode_lengths_consistent_with_from_to", "total_frames_ok", "fps_ok", "dims_ok", "cameras_ok",
            "task_count_ok", "task_string_ok", "all_episode_tasks_correct", "no_nan_inf",
            "data_and_video_references_ok",
        ]
    )
    return checks


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(force: bool = False) -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"canonical_task": CANONICAL_TASK}

    log.info("=== 0. snapshot originals (pre) ===")
    src_v3_before = snapshot_dir(SRC_V3)
    src_v4_before = snapshot_dir(SRC_V4)
    log.info(f"  V3 source: {len(src_v3_before)} files, V4 source: {len(src_v4_before)} files")

    log.info("=== 1. validate sources already canonical (no retask needed) ===")
    v3_validation = validate_source_dataset(SRC_V3, EXPECTED["v3"]["episodes"], EXPECTED["v3"]["frames"])
    v4_validation = validate_source_dataset(SRC_V4, EXPECTED["v4"]["episodes"], EXPECTED["v4"]["frames"])
    report["source_validation"] = {"v3": v3_validation, "v4": v4_validation}
    log.info(f"  V3 all_ok={v3_validation['all_ok']} (task={v3_validation['task_string']!r})")
    log.info(f"  V4 all_ok={v4_validation['all_ok']} (task={v4_validation['task_string']!r})")
    assert v3_validation["all_ok"], f"V3 source failed pre-merge validation: {v3_validation}"
    assert v4_validation["all_ok"], f"V4 source failed pre-merge validation: {v4_validation}"

    log.info("=== 2. assert feature schema equality before merge ===")
    schema_check = assert_feature_schema_equal(SRC_V3, SRC_V3.name, SRC_V4, SRC_V4.name)
    report["schema_equality_check"] = schema_check
    log.info(f"  safe_to_merge={schema_check['safe_to_merge']}")
    assert schema_check["safe_to_merge"], f"V3/V4 schemas differ, refusing to merge: {schema_check}"

    log.info("=== 3. merge V3(30ep) + V4(39ep) -> combined69 via lerobot.datasets.merge_datasets ===")
    combined_repo_id = DST_COMBINED.name
    merge_result = merge_v3_v4(SRC_V3, SRC_V3.name, SRC_V4, SRC_V4.name, DST_COMBINED, combined_repo_id, force=force)
    report["merge_result"] = merge_result
    log.info(f"  merged: {merge_result['total_episodes']} episodes, {merge_result['total_frames']} frames, tasks={merge_result['tasks']}")

    log.info("=== 4. prove originals untouched by the merge ===")
    src_v3_after = snapshot_dir(SRC_V3)
    src_v4_after = snapshot_dir(SRC_V4)
    v3_untouched = diff_snapshots(src_v3_before, src_v3_after)
    v4_untouched = diff_snapshots(src_v4_before, src_v4_after)
    report["originals_untouched"] = {"v3": v3_untouched, "v4": v4_untouched}
    assert v3_untouched["unchanged"], f"V3 source mutated! {v3_untouched}"
    assert v4_untouched["unchanged"], f"V4 source mutated! {v4_untouched}"
    log.info(f"  V3 source unchanged: {v3_untouched['unchanged']}, V4 source unchanged: {v4_untouched['unchanged']}")

    log.info("=== 5. validate combined69 integrity ===")
    combined_validation = validate_combined_dataset(DST_COMBINED)
    report["combined_validation"] = combined_validation
    log.info(f"  combined69 all_ok={combined_validation['all_ok']}")
    assert combined_validation["all_ok"], f"combined69 failed validation: {combined_validation}"

    json_path = REPORT_DIR / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    log.info(f"wrote {json_path}")
    log.info("=== DONE: data/so101_cube_pick_drop_v3_v4_combined69_v1 built and validated ===")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite pre-existing destination directory (never touches source datasets)")
    args = parser.parse_args()
    raise SystemExit(main(force=args.force))
