#!/usr/bin/env python3
"""Build synthetic (offline-proxy) T02-T10 "Shadow report" JSON files for the T01-T10 seed-sweep
generalization check.

Context / why this script exists
---------------------------------
``reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`` is a *real* Shadow Mode capture:
``scripts/run_shadow_mode.py`` running against live SO-101 hardware (real camera frames, real
read-only follower state). ``scripts/sweep_grid35_first_action_seed.py`` and
``scripts/analyze_seed_mitigation_strategies.py`` were built to consume that report's minimal
shape (``observation.state``, ``vla.raw_action``, ``observation.camera_frame_paths``,
``scene_metadata.label``).

No equivalent real hardware capture exists for T02-T10, and this analysis session has no SO-101
hardware attached (no ``/dev/serial/by-id/*``, no ``/dev/video*``) - so a new live Shadow report
cannot be produced here. Per explicit user decision (2026-08-08), this script instead builds a
**synthetic, offline proxy** "Shadow observation" for each of T02-T10 from the held-out
``data/so101_cube_xy_midpoint_test10_v2_clean`` dataset (10 episodes, never used in training),
taking each episode's frame_index=0 (state + first workspace/wrist frame) as that scene's
reference observation:

    episode_index 0 -> T01 (already has a real capture - not reused/overwritten here)
    episode_index 1 -> T02
    episode_index 2 -> T03
    ...
    episode_index 9 -> T10

This is explicitly **not** a live hardware Shadow capture. Every output JSON is marked
``"synthetic": true`` with a ``provenance`` block, and every downstream report generated from it
must carry that caveat forward.

Deliberately out of scope (same constraints as the parent scripts):
  * No training / fine-tuning, no modification of ``data/so101_cube_xy_midpoint_test10_v2_clean``
    or any other dataset.
  * No modification of Safety Gate thresholds.
  * No writes to any robot, real or simulated (this script does not even touch inference).
  * No modification of the LeRobot library itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.common.vla_contract import JOINT_ORDER  # noqa: E402

DEFAULT_SOURCE_DATASET = PROJECT_ROOT / "data" / "so101_cube_xy_midpoint_test10_v2_clean"
DEFAULT_SOURCE_REPO_ID = "local/so101_cube_xy_midpoint_test10_v2_clean"
DEFAULT_TASK = "Pick up the cube and place it in the target area."
DEFAULT_OUT_ROOT = PROJECT_ROOT / "reports"

# episode_index -> scene label, per the user-approved sequential mapping (2026-08-08).
# episode 0 == T01 is intentionally excluded: T01 already has a real hardware Shadow capture.
EPISODE_TO_SCENE = {i: f"T{i + 1:02d}" for i in range(1, 10)}  # {1: "T02", ..., 9: "T10"}


def extract_scene(dataset, episode_index: int) -> dict[str, Any]:
    """Pull frame_index=0 of one episode out of an already-loaded LeRobotDataset."""
    ep_row = dataset.meta.episodes[episode_index]
    global_idx = int(ep_row["dataset_from_index"])
    sample = dataset[global_idx]
    assert int(sample["episode_index"]) == episode_index
    assert int(sample["frame_index"]) == 0

    state = {j: float(sample["observation.state"][i]) for i, j in enumerate(JOINT_ORDER)}
    action = {j: float(sample["action"][i]) for i, j in enumerate(JOINT_ORDER)}

    images = {}
    for key in ("observation.images.workspace", "observation.images.wrist"):
        chw = sample[key].detach().cpu().numpy()  # float32 CHW in [0, 1]
        hwc_uint8 = np.clip(chw, 0.0, 1.0).transpose(1, 2, 0) * 255.0
        images[key] = hwc_uint8.round().astype(np.uint8)

    return {"state": state, "action": action, "images": images}


def write_scene_report(
    out_root: Path,
    scene_label: str,
    episode_index: int,
    scene: dict[str, Any],
    task: str,
    source_dataset: Path,
    source_repo_id: str,
) -> Path:
    from PIL import Image

    out_dir = out_root / f"grid35_v2_shadow_{scene_label}"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    workspace_path = images_dir / "workspace.jpg"
    wrist_path = images_dir / "wrist.jpg"
    Image.fromarray(scene["images"]["observation.images.workspace"]).save(workspace_path, quality=95)
    Image.fromarray(scene["images"]["observation.images.wrist"]).save(wrist_path, quality=95)

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "synthetic": True,
        "mode": "SYNTHETIC_OFFLINE_PROXY",
        "provenance": {
            "statement": (
                "NOT a live SO-101 hardware Shadow capture. This analysis session has no SO-101 "
                "hardware attached (no /dev/serial/by-id/*, no /dev/video*), so scripts/run_shadow_mode.py "
                "could not be run for this scene. Built instead from the held-out "
                f"{source_repo_id} dataset (never used in training) by taking episode "
                f"{episode_index}'s frame_index=0 (state + first workspace/wrist frame) as this "
                "scene's reference observation, per explicit user decision (2026-08-08)."
            ),
            "source_dataset": str(source_dataset),
            "source_repo_id": source_repo_id,
            "source_episode_index": episode_index,
            "source_frame_index": 0,
            "mapping_rule": "episode_index N -> T0(N+1), sequential (episode 0 -> T01 already has a real capture and is not reused here).",
            "generated_at": generated_at,
            "generated_by": "scripts/build_synthetic_midpoint_shadow_reports.py",
        },
        "generated_at": generated_at,
        "task": task,
        "scene_metadata": {
            "label": scene_label,
            "note": f"Synthetic offline proxy for held-out scene {scene_label} (see provenance).",
        },
        "observation": {
            "state": scene["state"],
            "camera_frame_paths": {
                "workspace_path": str(workspace_path.resolve()),
                "wrist_path": str(wrist_path.resolve()),
            },
        },
        "vla": {
            "raw_action": scene["action"],
            "raw_action_note": (
                "This is the held-out dataset's own recorded demonstration action at this frame, "
                "NOT a live VLA output (there was no live inference to source it from). It is used "
                "only as an informational value; the seed-sweep's GT-delta lookup "
                "(nearest_frame_trajectory_analysis) matches on observation.state alone against the "
                "*training* dataset, so this field does not influence the GT reference used for "
                "scoring."
            ),
        },
    }

    report_path = out_dir / f"shadow_synthetic_{scene_label}.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"[build] loading held-out dataset {args.source_repo_id} from {args.source_dataset}")
    dataset = LeRobotDataset(repo_id=args.source_repo_id, root=str(args.source_dataset.resolve()))
    assert len(dataset.meta.episodes) == 10, f"expected 10 held-out episodes, found {len(dataset.meta.episodes)}"

    written: dict[str, str] = {}
    for episode_index, scene_label in sorted(EPISODE_TO_SCENE.items()):
        print(f"[build] episode {episode_index} -> {scene_label}")
        scene = extract_scene(dataset, episode_index)
        report_path = write_scene_report(
            args.out_root.resolve(),
            scene_label,
            episode_index,
            scene,
            args.task,
            args.source_dataset.resolve(),
            args.source_repo_id,
        )
        written[scene_label] = str(report_path)
        print(f"[build]   wrote {report_path}")

    print("")
    print(f"[build] wrote {len(written)} synthetic scene reports:")
    for label, path in written.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
