#!/usr/bin/env python3
"""Generate ``simulation/mujoco/assets/scene_pick_drop.xml`` and
``configs/mujoco_rollout_scenes_v1.json`` for the MuJoCo full-rollout candidate-comparison
benchmark (``reports/mujoco_full_rollout_candidate_comparison_v1``).

This script does NOT modify ``simulation/mujoco/assets/scene.xml`` — every existing tool/test that
loads the default scene (dataset action replay, Shadow Mode, live web viewer, ``tests/test_mujoco_*``)
keeps working unmodified. It only reads that file's text and writes a new sibling file with a cube,
a bin, and a ``workspace_cam`` inserted before ``</worldbody>``.

# Why these particular coordinates (no fabricated "historical" positions)

No real cube/bin XY ground truth exists anywhere in this repository for any past experiment
(verified by a full-repo grep before this script was written — see the plan/report). Per explicit
instruction, this benchmark must not invent numbers and label them "historical". Instead:

1. The bin position and the 10-point cube grid below were chosen by sweeping the *actual* SO-101
   MuJoCo arm's joint ranges with ``mujoco.mj_forward`` and reading the real ``gripperframe`` site
   (already defined in the vendored MJCF) cartesian position at each sampled posture, then picking
   points that are within ~3cm of an empirically-reachable sample (i.e. within the gripper's own
   physical extent) — a physically grounded method, not a guess. See
   ``docs/mujoco_scene_to_so101_semantics.md`` for the full sweep methodology and validation numbers.
2. Each of the 10 scenes' *robot initial pose* is not invented either — it is the real recorded
   frame-0 ``observation.state`` of one of the 10 episodes of
   ``data/so101_cube_xy_midpoint_test10_v2_clean``, a dataset independently confirmed excluded from
   both candidate checkpoints' training data. These 10 episodes' start poses are themselves nearly
   identical (a fixed teleoperation "home" pose) — that is a real, expected property of the source
   data, not a bug in this script.

The resulting scenes are named ``mujoco_rollout_test01``..``mujoco_rollout_test10`` and must never be
referred to as "historical T01-T10" (that label belongs to a different, unrelated set of real-hardware
Shadow Mode sessions with no recorded object position).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSETS_DIR = PROJECT_ROOT / "simulation" / "mujoco" / "assets"
SRC_SCENE = ASSETS_DIR / "scene.xml"
DST_SCENE = ASSETS_DIR / "scene_pick_drop.xml"
SCENES_CONFIG_PATH = PROJECT_ROOT / "configs" / "mujoco_rollout_scenes_v1.json"
REAL_INITIAL_POSE_DATASET = "data/so101_cube_xy_midpoint_test10_v2_clean"

# Cube half-extent (12mm -> 24mm cube) — a small manipulable object matching the physical scale a
# SO-101 gripper (jaw opening ~ a few cm, per ctrlrange 0-100%) can enclose. No real cube dimension
# is recorded anywhere in the repo (see docs/mujoco_scene_to_so101_semantics.md) so this is a
# reasonable synthetic approximation, flagged as such.
CUBE_HALF_SIZE = 0.012
CUBE_MASS = 0.015  # 15g - light, plausible for a small 3D-printed/wood pick-drop training cube

BIN_CENTER = (0.30, -0.30)
BIN_INNER_HALF = 0.05  # 10cm x 10cm interior
BIN_WALL_HEIGHT = 0.035
BIN_WALL_THICKNESS = 0.004
BIN_FLOOR_Z = 0.0

# The 10-point cube grid (verified <=3.3cm from an FK-reachable sample point - see
# scripts/generate_mujoco_pick_drop_scene.py docstring / docs/mujoco_scene_to_so101_semantics.md).
CUBE_GRID_XY = [
    (0.16, 0.16),
    (0.22, 0.16),
    (0.28, 0.16),
    (0.16, 0.02),
    (0.22, 0.02),
    (0.28, 0.02),
    (0.16, -0.12),
    (0.22, -0.12),
    (0.28, -0.12),
    (0.22, 0.22),
]

WORKSPACE_CAM_POS = (0.55, 0.0, 0.45)
WORKSPACE_CAM_EULER = (0.0, 0.95, 1.5708)  # looking down/back toward the table+arm, front-above pose


def _cube_xml() -> str:
    return f"""
    <!-- === mujoco_rollout: pick/drop cube (synthetic; see scene_pick_drop.xml module docstring
         in scripts/generate_mujoco_pick_drop_scene.py for how position/size were chosen) === -->
    <body name="cube" pos="0.22 0.02 {CUBE_HALF_SIZE + 0.002}">
      <freejoint name="cube_freejoint"/>
      <inertial pos="0 0 0" mass="{CUBE_MASS}" diaginertia="1e-6 1e-6 1e-6"/>
      <geom name="cube_geom" type="box" size="{CUBE_HALF_SIZE} {CUBE_HALF_SIZE} {CUBE_HALF_SIZE}"
        rgba="0.85 0.15 0.15 1" condim="4" friction="1.0 0.01 0.001" solref="0.01 1" priority="1"/>
    </body>
"""


def _bin_xml() -> str:
    bx, by = BIN_CENTER
    hw = BIN_INNER_HALF
    wh = BIN_WALL_HEIGHT
    wt = BIN_WALL_THICKNESS
    z_wall = BIN_FLOOR_Z + wh / 2.0
    return f"""
    <!-- === mujoco_rollout: pick/drop bin (synthetic; static open-top container) === -->
    <body name="bin" pos="{bx} {by} 0">
      <geom name="bin_floor" type="box" size="{hw} {hw} 0.003" pos="0 0 {BIN_FLOOR_Z + 0.003}"
        rgba="0.2 0.2 0.6 1" condim="3"/>
      <geom name="bin_wall_neg_x" type="box" size="{wt} {hw} {wh / 2.0}" pos="{-hw} 0 {z_wall}"
        rgba="0.2 0.2 0.6 0.9" condim="3"/>
      <geom name="bin_wall_pos_x" type="box" size="{wt} {hw} {wh / 2.0}" pos="{hw} 0 {z_wall}"
        rgba="0.2 0.2 0.6 0.9" condim="3"/>
      <geom name="bin_wall_neg_y" type="box" size="{hw} {wt} {wh / 2.0}" pos="0 {-hw} {z_wall}"
        rgba="0.2 0.2 0.6 0.9" condim="3"/>
      <geom name="bin_wall_pos_y" type="box" size="{hw} {wt} {wh / 2.0}" pos="0 {hw} {z_wall}"
        rgba="0.2 0.2 0.6 0.9" condim="3"/>
      <site name="bin_center" pos="0 0 {BIN_FLOOR_Z + 0.01}" size="0.004" rgba="1 1 0 1"/>
    </body>
"""


def _workspace_cam_xml() -> str:
    px, py, pz = WORKSPACE_CAM_POS
    ex, ey, ez = WORKSPACE_CAM_EULER
    return f"""
    <!-- === mujoco_rollout: synthetic workspace camera - NO real external camera pose exists in
         this repo to copy (only wrist_cam is CAD-derived from the real mount). This pose is a
         generic "front-above, looking down at the table" placement, flagged as a visual
         domain-gap limitation in docs/mujoco_scene_to_so101_semantics.md. === -->
    <camera name="workspace_cam" mode="fixed" pos="{px} {py} {pz}" euler="{ex} {ey} {ez}"
      resolution="1920 1080" sensorsize="0.00576 0.00324" focal="0.0036 0.0036"/>
"""


def build_scene_pick_drop_xml() -> str:
    if not SRC_SCENE.is_file():
        raise FileNotFoundError(f"Source scene not found: {SRC_SCENE}")
    text = SRC_SCENE.read_text(encoding="utf-8")

    marker = "  </worldbody>"
    if marker not in text:
        raise ValueError(f"Expected '{marker.strip()}' in {SRC_SCENE} - scene.xml structure changed?")

    insertion = _cube_xml() + _bin_xml() + _workspace_cam_xml() + "\n"
    text = text.replace(marker, insertion + marker, 1)
    text = text.replace(
        'model="so101_dataset_action_replay_scene"',
        'model="so101_mujoco_rollout_pick_drop_scene"',
        1,
    )
    header_note = (
        "\n  <!-- Generated by scripts/generate_mujoco_pick_drop_scene.py from scene.xml "
        "(robot body chain unmodified) + a synthetic cube/bin/workspace_cam for "
        "reports/mujoco_full_rollout_candidate_comparison_v1. See "
        "docs/mujoco_scene_to_so101_semantics.md for what is real vs synthetic in this file. -->\n"
    )
    text = text.replace("<mujoco ", header_note + "<mujoco ", 1) if "<mujoco " in text[:60] else text
    return text


def build_scenes_config() -> dict:
    frame0_states_path = Path(__file__).resolve().parent.parent / "configs" / "generated" / "_midpoint10_frame0_states.json"
    # Loaded fresh each run from the dataset itself (see load_real_initial_poses) rather than only
    # trusting a cached file, so this config always reflects the actual dataset on disk.
    from simulation.mujoco.dataset_loader import load_dataset_info, load_episode
    from simulation.mujoco.so101_model import SO101_JOINT_NAMES

    info = load_dataset_info(REAL_INITIAL_POSE_DATASET)
    if info.total_episodes != len(CUBE_GRID_XY):
        raise ValueError(
            f"{REAL_INITIAL_POSE_DATASET} has {info.total_episodes} episodes, "
            f"expected {len(CUBE_GRID_XY)} to pair 1:1 with the cube grid."
        )

    scenes = []
    for i, (cx, cy) in enumerate(CUBE_GRID_XY):
        ep = load_episode(REAL_INITIAL_POSE_DATASET, i, info)
        state0 = ep.state[0]
        initial_pose_deg = {name: float(state0[j]) for j, name in enumerate(SO101_JOINT_NAMES)}
        scenes.append(
            {
                "scene_id": f"mujoco_rollout_test{i + 1:02d}",
                "cube_xy": [cx, cy],
                "cube_z_init": CUBE_HALF_SIZE + 0.002,
                "bin_center_xy": list(BIN_CENTER),
                "initial_pose_deg": initial_pose_deg,
                "initial_pose_source": {
                    "dataset": REAL_INITIAL_POSE_DATASET,
                    "episode_index": i,
                    "frame_index": 0,
                    "note": "Real recorded observation.state, frame 0 of this episode - not invented.",
                },
            }
        )

    return {
        "schema": "mujoco_rollout_scenes_v1",
        "note": (
            "These are NOT historical T01-T10 (no real cube/bin coordinates exist anywhere in this "
            "repo for that label - see docs/mujoco_scene_to_so101_semantics.md). cube_xy/bin_center_xy "
            "are synthetic but FK-reachability-verified; initial_pose_deg is real recorded data."
        ),
        "scene_path": "simulation/mujoco/assets/scene_pick_drop.xml",
        "cube_half_size": CUBE_HALF_SIZE,
        "cube_mass_kg": CUBE_MASS,
        "bin_inner_half": BIN_INNER_HALF,
        "bin_wall_height": BIN_WALL_HEIGHT,
        "scenes": scenes,
    }


def main() -> None:
    xml_text = build_scene_pick_drop_xml()
    DST_SCENE.write_text(xml_text, encoding="utf-8")
    print(f"[생성] {DST_SCENE}")

    config = build_scenes_config()
    SCENES_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCENES_CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[생성] {SCENES_CONFIG_PATH} ({len(config['scenes'])} scenes)")

    # Sanity-load the generated scene through the exact same loader every other tool uses.
    from simulation.mujoco.so101_model import load_model

    model = load_model(DST_SCENE)
    for name in ("cube", "bin"):
        import mujoco

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id >= 0, f"body '{name}' missing after generation"
    print("[검증] scene_pick_drop.xml 로딩 성공, cube/bin body 확인됨.")


if __name__ == "__main__":
    main()
