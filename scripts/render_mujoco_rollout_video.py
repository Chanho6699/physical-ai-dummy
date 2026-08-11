#!/usr/bin/env python3
"""저장된 rollout trajectory JSON(``qpos_deg``/``cube_quat`` 포함) -> MP4.

물리를 다시 돌리지 않고 저장된 qpos를 그대로 적용하는 kinematic scrub이다
(``scripts/run_mujoco_full_rollout_visual.py``의 ``--replay`` 경로와 같은 원칙). 기존
``simulation/mujoco/offscreen_recorder.py::OffscreenRecorder``를 그대로 재사용한다 - 새 비디오
인코딩 경로를 만들지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402

from runtime.common.vla_contract import JOINT_ORDER  # noqa: E402
from simulation.mujoco.offscreen_recorder import OffscreenRecorder  # noqa: E402
from simulation.mujoco.so101_model import load_model  # noqa: E402

SCENE_PATH = PROJECT_ROOT / "simulation" / "mujoco" / "assets" / "scene_pick_drop.xml"
DISPLAY_CAMERA = "workspace_cam"


def render_video(trajectory_path: Path, out_path: Path, *, camera: str = DISPLAY_CAMERA, fps: float = 30.0) -> None:
    data_json = json.loads(trajectory_path.read_text(encoding="utf-8"))
    records = data_json["step_records"]
    if not records or records[0].get("qpos_deg") is None:
        raise SystemExit(f"{trajectory_path}: qpos_deg가 없어 렌더링할 수 없습니다 (오래된 형식).")

    model = load_model(SCENE_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    jnt_id = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINT_ORDER}
    qpos_adr = {j: model.jnt_qposadr[jnt_id[j]] for j in JOINT_ORDER}
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qpos_adr = model.jnt_qposadr[cube_jnt_id]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with OffscreenRecorder(model, width=640, height=480, video_path=out_path, video_fps=fps) as recorder:
        # camera 인자는 offscreen_recorder.capture()가 받지 않으므로(항상 기본 시점) 여기서
        # renderer를 직접 만들어 camera를 지정한다 - OffscreenRecorder 내부 구현을 바꾸지 않고
        # 그 클래스가 이미 하는 PNG/MP4 저장/manifest 로직만 재사용한다.
        recorder._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        for raw in records:
            for j in JOINT_ORDER:
                data.qpos[qpos_adr[j]] = math.radians(raw["qpos_deg"][j])
            if raw.get("cube_quat"):
                data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = raw["cube_pos"]
                data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = raw["cube_quat"]
            mujoco.mj_forward(model, data)
            recorder._renderer.update_scene(data, camera=camera)
            frame = recorder._renderer.render()
            recorder._write_video_frame(frame)
            recorder._frame_index += 1

    print(f"저장됨: {out_path} ({len(records)} frames @ {fps}fps)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory", type=Path, help="trajectories/*.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--camera", default=DISPLAY_CAMERA)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    out = args.out or args.trajectory.with_suffix(".mp4")
    render_video(args.trajectory, out, camera=args.camera, fps=args.fps)


if __name__ == "__main__":
    main()
