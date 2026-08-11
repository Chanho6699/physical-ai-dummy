"""Secondary / exploratory track: MuJoCo-rendered synthetic camera -> SmolVLA -> true closed loop.

``docs/mujoco_scene_to_so101_semantics.md`` §4의 visual domain-gap 한계가 그대로 적용된다 - 이
트랙의 결과는 "실물에서 이렇게 동작할 것이다"가 아니라 "MuJoCo 렌더링이라는 낯선 입력에서
policy가 어떻게 반응하는가"로만 해석해야 한다. Primary track(``primary_replay_rollout.py`,
real observation)과 반드시 분리해서 보고한다.

observation -> policy inference -> action -> execution -> next observation -> 반복, 이라는 실제
closed-loop 시맨틱을 그대로 구현한다 (첫 observation에서 50-step chunk 하나만 뽑아 끝까지
open-loop replay하는 것과 다르다 - ``SmolVLAChunkRunner.next_queued_action``이 매 physics step마다
불리고, 내부 큐가 빌 때만 재추론한다 - 실제 배포(``select_action``)와 동일한 시맨틱).

재사용은 ``primary_replay_rollout.py``와 거의 동일하다 (action_adapter/safety_gate/
RealisticControlLayer/action_mapping/pick_drop_contacts/pick_drop_eval) - 다른 점은 관측이
dataset가 아니라 ``mujoco.Renderer``로 매 step 새로 렌더링된다는 것과, 다음 관측이 시뮬레이션
자신의 결과라는 것(진짜 closed loop)뿐이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY, JOINT_ORDER
from runtime.laptop.action_adapter import adapt_vla_action
from runtime.laptop.safety_gate import SafetyGate
from simulation.mujoco.action_mapping import build_default_mapping, map_positions_dict
from simulation.mujoco.pick_drop_contacts import cube_jaw_contact_active, get_cube_geom_id, get_jaw_geom_ids
from simulation.mujoco.pick_drop_eval import PickDropEvalResult, ReferenceZones, StepRecord, evaluate_trajectory
from simulation.mujoco.smolvla_chunk_runner import SmolVLAChunkRunner
from simulation.mujoco.so101_model import get_joint_limits
from simulation.realism.so101_control_profile import DEFAULT_PROFILE_PATH, load_control_profile
from simulation.realism.so101_realistic_control import RealisticControlConfig, RealisticControlLayer

DEFAULT_FPS = 30.0
DEFAULT_TASK = "Pick up the cube and drop it into the bin."
DEFAULT_MAX_STEPS = 400  # 13.3s @ 30fps - 접근/그립/lift/carry/release를 마칠 여유
# 실측(RTX 3050 8GB, MUJOCO_GL 기본 backend): 2-카메라 렌더링이 step당 ~200-250ms로 지배적
# (egl/osmesa backend는 이 환경(WSLg, /dev/dri 없음)에서 둘 다 즉시 실패해 기본 backend만
# 쓸 수 있음 - docs/mujoco_action_replay.md §11.8과 동일한 환경 제약). max_steps=400이면
# rollout 1회가 대략 80~100초 - benchmark 드라이버가 이 비용을 CLI/리포트에 명시한다.
DEFAULT_RENDER_SIZE = (640, 480)


class SyntheticRolloutError(RuntimeError):
    pass


@dataclass
class SyntheticRolloutResult:
    scene_id: str
    seed: int
    fps: float
    step_records: list[StepRecord]
    raw_command_log: list[dict[str, float]]
    safe_command_log: list[dict[str, float]]
    chunk_boundary_steps: list[int]
    ended_by_safety_reject: bool
    ended_reason: str  # "max_steps_reached" | "safety_reject" | "task_complete"
    eval_result: PickDropEvalResult
    real_follower_write_count: int = 0

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id, "seed": self.seed, "fps": self.fps,
            "chunk_boundary_steps": self.chunk_boundary_steps,
            "ended_by_safety_reject": self.ended_by_safety_reject, "ended_reason": self.ended_reason,
            "real_follower_write_count": self.real_follower_write_count,
            "eval_result": self.eval_result.to_dict(), "step_count": len(self.step_records),
        }


class SceneCameraRenderer:
    """``mujoco.Renderer`` 래핑 - ``offscreen_recorder.py``와 같은 단일 스레드 오프스크린
    경로를 재사용하되, PNG/MP4 저장이 아니라 매 step 두 카메라를 즉시 numpy 배열로 반환한다
    (VLA observation 입력용)."""

    def __init__(self, model: mujoco.MjModel, *, width: int = DEFAULT_RENDER_SIZE[0], height: int = DEFAULT_RENDER_SIZE[1]) -> None:
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        images = {}
        for camera_name, key in (("workspace_cam", CAMERA_WORKSPACE_KEY), ("wrist_cam", CAMERA_WRIST_KEY)):
            self._renderer.update_scene(data, camera=camera_name)
            images[key] = self._renderer.render().copy()
        return images

    def close(self) -> None:
        self._renderer.close()


def run_synthetic_closed_loop(
    *,
    chunk_runner: SmolVLAChunkRunner,
    model: mujoco.MjModel,
    safety_gate: SafetyGate,
    scene_id: str,
    initial_pose_deg: dict[str, float],
    cube_xy: tuple[float, float],
    cube_z_init: float,
    zones: ReferenceZones,
    seed: int,
    task: str = DEFAULT_TASK,
    fps: float = DEFAULT_FPS,
    max_steps: int = DEFAULT_MAX_STEPS,
    renderer: SceneCameraRenderer | None = None,
    on_step=None,  # optional callback(step_record, data) - visual mode console/live-viewer hook
) -> SyntheticRolloutResult:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    mapping = build_default_mapping([f"{j}.pos" for j in JOINT_ORDER])
    joint_limits = get_joint_limits(model)
    jnt_id = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINT_ORDER}
    qpos_adr = {j: model.jnt_qposadr[jnt_id[j]] for j in JOINT_ORDER}
    actuator_id = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINT_ORDER}
    gripperframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qpos_adr = model.jnt_qposadr[cube_jnt_id]
    cube_geom_id = get_cube_geom_id(model)
    jaw_geom_ids = get_jaw_geom_ids(model)

    def _clip_to_range(joint: str, value_deg: float) -> float:
        lo, hi = joint_limits[joint].joint_range
        return min(max(value_deg, math.degrees(lo)), math.degrees(hi))

    for j in JOINT_ORDER:
        target_rad = math.radians(_clip_to_range(j, initial_pose_deg[j]))
        data.qpos[qpos_adr[j]] = target_rad
        data.ctrl[actuator_id[j]] = target_rad
    data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = [cube_xy[0], cube_xy[1], cube_z_init]
    data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    owns_renderer = renderer is None
    if renderer is None:
        renderer = SceneCameraRenderer(model)

    profile = load_control_profile(DEFAULT_PROFILE_PATH)
    control_layer = RealisticControlLayer(profile, RealisticControlConfig())
    control_layer.reset(initial_state=initial_pose_deg)

    chunk_runner.reset()
    steps_per_frame = max(1, round((1.0 / fps) / model.opt.timestep))

    step_records: list[StepRecord] = []
    raw_log: list[dict[str, float]] = []
    safe_log: list[dict[str, float]] = []
    chunk_boundary_steps: list[int] = []
    ended_by_safety_reject = False
    ended_reason = "max_steps_reached"

    try:
        for step in range(max_steps):
            images = renderer.render(data)
            current_state_deg = {j: math.degrees(float(data.qpos[qpos_adr[j]])) for j in JOINT_ORDER}

            chunk_seed = seed if step == 0 else None
            raw_action, chunk_boundary = chunk_runner.next_queued_action(
                state_deg=current_state_deg, images=images, task=task, seed=chunk_seed
            )
            if chunk_boundary:
                chunk_boundary_steps.append(step)

            adapted = adapt_vla_action(raw_action)
            raw_log.append(dict(adapted.command_deg) if adapted.valid else dict(raw_action))

            decision = safety_gate.evaluate(
                adapted_action=adapted, current_state_deg=current_state_deg, observation_valid=True
            )

            if decision.decision == "REJECT":
                rec = StepRecord(
                    step=step, sim_time=step / fps,
                    ee_pos=tuple(data.site_xpos[gripperframe_id]), cube_pos=tuple(data.xpos[cube_body_id]),
                    gripper_cmd_percent=current_state_deg["gripper"],
                    gripper_raw_percent=adapted.command_deg.get("gripper", current_state_deg["gripper"]) if adapted.valid else current_state_deg["gripper"],
                    cube_jaw_contact=cube_jaw_contact_active(model, data, cube_geom_id=cube_geom_id, jaw_geom_ids=jaw_geom_ids),
                    safety_decision="REJECT", chunk_boundary=chunk_boundary,
                    qpos_deg=dict(current_state_deg), cube_quat=tuple(data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7]),
                )
                step_records.append(rec)
                safe_log.append(current_state_deg)
                if on_step is not None:
                    on_step(rec, data)
                ended_by_safety_reject = True
                ended_reason = "safety_reject"
                break

            # 시뮬레이션 시계 사용 이유는 primary_replay_rollout.py의 동일 지점 주석 참고 -
            # wall-clock을 쓰면 실시간 페이싱이 없는 이 루프에서 latency queue가 "시간이 거의
            # 안 지났다"고 착각해 명령을 계속 지연시키는 버그가 있었다.
            processed = control_layer.process(dict(decision.safe_action), now=step / fps, simulated_actual=current_state_deg)
            processed_rad = map_positions_dict(processed.processed_action, mapping)
            for j in JOINT_ORDER:
                data.ctrl[actuator_id[j]] = processed_rad[j]
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            contact = cube_jaw_contact_active(model, data, cube_geom_id=cube_geom_id, jaw_geom_ids=jaw_geom_ids)
            safe_log.append(dict(decision.safe_action))
            applied_deg = {j: math.degrees(float(data.qpos[qpos_adr[j]])) for j in JOINT_ORDER}
            rec = StepRecord(
                step=step, sim_time=step / fps,
                ee_pos=tuple(data.site_xpos[gripperframe_id]), cube_pos=tuple(data.xpos[cube_body_id]),
                gripper_cmd_percent=decision.safe_action["gripper"],
                gripper_raw_percent=adapted.command_deg.get("gripper", decision.safe_action["gripper"]) if adapted.valid else decision.safe_action["gripper"],
                cube_jaw_contact=contact, safety_decision=decision.decision, chunk_boundary=chunk_boundary,
                qpos_deg=dict(applied_deg), cube_quat=tuple(data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7]),
            )
            step_records.append(rec)
            if on_step is not None:
                on_step(rec, data)
    finally:
        if owns_renderer:
            renderer.close()

    eval_result = evaluate_trajectory(
        step_records, zones, per_step_command_deg=safe_log,
        ended_by_safety_reject=ended_by_safety_reject,
        ended_by_timeout=(not ended_by_safety_reject),
    )
    if eval_result.kinematic.kinematic_pick_drop_success and not ended_by_safety_reject:
        ended_reason = "task_complete"

    return SyntheticRolloutResult(
        scene_id=scene_id, seed=seed, fps=fps, step_records=step_records, raw_command_log=raw_log,
        safe_command_log=safe_log, chunk_boundary_steps=chunk_boundary_steps,
        ended_by_safety_reject=ended_by_safety_reject, ended_reason=ended_reason,
        eval_result=eval_result, real_follower_write_count=0,
    )
