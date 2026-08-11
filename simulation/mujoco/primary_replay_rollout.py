"""Primary track: real dataset observation -> SmolVLA action chunk -> MuJoCo 물리 실행.

``docs/mujoco_scene_to_so101_semantics.md``/plan amendment 참고. 이 모듈이 하는 일은 딱 하나다 -
**실제 기록된 카메라 이미지 + state**로 SmolVLA를 teacher-forced로 구동하고(각 chunk 경계마다
그 시점의 진짜 dataset 프레임을 사용, 시뮬레이션 상태를 관측으로 되먹임하지 않음), 그렇게 나온
action chunk 전체를 MuJoCo에서 물리적으로 실행해 joint safety/trajectory continuity/접근-그립-
lift-carry 방향을 채점한다. 관측이 항상 real이므로 visual domain gap이 없다 - synthetic
closed-loop(Secondary)와 반드시 분리해서 보고한다.

재사용:
    - ``simulation.mujoco.smolvla_chunk_runner.SmolVLAChunkRunner.predict_chunk`` (매 chunk 경계마다
      새로 샘플링 - 내부 큐를 쓰지 않는다, Secondary의 ``next_queued_action``과 다름)
    - ``runtime.laptop.action_adapter.adapt_vla_action`` (변경 없음)
    - ``runtime.laptop.safety_gate.SafetyGate.evaluate`` (변경 없음, 임계값도 변경 없음)
    - ``simulation.realism.so101_realistic_control.RealisticControlLayer`` (매 physics step마다
      호출 - ``simulation/mujoco/live_web_viewer.py::_apply_realistic_control``와 동일하게
      ``simulated_actual``을 매 step 현재 qpos에서 새로 계산한다, 세션 시작 시 1회만 쓰는
      ``mujoco_shadow_backend.py``와 다르다 - 여기는 loop이기 때문)
    - ``simulation.mujoco.action_mapping`` (deg/percent -> rad 변환, 변경 없음)
    - ``simulation.mujoco.pick_drop_contacts`` / ``simulation.mujoco.pick_drop_eval`` (Track A/B 채점)

실물 팔로워에는 어떤 write도 하지 않는다 - 이 모듈은 ``mujoco.MjData``에만 쓴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY, JOINT_ORDER
from runtime.laptop.action_adapter import adapt_vla_action
from runtime.laptop.safety_gate import SafetyGate
from simulation.mujoco.action_mapping import build_default_mapping, map_positions_dict
from simulation.mujoco.pick_drop_contacts import cube_jaw_contact_active, get_cube_geom_id, get_jaw_geom_ids
from simulation.mujoco.pick_drop_eval import (
    PickDropEvalResult,
    ReferenceZones,
    StepRecord,
    evaluate_trajectory,
)
from simulation.mujoco.smolvla_chunk_runner import SmolVLAChunkRunner
from simulation.mujoco.so101_model import get_joint_limits
from simulation.realism.so101_control_profile import DEFAULT_PROFILE_PATH, load_control_profile
from simulation.realism.so101_realistic_control import RealisticControlConfig, RealisticControlLayer

DEFAULT_FPS = 30.0
DEFAULT_TASK = "Pick up the cube and drop it into the bin."


class PrimaryReplayError(RuntimeError):
    pass


@dataclass
class PrimaryReplayResult:
    scene_id: str
    dataset_root: str
    episode_index: int
    seed: int
    fps: float
    step_records: list[StepRecord]
    raw_command_log: list[dict[str, float]]  # Safety Gate 이전 (adapt_vla_action 출력)
    safe_command_log: list[dict[str, float]]  # mj_step에 실제 적용된 값
    chunk_boundary_frames: list[int]  # 이번 rollout이 실제 사용한 real frame index들
    ended_by_safety_reject: bool
    ended_reason: str  # "episode_exhausted" | "safety_reject" | "max_chunks_reached"
    eval_result: PickDropEvalResult
    real_follower_write_count: int = 0

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "dataset_root": self.dataset_root,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "fps": self.fps,
            "chunk_boundary_frames": self.chunk_boundary_frames,
            "ended_by_safety_reject": self.ended_by_safety_reject,
            "ended_reason": self.ended_reason,
            "real_follower_write_count": self.real_follower_write_count,
            "eval_result": self.eval_result.to_dict(),
            "step_count": len(self.step_records),
        }


def _tensor_to_joint_dict(tensor) -> dict[str, float]:
    flat = tensor.detach().to("cpu").reshape(-1)
    if flat.shape[0] != len(JOINT_ORDER):
        raise PrimaryReplayError(f"observation.state 텐서 차원이 {len(JOINT_ORDER)}이 아닙니다: shape={tuple(flat.shape)}")
    return {joint: float(flat[i]) for i, joint in enumerate(JOINT_ORDER)}


def _tensor_chw_to_numpy_hwc_uint8(tensor) -> np.ndarray:
    arr = tensor.detach().to("cpu").numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def load_real_episode_dataset(dataset_root: str | Path):
    """``LeRobotDataset`` 로딩 - ``scripts/evaluate_smolvla_midpoint.py``와 동일한 패턴
    (repo_id는 local root와 함께 쓰이는 라벨일 뿐, HF hub에서 받아오지 않는다)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_root)
    return LeRobotDataset(repo_id=f"local/{root.name}", root=str(root))


def run_primary_replay(
    *,
    chunk_runner: SmolVLAChunkRunner,
    model: mujoco.MjModel,
    safety_gate: SafetyGate,
    scene_id: str,
    dataset_root: str | Path,
    episode_index: int,
    zones: ReferenceZones,
    cube_xy: tuple[float, float],
    cube_z_init: float,
    seed: int,
    chunk_size: int,
    task: str = DEFAULT_TASK,
    fps: float = DEFAULT_FPS,
    max_chunks: int | None = None,
    on_step=None,  # optional callback(step_record, data) - visual mode console/live-viewer hook
) -> PrimaryReplayResult:
    """``chunk_size``는 호출자가 명시적으로 넘긴다 (``chunk_runner.policy.config.chunk_size``에서
    가져오는 것은 benchmark 드라이버의 책임) - 이 함수를 fake chunk runner로 테스트할 때 실제
    LeRobot policy 객체 구조에 의존하지 않게 하기 위함."""
    dataset = load_real_episode_dataset(dataset_root)
    if episode_index >= dataset.num_episodes:
        raise PrimaryReplayError(f"{dataset_root}: episode_index={episode_index} >= num_episodes={dataset.num_episodes}")
    ep_meta = dataset.meta.episodes[episode_index]
    length = int(ep_meta["length"])
    base = int(ep_meta["dataset_from_index"])

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
        lo_deg, hi_deg = math.degrees(lo), math.degrees(hi)
        return min(max(value_deg, lo_deg), hi_deg)

    def _sync_robot_to_state(state_deg: dict[str, float]) -> None:
        for j in JOINT_ORDER:
            clipped = _clip_to_range(j, state_deg[j])
            target_rad = math.radians(clipped)
            data.qpos[qpos_adr[j]] = target_rad
            data.ctrl[actuator_id[j]] = target_rad
        mujoco.mj_forward(model, data)

    # cube 초기 위치 (1회) - resync은 로봇 팔에만 적용되고, cube는 이후 물리적으로 그대로 이어진다
    # (모듈 docstring 참고 - 실제 그 시점 세계 상태를 흉내내는 게 아니라, "지금까지 우리 chunk
    # 실행이 cube에 실제로 한 일"을 그대로 보존해야 하기 때문).
    data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = [cube_xy[0], cube_xy[1], cube_z_init]
    data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    profile = load_control_profile(DEFAULT_PROFILE_PATH)
    control_layer = RealisticControlLayer(profile, RealisticControlConfig())

    chunk_runner.reset()
    episode_seed = seed + episode_index

    step_records: list[StepRecord] = []
    raw_log: list[dict[str, float]] = []
    safe_log: list[dict[str, float]] = []
    chunk_boundary_frames: list[int] = []
    ended_by_safety_reject = False
    ended_reason = "episode_exhausted"
    steps_per_frame = max(1, round((1.0 / fps) / model.opt.timestep))
    global_step = 0
    used_seed_once = False

    frame_local = 0
    chunk_idx = 0
    while frame_local < length:
        if max_chunks is not None and chunk_idx >= max_chunks:
            ended_reason = "max_chunks_reached"
            break

        row = base + frame_local
        sample = dataset[row]
        state_deg = _tensor_to_joint_dict(sample["observation.state"])
        images = {
            CAMERA_WORKSPACE_KEY: _tensor_chw_to_numpy_hwc_uint8(sample[CAMERA_WORKSPACE_KEY]),
            CAMERA_WRIST_KEY: _tensor_chw_to_numpy_hwc_uint8(sample[CAMERA_WRIST_KEY]),
        }

        _sync_robot_to_state(state_deg)
        control_layer.reset(initial_state=state_deg)

        chunk_seed = episode_seed if not used_seed_once else None
        used_seed_once = True
        chunk_result = chunk_runner.predict_chunk(state_deg=state_deg, images=images, task=task, seed=chunk_seed)
        chunk_boundary_frames.append(frame_local)

        for i, raw_action in enumerate(chunk_result.chunk_deg):
            adapted = adapt_vla_action(raw_action)
            raw_log.append(dict(adapted.command_deg) if adapted.valid else dict(raw_action))

            current_state_deg = {j: math.degrees(float(data.qpos[qpos_adr[j]])) for j in JOINT_ORDER}
            decision = safety_gate.evaluate(
                adapted_action=adapted, current_state_deg=current_state_deg, observation_valid=True
            )

            if decision.decision == "REJECT":
                rec = StepRecord(
                    step=global_step, sim_time=global_step / fps,
                    ee_pos=tuple(data.site_xpos[gripperframe_id]), cube_pos=tuple(data.xpos[cube_body_id]),
                    gripper_cmd_percent=current_state_deg["gripper"],
                    gripper_raw_percent=adapted.command_deg.get("gripper", current_state_deg["gripper"]) if adapted.valid else current_state_deg["gripper"],
                    cube_jaw_contact=cube_jaw_contact_active(model, data, cube_geom_id=cube_geom_id, jaw_geom_ids=jaw_geom_ids),
                    safety_decision="REJECT", chunk_boundary=(i == 0),
                    qpos_deg=dict(current_state_deg), cube_quat=tuple(data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7]),
                )
                step_records.append(rec)
                safe_log.append(current_state_deg)
                if on_step is not None:
                    on_step(rec, data)
                ended_by_safety_reject = True
                ended_reason = "safety_reject"
                break

            desired_native = dict(decision.safe_action)
            simulated_actual_native = current_state_deg
            # 실제 wall-clock(time.monotonic())이 아니라 시뮬레이션 시계(global_step/fps)를 쓴다 -
            # 이 루프는 실시간 페이싱 없이 최대한 빠르게 도는데(live_web_viewer.py와 달리 매
            # step마다 실제 33ms를 기다리지 않음), RealisticControlLayer의 latency queue는
            # "now"가 실제로 몇 초 지났는지에 의존한다. wall-clock을 쓰면 여러 step이 실제로는
            # 수 ms 안에 몰아서 실행되어 latency 로직이 "거의 시간이 안 지났다"고 착각해 명령을
            # 계속 지연시키는 버그가 있었다 (sanity rollout에서 팔이 멈춘 것으로 발견) - 시뮬레이션
            # 시계를 쓰면 매 step이 항상 정확히 1/fps초씩 지난 것으로 취급되어 이 문제가 없다.
            processed = control_layer.process(desired_native, now=global_step / fps, simulated_actual=simulated_actual_native)
            processed_rad = map_positions_dict(processed.processed_action, mapping)
            for j in JOINT_ORDER:
                data.ctrl[actuator_id[j]] = processed_rad[j]

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            contact = cube_jaw_contact_active(model, data, cube_geom_id=cube_geom_id, jaw_geom_ids=jaw_geom_ids)
            applied_deg = {j: math.degrees(float(data.qpos[qpos_adr[j]])) for j in JOINT_ORDER}
            safe_log.append(dict(decision.safe_action))
            rec = StepRecord(
                step=global_step, sim_time=global_step / fps,
                ee_pos=tuple(data.site_xpos[gripperframe_id]), cube_pos=tuple(data.xpos[cube_body_id]),
                gripper_cmd_percent=decision.safe_action["gripper"],
                gripper_raw_percent=adapted.command_deg.get("gripper", applied_deg["gripper"]) if adapted.valid else applied_deg["gripper"],
                cube_jaw_contact=contact, safety_decision=decision.decision, chunk_boundary=(i == 0),
                qpos_deg=dict(applied_deg), cube_quat=tuple(data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7]),
            )
            step_records.append(rec)
            if on_step is not None:
                on_step(rec, data)
            global_step += 1

        if ended_by_safety_reject:
            break
        frame_local += chunk_size
        chunk_idx += 1

    eval_result = evaluate_trajectory(
        step_records, zones, per_step_command_deg=safe_log,
        ended_by_safety_reject=ended_by_safety_reject,
        ended_by_timeout=(not ended_by_safety_reject),
    )

    return PrimaryReplayResult(
        scene_id=scene_id, dataset_root=str(dataset_root), episode_index=episode_index, seed=seed, fps=fps,
        step_records=step_records, raw_command_log=raw_log, safe_command_log=safe_log,
        chunk_boundary_frames=chunk_boundary_frames, ended_by_safety_reject=ended_by_safety_reject,
        ended_reason=ended_reason, eval_result=eval_result, real_follower_write_count=0,
    )
