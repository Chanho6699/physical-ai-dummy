"""safe_action -> 기존 Realistic MuJoCo Control Layer -> 기존 MuJoCo executor.

이 모듈은 새 MuJoCo 실행기를 만들지 않는다. ``simulation/realism/so101_realistic_control.py``의
``RealisticControlLayer``와 ``simulation/mujoco/action_mapping.py``/``so101_model.py``를
그대로 재사용한다 (섹션 13 요구사항 - 중복 구현 금지).

세션 시작 시 실물 팔로워의 현재 state로 MuJoCo 초기 qpos/ctrl을 한 번만 맞춘다
(섹션 12 - "매 step 실제 state로 MuJoCo를 계속 덮어쓰지 않는다"). gripper도 나머지
5관절과 동일하게 ``action_mapping``의 기존 scale(pi/180)로 변환한다 - 이 프로젝트가
이미 확립한 관례("gripper percent를 degree 스케일로 재사용")를 그대로 따른다
(``simulation/mujoco/live_web_viewer.py``의 ``_apply_realistic_control`` 주석 참고,
새 스케일을 만들지 않는다).

이 backend는 실물 SO-101에 어떤 write도 하지 않는다 - MuJoCo(``mujoco.MjData``)에만
쓴다.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

from runtime.common.vla_contract import JOINT_ORDER
from simulation.mujoco.action_mapping import (
    ActionMappingError,
    JointMapping,
    build_default_mapping,
    map_positions_dict,
    validate_mapping_against_model,
)
from simulation.mujoco.so101_model import SO101ModelError, get_joint_limits, load_model, make_data
from simulation.realism.so101_control_profile import (
    DEFAULT_PROFILE_PATH,
    ControlProfile,
    ControlProfileError,
    load_control_profile,
)
from simulation.realism.so101_realistic_control import (
    RealisticControlConfig,
    RealisticControlLayer,
    compute_tracking_error,
)

DEFAULT_FPS = 30.0


class MuJoCoBackendError(RuntimeError):
    """preflight/실행 준비 실패 (모델/profile 로딩, mapping 검증 등)."""


@dataclass(frozen=True)
class MuJoCoStepResult:
    initial_state_deg: dict[str, float]
    processed_command_deg: dict[str, float]  # RealisticControlLayer 통과 후 (safe_action과 다를 수 있음)
    final_state_deg: dict[str, float]
    final_state_finite: bool
    joint_limits_deg: dict[str, tuple[float, float]]
    mechanical_violations: tuple[str, ...]  # final_state가 joint_limits_deg를 벗어난 관절
    tracking_error_deg: dict[str, float]
    max_abs_qvel: float
    realistic_diagnostics: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "initial_state_deg": self.initial_state_deg,
            "processed_command_deg": self.processed_command_deg,
            "final_state_deg": self.final_state_deg,
            "final_state_finite": self.final_state_finite,
            "joint_limits_deg": {k: list(v) for k, v in self.joint_limits_deg.items()},
            "mechanical_violations": list(self.mechanical_violations),
            "tracking_error_deg": self.tracking_error_deg,
            "max_abs_qvel": self.max_abs_qvel,
            "realistic_layer_diagnostics": self.realistic_diagnostics,
        }


class RealisticMuJoCoBackend:
    """single-step Shadow 실행 전용 - continuous rollout API가 아니다 (섹션 11)."""

    def __init__(
        self,
        *,
        scene_path: str | Path | None = None,
        profile_path: str | Path | None = None,
        control_config: RealisticControlConfig | None = None,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.scene_path = scene_path
        self.profile_path = profile_path
        self.control_config = control_config or RealisticControlConfig()
        self.fps = fps

        self._model = None
        self._data = None
        self._mapping: tuple[JointMapping, ...] | None = None
        self._profile: ControlProfile | None = None
        self._control_layer: RealisticControlLayer | None = None
        self._joint_limits_deg: dict[str, tuple[float, float]] = {}
        self._initial_state_deg: dict[str, float] | None = None

    def preflight(self) -> None:
        try:
            model = load_model(self.scene_path)
            mapping = build_default_mapping([f"{j}.pos" for j in JOINT_ORDER])
            validate_mapping_against_model(mapping, model)
        except (SO101ModelError, ActionMappingError) as exc:
            raise MuJoCoBackendError(f"MuJoCo 모델/매핑 준비 실패: {exc}") from exc

        try:
            profile = load_control_profile(self.profile_path or DEFAULT_PROFILE_PATH)
        except ControlProfileError as exc:
            raise MuJoCoBackendError(f"Realistic control profile 로딩 실패: {exc}") from exc

        self._model = model
        self._data = make_data(model)
        self._mapping = mapping
        self._profile = profile
        self._control_layer = RealisticControlLayer(profile, self.control_config)
        limits = get_joint_limits(model)
        self._joint_limits_deg = {
            name: (math.degrees(lim.joint_range[0]), math.degrees(lim.joint_range[1])) for name, lim in limits.items()
        }

    @property
    def joint_limits_deg(self) -> dict[str, tuple[float, float]]:
        return dict(self._joint_limits_deg)

    def sync_initial_state(self, state_deg: dict[str, float]) -> None:
        """실물 팔로워 현재 state -> MuJoCo 초기 qpos/ctrl (섹션 12, 세션 시작 시 1회)."""
        import mujoco

        if self._model is None or self._data is None or self._mapping is None or self._control_layer is None:
            raise MuJoCoBackendError("preflight()를 먼저 호출해야 합니다.")

        missing = [entry.dataset_name for entry in self._mapping if entry.dataset_name not in state_deg]
        if missing:
            raise MuJoCoBackendError(f"초기 state에 관절이 없습니다: {missing}")

        mujoco.mj_resetData(self._model, self._data)
        target_rad = map_positions_dict(state_deg, self._mapping)
        for entry in self._mapping:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
            qpos_adr = self._model.jnt_qposadr[joint_id]
            self._data.qpos[qpos_adr] = target_rad[entry.mujoco_actuator_name]
            actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, entry.mujoco_actuator_name)
            self._data.ctrl[actuator_id] = target_rad[entry.mujoco_actuator_name]
        mujoco.mj_forward(self._model, self._data)

        self._control_layer.reset(initial_state=state_deg)
        self._initial_state_deg = {entry.dataset_name: float(state_deg[entry.dataset_name]) for entry in self._mapping}

    def execute_single_step(self, safe_action_deg: dict[str, float], *, now: float) -> MuJoCoStepResult:
        import mujoco
        import numpy as np

        if self._model is None or self._data is None or self._mapping is None or self._control_layer is None:
            raise MuJoCoBackendError("preflight()를 먼저 호출해야 합니다.")
        if self._initial_state_deg is None:
            raise MuJoCoBackendError("sync_initial_state()를 먼저 호출해야 합니다.")

        model, data, mapping = self._model, self._data, self._mapping

        result = self._control_layer.process(safe_action_deg, now=now, simulated_actual=self._initial_state_deg)
        processed_rad = map_positions_dict(result.processed_action, mapping)

        for entry in mapping:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, entry.mujoco_actuator_name)
            data.ctrl[actuator_id] = processed_rad[entry.mujoco_actuator_name]

        steps_per_frame = max(1, round((1.0 / self.fps) / model.opt.timestep))
        max_abs_qvel = 0.0
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
            if np.isfinite(data.qvel).all():
                max_abs_qvel = max(max_abs_qvel, float(np.abs(data.qvel).max()))

        final_state_finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        final_state_deg: dict[str, float] = {}
        for entry in mapping:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
            qpos_adr = model.jnt_qposadr[joint_id]
            value = float(data.qpos[qpos_adr])
            final_state_deg[entry.dataset_name] = math.degrees(value) if math.isfinite(value) else float("nan")

        mechanical_violations: list[str] = []
        if final_state_finite:
            for joint, value in final_state_deg.items():
                lo, hi = self._joint_limits_deg.get(joint, (float("-inf"), float("inf")))
                if value < lo or value > hi:
                    mechanical_violations.append(f"{joint}: {value:.2f}deg (범위 [{lo:.2f}, {hi:.2f}])")

        tracking_error_deg = (
            compute_tracking_error(result.processed_action, final_state_deg) if final_state_finite else {}
        )
        realistic_diagnostics = {joint: dataclasses.asdict(diag) for joint, diag in result.diagnostics.items()}

        return MuJoCoStepResult(
            initial_state_deg=dict(self._initial_state_deg),
            processed_command_deg=dict(result.processed_action),
            final_state_deg=final_state_deg,
            final_state_finite=final_state_finite,
            joint_limits_deg=dict(self._joint_limits_deg),
            mechanical_violations=tuple(mechanical_violations),
            tracking_error_deg=tracking_error_deg,
            max_abs_qvel=max_abs_qvel,
            realistic_diagnostics=realistic_diagnostics,
        )
