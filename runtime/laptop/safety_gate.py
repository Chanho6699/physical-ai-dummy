"""Action Adapter 이후, Realistic MuJoCo 실행 이전의 Safety Gate.

결과는 항상 ``ACCEPT`` / ``WOULD_CLAMP`` / ``REJECT`` 셋 중 하나다 (섹션 9).

기존에 확립된 보수적 값을 그대로 재사용한다 (섹션 10 - control profile candidate의
p95/p99를 자동 hard limit으로 쓰지 않는다):

    - 관절 mechanical/calibration limit: ``configs/follower_safe_mapper.yaml``의
      ``fallback_raw_range``(또는 캘리브레이션 파일)를 ``simulation/mujoco/follower_safe_mapper.py``의
      ``load_follower_calibration``으로 그대로 로딩한다 - 몸통 5관절은 degree 범위,
      gripper는 이 프로젝트 전역 관례상 항상 percent_0_100([0, 100])이다
      (``hardware/state_server/readonly_so101_reader.py`` 확인 - follower_safe_mapper의
      UNVERIFIED_RANGE 취급과 달리, 여기서는 gripper 단위 자체는 확정적으로 알려져 있으므로
      [0, 100]을 그대로 mechanical bound로 쓴다).
    - 1-step 급격한 변화(excessive action step): ``configs/mujoco_so101.yaml``의
      ``safety.max_joint_delta_per_frame``(라디안, 실측 20-episode 데이터의 프레임간 최댓값에
      여유를 둔 기존 WARN 임계값)을 degree로 환산해 재사용한다.

``configs/generated/``의 CANDIDATE_ONLY 관측 프로파일 파일(``simulation/realism/so101_control_profile.py``의
``DEFAULT_PROFILE_PATH``)은 여기서 전혀 읽지 않는다 - 그 파일은 Realistic MuJoCo Layer(진단용)에서만 쓰인다.

REJECT vs WOULD_CLAMP 구분 원칙: 한계를 "약간" 벗어나면(clamp로 안전하게 흡수 가능한
수준) WOULD_CLAMP, 한계를 "터무니없이" 벗어나면(단위/스케일 자체가 잘못됐을 가능성이 높음)
REJECT한다 - clamp로 억지로 살리지 않는다 (섹션 9 요구사항). "터무니없이"의 기준은
``GROSS_RANGE_MULTIPLIER``/``GROSS_STEP_MULTIPLIER``로 명시한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from runtime.common.vla_contract import GRIPPER_JOINT, JOINT_ORDER
from runtime.laptop.action_adapter import AdaptedAction
from simulation.mujoco.follower_safe_mapper import (
    FollowerJointCalibration,
    FollowerSafeMapperError,
    load_config_yaml as load_follower_safe_mapper_config_yaml,
    load_follower_calibration,
)
from simulation.mujoco.safety_checks import SafetyConfigError, load_safety_config

Decision = Literal["ACCEPT", "WOULD_CLAMP", "REJECT"]

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

GRIPPER_RANGE_DEG = (0.0, 100.0)  # 실제로는 percent, readonly_so101_reader.py 관례 재사용

# 한계를 "터무니없이" 벗어났다고 볼 배수 - clamp가 아니라 REJECT로 처리한다.
GROSS_RANGE_MULTIPLIER = 3.0
GROSS_STEP_MULTIPLIER = 5.0

DEFAULT_FOLLOWER_SAFE_MAPPER_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "follower_safe_mapper.yaml"
)
DEFAULT_MUJOCO_SAFETY_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "mujoco_so101.yaml"


class SafetyGateConfigError(RuntimeError):
    """Safety Gate 설정 로딩 실패."""


@dataclass(frozen=True)
class SafetyGateConfig:
    """관절별 mechanical range(degree/percent) + excessive-step threshold(degree/percent)."""

    joint_range_deg: dict[str, tuple[float, float]]
    max_step_deg: dict[str, float]

    @staticmethod
    def from_repo_defaults(
        *,
        follower_safe_mapper_config_path: str | Path | None = None,
        mujoco_safety_config_path: str | Path | None = None,
    ) -> "SafetyGateConfig":
        try:
            mapper_config = load_follower_safe_mapper_config_yaml(
                follower_safe_mapper_config_path or DEFAULT_FOLLOWER_SAFE_MAPPER_CONFIG_PATH
            )
            calibrations = load_follower_calibration(
                calibration_file_path=mapper_config.calibration_file_path,
                fallback_raw_range=mapper_config.fallback_raw_range,
                motor_resolution=mapper_config.motor_resolution,
            )
        except FollowerSafeMapperError as exc:
            raise SafetyGateConfigError(f"follower_safe_mapper 설정/캘리브레이션 로딩 실패: {exc}") from exc

        joint_range_deg: dict[str, tuple[float, float]] = {}
        for joint in JOINT_ORDER:
            if joint == GRIPPER_JOINT:
                joint_range_deg[joint] = GRIPPER_RANGE_DEG
                continue
            cal: FollowerJointCalibration = calibrations[joint]
            if not cal.verified or cal.range_min_deg is None or cal.range_max_deg is None:
                raise SafetyGateConfigError(f"'{joint}' 캘리브레이션 범위를 확인할 수 없습니다.")
            joint_range_deg[joint] = (cal.range_min_deg, cal.range_max_deg)

        try:
            mujoco_safety = load_safety_config(mujoco_safety_config_path or DEFAULT_MUJOCO_SAFETY_CONFIG_PATH)
        except SafetyConfigError as exc:
            raise SafetyGateConfigError(f"mujoco_so101.yaml 로딩 실패: {exc}") from exc

        max_step_deg = {joint: mujoco_safety.max_joint_delta_per_frame[joint] * RAD2DEG for joint in JOINT_ORDER}

        return SafetyGateConfig(joint_range_deg=joint_range_deg, max_step_deg=max_step_deg)


@dataclass(frozen=True)
class JointSafetyReport:
    joint: str
    raw_value: float | None
    safe_value: float | None
    clamped: bool
    rejected: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "joint": self.joint,
            "raw_value": self.raw_value,
            "safe_value": self.safe_value,
            "clamped": self.clamped,
            "rejected": self.rejected,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SafetyDecision:
    decision: Decision
    reasons: tuple[str, ...]
    would_clamp: bool
    safe_action: dict[str, float] | None
    per_joint: dict[str, JointSafetyReport]

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "would_clamp": self.would_clamp,
            "safe_action": self.safe_action,
            "per_joint": {j: r.to_dict() for j, r in self.per_joint.items()},
        }


class SafetyGate:
    def __init__(self, config: SafetyGateConfig) -> None:
        self._config = config

    def evaluate(
        self,
        *,
        adapted_action: AdaptedAction,
        current_state_deg: dict[str, float] | None,
        observation_valid: bool,
        observation_reasons: tuple[str, ...] = (),
        state_stale: bool = False,
        state_stale_reason: str | None = None,
    ) -> SafetyDecision:
        # -- 0. 입력 자체가 무효/stale이면 관절별 판정 없이 즉시 REJECT ------------------
        global_reasons: list[str] = []
        if not observation_valid:
            global_reasons.append("OBSERVATION_INVALID: " + "; ".join(observation_reasons) if observation_reasons else "OBSERVATION_INVALID")
        if state_stale:
            global_reasons.append(f"STATE_STALE: {state_stale_reason or '팔로워 state가 stale합니다.'}")
        if current_state_deg is None:
            global_reasons.append("STATE_MISSING: 팔로워 현재 state가 없습니다.")
        if not adapted_action.valid:
            global_reasons.append(f"ACTION_SCHEMA_INVALID: {adapted_action.invalid_reason}")

        if global_reasons:
            return SafetyDecision(
                decision="REJECT", reasons=tuple(global_reasons), would_clamp=False, safe_action=None, per_joint={}
            )

        assert current_state_deg is not None
        command = adapted_action.command_deg
        per_joint: dict[str, JointSafetyReport] = {}
        any_reject = False
        any_clamp = False

        for joint in JOINT_ORDER:
            raw_value = command[joint]
            current_value = current_state_deg.get(joint)
            lo, hi = self._config.joint_range_deg[joint]
            span = hi - lo
            max_step = self._config.max_step_deg[joint]

            reasons: list[str] = []
            rejected = False
            safe_value = raw_value

            # -- 1. mechanical/calibration range ------------------------------------
            if raw_value < lo - span * GROSS_RANGE_MULTIPLIER or raw_value > hi + span * GROSS_RANGE_MULTIPLIER:
                rejected = True
                reasons.append(
                    f"MECHANICAL_LIMIT_GROSS_VIOLATION: {raw_value:.2f} (범위 [{lo:.2f}, {hi:.2f}]를 크게 벗어남)"
                )
            elif raw_value < lo or raw_value > hi:
                safe_value = min(max(raw_value, lo), hi)
                reasons.append(f"MECHANICAL_LIMIT_CLAMPED: {raw_value:.2f} -> {safe_value:.2f} (범위 [{lo:.2f}, {hi:.2f}])")

            # -- 2. excessive single-step delta (현재 실물 state 대비) -----------------
            if not rejected and current_value is not None and math.isfinite(current_value):
                delta = safe_value - current_value
                if abs(delta) > max_step * GROSS_STEP_MULTIPLIER:
                    rejected = True
                    reasons.append(
                        f"EXCESSIVE_STEP_GROSS: |delta|={abs(delta):.2f}deg (임계값 {max_step:.2f}deg의 "
                        f"{GROSS_STEP_MULTIPLIER}배 초과)"
                    )
                elif abs(delta) > max_step:
                    clamped_value = current_value + math.copysign(max_step, delta)
                    reasons.append(
                        f"EXCESSIVE_STEP_CLAMPED: {safe_value:.2f} -> {clamped_value:.2f} (임계값 {max_step:.2f}deg/step)"
                    )
                    safe_value = clamped_value

            if rejected:
                any_reject = True
                safe_value = None
            elif reasons:
                any_clamp = True

            per_joint[joint] = JointSafetyReport(
                joint=joint,
                raw_value=raw_value,
                safe_value=safe_value,
                clamped=(not rejected and bool(reasons)),
                rejected=rejected,
                reasons=tuple(reasons),
            )

        if any_reject:
            reject_reasons = [r for rep in per_joint.values() if rep.rejected for r in rep.reasons]
            return SafetyDecision(
                decision="REJECT", reasons=tuple(reject_reasons), would_clamp=False, safe_action=None, per_joint=per_joint
            )

        safe_action = {j: per_joint[j].safe_value for j in JOINT_ORDER}
        if any_clamp:
            clamp_reasons = [r for rep in per_joint.values() for r in rep.reasons]
            return SafetyDecision(
                decision="WOULD_CLAMP", reasons=tuple(clamp_reasons), would_clamp=True, safe_action=safe_action, per_joint=per_joint
            )

        return SafetyDecision(decision="ACCEPT", reasons=(), would_clamp=False, safe_action=safe_action, per_joint=per_joint)
