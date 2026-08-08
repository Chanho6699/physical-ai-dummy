"""Action Adapter 이후, Realistic MuJoCo 실행 이전의 Safety Gate.

결과는 항상 ``ACCEPT`` / ``WOULD_CLAMP`` / ``REJECT`` 셋 중 하나다 (섹션 9).

두 임계값의 출처를 의도적으로 분리한다 (2026-08 Grid35 실험 감사 근거 - 아래 "출처 분리"
절 참고):

    - 관절 mechanical/calibration limit (하드웨어 불변): ``configs/follower_safe_mapper.yaml``의
      ``fallback_raw_range``(또는 캘리브레이션 파일)를 ``simulation/mujoco/follower_safe_mapper.py``의
      ``load_follower_calibration``으로 그대로 로딩한다 - 몸통 5관절은 degree 범위,
      gripper는 이 프로젝트 전역 관례상 항상 percent_0_100([0, 100])이다
      (``hardware/state_server/readonly_so101_reader.py`` 확인 - follower_safe_mapper의
      UNVERIFIED_RANGE 취급과 달리, 여기서는 gripper 단위 자체는 확정적으로 알려져 있으므로
      [0, 100]을 그대로 mechanical bound로 쓴다). 실제 캘리브레이션 파일을 읽었는지
      fallback을 썼는지는 관절별로 ``joint_range_source``에 남는다(``source_summary()``).
    - 1-step 급격한 변화(excessive action step): ``configs/safety_gate.yaml``에서 읽는다.
      **이전에는** ``configs/mujoco_so101.yaml``의 ``safety.max_joint_delta_per_frame``
      (``data/so101_cube_xy_train_v1`` 20-episode 데이터셋에서 뽑은 WARN 임계값)을 그대로
      빌려 썼다 - 이 값을 새 checkpoint/dataset(예: Grid35)에 그대로 재사용하면, 그
      데이터셋의 정상적인 시연 속도가 새 실험과 다를 때 실제 시연조차 대량으로
      WOULD_CLAMP되는 문제가 있었다(``configs/safety_gate.yaml`` 상단 주석에 실측 근거
      기록). 그래서 이 값의 소유권을 MuJoCo action-replay 진단 도구(``mujoco_so101.yaml``)와
      완전히 분리된 ``configs/safety_gate.yaml``로 옮겼다 - 숫자 자체는 (다른 dataset
      통계로 다시 뽑지 않고) 그대로 유지했지만, 더 이상 "그 오래된 dataset을 검증하려고
      만든 진단 도구 설정"에 암묵적으로 얹혀가지 않는다.

``configs/generated/``의 CANDIDATE_ONLY 관측 프로파일 파일(``simulation/realism/so101_control_profile.py``의
``DEFAULT_PROFILE_PATH``)은 여기서 전혀 읽지 않는다 - 그 파일은 Realistic MuJoCo Layer(진단용)에서만 쓰인다.

REJECT vs WOULD_CLAMP 구분 원칙: 한계를 "약간" 벗어나면(clamp로 안전하게 흡수 가능한
수준) WOULD_CLAMP, 한계를 "터무니없이" 벗어나면(단위/스케일 자체가 잘못됐을 가능성이 높음)
REJECT한다 - clamp로 억지로 살리지 않는다 (섹션 9 요구사항). "터무니없이"의 기준은
``GROSS_RANGE_MULTIPLIER``/``GROSS_STEP_MULTIPLIER``로 명시한다.

# hardware-invariant Safety vs policy/model-quality diagnostic (구조적 분리 원칙)

mechanical range(A)는 실험/checkpoint가 바뀌어도 절대 흔들리지 않아야 하는 하드웨어
불변 값이다 - 그래서 dataset 통계가 아니라 calibration/fallback에서만 온다.
excessive-step(B)은 "특정 checkpoint의 예측이 비정상적으로 크다"를 걸러내려는
policy-dynamics 안전장치이지만, 실물 서보의 검증된 rate spec이 아직 없어 지금은 여전히
경험적 placeholder다(``configs/safety_gate.yaml`` 참고) - 이 둘을 같은 값/같은 파일로
섞지 않는 것이 이 모듈의 핵심 원칙이다. "이 checkpoint의 예측이 그 학습 dataset 자체의
시연 분포와 비교해 얼마나 튀는가"(model-quality diagnostic, 예: 학습/held-out dataset의
state->action delta 분포)는 이 Safety Gate가 판정하지 않는다 - 그런 진단은
``scripts/evaluate_smolvla_midpoint.py``의 오프라인 metric(예: ``gt_demo_state_delta``)이
담당하고, 여기서는 ACCEPT/WOULD_CLAMP/REJECT라는 하드웨어 보호 판정만 내린다.
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
# Safety Gate 전용 설정 - configs/mujoco_so101.yaml(MuJoCo action-replay 진단 도구, 다른
# 목적/다른 dataset에 종속된 설정)과는 더 이상 공유하지 않는다.
DEFAULT_SAFETY_GATE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "safety_gate.yaml"

# 캘리브레이션 출처 라벨 (SafetyGateConfig.joint_range_source 값들).
RANGE_SOURCE_CALIBRATION_FILE = "calibration_file"
RANGE_SOURCE_FALLBACK_CONFIG = "fallback_config"
RANGE_SOURCE_GRIPPER_CONVENTION = "gripper_percent_convention"  # follower_safe_mapper를 거치지 않는 고정 관례


class SafetyGateConfigError(RuntimeError):
    """Safety Gate 설정 로딩 실패."""


def load_excessive_step_config(path: str | Path) -> dict[str, float]:
    """``configs/safety_gate.yaml``의 ``excessive_step_deg`` 섹션만 읽는다 (관절 6개 필수)."""
    import yaml

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise SafetyGateConfigError(f"safety_gate 설정 파일을 찾을 수 없습니다: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SafetyGateConfigError(f"{resolved} 파싱 실패: {exc}") from exc

    excessive = raw.get("excessive_step_deg") if isinstance(raw, dict) else None
    if not isinstance(excessive, dict):
        raise SafetyGateConfigError(f"{resolved}에 excessive_step_deg 섹션이 없습니다.")
    missing = [j for j in JOINT_ORDER if j not in excessive]
    if missing:
        raise SafetyGateConfigError(f"{resolved}의 excessive_step_deg에 관절이 누락되었습니다: {missing}")
    return {joint: float(excessive[joint]) for joint in JOINT_ORDER}


@dataclass(frozen=True)
class SafetyGateConfig:
    """관절별 mechanical range(degree/percent) + excessive-step threshold(degree/percent).

    ``joint_range_source``/``calibration_file_path``/``excessive_step_config_path``는
    ``from_repo_defaults()``가 채우는 진단/출처 필드다 - 기존처럼 두 필드만 직접 넘겨
    구성하는 테스트/호출부(``SafetyGateConfig(joint_range_deg=..., max_step_deg=...)``)는
    기본값(빈 dict/None/False)으로 그대로 동작한다(하위 호환).
    """

    joint_range_deg: dict[str, tuple[float, float]]
    max_step_deg: dict[str, float]
    joint_range_source: dict[str, str] = field(default_factory=dict)
    calibration_file_path: str | None = None
    uses_calibration_fallback: bool = False
    excessive_step_config_path: str | None = None

    def source_summary(self) -> dict:
        """Shadow report 등에 넣기 위한 요약 - "real calibration을 썼는지 fallback을 썼는지"를
        명시적으로 드러낸다 (요구사항: Safety config source 기록)."""
        return {
            "joint_range_source": dict(self.joint_range_source),
            "uses_calibration_fallback": self.uses_calibration_fallback,
            "calibration_file_path": self.calibration_file_path,
            "excessive_step_config_path": self.excessive_step_config_path,
        }

    @staticmethod
    def from_repo_defaults(
        *,
        follower_safe_mapper_config_path: str | Path | None = None,
        safety_gate_config_path: str | Path | None = None,
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
        joint_range_source: dict[str, str] = {}
        for joint in JOINT_ORDER:
            if joint == GRIPPER_JOINT:
                joint_range_deg[joint] = GRIPPER_RANGE_DEG
                joint_range_source[joint] = RANGE_SOURCE_GRIPPER_CONVENTION
                continue
            cal: FollowerJointCalibration = calibrations[joint]
            if not cal.verified or cal.range_min_deg is None or cal.range_max_deg is None:
                raise SafetyGateConfigError(f"'{joint}' 캘리브레이션 범위를 확인할 수 없습니다.")
            joint_range_deg[joint] = (cal.range_min_deg, cal.range_max_deg)
            # cal.source는 "calibration_file" | "fallback_config" 중 하나 (follower_safe_mapper.py).
            joint_range_source[joint] = cal.source

        uses_calibration_fallback = any(
            source == RANGE_SOURCE_FALLBACK_CONFIG for source in joint_range_source.values()
        )

        max_step_deg = load_excessive_step_config(safety_gate_config_path or DEFAULT_SAFETY_GATE_CONFIG_PATH)

        return SafetyGateConfig(
            joint_range_deg=joint_range_deg,
            max_step_deg=max_step_deg,
            joint_range_source=joint_range_source,
            calibration_file_path=(
                str(Path(mapper_config.calibration_file_path).expanduser())
                if mapper_config.calibration_file_path
                else None
            ),
            uses_calibration_fallback=uses_calibration_fallback,
            excessive_step_config_path=str(Path(safety_gate_config_path or DEFAULT_SAFETY_GATE_CONFIG_PATH).expanduser()),
        )


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

    @property
    def config(self) -> SafetyGateConfig:
        """호출부(Shadow report 등)가 ``source_summary()``를 꺼내 쓸 수 있게 노출한다."""
        return self._config

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
