"""Phase C-3A (+C-3A.1 correction): 30Hz VLA trajectory -> 60Hz(configurable) real-time
control target.

# 전체 파이프라인 (C-3A.1에서 Intent Validation 추가 - 순서를 절대 안 바꿈)

::

    TrajectoryBuffer snapshot (여러 겹치는 chunk, Phase C-1B)
      -> TemporalEnsembler.compute_target(chunks, T)          (Phase C-2, 재구현 안 함)
      -> Intent Validation (raw target 자체가 신뢰 가능한 policy 의도인가)  (신규, intent_validation.py)
      -> Motion Guard (신뢰된 target에 velocity/acceleration/jerk 순차 제한)  (motion_guard.py)
      -> Final SafetyGate.evaluate(guarded_target)             (기존 코드, 무수정, 반드시 이 순서)
      -> ACCEPT일 때만 향후(다음 Phase) write 후보

이 모듈 자체는 follower에 아무것도 쓰지 않는다 - ``ControlTargetResult``를 만들 뿐이다.

# 왜 Intent Validation이 추가됐는가 (C-3A.1 correction - 문제 1)

C-3A 최초 구현은 ``raw target -> Motion Guard -> Final SafetyGate`` 순서였다. 이
순서에서는 raw target이 아무리 위험한 policy outlier(예: 실물 사례 wrist_flex
53.67->33.67, delta 20deg)여도 Motion Guard가 그걸 매 tick 작은 조각으로 잘라버려서
Final SafetyGate가 60 tick 내내 전부 ACCEPT했다 - mechanical hard limit 우회는
아니었지만, "이 policy 예측을 신뢰할 수 있는가"라는 outlier 판정 자체를 우회한 것이었다
(자세한 분석은 ``motion_guard.py`` 모듈 docstring "[정정]" 절 참고). Intent Validation을
Motion Guard **이전**에 넣어서, 신뢰할 수 없는 raw target은 Motion Guard가 절대 보지
못하게 막는다.

# 30Hz -> 60Hz (섹션 4, 변경 없음)

``TemporalEnsembler``가 이미 절대시간 T 기준 chunk 내부 linear interpolation을 하므로,
이 모듈은 그 interpolation을 다시 구현하지 않는다 - 그냥 60Hz(또는 임의 control_hz)
tick마다 다른 T(``now + lookahead_s``)를 ``compute_target()``에 넣을 뿐이다.

# Motion Guard feedforward velocity를 위한 2차 샘플링 (C-3A.1 correction - 문제 2)

Motion Guard가 "policy trajectory 자체의 순간 속도"(feedforward)를 알려면 같은
trajectory를 ``T``와 ``T+dt`` 두 시점에서 샘플링해야 한다(``motion_guard.py``의
"문제 2 교정" 절 참고) - 그래서 이 모듈은 매 tick ``TemporalEnsembler.compute_target()``을
**두 번**(``T``, ``T+dt``) 호출한다. ``T+dt``가 모든 chunk의 horizon 밖이면(거의 없는
경우 - chunk horizon이 통상 1.6초인데 dt는 16.7ms) feedforward=0으로 안전하게 degrade한다
(순수 correction만 적용).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.intent_validation import IntentValidationResult, PolicyIntentValidator
from runtime.laptop.motion_guard import (
    DEFAULT_CORRECTION_TIME_CONSTANT_S,
    DEFAULT_JOINT_MOTION_LIMITS,
    INITIAL_GUARD_STATE,
    GuardState,
    JointMotionLimits,
    MotionGuardError,
    apply_joint_motion_guard,
)
from runtime.laptop.safety_gate import SafetyGate
from runtime.laptop.temporal_ensemble import TemporalEnsembler
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

# chunks 자체가 비어있음(worker가 아직 아무 chunk도 만들지 못함 - 시작 직후 등).
STOP_REASON_NO_TARGET = "NO_TARGET"
# chunks는 있지만 이번 tick의 target_time을 커버하는 게 하나도 없음(worker가 뒤처짐/
# stall).
STOP_REASON_STALE_TRAJECTORY = "STALE_TRAJECTORY"
# Intent Validation이 raw target을 신뢰할 수 없다고 판정 - 뒤에 실제 decision이 붙는다
# (예: "INTENT_WOULD_CLAMP", "INTENT_REJECT"). Motion Guard는 이 tick에 대해 호출되지 않는다.
STOP_REASON_INTENT_PREFIX = "INTENT_"
# raw target/현재 state가 NaN/Inf이거나 dt<=0 - motion guard 계산 자체가 불가능.
STOP_REASON_GUARD_INVALID = "GUARD_INVALID_INPUT"
# Final SafetyGate가 guarded target을 ACCEPT하지 않음 - 뒤에 실제 decision이 붙는다
# (예: "SAFETY_WOULD_CLAMP", "SAFETY_REJECT").
STOP_REASON_SAFETY_PREFIX = "SAFETY_"


@dataclass(frozen=True)
class ControlTargetResult:
    """한 tick의 전체 계산 결과 - 각 단계 산출물을 전부 보존해 Intent Validation/Motion
    Guard가 각각 무엇을 했는지 추적 가능하게 한다."""

    target_time_monotonic: float
    raw_ensemble_target: dict[str, float] | None  # Temporal Ensemble 직후 (Intent/Guard 적용 전)
    intent_decision: str | None  # "ACCEPT" | "WOULD_CLAMP" | "REJECT" | None(target 계산 자체가 실패)
    intent_reasons: tuple[str, ...]
    smoothed_target: dict[str, float] | None  # Motion Guard 적용 후 (guarded_target과 동일 - 아래 참고)
    guarded_target: dict[str, float] | None  # MotionGuard 출력 / Final Safety 입력
    final_target: dict[str, float] | None  # Final Safety ACCEPT 또는 soft mechanical saturation 출력
    clamp_reasons: tuple[str, ...]
    safety_decision: str | None  # Final SafetyGate 판정. Intent에서 막히면 None(도달 안 함).
    safety_reasons: tuple[str, ...]
    contributing_sequences: tuple[int, ...]
    target_valid: bool  # True == intent_decision==ACCEPT AND safety_decision==ACCEPT
    stop_reason: str | None  # target_valid=False일 때만 채워짐

    def to_dict(self) -> dict:
        return {
            "target_time_monotonic": self.target_time_monotonic,
            "raw_ensemble_target": dict(self.raw_ensemble_target) if self.raw_ensemble_target else None,
            "intent_decision": self.intent_decision,
            "intent_reasons": list(self.intent_reasons),
            "smoothed_target": dict(self.smoothed_target) if self.smoothed_target else None,
            "guarded_target": dict(self.guarded_target) if self.guarded_target else None,
            "final_target": dict(self.final_target) if self.final_target else None,
            "clamp_reasons": list(self.clamp_reasons),
            "safety_decision": self.safety_decision,
            "safety_reasons": list(self.safety_reasons),
            "contributing_sequences": list(self.contributing_sequences),
            "target_valid": self.target_valid,
            "stop_reason": self.stop_reason,
        }


def _early_result(
    *, target_time: float, stop_reason: str, raw_target: dict[str, float] | None = None,
    contributing_sequences: tuple[int, ...] = (),
    intent_decision: str | None = None, intent_reasons: tuple[str, ...] = (),
) -> ControlTargetResult:
    return ControlTargetResult(
        target_time_monotonic=target_time, raw_ensemble_target=raw_target,
        intent_decision=intent_decision, intent_reasons=intent_reasons,
        smoothed_target=None, guarded_target=None, final_target=None, clamp_reasons=(),
        safety_decision=None, safety_reasons=(),
        contributing_sequences=contributing_sequences, target_valid=False, stop_reason=stop_reason,
    )


class RealTimeControlTargetGenerator:
    """``tick()`` 호출마다 결정적으로 하나의 ``ControlTargetResult``를 만든다. 아직 어떤
    background thread/스케줄러도 없다 - 실제 60Hz 루프는 향후 phase에서 이 클래스를
    주기적으로 호출하게 될 것이다.

    ``current_follower_state_deg``는 **매 tick마다** motion guard의 위치 기준점으로
    다시 쓰인다(``motion_guard.py`` docstring "Position 앵커링 설계" 참고) - 이 클래스
    자체는 이전 tick에서 "얼마나 움직였다고 생각하는지"를 position으로 기억하지 않는다.
    """

    def __init__(
        self,
        *,
        ensembler: TemporalEnsembler | None = None,
        safety_gate: SafetyGate,
        motion_limits: dict[str, JointMotionLimits] | None = None,
        control_hz: float = 60.0,
        lookahead_s: float = 0.0,
        correction_time_constant_s: float = DEFAULT_CORRECTION_TIME_CONSTANT_S,
    ) -> None:
        if control_hz <= 0:
            raise ValueError(f"control_hz는 양수여야 합니다: {control_hz}")
        self._ensembler = ensembler or TemporalEnsembler(lookahead_s=lookahead_s)
        self._safety_gate = safety_gate
        self._intent_validator = PolicyIntentValidator(safety_gate)  # 같은 SafetyGate 인스턴스 재사용
        self._motion_limits = motion_limits or DEFAULT_JOINT_MOTION_LIMITS
        missing = [j for j in JOINT_ORDER if j not in self._motion_limits]
        if missing:
            raise ValueError(f"motion_limits에 관절이 누락되었습니다: {missing}")
        self._control_hz = control_hz
        self._period_s = 1.0 / control_hz
        self._lookahead_s = lookahead_s
        self._correction_time_constant_s = correction_time_constant_s
        self._guard_states: dict[str, GuardState] = {}
        self._last_tick_time_monotonic: float | None = None

    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def period_s(self) -> float:
        return self._period_s

    def reset_guard_state(self) -> None:
        """Motion guard의 velocity/acceleration 이력을 전부 지운다 - 다음 ``tick()``은
        마치 첫 tick인 것처럼 bootstrap(velocity=0, acceleration=0)부터 다시 시작한다.
        ``_last_tick_time_monotonic``도 함께 지워서 다음 tick의 dt가 nominal
        ``period_s``로 계산되게 한다(변칙적인 큰 dt가 계산되지 않도록)."""
        self._guard_states = {}
        self._last_tick_time_monotonic = None
        self._intent_validator.reset_history()

    def tick(
        self,
        *,
        chunks: Sequence[TimestampedActionChunk],
        now_monotonic: float,
        current_follower_state_deg: dict[str, float],
    ) -> ControlTargetResult:
        target_time = now_monotonic + self._lookahead_s

        # -- 1. Stale trajectory fail-safe - 절대 추측/extrapolate 안 함 -------------------
        if not chunks:
            self._intent_validator.reset_history()
            return _early_result(target_time=target_time, stop_reason=STOP_REASON_NO_TARGET)

        ensembled = self._ensembler.compute_target(chunks, target_time)
        if ensembled is None:
            self._intent_validator.reset_history()
            return _early_result(target_time=target_time, stop_reason=STOP_REASON_STALE_TRAJECTORY)

        raw_target = ensembled.action

        # -- 2. Intent Validation (C-3A.1 신규) - Motion Guard보다 먼저, raw target에 --------
        intent: IntentValidationResult = self._intent_validator.check_intent(
            raw_target_deg=raw_target, current_state_deg=current_follower_state_deg,
        )
        if not intent.valid:
            return _early_result(
                target_time=target_time, stop_reason=f"{STOP_REASON_INTENT_PREFIX}{intent.decision}",
                raw_target=raw_target, contributing_sequences=ensembled.contributing_sequences,
                intent_decision=intent.decision, intent_reasons=intent.reasons,
            )

        # -- 3. dt 계산 ("variable dt handling") ------------------------------------------
        if self._last_tick_time_monotonic is None:
            dt_s = self._period_s  # 이력 없는 첫 tick(또는 reset 직후) - nominal period 사용
        else:
            dt_s = now_monotonic - self._last_tick_time_monotonic

        # -- 4. Motion Guard의 feedforward velocity용 T+dt 샘플 (C-3A.1 신규) --------------
        # 실패(None)해도 fail-closed로 전체를 막지 않는다 - feedforward=0(순수 correction)으로
        # 안전하게 degrade한다. chunk horizon(~1.6s)이 dt(~16.7ms)보다 훨씬 크므로 정상
        # 상황에서는 거의 항상 성공한다.
        ensembled_lookahead = self._ensembler.compute_target(chunks, target_time + dt_s) if dt_s > 0 else None
        target_lookahead = ensembled_lookahead.action if ensembled_lookahead is not None else raw_target

        # -- 5. Motion Guard (관절별, intent가 승인한 target에만 적용) ---------------------
        guarded: dict[str, float] = {}
        new_guard_states: dict[str, GuardState] = {}
        try:
            if dt_s <= 0 or not math.isfinite(dt_s):
                raise MotionGuardError(f"dt_s가 유효하지 않습니다: {dt_s}")
            for joint in JOINT_ORDER:
                prev_state = self._guard_states.get(joint, INITIAL_GUARD_STATE)
                guarded_value, new_state = apply_joint_motion_guard(
                    limits=self._motion_limits[joint],
                    current_state=current_follower_state_deg[joint],
                    target_now=raw_target[joint],
                    target_lookahead=target_lookahead[joint],
                    prev_guard_state=prev_state,
                    dt_s=dt_s,
                    correction_time_constant_s=self._correction_time_constant_s,
                )
                guarded[joint] = guarded_value
                new_guard_states[joint] = new_state
        except (MotionGuardError, KeyError) as exc:
            # guard 실패 시 이전 guard state/last_tick_time을 그대로 둔다(=이번 tick은
            # 버리고 다음 tick에서 다시 시도할 수 있게) - 잘못된 이력으로 오염시키지 않음.
            return _early_result(
                target_time=target_time, stop_reason=f"{STOP_REASON_GUARD_INVALID}: {exc}",
                raw_target=raw_target, contributing_sequences=ensembled.contributing_sequences,
                intent_decision=intent.decision, intent_reasons=intent.reasons,
            )

        # guard가 성공한 tick만 이력을 갱신한다.
        self._guard_states = new_guard_states
        self._last_tick_time_monotonic = now_monotonic

        # -- 6. Final SafetyGate (guarded target에, 반드시 motion guard 다음) --------------
        from runtime.laptop.action_adapter import adapt_vla_action

        adapted_guarded = adapt_vla_action(guarded)
        decision = self._safety_gate.evaluate(
            adapted_action=adapted_guarded, current_state_deg=current_follower_state_deg,
            observation_valid=True, check_excessive_step=False,
        )

        soft_mechanical_saturation = (
            decision.decision == "WOULD_CLAMP"
            and bool(decision.reasons)
            and all(reason.startswith("MECHANICAL_LIMIT_CLAMPED:") for reason in decision.reasons)
            and decision.safe_action is not None
        )
        target_valid = decision.decision == "ACCEPT" or soft_mechanical_saturation
        final_target = decision.safe_action if target_valid else None
        clamp_reasons = decision.reasons if soft_mechanical_saturation else ()
        stop_reason = None if target_valid else f"{STOP_REASON_SAFETY_PREFIX}{decision.decision}"

        return ControlTargetResult(
            target_time_monotonic=target_time,
            raw_ensemble_target=raw_target,
            intent_decision=intent.decision,
            intent_reasons=intent.reasons,
            smoothed_target=guarded,
            guarded_target=guarded,
            final_target=final_target,
            clamp_reasons=clamp_reasons,
            safety_decision=decision.decision,
            safety_reasons=decision.reasons,
            contributing_sequences=ensembled.contributing_sequences,
            target_valid=target_valid,
            stop_reason=stop_reason,
        )
