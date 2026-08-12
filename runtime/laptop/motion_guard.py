"""Phase C-3A(+C-3A.1 correction): velocity/acceleration/jerk-limited motion guard.

# Motion Guard vs Intent Validation vs Final SafetyGate (섹션 8 요구사항 - 역할 구분, C-3A.1에서 갱신)

이 모듈은 ``runtime/laptop/safety_gate.py``를 전혀 모르고, import하지도 않는다. 지금은
3단계 파이프라인(``runtime/laptop/realtime_control_target.py``)의 가운데 단계다:

    Temporal Ensemble -> **Intent Validation**(``intent_validation.py``, raw target
    자체가 신뢰 가능한 policy 의도인가 판정 - 아니면 이 모듈은 아예 호출되지 않는다) ->
    **Motion Guard(이 모듈)** -> Final SafetyGate(guarded target이 이번 tick 실행 가능한가)

- **Intent Validation**: raw ensemble target을 그대로 SafetyGate 기준으로 평가해,
  "이 policy 예측 자체를 신뢰할 만한가"를 판정한다. 여기서 막히면 Motion Guard는 그
  tick에 대해 전혀 호출되지 않는다 - "위험한 큰 target을 잘게 쪼개는" 상황 자체가
  발생할 수 없다(C-3A.1 correction의 핵심 - 아래 "우회 여부 재분석" 절 참고).
- **Motion Guard(이 모듈)**: Intent Validation을 통과한(=신뢰할 만하다고 판정된) target에
  "물리적으로 타당한 속도로 어떻게 접근할까"만 판단한다 - **부드럽게 rate-limit**해서
  이번 tick에 허용 가능한 만큼만 움직이는 ``guarded target``을 만든다. ACCEPT/REJECT
  이분법이 없다 - 항상 "이번 tick에 갈 수 있는 만큼"을 반환한다(입력이 애초에 invalid하지
  않는 한).
- **Final SafetyGate**(``safety_gate.py``, 이 모듈이 끝난 뒤 별도로 호출됨): guard가
  만든 최종 target을 절대 mechanical 범위/excessive-step 기준으로 다시
  ACCEPT/WOULD_CLAMP/REJECT 판정한다. **이 모듈은 Safety Gate를 대체하지 않는다.**

# Position 앵커링 설계 - "guard가 실제 로봇과 따로 노는" 문제를 구조적으로 막음 (섹션 9 근거)

이 guard는 매 tick마다 "이전에 guard가 계산했던 position"이 아니라 **그 tick에 실제로
측정된 follower state(``current_follower_state_deg``)를 위치의 기준점으로 다시
사용한다** - velocity/acceleration만 tick 사이에 이어지는 내부 이력으로 유지한다
(``GuardState``에 position이 없는 이유). 이렇게 설계한 이유:

만약 guard가 자기 자신이 "마지막으로 계산해서 내보낸 target"을 다음 tick의 위치
기준으로 삼았다면, Safety Gate가 그 target을 계속 WOULD_CLAMP/REJECT해서 실제로는 한
번도 write되지 않았는데도(이 Phase에서는 아직 write 자체가 없음) guard 내부 상태는
계속 "전진한 것처럼" 쌓여서, 실제 로봇 위치와 guard가 믿는 위치가 서서히 벌어질 수
있다 - 나중에 그 괴리가 갑자기 드러나면 오히려 큰 점프가 생길 위험이 있다. 매 tick
``current_follower_state_deg``로 위치를 다시 앵커링하면 이 문제가 구조적으로 발생할 수
없다: ``guarded_target``은 항상 "이번 tick 실측 위치 ± velocity_limit*dt" 안에 있다는
게 수식으로 보장된다(아래 ``apply_joint_motion_guard()`` 참고) - 몇 tick이 지났든,
이전에 뭐가 받아들여졌든 상관없다.

# [정정] 이게 Safety의 excessive-step 검사를 "우회"하는가? - C-3A 보고서의 결론을 수정함

C-3A 최초 구현 때는 이 절에서 "아니다"라고 결론 내렸다 - **그 결론은 절반만 맞았고
절반은 틀렸다. Phase C-3A.1에서 이 결론을 명시적으로 정정한다.**

**맞았던 부분(여전히 유효)**: mechanical hard limit은 절대 위치 기준이라 이 guard로
전혀 완화되지 않는다 - guard가 아무리 부드럽게 다가가도 목적지 자체가 mechanical 범위
밖이면 Final SafetyGate가 매 tick 그대로 다시 걸린다.

**틀렸던 부분(C-3A.1에서 발견)**: mechanical hard limit 우회가 아니어도, **policy
intent/outlier semantic은 실제로 우회됐다.** 실물 사례(wrist_flex 53.67->33.67, delta
20deg - 새 dataset의 robust gross-outlier regression으로 정의한
사례)를 C-3A 원래 파이프라인(raw -> guard -> safety)에 넣으면, guard가 그 20deg를
매 tick `velocity_limit*dt`(예: wrist_flex 29.01deg/s*16.7ms≈0.48deg)짜리 조각으로
잘라버리고, Final SafetyGate는 그 작은 조각만 보고 60 tick 내내 전부 ACCEPT해서 결국
원래(위험하다고 분류했던) 목표에 도달해버렸다(실측 확인, C-3A 보고서). 이건 "excessive-
step 임계값을 절대 못 넘는 속도로만 움직인다"는 점에서 진짜 mechanical bypass는
아니지만, **애초에 그 threshold가 하려던 일("이 policy 예측을 신뢰할 수 있는가"라는
outlier 판정)을 Motion Guard가 통째로 무력화**시킨 것이다 - guard 이후 단계는 항상
"작고 정상적인 delta"만 보게 되므로, 그 delta의 근거가 된 raw target이 애초에
비정상이었다는 사실 자체를 다시는 검사할 기회가 없다.

**수정된 결론**: mechanical bypass는 없다(여전히 참) + policy intent bypass는
**Intent Validation을 raw target에 대해 Motion Guard *이전*에 실행해야만** 막을 수
있다(``intent_validation.py``, ``realtime_control_target.py``가 이제 이 순서를
강제한다). Motion Guard 자신은 "얼마나 빨리 갈까"만 책임지고, "이 목적지 자체가
신뢰할 만한가"는 절대 책임지지 않는다 - 그 책임은 Intent Validation에 있다.

# 알려진 한계 - braking distance 없음

이 3단(jerk -> acceleration -> velocity) limiter는 "목표까지 남은 거리"를 고려한
감속(braking distance) 계산이 없다 - 그냥 매 tick 허용된 최대 가속으로 계속 목표를
향해 밀어붙인다. 이론적으로는 정지한(변하지 않는) 목표에 접근할 때 관성 때문에
오버슈트했다가 다시 돌아오는 감쇠 진동이 생길 수 있다. C-3A.1의 feedforward+bounded
correction 설계(위 "문제 2 교정" 참고)로 correction 성분이 훨씬 완만해졌고, Intent
Validation이 raw target 자체를 미리 걸러내므로(큰 outlier가 아예 이 모듈까지 오지
않음) 이 진동이 실제로 관측되는 빈도/폭은 C-3A 대비 줄었지만(오프라인 재현 결과 참고),
braking distance 계산 자체가 없다는 구조적 한계는 여전히 남아있다. 향후 개선 후보(이번
세션에서 구현하지 않음): 운동학적 정지거리 공식(``stopping_distance = velocity^2 /
(2*acceleration_limit)``)을 이용해 목표에 가까워지면 미리 감속을 시작하는 "S-curve
with braking" 방식으로 교체.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointMotionLimits:
    """한 관절의 속도/가속도/저크 상한. 단위는 몸통 5관절=deg, gripper=percent 기준
    (``vla_contract.JOINT_ORDER`` 관례와 동일) - 필드 이름 자체는 관절에 무관하게
    공용이다."""

    velocity_limit: float  # deg/s 또는 %/s (gripper)
    acceleration_limit: float  # deg/s^2 또는 %/s^2
    jerk_limit: float  # deg/s^3 또는 %/s^3

    def __post_init__(self) -> None:
        if self.velocity_limit <= 0 or self.acceleration_limit <= 0 or self.jerk_limit <= 0:
            raise ValueError(
                f"velocity/acceleration/jerk limit은 모두 양수여야 합니다: {self}"
            )


# ---------------------------------------------------------------------------
# 실측 근거 (섹션 5/6 요구사항 - 임의 hardcode 금지, 실제 데이터에서 도출)
# ---------------------------------------------------------------------------
#
# [C-3A.1 correction 섹션 5 - 재계산] 최초 C-3A는 data/so101_cube_pick_drop_v3_v4_combined69_v1
# (22,617 frame, fps=30, dt=33.333ms)의 observation.state를 **raw 30fps로 그대로 시간
# 미분**해서 velocity/acceleration/jerk 분포를 냈다. 이번 correction에서는 그 대신, 이
# 런타임이 실제로 쓰는 것과 정확히 같은 경로 - ``TemporalEnsembler``의 chunk-내부 linear
# interpolation - 로 각 episode를 60Hz로 리샘플링한 뒤 그 60Hz 시계열에서 미분했다
# (스크립트: 세션 스크래치패드 ``compute_vel_acc_jerk_60hz_resampled.py``, 69 episode 전부
# 사용, 결과 ``vel_acc_jerk_60hz_resampled_results.json``). 이 방식이 "guard가 실제로
# 무엇을 tracking하게 되는가"를 더 정확히 반영한다 - Motion Guard의 feedforward velocity는
# 바로 이 60Hz-interpolated 궤적에서 나오기 때문이다.
#
# **velocity는 old(raw 30fps diff)와 new(60Hz interpolation-resampled) 사이에 수학적으로
# 완전히 동일하다(ratio=1.000, 모든 관절)** - linear interpolation은 각 30Hz 구간
# 내부에서 속도를 상수로 보존하므로 더 촘촘히 샘플링해도 속도 분포 자체는 변하지 않는다.
# 그래서 velocity_limit은 그대로 유지한다.
#
# **acceleration/jerk는 significantly 달라진다(관절별 최대 acc 2.0배, jerk 5.1배까지)** -
# 이건 recompute 버그가 아니라 piecewise-linear interpolation의 잘 알려진 특성이다: 각
# 30Hz 구간 내부는 속도가 상수(가속도=0)이지만, 구간 경계(corner)에서는 속도가 순간적으로
# v_i에서 v_{i+1}로 바뀐다 - 그 경계를 걸치는 60Hz 차분은 (v_{i+1}-v_i)/dt_60Hz =
# (v_{i+1}-v_i)*60을 내는데, 이건 원래 raw 30fps 차분 (v_{i+1}-v_i)*30의 정확히 2배다
# (실측: shoulder_pan/elbow_flex acc ratio=2.000 정확히 일치). jerk는 이미 2배가 된
# acceleration의 경계 차분을 다시 60Hz로 나누므로 더 크게 증폭된다. **이 kink는
# "리샘플링 아티팩트"이지만 guard 입장에서는 허상이 아니라 실제로 매 tick 마주치는
# feedforward 요구치다** - RealTimeControlTargetGenerator가 실제로 TemporalEnsembler로
# 60Hz interpolation한 궤적을 그대로 feedforward에 쓰기 때문에, guard의
# acceleration_limit/jerk_limit이 raw-30fps 기준(old)으로 너무 타이트하면 **정상적인
# demo-like 궤적의 30Hz chunk 경계에서마다** guard가 불필요하게 개입하게 된다(섹션 6/8
# "guard activation 97~100%" 문제의 두 번째, error/dt 버그와는 독립적인 원인).
#
# 그래서 이 correction에서는 acceleration_limit/jerk_limit을 **60Hz-resampled 분포의
# p99**로 갱신한다(velocity_limit은 위 이유로 그대로). p99 채택 근거는 최초 C-3A와 동일 -
# "정상 teleop/demo motion은 거의 막지 않고 명백한 spike만 제한"(섹션 6).
#
# | joint | vel p99 (동일) | acc p99 OLD(30fps) -> NEW(60Hz resampled) | jerk p99 OLD -> NEW |
# |---|---:|---:|---:|
# | shoulder_pan  | 47.47 | 316.5 -> 633.0  (x2.00) | 14242 -> 37978 (x2.67) |
# | shoulder_lift | 47.47 | 395.6 -> 633.0  (x1.60) | 16615 -> 47472 (x2.86) |
# | elbow_flex    | 47.47 | 316.5 -> 633.0  (x2.00) | 16615 -> 37978 (x2.29) |
# | wrist_flex    | 29.01 | 316.5 -> 474.7  (x1.50) | 11869 -> 37979 (x3.20) |
# | wrist_roll    | 26.37 | 158.2 -> 158.2  (x1.00) |  7121 -> 18989 (x2.67) |
# | gripper       | 74.53 | 559.0 -> 869.6  (x1.56) | 13044 -> 67081 (x5.14) |
#
# (전체 old/new 비교 - max, p99.5 포함 - 는 최종 보고서 및
# ``vel_acc_jerk_60hz_resampled_results.json`` 참고. gripper가 다른 관절보다 배율이
# 큰 건 단위 스케일 차이(percent_0_100) 때문 - 최초 C-3A 표와 동일한 이유.)
DEFAULT_JOINT_MOTION_LIMITS: dict[str, JointMotionLimits] = {
    # so101_blue_cube_place_return_v1 41ep를 runtime 60Hz로 linear-resample한
    # |velocity|/|acceleration|/|jerk| p99.5.
    "shoulder_pan": JointMotionLimits(velocity_limit=42.20, acceleration_limit=791.21, jerk_limit=47472.89),
    "shoulder_lift": JointMotionLimits(velocity_limit=65.94, acceleration_limit=791.21, jerk_limit=56966.93),
    "elbow_flex": JointMotionLimits(velocity_limit=68.58, acceleration_limit=791.21, jerk_limit=56967.04),
    "wrist_flex": JointMotionLimits(velocity_limit=47.48, acceleration_limit=949.44, jerk_limit=66460.97),
    "wrist_roll": JointMotionLimits(velocity_limit=55.39, acceleration_limit=1265.94, jerk_limit=94945.05),
    "gripper": JointMotionLimits(velocity_limit=87.30, acceleration_limit=1533.01, jerk_limit=107310.13),
}


@dataclass(frozen=True)
class GuardState:
    """한 관절의 motion guard 내부 이력 - velocity/acceleration만 tick 사이에 이어간다
    (position은 없음 - 매 tick 실측 follower state로 다시 앵커링하므로, 모듈 docstring
    "Position 앵커링 설계" 참고). 첫 tick(이력 없음)에는 ``GuardState(velocity=0.0,
    acceleration=0.0)``를 bootstrap으로 쓴다(로봇이 정지해 있다고 가정)."""

    velocity: float
    acceleration: float


INITIAL_GUARD_STATE = GuardState(velocity=0.0, acceleration=0.0)


class MotionGuardError(RuntimeError):
    """입력 자체가 계산 불가능(NaN/Inf/dt<=0)한 경우 - fail closed(섹션 7)."""


# Phase C-3A.1 correction 근거: policy trajectory 자체의 시간 미분(feedforward velocity)을
# tracking error 보정(correction)과 별도로 다루기 위해, "이 tracking error를 몇 초에 걸쳐
# 닫을 것인가"를 이 상수로 정한다. dt(16.7ms @ 60Hz)로 나누면 사소한 위치 오차도 수백
# deg/s급 velocity로 폭증하는 원래 버그(문제 2, 섹션 6)를 만들었다 - 그래서 tick 주기와는
# 무관한, 훨씬 긴 시간축을 쓴다. VLA 추론 주기(이 저장소 실측 steady-state median 338ms,
# ``temporal_ensemble.py``의 ``DEFAULT_HALF_LIFE_S``와 동일 근거)를 그대로 재사용했다 -
# "policy trajectory 참조 자체가 갱신되는 시간 규모"와 같은 크기로 correction을 닫는 게
# 자연스럽기 때문이다(그보다 훨씬 빠르게 닫으려 하면 다시 원래 버그와 비슷한 성격의 과도한
# correction velocity가 나온다).
DEFAULT_CORRECTION_TIME_CONSTANT_S = 0.338


def apply_joint_motion_guard(
    *,
    limits: JointMotionLimits,
    current_state: float,
    target_now: float,
    target_lookahead: float,
    prev_guard_state: GuardState,
    dt_s: float,
    correction_time_constant_s: float = DEFAULT_CORRECTION_TIME_CONSTANT_S,
) -> tuple[float, GuardState]:
    """한 관절, 한 tick 분의 velocity->acceleration->jerk 순차 제한 (섹션 7 알고리즘,
    Phase C-3A.1에서 velocity 산출 방식을 교정함).

    # 문제 2 교정: "tracking error/dt"가 아니라 "trajectory feedforward + bounded correction"

    원래(C-3A) 구현은 ``requested_velocity = (raw_target - current_state) / dt_s``였다 -
    이건 "policy trajectory 자체가 지금 얼마나 빨리 움직이고 있는가"가 아니라 "지금 당장
    이 tick 안에 남은 tracking error를 전부 없애려면 얼마나 빨라야 하는가"를 계산한
    것이었다. ``dt_s``가 16.7ms(60Hz)로 아주 작기 때문에, 몇 deg짜리 사소한 tracking
    오차만 있어도(예: 이전 tick에서 guard가 조금 못 따라간 경우) 수백 deg/s급 값이
    나와서 거의 항상 velocity_limit에 saturate했다(실측: guard activation 97~100%,
    자세한 수치 증명은 별도 조사 스크립트/최종 보고서 참고).

    지금은 두 성분을 분리한다:

    1. **feedforward velocity** = ``(target_lookahead - target_now) / dt_s`` - 같은
       policy trajectory를 ``T``와 ``T+dt``에서 각각 샘플링(``TemporalEnsembler``가 이미
       절대시간 interpolation을 지원하므로 재구현 없음, 호출자가 두 샘플을 넘겨줌)한
       차이 - "trajectory 자체의 순간 속도"를 뜻한다. tracking error와 무관하다.
    2. **bounded correction velocity** = tracking error(``target_now - current_state``)를
       ``correction_time_constant_s``에 걸쳐 닫는 속도, ``±velocity_limit``로 clamp.
       tick 주기(``dt_s``)가 아니라 훨씬 긴 시간 상수를 쓰므로, 사소한 오차가 즉시
       velocity 폭증으로 이어지지 않는다.

    최종 ``requested_velocity = feedforward_velocity + correction_velocity``를 그 다음은
    기존과 동일하게 jerk -> acceleration -> velocity 순으로 clamp하고, position은
    여전히 ``current_state`` 기준으로 다시 앵커링한다(Position 앵커링 설계, 모듈
    docstring 참고 - 이 부분은 안 바뀜).

    Raises:
        MotionGuardError: 입력이 finite하지 않거나 ``dt_s``/``correction_time_constant_s``가
            유한한 양수가 아닌 경우 (fail closed).
    """
    import math

    if not all(math.isfinite(v) for v in (current_state, target_now, target_lookahead)):
        raise MotionGuardError(
            f"current_state/target_now/target_lookahead가 finite하지 않습니다: "
            f"current_state={current_state}, target_now={target_now}, target_lookahead={target_lookahead}"
        )
    if dt_s <= 0 or not math.isfinite(dt_s):
        raise MotionGuardError(f"dt_s는 유한한 양수여야 합니다: {dt_s}")
    if correction_time_constant_s <= 0 or not math.isfinite(correction_time_constant_s):
        raise MotionGuardError(f"correction_time_constant_s는 유한한 양수여야 합니다: {correction_time_constant_s}")

    feedforward_velocity = (target_lookahead - target_now) / dt_s
    position_error = target_now - current_state
    correction_velocity = _clamp(position_error / correction_time_constant_s, limits.velocity_limit)
    requested_velocity = feedforward_velocity + correction_velocity

    requested_acceleration = (requested_velocity - prev_guard_state.velocity) / dt_s
    requested_jerk = (requested_acceleration - prev_guard_state.acceleration) / dt_s

    clamped_jerk = _clamp(requested_jerk, limits.jerk_limit)
    next_acceleration = prev_guard_state.acceleration + clamped_jerk * dt_s
    clamped_acceleration = _clamp(next_acceleration, limits.acceleration_limit)
    next_velocity = prev_guard_state.velocity + clamped_acceleration * dt_s
    clamped_velocity = _clamp(next_velocity, limits.velocity_limit)

    guarded_position = current_state + clamped_velocity * dt_s

    if not math.isfinite(guarded_position):
        raise MotionGuardError(f"guarded_position 계산 결과가 finite하지 않습니다: {guarded_position}")

    return guarded_position, GuardState(velocity=clamped_velocity, acceleration=clamped_acceleration)


def _clamp(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


@dataclass(frozen=True)
class CoordinatedGuardState:
    """State shared by the six-axis trajectory-level motion guard."""

    positions: dict[str, float]
    velocities: dict[str, float]
    accelerations: dict[str, float]
    phase_scale: float = 1.0
    previous_actual_positions: dict[str, float] | None = None
    no_response_duration_s: float = 0.0
    opposite_response_duration_s: float = 0.0


DEFAULT_TRACKING_LEAD_LIMITS: dict[str, float] = {
    # Teleop observed max plus dataset-limit stopping distance (v^2/2a).
    # This is a phase-hold trigger, not a lag-only fatal threshold.
    "shoulder_pan": 6.9275880272960695,
    "shoulder_lift": 11.53895186826798,
    "elbow_flex": 9.126013088035567,
    "wrist_flex": 8.308079154583202,
    "wrist_roll": 5.607372765353356,
    "gripper": 11.04335337406125,
}
DEFAULT_LAG_SOFT_FRACTIONS: dict[str, float] = {
    "shoulder_pan": 5.165714285714285 / DEFAULT_TRACKING_LEAD_LIMITS["shoulder_pan"],
    "shoulder_lift": 8.066813186813185 / DEFAULT_TRACKING_LEAD_LIMITS["shoulder_lift"],
    "elbow_flex": 6.065934065934059 / DEFAULT_TRACKING_LEAD_LIMITS["elbow_flex"],
    "wrist_flex": 7.120879120879124 / DEFAULT_TRACKING_LEAD_LIMITS["wrist_flex"],
    "wrist_roll": 3.604395604395605 / DEFAULT_TRACKING_LEAD_LIMITS["wrist_roll"],
    "gripper": 8.488612836438925 / DEFAULT_TRACKING_LEAD_LIMITS["gripper"],
}
DEFAULT_LAG_SOFT_FRACTION = min(DEFAULT_LAG_SOFT_FRACTIONS.values())
DEFAULT_PHASE_RECOVERY_PER_S = 2.0
DEFAULT_NO_RESPONSE_TIMEOUT_S = 10.0
DEFAULT_ENCODER_RESPONSE_THRESHOLD_DEG = 0.0879120879120876
DEFAULT_OPPOSITE_RESPONSE_TIMEOUT_S = 10.0


def apply_coordinated_motion_guard(
    *,
    limits_by_joint: dict[str, JointMotionLimits],
    current_state: dict[str, float],
    target_now: dict[str, float],
    target_lookahead: dict[str, float],
    prev_state: CoordinatedGuardState | None,
    dt_s: float,
    correction_time_constant_s: float = DEFAULT_CORRECTION_TIME_CONSTANT_S,
    tracking_lead_limits: dict[str, float] | None = None,
    lag_soft_fraction: float | None = None,
    no_response_timeout_s: float = DEFAULT_NO_RESPONSE_TIMEOUT_S,
    response_threshold_deg: float = DEFAULT_ENCODER_RESPONSE_THRESHOLD_DEG,
    opposite_response_timeout_s: float = DEFAULT_OPPOSITE_RESPONSE_TIMEOUT_S,
    phase_recovery_per_s: float = DEFAULT_PHASE_RECOVERY_PER_S,
) -> tuple[dict[str, float], CoordinatedGuardState]:
    """Advance one virtual 6-D command with one common trajectory time scale.

    The virtual position is initialized from measured encoders, then integrated
    independently of them. Encoder state is used only to bound tracking lead and
    reduce the common phase rate. Velocity, acceleration and jerk constraints
    are intersected as scalar intervals, so normal trajectory progress never
    independently clips vector components.
    """
    import math

    joints = tuple(limits_by_joint)
    leads = tracking_lead_limits or DEFAULT_TRACKING_LEAD_LIMITS
    if dt_s <= 0 or not math.isfinite(dt_s):
        raise MotionGuardError(f"dt_s must be finite and positive: {dt_s}")
    if correction_time_constant_s <= 0 or not math.isfinite(correction_time_constant_s):
        raise MotionGuardError("correction_time_constant_s must be finite and positive")
    if lag_soft_fraction is not None and not 0 < lag_soft_fraction < 1:
        raise MotionGuardError("invalid lag slowdown configuration")
    if phase_recovery_per_s <= 0 or no_response_timeout_s <= 0 or response_threshold_deg <= 0:
        raise MotionGuardError("invalid lag response configuration")
    if opposite_response_timeout_s <= 0:
        raise MotionGuardError("invalid opposite-response configuration")
    for joint in joints:
        values = (current_state[joint], target_now[joint], target_lookahead[joint], leads[joint])
        if not all(math.isfinite(x) for x in values) or leads[joint] <= 0:
            raise MotionGuardError(f"invalid coordinated guard input for {joint}: {values}")

    if prev_state is None:
        virtual = {j: float(current_state[j]) for j in joints}
        previous_velocity = {j: 0.0 for j in joints}
        previous_acceleration = {j: 0.0 for j in joints}
        previous_phase_scale = 1.0
        previous_actual = dict(current_state)
        no_response_duration = 0.0
        opposite_response_duration = 0.0
    else:
        virtual = dict(prev_state.positions)
        previous_velocity = dict(prev_state.velocities)
        previous_acceleration = dict(prev_state.accelerations)
        previous_phase_scale = prev_state.phase_scale
        previous_actual = dict(prev_state.previous_actual_positions or current_state)
        no_response_duration = prev_state.no_response_duration_s
        opposite_response_duration = prev_state.opposite_response_duration_s

    desired_velocity = {
        j: (target_lookahead[j] - target_now[j]) / dt_s
        + (target_now[j] - virtual[j]) / correction_time_constant_s
        for j in joints
    }

    lag_scale = 1.0
    for joint in joints:
        lag = virtual[joint] - current_state[joint]
        desired = desired_velocity[joint]
        # Only trajectory progress that increases existing lag is slowed.
        if lag * desired <= 0:
            continue
        hard = leads[joint]
        soft_fraction = (
            lag_soft_fraction if lag_soft_fraction is not None
            else DEFAULT_LAG_SOFT_FRACTIONS.get(joint, DEFAULT_LAG_SOFT_FRACTION)
        )
        soft = hard * soft_fraction
        magnitude = abs(lag)
        if magnitude >= hard:
            lag_scale = 0.0
        elif magnitude > soft:
            lag_scale = min(lag_scale, (hard - magnitude) / (hard - soft))

    if lag_scale < previous_phase_scale:
        phase_scale = lag_scale
    else:
        phase_scale = min(lag_scale, previous_phase_scale + phase_recovery_per_s * dt_s)

    scaled_desired = {j: desired_velocity[j] * phase_scale for j in joints}
    scalar_lo, scalar_hi = 0.0, 1.0

    def intersect_velocity_interval(desired: float, low: float, high: float) -> None:
        nonlocal scalar_lo, scalar_hi
        if abs(desired) < 1e-12:
            if not low <= 0.0 <= high:
                scalar_lo, scalar_hi = 1.0, 0.0
            return
        a, b = low / desired, high / desired
        scalar_lo = max(scalar_lo, min(a, b))
        scalar_hi = min(scalar_hi, max(a, b))

    for joint in joints:
        motion = limits_by_joint[joint]
        prev_v = previous_velocity[joint]
        prev_a = previous_acceleration[joint]
        low = max(
            -motion.velocity_limit,
            prev_v - motion.acceleration_limit * dt_s,
            prev_v + (prev_a - motion.jerk_limit * dt_s) * dt_s,
        )
        high = min(
            motion.velocity_limit,
            prev_v + motion.acceleration_limit * dt_s,
            prev_v + (prev_a + motion.jerk_limit * dt_s) * dt_s,
        )
        lag = virtual[joint] - current_state[joint]
        hard = leads[joint]
        low = max(low, (-hard - lag) / dt_s)
        high = min(high, (hard - lag) / dt_s)
        intersect_velocity_interval(scaled_desired[joint], low, high)

    if scalar_hi >= max(0.0, scalar_lo):
        common_scale = min(1.0, scalar_hi)
        next_velocity = {j: scaled_desired[j] * common_scale for j in joints}
    else:
        # A sharp direction change can make exact vector scaling temporarily
        # infeasible. Hold trajectory phase and perform a bounded coordinated
        # brake; raw-path progress resumes only after the dynamic state permits it.
        phase_scale = 0.0
        next_velocity = {}
        for joint in joints:
            motion = limits_by_joint[joint]
            prev_v = previous_velocity[joint]
            prev_a = previous_acceleration[joint]
            desired_a = _clamp(-prev_v / dt_s, motion.acceleration_limit)
            delta_a = _clamp(desired_a - prev_a, motion.jerk_limit * dt_s)
            next_a = _clamp(prev_a + delta_a, motion.acceleration_limit)
            velocity = prev_v + next_a * dt_s
            next_velocity[joint] = _clamp(velocity, motion.velocity_limit)

    next_acceleration = {
        j: (next_velocity[j] - previous_velocity[j]) / dt_s for j in joints
    }
    next_positions = {j: virtual[j] + next_velocity[j] * dt_s for j in joints}
    # During a coordinated hold, encoder motion away from the command can make
    # lag grow even though virtual velocity is braking. Keep the emergency hold
    # envelope bounded without resuming any trajectory component.
    if phase_scale == 0.0:
        next_positions = {
            joint: max(current_state[joint] - leads[joint], min(current_state[joint] + leads[joint], position))
            for joint, position in next_positions.items()
        }

    blocking_joints = [
        joint for joint in joints
        if abs(virtual[joint] - current_state[joint]) >= leads[joint] - response_threshold_deg
        and (virtual[joint] - current_state[joint]) * desired_velocity[joint] > 0
    ]
    response_anchor = previous_actual
    if blocking_joints:
        same_direction_response = max(
            (current_state[joint] - response_anchor[joint])
            * (1.0 if desired_velocity[joint] > 0 else -1.0)
            for joint in blocking_joints
        )
        if same_direction_response >= response_threshold_deg:
            no_response_duration = 0.0
            opposite_response_duration = 0.0
            response_anchor = dict(current_state)
        else:
            no_response_duration += dt_s
            if same_direction_response < 0.0:
                opposite_response_duration += dt_s
            else:
                opposite_response_duration = 0.0
    else:
        no_response_duration = 0.0
        opposite_response_duration = 0.0

    if opposite_response_duration >= opposite_response_timeout_s:
        raise MotionGuardError(f"sustained opposite encoder response: joints={blocking_joints}")
    if no_response_duration >= no_response_timeout_s:
        raise MotionGuardError(
            f"sustained no-response at bounded tracking lead: joints={blocking_joints}"
        )

    state = CoordinatedGuardState(
        positions=next_positions,
        velocities=next_velocity,
        previous_actual_positions=response_anchor,
        no_response_duration_s=no_response_duration,
        opposite_response_duration_s=opposite_response_duration,
        accelerations=next_acceleration,
        phase_scale=phase_scale,
    )
    return dict(next_positions), state
