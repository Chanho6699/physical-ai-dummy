"""runtime/laptop/motion_guard.py의 apply_joint_motion_guard() 순수 함수 검증 -
Phase C-3A.1: feedforward velocity + bounded correction 설계로 교정된 버전.

velocity/acceleration/jerk 각각을 격리해서 정확한 수식으로 확인한다 (다른 두 limit은
절대 안 묶이도록 충분히 크게 잡아서 하나씩만 binding하게 설계). 또한 문제 2("tracking
error/dt가 거의 항상 velocity saturation을 일으킨다")가 실제로 고쳐졌는지 직접 증명하는
테스트를 추가했다.
"""

from __future__ import annotations

import math

import pytest

from runtime.laptop.motion_guard import (
    DEFAULT_CORRECTION_TIME_CONSTANT_S,
    INITIAL_GUARD_STATE,
    GuardState,
    JointMotionLimits,
    MotionGuardError,
    apply_joint_motion_guard,
)

DT = 1.0 / 60.0
HUGE = 1.0e9  # 사실상 안 걸리는 limit


def test_joint_motion_limits_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        JointMotionLimits(velocity_limit=0.0, acceleration_limit=1.0, jerk_limit=1.0)
    with pytest.raises(ValueError):
        JointMotionLimits(velocity_limit=1.0, acceleration_limit=-1.0, jerk_limit=1.0)


def test_correction_time_constant_default_matches_inference_cadence_basis() -> None:
    assert DEFAULT_CORRECTION_TIME_CONSTANT_S == pytest.approx(0.338)


# ---------------------------------------------------------------------------
# velocity/acceleration/jerk spike clamp - 정지한(target_lookahead==target_now) 큰
# 목표에 대해, 이전 C-3A와 정확히 같은 최종 수치가 나오는지 (correction이 충분히 크면
# 결국 각 stage에서 동일하게 saturate되므로 - 아래 각주 참고).
# ---------------------------------------------------------------------------


def test_velocity_spike_clamp_from_rest() -> None:
    limits = JointMotionLimits(velocity_limit=10.0, acceleration_limit=HUGE, jerk_limit=HUGE)
    guarded, new_state = apply_joint_motion_guard(
        limits=limits, current_state=0.0, target_now=1000.0, target_lookahead=1000.0,
        prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT,
    )
    # correction_velocity가 이미 velocity_limit에서 clamp되므로(1000/0.338≈2960 -> 10),
    # accel/jerk가 사실상 무제한이면 그대로 흘러 첫 tick에 velocity_limit까지 도달.
    assert guarded == pytest.approx(10.0 * DT, rel=1e-6)
    assert new_state.velocity == pytest.approx(10.0, rel=1e-6)


def test_velocity_not_clamped_when_within_limit() -> None:
    limits = JointMotionLimits(velocity_limit=10.0, acceleration_limit=HUGE, jerk_limit=HUGE)
    guarded, _ = apply_joint_motion_guard(
        limits=limits, current_state=0.0, target_now=0.01, target_lookahead=0.01,
        prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT,
    )
    # position_error=0.01, correction=0.01/0.338=0.0296 (velocity_limit 안 걸림) - 그대로 적용.
    expected_velocity = 0.01 / DEFAULT_CORRECTION_TIME_CONSTANT_S
    assert guarded == pytest.approx(expected_velocity * DT, rel=1e-4)


def test_acceleration_spike_clamp_from_rest() -> None:
    limits = JointMotionLimits(velocity_limit=HUGE, acceleration_limit=100.0, jerk_limit=HUGE)
    guarded, new_state = apply_joint_motion_guard(
        limits=limits, current_state=0.0, target_now=1000.0, target_lookahead=1000.0,
        prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT,
    )
    # correction_velocity = 1000/0.338 ≈ 2959.8 (velocity_limit=HUGE라 안 잘림) - 여전히
    # accel_limit(100)보다 훨씬 커서 jerk 무제한이면 acceleration이 즉시 acc_limit까지.
    expected = 100.0 * DT * DT
    assert guarded == pytest.approx(expected, rel=1e-6)
    assert new_state.acceleration == pytest.approx(100.0, rel=1e-6)


def test_jerk_spike_clamp_from_rest() -> None:
    limits = JointMotionLimits(velocity_limit=HUGE, acceleration_limit=HUGE, jerk_limit=1000.0)
    guarded, new_state = apply_joint_motion_guard(
        limits=limits, current_state=0.0, target_now=1000.0, target_lookahead=1000.0,
        prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT,
    )
    expected = 1000.0 * DT ** 3
    assert guarded == pytest.approx(expected, rel=1e-6)


def test_jerk_limited_acceleration_builds_up_over_ticks() -> None:
    limits = JointMotionLimits(velocity_limit=HUGE, acceleration_limit=HUGE, jerk_limit=1000.0)
    state = INITIAL_GUARD_STATE
    accelerations = []
    for _ in range(5):
        _, state = apply_joint_motion_guard(
            limits=limits, current_state=0.0, target_now=1000.0, target_lookahead=1000.0,
            prev_guard_state=state, dt_s=DT,
        )
        accelerations.append(state.acceleration)
    assert accelerations == sorted(accelerations)  # 단조 증가
    assert accelerations[0] == pytest.approx(1000.0 * DT, rel=1e-6)


def test_small_request_within_all_limits_passes_through_unclamped() -> None:
    limits = JointMotionLimits(velocity_limit=50.0, acceleration_limit=500.0, jerk_limit=20000.0)
    guarded, _ = apply_joint_motion_guard(
        limits=limits, current_state=10.0, target_now=10.001, target_lookahead=10.001,
        prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT,
    )
    expected_velocity = 0.001 / DEFAULT_CORRECTION_TIME_CONSTANT_S  # 아주 작은 correction
    assert guarded == pytest.approx(10.0 + expected_velocity * DT, rel=1e-3)


# ---------------------------------------------------------------------------
# [핵심 회귀] 문제 2 교정 검증: 작은 tracking error가 dt로 나뉘어 velocity를 폭증시키지
# 않는지 - "error/dt" 방식(옛 구현)과 직접 비교해서 수치로 증명한다 (섹션 6/7 요구사항).
# ---------------------------------------------------------------------------


def test_small_tracking_error_does_not_saturate_velocity_at_60hz() -> None:
    """C-3A 원래 버그 재현 조건: 2deg 정도의 사소한 tracking error, dt=16.7ms(60Hz).
    OLD 공식(error/dt)이었다면 2/0.0167≈120deg/s - 어떤 현실적인 velocity_limit도
    압도적으로 초과해 거의 항상 saturate했을 상황이다. NEW 공식은 correction_time_constant
    (0.338s)로 나누므로 정상상태 velocity가 2/0.338≈5.9deg/s에 수렴해야 한다 - 실제 demo
    기반 velocity_limit(47.47)에 한참 못 미쳐 saturate되지 않는다.

    (여러 tick을 돌려 jerk/accel의 첫 tick 램프업 구간을 지나 정상상태에 도달시킨다 - cold
    start(v=0,a=0)에서 단 1 tick 만에 목표 velocity에 도달하는 건애초에 jerk_limit이
    막는다, 이건 이 테스트의 관심사가 아니라 별도 "jerk spike clamp" 테스트가 이미 검증함.)
    """
    limits = JointMotionLimits(velocity_limit=47.47, acceleration_limit=395.6, jerk_limit=16615.0)
    old_formula_velocity = 2.0 / DT  # = 120 deg/s (옛 구현이라면 이렇게 계산됐을 값)
    assert old_formula_velocity > limits.velocity_limit * 2  # 옛 공식은 확실히 saturate시켰을 것

    state = INITIAL_GUARD_STATE
    current_state = 51.67
    target = 53.67  # 2deg 뒤처짐, 고정된 목표(target_lookahead==target_now)
    # 1차 지연계(시정수 0.338s) 근사: t=1초(≈2.96 시정수) 후 잔여오차는
    # 2.0*exp(-1/0.338)≈0.10deg로 0.2 tolerance에 여유 있게 수렴한다.
    for _ in range(60):  # 1.0초
        guarded, state = apply_joint_motion_guard(
            limits=limits, current_state=current_state, target_now=target, target_lookahead=target,
            prev_guard_state=state, dt_s=DT,
        )
        assert abs(state.velocity) <= limits.velocity_limit + 1e-6  # 매 tick 절대 saturate 안 됨
        current_state = guarded
    assert abs(current_state - target) < 0.2  # 1초 후 거의 수렴


def test_feedforward_velocity_reflects_trajectory_rate_not_tracking_error() -> None:
    """target_now==current_state(추적 오차 0)인데 trajectory 자체가 움직이고 있으면
    (target_lookahead != target_now), guard의 velocity는 그 trajectory 속도로
    수렴해야 한다 - 이게 "policy trajectory의 시간 미분"을 쓰는 이유다. (역시 cold-start
    jerk 램프업을 피하려고 여러 tick을 돈다.)"""
    limits = JointMotionLimits(velocity_limit=47.47, acceleration_limit=395.6, jerk_limit=16615.0)
    trajectory_velocity = 5.0  # deg/s - 정상적인 느긋한 움직임
    state = INITIAL_GUARD_STATE
    current_state = 10.0
    target_now = 10.0
    for _ in range(20):
        target_lookahead = target_now + trajectory_velocity * DT
        guarded, state = apply_joint_motion_guard(
            limits=limits, current_state=current_state, target_now=target_now, target_lookahead=target_lookahead,
            prev_guard_state=state, dt_s=DT,
        )
        current_state = guarded
        target_now = target_lookahead  # 다음 tick도 같은 속도로 계속 전진하는 trajectory
    assert state.velocity == pytest.approx(trajectory_velocity, rel=0.05)  # 정상상태에서 saturate 없이 수렴


def test_correction_and_feedforward_combine_additively() -> None:
    limits = JointMotionLimits(velocity_limit=HUGE, acceleration_limit=HUGE, jerk_limit=HUGE)
    trajectory_velocity = 3.0
    tracking_error = 1.0
    target_now = 20.0
    current_state = target_now - tracking_error
    target_lookahead = target_now + trajectory_velocity * DT
    _, new_state = apply_joint_motion_guard(
        limits=limits, current_state=current_state, target_now=target_now, target_lookahead=target_lookahead,
        prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT,
    )
    expected = trajectory_velocity + tracking_error / DEFAULT_CORRECTION_TIME_CONSTANT_S
    assert new_state.velocity == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Fail-closed: NaN/Inf/dt<=0/correction_time_constant<=0
# ---------------------------------------------------------------------------


def test_nan_target_now_raises() -> None:
    limits = JointMotionLimits(velocity_limit=10.0, acceleration_limit=100.0, jerk_limit=1000.0)
    with pytest.raises(MotionGuardError):
        apply_joint_motion_guard(
            limits=limits, current_state=0.0, target_now=math.nan, target_lookahead=0.0,
            prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT,
        )


def test_nan_target_lookahead_raises() -> None:
    limits = JointMotionLimits(velocity_limit=10.0, acceleration_limit=100.0, jerk_limit=1000.0)
    with pytest.raises(MotionGuardError):
        apply_joint_motion_guard(
            limits=limits, current_state=0.0, target_now=0.0, target_lookahead=math.inf,
            prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT,
        )


def test_inf_current_state_raises() -> None:
    limits = JointMotionLimits(velocity_limit=10.0, acceleration_limit=100.0, jerk_limit=1000.0)
    with pytest.raises(MotionGuardError):
        apply_joint_motion_guard(
            limits=limits, current_state=math.inf, target_now=0.0, target_lookahead=0.0,
            prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT,
        )


def test_zero_dt_raises() -> None:
    limits = JointMotionLimits(velocity_limit=10.0, acceleration_limit=100.0, jerk_limit=1000.0)
    with pytest.raises(MotionGuardError):
        apply_joint_motion_guard(
            limits=limits, current_state=0.0, target_now=1.0, target_lookahead=1.0,
            prev_guard_state=INITIAL_GUARD_STATE, dt_s=0.0,
        )


def test_negative_dt_raises() -> None:
    limits = JointMotionLimits(velocity_limit=10.0, acceleration_limit=100.0, jerk_limit=1000.0)
    with pytest.raises(MotionGuardError):
        apply_joint_motion_guard(
            limits=limits, current_state=0.0, target_now=1.0, target_lookahead=1.0,
            prev_guard_state=INITIAL_GUARD_STATE, dt_s=-0.01,
        )


def test_non_positive_correction_time_constant_raises() -> None:
    limits = JointMotionLimits(velocity_limit=10.0, acceleration_limit=100.0, jerk_limit=1000.0)
    with pytest.raises(MotionGuardError):
        apply_joint_motion_guard(
            limits=limits, current_state=0.0, target_now=1.0, target_lookahead=1.0,
            prev_guard_state=INITIAL_GUARD_STATE, dt_s=DT, correction_time_constant_s=0.0,
        )


# ---------------------------------------------------------------------------
# Position anchoring (변경 없음 - 여전히 성립해야 함)
# ---------------------------------------------------------------------------


def test_guarded_position_always_bounded_by_velocity_limit_times_dt_from_current_state() -> None:
    limits = JointMotionLimits(velocity_limit=5.0, acceleration_limit=HUGE, jerk_limit=HUGE)
    prev_state = GuardState(velocity=1000.0, acceleration=0.0)
    current_state = 0.0
    guarded, _ = apply_joint_motion_guard(
        limits=limits, current_state=current_state, target_now=0.0, target_lookahead=0.0,
        prev_guard_state=prev_state, dt_s=DT,
    )
    max_delta = limits.velocity_limit * DT
    assert abs(guarded - current_state) <= max_delta + 1e-9


def test_coordinated_guard_preserves_ratio_and_dynamic_limits() -> None:
    from runtime.laptop.motion_guard import apply_coordinated_motion_guard

    joints = ("shoulder_lift", "elbow_flex")
    limits = {j: JointMotionLimits(60.0, 600.0, 30000.0) for j in joints}
    current = {j: 0.0 for j in joints}
    target = {"shoulder_lift": 20.0, "elbow_flex": -18.0}
    state = None
    previous_velocity = {j: 0.0 for j in joints}
    previous_acceleration = {j: 0.0 for j in joints}
    for _ in range(360):
        guarded, state = apply_coordinated_motion_guard(
            limits_by_joint=limits, current_state=current, target_now=target,
            target_lookahead=target, prev_state=state, dt_s=DT,
            tracking_lead_limits={j: 2.0 for j in joints},
        )
        for joint in joints:
            velocity = state.velocities[joint]
            acceleration = state.accelerations[joint]
            jerk = (acceleration - previous_acceleration[joint]) / DT
            assert abs(velocity) <= limits[joint].velocity_limit + 1e-7
            assert abs(acceleration) <= limits[joint].acceleration_limit + 1e-7
            assert abs(jerk) <= limits[joint].jerk_limit + 1e-6
            previous_velocity[joint] = velocity
            previous_acceleration[joint] = acceleration
        current = guarded
    assert guarded["shoulder_lift"] / abs(guarded["elbow_flex"]) == pytest.approx(20 / 18, rel=1e-5)


def test_stuck_elbow_cannot_allow_shoulder_only_runaway() -> None:
    from runtime.laptop.motion_guard import apply_coordinated_motion_guard

    joints = ("shoulder_lift", "elbow_flex")
    limits = {j: JointMotionLimits(60.0, 600.0, 30000.0) for j in joints}
    current = {j: 0.0 for j in joints}
    target = {"shoulder_lift": 20.0, "elbow_flex": -18.0}
    state = None
    last_guarded = dict(current)
    with pytest.raises(MotionGuardError, match="sustained no-response"):
        for _ in range(660):
            last_guarded, state = apply_coordinated_motion_guard(
                limits_by_joint=limits, current_state=current, target_now=target,
                target_lookahead=target, prev_state=state, dt_s=DT,
                tracking_lead_limits={j: 2.0 for j in joints},
            )
            current["shoulder_lift"] = last_guarded["shoulder_lift"]
            # Endpoint elbow intentionally does not respond.
    assert abs(last_guarded["elbow_flex"]) <= 5.0 + 1e-7
    assert last_guarded["shoulder_lift"] < 4.0

def test_normal_teleop_p95_elbow_lead_is_diagnostic_only() -> None:
    from runtime.laptop.motion_guard import (
        CoordinatedGuardState,
        DEFAULT_TRACKING_LEAD_LIMITS,
        apply_coordinated_motion_guard,
    )

    joints = ("shoulder_lift", "elbow_flex")
    limits = {j: JointMotionLimits(60.0, 600.0, 30000.0) for j in joints}
    current = {j: 0.0 for j in joints}
    previous = CoordinatedGuardState(
        positions={"shoulder_lift": 0.0, "elbow_flex": -5.714285714285715},
        velocities={j: 0.0 for j in joints},
        accelerations={j: 0.0 for j in joints},
        phase_scale=1.0,
        previous_actual_positions=dict(current),
    )
    _, state = apply_coordinated_motion_guard(
        limits_by_joint=limits,
        current_state=current,
        target_now={"shoulder_lift": 10.0, "elbow_flex": -20.0},
        target_lookahead={"shoulder_lift": 10.0, "elbow_flex": -20.0},
        prev_state=previous,
        dt_s=DT,
    )
    assert DEFAULT_TRACKING_LEAD_LIMITS["elbow_flex"] > 5.714285714285715
    assert state.phase_scale == pytest.approx(1.0)


def test_sustained_no_response_at_data_driven_hold_fails_closed() -> None:
    from runtime.laptop.motion_guard import (
        CoordinatedGuardState,
        DEFAULT_TRACKING_LEAD_LIMITS,
        apply_coordinated_motion_guard,
    )

    joints = ("shoulder_lift", "elbow_flex")
    limits = {j: JointMotionLimits(HUGE, HUGE, HUGE) for j in joints}
    current = {j: 0.0 for j in joints}
    hold = DEFAULT_TRACKING_LEAD_LIMITS["elbow_flex"]
    state = CoordinatedGuardState(
        positions={"shoulder_lift": 0.0, "elbow_flex": -hold},
        velocities={j: 0.0 for j in joints},
        accelerations={j: 0.0 for j in joints},
        phase_scale=0.0,
        previous_actual_positions=dict(current),
    )
    with pytest.raises(MotionGuardError, match="sustained no-response"):
        for _ in range(660):
            _, state = apply_coordinated_motion_guard(
                limits_by_joint=limits,
                current_state=current,
                target_now={"shoulder_lift": 10.0, "elbow_flex": -20.0},
                target_lookahead={"shoulder_lift": 10.0, "elbow_flex": -20.0},
                prev_state=state,
                dt_s=DT,
            )


def test_temporary_lag_recovers_without_hard_block() -> None:
    from runtime.laptop.motion_guard import apply_coordinated_motion_guard

    joints = ("shoulder_lift", "elbow_flex")
    limits = {j: JointMotionLimits(60.0, 600.0, 30000.0) for j in joints}
    current = {j: 0.0 for j in joints}
    target = {"shoulder_lift": 20.0, "elbow_flex": -18.0}
    state = None
    scales = []
    for tick in range(180):
        guarded, state = apply_coordinated_motion_guard(
            limits_by_joint=limits, current_state=current, target_now=target,
            target_lookahead=target, prev_state=state, dt_s=DT,
        )
        # 0.5 s temporary elbow delay, then a responsive coordinated follower.
        if tick >= 30:
            current["elbow_flex"] += 0.5 * (guarded["elbow_flex"] - current["elbow_flex"])
        current["shoulder_lift"] += 0.5 * (guarded["shoulder_lift"] - current["shoulder_lift"])
        scales.append(state.phase_scale)
    assert min(scales) < 1.0
    assert scales[-1] > min(scales)
    assert abs(current["shoulder_lift"]) / abs(current["elbow_flex"]) < 2.0

def test_sustained_opposite_encoder_response_fails_closed() -> None:
    from runtime.laptop.motion_guard import (
        CoordinatedGuardState,
        DEFAULT_TRACKING_LEAD_LIMITS,
        apply_coordinated_motion_guard,
    )

    joints = ("shoulder_lift", "elbow_flex")
    limits = {j: JointMotionLimits(HUGE, HUGE, HUGE) for j in joints}
    current = {j: 0.0 for j in joints}
    hold = DEFAULT_TRACKING_LEAD_LIMITS["elbow_flex"]
    state = CoordinatedGuardState(
        positions={"shoulder_lift": 0.0, "elbow_flex": -hold},
        velocities={j: 0.0 for j in joints},
        accelerations={j: 0.0 for j in joints},
        phase_scale=0.0,
        previous_actual_positions=dict(current),
    )
    with pytest.raises(MotionGuardError, match="sustained opposite"):
        for _ in range(660):
            current["elbow_flex"] += 0.1
            _, state = apply_coordinated_motion_guard(
                limits_by_joint=limits,
                current_state=current,
                target_now={"shoulder_lift": 10.0, "elbow_flex": -20.0},
                target_lookahead={"shoulder_lift": 10.0, "elbow_flex": -20.0},
                prev_state=state,
                dt_s=DT,
            )
