"""hardware/safety/single_joint_test_planner.py 단위 테스트.

이 파일은 순수 계산 로직만 검증한다 - 하드웨어/파일/네트워크 접근이 전혀 없다.
"""

from __future__ import annotations

import pytest

from hardware.safety.single_joint_test_planner import (
    BLOCKED,
    DEFAULT_MOTOR_RESOLUTION,
    MAX_ALLOWED_STEP_SIZE_DEG,
    MAX_ALLOWED_TOTAL_DELTA_DEG,
    MAX_COMMAND_RAW_DELTA_TICKS,
    MAX_READBACK_ABS_RAW_DELTA_TICKS,
    MIN_COMMAND_RAW_DELTA_TICKS,
    PASS,
    READBACK_DIRECTION_MISMATCH,
    READBACK_NO_MOTION,
    READBACK_OVERSHOOT,
    READBACK_PASS,
    REQUIRED_ARMED_STEP_SIZE_DEG,
    REQUIRED_ARMED_TOTAL_DELTA_DEG,
    TARGET_JOINT,
    PlannerConfigError,
    build_armed_single_step_plan,
    build_dry_run_plan,
    check_expected_start_matches,
    classify_readback,
    compute_calibration_range_deg,
    degrees_to_raw,
    raw_to_degrees,
)

# 실제 chanho_follower.json의 wrist_roll calibration (full-turn: range_min=0, range_max=4095).
WRIST_ROLL_RANGE_MIN = 0
WRIST_ROLL_RANGE_MAX = 4095


# ---------------------------------------------------------------------------
# raw tick <-> degree 변환 (lerobot 공식과의 round-trip 검증)
# ---------------------------------------------------------------------------


def test_raw_to_degrees_matches_lerobot_formula_at_range_bounds():
    # mid=(0+4095)/2=2047.5, max_res=4095 -> raw=0 => -180.0, raw=4095 => +180.0
    lo = raw_to_degrees(0, range_min=0, range_max=4095)
    hi = raw_to_degrees(4095, range_min=0, range_max=4095)
    assert lo == pytest.approx(-180.0)
    assert hi == pytest.approx(180.0)


def test_raw_to_degrees_midpoint_is_zero():
    mid_deg = raw_to_degrees(2047.5, range_min=0, range_max=4095)
    assert mid_deg == pytest.approx(0.0, abs=1e-9)


def test_degrees_to_raw_round_trips_with_raw_to_degrees():
    for raw in (0, 500, 2048, 3000, 4095):
        deg = raw_to_degrees(raw, range_min=0, range_max=4095)
        back = degrees_to_raw(deg, range_min=0, range_max=4095)
        assert back == pytest.approx(raw, abs=1)


def test_raw_to_degrees_does_not_clamp_out_of_range_input():
    # lerobot MotorsBus._normalize의 DEGREES 분기는 clamp하지 않는다 (모듈 docstring 근거).
    deg = raw_to_degrees(5000, range_min=0, range_max=4095)
    assert deg > 180.0


def test_equal_min_max_raises_planner_config_error():
    with pytest.raises(PlannerConfigError):
        raw_to_degrees(100, range_min=50, range_max=50)
    with pytest.raises(PlannerConfigError):
        degrees_to_raw(0.0, range_min=50, range_max=50)


# ---------------------------------------------------------------------------
# calibration 내부 안전 구간 (margin 적용) / full-turn 감지
# ---------------------------------------------------------------------------


def test_wrist_roll_calibration_is_detected_as_full_turn():
    cal = compute_calibration_range_deg(range_min=0, range_max=4095, margin_deg=15.0)
    assert cal.is_full_turn is True
    assert cal.full_deg_min == pytest.approx(-180.0)
    assert cal.full_deg_max == pytest.approx(180.0)
    assert cal.inner_deg_min == pytest.approx(-165.0)
    assert cal.inner_deg_max == pytest.approx(165.0)


def test_non_full_turn_joint_is_not_flagged_full_turn():
    cal = compute_calibration_range_deg(range_min=1052, range_max=2977, margin_deg=15.0)
    assert cal.is_full_turn is False


def test_negative_margin_rejected():
    with pytest.raises(PlannerConfigError):
        compute_calibration_range_deg(range_min=0, range_max=4095, margin_deg=-1.0)


# ---------------------------------------------------------------------------
# dry-run 계획: positive/negative 10 스텝
# ---------------------------------------------------------------------------


def _wrist_roll_plan(**overrides):
    kwargs = dict(
        start_deg=0.0,
        start_raw=2048,
        direction="positive",
        range_min=WRIST_ROLL_RANGE_MIN,
        range_max=WRIST_ROLL_RANGE_MAX,
    )
    kwargs.update(overrides)
    return build_dry_run_plan(**kwargs)


def test_positive_direction_generates_ten_tenth_degree_steps():
    plan = _wrist_roll_plan(direction="positive")
    assert len(plan.steps) == 10
    expected = [round(0.1 * i, 6) for i in range(1, 11)]
    actual = [round(s.delta_deg, 6) for s in plan.steps]
    assert actual == expected
    assert plan.steps[-1].target_deg == pytest.approx(1.0)
    assert plan.final_verdict == PASS


def test_negative_direction_generates_ten_tenth_degree_steps():
    plan = _wrist_roll_plan(direction="negative")
    assert len(plan.steps) == 10
    expected = [round(-0.1 * i, 6) for i in range(1, 11)]
    actual = [round(s.delta_deg, 6) for s in plan.steps]
    assert actual == expected
    assert plan.steps[-1].target_deg == pytest.approx(-1.0)
    assert plan.final_verdict == PASS


def test_direction_never_auto_switches_even_when_blocked():
    # 시작 위치를 이음매 바로 안쪽(양의 방향으로 가면 바로 이음매를 넘는 지점)에 두고
    # positive를 요청하면, negative로 자동 전환되지 않고 그냥 BLOCKED여야 한다.
    plan = _wrist_roll_plan(start_deg=164.95, direction="positive")
    assert plan.direction == "positive"  # 반대 방향으로 바뀌지 않았다
    assert plan.final_verdict == BLOCKED
    assert all(s.delta_deg > 0 for s in plan.steps)  # 모든 스텝이 여전히 양의 방향


def test_invalid_direction_raises():
    with pytest.raises(PlannerConfigError):
        _wrist_roll_plan(direction="sideways")


# ---------------------------------------------------------------------------
# 1° / 0.1° 상한 검증
# ---------------------------------------------------------------------------


def test_total_delta_exceeding_one_degree_is_rejected():
    with pytest.raises(PlannerConfigError):
        _wrist_roll_plan(requested_total_delta_deg=MAX_ALLOWED_TOTAL_DELTA_DEG + 0.1, step_size_deg=0.1)


def test_zero_or_negative_total_delta_is_rejected():
    with pytest.raises(PlannerConfigError):
        _wrist_roll_plan(requested_total_delta_deg=0.0)
    with pytest.raises(PlannerConfigError):
        _wrist_roll_plan(requested_total_delta_deg=-0.5)


def test_step_size_exceeding_point_one_degree_is_rejected():
    with pytest.raises(PlannerConfigError):
        _wrist_roll_plan(step_size_deg=MAX_ALLOWED_STEP_SIZE_DEG + 0.05)


def test_zero_or_negative_step_size_is_rejected():
    with pytest.raises(PlannerConfigError):
        _wrist_roll_plan(step_size_deg=0.0)
    with pytest.raises(PlannerConfigError):
        _wrist_roll_plan(step_size_deg=-0.1)


def test_step_size_not_evenly_dividing_total_delta_is_rejected():
    with pytest.raises(PlannerConfigError):
        _wrist_roll_plan(requested_total_delta_deg=1.0, step_size_deg=0.07)


# ---------------------------------------------------------------------------
# calibration 내부 범위 위반 -> BLOCKED
# ---------------------------------------------------------------------------


def test_step_violating_inner_calibration_range_is_blocked():
    # inner range는 [-165, 165]. start=164.95에서 +0.1 스텝 몇 개는 165를 넘는다.
    plan = _wrist_roll_plan(start_deg=164.95, direction="positive")
    assert any(s.calibration_check == BLOCKED for s in plan.steps)
    assert plan.final_verdict == BLOCKED


def test_start_position_outside_inner_band_blocks_entire_plan_without_computing_direction():
    # start=170은 이미 이음매 쪽 margin 구역 안(inner_max=165보다 큼) - 계획 전체 BLOCKED.
    plan = _wrist_roll_plan(start_deg=170.0, direction="positive")
    assert plan.final_verdict == BLOCKED
    assert any("이음매" in reason or "안전 구간" in reason for reason in plan.block_reasons)
    # 그래도 스텝 자체는 여전히 요청한 방향으로만 계산된다 (자동 반전 없음).
    assert all(s.calibration_check == BLOCKED for s in plan.steps)


def test_well_centered_start_position_passes_full_plan():
    plan = _wrist_roll_plan(start_deg=0.0, direction="positive")
    assert plan.final_verdict == PASS
    assert all(s.verdict == PASS for s in plan.steps)


# ---------------------------------------------------------------------------
# full-turn wrap 경계 검증
# ---------------------------------------------------------------------------


def test_target_beyond_physical_180_degree_bound_is_blocked():
    # margin_deg=0으로 두면 inner band == 물리적 [-180, 180] 그대로라, "180을 넘는 순간
    # BLOCKED"라는 물리적 경계 검사 자체를 margin 로직과 분리해서 검증할 수 있다.
    # start=179.5에서 positive 방향 1° 이동 시 179.6..180.5까지 감 - 180을 넘는 스텝만 BLOCKED.
    plan = _wrist_roll_plan(start_deg=179.5, direction="positive", margin_deg=0.0)
    under_or_at_180_steps = [s for s in plan.steps if s.target_deg <= 180.0]
    over_180_steps = [s for s in plan.steps if s.target_deg > 180.0]
    assert under_or_at_180_steps and over_180_steps  # 두 그룹 모두 존재해야 의미 있는 검증
    assert all(s.calibration_check == PASS for s in under_or_at_180_steps)
    assert all(s.calibration_check == BLOCKED for s in over_180_steps)
    assert plan.final_verdict == BLOCKED  # 스텝 하나라도 BLOCKED면 전체 계획이 BLOCKED


def test_raw_tick_near_wrap_seam_round_trips_without_modulo_assumption():
    # raw=1 (거의 0, 이음매 바로 옆)과 raw=4094 (거의 4095, 반대쪽 이음매 바로 옆)은
    # 물리적으로는 인접하지만 degree 선형 변환에서는 거의 -180/+180으로 정반대다.
    deg_near_zero_raw = raw_to_degrees(1, range_min=0, range_max=4095)
    deg_near_max_raw = raw_to_degrees(4094, range_min=0, range_max=4095)
    assert deg_near_zero_raw == pytest.approx(-180.0, abs=0.2)
    assert deg_near_max_raw == pytest.approx(180.0, abs=0.2)
    # 이 모듈은 이 둘을 "가깝다"고 취급하지 않는다 (모듈러 wrap 계산을 하지 않음).
    assert abs(deg_near_max_raw - deg_near_zero_raw) > 300


# ---------------------------------------------------------------------------
# 계획 결과 직렬화 (JSON 리포트에 들어갈 dict) - write_count / 민감정보 없음
# ---------------------------------------------------------------------------


def test_plan_to_dict_reports_write_count_zero_and_no_sensitive_fields():
    plan = _wrist_roll_plan(direction="positive")
    d = plan.to_dict()
    assert d["write_count"] == 0
    assert d["joint"] == TARGET_JOINT
    serialized = str(d)
    for forbidden in ("token", "Token", "TOKEN", "password", "secret", "Authorization", "/home/"):
        assert forbidden not in serialized


def test_plan_module_exposes_no_write_like_callables():
    """이 모듈에는 호출하면 실제로 무언가를 쓰는(write) 함수/메서드가 하나도 없어야 한다.

    ``WRITE_FAILED`` 같은 상태 라벨(문자열 상수 - armed writer가 반환하는 판정값 이름일
    뿐, 그 자체를 호출해도 아무 일도 일어나지 않는다)은 이름에 "write"가 들어가도
    무방하다 - 실제로 호출 가능한(callable) 대상만 감사한다.
    """
    import hardware.safety.single_joint_test_planner as planner_module

    public_names = [name for name in dir(planner_module) if not name.startswith("_")]
    banned_substrings = ("write", "send_action", "enable_torque", "disable_torque", "goal_position", "calibrate")
    for name in public_names:
        value = getattr(planner_module, name)
        if not callable(value):
            continue  # 문자열/숫자 상수(PASS, BLOCKED, WRITE_FAILED 등)는 감사 대상이 아니다.
        lowered = name.lower()
        for banned in banned_substrings:
            assert banned not in lowered, f"'{name}'에 금지된 패턴 '{banned}'가 포함되어 있습니다."


# ---------------------------------------------------------------------------
# armed 단발(0.1°) 계획: build_armed_single_step_plan
# ---------------------------------------------------------------------------


def test_armed_plan_requires_exactly_point_one_degree_delta_and_step():
    with pytest.raises(PlannerConfigError):
        build_armed_single_step_plan(
            start_deg=0.0, start_raw=2048, direction="positive", range_min=0, range_max=4095, requested_delta_deg=0.2
        )
    with pytest.raises(PlannerConfigError):
        build_armed_single_step_plan(
            start_deg=0.0, start_raw=2048, direction="positive", range_min=0, range_max=4095, step_size_deg=0.2
        )
    with pytest.raises(PlannerConfigError):
        build_armed_single_step_plan(
            start_deg=0.0, start_raw=2048, direction="positive", range_min=0, range_max=4095, requested_delta_deg=1.0
        )


def test_armed_plan_defaults_match_required_constants():
    assert REQUIRED_ARMED_TOTAL_DELTA_DEG == 0.1
    assert REQUIRED_ARMED_STEP_SIZE_DEG == 0.1
    assert MIN_COMMAND_RAW_DELTA_TICKS == 1
    assert MAX_COMMAND_RAW_DELTA_TICKS == 2


def test_armed_plan_well_centered_start_passes_with_one_or_two_tick_delta():
    start_raw = 2048
    start_deg = raw_to_degrees(start_raw, range_min=0, range_max=4095)
    plan = build_armed_single_step_plan(start_deg=start_deg, start_raw=start_raw, direction="positive", range_min=0, range_max=4095)
    assert plan.final_verdict == PASS
    assert MIN_COMMAND_RAW_DELTA_TICKS <= abs(plan.command_raw_delta) <= MAX_COMMAND_RAW_DELTA_TICKS


def test_armed_plan_negative_direction_moves_start_raw_down():
    start_raw = 2048
    start_deg = raw_to_degrees(start_raw, range_min=0, range_max=4095)
    plan = build_armed_single_step_plan(start_deg=start_deg, start_raw=start_raw, direction="negative", range_min=0, range_max=4095)
    assert plan.direction == "negative"
    assert plan.command_raw_delta < 0
    assert plan.final_verdict == PASS


def test_armed_plan_seam_start_is_blocked_with_zero_step_plan():
    # 이음매 근처(raw≈4090, 약 +179.6°)에서는 방향 계산 자체가 무의미하므로 BLOCKED.
    plan = build_armed_single_step_plan(start_deg=179.6, start_raw=4090, direction="positive", range_min=0, range_max=4095)
    assert plan.checks["seam_avoidance_start_check"] == BLOCKED
    assert plan.final_verdict == BLOCKED


def test_armed_plan_start_outside_inner_margin_is_blocked():
    # margin 기본 15도 -> inner range [-165, 165]. start=170은 그 밖.
    plan = build_armed_single_step_plan(start_deg=170.0, start_raw=3981, direction="positive", range_min=0, range_max=4095)
    assert plan.checks["seam_avoidance_start_check"] == BLOCKED
    assert plan.final_verdict == BLOCKED


def test_armed_plan_target_raw_out_of_bounds_is_blocked():
    # start_deg를 물리적 최댓값(+180°) 바로 아래에 둬서 +0.1° 이동 시 목표 raw가
    # 4095(motor_resolution-1)를 넘도록 강제한다. 이음매 회피 체크도 함께 BLOCKED되지만,
    # target_raw_bounds_check 자체가 독립적으로 BLOCKED로 기록되는지 확인하는 것이 목적이다.
    start_deg = 179.99
    start_raw = degrees_to_raw(start_deg, range_min=0, range_max=4095)
    plan = build_armed_single_step_plan(start_deg=start_deg, start_raw=start_raw, direction="positive", range_min=0, range_max=4095)
    assert plan.target_raw > 4095
    assert plan.checks["target_raw_bounds_check"] == BLOCKED
    assert plan.final_verdict == BLOCKED


def test_armed_plan_invalid_direction_raises():
    with pytest.raises(PlannerConfigError):
        build_armed_single_step_plan(start_deg=0.0, start_raw=2048, direction="sideways", range_min=0, range_max=4095)


def test_armed_plan_to_dict_has_no_write_count_leftover_field_and_serializes():
    plan = build_armed_single_step_plan(start_deg=0.0, start_raw=2048, direction="positive", range_min=0, range_max=4095)
    d = plan.to_dict()
    assert d["joint"] == TARGET_JOINT
    assert isinstance(d["checks"], dict)
    assert d["final_verdict"] in (PASS, BLOCKED)


# ---------------------------------------------------------------------------
# expected-start 확인: check_expected_start_matches
# ---------------------------------------------------------------------------


def test_expected_start_missing_both_is_blocked():
    result, reason = check_expected_start_matches(measured_raw=2048, measured_deg=0.0, expected_raw=None, expected_deg=None)
    assert result == BLOCKED
    assert reason is not None


def test_expected_start_raw_within_tolerance_passes():
    result, reason = check_expected_start_matches(measured_raw=2048, measured_deg=0.0, expected_raw=2049, expected_deg=None)
    assert result == PASS
    assert reason is None


def test_expected_start_raw_outside_tolerance_is_blocked():
    result, reason = check_expected_start_matches(measured_raw=2048, measured_deg=0.0, expected_raw=2060, expected_deg=None)
    assert result == BLOCKED


def test_expected_start_deg_within_tolerance_passes():
    result, _ = check_expected_start_matches(measured_raw=2048, measured_deg=0.0, expected_raw=None, expected_deg=0.2)
    assert result == PASS


def test_expected_start_deg_outside_tolerance_is_blocked():
    result, _ = check_expected_start_matches(measured_raw=2048, measured_deg=0.0, expected_raw=None, expected_deg=1.0)
    assert result == BLOCKED


def test_expected_start_both_provided_both_must_pass():
    # raw는 통과, deg는 실패 -> 전체 BLOCKED.
    result, _ = check_expected_start_matches(measured_raw=2048, measured_deg=0.0, expected_raw=2049, expected_deg=5.0)
    assert result == BLOCKED


# ---------------------------------------------------------------------------
# readback 판정: classify_readback
# ---------------------------------------------------------------------------


def test_classify_readback_zero_delta_is_no_motion():
    assert classify_readback(direction="positive", command_raw_delta=1, actual_raw_delta=0) == READBACK_NO_MOTION


def test_classify_readback_matching_direction_within_bound_is_pass():
    assert classify_readback(direction="positive", command_raw_delta=1, actual_raw_delta=1) == READBACK_PASS
    assert classify_readback(direction="negative", command_raw_delta=-2, actual_raw_delta=-2) == READBACK_PASS


def test_classify_readback_opposite_direction_is_direction_mismatch():
    assert classify_readback(direction="positive", command_raw_delta=1, actual_raw_delta=-1) == READBACK_DIRECTION_MISMATCH
    assert classify_readback(direction="negative", command_raw_delta=-1, actual_raw_delta=1) == READBACK_DIRECTION_MISMATCH


def test_classify_readback_exceeding_max_ticks_is_overshoot():
    over = MAX_READBACK_ABS_RAW_DELTA_TICKS + 2
    assert classify_readback(direction="positive", command_raw_delta=1, actual_raw_delta=over) == READBACK_OVERSHOOT


def test_classify_readback_at_exact_max_tick_boundary_is_pass():
    assert (
        classify_readback(direction="positive", command_raw_delta=1, actual_raw_delta=MAX_READBACK_ABS_RAW_DELTA_TICKS)
        == READBACK_PASS
    )
