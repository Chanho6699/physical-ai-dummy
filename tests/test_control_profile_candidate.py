"""hardware/diagnostics/control_profile_candidate.py 단위 테스트 - 전부 offline, 합성 fixture.

실제 aggregate_6runs_*.json의 schema를 최소한으로 흉내낸 합성 dict만 쓴다. lerobot/serial
접근이 전혀 없으므로 lerobot 설치 여부와 무관하게 항상 실행 가능해야 한다.
"""

from __future__ import annotations

import copy

import pytest

from hardware.diagnostics.control_profile_candidate import (
    CONFIDENCE_HIGH,
    CONFIDENCE_INSUFFICIENT_DATA,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    DEG_PER_WRIST_ROLL_TICK,
    GRIPPER_JOINT,
    LATENCY_SCOPE_LOCAL_INSTRUMENTED_TELEOP,
    VERDICT_LOOSER,
    VERDICT_MORE_CONSERVATIVE,
    VERDICT_NO_EXISTING_LIMIT,
    VERDICT_NO_OBSERVED_DATA,
    ControlProfileCandidateError,
    build_control_profile_candidate,
    compare_with_existing_rate_limits,
    confidence_from_cv,
    tick_to_degree_table,
)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _joint_aggregate_block(*, range_min=-10.0, range_max=10.0, p01=-9.0, p99=9.0) -> dict:
    return {
        "joint": "x",
        "aggregate": {
            "range": {"min": range_min, "max": range_max, "p01": p01, "p99": p99, "run_min_max_spread": range_max - range_min},
            "frame_delta": {"mean": 0.1, "p50": 0.0, "p90": 0.5, "p95": 0.8, "p99": 1.2, "max": 1.5, "min": 0.0, "n": 100},
            "velocity": {"mean": 5.0, "p50": 0.0, "p90": 20.0, "p95": 30.0, "p99": 40.0, "max": 50.0, "min": 0.0, "n": 100},
            "tracking_error": {"mean": 1.0, "p50": 0.5, "p90": 2.0, "p95": 3.0, "p99": 4.0, "max": 5.0, "min": 0.0, "n": 100},
        },
    }


def _stability_block(*, tracking_cv=0.1, velocity_cv=0.1) -> dict:
    return {
        "tracking_error_mean_across_runs": {"mean": 1.0, "std": 0.1, "cv": tracking_cv, "n": 6},
        "mean_velocity_across_runs": {"mean": 5.0, "std": 0.5, "cv": velocity_cv, "n": 6},
    }


def _deadband_bucket(ticks, region, *, response_fraction=0.0, sample_count=100, runs_with_any_response=0):
    return {
        "abs_goal_present_error_ticks": ticks,
        "sample_count": sample_count,
        "response_count": int(response_fraction * sample_count),
        "no_response_count": sample_count - int(response_fraction * sample_count),
        "opposite_motion_count": 0,
        "response_fraction": response_fraction,
        "region_candidate": region,
        "runs_with_any_response": runs_with_any_response,
    }


def _minimal_aggregate(*, run_count=6, cv_by_joint: dict | None = None, deadband_buckets=None) -> dict:
    cv_by_joint = cv_by_joint or {}
    joint_aggregates = {j: _joint_aggregate_block() for j in JOINT_NAMES}
    per_joint_stability = {}
    for j in JOINT_NAMES:
        tcv, vcv = cv_by_joint.get(j, (0.1, 0.1))
        per_joint_stability[j] = _stability_block(tracking_cv=tcv, velocity_cv=vcv)

    if deadband_buckets is None:
        deadband_buckets = [
            _deadband_bucket(0, "NO_RESPONSE_REGION", response_fraction=0.0),
            _deadband_bucket(1, "NO_RESPONSE_REGION", response_fraction=0.001, runs_with_any_response=1),
            _deadband_bucket(2, "NO_RESPONSE_REGION", response_fraction=0.002, runs_with_any_response=1),
            _deadband_bucket(3, "NO_RESPONSE_REGION", response_fraction=0.005, runs_with_any_response=3),
            _deadband_bucket(4, "NO_RESPONSE_REGION", response_fraction=0.002, runs_with_any_response=2),
            _deadband_bucket(5, "NO_RESPONSE_REGION", response_fraction=0.003, runs_with_any_response=2),
            _deadband_bucket("6+", "TRANSITION_REGION", response_fraction=0.706, runs_with_any_response=4),
        ]

    return {
        "generated_at": "2026-08-07T09:45:32+00:00",
        "run_count": run_count,
        "joint_aggregates": joint_aggregates,
        "run_to_run_stability": {
            "actual_loop_hz": {"mean": 59.13, "std": 0.07, "cv": 0.0012, "n": 6},
            "latency_ms": {"mean": 88.76, "std": 13.97, "cv": 0.157, "n": 4},
            "per_joint": per_joint_stability,
        },
        "latency_aggregate": {
            "verdict": "AVAILABLE",
            "n_runs_with_valid_lag": 4,
            "n_runs_total": 6,
            "lag_ms_median": 92.9,
            "lag_ms_mean": 88.76,
            "lag_ms_min": 67.69,
            "lag_ms_max": 101.57,
            "lag_ms_std": 13.97,
        },
        "deadband_aggregate": {
            "verdict": "DEADBAND_AGGREGATE_AVAILABLE",
            "buckets": deadband_buckets,
        },
    }


# ---------------------------------------------------------------------------
# 1. aggregate JSON parsing / provenance / schema 검증
# ---------------------------------------------------------------------------


def test_missing_required_keys_raise_error():
    with pytest.raises(ControlProfileCandidateError):
        build_control_profile_candidate({"run_count": 6})


def test_provenance_fields_present_and_correct():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg, source_aggregate_path="reports/x/aggregate_6runs_x.json")

    assert profile["status"] == "CANDIDATE_ONLY"
    assert profile["source"] == "instrumented_teleop_6runs"
    assert profile["run_count"] == 6
    assert profile["apply_automatically"] is False
    assert profile["source_aggregate_path"] == "reports/x/aggregate_6runs_x.json"
    assert profile["source_aggregate_generated_at"] == agg["generated_at"]


# ---------------------------------------------------------------------------
# 2. CANDIDATE_ONLY 라벨이 모든 하위 블록에 붙는지 + hard_limit이라 부르지 않는지
# ---------------------------------------------------------------------------


def test_candidate_only_label_present_everywhere():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)

    for joint in JOINT_NAMES:
        assert profile["joints"][joint]["label"] == "CANDIDATE_ONLY"
    assert profile["wrist_roll_deadband_analysis"]["label"] == "CANDIDATE_ONLY"
    assert profile["timing"]["label"] == "CANDIDATE_ONLY"


def test_historical_range_not_called_hard_limit():
    """섹션 4-A 요구사항: 필드 이름 자체가 ``historical_operating_range``여야 하고, ``hard_limit``
    이라는 키가 있으면 안 된다 (설명 문구 안에서 "이것은 hard_limit이 아니다"라고 명시적으로
    부인하는 것은 허용된다 - 오히려 요구사항이다).
    """
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    joint_block = profile["joints"]["shoulder_pan"]
    assert "historical_operating_range" in joint_block
    assert "hard_limit" not in joint_block
    rng = joint_block["historical_operating_range"]
    assert "hard_limit" not in rng
    assert rng["candidate_historical_inner_range"] == [-9.0, 9.0]
    assert rng["observed_min"] == -10.0 and rng["observed_max"] == 10.0


# ---------------------------------------------------------------------------
# 3. wrist_roll deadband mapping / HIGH_RESPONSE 미확립 처리
# ---------------------------------------------------------------------------


def test_wrist_roll_deadband_no_response_region_is_0_to_5():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    wr = profile["wrist_roll_deadband_analysis"]
    assert wr["no_response_region_ticks"] == [0, 5]
    assert wr["transition_region_start_ticks"] == 6


def test_wrist_roll_high_response_region_not_established_at_70_percent():
    """6+ tick response_fraction=70.6%는 TRANSITION_REGION이지 HIGH_RESPONSE_REGION이 아니다 -
    이 모듈은 이 사실을 절대 'HIGH_RESPONSE_REGION'이나 'guaranteed motion'으로 승격하면 안 된다.
    """
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    wr = profile["wrist_roll_deadband_analysis"]
    assert wr["high_response_region"] == "NOT_ESTABLISHED"
    assert "guaranteed" not in wr["note"].lower()
    assert "70.6" in wr["high_response_region_rationale"]


def test_wrist_roll_transition_region_confidence_is_not_high():
    """6개 run 중 2개가 이 구간에서 표본이 없었으므로(응답 없음이 아니라 데이터 없음) 과신 금지."""
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    wr = profile["wrist_roll_deadband_analysis"]
    assert wr["transition_region_confidence"] != CONFIDENCE_HIGH


def test_wrist_roll_no_response_region_confidence_is_high_when_consistently_near_zero():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    wr = profile["wrist_roll_deadband_analysis"]
    assert wr["no_response_region_confidence"] == CONFIDENCE_HIGH


def test_wrist_roll_high_response_region_established_only_when_bucket_says_so():
    """만약 미래에 실제로 HIGH_RESPONSE_REGION 버킷이 aggregate에 나타나면(예: 다른 tick에서
    >80% 반응), 그때도 이 모듈은 재분류(reclassification)를 하지 않고 기존
    ``deadband_aggregate``의 ``region_candidate``를 그대로 신뢰한다 - 하지만 여전히
    'high_response_region' 최상위 필드는 NOT_ESTABLISHED로 고정한다(섹션 5 요구사항: 이번
    작업 범위에서 HIGH_RESPONSE_REGION을 확정하지 않는다).
    """
    buckets = [
        _deadband_bucket(0, "NO_RESPONSE_REGION", response_fraction=0.0),
        _deadband_bucket(1, "NO_RESPONSE_REGION", response_fraction=0.0),
        _deadband_bucket(2, "NO_RESPONSE_REGION", response_fraction=0.0),
        _deadband_bucket(3, "NO_RESPONSE_REGION", response_fraction=0.0),
        _deadband_bucket(4, "NO_RESPONSE_REGION", response_fraction=0.0),
        _deadband_bucket(5, "HIGH_RESPONSE_REGION", response_fraction=0.95, runs_with_any_response=6),
        _deadband_bucket("6+", "HIGH_RESPONSE_REGION", response_fraction=0.98, runs_with_any_response=6),
    ]
    agg = _minimal_aggregate(deadband_buckets=buckets)
    profile = build_control_profile_candidate(agg)
    wr = profile["wrist_roll_deadband_analysis"]
    assert wr["high_response_region"] == "NOT_ESTABLISHED"


def test_wrist_roll_deadband_insufficient_when_aggregate_unavailable():
    agg = _minimal_aggregate()
    agg["deadband_aggregate"] = {"verdict": "INSUFFICIENT_FOR_DEADBAND_ESTIMATE"}
    profile = build_control_profile_candidate(agg)
    wr = profile["wrist_roll_deadband_analysis"]
    assert wr["verdict"] == "INSUFFICIENT_DATA"
    assert wr["high_response_region"] == "NOT_ESTABLISHED"
    assert wr["no_response_region_ticks"] is None


def test_wrist_roll_degree_equivalents_match_expected_values():
    """섹션 5 요구사항의 실측 값(1 tick ~= 0.0879 deg, 5 tick ~= 0.4396 deg, 6 tick ~= 0.5275 deg)와 일치해야 한다."""
    table = tick_to_degree_table(6)
    assert table["1"] == pytest.approx(0.0879, abs=1e-3)
    assert table["5"] == pytest.approx(0.4396, abs=1e-3)
    assert table["6"] == pytest.approx(0.5275, abs=1e-3)
    assert DEG_PER_WRIST_ROLL_TICK == pytest.approx(360.0 / 4095.0)


# ---------------------------------------------------------------------------
# 4. Timing scope 표시
# ---------------------------------------------------------------------------


def test_timing_scope_is_labeled_local_not_end_to_end():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    timing = profile["timing"]
    assert timing["latency_scope"] == LATENCY_SCOPE_LOCAL_INSTRUMENTED_TELEOP
    assert "end-to-end" in timing["latency_scope_note"] or "end to end" in timing["latency_scope_note"].lower()
    assert timing["observed_latency_ms"]["valid_runs"] == "4/6"


def test_timing_latency_confidence_capped_when_not_all_runs_valid():
    """latency cv 자체는 낮아도(예: 0.157) valid_runs가 6/6이 아니면 HIGH로 과신하지 않는다."""
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    assert profile["timing"]["latency_confidence"] != CONFIDENCE_HIGH


# ---------------------------------------------------------------------------
# 5. confidence mapping (cv -> HIGH/MEDIUM/LOW/INSUFFICIENT_DATA)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cv", "expected"),
    [
        (None, CONFIDENCE_INSUFFICIENT_DATA),
        (0.0, CONFIDENCE_HIGH),
        (0.29, CONFIDENCE_HIGH),
        (0.3, CONFIDENCE_MEDIUM),
        (0.59, CONFIDENCE_MEDIUM),
        (0.6, CONFIDENCE_LOW),
        (2.0, CONFIDENCE_LOW),
    ],
)
def test_confidence_from_cv_thresholds(cv, expected):
    assert confidence_from_cv(cv) == expected


def test_joint_confidence_reflects_run_to_run_stability():
    agg = _minimal_aggregate(
        cv_by_joint={
            "elbow_flex": (0.1, 0.1),  # HIGH
            "wrist_roll": (0.9, 0.9),  # LOW
        }
    )
    profile = build_control_profile_candidate(agg)
    assert profile["joints"]["elbow_flex"]["confidence"] == CONFIDENCE_HIGH
    assert profile["joints"]["wrist_roll"]["confidence"] == CONFIDENCE_LOW


def test_joint_confidence_insufficient_when_no_stability_data():
    agg = _minimal_aggregate()
    agg["run_to_run_stability"]["per_joint"].pop("gripper")
    profile = build_control_profile_candidate(agg)
    assert profile["joints"]["gripper"]["confidence"] == CONFIDENCE_INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# 6. gripper unit handling
# ---------------------------------------------------------------------------


def test_gripper_unit_is_percent_not_degree():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    assert profile["joints"][GRIPPER_JOINT]["unit"] == "percent_0_100"
    for joint in JOINT_NAMES:
        if joint == GRIPPER_JOINT:
            continue
        assert profile["joints"][joint]["unit"] == "degree"
    assert "percent_0_100" in profile["gripper_unit_note"]


def test_gripper_comparison_row_flags_unit_ambiguity():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    comparison = compare_with_existing_rate_limits(profile, {j: 20.0 for j in JOINT_NAMES})
    assert "percent" in comparison["joints"][GRIPPER_JOINT]["existing_rate_limit_unit"].lower()
    assert comparison["joints"]["shoulder_pan"]["existing_rate_limit_unit"] == "deg/s"


# ---------------------------------------------------------------------------
# 7. 기존 rate limit 비교 verdict
# ---------------------------------------------------------------------------


def test_comparison_verdict_more_conservative_when_existing_limit_below_observed_p95():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    # fixture 관절들의 velocity p95는 30.0 (see _joint_aggregate_block) - existing limit을 더 낮게 둔다.
    comparison = compare_with_existing_rate_limits(profile, {"shoulder_pan": 5.0})
    assert comparison["joints"]["shoulder_pan"]["verdict"] == VERDICT_MORE_CONSERVATIVE


def test_comparison_verdict_looser_when_existing_limit_above_observed_p95():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    comparison = compare_with_existing_rate_limits(profile, {"shoulder_pan": 500.0})
    assert comparison["joints"]["shoulder_pan"]["verdict"] == VERDICT_LOOSER


def test_comparison_verdict_no_existing_limit_when_joint_missing():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    comparison = compare_with_existing_rate_limits(profile, {})
    assert comparison["joints"]["shoulder_pan"]["verdict"] == VERDICT_NO_EXISTING_LIMIT


def test_comparison_verdict_no_observed_data_when_joint_aggregate_missing():
    agg = _minimal_aggregate()
    del agg["joint_aggregates"]["wrist_flex"]
    profile = build_control_profile_candidate(agg)
    assert profile["joints"]["wrist_flex"]["confidence"] == CONFIDENCE_INSUFFICIENT_DATA
    comparison = compare_with_existing_rate_limits(profile, {"wrist_flex": 15.0})
    assert comparison["joints"]["wrist_flex"]["verdict"] == VERDICT_NO_OBSERVED_DATA


def test_comparison_does_not_mutate_inputs():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    existing = {"shoulder_pan": 20.0}
    existing_before = copy.deepcopy(existing)
    profile_before = copy.deepcopy(profile)
    compare_with_existing_rate_limits(profile, existing)
    assert existing == existing_before
    assert profile == profile_before


# ---------------------------------------------------------------------------
# 8. apply_automatically / usage_restrictions
# ---------------------------------------------------------------------------


def test_apply_automatically_is_false_and_restrictions_present():
    agg = _minimal_aggregate()
    profile = build_control_profile_candidate(agg)
    assert profile["apply_automatically"] is False
    assert len(profile["usage_restrictions"]) >= 1
    assert all("적용" in r or "connect" in r or "safety" in r.lower() for r in profile["usage_restrictions"])


# ---------------------------------------------------------------------------
# 9. 이 모듈은 lerobot/하드웨어를 import하지 않는다
# ---------------------------------------------------------------------------


def test_module_never_imports_lerobot_or_hardware_safety():
    import inspect

    import hardware.diagnostics.control_profile_candidate as module

    source = inspect.getsource(module)
    for forbidden in ("import lerobot", "from lerobot", "FeetechMotorsBus", "hardware.safety.single_joint"):
        assert forbidden not in source
