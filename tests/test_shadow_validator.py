"""runtime/laptop/shadow_validator.py 테스트."""

from __future__ import annotations

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.mujoco_shadow_backend import MuJoCoStepResult
from runtime.laptop.shadow_validator import OSCILLATION_QVEL_WARN, TRACKING_ERROR_WARN_DEG, validate_mujoco_step


def _zero() -> dict[str, float]:
    return {j: 0.0 for j in JOINT_ORDER}


def _diagnostics(*, rate_limited: set[str] = frozenset()) -> dict[str, dict]:
    return {
        j: {
            "joint": j,
            "raw_action": 0.0,
            "processed_action": 0.0,
            "simulated_actual_state": 0.0,
            "command_delta": 0.0,
            "deadband_applied": False,
            "latency_applied_ms": 0.0,
            "rate_limited": j in rate_limited,
            "outside_historical_range": False,
            "tracking_error": None,
        }
        for j in JOINT_ORDER
    }


def _result(**overrides) -> MuJoCoStepResult:
    base = dict(
        initial_state_deg=_zero(),
        processed_command_deg=_zero(),
        final_state_deg=_zero(),
        final_state_finite=True,
        joint_limits_deg={j: (-90.0, 90.0) for j in JOINT_ORDER},
        mechanical_violations=(),
        tracking_error_deg={j: 0.0 for j in JOINT_ORDER},
        max_abs_qvel=0.0,
        realistic_diagnostics=_diagnostics(),
    )
    base.update(overrides)
    return MuJoCoStepResult(**base)


def test_clean_step_passes() -> None:
    result = validate_mujoco_step(_result())
    assert result.passed is True
    assert result.failures == ()


def test_non_finite_state_fails_immediately() -> None:
    result = validate_mujoco_step(_result(final_state_finite=False))
    assert result.passed is False
    assert any("NaN" in f or "Inf" in f for f in result.failures)
    # 발산 상태에서는 이후 검사가 의미 없으므로 checks에 finite 하나만 채워진다.
    assert set(result.checks) == {"final_state_finite"}


def test_mechanical_violation_fails() -> None:
    result = validate_mujoco_step(_result(mechanical_violations=("shoulder_pan: 95.00deg (범위 [-90.00, 90.00])",)))
    assert result.passed is False
    assert result.checks["no_mechanical_violation"] is False


def test_rate_limited_is_warning_not_failure() -> None:
    result = validate_mujoco_step(_result(realistic_diagnostics=_diagnostics(rate_limited={"wrist_roll"})))
    assert result.passed is True
    assert any("wrist_roll" in w for w in result.warnings)


def test_severe_oscillation_fails() -> None:
    result = validate_mujoco_step(_result(max_abs_qvel=OSCILLATION_QVEL_WARN + 1.0))
    assert result.passed is False
    assert result.checks["no_severe_oscillation"] is False


def test_incomplete_realistic_diagnostics_fails() -> None:
    diag = _diagnostics()
    del diag["gripper"]
    result = validate_mujoco_step(_result(realistic_diagnostics=diag))
    assert result.passed is False
    assert result.checks["realistic_layer_complete"] is False


def test_large_tracking_error_is_warning_only() -> None:
    tracking_error = {j: 0.0 for j in JOINT_ORDER}
    tracking_error["wrist_flex"] = TRACKING_ERROR_WARN_DEG + 5.0
    result = validate_mujoco_step(_result(tracking_error_deg=tracking_error))
    assert result.passed is True
    assert result.checks["tracking_error_within_bounds"] is False
    assert any("wrist_flex" in w for w in result.warnings)


def test_direction_mismatch_is_warning() -> None:
    command = _zero()
    command["shoulder_pan"] = 10.0  # 명령은 +방향
    final = _zero()
    final["shoulder_pan"] = -10.0  # 실제는 -방향으로 움직임 (mismatch)
    result = validate_mujoco_step(_result(processed_command_deg=command, final_state_deg=final))
    assert result.passed is True
    assert result.checks["direction_consistent"] is False


def test_small_command_delta_is_not_direction_checked() -> None:
    command = _zero()
    command["shoulder_pan"] = 0.05  # DIRECTION_CHECK_MIN_DELTA_DEG(0.5)보다 작음
    final = _zero()
    final["shoulder_pan"] = -0.05
    result = validate_mujoco_step(_result(processed_command_deg=command, final_state_deg=final))
    assert result.checks["direction_consistent"] is True
