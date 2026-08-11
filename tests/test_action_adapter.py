"""runtime/laptop/action_adapter.py 테스트."""

from __future__ import annotations

import math

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.action_adapter import GRIPPER_TINY_NEGATIVE_EPSILON, adapt_vla_action


def _good() -> dict[str, float]:
    return {joint: float(i) for i, joint in enumerate(JOINT_ORDER)}


def test_valid_action_passes_through_unchanged() -> None:
    raw = _good()
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert result.command_deg == raw
    assert result.warnings == ()


def test_wrong_dimension_rejected() -> None:
    raw = _good()
    del raw["wrist_roll"]
    result = adapt_vla_action(raw)
    assert result.valid is False
    assert "wrist_roll" in result.invalid_reason


def test_nan_rejected() -> None:
    raw = _good()
    raw["elbow_flex"] = math.nan
    result = adapt_vla_action(raw)
    assert result.valid is False


def test_inf_rejected() -> None:
    raw = _good()
    raw["elbow_flex"] = math.inf
    result = adapt_vla_action(raw)
    assert result.valid is False


def test_not_a_dict_rejected() -> None:
    result = adapt_vla_action([1, 2, 3, 4, 5, 6])
    assert result.valid is False


def test_gripper_far_outside_percent_range_warns_but_valid() -> None:
    raw = _good()
    raw["gripper"] = 500.0
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert any("gripper" in w for w in result.warnings)


def test_gripper_within_percent_range_no_warning() -> None:
    raw = _good()
    raw["gripper"] = 50.0
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert result.warnings == ()


# ---------------------------------------------------------------------------
# gripper tiny-negative normalization (2026-08 실물 Stage 3 조사 근거 - 모듈 docstring
# "gripper tiny-negative normalization" 절 참고). epsilon=GRIPPER_TINY_NEGATIVE_EPSILON=0.05.
# ---------------------------------------------------------------------------


def test_gripper_epsilon_constant_is_0_05() -> None:
    # 조사에서 근거 있는 후보 범위(0.05~0.1) 중 더 보수적인 하한을 쓴다는 결정 자체를
    # 회귀로부터 고정한다 - 값이 조용히 바뀌면 이 테스트가 먼저 깨져야 한다.
    assert GRIPPER_TINY_NEGATIVE_EPSILON == 0.05


def test_gripper_tiny_negative_normalized_to_zero() -> None:
    raw = _good()
    raw["gripper"] = -0.025
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert result.command_deg["gripper"] == 0.0


def test_gripper_exactly_at_epsilon_boundary_normalized_to_zero() -> None:
    raw = _good()
    raw["gripper"] = -0.05
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert result.command_deg["gripper"] == 0.0


def test_gripper_just_past_epsilon_boundary_not_normalized() -> None:
    raw = _good()
    raw["gripper"] = -0.051
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert result.command_deg["gripper"] == -0.051


def test_gripper_moderate_negative_not_normalized() -> None:
    raw = _good()
    raw["gripper"] = -0.3
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert result.command_deg["gripper"] == -0.3


def test_gripper_exactly_zero_unchanged() -> None:
    raw = _good()
    raw["gripper"] = 0.0
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert result.command_deg["gripper"] == 0.0


def test_gripper_positive_value_unchanged() -> None:
    raw = _good()
    raw["gripper"] = 0.5
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert result.command_deg["gripper"] == 0.5


def test_gripper_tiny_negative_normalization_does_not_touch_arm_joints() -> None:
    """gripper만 정규화 대상이다 - 다른 5개 arm 관절은 음수여도 절대 바뀌면 안 된다."""
    raw = _good()
    raw["gripper"] = -0.025
    for joint in JOINT_ORDER:
        if joint == "gripper":
            continue
        raw[joint] = -0.025  # arm 관절에도 동일하게 작은 음수를 줘서 대조군으로 삼는다
    result = adapt_vla_action(raw)
    assert result.valid is True
    assert result.command_deg["gripper"] == 0.0
    for joint in JOINT_ORDER:
        if joint == "gripper":
            continue
        assert result.command_deg[joint] == -0.025, f"{joint}가 정규화되면 안 되는데 바뀜: {result.command_deg[joint]}"
