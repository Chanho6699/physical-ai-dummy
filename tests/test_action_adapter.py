"""runtime/laptop/action_adapter.py 테스트."""

from __future__ import annotations

import math

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.action_adapter import adapt_vla_action


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
