"""runtime/common/vla_contract.py의 ``validate_joint_dict`` 검증."""

from __future__ import annotations

import math

from runtime.common.vla_contract import JOINT_ORDER, validate_joint_dict


def _good() -> dict[str, float]:
    return {joint: float(i) for i, joint in enumerate(JOINT_ORDER)}


def test_valid_dict_passes() -> None:
    result, reason = validate_joint_dict(_good(), context="test")
    assert reason is None
    assert result == _good()


def test_not_a_dict_rejected() -> None:
    result, reason = validate_joint_dict([1, 2, 3], context="test")
    assert result is None
    assert "object" in reason


def test_missing_joint_rejected() -> None:
    raw = _good()
    del raw["gripper"]
    result, reason = validate_joint_dict(raw, context="test")
    assert result is None
    assert "gripper" in reason


def test_extra_key_rejected() -> None:
    raw = _good()
    raw["elbow"] = 1.0
    result, reason = validate_joint_dict(raw, context="test")
    assert result is None
    assert "elbow" in reason


def test_nan_rejected() -> None:
    raw = _good()
    raw["wrist_roll"] = math.nan
    result, reason = validate_joint_dict(raw, context="test")
    assert result is None
    assert "NaN" in reason


def test_inf_rejected() -> None:
    raw = _good()
    raw["wrist_roll"] = math.inf
    result, reason = validate_joint_dict(raw, context="test")
    assert result is None
    assert "Inf" in reason


def test_non_numeric_rejected() -> None:
    raw = _good()
    raw["gripper"] = "20"
    result, reason = validate_joint_dict(raw, context="test")
    assert result is None
    assert "gripper" in reason


def test_bool_rejected_as_non_numeric() -> None:
    raw = _good()
    raw["gripper"] = True
    result, reason = validate_joint_dict(raw, context="test")
    assert result is None
