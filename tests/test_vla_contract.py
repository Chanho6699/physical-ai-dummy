"""runtime/common/vla_contract.py의 ``validate_joint_dict``/``validate_action_chunk`` 검증."""

from __future__ import annotations

import math

from runtime.common.vla_contract import JOINT_ORDER, validate_action_chunk, validate_joint_dict


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


# ---------------------------------------------------------------------------
# validate_action_chunk (Phase C-1A, /predict_chunk 전용 - 섹션 8 fail-closed 요구사항)
# ---------------------------------------------------------------------------


def _good_chunk(n: int = 3) -> list[dict[str, float]]:
    return [_good() for _ in range(n)]


def test_valid_chunk_passes() -> None:
    chunk, reason = validate_action_chunk(_good_chunk(), context="chunk")
    assert reason is None
    assert chunk == _good_chunk()


def test_chunk_not_a_list_rejected() -> None:
    chunk, reason = validate_action_chunk({"not": "a list"}, context="chunk")
    assert chunk is None
    assert "list" in reason


def test_empty_chunk_rejected() -> None:
    chunk, reason = validate_action_chunk([], context="chunk")
    assert chunk is None
    assert "비어" in reason


def test_chunk_length_mismatch_rejected() -> None:
    chunk, reason = validate_action_chunk(_good_chunk(3), context="chunk", expected_length=50)
    assert chunk is None
    assert "길이" in reason


def test_chunk_length_match_passes() -> None:
    chunk, reason = validate_action_chunk(_good_chunk(50), context="chunk", expected_length=50)
    assert reason is None
    assert len(chunk) == 50


def test_chunk_with_one_nan_element_rejects_whole_chunk() -> None:
    """원소 하나만 무효해도 chunk 전체를 무효 처리한다 - "일부만 쓸 수 있는 chunk"는 없다."""
    raw = _good_chunk(5)
    raw[3]["wrist_roll"] = math.nan
    chunk, reason = validate_action_chunk(raw, context="chunk")
    assert chunk is None
    assert "NaN" in reason
    assert "chunk[3]" in reason


def test_chunk_with_missing_joint_in_one_element_rejected() -> None:
    raw = _good_chunk(5)
    del raw[1]["gripper"]
    chunk, reason = validate_action_chunk(raw, context="chunk")
    assert chunk is None
    assert "gripper" in reason


def test_chunk_element_not_dict_rejected() -> None:
    raw = _good_chunk(3)
    raw[2] = [1, 2, 3, 4, 5, 6]  # dict가 아님
    chunk, reason = validate_action_chunk(raw, context="chunk")
    assert chunk is None
