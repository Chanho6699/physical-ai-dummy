"""runtime/laptop/intent_validation.py의 PolicyIntentValidator 단위 검증 (Phase C-3A.1
correction). 이 모듈은 SafetyGate.evaluate()를 재구현하지 않고 위임만 하므로, 여기서는
주로 (1) 위임이 정확히 이뤄지는지, (2) 반환 타입/필드 매핑이 올바른지, (3) 실제
재캘리브레이션된 threshold 기준 정상/위험 케이스를 검증한다. SafetyGate 자체의 로직
정확성은 tests/test_safety_gate.py가 이미 검증한다 - 여기서 중복하지 않는다.
"""

from __future__ import annotations

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.intent_validation import IntentValidationResult, PolicyIntentValidator
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig


def _neutral(value: float = 0.0) -> dict[str, float]:
    return {j: value for j in JOINT_ORDER}


def _real_validator() -> PolicyIntentValidator:
    return PolicyIntentValidator(SafetyGate(SafetyGateConfig.from_repo_defaults()))


# ---------------------------------------------------------------------------
# 위임 정확성 - PolicyIntentValidator는 SafetyGate.evaluate()의 decision/reasons를
# 그대로(재해석만 해서) 전달해야 한다.
# ---------------------------------------------------------------------------


def test_check_intent_delegates_to_same_safety_gate_instance() -> None:
    gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    validator = PolicyIntentValidator(gate)
    current = _neutral(0.0)
    raw = dict(current)
    raw["elbow_flex"] = 7.08  # threshold(8.50) 아래 - ACCEPT

    result = validator.check_intent(raw_target_deg=raw, current_state_deg=current)
    direct = gate.evaluate(
        adapted_action=__import__("runtime.laptop.action_adapter", fromlist=["adapt_vla_action"]).adapt_vla_action(raw),
        current_state_deg=current, observation_valid=True,
    )
    assert result.decision == direct.decision
    assert result.reasons == direct.reasons
    assert result.valid == (direct.decision == "ACCEPT")


def test_intent_validation_result_is_frozen_dataclass_with_expected_fields() -> None:
    result = IntentValidationResult(valid=True, decision="ACCEPT", reasons=())
    assert result.valid is True
    assert result.decision == "ACCEPT"
    assert result.reasons == ()
    try:
        result.valid = False  # type: ignore[misc]
        assert False, "frozen dataclass should reject mutation"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 정상 Candidate B target (재캘리브레이션된 threshold: pan 6.00/lift 9.00/elbow 8.50/
# wrist_flex 4.01/wrist_roll 1.15/gripper 9.17) - ACCEPT
# ---------------------------------------------------------------------------


def test_normal_elbow_delta_within_threshold_is_valid() -> None:
    validator = _real_validator()
    current = _neutral(0.0)
    raw = dict(current)
    raw["elbow_flex"] = 7.08  # < 8.50
    result = validator.check_intent(raw_target_deg=raw, current_state_deg=current)
    assert result.valid is True
    assert result.decision == "ACCEPT"


def test_normal_pan_delta_within_threshold_is_valid() -> None:
    validator = _real_validator()
    current = _neutral(0.0)
    raw = dict(current)
    raw["shoulder_pan"] = 5.45  # < 6.00
    result = validator.check_intent(raw_target_deg=raw, current_state_deg=current)
    assert result.valid is True
    assert result.decision == "ACCEPT"


def test_normal_lift_delta_within_threshold_is_valid() -> None:
    validator = _real_validator()
    current = _neutral(0.0)
    raw = dict(current)
    raw["shoulder_lift"] = 5.63  # < 9.00
    result = validator.check_intent(raw_target_deg=raw, current_state_deg=current)
    assert result.valid is True
    assert result.decision == "ACCEPT"


# ---------------------------------------------------------------------------
# 위험한 wrist outlier (실물 사례 재현: wrist_flex 53.67 -> 40.49, delta=13.18deg) - 반드시
# WOULD_CLAMP(=valid False), threshold(4.01)의 gross 5x(20.05deg) 안이므로 REJECT는 아님.
# ---------------------------------------------------------------------------


def test_dangerous_wrist_delta_is_invalid_would_clamp() -> None:
    validator = _real_validator()
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703
    raw = dict(current)
    raw["wrist_flex"] = 40.4879
    result = validator.check_intent(raw_target_deg=raw, current_state_deg=current)
    assert result.valid is False
    assert result.decision == "WOULD_CLAMP"
    assert len(result.reasons) > 0


def test_grossly_dangerous_wrist_delta_is_invalid_reject() -> None:
    """gross multiplier(5x=20.05deg)를 넘는 훨씬 더 큰 delta는 REJECT여야 한다 - 이
    경계도 Intent 단계에서 valid=False로 정확히 분류되는지 확인."""
    validator = _real_validator()
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703
    raw = dict(current)
    raw["wrist_flex"] = 53.6703 - 25.0  # delta=25.0 > 20.05
    result = validator.check_intent(raw_target_deg=raw, current_state_deg=current)
    assert result.valid is False
    assert result.decision == "REJECT"


# ---------------------------------------------------------------------------
# adapt_vla_action 경유 확인 - gripper tiny-negative epsilon 등 기존 adapter 로직이
# Intent 단계에서도 동일하게 적용돼야 한다(별도 재구현 없이 재사용).
# ---------------------------------------------------------------------------


def test_gripper_tiny_negative_is_normalized_before_intent_check() -> None:
    validator = _real_validator()
    current = _neutral(0.0)
    current["gripper"] = 0.0
    raw = dict(current)
    raw["gripper"] = -0.02  # GRIPPER_TINY_NEGATIVE_EPSILON(0.05) 안 - 0.0으로 정규화돼야 함
    result = validator.check_intent(raw_target_deg=raw, current_state_deg=current)
    assert result.valid is True
    assert result.decision == "ACCEPT"
