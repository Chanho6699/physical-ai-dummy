"""runtime/laptop/safety_gate.py 테스트 - ACCEPT/WOULD_CLAMP/REJECT 3분류 검증."""

from __future__ import annotations

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.action_adapter import AdaptedAction
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig

RANGE = (-90.0, 90.0)
GRIPPER_RANGE = (0.0, 100.0)
MAX_STEP = 5.0


def _config() -> SafetyGateConfig:
    joint_range = {j: (GRIPPER_RANGE if j == "gripper" else RANGE) for j in JOINT_ORDER}
    max_step = {j: MAX_STEP for j in JOINT_ORDER}
    return SafetyGateConfig(joint_range_deg=joint_range, max_step_deg=max_step)


def _current_state(value: float = 0.0) -> dict[str, float]:
    return {j: value for j in JOINT_ORDER}


def _action(**overrides) -> AdaptedAction:
    command = _current_state(0.0)
    command.update(overrides)
    return AdaptedAction(valid=True, command_deg=command)


def _gate() -> SafetyGate:
    return SafetyGate(_config())


def test_accept_when_within_range_and_small_step() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(shoulder_pan=2.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "ACCEPT"
    assert decision.would_clamp is False
    assert decision.safe_action["shoulder_pan"] == 2.0


def test_would_clamp_on_mild_mechanical_range_violation() -> None:
    # 현재/명령 모두 range 경계 근처 - step 자체는 작게 유지해 rate limit이 아니라
    # mechanical range 위반만 단독으로 관측되게 한다. shoulder_pan만 90도에서 시작하고
    # 나머지 관절은 command/current 둘 다 기본값(0.0)으로 맞춰 다른 관절에서 우발적인
    # excessive-step REJECT가 섞이지 않게 한다.
    current = _current_state(0.0)
    current["shoulder_pan"] = 90.0
    decision = _gate().evaluate(adapted_action=_action(shoulder_pan=92.0), current_state_deg=current, observation_valid=True)
    assert decision.decision == "WOULD_CLAMP"
    assert decision.safe_action["shoulder_pan"] == 90.0
    assert any("MECHANICAL_LIMIT_CLAMPED" in r for r in decision.reasons)


def test_reject_on_gross_mechanical_range_violation() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(shoulder_pan=10_000.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "REJECT"
    assert decision.safe_action is None
    assert any("MECHANICAL_LIMIT_GROSS_VIOLATION" in r for r in decision.reasons)


def test_would_clamp_on_excessive_step() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(wrist_flex=8.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "WOULD_CLAMP"
    assert decision.safe_action["wrist_flex"] == MAX_STEP
    assert any("EXCESSIVE_STEP_CLAMPED" in r for r in decision.reasons)


def test_reject_on_gross_excessive_step() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(wrist_flex=89.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "REJECT"
    assert any("EXCESSIVE_STEP_GROSS" in r for r in decision.reasons)


def test_reject_on_invalid_action_schema() -> None:
    bad_action = AdaptedAction(valid=False, command_deg={}, invalid_reason="차원이 틀렸습니다")
    decision = _gate().evaluate(adapted_action=bad_action, current_state_deg=_current_state(0.0), observation_valid=True)
    assert decision.decision == "REJECT"
    assert decision.safe_action is None
    assert any("ACTION_SCHEMA_INVALID" in r for r in decision.reasons)


def test_reject_on_stale_state() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(),
        current_state_deg=_current_state(0.0),
        observation_valid=True,
        state_stale=True,
        state_stale_reason="너무 오래된 값",
    )
    assert decision.decision == "REJECT"
    assert any("STATE_STALE" in r for r in decision.reasons)


def test_reject_on_invalid_observation() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(),
        current_state_deg=_current_state(0.0),
        observation_valid=False,
        observation_reasons=("wrist 카메라가 없습니다",),
    )
    assert decision.decision == "REJECT"
    assert any("OBSERVATION_INVALID" in r for r in decision.reasons)


def test_reject_on_missing_current_state() -> None:
    decision = _gate().evaluate(adapted_action=_action(), current_state_deg=None, observation_valid=True)
    assert decision.decision == "REJECT"
    assert any("STATE_MISSING" in r for r in decision.reasons)


def test_gripper_semantic_mismatch_mild_clamps_to_percent_range() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(gripper=105.0), current_state_deg=_current_state(0.0) | {"gripper": 100.0},
        observation_valid=True,
    )
    assert decision.decision == "WOULD_CLAMP"
    assert decision.safe_action["gripper"] == 100.0


def test_gripper_semantic_mismatch_gross_is_rejected() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(gripper=99999.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "REJECT"
