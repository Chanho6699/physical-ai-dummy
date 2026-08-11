from __future__ import annotations

import pytest

from hardware.safety.staged_follower_writer import (
    JOINT_ORDER,
    StagedFollowerArmedWriter,
    StagedWriteBudget,
    StagedWriteBudgetExceededError,
    WriteAttemptResult,
)
from runtime.common.vla_contract import JOINT_ORDER as VLA_JOINT_ORDER


def test_joint_order_matches_vla_contract():
    """새 JOINT_ORDER 복제본이 원본과 어긋나면 write가 잘못된 관절에 갈 수 있다 - 항상 동일해야 한다."""
    assert JOINT_ORDER == VLA_JOINT_ORDER


class FakeFollower:
    """``_FollowerLike`` duck type을 흉내내는 fake - 실제 시리얼/lerobot 의존성 없음."""

    def __init__(self, initial_state_deg: dict[str, float]):
        self.state = dict(initial_state_deg)
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.send_action_calls: list[dict] = []
        self.get_observation_calls = 0
        self.raise_on_send: Exception | None = None
        self.clip_send_action: bool = False  # LeRobot의 max_relative_target 자체 clip 흉내

    def connect(self) -> None:
        self.connect_calls += 1

    def get_observation(self) -> dict:
        self.get_observation_calls += 1
        return {f"{j}.pos": v for j, v in self.state.items()}

    def send_action(self, action: dict) -> dict:
        self.send_action_calls.append(dict(action))
        if self.raise_on_send is not None:
            raise self.raise_on_send
        sent = dict(action)
        if self.clip_send_action:
            # LeRobot이 max_relative_target으로 자체 clip했다고 가정 - 절반만 실제로 보냄.
            sent = {k: (self.state[k.removesuffix(".pos")] + v) / 2 for k, v in action.items()}
        for k, v in sent.items():
            self.state[k.removesuffix(".pos")] = v
        return sent

    def disconnect(self) -> None:
        self.disconnect_calls += 1


def _neutral_state() -> dict[str, float]:
    return {j: 0.0 for j in JOINT_ORDER}


def test_write_budget_enforced_before_any_wire_call():
    budget = StagedWriteBudget(max_write_count=2)
    budget.consume()
    budget.consume()
    with pytest.raises(StagedWriteBudgetExceededError):
        budget.consume()


def test_write_budget_rejects_zero_or_negative_max():
    with pytest.raises(ValueError):
        StagedWriteBudget(max_write_count=0)


def test_single_write_succeeds_and_state_updates():
    follower = FakeFollower(_neutral_state())
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=1)
    writer.connect()
    assert follower.connect_calls == 1

    before = writer.read_state_deg()
    assert before == _neutral_state()

    target = {**_neutral_state(), "shoulder_pan": 5.0}
    result = writer.write_action_once(target)
    assert result.executed
    assert result.error is None
    assert result.sent_action_deg == target
    assert result.write_count_before == 0
    assert result.write_count_after == 1
    assert len(follower.send_action_calls) == 1
    # send_action은 항상 ".pos" 접미사가 붙은 키만 받는다.
    assert set(follower.send_action_calls[0].keys()) == {f"{j}.pos" for j in JOINT_ORDER}

    after = writer.read_state_deg()
    assert after["shoulder_pan"] == 5.0


def test_second_write_beyond_budget_never_calls_send_action():
    follower = FakeFollower(_neutral_state())
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=1)
    writer.connect()

    writer.write_action_once(_neutral_state())
    assert len(follower.send_action_calls) == 1

    result = writer.write_action_once({**_neutral_state(), "gripper": 50.0})
    assert not result.executed
    assert result.error is not None
    assert "budget" in result.error.lower() or "예산" in result.error or "초과" in result.error
    # 두 번째 시도는 send_action 자체가 호출되지 않아야 한다 (핵심 불변식).
    assert len(follower.send_action_calls) == 1


def test_write_action_once_requires_all_joints():
    follower = FakeFollower(_neutral_state())
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=1)
    writer.connect()
    incomplete = {j: 0.0 for j in JOINT_ORDER if j != "gripper"}
    with pytest.raises(ValueError):
        writer.write_action_once(incomplete)
    assert len(follower.send_action_calls) == 0


def test_send_action_exception_is_captured_not_raised():
    follower = FakeFollower(_neutral_state())
    follower.raise_on_send = ConnectionError("simulated serial failure")
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=1)
    writer.connect()

    result = writer.write_action_once(_neutral_state())
    assert not result.executed
    assert "simulated serial failure" in (result.error or "")
    # budget은 시도 자체로 이미 소비된다 (요구사항: 재시도 금지 - 실패해도 예산은 줄어든다).
    assert writer.write_count == 1


def test_captures_lerobot_self_clip_in_sent_action():
    """LeRobot의 max_relative_target이 설정돼 있으면 send_action이 실제로 보낸 값을
    요청값과 다르게 반환할 수 있다 - writer는 그 반환값을 그대로 캡처해야 한다."""
    follower = FakeFollower(_neutral_state())
    follower.clip_send_action = True
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=1)
    writer.connect()

    target = {**_neutral_state(), "shoulder_pan": 10.0}
    result = writer.write_action_once(target)
    assert result.executed
    assert result.requested_action_deg["shoulder_pan"] == 10.0
    assert result.sent_action_deg["shoulder_pan"] == 5.0  # clipped to half by the fake


def test_disconnect_is_idempotent_and_only_after_connect():
    follower = FakeFollower(_neutral_state())
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=1)
    writer.disconnect()  # connect 없이 호출해도 안전해야 한다
    assert follower.disconnect_calls == 0

    writer.connect()
    writer.disconnect()
    writer.disconnect()
    assert follower.disconnect_calls == 1


def test_no_torque_or_calibration_methods_referenced_anywhere_in_module():
    """소스 코드 자체를 감사(audit)한다 - single_joint_writer.py와 동일한 방어적 테스트 스타일.

    docstring 설명문에는 이 이름들이 (금지 대상이라는 걸 설명하려고) 그대로 등장하므로 단순
    substring 검사 대신, 실제 '호출 형태'(``.name(``)만 감사한다."""
    import inspect

    import hardware.safety.staged_follower_writer as mod

    source = inspect.getsource(mod)
    forbidden = ["enable_torque", "disable_torque", "write_calibration", "set_half_turn_homings", "configure_motors"]
    for name in forbidden:
        assert f".{name}(" not in source, f"staged_follower_writer.py에 금지된 호출이 있습니다: .{name}("
