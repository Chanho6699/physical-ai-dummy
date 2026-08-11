"""runtime/laptop/follower_action_writer.py 검증.

``SO101FollowerActionWriter``도 ``test_staged_follower_writer.py``와 동일한
``FakeFollower`` double만 사용한다 - 실제 포트/시리얼 접근 없음. 어디에서도
``connect()``를 실제 follower 객체로 호출하지 않는다(이 테스트 포함).
"""

from __future__ import annotations

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.follower_action_writer import (
    DEFAULT_SESSION_WRITE_CAP,
    FakeFollowerWriter,
    RecordingFollowerWriter,
    SO101FollowerActionWriter,
    WriteResult,
)


def _neutral(v: float = 0.0) -> dict[str, float]:
    return {j: v for j in JOINT_ORDER}


class FakeFollower:
    """test_staged_follower_writer.py::FakeFollower와 동일한 duck type."""

    def __init__(self, initial_state_deg: dict[str, float]):
        self.state = dict(initial_state_deg)
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.send_action_calls: list[dict] = []
        self.raise_on_send: Exception | None = None

    def connect(self) -> None:
        self.connect_calls += 1

    def get_observation(self) -> dict:
        return {f"{j}.pos": v for j, v in self.state.items()}

    def send_action(self, action: dict) -> dict:
        self.send_action_calls.append(dict(action))
        if self.raise_on_send is not None:
            raise self.raise_on_send
        for k, v in action.items():
            self.state[k.removesuffix(".pos")] = v
        return dict(action)

    def disconnect(self) -> None:
        self.disconnect_calls += 1


# ---------------------------------------------------------------------------
# FakeFollowerWriter / RecordingFollowerWriter (오프라인 전용)
# ---------------------------------------------------------------------------


def test_fake_writer_records_calls_and_reports_executed() -> None:
    writer = FakeFollowerWriter()
    result = writer.write(_neutral(1.0))
    assert isinstance(result, WriteResult)
    assert result.executed is True
    assert result.sent_action_deg == _neutral(1.0)
    assert result.write_index == 0
    assert writer.calls == [_neutral(1.0)]
    assert writer.write_count == 1


def test_fake_writer_rejects_missing_joint() -> None:
    writer = FakeFollowerWriter()
    incomplete = _neutral(0.0)
    del incomplete["wrist_roll"]
    with pytest.raises(ValueError):
        writer.write(incomplete)


def test_fake_writer_injected_failure_does_not_raise() -> None:
    writer = FakeFollowerWriter(fail_on_indices=frozenset({1}))
    r0 = writer.write(_neutral(0.0))
    r1 = writer.write(_neutral(1.0))
    r2 = writer.write(_neutral(2.0))
    assert r0.executed is True
    assert r1.executed is False
    assert r1.error is not None
    assert r2.executed is True
    # 실패한 write도 calls에는 기록된다(호출은 됐다 - "이 값을 쓰려고 시도했다"는 사실은 남아야 함).
    assert len(writer.calls) == 3


def test_recording_writer_tracks_monotonic_timestamps() -> None:
    times = iter([10.0, 10.02, 10.04])
    writer = RecordingFollowerWriter(monotonic_fn=lambda: next(times))
    writer.write(_neutral(0.0))
    writer.write(_neutral(1.0))
    writer.write(_neutral(2.0))
    assert writer.call_monotonic_times == [10.0, 10.02, 10.04]


# ---------------------------------------------------------------------------
# SO101FollowerActionWriter - production writer, fake follower double만 사용
# ---------------------------------------------------------------------------


def test_so101_writer_composes_staged_writer_and_calls_send_action() -> None:
    follower = FakeFollower(_neutral(0.0))
    writer = SO101FollowerActionWriter(follower=follower, session_write_cap=10)
    writer.connect()
    result = writer.write({**_neutral(0.0), "elbow_flex": 5.0})
    assert result.executed is True
    assert result.sent_action_deg["elbow_flex"] == pytest.approx(5.0)
    assert follower.connect_calls == 1
    assert len(follower.send_action_calls) == 1
    writer.disconnect()
    assert follower.disconnect_calls == 1


def test_so101_writer_session_cap_is_finite_circuit_breaker() -> None:
    follower = FakeFollower(_neutral(0.0))
    writer = SO101FollowerActionWriter(follower=follower, session_write_cap=2)
    writer.connect()
    r0 = writer.write(_neutral(1.0))
    r1 = writer.write(_neutral(2.0))
    r2 = writer.write(_neutral(3.0))  # cap 초과
    assert r0.executed and r1.executed
    assert r2.executed is False
    assert r2.error is not None
    # cap을 넘긴 뒤에는 send_action 자체가 호출되지 않는다(budget 소진 시 wire call 없음).
    assert len(follower.send_action_calls) == 2


def test_so101_writer_default_session_cap_is_large_not_unlimited() -> None:
    # "무제한"이 아니라 명시적 circuit breaker라는 설계 의도를 고정한다.
    assert DEFAULT_SESSION_WRITE_CAP > 1_000_000
    assert DEFAULT_SESSION_WRITE_CAP < 10**12


def test_so101_writer_never_calls_forbidden_methods() -> None:
    """SO101FollowerActionWriter가 send_action/connect/disconnect/get_observation 외
    다른 메서드를 참조하지 않는지 - torque/calibration 관련 write 경로가 없음을 재확인."""
    follower = FakeFollower(_neutral(0.0))

    class StrictFollower(FakeFollower):
        def __getattr__(self, name):
            raise AssertionError(f"허용되지 않은 메서드 접근: {name}")

    strict = StrictFollower(_neutral(0.0))
    writer = SO101FollowerActionWriter(follower=strict, session_write_cap=5)
    writer.connect()
    writer.write(_neutral(1.0))
    writer.disconnect()  # 여기서 AssertionError가 나지 않으면 통과
