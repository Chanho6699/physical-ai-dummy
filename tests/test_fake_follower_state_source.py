"""runtime/laptop/fake_follower_state_source.py의 FakeFollowerStateSource 검증.

오프라인 전용 - 하드웨어 접근 없음(애초에 이 클래스에 connect/write 메서드 자체가 없다).
"""

from __future__ import annotations

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.fake_follower_state_source import FakeFollowerStateSource, FakeFollowerStateSourceError


def _neutral(v: float = 0.0) -> dict[str, float]:
    return {j: v for j in JOINT_ORDER}


def test_no_write_methods_exist() -> None:
    source = FakeFollowerStateSource()
    for name in ("write", "send_action", "connect", "sync_write", "enable_torque", "disable_torque"):
        assert not hasattr(source, name), f"FakeFollowerStateSource에 '{name}' 메서드가 있으면 안 됩니다."


def test_read_returns_initial_state() -> None:
    initial = {**_neutral(0.0), "shoulder_pan": 12.5}
    source = FakeFollowerStateSource(initial_state_deg=initial)
    snapshot = source.read()
    assert snapshot.positions_deg == initial
    assert snapshot.read_at_monotonic > 0
    assert snapshot.read_at_wall > 0


def test_set_state_updates_subsequent_reads() -> None:
    source = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    source.set_state({**_neutral(0.0), "elbow_flex": 7.08})
    snapshot = source.read()
    assert snapshot.positions_deg["elbow_flex"] == pytest.approx(7.08)


def test_set_state_rejects_missing_joint() -> None:
    source = FakeFollowerStateSource()
    incomplete = _neutral(0.0)
    del incomplete["gripper"]
    with pytest.raises(ValueError):
        source.set_state(incomplete)


def test_fail_next_read_raises_once_then_recovers() -> None:
    source = FakeFollowerStateSource(initial_state_deg=_neutral(1.0))
    source.fail_next_read()
    with pytest.raises(FakeFollowerStateSourceError):
        source.read()
    # 실패는 딱 한 번만 - 다음 read()는 정상.
    snapshot = source.read()
    assert snapshot.positions_deg == _neutral(1.0)


def test_read_count_increments_even_on_failure() -> None:
    source = FakeFollowerStateSource()
    source.fail_next_read()
    with pytest.raises(FakeFollowerStateSourceError):
        source.read()
    source.read()
    assert source.read_count == 2


def test_monotonic_fn_is_injectable() -> None:
    calls = []

    def fake_monotonic() -> float:
        calls.append(1)
        return 42.0

    source = FakeFollowerStateSource(monotonic_fn=fake_monotonic)
    snapshot = source.read()
    assert snapshot.read_at_monotonic == 42.0
    assert calls
