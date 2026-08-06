"""hardware/state_server/state_service.py (StatePoller) 단위 테스트.

실물 하드웨어 대신 ReadOnlySO101Reader와 같은 인터페이스
(``connect``/``is_connected``/``read_positions``/``read_raw_positions``/``disconnect``)를
구현한 가짜 reader를 사용한다. 백그라운드 스레드(``start()``)는 타이밍에 의존하므로
사용하지 않고, ``poll_once()``를 직접 호출해 한 틱씩 결정적으로(deterministic) 진행한다.
"""

from __future__ import annotations

import math
import time

import pytest

from hardware.state_server.state_models import JOINT_NAMES
from hardware.state_server.state_service import StatePoller

GOOD_LEADER = {joint: float(i) for i, joint in enumerate(JOINT_NAMES)}
GOOD_FOLLOWER = {joint: float(i) + 0.5 for i, joint in enumerate(JOINT_NAMES)}


class FakeReader:
    """ReadOnlySO101Reader와 동일한 공개 인터페이스를 갖는 테스트 전용 가짜 reader.

    ``positions_queue``에 넣어둔 항목을 순서대로 하나씩 소비한다. 항목이
    ``Exception``이면 ``read_positions()``가 그 예외를 던진다 (읽기 실패 시뮬레이션).
    큐가 소진되면 마지막 항목을 반복 반환한다.
    """

    def __init__(self, name: str, positions_queue: list, *, fail_connect: bool = False) -> None:
        self.name = name
        self._queue = list(positions_queue)
        self._index = 0
        self._fail_connect = fail_connect
        self._connected = False
        self.disconnect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._fail_connect:
            raise RuntimeError(f"{self.name} 연결 실패 (시뮬레이션)")
        self._connected = True

    def _next(self):
        if not self._queue:
            raise RuntimeError(f"{self.name}: 큐에 데이터가 없습니다.")
        item = self._queue[min(self._index, len(self._queue) - 1)]
        self._index += 1
        return item

    def read_positions(self) -> dict[str, float]:
        item = self._next()
        if isinstance(item, Exception):
            raise item
        return dict(item)

    def read_raw_positions(self) -> dict[str, int] | None:
        return None

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False


def _make_poller(leader: FakeReader, follower: FakeReader, **kwargs) -> StatePoller:
    defaults = dict(rate_hz=30.0, stale_after_ms=500.0, max_read_errors=3)
    defaults.update(kwargs)
    return StatePoller(leader_reader=leader, follower_reader=follower, **defaults)


# ---------------------------------------------------------------------------
# 정상 상태
# ---------------------------------------------------------------------------


def test_snapshot_reports_both_arms_when_healthy():
    leader = FakeReader("leader", [GOOD_LEADER])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()
    poller.poll_once()

    snap = poller.snapshot()

    assert snap.leader.connected is True
    assert snap.leader.positions_deg == GOOD_LEADER
    assert snap.follower.positions_deg == GOOD_FOLLOWER
    assert snap.leader.stale is False
    assert snap.follower.stale is False
    assert snap.warnings == []


def test_health_ok_when_both_connected_and_fresh():
    leader = FakeReader("leader", [GOOD_LEADER])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()
    poller.poll_once()

    health = poller.health()

    assert health.status == "ok"
    assert health.leader_connected is True
    assert health.follower_connected is True
    assert health.write_enabled is False
    assert health.errors == []


# ---------------------------------------------------------------------------
# 연결 실패
# ---------------------------------------------------------------------------


def test_health_degraded_when_follower_connect_fails():
    leader = FakeReader("leader", [GOOD_LEADER])
    follower = FakeReader("follower", [GOOD_FOLLOWER], fail_connect=True)
    poller = _make_poller(leader, follower)

    errors = poller.connect_all()

    assert any("팔로워암" in e for e in errors)
    health = poller.health()
    assert health.status == "degraded"
    assert health.follower_connected is False
    assert any("팔로워암" in e for e in health.errors)


def test_connect_all_still_connects_leader_when_follower_fails():
    leader = FakeReader("leader", [GOOD_LEADER])
    follower = FakeReader("follower", [GOOD_FOLLOWER], fail_connect=True)
    poller = _make_poller(leader, follower)

    poller.connect_all()

    assert leader.is_connected is True
    assert follower.is_connected is False


# ---------------------------------------------------------------------------
# 읽기 실패 / stale / 거짓 정상값 금지
# ---------------------------------------------------------------------------


def test_read_failure_preserves_last_good_and_marks_stale_after_expiry():
    leader = FakeReader("leader", [GOOD_LEADER, RuntimeError("통신 오류")])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower, stale_after_ms=1.0)
    poller.connect_all()

    poller.poll_once()  # 정상 1회
    time.sleep(0.01)  # stale_after_ms(1ms)를 확실히 초과시킨다
    poller.poll_once()  # 실패 1회 (leader만)

    snap = poller.snapshot()
    # 마지막 정상값(=GOOD_LEADER)이 계속 제공되어야 한다 - 거짓 정상값이 아니라 "이전" 정상값.
    assert snap.leader.positions_deg == GOOD_LEADER
    assert snap.leader.stale is True
    assert snap.leader.read_error_count == 1


def test_never_serves_a_value_when_no_successful_read_ever_happened():
    leader = FakeReader("leader", [RuntimeError("최초부터 실패")])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()
    poller.poll_once()

    snap = poller.snapshot()

    assert snap.leader.positions_deg is None  # 거짓 정상값을 만들지 않는다
    assert snap.leader.stale is True
    assert snap.leader.read_error_count == 1


def test_health_becomes_degraded_after_max_read_errors_reached():
    leader = FakeReader("leader", [RuntimeError("실패")] * 5)
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower, max_read_errors=3)
    poller.connect_all()

    for _ in range(3):
        poller.poll_once()

    health = poller.health()
    assert health.status == "degraded"
    assert any("리더암" in e and "3회" in e for e in health.errors)


def test_recovers_after_successful_read_following_errors():
    leader = FakeReader("leader", [RuntimeError("실패1"), RuntimeError("실패2"), GOOD_LEADER])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower, max_read_errors=3)
    poller.connect_all()

    poller.poll_once()
    poller.poll_once()
    poller.poll_once()  # 여기서 회복

    snap = poller.snapshot()
    assert snap.leader.positions_deg == GOOD_LEADER
    assert snap.leader.read_error_count == 0


# ---------------------------------------------------------------------------
# NaN/Inf, 관절 누락, 잘못된 shape 거부
# ---------------------------------------------------------------------------


def test_nan_position_is_rejected_and_treated_as_read_error():
    bad = dict(GOOD_LEADER)
    bad["wrist_flex"] = math.nan
    leader = FakeReader("leader", [bad])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()
    poller.poll_once()

    snap = poller.snapshot()
    assert snap.leader.positions_deg is None
    assert snap.leader.read_error_count == 1


def test_inf_position_is_rejected():
    bad = dict(GOOD_LEADER)
    bad["gripper"] = math.inf
    leader = FakeReader("leader", [bad])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()
    poller.poll_once()

    assert poller.snapshot().leader.positions_deg is None


def test_missing_joint_is_rejected():
    incomplete = dict(GOOD_LEADER)
    del incomplete["gripper"]
    leader = FakeReader("leader", [incomplete])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()
    poller.poll_once()

    snap = poller.snapshot()
    assert snap.leader.positions_deg is None
    assert snap.leader.read_error_count == 1


def test_wrong_shape_extra_key_is_rejected():
    wrong = dict(GOOD_LEADER)
    wrong["extra_unexpected_joint"] = 1.0
    leader = FakeReader("leader", [wrong])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()
    poller.poll_once()

    assert poller.snapshot().leader.positions_deg is None


# ---------------------------------------------------------------------------
# sequence 증가 / difference_deg 계산
# ---------------------------------------------------------------------------


def test_sequence_increments_once_per_poll_once_call():
    leader = FakeReader("leader", [GOOD_LEADER] * 5)
    follower = FakeReader("follower", [GOOD_FOLLOWER] * 5)
    poller = _make_poller(leader, follower)
    poller.connect_all()

    assert poller.snapshot().sequence == 0
    poller.poll_once()
    assert poller.snapshot().sequence == 1
    poller.poll_once()
    poller.poll_once()
    assert poller.snapshot().sequence == 3


def test_difference_deg_is_leader_minus_follower():
    leader = FakeReader("leader", [GOOD_LEADER])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()
    poller.poll_once()

    diff = poller.snapshot().difference_deg
    for joint in JOINT_NAMES:
        assert diff[joint] == pytest.approx(GOOD_LEADER[joint] - GOOD_FOLLOWER[joint])


def test_difference_deg_empty_and_warning_when_one_side_missing():
    leader = FakeReader("leader", [RuntimeError("실패")])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()
    poller.poll_once()

    snap = poller.snapshot()
    assert snap.difference_deg == {}
    assert any("difference_deg" in w for w in snap.warnings)


# ---------------------------------------------------------------------------
# 종료(disconnect) 처리
# ---------------------------------------------------------------------------


def test_disconnect_all_calls_reader_disconnect_for_both_arms():
    leader = FakeReader("leader", [GOOD_LEADER])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()

    errors = poller.disconnect_all()

    assert errors == []
    assert leader.disconnect_calls == 1
    assert follower.disconnect_calls == 1


def test_disconnect_all_reports_errors_without_raising():
    class RaisingDisconnectReader(FakeReader):
        def disconnect(self) -> None:
            raise RuntimeError("disconnect 실패 (시뮬레이션)")

    leader = RaisingDisconnectReader("leader", [GOOD_LEADER])
    follower = FakeReader("follower", [GOOD_FOLLOWER])
    poller = _make_poller(leader, follower)
    poller.connect_all()

    errors = poller.disconnect_all()

    assert len(errors) == 1
    assert "리더암" in errors[0]
    # 팔로워는 리더 disconnect 실패와 무관하게 정상적으로 disconnect되어야 한다.
    assert follower.disconnect_calls == 1
