"""runtime/laptop/follower_state_source.py 테스트.

``hardware/state_server/readonly_so101_reader.ReadOnlySO101Reader``를 감싼 wrapper가
1) 위임 로직이 올바른지, 2) 공개 API에 write 계열 메서드를 하나도 노출하지 않는지
(섹션 17 - "boolean 하나에만 의존하지 않는" 구조적 보장)를 함께 검증한다.
"""

from __future__ import annotations

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.follower_state_source import (
    REAL_FOLLOWER_WRITE_COUNT,
    FollowerStateSourceError,
    ReadOnlyRealFollowerStateSource,
)

FORBIDDEN_METHOD_NAMES = (
    "write",
    "sync_write",
    "send_action",
    "send_feedback",
    "enable_torque",
    "disable_torque",
    "write_calibration",
    "reset_calibration",
    "set_half_turn_homings",
    "calibrate",
    "configure",
    "setup_motor",
    "teleop_step",
)


class FakeReader:
    def __init__(self, *, positions: dict[str, float] | None = None, fail_connect: bool = False, fail_read: bool = False) -> None:
        self._positions = positions if positions is not None else {j: float(i) for i, j in enumerate(JOINT_ORDER)}
        self._fail_connect = fail_connect
        self._fail_read = fail_read
        self._connected = False
        self.disconnect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._fail_connect:
            raise RuntimeError("연결 실패 (시뮬레이션)")
        self._connected = True

    def read_positions(self) -> dict[str, float]:
        if self._fail_read:
            raise RuntimeError("읽기 실패 (시뮬레이션)")
        return dict(self._positions)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False


def test_public_api_exposes_no_write_methods() -> None:
    """클래스 감사 - ReadOnlyRealFollowerStateSource 자체가 write 메서드를 노출하지 않는다."""
    public_methods = {name for name in dir(ReadOnlyRealFollowerStateSource) if not name.startswith("_")}
    overlap = public_methods & set(FORBIDDEN_METHOD_NAMES)
    assert overlap == set(), f"금지된 write 메서드가 노출되어 있습니다: {overlap}"


def test_connect_delegates_to_reader() -> None:
    reader = FakeReader()
    source = ReadOnlyRealFollowerStateSource(reader)
    assert source.is_connected is False
    source.connect()
    assert source.is_connected is True


def test_connect_failure_raises_typed_error() -> None:
    source = ReadOnlyRealFollowerStateSource(FakeReader(fail_connect=True))
    with pytest.raises(FollowerStateSourceError):
        source.connect()


def test_read_returns_snapshot_with_all_joints() -> None:
    reader = FakeReader(positions={j: 1.5 for j in JOINT_ORDER})
    source = ReadOnlyRealFollowerStateSource(reader)
    snapshot = source.read()
    assert set(snapshot.positions_deg) == set(JOINT_ORDER)
    assert all(v == 1.5 for v in snapshot.positions_deg.values())


def test_read_failure_raises_typed_error() -> None:
    source = ReadOnlyRealFollowerStateSource(FakeReader(fail_read=True))
    with pytest.raises(FollowerStateSourceError):
        source.read()


def test_read_missing_joint_raises() -> None:
    reader = FakeReader(positions={j: 0.0 for j in JOINT_ORDER if j != "gripper"})
    source = ReadOnlyRealFollowerStateSource(reader)
    with pytest.raises(FollowerStateSourceError):
        source.read()


def test_disconnect_delegates_to_reader() -> None:
    reader = FakeReader()
    source = ReadOnlyRealFollowerStateSource(reader)
    source.disconnect()
    assert reader.disconnect_calls == 1


def test_real_follower_write_count_constant_is_zero() -> None:
    assert REAL_FOLLOWER_WRITE_COUNT == 0
