"""hardware/state_server/readonly_so101_reader.py 단위 테스트.

실물 serial 포트에 연결하지 않는다. ``ReadOnlySO101Reader``가 생성한 실제
``FeetechMotorsBus`` 인스턴스(``reader._bus``)를 테스트 전용 가짜 버스로 바꿔치기해서,
reader가 오직 ``connect``/``sync_read``/``disconnect(disable_torque=False)``만 호출하고
그 어떤 쓰기 메서드도 절대 호출하지 않는다는 것을 검증한다.

이 파일은 두 가지를 함께 검증한다:
  1) (동작) reader가 가짜 버스에 위임하는 로직이 올바른가.
  2) (감사) ReadOnlySO101Reader 클래스가 공개 쓰기 메서드를 노출하지 않는가 - 이름 기반
     감사이며, 실제 호출 여부는 1)의 AssertionError 트리거로 별도 확인한다.
"""

from __future__ import annotations

import pytest

from hardware.state_server.readonly_so101_reader import (
    JOINT_ORDER,
    ReadOnlyReaderError,
    ReadOnlySO101Reader,
)

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")

from lerobot.motors import MotorCalibration  # noqa: E402


def _make_calibration() -> dict[str, MotorCalibration]:
    return {
        joint: MotorCalibration(id=i + 1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)
        for i, joint in enumerate(JOINT_ORDER)
    }


class _ForbiddenWriteCalled(AssertionError):
    """가짜 버스의 쓰기 계열 메서드가 호출되면 이 예외로 즉시 테스트를 실패시킨다."""


class FakeFeetechBus:
    """FeetechMotorsBus를 대체하는 테스트 전용 가짜 버스.

    read 계열 메서드만 정상 동작하고, 쓰기 계열 메서드는 호출되는 즉시 실패한다.
    """

    def __init__(self) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []  # disable_torque 인자 기록
        self.port = "/dev/null"  # ReadOnlySO101Reader.connect()의 로그 메시지가 참조한다

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, handshake: bool = True) -> None:
        self.connect_calls += 1
        self.connected = True

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        assert data_name == "Present_Position"
        if normalize:
            return dict.fromkeys(JOINT_ORDER, 12.5)
        return dict.fromkeys(JOINT_ORDER, 2048)

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls.append(disable_torque)
        self.connected = False

    # -- 쓰기 계열: 절대 호출되면 안 된다 --------------------------------

    def write(self, *args, **kwargs):
        raise _ForbiddenWriteCalled(f"write() 호출됨: args={args} kwargs={kwargs}")

    def sync_write(self, *args, **kwargs):
        raise _ForbiddenWriteCalled(f"sync_write() 호출됨: args={args} kwargs={kwargs}")

    def enable_torque(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("enable_torque() 호출됨")

    def disable_torque(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("disable_torque() 호출됨")

    def write_calibration(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("write_calibration() 호출됨")

    def reset_calibration(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("reset_calibration() 호출됨")

    def set_half_turn_homings(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("set_half_turn_homings() 호출됨")


def _build_reader_with_fake_bus(name: str = "테스트암") -> tuple[ReadOnlySO101Reader, FakeFeetechBus]:
    reader = ReadOnlySO101Reader(name=name, port="/dev/null", calibration=_make_calibration())
    fake_bus = FakeFeetechBus()
    reader._bus = fake_bus  # noqa: SLF001 - 의도적인 테스트용 내부 교체
    return reader, fake_bus


# ---------------------------------------------------------------------------
# 동작 테스트
# ---------------------------------------------------------------------------


def test_missing_joint_in_calibration_raises():
    incomplete = _make_calibration()
    del incomplete["gripper"]
    with pytest.raises(ReadOnlyReaderError, match="gripper"):
        ReadOnlySO101Reader(name="x", port="/dev/null", calibration=incomplete)


def test_connect_delegates_to_bus_connect_only():
    reader, fake_bus = _build_reader_with_fake_bus()
    assert not reader.is_connected

    reader.connect()

    assert fake_bus.connect_calls == 1
    assert reader.is_connected


def test_connect_is_idempotent_when_already_connected():
    reader, fake_bus = _build_reader_with_fake_bus()
    reader.connect()
    reader.connect()
    assert fake_bus.connect_calls == 1  # 이미 연결되어 있으면 다시 connect()를 부르지 않는다


def test_read_positions_returns_all_six_joints_normalized():
    reader, _ = _build_reader_with_fake_bus()
    reader.connect()

    positions = reader.read_positions()

    assert set(positions) == set(JOINT_ORDER)
    assert all(value == 12.5 for value in positions.values())


def test_read_raw_positions_returns_raw_ticks():
    reader, _ = _build_reader_with_fake_bus()
    reader.connect()

    raw = reader.read_raw_positions()

    assert set(raw) == set(JOINT_ORDER)
    assert all(value == 2048 for value in raw.values())
    assert all(isinstance(value, int) for value in raw.values())


def test_disconnect_never_disables_torque_via_write():
    reader, fake_bus = _build_reader_with_fake_bus()
    reader.connect()

    reader.disconnect()

    assert fake_bus.disconnect_calls == [False]  # disable_torque=False 고정
    assert not reader.is_connected


def test_disconnect_when_never_connected_is_a_noop():
    reader, fake_bus = _build_reader_with_fake_bus()
    reader.disconnect()
    assert fake_bus.disconnect_calls == []


def test_no_write_path_is_ever_triggered_by_normal_usage():
    """정상적인 connect -> read -> read_raw -> disconnect 흐름에서 쓰기 메서드가 전혀
    호출되지 않는다는 것을 가짜 버스의 AssertionError 트리거로 확인한다."""

    reader, _ = _build_reader_with_fake_bus()
    reader.connect()
    reader.read_positions()
    reader.read_raw_positions()
    reader.disconnect()
    # 예외 없이 여기 도달하면 write 계열 메서드가 한 번도 호출되지 않았다는 뜻이다.


# ---------------------------------------------------------------------------
# 감사(audit) 테스트: 클래스가 쓰기 메서드를 공개하지 않는가
# ---------------------------------------------------------------------------

_BANNED_NAME_SUBSTRINGS = (
    "write",
    "send_action",
    "send_feedback",
    "teleop_step",
    "set_goal",
    "goal_position",
    "enable_torque",
    "disable_torque",
    "calibrate",
    "configure",
    "setup_motor",
)


def test_reader_exposes_only_the_expected_read_only_interface():
    # 클래스 자체에는 self.name 같은 인스턴스 속성이 dir()에 나타나지 않으므로 클래스 레벨
    # (메서드/프로퍼티)과 인스턴스 레벨을 각각 확인한다.
    class_level_attrs = {name for name in dir(ReadOnlySO101Reader) if not name.startswith("_")}
    assert class_level_attrs == {"is_connected", "connect", "read_positions", "read_raw_positions", "disconnect"}

    reader, _ = _build_reader_with_fake_bus()
    instance_attrs = {name for name in dir(reader) if not name.startswith("_")}
    assert instance_attrs == class_level_attrs | {"name"}


def test_reader_public_method_names_contain_no_banned_write_substrings():
    public_attrs = [name for name in dir(ReadOnlySO101Reader) if not name.startswith("_")]
    for attr_name in public_attrs:
        lowered = attr_name.lower()
        for banned in _BANNED_NAME_SUBSTRINGS:
            assert banned not in lowered, f"'{attr_name}'에 금지된 패턴 '{banned}'가 포함되어 있습니다."
