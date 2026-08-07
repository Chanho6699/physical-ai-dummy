"""hardware/safety/single_joint_hardware_inspector.py 단위 테스트.

실물 serial 포트에 연결하지 않는다. ``SingleJointInspector``가 생성한 실제
``FeetechMotorsBus`` 인스턴스를 테스트 전용 가짜 버스로 바꿔치기해서(기존
``tests/test_readonly_so101_reader.py``와 동일한 패턴), 이 inspector가
``connect``/``sync_read``/``disconnect(disable_torque=False)``만 호출하고 어떤 쓰기
메서드도 절대 호출하지 않는다는 것과, wrist_roll 외 다른 관절에는 아예 접근하지
않는다는 것을 검증한다.
"""

from __future__ import annotations

import pytest

from hardware.safety.single_joint_hardware_inspector import SingleJointInspector, WristRollCalibration
from hardware.safety.single_joint_test_planner import TARGET_JOINT

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")


class _ForbiddenWriteCalled(AssertionError):
    """가짜 버스의 쓰기 계열 메서드가 호출되면 이 예외로 즉시 테스트를 실패시킨다."""


class FakeSingleJointBus:
    """FeetechMotorsBus를 대체하는 테스트 전용 가짜 버스 - wrist_roll 하나만 다룬다."""

    def __init__(self) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []
        self.sync_read_motor_args: list[list[str]] = []
        self.port = "/dev/null"

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, handshake: bool = True) -> None:
        self.connect_calls += 1
        self.connected = True

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        assert data_name == "Present_Position"
        self.sync_read_motor_args.append(list(motors) if motors is not None else [])
        if normalize:
            return {TARGET_JOINT: 12.5}
        return {TARGET_JOINT: 2048}

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

    def set_half_turn_homings(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("set_half_turn_homings() 호출됨")


def _make_calibration() -> WristRollCalibration:
    return WristRollCalibration(motor_id=5, drive_mode=0, homing_offset=1627, range_min=0, range_max=4095)


def _build_inspector_with_fake_bus() -> tuple[SingleJointInspector, FakeSingleJointBus]:
    inspector = SingleJointInspector(port="/dev/null", calibration=_make_calibration())
    fake_bus = FakeSingleJointBus()
    inspector._bus = fake_bus  # noqa: SLF001 - 의도적인 테스트용 내부 교체
    return inspector, fake_bus


# ---------------------------------------------------------------------------
# wrist_roll 외 관절 접근 금지
# ---------------------------------------------------------------------------


def test_bus_only_registers_wrist_roll_motor():
    inspector = SingleJointInspector(port="/dev/null", calibration=_make_calibration())
    assert set(inspector._bus.motors) == {TARGET_JOINT}  # noqa: SLF001


def test_sync_read_never_requests_other_joints():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_raw()
    inspector.read_degrees()
    for requested_motors in fake_bus.sync_read_motor_args:
        assert requested_motors == [TARGET_JOINT]


# ---------------------------------------------------------------------------
# 연결/읽기 동작 + write 0회
# ---------------------------------------------------------------------------


def test_connect_delegates_to_bus_connect_only():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    assert not inspector.is_connected
    inspector.connect()
    assert fake_bus.connect_calls == 1
    assert inspector.is_connected


def test_connect_is_idempotent():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.connect()
    assert fake_bus.connect_calls == 1


def test_read_raw_returns_int_tick():
    inspector, _ = _build_inspector_with_fake_bus()
    inspector.connect()
    raw = inspector.read_raw()
    assert raw == 2048
    assert isinstance(raw, int)


def test_read_degrees_returns_normalized_value():
    inspector, _ = _build_inspector_with_fake_bus()
    inspector.connect()
    deg = inspector.read_degrees()
    assert deg == pytest.approx(12.5)


def test_disconnect_never_disables_torque_via_write():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.disconnect()
    assert fake_bus.disconnect_calls == [False]  # disable_torque=False 고정
    assert not inspector.is_connected


def test_full_inspect_only_flow_triggers_zero_writes():
    """connect -> read_raw -> read_degrees -> disconnect 전 과정에서 쓰기 메서드가 전혀
    호출되지 않는다는 것을 가짜 버스의 AssertionError 트리거로 확인한다."""
    inspector, _ = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_raw()
    inspector.read_degrees()
    inspector.disconnect()
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


def test_inspector_exposes_only_expected_read_only_interface():
    class_level_attrs = {name for name in dir(SingleJointInspector) if not name.startswith("_")}
    assert class_level_attrs == {"is_connected", "connect", "read_raw", "read_degrees", "disconnect"}


def test_inspector_public_method_names_contain_no_banned_write_substrings():
    public_attrs = [name for name in dir(SingleJointInspector) if not name.startswith("_")]
    for attr_name in public_attrs:
        lowered = attr_name.lower()
        for banned in _BANNED_NAME_SUBSTRINGS:
            assert banned not in lowered, f"'{attr_name}'에 금지된 패턴 '{banned}'가 포함되어 있습니다."
