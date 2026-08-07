"""hardware/safety/single_joint_register_diagnostic.py 단위 테스트.

실물 ``/dev/tty*``/``/dev/serial*``에는 절대 접근하지 않는다. 두 층으로 나눠 검증한다:

1. ``classify_diagnostic`` (순수 함수, lerobot 불필요) - 6개 판정 라벨 전부.
2. ``RegisterDiagnosticInspector`` + 가짜 ``FeetechMotorsBus`` - motor id 5 외 접근
   금지, write 계열 메서드가 호출되는 즉시 ``AssertionError``, disconnect 중 write 없음.
"""

from __future__ import annotations

import pytest

from hardware.safety.single_joint_bus import WristRollCalibration
from hardware.safety.single_joint_register_diagnostic import (
    HARDWARE_ERROR_STATUS_NOT_AVAILABLE_NOTE,
    MOVING_STATUS_NOT_AVAILABLE_NOTE,
    OPTIONAL_REGISTERS,
    REQUIRED_REGISTERS,
    DiagnosticVerdict,
    RegisterSnapshot,
    classify_diagnostic,
)

WRIST_ROLL_CALIBRATION = WristRollCalibration(motor_id=5, drive_mode=0, homing_offset=1627, range_min=0, range_max=4095)


def _snapshot(**overrides) -> RegisterSnapshot:
    base = dict(
        torque_enable=1,
        goal_position_raw=2024,
        present_position_raw=2024,
        moving=0,
        present_load=0,
        present_current=0,
        present_velocity=0,
        present_voltage=74,
        present_temperature=30,
        status_raw=0,
        read_errors={},
    )
    base.update(overrides)
    return RegisterSnapshot(**base)


# ---------------------------------------------------------------------------
# classify_diagnostic: 6개 판정 라벨 (순수 함수)
# ---------------------------------------------------------------------------


def test_torque_disabled_verdict():
    snap = _snapshot(torque_enable=0)
    verdict, reasons = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=2024)
    assert verdict == DiagnosticVerdict.TORQUE_DISABLED
    assert reasons


def test_command_latched_but_no_motion_verdict():
    # 실제 armed 실행에서 보고된 값 그대로: torque=1, goal=2024, present=2023, moving=0.
    snap = _snapshot(torque_enable=1, goal_position_raw=2024, present_position_raw=2023, moving=0)
    verdict, reasons = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=2024)
    assert verdict == DiagnosticVerdict.COMMAND_LATCHED_BUT_NO_MOTION
    assert reasons


def test_goal_not_latched_verdict():
    snap = _snapshot(goal_position_raw=2048)  # expected_goal_raw=2024와 다름
    verdict, reasons = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=2024)
    assert verdict == DiagnosticVerdict.GOAL_NOT_LATCHED
    assert reasons


def test_motor_still_moving_verdict():
    snap = _snapshot(moving=1)
    verdict, reasons = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=2024)
    assert verdict == DiagnosticVerdict.MOTOR_STILL_MOVING
    assert reasons


def test_fault_or_protection_verdict():
    snap = _snapshot(status_raw=4)  # 0이 아님
    verdict, reasons = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=2024)
    assert verdict == DiagnosticVerdict.FAULT_OR_PROTECTION
    assert reasons


def test_required_register_read_failure_is_unknown():
    snap = _snapshot(torque_enable=None, read_errors={"Torque_Enable": "comm timeout"})
    verdict, reasons = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=2024)
    assert verdict == DiagnosticVerdict.UNKNOWN
    assert "Torque_Enable" in reasons[0]


def test_goal_equals_present_and_no_issues_is_unknown_not_a_failure_label():
    snap = _snapshot(goal_position_raw=2024, present_position_raw=2024)
    verdict, _ = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=2024)
    assert verdict == DiagnosticVerdict.UNKNOWN


def test_fault_takes_priority_over_torque_disabled():
    snap = _snapshot(status_raw=1, torque_enable=0)
    verdict, _ = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=2024)
    assert verdict == DiagnosticVerdict.FAULT_OR_PROTECTION


def test_torque_disabled_takes_priority_over_goal_not_latched():
    snap = _snapshot(torque_enable=0, goal_position_raw=9999)
    verdict, _ = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=2024)
    assert verdict == DiagnosticVerdict.TORQUE_DISABLED


def test_goal_not_latched_skipped_when_expected_goal_raw_is_none():
    # expected_goal_raw가 없으면 latch 비교 자체를 생략하고 다음 단계로 넘어간다.
    snap = _snapshot(goal_position_raw=9999, present_position_raw=2023, moving=0)
    verdict, _ = classify_diagnostic(snapshot=snap, expected_start_raw=2023, expected_goal_raw=None)
    assert verdict == DiagnosticVerdict.COMMAND_LATCHED_BUT_NO_MOTION


def test_moving_status_and_hardware_error_status_notes_are_explicit_about_absence():
    assert "Moving_Status" in MOVING_STATUS_NOT_AVAILABLE_NOTE
    assert "Hardware_Error_Status" in HARDWARE_ERROR_STATUS_NOT_AVAILABLE_NOTE
    assert "존재하지" in MOVING_STATUS_NOT_AVAILABLE_NOTE or "정의되어 있지" in MOVING_STATUS_NOT_AVAILABLE_NOTE


def test_register_snapshot_to_dict_reports_moving_status_as_not_available():
    snap = _snapshot()
    d = snap.to_dict()
    assert d["Moving_Status"] == "NOT_AVAILABLE_IN_INSTALLED_TABLE"
    assert d["Torque_Enable"] == 1


def test_required_and_optional_register_names_match_installed_control_table():
    # 이 값들은 ~/lerobot/src/lerobot/motors/feetech/tables.py의 STS_SMS_SERIES_CONTROL_TABLE
    # 실제 키와 일치해야 한다 (추측 이름 금지).
    assert REQUIRED_REGISTERS == ("Torque_Enable", "Goal_Position", "Present_Position", "Moving")
    assert set(OPTIONAL_REGISTERS) == {
        "Present_Load",
        "Present_Current",
        "Present_Velocity",
        "Present_Voltage",
        "Present_Temperature",
        "Status",
    }
    assert "Moving_Status" not in REQUIRED_REGISTERS
    assert "Moving_Status" not in OPTIONAL_REGISTERS
    assert "Hardware_Error_Status" not in OPTIONAL_REGISTERS


# ---------------------------------------------------------------------------
# RegisterDiagnosticInspector + 가짜 FeetechMotorsBus
# ---------------------------------------------------------------------------

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")

from hardware.safety.single_joint_register_diagnostic import RegisterDiagnosticInspector  # noqa: E402


class _ForbiddenWriteCalled(AssertionError):
    pass


# 실제 STS_SMS_SERIES_CONTROL_TABLE의 sts3215 서브셋 (필수 + 선택 레지스터 전부 포함).
_FAKE_CTRL_TABLE = {
    "sts3215": {
        "Torque_Enable": (40, 1),
        "Goal_Position": (42, 2),
        "Present_Position": (56, 2),
        "Moving": (66, 1),
        "Present_Load": (60, 2),
        "Present_Current": (69, 2),
        "Present_Velocity": (58, 2),
        "Present_Voltage": (62, 1),
        "Present_Temperature": (63, 1),
        "Status": (65, 1),
    }
}


class FakeDiagnosticBus:
    """RegisterDiagnosticInspector가 감싸는 FeetechMotorsBus를 대체하는 가짜 버스.

    read만 정상 동작하고, 쓰기 계열 메서드는 호출되는 즉시 실패한다.
    """

    def __init__(self, *, values: dict | None = None, ctrl_table: dict | None = None) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []
        self.read_calls: list[tuple[str, str]] = []
        self.port = "/dev/null"
        self.model_ctrl_table = ctrl_table if ctrl_table is not None else _FAKE_CTRL_TABLE
        self._values = values or {
            "Torque_Enable": 1,
            "Goal_Position": 2024,
            "Present_Position": 2023,
            "Moving": 0,
            "Present_Load": 0,
            "Present_Current": 0,
            "Present_Velocity": 0,
            "Present_Voltage": 74,
            "Present_Temperature": 30,
            "Status": 0,
        }

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, handshake: bool = True) -> None:
        self.connect_calls += 1
        self.connected = True

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        assert motor == "wrist_roll", f"wrist_roll 외 관절 read 시도: motor={motor}"
        self.read_calls.append((data_name, motor))
        if data_name not in self._values:
            raise ConnectionError(f"register not available in fake: {data_name}")
        return self._values[data_name]

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

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls.append(disable_torque)
        self.connected = False


def _build_inspector_with_fake_bus(**bus_kwargs) -> tuple[RegisterDiagnosticInspector, FakeDiagnosticBus]:
    inspector = RegisterDiagnosticInspector(port="/dev/null", calibration=WRIST_ROLL_CALIBRATION)
    fake_bus = FakeDiagnosticBus(**bus_kwargs)
    inspector._bus = fake_bus  # noqa: SLF001 - 의도적인 테스트용 내부 교체
    return inspector, fake_bus


def test_inspector_bus_only_registers_wrist_roll_motor():
    inspector = RegisterDiagnosticInspector(port="/dev/null", calibration=WRIST_ROLL_CALIBRATION)
    assert set(inspector._bus.motors) == {"wrist_roll"}  # noqa: SLF001


def test_read_snapshot_returns_all_required_and_optional_values():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    snapshot = inspector.read_snapshot()

    assert snapshot.torque_enable == 1
    assert snapshot.goal_position_raw == 2024
    assert snapshot.present_position_raw == 2023
    assert snapshot.moving == 0
    assert snapshot.status_raw == 0
    assert snapshot.read_errors == {}


def test_read_snapshot_only_reads_wrist_roll_motor():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_snapshot()
    for _data_name, motor in fake_bus.read_calls:
        assert motor == "wrist_roll"


def test_read_snapshot_skips_registers_not_in_installed_control_table():
    # ctrl_table에 Status/Present_Current가 없는 경우 - 시도조차 하지 않아야 한다.
    limited_table = {
        "sts3215": {
            "Torque_Enable": (40, 1),
            "Goal_Position": (42, 2),
            "Present_Position": (56, 2),
            "Moving": (66, 1),
        }
    }
    inspector, fake_bus = _build_inspector_with_fake_bus(ctrl_table=limited_table)
    inspector.connect()
    snapshot = inspector.read_snapshot()

    read_register_names = {name for name, _motor in fake_bus.read_calls}
    assert "Status" not in read_register_names
    assert "Present_Current" not in read_register_names
    assert snapshot.status_raw is None
    assert snapshot.present_current is None


def test_read_snapshot_reports_read_error_without_raising():
    values = {
        "Goal_Position": 2024,
        "Present_Position": 2023,
        "Moving": 0,
        # Torque_Enable을 일부러 빼서 read 실패를 유도한다.
        "Present_Load": 0,
        "Present_Current": 0,
        "Present_Velocity": 0,
        "Present_Voltage": 74,
        "Present_Temperature": 30,
        "Status": 0,
    }
    inspector, _ = _build_inspector_with_fake_bus(values=values)
    inspector.connect()
    snapshot = inspector.read_snapshot()

    assert snapshot.torque_enable is None
    assert "Torque_Enable" in snapshot.read_errors


def test_disconnect_never_writes_and_disables_torque_flag_is_false():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_snapshot()
    inspector.disconnect()

    assert fake_bus.disconnect_calls == [False]
    assert not inspector.is_connected


def test_full_diagnostic_flow_triggers_zero_writes():
    """connect -> read_snapshot -> disconnect 전 과정에서 쓰기 메서드가 전혀 호출되지
    않는다는 것을 가짜 버스의 AssertionError 트리거로 확인한다."""
    inspector, _ = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_snapshot()
    inspector.disconnect()
    # 예외 없이 여기 도달하면 write 계열 메서드가 한 번도 호출되지 않았다는 뜻이다.


def test_inspector_exposes_only_expected_read_only_interface():
    class_level_attrs = {name for name in dir(RegisterDiagnosticInspector) if not name.startswith("_")}
    assert class_level_attrs == {"is_connected", "connect", "read_snapshot", "disconnect"}


_BANNED_NAME_SUBSTRINGS = (
    "write",
    "send_action",
    "enable_torque",
    "disable_torque",
    "goal_position",
    "calibrate",
    "configure",
    "setup_motor",
)


def test_inspector_public_method_names_contain_no_banned_write_substrings():
    public_attrs = [name for name in dir(RegisterDiagnosticInspector) if not name.startswith("_")]
    for attr_name in public_attrs:
        lowered = attr_name.lower()
        for banned in _BANNED_NAME_SUBSTRINGS:
            assert banned not in lowered, f"'{attr_name}'에 금지된 패턴 '{banned}'가 포함되어 있습니다."
