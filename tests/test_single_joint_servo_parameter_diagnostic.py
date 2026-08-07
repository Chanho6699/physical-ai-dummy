"""hardware/safety/single_joint_servo_parameter_diagnostic.py 단위 테스트.

실물 ``/dev/tty*``/``/dev/serial*``에는 절대 접근하지 않는다. 세 층으로 나눠 검증한다:

1. ``classify_servo_parameters`` (순수 함수, lerobot 불필요) - 여러 판정 라벨.
2. ``compute_next_step_candidates`` (순수 함수) - 다음 단발 후보가 0.5° 미만인지 등.
3. ``ServoParameterDiagnosticInspector`` + 가짜 ``FeetechMotorsBus`` - motor id 5 외
   접근 금지, table에 없는 레지스터 read 시도 없음, write 계열 메서드가 호출되는 즉시
   ``AssertionError``.
"""

from __future__ import annotations

import pytest

from hardware.safety.single_joint_bus import WristRollCalibration
from hardware.safety.single_joint_servo_parameter_diagnostic import (
    PARAMETER_REGISTERS,
    STATE_REGISTERS,
    ServoParameterSnapshot,
    ServoParameterVerdict,
    classify_servo_parameters,
    compute_next_step_candidates,
)

WRIST_ROLL_CALIBRATION = WristRollCalibration(motor_id=5, drive_mode=0, homing_offset=1627, range_min=0, range_max=4095)


def _snapshot(**overrides) -> ServoParameterSnapshot:
    base = dict(
        torque_enable=1,
        goal_position_raw=2024,
        present_position_raw=2023,
        moving=0,
        status_raw=0,
        cw_dead_zone=0,
        ccw_dead_zone=0,
        minimum_startup_force=0,
        operating_mode=0,
        acceleration=254,
        maximum_acceleration=254,
        goal_velocity=0,
        maximum_velocity_limit=0,
        moving_velocity_threshold=0,
        torque_limit=1000,
        max_torque_limit=1000,
        p_coefficient=32,
        i_coefficient=0,
        d_coefficient=0,
        lock=1,
        angular_resolution=1,
        read_errors={},
        unavailable_registers=(),
    )
    base.update(overrides)
    return ServoParameterSnapshot(**base)


# ---------------------------------------------------------------------------
# classify_servo_parameters
# ---------------------------------------------------------------------------


def test_no_configuration_cause_found_when_everything_looks_normal():
    snap = _snapshot()  # dead zone/startup force=0, mode=0, accel=254, torque_limit=1000
    verdicts, reasons = classify_servo_parameters(snap)
    assert verdicts == (ServoParameterVerdict.NO_CONFIGURATION_CAUSE_FOUND,)
    assert reasons


def test_dead_zone_likely_when_dead_zone_nonzero():
    snap = _snapshot(cw_dead_zone=3, ccw_dead_zone=3)
    verdicts, _ = classify_servo_parameters(snap)
    assert ServoParameterVerdict.DEAD_ZONE_LIKELY in verdicts


def test_startup_force_threshold_likely_when_nonzero():
    snap = _snapshot(minimum_startup_force=50)
    verdicts, _ = classify_servo_parameters(snap)
    assert ServoParameterVerdict.STARTUP_FORCE_THRESHOLD_LIKELY in verdicts


def test_control_mode_restriction_when_operating_mode_not_position():
    snap = _snapshot(operating_mode=1)  # VELOCITY, lerobot OperatingMode Enum 기준 확정
    verdicts, reasons = classify_servo_parameters(snap)
    assert ServoParameterVerdict.CONTROL_MODE_RESTRICTION in verdicts
    assert any("OperatingMode" in r for r in reasons)


def test_velocity_or_acceleration_restriction_when_acceleration_zero():
    snap = _snapshot(acceleration=0)
    verdicts, _ = classify_servo_parameters(snap)
    assert ServoParameterVerdict.VELOCITY_OR_ACCELERATION_RESTRICTION in verdicts


def test_torque_limit_restriction_when_torque_limit_zero():
    snap = _snapshot(torque_limit=0)
    verdicts, _ = classify_servo_parameters(snap)
    assert ServoParameterVerdict.TORQUE_LIMIT_RESTRICTION in verdicts


def test_multiple_verdicts_can_coexist():
    snap = _snapshot(cw_dead_zone=5, operating_mode=2, acceleration=0)
    verdicts, _ = classify_servo_parameters(snap)
    assert ServoParameterVerdict.DEAD_ZONE_LIKELY in verdicts
    assert ServoParameterVerdict.CONTROL_MODE_RESTRICTION in verdicts
    assert ServoParameterVerdict.VELOCITY_OR_ACCELERATION_RESTRICTION in verdicts


def test_required_state_read_failure_is_unknown():
    snap = _snapshot(torque_enable=None, read_errors={"Torque_Enable": "comm timeout"})
    verdicts, reasons = classify_servo_parameters(snap)
    assert verdicts == (ServoParameterVerdict.UNKNOWN,)
    assert reasons


def test_verdicts_never_claim_confirmed_only_likely_or_restriction_labels():
    # 요구사항: "원인을 과하게 확정하지 말 것" - CONFIRMED라는 라벨 자체가 존재하지 않는다.
    assert not any("CONFIRMED" in v for v in ServoParameterVerdict.ALL)


# ---------------------------------------------------------------------------
# compute_next_step_candidates
# ---------------------------------------------------------------------------


def test_default_candidates_are_positive_and_negative_three_ticks():
    candidates = compute_next_step_candidates(start_deg=-2.1538, start_raw=2023, range_min=0, range_max=4095)
    assert len(candidates) == 2
    directions = {c.direction for c in candidates}
    assert directions == {"positive", "negative"}
    for c in candidates:
        assert c.tick_count == 3
        assert c.expected_raw_delta in (3, -3)


def test_candidates_are_under_half_degree():
    candidates = compute_next_step_candidates(start_deg=-2.1538, start_raw=2023, range_min=0, range_max=4095)
    for c in candidates:
        assert abs(c.requested_delta_deg) < 0.5
        assert c.under_max_degree is True


def test_candidates_expected_target_raw_matches_start_plus_tick_delta():
    candidates = compute_next_step_candidates(
        start_deg=-2.1538, start_raw=2023, range_min=0, range_max=4095, candidate_specs=(("positive", 4),)
    )
    c = candidates[0]
    assert c.expected_target_raw == 2027
    assert c.expected_raw_delta == 4


def test_candidates_flagged_unsafe_when_outside_inner_range():
    # start를 이음매 근처(raw≈4090)에 두면 안전 구간 밖 -> within_calibration_inner_range=False.
    candidates = compute_next_step_candidates(
        start_deg=179.6, start_raw=4090, range_min=0, range_max=4095, candidate_specs=(("positive", 3),)
    )
    c = candidates[0]
    assert c.within_calibration_inner_range is False
    assert c.to_dict()["safe_candidate"] is False


def test_candidate_specs_are_not_hardcoded_to_three_or_four_ticks():
    # 호출부가 다른 tick 수를 넘기면 그대로 반영되어야 한다 - 함수가 3/4을 강제하지 않는다.
    candidates = compute_next_step_candidates(
        start_deg=0.0, start_raw=2048, range_min=0, range_max=4095, candidate_specs=(("positive", 7),)
    )
    assert candidates[0].tick_count == 7


def test_required_registers_match_installed_control_table_names():
    assert STATE_REGISTERS == ("Torque_Enable", "Goal_Position", "Present_Position", "Moving", "Status")
    assert set(PARAMETER_REGISTERS) == {
        "CW_Dead_Zone",
        "CCW_Dead_Zone",
        "Minimum_Startup_Force",
        "Operating_Mode",
        "Acceleration",
        "Maximum_Acceleration",
        "Goal_Velocity",
        "Maximum_Velocity_Limit",
        "Moving_Velocity_Threshold",
        "Torque_Limit",
        "Max_Torque_Limit",
        "P_Coefficient",
        "I_Coefficient",
        "D_Coefficient",
        "Lock",
        "Angular_Resolution",
        "Acceleration_Multiplier ",
    }


# ---------------------------------------------------------------------------
# ServoParameterDiagnosticInspector + 가짜 FeetechMotorsBus
# ---------------------------------------------------------------------------

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")

from hardware.safety.single_joint_servo_parameter_diagnostic import ServoParameterDiagnosticInspector  # noqa: E402


class _ForbiddenWriteCalled(AssertionError):
    pass


_FULL_FAKE_CTRL_TABLE = {
    "sts3215": {
        "Torque_Enable": (40, 1),
        "Goal_Position": (42, 2),
        "Present_Position": (56, 2),
        "Moving": (66, 1),
        "Status": (65, 1),
        "CW_Dead_Zone": (26, 1),
        "CCW_Dead_Zone": (27, 1),
        "Minimum_Startup_Force": (24, 2),
        "Operating_Mode": (33, 1),
        "Acceleration": (41, 1),
        "Maximum_Acceleration": (85, 1),
        "Goal_Velocity": (46, 2),
        "Maximum_Velocity_Limit": (84, 1),
        "Moving_Velocity_Threshold": (80, 1),
        "Torque_Limit": (48, 2),
        "Max_Torque_Limit": (16, 2),
        "P_Coefficient": (21, 1),
        "I_Coefficient": (23, 1),
        "D_Coefficient": (22, 1),
        "Lock": (55, 1),
        "Angular_Resolution": (30, 1),
        "Acceleration_Multiplier ": (86, 1),  # 원본 dict key 그대로 (끝 공백 포함)
    }
}

_DEFAULT_VALUES = {
    "Torque_Enable": 1,
    "Goal_Position": 2021,
    "Present_Position": 2023,
    "Moving": 0,
    "Status": 0,
    "CW_Dead_Zone": 0,
    "CCW_Dead_Zone": 0,
    "Minimum_Startup_Force": 0,
    "Operating_Mode": 0,
    "Acceleration": 0,
    "Maximum_Acceleration": 0,
    "Goal_Velocity": 0,
    "Maximum_Velocity_Limit": 0,
    "Moving_Velocity_Threshold": 0,
    "Torque_Limit": 1000,
    "Max_Torque_Limit": 1000,
    "P_Coefficient": 32,
    "I_Coefficient": 0,
    "D_Coefficient": 0,
    "Lock": 1,
    "Angular_Resolution": 1,
    "Acceleration_Multiplier ": 64,
}


class FakeServoParameterBus:
    def __init__(self, *, values: dict | None = None, ctrl_table: dict | None = None) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []
        self.read_calls: list[tuple[str, str]] = []
        self.port = "/dev/null"
        self.model_ctrl_table = ctrl_table if ctrl_table is not None else _FULL_FAKE_CTRL_TABLE
        self._values = values if values is not None else dict(_DEFAULT_VALUES)

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, handshake: bool = True) -> None:
        self.connect_calls += 1
        self.connected = True

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        assert motor == "wrist_roll", f"wrist_roll 외 관절 read 시도: motor={motor}"
        assert data_name in self.model_ctrl_table["sts3215"], f"control table에 없는 레지스터 read 시도: {data_name}"
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


def _build_inspector_with_fake_bus(**bus_kwargs) -> tuple[ServoParameterDiagnosticInspector, FakeServoParameterBus]:
    inspector = ServoParameterDiagnosticInspector(port="/dev/null", calibration=WRIST_ROLL_CALIBRATION)
    fake_bus = FakeServoParameterBus(**bus_kwargs)
    inspector._bus = fake_bus  # noqa: SLF001 - 의도적인 테스트용 내부 교체
    return inspector, fake_bus


def test_inspector_bus_only_registers_wrist_roll_motor():
    inspector = ServoParameterDiagnosticInspector(port="/dev/null", calibration=WRIST_ROLL_CALIBRATION)
    assert set(inspector._bus.motors) == {"wrist_roll"}  # noqa: SLF001


def test_read_snapshot_returns_state_and_parameter_values():
    inspector, _ = _build_inspector_with_fake_bus()
    inspector.connect()
    snapshot = inspector.read_snapshot()

    assert snapshot.torque_enable == 1
    assert snapshot.goal_position_raw == 2021  # 마지막 negative armed 명령의 target
    assert snapshot.present_position_raw == 2023
    assert snapshot.moving == 0
    assert snapshot.status_raw == 0
    assert snapshot.cw_dead_zone == 0
    assert snapshot.operating_mode == 0
    assert snapshot.acceleration == 0


def test_read_snapshot_reads_acceleration_multiplier_despite_trailing_space_key():
    """tables.py의 'Acceleration_Multiplier ' 키는 끝에 공백이 있는 실제 dict key다 -
    이 진단 모듈이 그 정확한 문자열로 read를 시도하고, 값을 필드에 정상 매핑하는지 확인한다."""
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    snapshot = inspector.read_snapshot()

    assert snapshot.acceleration_multiplier == 64
    assert ("Acceleration_Multiplier ", "wrist_roll") in fake_bus.read_calls


def test_read_snapshot_only_touches_wrist_roll_motor():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_snapshot()
    for _data_name, motor in fake_bus.read_calls:
        assert motor == "wrist_roll"


def test_read_snapshot_never_attempts_register_not_in_control_table():
    limited_table = {
        "sts3215": {
            "Torque_Enable": (40, 1),
            "Goal_Position": (42, 2),
            "Present_Position": (56, 2),
            "Moving": (66, 1),
            "Status": (65, 1),
            "Operating_Mode": (33, 1),
            # dead zone / startup force / velocity / torque limit / PID / lock 없음
        }
    }
    inspector, fake_bus = _build_inspector_with_fake_bus(ctrl_table=limited_table)
    inspector.connect()
    snapshot = inspector.read_snapshot()

    read_names = {name for name, _motor in fake_bus.read_calls}
    assert "CW_Dead_Zone" not in read_names
    assert "Minimum_Startup_Force" not in read_names
    assert snapshot.cw_dead_zone is None
    assert "CW_Dead_Zone" in snapshot.unavailable_registers


def test_disconnect_never_writes():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_snapshot()
    inspector.disconnect()
    assert fake_bus.disconnect_calls == [False]


def test_full_diagnostic_flow_triggers_zero_writes():
    inspector, _ = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_snapshot()
    inspector.disconnect()
    # 예외 없이 여기 도달하면 write 계열 메서드가 한 번도 호출되지 않았다는 뜻이다.


def test_inspector_exposes_only_expected_read_only_interface():
    class_level_attrs = {name for name in dir(ServoParameterDiagnosticInspector) if not name.startswith("_")}
    assert class_level_attrs == {"is_connected", "connect", "read_snapshot", "disconnect"}
