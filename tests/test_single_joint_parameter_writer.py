"""hardware/safety/single_joint_parameter_writer.py 단위 테스트.

실물 ``/dev/tty*``/``/dev/serial*``에는 절대 접근하지 않는다. 두 층으로 나눠 검증한다:

1. ``SingleJointParameterWriter`` + 가짜 ``FeetechMotorsBus`` - Acceleration 외
   레지스터 write, wrist_roll 외 관절 write, 두 번째 write가 즉시 ``AssertionError``.
2. ``execute_single_parameter_write`` + 순수 파이썬 ``FakeParameterWriter``(lerobot
   불필요) - 확인 플래그/현재값 일치/write budget/readback 판정 오케스트레이션 전체.

이 파일은 ``--mode acceleration-write`` CLI 진입점(``scripts/run_single_joint_hardware_test.py``
의 ``_run_acceleration_write``)을 어디에서도 호출하지 않는다.
"""

from __future__ import annotations

import pytest

from hardware.safety.single_joint_bus import WristRollCalibration
from hardware.safety.single_joint_parameter_writer import (
    ALLOWED_REGISTER_NAME,
    DEFAULT_EXPECTED_CURRENT_ACCELERATION,
    FORBIDDEN_REGISTER_NAMES,
    READBACK_MISMATCH,
    REQUIRED_CONFIRMATION_FLAGS,
    TARGET_ACCELERATION_VALUE,
    ParameterWriteResult,
    SingleJointParameterWriter,
    WriteBudgetExceededError,
    execute_single_parameter_write,
)
from hardware.safety.single_joint_test_planner import BLOCKED, PASS, READBACK_FAILED, TARGET_JOINT, WRITE_FAILED

WRIST_ROLL_CALIBRATION = WristRollCalibration(motor_id=5, drive_mode=0, homing_offset=1627, range_min=0, range_max=4095)
WRONG_MOTOR_CALIBRATION = WristRollCalibration(motor_id=1, drive_mode=0, homing_offset=0, range_min=1070, range_max=3135)

BOTH_CONFIRMED = {"i_understand_this_changes_servo_state": True, "confirm_acceleration_write": True}


def _call(writer, *, expected_current_acceleration=DEFAULT_EXPECTED_CURRENT_ACCELERATION, confirmation_flags=None):
    return execute_single_parameter_write(
        writer=writer,
        expected_current_acceleration=expected_current_acceleration,
        confirmation_flags=BOTH_CONFIRMED if confirmation_flags is None else confirmation_flags,
    )


# ---------------------------------------------------------------------------
# 상수 감사
# ---------------------------------------------------------------------------


def test_only_acceleration_is_allowed():
    assert ALLOWED_REGISTER_NAME == "Acceleration"
    assert "Acceleration" not in FORBIDDEN_REGISTER_NAMES


def test_forbidden_registers_cover_all_dangerous_settings():
    expected = {
        "Torque_Enable",
        "Goal_Position",
        "Operating_Mode",
        "P_Coefficient",
        "I_Coefficient",
        "D_Coefficient",
        "CW_Dead_Zone",
        "CCW_Dead_Zone",
        "Minimum_Startup_Force",
        "Maximum_Acceleration",
        "Torque_Limit",
        "Max_Torque_Limit",
        "Lock",
        "Homing_Offset",
    }
    assert set(FORBIDDEN_REGISTER_NAMES) == expected


def test_required_confirmation_flags_has_exactly_two_entries():
    assert set(REQUIRED_CONFIRMATION_FLAGS) == {"i_understand_this_changes_servo_state", "confirm_acceleration_write"}


def test_target_and_default_expected_values():
    assert TARGET_ACCELERATION_VALUE == 254
    assert DEFAULT_EXPECTED_CURRENT_ACCELERATION == 0


def test_write_acceleration_once_signature_takes_no_register_or_value_argument():
    """이 write 메서드는 register 이름이나 값을 인자로 받지 않는다 - 구조적으로 다른
    레지스터/값을 쓰는 것이 불가능함을 시그니처 레벨에서도 확인한다."""
    import inspect

    sig = inspect.signature(SingleJointParameterWriter.write_acceleration_once)
    params = [p for p in sig.parameters if p != "self"]
    assert params == []


def test_module_source_never_writes_forbidden_registers():
    import inspect

    import hardware.safety.single_joint_parameter_writer as writer_module

    source = inspect.getsource(writer_module)
    # FORBIDDEN_REGISTER_NAMES 정의 자체(문자열 리터럴로만 등장)는 제외하고, "self._bus.write("
    # 호출부에 그 이름들이 실제로 등장하지 않는지 확인한다.
    write_call_lines = [line for line in source.splitlines() if ".write(" in line and "_bus" in line]
    assert len(write_call_lines) == 1  # write_acceleration_once() 안의 단 한 줄
    for forbidden in FORBIDDEN_REGISTER_NAMES:
        assert forbidden not in write_call_lines[0]
    assert "ALLOWED_REGISTER_NAME" in write_call_lines[0]


# ---------------------------------------------------------------------------
# SingleJointParameterWriter + 가짜 FeetechMotorsBus
# ---------------------------------------------------------------------------

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")


class _ForbiddenWriteCalled(AssertionError):
    pass


class FakeParameterFeetechBus:
    def __init__(self, *, current_acceleration: int = 0) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []
        self.write_calls: list[tuple] = []
        self.read_calls: list[tuple[str, str]] = []
        self.port = "/dev/null"
        self._acceleration = current_acceleration
        self._context_values = {"Torque_Enable": 1, "Goal_Position": 2021, "Present_Position": 2023}

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, handshake: bool = True) -> None:
        self.connect_calls += 1
        self.connected = True

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        self.read_calls.append((data_name, motor))
        if data_name == "Acceleration":
            return self._acceleration
        if data_name in self._context_values:
            return self._context_values[data_name]
        raise ConnectionError(f"unexpected register in fake: {data_name}")

    def write(self, data_name, motor, value, *, normalize=True, num_retry=0):
        if motor != TARGET_JOINT:
            raise _ForbiddenWriteCalled(f"wrist_roll 외 관절 write 시도: motor={motor}")
        if data_name != "Acceleration":
            raise _ForbiddenWriteCalled(f"Acceleration 외 레지스터 write 시도: data_name={data_name}")
        if len(self.write_calls) >= 1:
            raise _ForbiddenWriteCalled("두 번째 write 시도")
        self.write_calls.append((data_name, motor, value, normalize, num_retry))
        self._acceleration = value

    def sync_write(self, *args, **kwargs):
        raise _ForbiddenWriteCalled(f"sync_write() 호출됨(전체 bus write 금지): args={args} kwargs={kwargs}")

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


def _build_writer_with_fake_bus(**bus_kwargs) -> tuple[SingleJointParameterWriter, FakeParameterFeetechBus]:
    writer = SingleJointParameterWriter(port="/dev/null", calibration=WRIST_ROLL_CALIBRATION)
    fake_bus = FakeParameterFeetechBus(**bus_kwargs)
    writer._bus = fake_bus  # noqa: SLF001 - 의도적인 테스트용 내부 교체
    return writer, fake_bus


def test_writer_bus_only_registers_wrist_roll_motor():
    writer = SingleJointParameterWriter(port="/dev/null", calibration=WRIST_ROLL_CALIBRATION)
    assert set(writer._bus.motors) == {TARGET_JOINT}  # noqa: SLF001


def test_write_acceleration_once_writes_target_value_on_wrist_roll_only():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    writer.write_acceleration_once()

    assert len(fake_bus.write_calls) == 1
    data_name, motor, value, normalize, num_retry = fake_bus.write_calls[0]
    assert data_name == "Acceleration"
    assert motor == TARGET_JOINT
    assert value == TARGET_ACCELERATION_VALUE
    assert num_retry == 0
    assert writer.write_count == 1


def test_second_write_attempt_raises_before_touching_bus():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    writer.write_acceleration_once()

    with pytest.raises(WriteBudgetExceededError):
        writer.write_acceleration_once()

    assert len(fake_bus.write_calls) == 1  # 두 번째 시도는 가짜 버스에 도달하지도 않는다


def test_fake_bus_raises_if_writer_ever_targets_other_register():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    with pytest.raises(_ForbiddenWriteCalled):
        fake_bus.write("Torque_Enable", TARGET_JOINT, 1)


def test_fake_bus_raises_if_writer_ever_targets_other_motor():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    with pytest.raises(_ForbiddenWriteCalled):
        fake_bus.write("Acceleration", "gripper", 254)


def test_fake_bus_raises_on_sync_write_call():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    with pytest.raises(_ForbiddenWriteCalled):
        fake_bus.sync_write("Acceleration", {TARGET_JOINT: 254})


def test_read_acceleration_and_read_context_never_write():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    writer.read_acceleration()
    writer.read_context()
    assert fake_bus.write_calls == []


def test_disconnect_never_writes():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    writer.disconnect()
    assert fake_bus.disconnect_calls == [False]
    assert fake_bus.write_calls == []


def test_writer_exposes_only_expected_interface():
    # dir(class)는 인스턴스 속성(port/calibration - __init__에서 self.x = ...로 설정됨)을
    # 보여주지 않는다 - 프로퍼티/메서드만 확인한다(tests/test_single_joint_hardware_inspector.py
    # 의 동일 패턴 참고).
    class_level_attrs = {name for name in dir(SingleJointParameterWriter) if not name.startswith("_")}
    assert class_level_attrs == {
        "is_connected",
        "write_count",
        "connect",
        "read_acceleration",
        "read_context",
        "write_acceleration_once",
        "disconnect",
    }


# ---------------------------------------------------------------------------
# execute_single_parameter_write + FakeParameterWriter (순수 파이썬, lerobot 불필요)
# ---------------------------------------------------------------------------


class FakeParameterWriter:
    """``_ParameterWriterLike`` 프로토콜만 구현하는 순수 파이썬 가짜."""

    def __init__(
        self,
        *,
        current_acceleration: int,
        readback_acceleration: int | None = None,
        calibration: WristRollCalibration = WRIST_ROLL_CALIBRATION,
        raise_on_current_read: Exception | None = None,
        raise_on_readback: Exception | None = None,
        raise_on_write: Exception | None = None,
    ) -> None:
        self._current = current_acceleration
        self._readback = readback_acceleration if readback_acceleration is not None else current_acceleration
        self.calibration = calibration
        self._raise_on_current_read = raise_on_current_read
        self._raise_on_readback = raise_on_readback
        self._raise_on_write = raise_on_write
        self._write_count = 0
        self._written = False
        self.write_calls = 0
        self.context_read_calls = 0

    @property
    def write_count(self) -> int:
        return self._write_count

    def read_acceleration(self) -> int:
        if not self._written and self._raise_on_current_read is not None:
            raise self._raise_on_current_read
        if self._written and self._raise_on_readback is not None:
            raise self._raise_on_readback
        return self._readback if self._written else self._current

    def read_context(self) -> dict:
        self.context_read_calls += 1
        return {"Torque_Enable": 1, "Goal_Position": 2021, "Present_Position": 2023}

    def write_acceleration_once(self) -> None:
        if self._write_count >= 1:
            raise WriteBudgetExceededError("두 번째 write 시도")
        self._write_count += 1
        self.write_calls += 1
        if self._raise_on_write is not None:
            raise self._raise_on_write
        self._written = True


# -- 확인 플래그 --------------------------------------------------------------


def test_no_confirmation_flags_blocks_and_zero_writes():
    writer = FakeParameterWriter(current_acceleration=0)
    result = _call(writer, confirmation_flags={})
    assert result.final_verdict == BLOCKED
    assert result.write_count_after == 0
    assert writer.write_calls == 0


def test_one_confirmation_flag_blocks_and_zero_writes():
    writer = FakeParameterWriter(current_acceleration=0)
    result = _call(
        writer,
        confirmation_flags={"i_understand_this_changes_servo_state": True, "confirm_acceleration_write": False},
    )
    assert result.final_verdict == BLOCKED
    assert result.checks["confirmation_flags_check"] == BLOCKED
    assert result.write_count_after == 0


def test_both_confirmation_flags_present_proceeds_past_that_check():
    writer = FakeParameterWriter(current_acceleration=0)
    result = _call(writer)
    assert result.checks["confirmation_flags_check"] == PASS


# -- motor id --------------------------------------------------------------


def test_wrong_motor_id_blocks_and_zero_writes():
    writer = FakeParameterWriter(current_acceleration=0, calibration=WRONG_MOTOR_CALIBRATION)
    result = _call(writer)
    assert result.final_verdict == BLOCKED
    assert result.checks["motor_id_check"] == BLOCKED
    assert result.write_count_after == 0
    assert writer.write_calls == 0


# -- 현재값 불일치 -------------------------------------------------------------


def test_current_value_mismatch_blocks_and_zero_writes():
    writer = FakeParameterWriter(current_acceleration=100)  # expected 기본값 0과 다름
    result = _call(writer)
    assert result.final_verdict == BLOCKED
    assert result.checks["expected_current_match_check"] == BLOCKED
    assert result.measured_current_acceleration == 100
    assert result.write_count_after == 0
    assert writer.write_calls == 0


def test_current_value_matches_expected_proceeds_to_write():
    writer = FakeParameterWriter(current_acceleration=0, readback_acceleration=254)
    result = _call(writer, expected_current_acceleration=0)
    assert result.checks["expected_current_match_check"] == PASS
    assert result.write_executed is True


def test_current_read_failure_blocks_before_any_write():
    writer = FakeParameterWriter(current_acceleration=0, raise_on_current_read=ConnectionError("comm lost"))
    result = _call(writer)
    assert result.checks["current_read_check"] == BLOCKED
    assert result.final_verdict == BLOCKED
    assert result.write_count_after == 0
    assert writer.write_calls == 0


# -- write budget ------------------------------------------------------------


def test_write_budget_exceeded_blocks_second_orchestration_call():
    writer = FakeParameterWriter(current_acceleration=0, readback_acceleration=254)
    first = _call(writer)
    assert first.write_executed is True
    assert writer.write_count == 1

    second = _call(writer, expected_current_acceleration=254)  # 이제 실측이 254로 바뀜
    assert second.checks["write_budget_check"] == BLOCKED
    assert second.write_executed is False
    assert writer.write_count == 1  # 두 번째 호출로 늘지 않는다(retry 없음)


# -- write/readback 성공 ------------------------------------------------------


def test_successful_write_and_matching_readback_is_pass():
    writer = FakeParameterWriter(current_acceleration=0, readback_acceleration=254)
    result = _call(writer, expected_current_acceleration=0)
    assert result.write_executed is True
    assert result.readback_acceleration == 254
    assert result.readback_matches_target is True
    assert result.final_verdict == PASS


def test_readback_mismatch_after_successful_write():
    writer = FakeParameterWriter(current_acceleration=0, readback_acceleration=100)  # write는 됐지만 다른 값으로 읽힘
    result = _call(writer, expected_current_acceleration=0)
    assert result.write_executed is True
    assert result.readback_matches_target is False
    assert result.final_verdict == READBACK_MISMATCH


def test_write_call_raising_is_write_failed_with_no_readback_attempt():
    writer = FakeParameterWriter(current_acceleration=0, raise_on_write=ConnectionError("packet timeout"))
    result = _call(writer, expected_current_acceleration=0)
    assert result.final_verdict == WRITE_FAILED
    assert result.write_executed is False
    assert result.write_count_after == 1  # budget은 시도 자체로 이미 소비됨(retry 불가)


def test_readback_read_failure_is_readback_failed():
    writer = FakeParameterWriter(current_acceleration=0, raise_on_readback=ConnectionError("comm lost"))
    result = _call(writer, expected_current_acceleration=0)
    assert result.write_executed is True
    assert result.final_verdict == READBACK_FAILED


# -- context 로깅 (참고용, 판정에 영향 없음) -----------------------------------


def test_context_is_read_and_included_in_result():
    writer = FakeParameterWriter(current_acceleration=0, readback_acceleration=254)
    result = _call(writer, expected_current_acceleration=0)
    assert writer.context_read_calls == 1
    assert result.context == {"Torque_Enable": 1, "Goal_Position": 2021, "Present_Position": 2023}


def test_context_not_read_when_blocked_before_that_step():
    writer = FakeParameterWriter(current_acceleration=100)  # expected_current_match에서 BLOCKED
    _call(writer)
    assert writer.context_read_calls == 0


# -- write budget 전체 상한 (요구사항: 이 모듈로 인한 write는 항상 최대 1) --------


def test_write_count_never_exceeds_one_across_multiple_scenarios():
    scenarios = [
        FakeParameterWriter(current_acceleration=0, readback_acceleration=254),
        FakeParameterWriter(current_acceleration=100),  # BLOCKED, write 0
        FakeParameterWriter(current_acceleration=0, readback_acceleration=100),  # MISMATCH지만 write는 1회
    ]
    for writer in scenarios:
        result = _call(writer, expected_current_acceleration=0)
        assert result.write_count_after <= 1
        assert writer.write_count <= 1


def test_execute_single_parameter_write_never_references_goal_position_write():
    import inspect

    source = inspect.getsource(execute_single_parameter_write)
    assert "write_goal_position_once" not in source
    assert "execute_single_armed_write" not in source


def test_parameter_write_result_to_dict_has_no_secrets_or_home_paths():
    writer = FakeParameterWriter(current_acceleration=0, readback_acceleration=254)
    result = _call(writer, expected_current_acceleration=0)
    d = result.to_dict()
    serialized = str(d)
    for forbidden in ("token", "Token", "TOKEN", "password", "secret", "Authorization", "Bearer", "/home/"):
        assert forbidden not in serialized


def test_parameter_write_result_default_final_verdict_is_blocked():
    assert ParameterWriteResult().final_verdict == BLOCKED
