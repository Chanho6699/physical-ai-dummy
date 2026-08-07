"""hardware/safety/single_joint_writer.py 단위 테스트.

실물 ``/dev/tty*``/``/dev/serial*``에는 절대 접근하지 않는다. 두 층으로 나눠 검증한다:

1. ``SingleJointArmedWriter`` + 가짜 ``FeetechMotorsBus``(``tests/test_readonly_so101_reader.py``,
   ``tests/test_single_joint_hardware_inspector.py``와 동일한 패턴) - motor id 5 외
   write, ``Goal_Position`` 외 레지스터 write, 두 번째 write가 즉시 ``AssertionError``를
   내는지 확인한다. 여기서는 lerobot이 필요하다(``pytest.importorskip``).
2. ``execute_single_armed_write`` + 순수 파이썬 ``FakeArmedWriter``(``_WriterLike``
   프로토콜만 구현, lerobot 불필요) - 확인 플래그/expected-start/이음매·calibration/
   raw delta 범위/readback 판정 전체 오케스트레이션을 검증한다.

이 파일은 ``--mode armed`` CLI 진입점(``scripts/run_single_joint_hardware_test.py``의
``_run_armed``)을 어디에서도 호출하지 않는다 - 작업 지시(Claude Code가 armed 모드를
실행해서는 안 됨)를 그대로 따른 것이다. ``execute_single_armed_write``/
``SingleJointArmedWriter``는 armed 모드가 내부적으로 쓰는 "라이브러리 함수"일 뿐이며,
이를 fake bus로 단위 테스트하는 것은 CLI의 ``--mode armed``를 실행하는 것과 다르다.
"""

from __future__ import annotations

import pytest

from hardware.safety.single_joint_bus import WristRollCalibration
from hardware.safety.single_joint_test_planner import (
    BLOCKED,
    PASS,
    READBACK_DIRECTION_MISMATCH,
    READBACK_FAILED,
    READBACK_NO_MOTION,
    READBACK_OVERSHOOT,
    READBACK_PASS,
    TARGET_JOINT,
    WRITE_FAILED,
)
from hardware.safety.single_joint_writer import (
    REQUIRED_CONFIRMATION_FLAGS,
    ArmedWriteResult,
    SingleJointArmedWriter,
    WriteBudget,
    WriteBudgetExceededError,
    execute_single_armed_write,
)

WRIST_ROLL_CALIBRATION = WristRollCalibration(motor_id=5, drive_mode=0, homing_offset=1627, range_min=0, range_max=4095)

BOTH_CONFIRMED = {"i_have_read_the_safety_plan": True, "confirm_single_write": True}

# 실제 lerobot DEGREES 정규화 공식과 동일 (raw=2048 근처는 조사 세션에서 실측된 값).
_RANGE_MIN, _RANGE_MAX, _MOTOR_RES = 0, 4095, 4096


def _raw_to_deg(raw: float) -> float:
    mid = (_RANGE_MIN + _RANGE_MAX) / 2.0
    return (raw - mid) * 360.0 / (_MOTOR_RES - 1)


# ---------------------------------------------------------------------------
# WriteBudget: 순수 로직
# ---------------------------------------------------------------------------


def test_write_budget_allows_exactly_one_consume():
    budget = WriteBudget(max_write_count=1)
    assert budget.write_count == 0
    budget.consume()
    assert budget.write_count == 1


def test_write_budget_second_consume_raises():
    budget = WriteBudget(max_write_count=1)
    budget.consume()
    with pytest.raises(WriteBudgetExceededError):
        budget.consume()
    assert budget.write_count == 1  # 실패한 두 번째 시도로 카운트가 더 늘지 않는다


# ---------------------------------------------------------------------------
# SingleJointArmedWriter + 가짜 FeetechMotorsBus (motor id 5 / Goal_Position / 1회 한정)
# ---------------------------------------------------------------------------

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")


class _ForbiddenWriteCalled(AssertionError):
    """가짜 버스가 허용되지 않은 쓰기를 감지하면 즉시 테스트를 실패시킨다."""


class FakeArmedFeetechBus:
    """``FeetechMotorsBus``를 대체하는 테스트 전용 가짜 버스.

    - motor id 5(wrist_roll) 외 write는 즉시 ``AssertionError``.
    - ``Goal_Position`` 외 레지스터 write는 즉시 ``AssertionError``.
    - 두 번째 write는 즉시 ``AssertionError``.
    - ``sync_write``(전체 bus write) 호출은 무조건 ``AssertionError``.
    """

    def __init__(self, *, start_raw: int = 2048) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []
        self.write_calls: list[tuple] = []
        self.sync_read_motor_args: list[list[str]] = []
        self.port = "/dev/null"
        self._raw_position = start_raw

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, handshake: bool = True) -> None:
        self.connect_calls += 1
        self.connected = True

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        assert data_name == "Present_Position"
        motor_list = list(motors) if motors is not None else []
        assert motor_list == [TARGET_JOINT], f"wrist_roll 외 관절 read 시도: {motor_list}"
        self.sync_read_motor_args.append(motor_list)
        if normalize:
            return {TARGET_JOINT: _raw_to_deg(self._raw_position)}
        return {TARGET_JOINT: self._raw_position}

    def write(self, data_name, motor, value, *, normalize=True, num_retry=0):
        if motor != TARGET_JOINT:
            raise _ForbiddenWriteCalled(f"wrist_roll 외 관절 write 시도: motor={motor}")
        if data_name != "Goal_Position":
            raise _ForbiddenWriteCalled(f"Goal_Position 외 레지스터 write 시도: data_name={data_name}")
        if len(self.write_calls) >= 1:
            raise _ForbiddenWriteCalled("두 번째 write 시도")
        self.write_calls.append((data_name, motor, value, normalize, num_retry))

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


def _build_writer_with_fake_bus(**bus_kwargs) -> tuple[SingleJointArmedWriter, FakeArmedFeetechBus]:
    writer = SingleJointArmedWriter(port="/dev/null", calibration=WRIST_ROLL_CALIBRATION)
    fake_bus = FakeArmedFeetechBus(**bus_kwargs)
    writer._bus = fake_bus  # noqa: SLF001 - 의도적인 테스트용 내부 교체
    return writer, fake_bus


def test_writer_bus_only_registers_wrist_roll_motor():
    writer = SingleJointArmedWriter(port="/dev/null", calibration=WRIST_ROLL_CALIBRATION)
    assert set(writer._bus.motors) == {TARGET_JOINT}  # noqa: SLF001


def test_write_goal_position_once_calls_goal_position_on_wrist_roll_only():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    writer.write_goal_position_once(target_deg=0.1)

    assert len(fake_bus.write_calls) == 1
    data_name, motor, value, normalize, num_retry = fake_bus.write_calls[0]
    assert data_name == "Goal_Position"
    assert motor == TARGET_JOINT
    assert normalize is True
    assert num_retry == 0
    assert writer.write_count == 1


def test_second_write_attempt_raises_before_touching_bus():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    writer.write_goal_position_once(target_deg=0.1)

    with pytest.raises(WriteBudgetExceededError):
        writer.write_goal_position_once(target_deg=0.2)

    # budget이 먼저 막아서 가짜 버스의 write()에는 두 번째 호출이 도달하지도 않는다.
    assert len(fake_bus.write_calls) == 1


def test_fake_bus_raises_if_writer_ever_targets_other_motor():
    """writer가 (버그로) 다른 관절 이름을 넘기면 가짜 버스가 즉시 AssertionError를 낸다."""
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    with pytest.raises(_ForbiddenWriteCalled):
        fake_bus.write("Goal_Position", "gripper", 0.0)


def test_fake_bus_raises_if_writer_ever_targets_other_register():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    with pytest.raises(_ForbiddenWriteCalled):
        fake_bus.write("Torque_Enable", TARGET_JOINT, 1)


def test_fake_bus_raises_on_sync_write_call():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    with pytest.raises(_ForbiddenWriteCalled):
        fake_bus.sync_write("Goal_Position", {TARGET_JOINT: 0.1})


def test_read_raw_and_read_degrees_never_touch_other_joints():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    writer.read_raw()
    writer.read_degrees()
    for requested in fake_bus.sync_read_motor_args:
        assert requested == [TARGET_JOINT]


def test_disconnect_never_writes_and_disables_torque_flag_is_false():
    writer, fake_bus = _build_writer_with_fake_bus()
    writer.connect()
    writer.disconnect()
    assert fake_bus.disconnect_calls == [False]
    assert fake_bus.write_calls == []


def test_writer_exposes_only_expected_interface():
    class_level_attrs = {name for name in dir(SingleJointArmedWriter) if not name.startswith("_")}
    assert class_level_attrs == {
        "is_connected",
        "write_count",
        "connect",
        "read_raw",
        "read_degrees",
        "write_goal_position_once",
        "disconnect",
    }


# ---------------------------------------------------------------------------
# execute_single_armed_write + FakeArmedWriter (순수 파이썬, lerobot 불필요)
# ---------------------------------------------------------------------------


class FakeArmedWriter:
    """``_WriterLike`` 프로토콜만 구현하는 순수 파이썬 가짜 (lerobot/serial 전혀 사용 안 함)."""

    def __init__(
        self,
        *,
        start_raw: int,
        post_write_raw: int | None = None,
        raise_on_initial_read: Exception | None = None,
        raise_on_readback: Exception | None = None,
        raise_on_write: Exception | None = None,
    ) -> None:
        self._start_raw = start_raw
        self._post_write_raw = post_write_raw if post_write_raw is not None else start_raw
        self._raise_on_initial_read = raise_on_initial_read
        self._raise_on_readback = raise_on_readback
        self._raise_on_write = raise_on_write
        self._write_count = 0
        self._written = False
        self.write_calls: list[float] = []

    @property
    def write_count(self) -> int:
        return self._write_count

    def read_raw(self) -> int:
        if not self._written and self._raise_on_initial_read is not None:
            raise self._raise_on_initial_read
        if self._written and self._raise_on_readback is not None:
            raise self._raise_on_readback
        return self._post_write_raw if self._written else self._start_raw

    def read_degrees(self) -> float:
        return _raw_to_deg(self.read_raw())

    def write_goal_position_once(self, *, target_deg: float) -> None:
        if self._write_count >= 1:
            raise WriteBudgetExceededError("두 번째 write 시도")
        self._write_count += 1
        self.write_calls.append(target_deg)
        if self._raise_on_write is not None:
            raise self._raise_on_write
        self._written = True


def _no_sleep(_seconds: float) -> None:
    return None


def _call(writer, *, direction="positive", expected_start_raw=None, expected_start_deg=None, confirmation_flags=None):
    return execute_single_armed_write(
        writer=writer,
        direction=direction,
        calibration=WRIST_ROLL_CALIBRATION,
        expected_start_raw=expected_start_raw,
        expected_start_deg=expected_start_deg,
        confirmation_flags=BOTH_CONFIRMED if confirmation_flags is None else confirmation_flags,
        sleep_fn=_no_sleep,
    )


# -- 1~4: 확인 플래그 / expected start --------------------------------------


def test_no_confirmation_flags_blocks_and_zero_writes():
    writer = FakeArmedWriter(start_raw=2048)
    result = _call(writer, expected_start_raw=2048, confirmation_flags={})
    assert result.final_verdict == BLOCKED
    assert result.write_count_after == 0
    assert writer.write_calls == []


def test_one_confirmation_flag_blocks_and_zero_writes():
    writer = FakeArmedWriter(start_raw=2048)
    result = _call(
        writer, expected_start_raw=2048, confirmation_flags={"i_have_read_the_safety_plan": True, "confirm_single_write": False}
    )
    assert result.final_verdict == BLOCKED
    assert result.checks["confirmation_flags_check"] == BLOCKED
    assert result.write_count_after == 0


def test_missing_expected_start_blocks_and_zero_writes():
    writer = FakeArmedWriter(start_raw=2048)
    result = _call(writer, expected_start_raw=None, expected_start_deg=None)
    assert result.final_verdict == BLOCKED
    assert result.checks["expected_start_provided_check"] == BLOCKED
    assert result.write_count_after == 0


def test_expected_start_mismatch_blocks_and_zero_writes():
    writer = FakeArmedWriter(start_raw=2048)
    result = _call(writer, expected_start_raw=2048 + 10)  # 허용 오차(±2 tick)를 크게 초과
    assert result.final_verdict == BLOCKED
    assert result.checks["expected_start_match_check"] == BLOCKED
    assert result.write_count_after == 0


def test_expected_start_within_tolerance_proceeds_to_write():
    writer = FakeArmedWriter(start_raw=2048, post_write_raw=2049)
    result = _call(writer, direction="positive", expected_start_raw=2048 + 1)  # ±2 tick 이내
    assert result.checks["expected_start_match_check"] == PASS
    assert result.write_executed is True


# -- 5~8: 0.1°/1~2 tick 상한 (build_armed_single_step_plan을 통해) -----------


def test_delta_0_2_degree_is_blocked_at_planner_level():
    from hardware.safety.single_joint_test_planner import PlannerConfigError, build_armed_single_step_plan

    with pytest.raises(PlannerConfigError):
        build_armed_single_step_plan(
            start_deg=0.0, start_raw=2048, direction="positive", range_min=0, range_max=4095, requested_delta_deg=0.2
        )


def test_step_0_2_degree_is_blocked_at_planner_level():
    from hardware.safety.single_joint_test_planner import PlannerConfigError, build_armed_single_step_plan

    with pytest.raises(PlannerConfigError):
        build_armed_single_step_plan(
            start_deg=0.0, start_raw=2048, direction="positive", range_min=0, range_max=4095, step_size_deg=0.2
        )


def test_raw_delta_zero_tick_is_blocked():
    from hardware.safety.single_joint_test_planner import build_armed_single_step_plan

    # start_deg=0.0, target_deg=0.1 -> target_raw=int(0.1*4095/360+2047.5)=2048.
    # start_raw를 일부러 2048로 맞춰 delta=0을 강제한다 (경계값 검증 목적).
    plan = build_armed_single_step_plan(start_deg=0.0, start_raw=2048, direction="positive", range_min=0, range_max=4095)
    assert plan.command_raw_delta == 0
    assert plan.checks["command_raw_delta_check"] == BLOCKED
    assert plan.final_verdict == BLOCKED


def test_raw_delta_three_ticks_is_blocked():
    from hardware.safety.single_joint_test_planner import build_armed_single_step_plan

    # 위와 동일한 target_raw=2048 기준, start_raw=2045로 두면 delta=3.
    plan = build_armed_single_step_plan(start_deg=0.0, start_raw=2045, direction="positive", range_min=0, range_max=4095)
    assert plan.command_raw_delta == 3
    assert plan.checks["command_raw_delta_check"] == BLOCKED
    assert plan.final_verdict == BLOCKED


# -- 9~12: fake bus write 위반 (SingleJointArmedWriter 레벨에서 이미 위에서 검증) ----
# (test_fake_bus_raises_if_writer_ever_targets_other_motor 등 - 위 섹션 참고)


# -- 13~14: 방향 일치 readback -> PASS --------------------------------------


def test_positive_command_with_positive_readback_passes():
    writer = FakeArmedWriter(start_raw=2048, post_write_raw=2049)  # +1 tick
    result = _call(writer, direction="positive", expected_start_raw=2048)
    assert result.write_executed is True
    assert result.readback_verdict == READBACK_PASS
    assert result.final_verdict == READBACK_PASS
    assert result.actual_raw_delta == 1


def test_negative_command_with_negative_readback_passes():
    writer = FakeArmedWriter(start_raw=2048, post_write_raw=2046)  # -2 tick
    result = _call(writer, direction="negative", expected_start_raw=2048)
    assert result.write_executed is True
    assert result.readback_verdict == READBACK_PASS
    assert result.final_verdict == READBACK_PASS
    assert result.actual_raw_delta == -2


# -- 15~18: readback 이상 판정 ------------------------------------------------


def test_no_change_in_readback_is_no_motion():
    writer = FakeArmedWriter(start_raw=2048, post_write_raw=2048)  # 변화 없음
    result = _call(writer, direction="positive", expected_start_raw=2048)
    assert result.readback_verdict == READBACK_NO_MOTION
    assert result.final_verdict == READBACK_NO_MOTION


def test_opposite_direction_motion_is_direction_mismatch():
    writer = FakeArmedWriter(start_raw=2048, post_write_raw=2047)  # positive 명령인데 -1
    result = _call(writer, direction="positive", expected_start_raw=2048)
    assert result.readback_verdict == READBACK_DIRECTION_MISMATCH
    assert result.final_verdict == READBACK_DIRECTION_MISMATCH


def test_overshoot_beyond_four_ticks_is_overshoot():
    writer = FakeArmedWriter(start_raw=2048, post_write_raw=2048 + 6)  # +6 tick, 방향은 일치
    result = _call(writer, direction="positive", expected_start_raw=2048)
    assert result.readback_verdict == READBACK_OVERSHOOT
    assert result.final_verdict == READBACK_OVERSHOOT


def test_readback_read_failure_is_readback_failed():
    writer = FakeArmedWriter(start_raw=2048, raise_on_readback=ConnectionError("comm lost"))
    result = _call(writer, direction="positive", expected_start_raw=2048)
    assert result.write_executed is True
    assert result.readback_verdict == READBACK_FAILED
    assert result.final_verdict == READBACK_FAILED


def test_initial_read_failure_blocks_before_any_write():
    writer = FakeArmedWriter(start_raw=2048, raise_on_initial_read=ConnectionError("comm lost"))
    result = _call(writer, direction="positive", expected_start_raw=2048)
    assert result.checks["initial_read_check"] == BLOCKED
    assert result.final_verdict == BLOCKED
    assert result.write_count_after == 0
    assert writer.write_calls == []


def test_write_call_raising_is_write_failed_with_no_readback_attempt():
    writer = FakeArmedWriter(start_raw=2048, raise_on_write=ConnectionError("packet timeout"))
    result = _call(writer, direction="positive", expected_start_raw=2048)
    assert result.final_verdict == WRITE_FAILED
    assert result.write_executed is False
    # write budget은 시도 자체로 이미 소비되어 재시도가 불가능하다.
    assert result.write_count_after == 1


# -- 19~20: 이음매 / calibration 내부 범위 -----------------------------------


def test_seam_start_position_blocks_with_zero_writes():
    writer = FakeArmedWriter(start_raw=4090)  # raw=4090 -> deg 약 +179.2 (이음매 근처)
    result = _call(writer, direction="positive", expected_start_raw=4090)
    assert result.checks["seam_avoidance_start_check"] == BLOCKED
    assert result.final_verdict == BLOCKED
    assert result.write_count_after == 0


def test_start_outside_calibration_inner_range_blocks_with_zero_writes():
    # margin 기본 15도이므로 inner range=[-165,165]. raw로 약 +170도에 해당하는 지점 사용.
    near_edge_raw = int(170.0 * (4096 - 1) / 360.0 + (0 + 4095) / 2.0)
    writer = FakeArmedWriter(start_raw=near_edge_raw)
    result = _call(writer, direction="positive", expected_start_raw=near_edge_raw)
    assert result.checks["seam_avoidance_start_check"] == BLOCKED
    assert result.final_verdict == BLOCKED
    assert result.write_count_after == 0


# -- 24: 두 번째 orchestration 호출(같은 writer 재사용) -> write 재시도 금지 --------


def test_second_orchestration_call_on_same_writer_is_blocked_without_retry():
    writer = FakeArmedWriter(start_raw=2048, post_write_raw=2049)
    first = _call(writer, direction="positive", expected_start_raw=2048)
    assert first.write_executed is True
    assert writer.write_count == 1

    second = _call(writer, direction="positive", expected_start_raw=2049)
    assert second.checks["write_budget_check"] == BLOCKED
    assert second.write_executed is False
    assert writer.write_count == 1  # 두 번째 호출로 늘지 않는다


# -- 25: 이 파일의 모든 테스트에서 write_count가 1을 넘지 않는다 (개별 assert들로 이미
#        보장되지만, 대표적으로 "여러 시나리오를 순서대로 실행해도 항상 <=1"을 재확인) --


def test_write_count_never_exceeds_one_across_multiple_scenarios():
    scenarios = [
        FakeArmedWriter(start_raw=2048, post_write_raw=2049),
        FakeArmedWriter(start_raw=2048, post_write_raw=2046),
        FakeArmedWriter(start_raw=2048),  # NO_MOTION
    ]
    for writer in scenarios:
        result = _call(writer, direction="positive", expected_start_raw=2048)
        assert result.write_count_after <= 1
        assert writer.write_count <= 1


# -- 26: wrist_roll 외 관절 접근 없음 (FakeArmedWriter/execute_single_armed_write 레벨) --


def test_execute_single_armed_write_never_references_other_joint_names():
    import inspect

    source = inspect.getsource(execute_single_armed_write)
    for other_joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper"):
        assert other_joint not in source


# -- 27: JSON 리포트(ArmedWriteResult.to_dict())에 민감정보 없음 --------------------


def test_armed_write_result_to_dict_has_no_secrets_or_home_paths():
    writer = FakeArmedWriter(start_raw=2048, post_write_raw=2049)
    result = _call(writer, direction="positive", expected_start_raw=2048)
    d = result.to_dict()
    serialized = str(d)
    for forbidden in ("token", "Token", "TOKEN", "password", "secret", "Authorization", "Bearer", "/home/"):
        assert forbidden not in serialized


def test_required_confirmation_flags_constant_has_exactly_two_entries():
    assert set(REQUIRED_CONFIRMATION_FLAGS) == {"i_have_read_the_safety_plan", "confirm_single_write"}


def test_armed_write_result_default_final_verdict_is_blocked():
    # 아무 필드도 안 채우고 만들면(방어적 기본값) final_verdict는 BLOCKED여야 한다 -
    # "명시적으로 PASS를 만들지 않는 한 항상 안전 측" 원칙.
    assert ArmedWriteResult().final_verdict == BLOCKED
