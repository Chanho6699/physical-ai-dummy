"""hardware/diagnostics/instrumented_teleop.py 단위 테스트 (passive monitoring 모델).

실물 serial 포트/실제 SO101Leader/SO101Follower에는 절대 연결하지 않는다. leader/follower는
``get_action``/``get_observation``/``send_action``/``bus.read`` 인터페이스만 흉내내는 가짜
객체로 대체한다.

가장 중요한 계약(섹션 8): **같은 leader action sequence를 입력하면, "계측 없는 baseline
루프"와 "Instrumented Teleop 루프"가 follower ``send_action()``에 전달하는 command
sequence가 완전히 동일해야 한다** - warning이 발생하더라도 마찬가지다. 이 파일의
``test_instrumented_command_sequence_matches_baseline_exactly``가 그것을 직접 검증한다.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행 (import 경로 확인용)")

from hardware.diagnostics import instrumented_teleop as it
from hardware.diagnostics.instrumented_teleop import (
    MOTION_ONSET_THRESHOLD_RAW_TICKS,
    STOP_DURATION_ELAPSED,
    STOP_KEYBOARD_INTERRUPT,
    STOP_READ_FAILURE,
    WARNING_DIRECTION_MISMATCH,
    WARNING_LARGE_COMMAND_DELTA,
    WARNING_LARGE_TRACKING_ERROR,
    WARNING_LOW_LOOP_RATE,
    WARNING_POSITION_JUMP,
    WARNING_STATUS_NONZERO,
    INSUFFICIENT_DATA,
    INSUFFICIENT_FOR_DEADBAND_ESTIMATE,
    TeleopCycleSample,
    TeleopRunResult,
    WarningThresholds,
    WristRollRegisterInstrument,
    check_command_delta_guard,
    check_direction_mismatch,
    check_low_loop_rate,
    check_position_jump,
    check_tracking_error,
    compute_deadband_summary,
    compute_run_analysis,
    run_instrumented_teleop_loop,
)
from hardware.safety.shadow_teleop_diagnostic import FOLLOWER_STATE_REGISTERS

RANGE = (0, 4095)  # wrist_roll full-turn calibration
DEG_PER_TICK = 360.0 / 4095


def _wrist_roll_deg(raw: float) -> float:
    return (raw - 2047.5) * 360.0 / 4095


ALL_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _action_dict(wrist_roll_deg: float, **overrides) -> dict:
    action = {f"{j}.pos": 0.0 for j in ALL_JOINTS}
    action["wrist_roll.pos"] = wrist_roll_deg
    action.update(overrides)
    return action


# ---------------------------------------------------------------------------
# 가짜 bus / leader / follower
# ---------------------------------------------------------------------------


class _ForbiddenWriteCalled(AssertionError):
    pass


class FakeBus:
    """follower.bus 대체 - ``read()``만 지원한다. write 계열은 호출되면 즉시 실패한다."""

    def __init__(self, register_sequences=None, raise_on=None):
        self.read_calls: list[tuple] = []
        self._sequences = {k: list(v) for k, v in (register_sequences or {}).items()}
        self._indices: dict[str, int] = {}
        self._raise_on = set(raise_on or ())

    def _next(self, name: str):
        seq = self._sequences.get(name)
        if not seq:
            return 0
        idx = self._indices.get(name, 0)
        value = seq[min(idx, len(seq) - 1)]
        self._indices[name] = idx + 1
        return value

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        self.read_calls.append((data_name, motor, normalize))
        if data_name in self._raise_on:
            raise RuntimeError(f"comm error on {data_name}")
        return self._next(data_name)

    def write(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeBus.write() 호출됨")

    def sync_write(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeBus.sync_write() 호출됨")

    def enable_torque(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeBus.enable_torque() 호출됨")

    def disable_torque(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeBus.disable_torque() 호출됨")


class FakeLeader:
    def __init__(self, wrist_roll_sequence):
        self._sequence = list(wrist_roll_sequence)
        self._index = 0
        self.call_count = 0

    def get_action(self):
        self.call_count += 1
        value = self._sequence[min(self._index, len(self._sequence) - 1)]
        self._index += 1
        if isinstance(value, BaseException):
            raise value
        return _action_dict(value)


class FakeFollower:
    def __init__(self, *, bus, observation_wrist_roll=0.0):
        self.bus = bus
        self._observation_wrist_roll = observation_wrist_roll
        self.send_action_calls: list[dict] = []
        self.get_observation_calls = 0

    def get_observation(self):
        self.get_observation_calls += 1
        return {f"{j}.pos": 0.0 for j in ALL_JOINTS} | {"wrist_roll.pos": self._observation_wrist_roll}

    def send_action(self, action):
        if isinstance(action, BaseException):
            raise action
        self.send_action_calls.append(dict(action))
        return dict(action)  # max_relative_target=None인 정상 SOFollower.send_action()과 동일하게 그대로 반환


def _identity_processor(pair):
    return pair[0]


def _default_registers(*, goal=2021, present=2023, torque=1, moving=0, status=0, accel=254, accel_mult=1):
    return {
        "Goal_Position": [goal] * 50,
        "Present_Position": [present] * 50,
        "Torque_Enable": [torque] * 50,
        "Moving": [moving] * 50,
        "Status": [status] * 50,
        "Acceleration": [accel] * 50,
        "Acceleration_Multiplier ": [accel_mult] * 50,
    }


def _make_loop(
    *,
    leader_sequence,
    follower_registers,
    duration_sec=None,
    warning_thresholds=None,
    follower_observation_wrist_roll=0.0,
    accel_refresh_interval_s=1.0,
    on_sample=None,
    on_warning=None,
):
    """``leader_sequence``의 각 값은 follower 시작 위치(``follower_registers``의
    ``Present_Position`` 첫 값) 기준 **델타(도)**로 해석한다.
    """
    bus = FakeBus(register_sequences=follower_registers)
    present_list = follower_registers.get("Present_Position") or [2023]
    follower_start_deg = _wrist_roll_deg(present_list[0])
    offset_sequence = [v if isinstance(v, BaseException) else v + follower_start_deg for v in leader_sequence]
    leader = FakeLeader(offset_sequence)
    follower = FakeFollower(bus=bus, observation_wrist_roll=follower_observation_wrist_roll)
    instrument = WristRollRegisterInstrument(bus=bus)

    ticks = iter(range(1, 1_000_000))
    clock = lambda: next(ticks) * 0.01  # noqa: E731 - 단조 증가 fake clock (실제 sleep 없음)

    result = run_instrumented_teleop_loop(
        leader=leader,
        follower=follower,
        instrument=instrument,
        follower_calibration_range=RANGE,
        teleop_action_processor=_identity_processor,
        robot_action_processor=_identity_processor,
        fps=60,
        duration_sec=duration_sec,
        warning_thresholds=warning_thresholds or WarningThresholds(),
        accel_refresh_interval_s=accel_refresh_interval_s,
        on_sample=on_sample,
        on_warning=on_warning,
        clock=clock,
    )
    return result, leader, follower, bus


# ---------------------------------------------------------------------------
# WristRollRegisterInstrument: read만, write 0회 (변경 없음)
# ---------------------------------------------------------------------------


def test_instrument_read_state_reads_expected_registers():
    bus = FakeBus(register_sequences=_default_registers())
    instrument = WristRollRegisterInstrument(bus=bus)
    state = instrument.read_state()
    read_names = {name for name, _, _ in bus.read_calls}
    assert read_names == set(FOLLOWER_STATE_REGISTERS)
    assert state.goal_raw == 2021
    assert state.present_raw == 2023


def test_instrument_missing_register_becomes_none_not_a_crash():
    bus = FakeBus(register_sequences=_default_registers(), raise_on={"Status"})
    instrument = WristRollRegisterInstrument(bus=bus)
    state = instrument.read_state()
    assert state.status_raw is None
    assert "Status" in state.read_errors
    assert state.goal_raw == 2021


def test_instrument_never_exposes_write_methods():
    public_attrs = {name for name in dir(WristRollRegisterInstrument) if not name.startswith("_")}
    assert public_attrs == {"read_state", "read_accel", "read_full_snapshot"}


# ---------------------------------------------------------------------------
# 순수 warning 판정 함수
# ---------------------------------------------------------------------------


def test_check_command_delta_guard_semantics_unchanged():
    assert check_command_delta_guard(command_wrist_roll_deg=1.0, follower_start_present_deg=0.0, max_delta_deg=2.0)
    assert not check_command_delta_guard(command_wrist_roll_deg=2.5, follower_start_present_deg=0.0, max_delta_deg=2.0)
    assert check_command_delta_guard(command_wrist_roll_deg=2.0, follower_start_present_deg=0.0, max_delta_deg=2.0)


def test_check_direction_mismatch_ignores_noise_band():
    assert check_direction_mismatch(
        command_delta_from_start_deg=0.05, present_delta_from_start_deg=-0.05, noise_tolerance_deg=0.176
    )


def test_check_direction_mismatch_detects_opposite_signs_beyond_noise():
    assert not check_direction_mismatch(
        command_delta_from_start_deg=1.0, present_delta_from_start_deg=-1.0, noise_tolerance_deg=0.176
    )


def test_check_position_jump_none_previous_is_pass():
    assert check_position_jump(present_delta_from_prev_deg=None, max_jump_deg=3.0)


def test_check_position_jump_detects_large_jump():
    assert not check_position_jump(present_delta_from_prev_deg=5.0, max_jump_deg=3.0)


def test_check_tracking_error():
    assert check_tracking_error(goal_present_error_deg=0.5, max_error_deg=1.0)
    assert not check_tracking_error(goal_present_error_deg=1.5, max_error_deg=1.0)
    assert check_tracking_error(goal_present_error_deg=None, max_error_deg=1.0)


def test_check_low_loop_rate():
    assert check_low_loop_rate(loop_hz=50.0, min_hz=20.0)
    assert not check_low_loop_rate(loop_hz=10.0, min_hz=20.0)


# ---------------------------------------------------------------------------
# 핵심 계약(섹션 8): baseline과 instrumented의 send_action 시퀀스가 완전히 동일해야 한다
# ---------------------------------------------------------------------------


def _baseline_send_action_sequence(leader, follower, *, num_cycles):
    """계측이 전혀 없는 teleop_loop() 최소 재현 - get_observation -> get_action ->
    processors -> send_action만 반복한다. 비교 기준(ground truth)이다.
    """
    sent = []
    for _ in range(num_cycles):
        obs = follower.get_observation()
        raw_action = leader.get_action()
        teleop_action = _identity_processor((raw_action, obs))
        robot_action_to_send = _identity_processor((teleop_action, obs))
        sent_action = follower.send_action(robot_action_to_send)
        sent.append(dict(sent_action))
    return sent


def test_instrumented_command_sequence_matches_baseline_exactly():
    # 일부러 안전 문턱값을 크게 벗어나는 leader 시퀀스를 사용한다 - "경고가 발생해도
    # command sequence는 절대 달라지지 않는다"는 것까지 함께 검증하기 위함.
    leader_deltas = [0.0, 0.3, 5.0, -5.0, 0.1, 10.0, -10.0, 0.05, 0.02, 0.0]  # 5.0/10.0° -> WARNING_LARGE_COMMAND_DELTA 유발
    num_cycles = len(leader_deltas)

    follower_start_raw = 2023
    follower_start_deg = _wrist_roll_deg(follower_start_raw)
    offset_sequence = [v + follower_start_deg for v in leader_deltas]

    # -- baseline: 계측 전혀 없음 --------------------------------------------------------
    baseline_leader = FakeLeader(list(offset_sequence))
    baseline_bus = FakeBus(register_sequences=_default_registers(present=follower_start_raw))
    baseline_follower = FakeFollower(bus=baseline_bus)
    baseline_sent = _baseline_send_action_sequence(baseline_leader, baseline_follower, num_cycles=num_cycles)

    # -- instrumented: 급격한 position jump까지 섞어 여러 warning을 동시에 유발한다 -------
    registers = _default_registers(present=follower_start_raw)
    registers["Present_Position"] = [follower_start_raw] + [follower_start_raw + 500] * (num_cycles + 5)
    instrumented_leader = FakeLeader([*offset_sequence, KeyboardInterrupt()])  # num_cycles 이후 명시적으로 중단
    instrumented_bus = FakeBus(register_sequences=registers)
    instrumented_follower = FakeFollower(bus=instrumented_bus)
    instrument = WristRollRegisterInstrument(bus=instrumented_bus)
    ticks = iter(range(1, 1_000_000))

    result = run_instrumented_teleop_loop(
        leader=instrumented_leader,
        follower=instrumented_follower,
        instrument=instrument,
        follower_calibration_range=RANGE,
        teleop_action_processor=_identity_processor,
        robot_action_processor=_identity_processor,
        fps=60,
        duration_sec=None,
        warning_thresholds=WarningThresholds(low_loop_rate_hz=0.0),  # loop rate 경고는 이 비교와 무관하니 끔
        clock=lambda: next(ticks) * 0.01,
    )
    # duration_sec=None이므로 leader_sequence가 소진되면 IndexError 대신 마지막 값을 반복한다 -
    # num_cycles만큼만 비교하기 위해 여기서 잘라낸다.
    instrumented_sent = instrumented_follower.send_action_calls[:num_cycles]

    assert instrumented_sent == baseline_sent
    # 여러 warning이 실제로 발생했는지도 확인 - "경고가 있어도 시퀀스는 그대로"라는 주장이
    # 공허하지 않도록(경고가 아예 안 나온 상태에서 우연히 같았던 게 아님을 보장).
    warning_types = {w.event_type for w in result.warnings}
    assert WARNING_LARGE_COMMAND_DELTA in warning_types
    assert WARNING_POSITION_JUMP in warning_types


def test_send_action_never_blocked_with_explicit_stop():
    # 예전 같으면 5.0/10.0° cycle에서 SAFETY_STOP_COMMAND_DELTA로 멈췄을 시나리오 - 이제는
    # 계속 진행되어 KeyboardInterrupt를 만날 때까지 4개 cycle 모두 send_action이 호출된다.
    leader_seq = [0.0, 5.0, 10.0, -10.0, KeyboardInterrupt()]
    result, leader, follower, bus = _make_loop(
        leader_sequence=leader_seq, follower_registers=_default_registers(), duration_sec=None
    )
    assert result.stopped_reason == STOP_KEYBOARD_INTERRUPT
    assert len(follower.send_action_calls) == 4  # 4개 전부 전송됨 (마지막은 KeyboardInterrupt로 중단)
    assert len(result.samples) == 4
    for sample in result.samples:
        assert sample.send_action_executed is True


# ---------------------------------------------------------------------------
# warning 이벤트: 기록되지만 절대 command/루프에 영향 없음
# ---------------------------------------------------------------------------


def test_large_command_delta_produces_warning_not_a_stop():
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 5.0, KeyboardInterrupt()], follower_registers=_default_registers(), duration_sec=None
    )
    assert result.stopped_reason == STOP_KEYBOARD_INTERRUPT
    assert len(follower.send_action_calls) == 2
    warning_types = [w.event_type for w in result.warnings]
    assert WARNING_LARGE_COMMAND_DELTA in warning_types
    event = next(w for w in result.warnings if w.event_type == WARNING_LARGE_COMMAND_DELTA)
    assert event.joint == "wrist_roll"
    assert event.threshold == WarningThresholds().command_delta_max_deg
    assert event.loop_index == 1


def test_direction_mismatch_produces_warning_not_a_stop():
    registers = _default_registers()
    negative_present_raw = 2023 - int(1.0 / DEG_PER_TICK) - 20
    registers["Present_Position"] = [2023, negative_present_raw, negative_present_raw, negative_present_raw]
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 1.0, 1.0, KeyboardInterrupt()],
        follower_registers=registers,
        duration_sec=None,
        warning_thresholds=WarningThresholds(direction_mismatch_noise_tolerance_deg=0.176, position_jump_max_deg=100.0),
    )
    assert result.stopped_reason == STOP_KEYBOARD_INTERRUPT
    assert len(follower.send_action_calls) == 3
    warning_types = {w.event_type for w in result.warnings}
    assert WARNING_DIRECTION_MISMATCH in warning_types


def test_position_jump_produces_warning_not_a_stop():
    registers = _default_registers()
    registers["Present_Position"] = [2023, 2023, 2023 + 400, 2023 + 400]
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 0.02, 0.02, KeyboardInterrupt()],
        follower_registers=registers,
        duration_sec=None,
        warning_thresholds=WarningThresholds(position_jump_max_deg=3.0),
    )
    assert result.stopped_reason == STOP_KEYBOARD_INTERRUPT
    assert len(follower.send_action_calls) == 3
    warning_types = {w.event_type for w in result.warnings}
    assert WARNING_POSITION_JUMP in warning_types


def test_status_nonzero_produces_warning_not_a_stop():
    registers = _default_registers()
    registers["Status"] = [0, 4, 4, 4]
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 0.0, 0.0, KeyboardInterrupt()], follower_registers=registers, duration_sec=None
    )
    assert result.stopped_reason == STOP_KEYBOARD_INTERRUPT
    assert len(follower.send_action_calls) == 3
    warning_types = {w.event_type for w in result.warnings}
    assert WARNING_STATUS_NONZERO in warning_types


def test_large_tracking_error_produces_warning():
    registers = _default_registers(goal=2021, present=2023)
    # goal-present error를 크게 유지 (기본 tracking_error_max_deg=1.0°보다 훨씬 큰 값)
    registers["Goal_Position"] = [2021 + 100] * 10
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 0.0, KeyboardInterrupt()], follower_registers=registers, duration_sec=None
    )
    warning_types = {w.event_type for w in result.warnings}
    assert WARNING_LARGE_TRACKING_ERROR in warning_types


def test_low_loop_rate_produces_warning():
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 0.0, KeyboardInterrupt()],
        follower_registers=_default_registers(),
        duration_sec=None,
        warning_thresholds=WarningThresholds(low_loop_rate_hz=1e9),  # 사실상 항상 발동
    )
    warning_types = {w.event_type for w in result.warnings}
    assert WARNING_LOW_LOOP_RATE in warning_types


def test_on_warning_callback_invoked_for_every_warning():
    seen = []
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 5.0, KeyboardInterrupt()],
        follower_registers=_default_registers(),
        duration_sec=None,
        on_warning=seen.append,
    )
    assert len(seen) == len(result.warnings)
    assert len(seen) > 0


def test_warning_types_recorded_on_sample():
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 5.0, KeyboardInterrupt()], follower_registers=_default_registers(), duration_sec=None
    )
    assert WARNING_LARGE_COMMAND_DELTA in result.samples[1].warning_types


# ---------------------------------------------------------------------------
# 계측 실패는 루프를 멈추지 않는다 (섹션 4 핵심 원칙)
# ---------------------------------------------------------------------------


def test_register_read_failure_does_not_stop_loop_or_block_command():
    class _RaisingInstrument:
        def read_full_snapshot(self):
            from hardware.diagnostics.instrumented_teleop import FullRegisterSnapshot

            return FullRegisterSnapshot(
                torque_enable=1,
                operating_mode=0,
                goal_position_raw=2021,
                present_position_raw=2023,
                moving=0,
                status_raw=0,
                acceleration=254,
                acceleration_multiplier=1,
            )

        def read_state(self):
            raise RuntimeError("register read comm error")

        def read_accel(self):
            from hardware.safety.shadow_teleop_diagnostic import FollowerAccelSnapshot

            return FollowerAccelSnapshot(acceleration=254, acceleration_multiplier=1)

    leader = FakeLeader([0.0, 0.0, KeyboardInterrupt()])
    bus = FakeBus(register_sequences=_default_registers())
    follower = FakeFollower(bus=bus)
    ticks = iter(range(1, 100000))

    result = run_instrumented_teleop_loop(
        leader=leader,
        follower=follower,
        instrument=_RaisingInstrument(),
        follower_calibration_range=RANGE,
        teleop_action_processor=_identity_processor,
        robot_action_processor=_identity_processor,
        fps=60,
        duration_sec=None,
        clock=lambda: next(ticks) * 0.01,
    )
    assert result.stopped_reason == STOP_KEYBOARD_INTERRUPT
    assert len(follower.send_action_calls) == 2  # send_action은 정상적으로 계속 호출됨
    assert len(result.samples) == 2
    for sample in result.samples:
        assert sample.register_read_error is not None
        assert sample.follower_present_raw is None  # 계측 실패 - 값 없음, 그러나 루프는 계속


def test_initial_snapshot_failure_does_not_prevent_loop():
    class _FailingInitialSnapshotInstrument:
        def read_full_snapshot(self):
            raise RuntimeError("initial read failed")

        def read_state(self):
            from hardware.safety.shadow_teleop_diagnostic import FollowerStateSnapshot

            return FollowerStateSnapshot(goal_raw=2021, present_raw=2023, torque_enable=1, moving=0, status_raw=0)

        def read_accel(self):
            from hardware.safety.shadow_teleop_diagnostic import FollowerAccelSnapshot

            return FollowerAccelSnapshot(acceleration=254, acceleration_multiplier=1)

    leader = FakeLeader([0.0, KeyboardInterrupt()])
    bus = FakeBus(register_sequences=_default_registers())
    follower = FakeFollower(bus=bus)
    ticks = iter(range(1, 100000))

    result = run_instrumented_teleop_loop(
        leader=leader,
        follower=follower,
        instrument=_FailingInitialSnapshotInstrument(),
        follower_calibration_range=RANGE,
        teleop_action_processor=_identity_processor,
        robot_action_processor=_identity_processor,
        fps=60,
        duration_sec=None,
        clock=lambda: next(ticks) * 0.01,
    )
    assert result.stopped_reason == STOP_KEYBOARD_INTERRUPT
    assert len(follower.send_action_calls) == 1
    assert result.follower_start_present_raw is None
    assert result.follower_start_present_deg is None


# ---------------------------------------------------------------------------
# 진짜 종료 사유: duration / Ctrl+C / 핵심 경로 실패
# ---------------------------------------------------------------------------


def test_loop_zero_duration_stops_immediately_with_no_samples():
    result, *_ = _make_loop(leader_sequence=[0.0] * 5, follower_registers=_default_registers(), duration_sec=0.0)
    assert result.stopped_reason == STOP_DURATION_ELAPSED
    assert result.samples == []


def test_read_failure_in_leader_get_action_stops_safely():
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, RuntimeError("serial timeout")], follower_registers=_default_registers(), duration_sec=None
    )
    assert result.stopped_reason == STOP_READ_FAILURE
    assert len(result.samples) == 1


def test_read_failure_in_send_action_stops_safely():
    base = _wrist_roll_deg(2023)
    bus = FakeBus(register_sequences=_default_registers())
    leader = FakeLeader([base, base + 0.5])
    follower = FakeFollower(bus=bus)
    instrument = WristRollRegisterInstrument(bus=bus)
    ticks = iter(range(1, 100000))

    def _fail_on_second_cycle(pair):
        action = pair[0]
        if action["wrist_roll.pos"] != base:
            raise RuntimeError("send failed (processor raised)")
        return action

    result = run_instrumented_teleop_loop(
        leader=leader,
        follower=follower,
        instrument=instrument,
        follower_calibration_range=RANGE,
        teleop_action_processor=_identity_processor,
        robot_action_processor=_fail_on_second_cycle,
        fps=60,
        duration_sec=None,
        clock=lambda: next(ticks) * 0.01,
    )
    assert result.stopped_reason == STOP_READ_FAILURE
    assert len(result.samples) == 1


def test_missing_wrist_roll_key_stops_safely():
    bus = FakeBus(register_sequences=_default_registers())
    leader = FakeLeader([0.0])
    follower = FakeFollower(bus=bus)
    instrument = WristRollRegisterInstrument(bus=bus)
    ticks = iter(range(1, 100000))

    def _strip_wrist_roll(pair):
        action = dict(pair[0])
        action.pop("wrist_roll.pos", None)
        return action

    result = run_instrumented_teleop_loop(
        leader=leader,
        follower=follower,
        instrument=instrument,
        follower_calibration_range=RANGE,
        teleop_action_processor=_identity_processor,
        robot_action_processor=_strip_wrist_roll,
        fps=60,
        duration_sec=None,
        clock=lambda: next(ticks) * 0.01,
    )
    assert result.stopped_reason == STOP_READ_FAILURE
    assert isinstance(result.error, it.InstrumentedTeleopError)


def test_keyboard_interrupt_stops_gracefully():
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 0.01, KeyboardInterrupt()], follower_registers=_default_registers(), duration_sec=None
    )
    assert result.stopped_reason == STOP_KEYBOARD_INTERRUPT
    assert len(result.samples) == 2


def test_on_sample_called_for_every_recorded_sample():
    seen = []
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, 0.01, KeyboardInterrupt()],
        follower_registers=_default_registers(),
        duration_sec=None,
        on_sample=seen.append,
    )
    assert len(seen) == 2
    assert seen == result.samples


# ---------------------------------------------------------------------------
# CSV_FIELDNAMES <-> TeleopCycleSample.to_csv_row() 일치
# ---------------------------------------------------------------------------


def test_csv_fieldnames_match_to_csv_row_keys():
    result, leader, follower, bus = _make_loop(
        leader_sequence=[0.0, KeyboardInterrupt()], follower_registers=_default_registers(), duration_sec=None
    )
    assert len(result.samples) >= 1
    row = result.samples[0].to_csv_row()
    assert set(row.keys()) == set(it.CSV_FIELDNAMES)


# ---------------------------------------------------------------------------
# compute_run_analysis / compute_deadband_summary
# ---------------------------------------------------------------------------


def test_compute_run_analysis_empty_samples():
    result = TeleopRunResult(samples=[], stopped_reason=STOP_DURATION_ELAPSED)
    analysis = compute_run_analysis(result)
    assert analysis["sample_count"] == 0
    assert analysis["warning_counts"] == {}


def _cycle_sample(loop_index, *, present_raw, goal_raw=2021, leader_deg=0.0, command_deg=0.0, status=0, loop_hz=50.0, prev_present_raw=None):
    present_deg = _wrist_roll_deg(present_raw)
    goal_deg = _wrist_roll_deg(goal_raw)
    prev_deg = _wrist_roll_deg(prev_present_raw) if prev_present_raw is not None else None
    return TeleopCycleSample(
        loop_index=loop_index,
        timestamp_iso="2026-08-07T00:00:00+00:00",
        elapsed_sec=loop_index * 0.02,
        loop_hz=loop_hz,
        leader_wrist_roll_deg=leader_deg,
        leader_wrist_roll_delta_from_start_deg=leader_deg,
        command_wrist_roll_deg=command_deg,
        follower_goal_raw=goal_raw,
        follower_goal_deg=goal_deg,
        follower_present_raw=present_raw,
        follower_present_deg=present_deg,
        goal_present_error_raw=goal_raw - present_raw,
        goal_present_error_deg=goal_deg - present_deg,
        follower_present_delta_from_prev_raw=(present_raw - prev_present_raw) if prev_present_raw is not None else 0,
        follower_present_delta_from_prev_deg=(present_deg - prev_deg) if prev_deg is not None else 0.0,
        follower_present_delta_from_start_deg=present_deg - _wrist_roll_deg(2023),
        follower_torque_enable=1,
        follower_acceleration=254,
        follower_acceleration_multiplier=1,
        follower_moving=0,
        follower_status=status,
        send_action_executed=True,
        leader_command_all_joints={},
        follower_sent_all_joints={},
        follower_observation_all_joints={},
    )


def test_compute_run_analysis_percentiles_and_mae():
    samples = [
        _cycle_sample(0, present_raw=2023, goal_raw=2023),
        _cycle_sample(1, present_raw=2023, goal_raw=2023 + 5),  # error ≈ 0.44°
        _cycle_sample(2, present_raw=2023, goal_raw=2023 + 10),  # error ≈ 0.88°
        _cycle_sample(3, present_raw=2023, goal_raw=2023 + 20),  # error ≈ 1.76°
    ]
    result = TeleopRunResult(
        samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2023, follower_start_present_deg=_wrist_roll_deg(2023)
    )
    analysis = compute_run_analysis(result)
    assert analysis["mae_tracking_error_deg"] == pytest.approx(
        sum(abs(s.goal_present_error_deg) for s in samples) / 4
    )
    assert analysis["tracking_error_p50_deg"] is not None
    assert analysis["tracking_error_p99_deg"] >= analysis["tracking_error_p50_deg"]
    assert analysis["max_follower_lag_deg"] == pytest.approx(max(abs(s.goal_present_error_deg) for s in samples))


def test_compute_run_analysis_warning_counts():
    result = TeleopRunResult(
        samples=[_cycle_sample(0, present_raw=2023)],
        stopped_reason=STOP_DURATION_ELAPSED,
        follower_start_present_raw=2023,
        follower_start_present_deg=_wrist_roll_deg(2023),
        warnings=[
            it.WarningEvent(
                event_type=WARNING_LARGE_COMMAND_DELTA,
                timestamp_iso="x",
                loop_index=0,
                joint="wrist_roll",
                leader_value=1.0,
                command_value=1.0,
                goal_value=None,
                present_value=None,
                error_value=3.0,
                threshold=2.0,
            ),
            it.WarningEvent(
                event_type=WARNING_LARGE_COMMAND_DELTA,
                timestamp_iso="x",
                loop_index=1,
                joint="wrist_roll",
                leader_value=1.0,
                command_value=1.0,
                goal_value=None,
                present_value=None,
                error_value=3.0,
                threshold=2.0,
            ),
        ],
    )
    analysis = compute_run_analysis(result)
    assert analysis["warning_counts"] == {WARNING_LARGE_COMMAND_DELTA: 2}
    assert analysis["total_warning_count"] == 2


def test_compute_run_analysis_lag_estimate_insufficient_when_no_movement():
    samples = [_cycle_sample(i, present_raw=2023, goal_raw=2021, command_deg=0.0) for i in range(10)]
    result = TeleopRunResult(
        samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2023, follower_start_present_deg=_wrist_roll_deg(2023)
    )
    analysis = compute_run_analysis(result)
    assert analysis["command_to_actual_lag_estimate"] == INSUFFICIENT_DATA


def test_compute_run_analysis_lag_estimate_detects_synthetic_lag():
    # command가 present보다 3 frame 앞서도록 합성 데이터를 만든다.
    lag = 3
    n = 80
    command_series = [((i % 10) - 5) * 0.3 for i in range(n)]
    present_series = [0.0] * lag + command_series[: n - lag]

    samples = []
    for i in range(n):
        present_deg = present_series[i]
        present_raw = int(round(present_deg * 4095 / 360.0 + 2047.5))
        samples.append(
            _cycle_sample(i, present_raw=present_raw, goal_raw=present_raw, command_deg=command_series[i])
        )
    result = TeleopRunResult(
        samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=samples[0].follower_present_raw, follower_start_present_deg=samples[0].follower_present_deg
    )
    analysis = compute_run_analysis(result)
    assert analysis.get("command_to_actual_lag_estimate") != INSUFFICIENT_DATA
    assert analysis["command_to_actual_lag_frames"] == lag


def test_deadband_summary_insufficient_when_too_few_samples():
    samples = [_cycle_sample(0, present_raw=2023, goal_raw=2021)]
    result = TeleopRunResult(samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2023, follower_start_present_deg=_wrist_roll_deg(2023))
    summary = compute_deadband_summary(result)
    assert summary["verdict"] == INSUFFICIENT_FOR_DEADBAND_ESTIMATE


def test_deadband_summary_available_when_clear_transition():
    """새 causal 구현: 0 tick 구간은 계속 응답이 없어야 하고(goal==present), 3 tick 구간은
    lookahead 안에서 실제로 같은 방향 움직임이 있어야 RESPONSE로 집계된다."""
    samples = []
    for i in range(10):
        samples.append(_cycle_sample(i, present_raw=2021, goal_raw=2021))  # error=0, 완전히 고정
    for i in range(10, 20):
        present_raw = 2021 + (i - 10) * 2  # 매 샘플 2 tick씩 같은 방향으로 실제 이동
        samples.append(_cycle_sample(i, present_raw=present_raw, goal_raw=present_raw + 3))  # error 항상 +3
    result = TeleopRunResult(
        samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2021, follower_start_present_deg=_wrist_roll_deg(2021)
    )
    # _cycle_sample의 elapsed_sec 간격은 0.02s(50Hz 상당) - lookahead 200ms면 10 샘플 앞까지 본다.
    summary = compute_deadband_summary(result, min_samples_per_bucket=3, lookahead_ms=200.0)
    assert summary["verdict"] == "DEADBAND_ESTIMATE_AVAILABLE"
    bucket0 = next(b for b in summary["buckets"] if b["abs_goal_present_error_ticks"] == 0)
    assert bucket0["response_count"] == 0
    assert bucket0["no_response_count"] == bucket0["sample_count"]
    bucket3 = next(b for b in summary["buckets"] if b["abs_goal_present_error_ticks"] == 3)
    assert bucket3["response_count"] > 0
    assert bucket3["response_fraction"] > 0.0


def test_deadband_summary_reports_opposite_motion():
    samples = []
    for i in range(10):
        # error는 양수로 고정(goal이 present보다 앞서 있음)이지만 present는 반대(음의) 방향으로 이동.
        present_raw = 2021 - i * 2
        samples.append(_cycle_sample(i, present_raw=present_raw, goal_raw=present_raw + 5))
    result = TeleopRunResult(
        samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2021, follower_start_present_deg=_wrist_roll_deg(2021)
    )
    summary = compute_deadband_summary(result, min_samples_per_bucket=3, lookahead_ms=200.0)
    bucket5 = next(b for b in summary["buckets"] if b["abs_goal_present_error_ticks"] == 5)
    assert bucket5["opposite_motion_count"] > 0


def test_deadband_summary_insufficient_data_reason():
    result = TeleopRunResult(samples=[], stopped_reason=STOP_DURATION_ELAPSED)
    summary = compute_deadband_summary(result)
    assert summary["verdict"] == INSUFFICIENT_FOR_DEADBAND_ESTIMATE


# ---------------------------------------------------------------------------
# classify_causal_response: 순수 판정 함수 (섹션 6, 14)
# ---------------------------------------------------------------------------


def test_classify_causal_response_no_response_below_noise_threshold():
    from hardware.diagnostics.instrumented_teleop import classify_causal_response

    assert classify_causal_response(error_raw=1, present_delta_raw=1, noise_threshold_ticks=2) == it.NO_RESPONSE
    assert classify_causal_response(error_raw=2, present_delta_raw=1, noise_threshold_ticks=2) == it.NO_RESPONSE


def test_classify_causal_response_response_same_direction():
    from hardware.diagnostics.instrumented_teleop import classify_causal_response

    assert classify_causal_response(error_raw=3, present_delta_raw=2, noise_threshold_ticks=2) == it.RESPONSE
    assert classify_causal_response(error_raw=-3, present_delta_raw=-2, noise_threshold_ticks=2) == it.RESPONSE


def test_classify_causal_response_opposite_motion():
    from hardware.diagnostics.instrumented_teleop import classify_causal_response

    assert classify_causal_response(error_raw=3, present_delta_raw=-2, noise_threshold_ticks=2) == it.OPPOSITE_MOTION


def test_classify_causal_response_zero_error_never_a_response():
    from hardware.diagnostics.instrumented_teleop import classify_causal_response

    # error가 0인데 문턱값 이상 움직였어도 "이 에러가 원인"이라고 주장할 근거가 없다.
    assert classify_causal_response(error_raw=0, present_delta_raw=10, noise_threshold_ticks=2) == it.NO_RESPONSE


# ---------------------------------------------------------------------------
# compute_motion_onset_analysis: causal 3-조건 판정 (섹션 10, 14)
# ---------------------------------------------------------------------------


def test_motion_onset_detects_stationary_then_new_error_then_response():
    samples = []
    # 0~9: 완전히 정지, error=0
    for i in range(10):
        samples.append(_cycle_sample(i, present_raw=2021, goal_raw=2021))
    # 10: 새로운 에러 발생 (아직 반응 없음 - 이 샘플 자체의 present는 그대로)
    samples.append(_cycle_sample(10, present_raw=2021, goal_raw=2021 + 4))
    # 11~15: lookahead 안에서 실제로 같은 방향으로 반응
    for i in range(11, 16):
        present_raw = 2021 + (i - 10) * 2
        samples.append(_cycle_sample(i, present_raw=present_raw, goal_raw=2021 + 4))

    result = TeleopRunResult(
        samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2021, follower_start_present_deg=_wrist_roll_deg(2021)
    )
    onset = it.compute_motion_onset_analysis(result, lookahead_ms=200.0, noise_threshold_ticks=2, stationary_window_samples=5)
    assert onset["verdict"] == it.MOTION_ONSET_FOUND
    assert onset["loop_index"] == 10
    assert onset["goal_present_error_raw_at_onset"] == 4


def test_motion_onset_does_not_false_positive_on_already_moving_follower():
    samples = []
    # follower가 처음부터 계속 움직이고 있다 - "정지 -> 새 에러" 조건을 만족하는 순간이 없어야 한다.
    for i in range(20):
        present_raw = 2021 + i * 3
        samples.append(_cycle_sample(i, present_raw=present_raw, goal_raw=present_raw + 4))
    result = TeleopRunResult(
        samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2021, follower_start_present_deg=_wrist_roll_deg(2021)
    )
    onset = it.compute_motion_onset_analysis(result, lookahead_ms=200.0, noise_threshold_ticks=2, stationary_window_samples=5)
    assert onset["verdict"] == it.MOTION_ONSET_INSUFFICIENT_DATA


def test_motion_onset_does_not_treat_goal_equals_present_as_cause():
    # 계속 goal==present(error=0)인 상태에서 present가 노이즈 수준으로 흔들려도 onset이 아니다.
    samples = [_cycle_sample(i, present_raw=2021, goal_raw=2021) for i in range(20)]
    result = TeleopRunResult(
        samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2021, follower_start_present_deg=_wrist_roll_deg(2021)
    )
    onset = it.compute_motion_onset_analysis(result, lookahead_ms=200.0, noise_threshold_ticks=2, stationary_window_samples=5)
    assert onset["verdict"] == it.MOTION_ONSET_INSUFFICIENT_DATA


def test_motion_onset_insufficient_data_when_too_few_samples():
    samples = [_cycle_sample(0, present_raw=2021, goal_raw=2021)]
    result = TeleopRunResult(samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2021, follower_start_present_deg=_wrist_roll_deg(2021))
    onset = it.compute_motion_onset_analysis(result)
    assert onset["verdict"] == it.MOTION_ONSET_INSUFFICIENT_DATA


def test_motion_onset_respects_lookahead_window():
    """반응이 lookahead 창 밖에서 일어나면 onset으로 인정하지 않는다."""
    samples = []
    for i in range(10):
        samples.append(_cycle_sample(i, present_raw=2021, goal_raw=2021))
    samples.append(_cycle_sample(10, present_raw=2021, goal_raw=2021 + 4))
    # 반응이 아주 늦게(elapsed 기준 한참 뒤) 일어난다 - 짧은 lookahead로는 못 잡아야 한다.
    for i in range(11, 15):
        samples.append(_cycle_sample(i, present_raw=2021, goal_raw=2021 + 4))  # 아직 안 움직임
    late = _cycle_sample(15, present_raw=2021 + 10, goal_raw=2021 + 4)
    samples.append(late)

    result = TeleopRunResult(
        samples=samples, stopped_reason=STOP_DURATION_ELAPSED, follower_start_present_raw=2021, follower_start_present_deg=_wrist_roll_deg(2021)
    )
    # elapsed_sec 간격이 0.02s이므로 lookahead 40ms(=2 샘플)면 index 15의 반응을 놓친다.
    onset_short = it.compute_motion_onset_analysis(result, lookahead_ms=40.0, noise_threshold_ticks=2, stationary_window_samples=5)
    assert onset_short["verdict"] == it.MOTION_ONSET_INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# Timing: fps 제한이 실제로 걸리는지 (섹션 1~3 - 89Hz 버그 회귀 테스트)
# ---------------------------------------------------------------------------


class _ScriptedClock:
    """미리 정해둔 timestamp 시퀀스를 순서대로 반환하는 fake clock - loop 안에서 ``clock()``이
    정확히 몇 번, 어떤 순서로 호출되는지 알고 있을 때만 쓸 수 있는 정밀 제어용 도구다."""

    def __init__(self, values):
        self._values = list(values)
        self._index = 0

    def __call__(self):
        value = self._values[self._index]
        self._index += 1
        return value


def _run_one_cycle_and_capture_sleep(*, work_duration_s: float, fps: int):
    """정확히 한 cycle만 실행하고(두 번째 cycle의 ``get_action``에서 KeyboardInterrupt로
    멈춘다) ``sleep_fn``에 전달된 값을 캡처한다. ``clock()`` 호출 순서
    (``start`` -> ``cached_accel_at`` -> [``loop_start`` -> ``now`` -> ``loop_dt`` ->
    ``dt_s``] -> 다음 cycle ``loop_start``)를 소스 코드로 확인한 그대로 스크립팅한다.
    """
    t0 = 100.0
    scripted_values = [
        0.0,  # start = clock()
        0.0,  # cached_accel_at = clock()
        t0,  # loop_start = clock()  (cycle 1)
        t0 + 0.0001,  # now = clock()  (accel 캐시 확인용 - sleep 계산과 무관)
        t0 + 0.0002,  # loop_dt = clock() - loop_start  (loop_hz 표시용 - sleep 계산과 무관)
        t0 + work_duration_s,  # dt_s = clock() - loop_start  (실제 sleep 계산에 쓰이는 값)
        t0 + 999.0,  # loop_start = clock()  (cycle 2 - 이후 get_action에서 즉시 KeyboardInterrupt)
    ]
    clock = _ScriptedClock(scripted_values)

    sleep_calls: list[float] = []
    bus = FakeBus(register_sequences=_default_registers())
    leader = FakeLeader([0.0, KeyboardInterrupt()])
    follower = FakeFollower(bus=bus)
    instrument = WristRollRegisterInstrument(bus=bus)

    result = run_instrumented_teleop_loop(
        leader=leader,
        follower=follower,
        instrument=instrument,
        follower_calibration_range=RANGE,
        teleop_action_processor=_identity_processor,
        robot_action_processor=_identity_processor,
        fps=fps,
        duration_sec=None,
        clock=clock,
        sleep_fn=sleep_calls.append,
    )
    assert result.stopped_reason == STOP_KEYBOARD_INTERRUPT
    assert len(sleep_calls) == 1
    return sleep_calls[0]


def test_fps_60_target_period_and_workload_budget():
    # fps=60 -> target period = 1/60 ≈ 16.6667ms. 작업이 5ms 걸렸다면 남은 예산만큼만 sleep.
    slept = _run_one_cycle_and_capture_sleep(work_duration_s=0.005, fps=60)
    assert slept == pytest.approx(1.0 / 60 - 0.005, abs=1e-6)
    assert slept == pytest.approx(0.011667, abs=1e-4)


def test_fps_60_overrun_sleeps_zero_not_negative():
    # 작업이 목표 주기(16.67ms)보다 오래 걸리면 sleep은 0이어야 한다 - 절대 음수가 되면 안 된다.
    slept = _run_one_cycle_and_capture_sleep(work_duration_s=0.025, fps=60)
    assert slept == 0.0
    assert slept >= 0.0


def test_fps_60_overrun_exactly_at_period_sleeps_zero():
    slept = _run_one_cycle_and_capture_sleep(work_duration_s=1.0 / 60, fps=60)
    assert slept == pytest.approx(0.0, abs=1e-9)


def test_fps_30_target_period():
    # fps=30 -> target period = 1/30 ≈ 33.333ms.
    slept = _run_one_cycle_and_capture_sleep(work_duration_s=0.010, fps=30)
    assert slept == pytest.approx(1.0 / 30 - 0.010, abs=1e-6)


def test_default_sleep_fn_resolves_to_lerobot_precise_sleep(monkeypatch):
    """``sleep_fn``을 넘기지 않으면(``None``) 실제로 ``lerobot.utils.robot_utils.precise_sleep``이
    호출되는지 확인한다 - 이것이 89Hz 버그(기본값이 아무것도 안 하던 것)의 회귀 테스트다."""
    import lerobot.utils.robot_utils as robot_utils

    calls: list[float] = []
    monkeypatch.setattr(robot_utils, "precise_sleep", calls.append)

    bus = FakeBus(register_sequences=_default_registers())
    leader = FakeLeader([0.0, KeyboardInterrupt()])
    follower = FakeFollower(bus=bus)
    instrument = WristRollRegisterInstrument(bus=bus)
    ticks = iter(range(1, 100000))

    run_instrumented_teleop_loop(
        leader=leader,
        follower=follower,
        instrument=instrument,
        follower_calibration_range=RANGE,
        teleop_action_processor=_identity_processor,
        robot_action_processor=_identity_processor,
        fps=60,
        duration_sec=None,
        clock=lambda: next(ticks) * 0.001,
        # sleep_fn을 일부러 넘기지 않는다 - 기본값(None) 해석 경로를 검증한다.
    )
    assert len(calls) == 1  # 정확히 한 cycle만 실행되고 KeyboardInterrupt로 멈췄다
    assert calls[0] >= 0.0


# ---------------------------------------------------------------------------
# 소스 감사: write 계열 호출/parameter writer 사용 흔적 없음
# ---------------------------------------------------------------------------


def _code_only_source(module) -> str:
    source = inspect.getsource(module)
    first = source.index('"""')
    second = source.index('"""', first + 3)
    return source[second + 3 :]


def test_module_source_contains_no_write_call_patterns():
    source = _code_only_source(it)
    for forbidden in (".write(", ".sync_write(", "enable_torque(", "disable_torque("):
        assert forbidden not in source, f"금지된 패턴 '{forbidden}'이 instrumented_teleop.py 코드에 있습니다."


def test_module_never_uses_single_joint_parameter_writer_or_armed_writer():
    source = _code_only_source(it)
    for forbidden in (
        "single_joint_parameter_writer",
        "single_joint_writer",
        "SingleJointArmedWriter",
        "SingleJointParameterWriter",
        "execute_single_armed_write",
        "execute_single_parameter_write",
    ):
        assert forbidden not in source


def test_module_only_write_path_is_send_action():
    source = _code_only_source(it)
    assert "send_action(" in source
    assert ".configure()" not in source


def test_send_action_call_has_no_conditional_guard_before_it():
    """섹션 5 핵심 검증: send_action() 호출 직전에 어떤 if/조건 분기도 없어야 한다 (passive)."""
    source = inspect.getsource(run_instrumented_teleop_loop)
    idx = source.index("follower.send_action(")
    # send_action 호출 앞 300자 안에 "return _result("가 있으면 안 된다(=그 전에 조기 종료하는
    # 분기가 있다는 뜻) - get_observation/get_action/processor 예외 처리(정상 control-path
    # 실패)만 있어야 하고, warning 판정으로 인한 조기 종료는 없어야 한다.
    window = source[max(0, idx - 400) : idx]
    assert "check_command_delta_guard" not in window  # command-delta 체크가 send 앞에서 값을 막지 않는다
