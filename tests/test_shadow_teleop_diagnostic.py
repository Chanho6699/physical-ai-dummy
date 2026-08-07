"""hardware/safety/shadow_teleop_diagnostic.py 단위 테스트.

실물 serial 포트에는 절대 접근하지 않는다 - ``LeaderWristRollReader``/
``FollowerWristRollStateReader``가 생성한 실제 ``FeetechMotorsBus`` 인스턴스를 가짜 버스로
바꿔치기해서(``tests/test_single_joint_hardware_inspector.py``와 동일한 패턴), read만
호출하고 write 계열 메서드는 절대 호출하지 않는다는 것과, 필수/선택 레지스터 read 실패를
안전하게 처리한다는 것을 검증한다.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")

from hardware.safety import shadow_teleop_diagnostic as std
from hardware.safety.shadow_teleop_diagnostic import (
    DEFAULT_FOLLOWER_MOVE_ABORT_THRESHOLD_DEG,
    FOLLOWER_ACCEL_REGISTERS,
    FOLLOWER_STATE_REGISTERS,
    FollowerMovedUnexpectedlyError,
    FollowerWristRollStateReader,
    LeaderWristRollReader,
    ShadowSample,
    ShadowTeleopReadError,
    ShadowTeleopSampler,
    WristRollCalibration,
    WriteGuardTriggeredError,
    build_command_delta_reference_table,
    compute_run_analysis,
    degrees_per_tick_for_calibration,
    run_sampling_loop,
)

LEADER_CALIBRATION = WristRollCalibration(motor_id=5, drive_mode=0, homing_offset=-1426, range_min=0, range_max=4095)
FOLLOWER_CALIBRATION = WristRollCalibration(motor_id=5, drive_mode=0, homing_offset=1627, range_min=0, range_max=4095)


class _ForbiddenWriteCalled(AssertionError):
    """가짜 버스의 쓰기 계열 메서드가 호출되면 즉시 테스트를 실패시킨다."""


class FakeLeaderBus:
    """리더용 가짜 FeetechMotorsBus - Present_Position만 안다."""

    def __init__(self, raw_sequence=None):
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []
        self.sync_read_calls: list[tuple] = []
        self.port = "/dev/null"
        self._raw_sequence = list(raw_sequence) if raw_sequence is not None else [2048]
        self._raw_index = 0

    @property
    def is_connected(self):
        return self.connected

    def connect(self, handshake: bool = True):
        self.connect_calls += 1
        self.connected = True

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        assert data_name == "Present_Position"
        self.sync_read_calls.append((data_name, tuple(motors or ()), normalize))
        raw = self._raw_sequence[min(self._raw_index, len(self._raw_sequence) - 1)]
        self._raw_index += 1
        return {"wrist_roll": raw}

    def disconnect(self, disable_torque: bool = True):
        self.disconnect_calls.append(disable_torque)
        self.connected = False

    def write(self, *a, **k):
        raise _ForbiddenWriteCalled("write() 호출됨")

    def sync_write(self, *a, **k):
        raise _ForbiddenWriteCalled("sync_write() 호출됨")

    def enable_torque(self, *a, **k):
        raise _ForbiddenWriteCalled("enable_torque() 호출됨")

    def disable_torque(self, *a, **k):
        raise _ForbiddenWriteCalled("disable_torque() 호출됨")


class FakeFollowerBus:
    """팔로워용 가짜 FeetechMotorsBus - register별 개별 read()를 흉내낸다."""

    def __init__(self, register_values=None, raise_on=None):
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []
        self.read_calls: list[tuple] = []
        self.port = "/dev/null"
        self._values = dict(
            register_values
            or {
                "Goal_Position": 2021,
                "Present_Position": 2023,
                "Torque_Enable": 1,
                "Moving": 0,
                "Status": 0,
                "Acceleration": 0,
                "Acceleration_Multiplier ": 1,
            }
        )
        self._raise_on = set(raise_on or ())

    @property
    def is_connected(self):
        return self.connected

    def connect(self, handshake: bool = True):
        self.connect_calls += 1
        self.connected = True

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        self.read_calls.append((data_name, motor, normalize))
        if data_name in self._raise_on:
            raise RuntimeError(f"comm error on {data_name}")
        return self._values[data_name]

    def disconnect(self, disable_torque: bool = True):
        self.disconnect_calls.append(disable_torque)
        self.connected = False

    def write(self, *a, **k):
        raise _ForbiddenWriteCalled("write() 호출됨")

    def sync_write(self, *a, **k):
        raise _ForbiddenWriteCalled("sync_write() 호출됨")

    def enable_torque(self, *a, **k):
        raise _ForbiddenWriteCalled("enable_torque() 호출됨")

    def disable_torque(self, *a, **k):
        raise _ForbiddenWriteCalled("disable_torque() 호출됨")


def _build_leader_with_fake_bus(raw_sequence=None) -> tuple[LeaderWristRollReader, FakeLeaderBus]:
    reader = LeaderWristRollReader(port="/dev/null", calibration=LEADER_CALIBRATION)
    fake_bus = FakeLeaderBus(raw_sequence=raw_sequence)
    reader._bus = fake_bus  # noqa: SLF001 - 의도적인 테스트용 내부 교체
    return reader, fake_bus


def _build_follower_with_fake_bus(register_values=None, raise_on=None) -> tuple[FollowerWristRollStateReader, FakeFollowerBus]:
    reader = FollowerWristRollStateReader(port="/dev/null", calibration=FOLLOWER_CALIBRATION)
    fake_bus = FakeFollowerBus(register_values=register_values, raise_on=raise_on)
    reader._bus = fake_bus  # noqa: SLF001
    return reader, fake_bus


# ---------------------------------------------------------------------------
# LeaderWristRollReader: read만, write 0회
# ---------------------------------------------------------------------------


def test_leader_bus_only_registers_wrist_roll_motor():
    reader = LeaderWristRollReader(port="/dev/null", calibration=LEADER_CALIBRATION)
    assert set(reader._bus.motors) == {"wrist_roll"}  # noqa: SLF001


def test_leader_connect_read_disconnect_zero_writes():
    reader, fake_bus = _build_leader_with_fake_bus(raw_sequence=[2048])
    assert not reader.is_connected
    reader.connect()
    assert fake_bus.connect_calls == 1
    assert reader.read_raw() == 2048
    reader.disconnect()
    assert fake_bus.disconnect_calls == [False]  # disable_torque=False 고정
    # 예외 없이 여기 도달하면 write 계열 메서드가 한 번도 호출되지 않았다는 뜻이다.


def test_leader_connect_is_idempotent():
    reader, fake_bus = _build_leader_with_fake_bus()
    reader.connect()
    reader.connect()
    assert fake_bus.connect_calls == 1


def test_leader_sync_read_only_requests_wrist_roll():
    reader, fake_bus = _build_leader_with_fake_bus()
    reader.connect()
    reader.read_raw()
    for _, motors, normalize in fake_bus.sync_read_calls:
        assert motors == ("wrist_roll",)
        assert normalize is False


def test_leader_exposes_only_expected_read_only_interface():
    public_attrs = {name for name in dir(LeaderWristRollReader) if not name.startswith("_")}
    assert public_attrs == {"is_connected", "connect", "read_raw", "disconnect"}


# ---------------------------------------------------------------------------
# FollowerWristRollStateReader: 필요한 레지스터만, write 0회, 개별 read 실패 처리
# ---------------------------------------------------------------------------


def test_follower_bus_only_registers_wrist_roll_motor():
    reader = FollowerWristRollStateReader(port="/dev/null", calibration=FOLLOWER_CALIBRATION)
    assert set(reader._bus.motors) == {"wrist_roll"}  # noqa: SLF001


def test_follower_read_state_reads_only_expected_registers():
    reader, fake_bus = _build_follower_with_fake_bus()
    reader.connect()
    state = reader.read_state()
    read_names = {name for name, _, _ in fake_bus.read_calls}
    assert read_names == set(FOLLOWER_STATE_REGISTERS)
    assert state.goal_raw == 2021
    assert state.present_raw == 2023
    assert state.torque_enable == 1
    assert state.moving == 0
    assert state.status_raw == 0
    assert state.read_errors == {}


def test_follower_read_accel_reads_only_accel_registers():
    reader, fake_bus = _build_follower_with_fake_bus()
    reader.connect()
    accel = reader.read_accel()
    read_names = {name for name, _, _ in fake_bus.read_calls}
    assert read_names == set(FOLLOWER_ACCEL_REGISTERS)
    assert accel.acceleration == 0
    assert accel.acceleration_multiplier == 1


def test_follower_missing_register_becomes_none_with_reason_not_a_crash():
    reader, _ = _build_follower_with_fake_bus(raise_on={"Status"})
    reader.connect()
    state = reader.read_state()
    assert state.status_raw is None
    assert "Status" in state.read_errors
    # 다른 레지스터는 정상적으로 계속 읽힌다.
    assert state.goal_raw == 2021
    assert state.present_raw == 2023


def test_follower_disconnect_never_disables_torque_via_write():
    reader, fake_bus = _build_follower_with_fake_bus()
    reader.connect()
    reader.disconnect()
    assert fake_bus.disconnect_calls == [False]


def test_follower_exposes_only_expected_read_only_interface():
    public_attrs = {name for name in dir(FollowerWristRollStateReader) if not name.startswith("_")}
    assert public_attrs == {"is_connected", "connect", "read_state", "read_accel", "disconnect"}


# ---------------------------------------------------------------------------
# write guard: bus에 설치된 defense-in-depth 가드
# ---------------------------------------------------------------------------


def test_write_guard_blocks_write_and_sync_write_and_torque_methods():
    class _Dummy:
        pass

    bus = _Dummy()
    std._install_write_guard(bus)  # noqa: SLF001 - 내부 함수를 직접 테스트

    with pytest.raises(WriteGuardTriggeredError):
        bus.write("Goal_Position", "wrist_roll", 0)
    with pytest.raises(WriteGuardTriggeredError):
        bus.sync_write("Goal_Position", {"wrist_roll": 0})
    with pytest.raises(WriteGuardTriggeredError):
        bus.enable_torque()
    with pytest.raises(WriteGuardTriggeredError):
        bus.disable_torque()


def test_leader_reader_bus_has_write_guard_installed():
    reader = LeaderWristRollReader(port="/dev/null", calibration=LEADER_CALIBRATION)
    with pytest.raises(WriteGuardTriggeredError):
        reader._bus.write("Goal_Position", "wrist_roll", 0)  # noqa: SLF001


def test_follower_reader_bus_has_write_guard_installed():
    reader = FollowerWristRollStateReader(port="/dev/null", calibration=FOLLOWER_CALIBRATION)
    with pytest.raises(WriteGuardTriggeredError):
        reader._bus.sync_write("Goal_Position", {"wrist_roll": 0})  # noqa: SLF001


# ---------------------------------------------------------------------------
# ShadowTeleopSampler: normalize/delta/오차 계산 + 필수 레지스터 read 실패 시 안전 종료
# ---------------------------------------------------------------------------


def _make_sampler(leader_reader, follower_reader, *, clock=None, accel_refresh_interval_s=1.0):
    ticks = iter(range(1_000_000))
    fixed_wall = __import__("datetime").datetime(2026, 8, 7, tzinfo=__import__("datetime").timezone.utc)
    return ShadowTeleopSampler(
        leader_reader=leader_reader,
        follower_reader=follower_reader,
        leader_calibration=LEADER_CALIBRATION,
        follower_calibration=FOLLOWER_CALIBRATION,
        accel_refresh_interval_s=accel_refresh_interval_s,
        clock=clock or (lambda: next(ticks) * 0.01),
        wall_clock=lambda: fixed_wall,
    )


def test_sampler_computes_normalized_degrees_independently_per_calibration():
    # 리더/팔로워 calibration의 homing_offset이 서로 다르지만(range는 둘 다 0~4095), raw=2048은
    # 두 calibration 모두 mid=(0+4095)/2=2047.5에 매우 가까워 두 팔 모두 거의 0도가 나온다 -
    # "raw를 그대로 빼지 않고 각자 calibration으로 변환한 뒤 비교한다"를 명시적으로 검증하기
    # 위해 서로 다른 raw를 넣어 서로 다른 결과가 나오는지 확인한다.
    leader, _ = _build_leader_with_fake_bus(raw_sequence=[2148])  # mid보다 100 tick 위
    follower, _ = _build_follower_with_fake_bus(register_values={"Goal_Position": 2021, "Present_Position": 1948})
    leader.connect()
    follower.connect()
    sampler = _make_sampler(leader, follower)

    sample = sampler.sample()

    expected_leader_deg = (2148 - 2047.5) * 360.0 / 4095
    expected_follower_present_deg = (1948 - 2047.5) * 360.0 / 4095
    assert sample.leader_wrist_roll_deg == pytest.approx(expected_leader_deg)
    assert sample.follower_present_deg == pytest.approx(expected_follower_present_deg)
    assert sample.leader_vs_follower_present_deg == pytest.approx(expected_leader_deg - expected_follower_present_deg)


def test_sampler_first_sample_has_zero_deltas():
    leader, _ = _build_leader_with_fake_bus(raw_sequence=[2048])
    follower, _ = _build_follower_with_fake_bus()
    leader.connect()
    follower.connect()
    sampler = _make_sampler(leader, follower)

    sample = sampler.sample()

    assert sample.leader_delta_from_start_deg == pytest.approx(0.0)
    assert sample.follower_present_delta_from_start_deg == pytest.approx(0.0)
    assert sample.sample_index == 0


def test_sampler_deltas_track_start_reference():
    leader, _ = _build_leader_with_fake_bus(raw_sequence=[2048, 2148])
    follower, _ = _build_follower_with_fake_bus()
    leader.connect()
    follower.connect()
    sampler = _make_sampler(leader, follower)

    first = sampler.sample()
    second = sampler.sample()

    assert first.leader_delta_from_start_deg == pytest.approx(0.0)
    expected_delta = (2148 - 2047.5) * 360.0 / 4095 - (2048 - 2047.5) * 360.0 / 4095
    assert second.leader_delta_from_start_deg == pytest.approx(expected_delta)
    assert second.sample_index == 1


def test_sampler_goal_present_error_raw_and_deg():
    leader, _ = _build_leader_with_fake_bus()
    follower, _ = _build_follower_with_fake_bus(register_values={"Goal_Position": 2021, "Present_Position": 2023})
    leader.connect()
    follower.connect()
    sampler = _make_sampler(leader, follower)

    sample = sampler.sample()

    assert sample.goal_present_error_raw == 2021 - 2023
    assert sample.goal_present_error_deg == pytest.approx(sample.follower_goal_deg - sample.follower_present_deg)


def test_sampler_elapsed_sec_is_monotonic_across_samples():
    leader, _ = _build_leader_with_fake_bus(raw_sequence=[2048, 2048, 2048])
    follower, _ = _build_follower_with_fake_bus()
    leader.connect()
    follower.connect()
    sampler = _make_sampler(leader, follower)

    elapsed = [sampler.sample().elapsed_sec for _ in range(3)]
    assert elapsed[0] == pytest.approx(0.0)
    assert elapsed[1] > elapsed[0]
    assert elapsed[2] > elapsed[1]


def test_sampler_raises_on_leader_read_failure():
    class _RaisingLeaderReader:
        def read_raw(self):
            raise RuntimeError("serial timeout")

    follower, _ = _build_follower_with_fake_bus()
    follower.connect()
    sampler = _make_sampler(_RaisingLeaderReader(), follower)

    with pytest.raises(ShadowTeleopReadError):
        sampler.sample()


def test_sampler_raises_when_follower_present_position_missing():
    leader, _ = _build_leader_with_fake_bus()
    leader.connect()
    follower, _ = _build_follower_with_fake_bus(raise_on={"Present_Position"})
    follower.connect()
    sampler = _make_sampler(leader, follower)

    with pytest.raises(ShadowTeleopReadError):
        sampler.sample()


def test_sampler_raises_when_follower_goal_position_missing():
    leader, _ = _build_leader_with_fake_bus()
    leader.connect()
    follower, _ = _build_follower_with_fake_bus(raise_on={"Goal_Position"})
    follower.connect()
    sampler = _make_sampler(leader, follower)

    with pytest.raises(ShadowTeleopReadError):
        sampler.sample()


def test_sampler_survives_missing_optional_status_and_moving():
    leader, _ = _build_leader_with_fake_bus()
    leader.connect()
    follower, _ = _build_follower_with_fake_bus(raise_on={"Status", "Moving", "Torque_Enable"})
    follower.connect()
    sampler = _make_sampler(leader, follower)

    sample = sampler.sample()

    assert sample.follower_status is None
    assert sample.follower_moving is None
    assert sample.follower_torque_enable is None
    assert sample.follower_present_raw == 2023  # 필수 레지스터는 정상 처리됨


def test_sampler_survives_missing_accel_registers():
    leader, _ = _build_leader_with_fake_bus()
    leader.connect()
    follower, _ = _build_follower_with_fake_bus(raise_on={"Acceleration", "Acceleration_Multiplier "})
    follower.connect()
    sampler = _make_sampler(leader, follower)

    sample = sampler.sample()

    assert sample.follower_acceleration is None
    assert sample.follower_acceleration_multiplier is None


def test_sampler_caches_accel_and_does_not_reread_within_interval():
    leader, _ = _build_leader_with_fake_bus(raw_sequence=[2048] * 5)
    follower, fake_follower_bus = _build_follower_with_fake_bus()
    leader.connect()
    follower.connect()

    clock_values = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    sampler = _make_sampler(leader, follower, clock=lambda: next(clock_values), accel_refresh_interval_s=1.0)

    for _ in range(5):
        sampler.sample()

    accel_read_count = sum(1 for name, _, _ in fake_follower_bus.read_calls if name in FOLLOWER_ACCEL_REGISTERS)
    # 5개 샘플 모두 accel_refresh_interval_s(1.0초) 안에 들어오므로 accel 레지스터는 최초 1회씩만 읽혀야 한다.
    assert accel_read_count == len(FOLLOWER_ACCEL_REGISTERS)


def test_sampler_refreshes_accel_after_interval_elapses():
    leader, _ = _build_leader_with_fake_bus(raw_sequence=[2048] * 2)
    follower, fake_follower_bus = _build_follower_with_fake_bus()
    leader.connect()
    follower.connect()

    clock_values = iter([0.0, 2.0])  # 두 번째 샘플은 refresh interval(1.0s)을 넘겼다
    sampler = _make_sampler(leader, follower, clock=lambda: next(clock_values), accel_refresh_interval_s=1.0)

    sampler.sample()
    sampler.sample()

    accel_read_count = sum(1 for name, _, _ in fake_follower_bus.read_calls if name in FOLLOWER_ACCEL_REGISTERS)
    assert accel_read_count == len(FOLLOWER_ACCEL_REGISTERS) * 2


# ---------------------------------------------------------------------------
# run_sampling_loop: duration/Ctrl+C/read 실패/follower 이동 - 네 가지 안전 종료 경로
# ---------------------------------------------------------------------------


def _fake_sample(index: int, *, follower_delta_deg: float = 0.0) -> ShadowSample:
    return ShadowSample(
        sample_index=index,
        timestamp_iso="2026-08-07T00:00:00+00:00",
        elapsed_sec=float(index) * 0.05,
        leader_wrist_roll_raw=2048,
        leader_wrist_roll_deg=0.0,
        leader_delta_from_start_deg=0.0,
        follower_goal_raw=2021,
        follower_goal_deg=-0.02,
        follower_present_raw=2023,
        follower_present_deg=-0.01,
        follower_present_delta_from_start_deg=follower_delta_deg,
        goal_present_error_raw=-2,
        goal_present_error_deg=-0.01,
        leader_vs_follower_present_deg=0.01,
        follower_acceleration=0,
        follower_acceleration_multiplier=1,
        follower_torque_enable=1,
        follower_moving=0,
        follower_status=0,
    )


class _CannedSampler:
    """``ShadowTeleopSampler``를 대체하는 고정 시퀀스 fake - run_sampling_loop 전용 테스트."""

    def __init__(self, samples_or_exceptions):
        self._items = list(samples_or_exceptions)
        self._index = 0

    def sample(self):
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, BaseException):
            raise item
        return item


def test_run_sampling_loop_stops_on_duration_elapsed():
    sampler = _CannedSampler([_fake_sample(0), _fake_sample(1), _fake_sample(2)])
    clock_values = iter([0.0, 0.0, 0.1, 0.2, 5.0])  # start, 3번 체크 후 duration 초과
    result = run_sampling_loop(sampler, duration_sec=1.0, clock=lambda: next(clock_values))
    assert result.stopped_reason == "duration_elapsed"
    assert len(result.samples) <= 3


def test_run_sampling_loop_stops_on_keyboard_interrupt():
    class _InterruptingSampler:
        def __init__(self):
            self.calls = 0

        def sample(self):
            self.calls += 1
            if self.calls > 2:
                raise KeyboardInterrupt
            return _fake_sample(self.calls - 1)

    result = run_sampling_loop(_InterruptingSampler(), duration_sec=None)
    assert result.stopped_reason == "keyboard_interrupt"
    assert len(result.samples) == 2


def test_run_sampling_loop_stops_on_read_error_without_propagating():
    sampler = _CannedSampler([_fake_sample(0), ShadowTeleopReadError("comm lost")])
    result = run_sampling_loop(sampler, duration_sec=None)
    assert result.stopped_reason == "read_error"
    assert len(result.samples) == 1
    assert isinstance(result.error, ShadowTeleopReadError)


def test_run_sampling_loop_stops_on_follower_moved_unexpectedly():
    sampler = _CannedSampler([_fake_sample(0, follower_delta_deg=0.0), _fake_sample(1, follower_delta_deg=1.5)])
    result = run_sampling_loop(
        sampler, duration_sec=None, follower_move_abort_threshold_deg=DEFAULT_FOLLOWER_MOVE_ABORT_THRESHOLD_DEG
    )
    assert result.stopped_reason == "follower_moved_unexpectedly"
    assert isinstance(result.error, FollowerMovedUnexpectedlyError)
    assert len(result.samples) == 2  # 문제가 된 샘플도 기록된 뒤 멈춘다


def test_run_sampling_loop_does_not_abort_when_threshold_is_none():
    sampler = _CannedSampler([_fake_sample(0, follower_delta_deg=5.0), KeyboardInterrupt()])
    result = run_sampling_loop(sampler, duration_sec=None, follower_move_abort_threshold_deg=None)
    # threshold를 안 넘겼다는 얘기가 아니라, threshold 자체를 끈 경우(None) 안전장치가 비활성임을 확인.
    assert result.stopped_reason == "keyboard_interrupt"
    assert len(result.samples) == 1


def test_run_sampling_loop_calls_on_sample_for_every_collected_sample():
    sampler = _CannedSampler([_fake_sample(0), _fake_sample(1), KeyboardInterrupt()])
    seen = []
    result = run_sampling_loop(sampler, duration_sec=None, on_sample=seen.append)
    assert len(seen) == 2
    assert seen == result.samples


# ---------------------------------------------------------------------------
# compute_run_analysis: 섹션 11 요약 통계 (순수 계산)
# ---------------------------------------------------------------------------


def test_compute_run_analysis_empty_samples():
    assert compute_run_analysis([]) == {"sample_count": 0}


def test_compute_run_analysis_reports_leader_moved_while_follower_fixed():
    samples = [
        ShadowSample(
            sample_index=i,
            timestamp_iso="2026-08-07T00:00:00+00:00",
            elapsed_sec=i * 0.05,
            leader_wrist_roll_raw=2048 + i,
            leader_wrist_roll_deg=i * 0.5,  # 리더는 뚜렷하게 움직인다
            leader_delta_from_start_deg=i * 0.5,
            follower_goal_raw=2021,
            follower_goal_deg=-0.02,
            follower_present_raw=2023,
            follower_present_deg=-0.01,  # 팔로워는 완전히 고정
            follower_present_delta_from_start_deg=0.0,
            goal_present_error_raw=-2,
            goal_present_error_deg=-0.01,
            leader_vs_follower_present_deg=i * 0.5,
            follower_acceleration=0,
            follower_acceleration_multiplier=1,
            follower_torque_enable=1,
            follower_moving=0,
            follower_status=0,
        )
        for i in range(5)
    ]
    analysis = compute_run_analysis(samples)
    assert analysis["sample_count"] == 5
    assert analysis["leader_moved_while_follower_fixed"] is True
    assert analysis["follower_present_deg_range"] == pytest.approx(0.0)
    assert analysis["moving_ever_nonzero"] is False
    assert analysis["status_ever_nonzero"] is False
    assert analysis["torque_enable_changed"] is False
    assert analysis["acceleration_changed"] is False
    assert analysis["write_count"] == 0


def test_compute_run_analysis_detects_status_and_moving_and_torque_changes():
    def _sample(i, *, status, moving, torque):
        return ShadowSample(
            sample_index=i,
            timestamp_iso="2026-08-07T00:00:00+00:00",
            elapsed_sec=i * 0.05,
            leader_wrist_roll_raw=2048,
            leader_wrist_roll_deg=0.0,
            leader_delta_from_start_deg=0.0,
            follower_goal_raw=2021,
            follower_goal_deg=-0.02,
            follower_present_raw=2023,
            follower_present_deg=-0.01,
            follower_present_delta_from_start_deg=0.0,
            goal_present_error_raw=-2,
            goal_present_error_deg=-0.01,
            leader_vs_follower_present_deg=0.01,
            follower_acceleration=0,
            follower_acceleration_multiplier=1,
            follower_torque_enable=torque,
            follower_moving=moving,
            follower_status=status,
        )

    samples = [
        _sample(0, status=0, moving=0, torque=1),
        _sample(1, status=4, moving=1, torque=0),
    ]
    analysis = compute_run_analysis(samples)
    assert analysis["status_ever_nonzero"] is True
    assert analysis["moving_ever_nonzero"] is True
    assert analysis["torque_enable_changed"] is True


# ---------------------------------------------------------------------------
# 섹션 12: tick<->degree 참고표 (계산만, write 없음)
# ---------------------------------------------------------------------------


def test_degrees_per_tick_matches_known_sts3215_scale():
    deg_per_tick = degrees_per_tick_for_calibration(FOLLOWER_CALIBRATION)
    assert deg_per_tick == pytest.approx(360.0 / 4095)


def test_build_command_delta_reference_table_values():
    table = build_command_delta_reference_table(
        leader_calibration=LEADER_CALIBRATION, follower_calibration=FOLLOWER_CALIBRATION
    )
    assert table["leader_and_follower_scale_equal"] is True
    follower = table["follower"]
    deg_per_tick = 360.0 / 4095
    assert follower["degrees_to_ticks"]["0.1_deg_in_ticks"] == pytest.approx(0.1 / deg_per_tick)
    assert follower["degrees_to_ticks"]["1_deg_in_ticks"] == pytest.approx(1.0 / deg_per_tick)
    assert follower["ticks_to_degrees"]["1_tick_in_deg"] == pytest.approx(deg_per_tick)
    assert follower["ticks_to_degrees"]["5_tick_in_deg"] == pytest.approx(5 * deg_per_tick)


def test_build_command_delta_reference_table_never_touches_hardware():
    # calibration만으로 순수 계산되는지 소스로도 확인 - bus/port 인자를 받지 않는다.
    sig = inspect.signature(build_command_delta_reference_table)
    assert "port" not in sig.parameters
    assert "bus" not in sig.parameters


# ---------------------------------------------------------------------------
# 소스 감사: write 계열 호출/register write 흔적이 이 모듈 어디에도 없음
# ---------------------------------------------------------------------------


def test_module_source_contains_no_write_call_patterns():
    source = inspect.getsource(std)
    for forbidden in (".write(", ".sync_write(", "enable_torque(", "disable_torque("):
        assert forbidden not in source, f"금지된 패턴 '{forbidden}'이 shadow_teleop_diagnostic.py에 있습니다."


def test_module_never_imports_so_follower_or_so_leader_classes():
    source = inspect.getsource(std)
    assert "SOFollower" not in source
    assert "SOLeader" not in source
    assert ".configure(" not in source
    assert ".calibrate(" not in source


def test_csv_fieldnames_match_shadow_sample_to_csv_row_keys():
    sample = _fake_sample(0)
    assert set(sample.to_csv_row().keys()) == set(std.CSV_FIELDNAMES)
