"""runtime/laptop/realtime_control_loop.py의 RealTimeFollowerControlLoop 검증
(Phase C-3B, 섹션 11~16 요구사항 전부).

두 그룹으로 나뉜다:

    - **결정적(deterministic) 그룹**: ``loop._do_tick(scheduled_time=...)``을 테스트
      스레드에서 직접, controllable ``FakeClock``으로 호출한다(실제 background thread
      없음) - state machine/hold/quarantine/writer invariant/failure injection처럼
      "정확히 무슨 일이 일어났는지"를 경계값까지 검증해야 하는 로직용.
    - **실 스레드(real-thread) 그룹**: ``start()``/``stop()``으로 진짜 background
      thread를 띄워 최소 수 초 실행한다(섹션 11/12) - deadline scheduler 자체의 실제
      cadence/jitter, 그리고 "control loop이 inference latency에 안 막힌다"는 사실은
      진짜 시간 흐름 없이는 증명할 수 없기 때문이다.

실제 하드웨어 접근 없음 - connect/send_action을 실제 포트로 호출하지 않는다(writer는
전부 Fake, state_source는 전부 Fake).
"""

from __future__ import annotations

import math
import threading
import time

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.fake_follower_state_source import FakeFollowerStateSource
from runtime.laptop.follower_action_writer import FakeFollowerWriter, RecordingFollowerWriter
from runtime.laptop.motion_guard import DEFAULT_JOINT_MOTION_LIMITS, JointMotionLimits
from runtime.laptop.realtime_control_loop import (
    ControlLoopState,
    HoldPolicy,
    RealTimeFollowerControlLoop,
    RealTimeFollowerControlLoopConfig,
)
from runtime.laptop.realtime_control_target import RealTimeControlTargetGenerator
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
from runtime.laptop.temporal_ensemble import TemporalEnsembler
from runtime.laptop.trajectory_buffer import TrajectoryBuffer
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

SPACING = 1.0 / 30.0
CHUNK_SIZE = 50
DT_60HZ = 1.0 / 60.0


def _neutral(v: float = 0.0) -> dict[str, float]:
    return {j: v for j in JOINT_ORDER}


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        self._t += dt
        return self._t

    def __call__(self) -> float:
        return self._t


def _chunk(*, sequence: int, obs_time: float, action: dict[str, float], chunk_size: int = CHUNK_SIZE, spacing: float = SPACING) -> TimestampedActionChunk:
    actions = tuple(dict(action) for _ in range(chunk_size))
    return TimestampedActionChunk(
        sequence=sequence, session_id="s1", observation_time_monotonic=obs_time,
        request_started_time_monotonic=obs_time, response_received_time_monotonic=obs_time + 0.05,
        server_received_at=None, server_responded_at=None, inference_latency_ms=50.0,
        chunk_index_spacing_s=spacing, chunk_size=chunk_size, actions=actions,
        model_id="fake", backend="fake",
    )


def _generous_safety_gate() -> SafetyGate:
    joint_range = {j: (-1000.0, 1000.0) for j in JOINT_ORDER}
    max_step = {j: 1000.0 for j in JOINT_ORDER}
    return SafetyGate(SafetyGateConfig(joint_range_deg=joint_range, max_step_deg=max_step))


def _generous_motion_limits() -> dict[str, JointMotionLimits]:
    return {j: JointMotionLimits(velocity_limit=1000.0, acceleration_limit=1e7, jerk_limit=1e10) for j in JOINT_ORDER}


def _make_loop(
    *, clock: FakeClock, safety_gate=None, motion_limits=None, control_hz: float = 60.0,
    hold_policy: HoldPolicy = HoldPolicy.NO_WRITE, hold_timeout_s: float = 0.5,
    trajectory_buffer: TrajectoryBuffer | None = None, state_source=None, writer=None,
    health_source=None,
):
    gate = safety_gate or _generous_safety_gate()
    gen = RealTimeControlTargetGenerator(
        ensembler=TemporalEnsembler(half_life_s=0.338), safety_gate=gate,
        motion_limits=motion_limits or _generous_motion_limits(), control_hz=control_hz,
    )
    buf = trajectory_buffer if trajectory_buffer is not None else TrajectoryBuffer(max_chunks=4)
    state_src = state_source or FakeFollowerStateSource(initial_state_deg=_neutral(0.0), monotonic_fn=clock)
    w = writer if writer is not None else RecordingFollowerWriter(monotonic_fn=clock)
    config = RealTimeFollowerControlLoopConfig(control_hz=control_hz, hold_policy=hold_policy, hold_timeout_s=hold_timeout_s)
    loop = RealTimeFollowerControlLoop(
        generator=gen, safety_gate=gate, trajectory_buffer=buf, state_source=state_src, writer=w,
        config=config, health_source=health_source, monotonic_fn=clock,
    )
    return loop, buf, state_src, w, gate


# ===========================================================================
# 결정적(deterministic) 그룹 - _do_tick()을 직접 호출
# ===========================================================================


# ---------------------------------------------------------------------------
# 섹션 6: fail-safe state machine
# ---------------------------------------------------------------------------


def test_no_trajectory_state_and_no_write_at_startup() -> None:
    clock = FakeClock()
    loop, buf, state_src, writer, gate = _make_loop(clock=clock)
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.NO_TRAJECTORY
    assert record.write_attempted is False
    assert writer.write_count == 0


def test_running_state_and_write_when_trajectory_valid() -> None:
    clock = FakeClock()
    loop, buf, state_src, writer, gate = _make_loop(clock=clock)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.RUNNING
    assert record.target_valid is True
    assert record.write_executed is True
    assert writer.write_count == 1


def test_stale_trajectory_after_chunk_horizon_elapses() -> None:
    clock = FakeClock()
    loop, buf, state_src, writer, gate = _make_loop(clock=clock)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0), chunk_size=5))  # 짧은 horizon
    clock.advance(100.0)  # horizon을 훨씬 지남
    record = loop._do_tick(scheduled_time=None)
    assert record.state in (ControlLoopState.NO_TRAJECTORY, ControlLoopState.STALE_TRAJECTORY)
    assert record.write_attempted is False


def test_intent_blocked_state_no_write() -> None:
    clock = FakeClock()
    gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, safety_gate=gate, motion_limits=DEFAULT_JOINT_MOTION_LIMITS)
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703
    state_src.set_state(current)
    target = dict(current)
    target["wrist_flex"] = 40.4879  # delta=13.18, 위험한 실물 사례
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=target))
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.INTENT_BLOCKED
    assert record.intent_decision == "WOULD_CLAMP"
    assert record.write_attempted is False
    assert writer.write_count == 0


def test_safety_blocked_state_when_mechanical_range_violated() -> None:
    clock = FakeClock()
    tight = SafetyGate(SafetyGateConfig(
        joint_range_deg={j: (-1.0, 1.0) for j in JOINT_ORDER}, max_step_deg={j: 1000.0 for j in JOINT_ORDER},
    ))
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, safety_gate=tight)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(50.0)))
    record = loop._do_tick(scheduled_time=None)
    # 이 설정에서는 raw target 자체가 range 밖이라 Intent 단계에서 막힌다 -
    # INTENT_BLOCKED든 SAFETY_BLOCKED든(Final 단계까지 도달한 경우) 둘 다 "차단된 상태"이고
    # write는 절대 없어야 한다는 게 핵심 불변량이다.
    assert record.state in (ControlLoopState.INTENT_BLOCKED, ControlLoopState.SAFETY_BLOCKED)
    assert record.write_attempted is False
    assert writer.write_count == 0


def test_nan_current_state_fails_closed() -> None:
    """NaN current_state는 generator 내부에서 예외가 아니라(MotionGuard가
    ``GUARD_INVALID_INPUT``으로 정상 fail-closed 처리, ``realtime_control_target.py``
    참고) 정상적인 ``ControlTargetResult``로 돌아온다 - 그래도 write는 절대 없어야
    한다는 게 핵심이다(FAULT이든 SAFETY_BLOCKED든 fail-closed라면 둘 다 통과)."""
    clock = FakeClock()
    loop, buf, state_src, writer, _ = _make_loop(clock=clock)
    bad_state = _neutral(0.0)
    bad_state["wrist_roll"] = math.nan
    state_src.set_state(bad_state)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    record = loop._do_tick(scheduled_time=None)
    assert record.state in (ControlLoopState.FAULT, ControlLoopState.SAFETY_BLOCKED)
    assert record.write_attempted is False
    assert writer.write_count == 0


def test_state_read_exception_is_fault_and_does_not_crash_loop() -> None:
    clock = FakeClock()
    loop, buf, state_src, writer, _ = _make_loop(clock=clock)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    state_src.fail_next_read()
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.FAULT
    assert writer.write_count == 0
    assert any("state_read" in e for e in record.errors)
    # 다음 tick은 정상 회복 - thread가 죽지 않는다는 것을 tick 재호출 성공으로 증명.
    record2 = loop._do_tick(scheduled_time=None)
    assert record2.state == ControlLoopState.RUNNING


def test_writer_exception_is_fault_and_does_not_crash_loop() -> None:
    clock = FakeClock()

    class FlakyOnceWriter:
        """딱 첫 호출만 예외를 던지고, 그 이후는 정상 - "일시적 통신 오류에서 재시도로
        회복"하는 시나리오를 재현한다(항상 실패하는 writer는 이후 tick도 항상 FAULT가
        되는 게 당연하므로 회복 시나리오를 검증하려면 이렇게 해야 한다)."""

        def __init__(self) -> None:
            self._called = 0

        def write(self, action_deg):
            self._called += 1
            if self._called == 1:
                raise RuntimeError("writer 통신 오류 (시뮬레이션, 1회성)")
            return FakeFollowerWriter().write(action_deg)

    loop, buf, state_src, _, _ = _make_loop(clock=clock, writer=FlakyOnceWriter())
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.FAULT
    assert record.stop_reason == "WRITER_FAULT"
    assert record.write_attempted is True
    assert record.write_executed is False
    # 다음 tick은 writer가 회복돼 정상 재시도된다(thread가 안 죽음) - dt>0이 되도록
    # clock을 한 tick 전진.
    clock.advance(DT_60HZ)
    record2 = loop._do_tick(scheduled_time=None)
    assert record2.state == ControlLoopState.RUNNING


def test_generator_exception_is_fault_and_does_not_crash_loop() -> None:
    clock = FakeClock()

    class ExplodingGenerator:
        def tick(self, **kwargs):
            raise RuntimeError("generator 내부 오류 (시뮬레이션)")

    loop, buf, state_src, writer, _ = _make_loop(clock=clock)
    loop._generator = ExplodingGenerator()
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.FAULT
    assert writer.write_count == 0


def test_malformed_nan_chunk_fails_closed() -> None:
    """buffer에 NaN이 섞인 chunk가 올라와도(방어적 상황 - 정상 경로에서는
    AsyncVLAChunkInferenceWorker/TrajectoryBuffer.publish()가 이미 거르지만)
    control loop 레벨에서도 fail-closed여야 한다(defense-in-depth)."""
    clock = FakeClock()
    loop, buf, state_src, writer, _ = _make_loop(clock=clock)
    nan_action = {**_neutral(0.0), "elbow_flex": math.nan}
    chunk = _chunk(sequence=0, obs_time=clock.now(), action=nan_action)
    ok, reason = chunk.validate()
    assert ok is False  # TimestampedActionChunk 자체가 이미 NaN을 거부함(확인)
    # publish()도 이 chunk를 거부한다 - buffer에 애초에 안 들어감.
    result = buf.publish(chunk)
    assert result.accepted is False
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.NO_TRAJECTORY  # buffer가 비어있으므로
    assert writer.write_count == 0


# ---------------------------------------------------------------------------
# 섹션 8: Intent quarantine + recovery
# ---------------------------------------------------------------------------


def test_intent_blocked_chunk_is_quarantined_and_excluded_next_tick() -> None:
    clock = FakeClock()
    gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, safety_gate=gate, motion_limits=DEFAULT_JOINT_MOTION_LIMITS)
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703
    state_src.set_state(current)
    dangerous = dict(current)
    dangerous["wrist_flex"] = 40.4879
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=dangerous))

    r0 = loop._do_tick(scheduled_time=None)
    assert r0.state == ControlLoopState.INTENT_BLOCKED
    assert 0 in loop.quarantined_sequences

    clock.advance(DT_60HZ)
    r1 = loop._do_tick(scheduled_time=None)
    # 같은 sequence=0 chunk가 quarantine되어 buffer.valid_chunks()에서 필터링됨 ->
    # 사실상 "쓸 chunk가 없음" -> NO_TRAJECTORY(반복 BLOCK 로그가 아니라 명확한
    # "폐기됨" 상태로 바뀐다).
    assert r1.state == ControlLoopState.NO_TRAJECTORY
    assert 0 in r1.quarantined_sequences_excluded
    assert writer.write_count == 0


def test_newer_valid_chunk_recovers_after_quarantine() -> None:
    clock = FakeClock()
    gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, safety_gate=gate, motion_limits=DEFAULT_JOINT_MOTION_LIMITS)
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703
    state_src.set_state(current)
    dangerous = dict(current)
    dangerous["wrist_flex"] = 40.4879
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=dangerous))
    loop._do_tick(scheduled_time=None)
    assert 0 in loop.quarantined_sequences

    # 더 최신 observation 기반의 정상 chunk 도착(sequence=1, 더 큰 번호).
    clock.advance(0.05)
    safe = dict(current)
    safe["elbow_flex"] = 7.08
    buf.publish(_chunk(sequence=1, obs_time=clock.now(), action=safe))
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.RUNNING
    assert record.write_executed is True
    assert 1 not in loop.quarantined_sequences


# ---------------------------------------------------------------------------
# 섹션 7: Hold policy A(no write) / B(last commanded) / C(measured)
# ---------------------------------------------------------------------------


def test_hold_policy_no_write_default_never_writes_when_trajectory_gone() -> None:
    clock = FakeClock()
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, hold_policy=HoldPolicy.NO_WRITE)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    loop._do_tick(scheduled_time=None)  # RUNNING, 1 write
    assert writer.write_count == 1
    buf.clear()  # trajectory 사라짐
    clock.advance(DT_60HZ)
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.NO_TRAJECTORY
    assert record.write_attempted is False
    assert writer.write_count == 1  # 늘지 않음


def test_hold_policy_last_commanded_writes_static_value_within_timeout() -> None:
    clock = FakeClock()
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, hold_policy=HoldPolicy.HOLD_LAST_COMMANDED, hold_timeout_s=1.0)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    loop._do_tick(scheduled_time=None)
    last_commanded = dict(writer.calls[-1])
    buf.clear()
    clock.advance(DT_60HZ)
    record = loop._do_tick(scheduled_time=None)
    assert record.write_attempted is True
    assert record.write_path == "hold_last_commanded"
    assert writer.calls[-1] == last_commanded  # 마지막 명령값 그대로 재전송(추정/추론값 아님)


def test_hold_policy_measured_writes_current_state_within_timeout() -> None:
    clock = FakeClock()
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, hold_policy=HoldPolicy.HOLD_MEASURED, hold_timeout_s=1.0)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    loop._do_tick(scheduled_time=None)
    buf.clear()
    measured = {**_neutral(0.0), "shoulder_pan": 3.3}
    state_src.set_state(measured)
    clock.advance(DT_60HZ)
    record = loop._do_tick(scheduled_time=None)
    assert record.write_path == "hold_measured"
    assert writer.calls[-1]["shoulder_pan"] == pytest.approx(3.3)  # 실측값 그대로(외삽 없음)


def test_hold_policy_expires_after_hold_timeout() -> None:
    clock = FakeClock()
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, hold_policy=HoldPolicy.HOLD_LAST_COMMANDED, hold_timeout_s=0.2)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    loop._do_tick(scheduled_time=None)
    buf.clear()
    clock.advance(0.3)  # hold_timeout_s(0.2) 초과
    record = loop._do_tick(scheduled_time=None)
    assert record.write_attempted is False  # hold도 포기하고 순수 NO_WRITE로 degrade
    assert record.write_path is None


def test_hold_never_extrapolates_new_trajectory_only_static_values() -> None:
    """hold 경로는 오직 마지막 실제 commanded 값이나 이번 tick 실측값만 쓴다는 것을
    직접 증명 - 두 값이 서로 다를 때 각 정책이 정확히 그 값만 쓰는지 확인."""
    clock = FakeClock()
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, hold_policy=HoldPolicy.HOLD_LAST_COMMANDED, hold_timeout_s=1.0)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(2.0)))
    loop._do_tick(scheduled_time=None)
    last_commanded_pan = writer.calls[-1]["shoulder_pan"]
    buf.clear()
    state_src.set_state({**_neutral(0.0), "shoulder_pan": 99.0})  # 실측은 전혀 다른 값
    clock.advance(DT_60HZ)
    record = loop._do_tick(scheduled_time=None)
    # HOLD_LAST_COMMANDED이므로 실측(99.0)이 아니라 마지막 commanded 값이 나가야 한다.
    assert writer.calls[-1]["shoulder_pan"] == pytest.approx(last_commanded_pan)
    assert writer.calls[-1]["shoulder_pan"] != pytest.approx(99.0)


# ---------------------------------------------------------------------------
# 섹션 16: Writer invariant (가장 중요)
# ---------------------------------------------------------------------------


def test_writer_invariant_write_only_when_target_valid_and_accept() -> None:
    clock = FakeClock()
    gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, safety_gate=gate, motion_limits=DEFAULT_JOINT_MOTION_LIMITS)

    scenarios_written = []
    # 1) no trajectory
    r = loop._do_tick(scheduled_time=None)
    scenarios_written.append((r.target_valid, r.write_executed))
    # 2) dangerous intent block
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703
    state_src.set_state(current)
    dangerous = dict(current)
    dangerous["wrist_flex"] = 40.4879
    clock.advance(DT_60HZ)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=dangerous))
    r = loop._do_tick(scheduled_time=None)
    scenarios_written.append((r.target_valid, r.write_executed))
    # 3) normal accept
    clock.advance(DT_60HZ)
    safe = dict(current)
    safe["elbow_flex"] = 7.08
    buf.publish(_chunk(sequence=1, obs_time=clock.now(), action=safe))
    r = loop._do_tick(scheduled_time=None)
    scenarios_written.append((r.target_valid, r.write_executed))

    for target_valid, write_executed in scenarios_written:
        assert write_executed == target_valid, "writer invariant 위반: target_valid와 write_executed가 어긋남"
    assert writer.write_count == sum(1 for v, _ in scenarios_written if v)


def test_writer_invariant_holds_across_many_random_like_ticks() -> None:
    """여러 tick에 걸쳐 정상/위험 target을 번갈아 흘려보내며 invariant가 매번 성립하는지."""
    clock = FakeClock()
    gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, safety_gate=gate, motion_limits=DEFAULT_JOINT_MOTION_LIMITS)
    current = _neutral(0.0)
    state_src.set_state(current)

    expected_writes = 0
    for i in range(20):
        clock.advance(DT_60HZ)
        if i % 3 == 0:
            action = dict(current)
            action["wrist_flex"] = current.get("wrist_flex", 0.0) + 13.18  # 위험
        else:
            action = dict(current)
            action["elbow_flex"] = current.get("elbow_flex", 0.0) + 7.08  # 정상
        buf.publish(_chunk(sequence=i, obs_time=clock.now(), action=action))
        record = loop._do_tick(scheduled_time=None)
        assert record.write_executed == record.target_valid
        if record.write_executed:
            expected_writes += 1
            state_src.set_state(dict(writer.calls[-1]))

    assert writer.write_count == expected_writes


# ===========================================================================
# 실 스레드(real-thread) 그룹 - start()/stop()로 진짜 background thread
# ===========================================================================


def _publish_constant_chunk(buf: TrajectoryBuffer, *, sequence: int, action: dict[str, float]) -> None:
    now = time.monotonic()
    buf.publish(_chunk(sequence=sequence, obs_time=now, action=action, chunk_size=CHUNK_SIZE, spacing=SPACING))


def test_real_thread_scheduler_maintains_60hz_cadence() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    _publish_constant_chunk(buf, sequence=0, action=_neutral(0.0))
    gate = _generous_safety_gate()
    gen = RealTimeControlTargetGenerator(
        ensembler=TemporalEnsembler(half_life_s=0.338), safety_gate=gate, motion_limits=_generous_motion_limits(), control_hz=60.0,
    )
    state_src = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = RecordingFollowerWriter()
    loop = RealTimeFollowerControlLoop(
        generator=gen, safety_gate=gate, trajectory_buffer=buf, state_source=state_src, writer=writer,
        config=RealTimeFollowerControlLoopConfig(control_hz=60.0),
    )
    loop.start()
    time.sleep(1.5)
    loop.stop()

    stats = loop.stats()
    # 실 스레드/실 OS 스케줄링 jitter를 감안한 tolerance - 60Hz 목표 대비 ±15% 이내,
    # deadline miss는 드물어야 한다(과반 이상이면 스케줄러 버그).
    assert stats.actual_hz is not None
    assert 51.0 <= stats.actual_hz <= 69.0
    assert stats.deadline_miss_rate < 0.2
    assert stats.jitter_ms < 5.0


def test_real_thread_control_loop_not_blocked_by_slow_inference() -> None:
    """섹션 11: 3초간 (fake) inference ~3Hz + control loop ~60Hz를 독립된 thread로
    동시 실행 - control tick 수가 inference latency에 좌우되지 않는지 확인."""
    buf = TrajectoryBuffer(max_chunks=4)
    gate = _generous_safety_gate()
    gen = RealTimeControlTargetGenerator(
        ensembler=TemporalEnsembler(half_life_s=0.338), safety_gate=gate, motion_limits=_generous_motion_limits(), control_hz=60.0,
    )
    state_src = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = RecordingFollowerWriter()
    loop = RealTimeFollowerControlLoop(
        generator=gen, safety_gate=gate, trajectory_buffer=buf, state_source=state_src, writer=writer,
        config=RealTimeFollowerControlLoopConfig(control_hz=60.0),
    )

    stop_inference = threading.Event()
    inference_count = [0]

    def fake_inference_loop():
        seq = 0
        while not stop_inference.is_set():
            time.sleep(0.33)  # 실측 steady-state inference latency와 동일한 크기
            if stop_inference.is_set():
                break
            _publish_constant_chunk(buf, sequence=seq, action=_neutral(float(seq % 3)))
            inference_count[0] += 1
            seq += 1

    inference_thread = threading.Thread(target=fake_inference_loop, daemon=True)
    loop.start()
    inference_thread.start()
    time.sleep(3.0)
    stop_inference.set()
    inference_thread.join(timeout=2.0)
    loop.stop()

    stats = loop.stats()
    # 3초 * 60Hz = 180틱 근방(스레드 시작 지연/스케줄링 여유로 넉넉한 tolerance).
    assert 130 <= stats.n_ticks <= 210
    # 3초 / 0.33s ≈ 9회 근방.
    assert 6 <= inference_count[0] <= 12
    # control loop 자체 cadence는 inference 존재와 무관하게 유지돼야 한다.
    assert stats.actual_hz is not None and stats.actual_hz >= 45.0


@pytest.mark.parametrize("fake_latency_s", [0.2, 0.34, 0.4, 0.8])
def test_real_thread_control_hz_independent_of_inference_latency(fake_latency_s: float) -> None:
    """섹션 12: fake inference latency를 바꿔도 control scheduler 자체 cadence가 유지되는지."""
    buf = TrajectoryBuffer(max_chunks=4)
    gate = _generous_safety_gate()
    gen = RealTimeControlTargetGenerator(
        ensembler=TemporalEnsembler(half_life_s=0.338), safety_gate=gate, motion_limits=_generous_motion_limits(), control_hz=60.0,
    )
    state_src = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = RecordingFollowerWriter()
    loop = RealTimeFollowerControlLoop(
        generator=gen, safety_gate=gate, trajectory_buffer=buf, state_source=state_src, writer=writer,
        config=RealTimeFollowerControlLoopConfig(control_hz=60.0),
    )

    stop_inference = threading.Event()

    def fake_inference_loop():
        seq = 0
        while not stop_inference.is_set():
            time.sleep(fake_latency_s)
            if stop_inference.is_set():
                break
            _publish_constant_chunk(buf, sequence=seq, action=_neutral(0.0))
            seq += 1

    inference_thread = threading.Thread(target=fake_inference_loop, daemon=True)
    loop.start()
    inference_thread.start()
    time.sleep(1.5)
    stop_inference.set()
    inference_thread.join(timeout=2.0)
    loop.stop()

    stats = loop.stats()
    assert stats.actual_hz is not None and stats.actual_hz >= 45.0  # inference latency와 무관하게 유지


# ---------------------------------------------------------------------------
# 섹션 9: inference health 연동 (INFERENCE_DEGRADED)
# ---------------------------------------------------------------------------


def test_inference_degraded_state_when_health_shows_many_failures_but_trajectory_still_valid() -> None:
    from runtime.laptop.async_chunk_inference_worker import WorkerHealthSnapshot

    clock = FakeClock()
    unhealthy = WorkerHealthSnapshot(
        running=True, consecutive_failures=5, last_success_time_monotonic=clock.now() - 1.0,
        last_error="fake HTTP 500 (시뮬레이션)", latest_sequence=3,
        total_requests=10, total_published=4, total_discarded_stale=0,
    )
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, health_source=lambda: unhealthy)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    record = loop._do_tick(scheduled_time=None)
    # trajectory 자체는 여전히 유효하므로(buffer에 usable chunk가 있음) - INFERENCE_DEGRADED로
    # "경고"만 표시하고, write는 계속된다(섹션 9: "inference 한 번 실패했다고 motor
    # thread가 즉시 crash하면 안 됨" - 여기서는 "write를 멈추면 안 됨"으로 확장 해석).
    assert record.state == ControlLoopState.INFERENCE_DEGRADED
    assert record.write_executed is True
    assert writer.write_count == 1


def test_healthy_inference_does_not_trigger_degraded_state() -> None:
    from runtime.laptop.async_chunk_inference_worker import WorkerHealthSnapshot

    clock = FakeClock()
    healthy = WorkerHealthSnapshot(
        running=True, consecutive_failures=0, last_success_time_monotonic=clock.now(),
        last_error=None, latest_sequence=3, total_requests=10, total_published=10, total_discarded_stale=0,
    )
    loop, buf, state_src, writer, _ = _make_loop(clock=clock, health_source=lambda: healthy)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    record = loop._do_tick(scheduled_time=None)
    assert record.state == ControlLoopState.RUNNING


def test_health_source_exception_does_not_crash_tick() -> None:
    clock = FakeClock()

    def exploding_health_source():
        raise RuntimeError("health snapshot 조회 실패 (시뮬레이션)")

    loop, buf, state_src, writer, _ = _make_loop(clock=clock, health_source=exploding_health_source)
    buf.publish(_chunk(sequence=0, obs_time=clock.now(), action=_neutral(0.0)))
    record = loop._do_tick(scheduled_time=None)
    # health 조회 실패가 tick 자체를 방해하면 안 된다 - RUNNING으로 정상 처리(진단
    # 조회는 곁가지일 뿐, 주 경로에 영향을 주지 않는다).
    assert record.state == ControlLoopState.RUNNING
    assert record.write_executed is True


# ---------------------------------------------------------------------------
# 섹션 13 보완: control tick overrun (실 스레드, 인위적으로 느린 state_source)
# ---------------------------------------------------------------------------


def test_control_tick_overrun_recorded_and_loop_recovers() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    _publish_constant_chunk(buf, sequence=0, action=_neutral(0.0))
    gate = _generous_safety_gate()
    gen = RealTimeControlTargetGenerator(
        ensembler=TemporalEnsembler(half_life_s=0.338), safety_gate=gate, motion_limits=_generous_motion_limits(), control_hz=60.0,
    )

    class SlowOnceStateSource:
        def __init__(self) -> None:
            self._calls = 0
            self._inner = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))

        def read(self):
            self._calls += 1
            if self._calls == 5:
                time.sleep(0.1)  # 60Hz period(16.7ms)의 약 6배 - 명백한 overrun 유발
            return self._inner.read()

    writer = RecordingFollowerWriter()
    loop = RealTimeFollowerControlLoop(
        generator=gen, safety_gate=gate, trajectory_buffer=buf, state_source=SlowOnceStateSource(), writer=writer,
        config=RealTimeFollowerControlLoopConfig(control_hz=60.0),
    )
    loop.start()
    time.sleep(1.0)
    loop.stop()

    records = loop.tick_history()
    assert any(r.deadline_overrun_ms > 50.0 for r in records)  # 인위적 지연이 기록됨
    # overrun 이후에도 계속 정상 동작(무한 catch-up으로 멈추지 않고 회복) - 마지막 여러
    # tick은 다시 정상 RUNNING이어야 한다.
    assert all(r.state == ControlLoopState.RUNNING for r in records[-5:])
    stats = loop.stats()
    assert stats.max_overrun_ms > 50.0


# ---------------------------------------------------------------------------
# 섹션 14: dangerous wrist regression - 실 스레드 전체 스택
# ---------------------------------------------------------------------------


def test_dangerous_wrist_full_stack_real_thread_blocks_then_recovers() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    gen = RealTimeControlTargetGenerator(
        ensembler=TemporalEnsembler(half_life_s=0.338), safety_gate=gate, motion_limits=DEFAULT_JOINT_MOTION_LIMITS, control_hz=60.0,
    )
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703
    state_src = FakeFollowerStateSource(initial_state_deg=current)
    writer = RecordingFollowerWriter()
    loop = RealTimeFollowerControlLoop(
        generator=gen, safety_gate=gate, trajectory_buffer=buf, state_source=state_src, writer=writer,
        config=RealTimeFollowerControlLoopConfig(control_hz=60.0),
    )
    dangerous = dict(current)
    dangerous["wrist_flex"] = 40.4879
    _publish_constant_chunk(buf, sequence=0, action=dangerous)

    loop.start()
    time.sleep(0.5)
    assert writer.write_count == 0  # 위험한 target은 0.5초 내내 단 한 번도 write되지 않음
    assert loop.state in (ControlLoopState.INTENT_BLOCKED, ControlLoopState.NO_TRAJECTORY)
    assert 0 in loop.quarantined_sequences

    # 더 최신 정상 chunk 도착 - recovery.
    safe = dict(current)
    safe["elbow_flex"] = 7.08
    _publish_constant_chunk(buf, sequence=1, action=safe)
    time.sleep(0.3)
    loop.stop()

    assert writer.write_count > 0  # 새 정상 chunk 이후로는 write가 재개됨
    for call in writer.calls:
        assert abs(call["wrist_flex"] - current["wrist_flex"]) < 1.0  # 위험한 방향으로 전혀 이동 안 함


# ---------------------------------------------------------------------------
# 섹션 15: normal trajectory continuity - 실 스레드, 데모 유사 ramp를 수 초 재생
# ---------------------------------------------------------------------------


def test_normal_trajectory_continuity_over_several_seconds() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    gen = RealTimeControlTargetGenerator(
        ensembler=TemporalEnsembler(half_life_s=0.338), safety_gate=gate, motion_limits=DEFAULT_JOINT_MOTION_LIMITS, control_hz=60.0,
    )
    state_src = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = RecordingFollowerWriter()
    loop = RealTimeFollowerControlLoop(
        generator=gen, safety_gate=gate, trajectory_buffer=buf, state_source=state_src, writer=writer,
        config=RealTimeFollowerControlLoopConfig(control_hz=60.0),
    )

    stop_inference = threading.Event()

    def demo_like_inference_loop():
        seq = 0
        rate = 3.0  # deg/s - demo-like 느긋한 속도
        t0 = time.monotonic()
        while not stop_inference.is_set():
            time.sleep(0.33)
            if stop_inference.is_set():
                break
            elapsed = time.monotonic() - t0
            _publish_constant_chunk(buf, sequence=seq, action=_neutral(min(rate * elapsed, 5.0)))
            state_src.set_state(_neutral(min(rate * elapsed, 5.0)))  # follower가 실제로 따라갔다고 가정
            seq += 1

    inference_thread = threading.Thread(target=demo_like_inference_loop, daemon=True)
    loop.start()
    inference_thread.start()
    time.sleep(2.5)
    stop_inference.set()
    inference_thread.join(timeout=2.0)
    loop.stop()

    records = loop.tick_history()
    intent_decisions = [r.intent_decision for r in records if r.intent_decision is not None]
    assert intent_decisions, "적어도 일부 tick은 Intent 판정을 거쳐야 한다"
    accept_rate = sum(1 for d in intent_decisions if d == "ACCEPT") / len(intent_decisions)
    assert accept_rate > 0.9  # 정상 demo-like 궤적은 대부분/전부 ACCEPT

    write_indices = [i for i, r in enumerate(records) if r.write_executed]
    assert len(write_indices) > 30  # 연속적으로 target을 받음(멈추지 않음)
    # long unintended pause 없음: 연속된 write 사이 tick 간격이 비정상적으로 길지 않은지
    # (전체 tick 수 대비 write가 듬성듬성이지 않은지) - 인접 write 인덱스 간 최대 gap 확인.
    if len(write_indices) >= 2:
        gaps = [write_indices[i + 1] - write_indices[i] for i in range(len(write_indices) - 1)]
        assert max(gaps) <= 10  # 60Hz 기준 10 tick(~167ms) 이상 write가 끊기지 않음


if __name__ == "__main__":  # pragma: no cover
    pass
