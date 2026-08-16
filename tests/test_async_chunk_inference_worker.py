"""runtime/laptop/async_chunk_inference_worker.py의 AsyncVLAChunkInferenceWorker 검증.

Case A/B/D/E/F는 실제 background thread 없이 ``worker._run_one_iteration()``을 테스트
스레드에서 직접, 결정적으로 호출해서 검증한다(controllable fake clock 사용, 섹션 10
"fake clock 또는 controllable timestamps로 검증" 요구사항) - race condition 없이 정확한
경계값을 재현하기 위한 의도적 선택이다. Case C(out-of-order)는
``tests/test_trajectory_buffer.py``에서 이미 직접 커버한다 - 이 worker는 항상 순차적으로만
요청하므로(한 번에 하나의 predict_chunk 호출만 진행) 진짜 out-of-order 응답을 이 worker
혼자서는 재현할 수 없다(향후 pipelined/overlapping 요청을 도입하면 그때 이 worker
레벨에서도 재현 가능해질 것).

Case G(clean shutdown)와 기본 통합 스모크는 실제 ``threading.Thread``를 띄워서 검증한다.

실제 하드웨어 접근 없음 - 전부 Fake observation provider + Fake VLA client.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.async_chunk_inference_worker import AsyncVLAChunkInferenceWorker
from runtime.laptop.observation_snapshot import ObservationSnapshot
from runtime.laptop.trajectory_buffer import TrajectoryBuffer

SPACING = 1.0 / 30.0
CHUNK_SIZE = 50


class FakeClock:
    """테스트가 직접 시간을 전진시키는 controllable monotonic clock (섹션 10)."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        self._t += dt
        return self._t

    def __call__(self) -> float:  # worker의 monotonic_fn으로 바로 주입 가능
        return self._t


@dataclass
class FakeObservationSnapshotProvider:
    clock: FakeClock
    state: dict[str, float] = field(default_factory=lambda: {j: 0.0 for j in JOINT_ORDER})
    task: str = "pick up the cube"
    fail: bool = False
    capture_calls: list[int] = field(default_factory=list)

    def capture(self, *, sequence: int) -> ObservationSnapshot:
        self.capture_calls.append(sequence)
        if self.fail:
            raise RuntimeError("카메라 캡처 실패 (시뮬레이션)")
        img = {"observation.images.workspace": None, "observation.images.wrist": None}
        return ObservationSnapshot(
            images=img, state=dict(self.state), task=self.task,
            capture_monotonic_time=self.clock.now(), sequence=sequence,
        )


@dataclass
class _FakeChunkResult:
    ok: bool
    chunk: list[dict[str, float]] | None
    chunk_size: int | None
    chunk_index_spacing_s: float | None
    model_id: str | None
    backend: str | None
    inference_latency_ms: float | None
    server_received_at: float | None
    server_responded_at: float | None
    error_kind: str | None
    error_message: str | None


@dataclass
class ChunkPlan:
    """``ScriptedVLAChunkClient``의 한 iteration 계획 - clock을 얼마나 전진시키고
    (=inference/통신 소요시간 시뮬레이션) 무엇을 반환할지."""

    latency_s: float = 0.0
    ok: bool = True
    chunk_size: int = CHUNK_SIZE
    spacing: float = SPACING
    error_kind: str | None = None
    error_message: str | None = None
    raise_exception: bool = False
    inject_nan: bool = False  # ok=True인데 chunk 안에 NaN을 심어서 defense-in-depth 테스트


@dataclass
class ScriptedVLAChunkClient:
    clock: FakeClock
    plans: list[ChunkPlan]
    calls: list[dict] = field(default_factory=list)

    def predict_chunk(self, *, session_id, task, sequence, state, images):
        self.calls.append({"session_id": session_id, "sequence": sequence, "state": dict(state)})
        idx = len(self.calls) - 1
        plan = self.plans[idx] if idx < len(self.plans) else self.plans[-1]
        self.clock.advance(plan.latency_s)
        if plan.raise_exception:
            raise RuntimeError("predict_chunk 통신 예외 (시뮬레이션)")
        if not plan.ok:
            return _FakeChunkResult(
                ok=False, chunk=None, chunk_size=None, chunk_index_spacing_s=None, model_id=None, backend=None,
                inference_latency_ms=None, server_received_at=None, server_responded_at=None,
                error_kind=plan.error_kind or "communication", error_message=plan.error_message or "fake 실패",
            )
        actions = [dict(state) for _ in range(plan.chunk_size)]
        if plan.inject_nan:
            actions[0] = {**actions[0], "wrist_roll": math.nan}
        return _FakeChunkResult(
            ok=True, chunk=actions, chunk_size=plan.chunk_size, chunk_index_spacing_s=plan.spacing,
            model_id="fake-model", backend="fake", inference_latency_ms=plan.latency_s * 1000.0,
            server_received_at=None, server_responded_at=None, error_kind=None, error_message=None,
        )


def _make_worker(*, clock: FakeClock, plans: list[ChunkPlan], obs_provider=None, max_chunks=4, min_interval_s=0.0):
    vla_client = ScriptedVLAChunkClient(clock=clock, plans=plans)
    provider = obs_provider or FakeObservationSnapshotProvider(clock=clock)
    buffer = TrajectoryBuffer(max_chunks=max_chunks)
    worker = AsyncVLAChunkInferenceWorker(
        vla_client=vla_client, observation_provider=provider, buffer=buffer, session_id="s1",
        task="pick up the cube", min_interval_s=min_interval_s, monotonic_fn=clock,
    )
    return worker, vla_client, provider, buffer


# ---------------------------------------------------------------------------
# Case A (섹션 10/11): inference latency=330ms, spacing=33.33ms, 50-step chunk 수신
# -> response 시점에서 약 index 9~10부터 미래
# ---------------------------------------------------------------------------


def test_case_a_latency_330ms_leaves_correct_remaining_future() -> None:
    clock = FakeClock(start=1000.0)
    worker, vla_client, provider, buffer = _make_worker(clock=clock, plans=[ChunkPlan(latency_s=0.330)])

    worker._run_one_iteration()

    chunk = buffer.latest()
    assert chunk is not None
    assert chunk.observation_time_monotonic == pytest.approx(1000.0)  # capture 시점(clock 전진 전)
    now = clock.now()  # 330ms 전진된 현재 시각(response_received와 동일 시점)
    assert chunk.current_index(now) == 9
    assert chunk.first_future_index(now) == 10
    assert chunk.remaining_future_count(now) == 40


# ---------------------------------------------------------------------------
# Case B (섹션 10): 다음 chunk가 340ms 뒤 도착 -> latest 교체, old chunk도 history에 남음
# ---------------------------------------------------------------------------


def test_case_b_second_chunk_340ms_later_becomes_latest_keeps_history() -> None:
    clock = FakeClock(start=2000.0)
    worker, vla_client, provider, buffer = _make_worker(
        clock=clock, plans=[ChunkPlan(latency_s=0.05), ChunkPlan(latency_s=0.05)], max_chunks=4,
    )

    worker._run_one_iteration()  # seq 0, obs=2000.0
    first_latest = buffer.latest()
    assert first_latest.sequence == 0

    clock.advance(0.34 - 0.05)  # 다음 관측 캡처 시점을 "이전 응답 후 340ms 뒤"로 맞춤
    worker._run_one_iteration()  # seq 1

    assert buffer.latest().sequence == 1
    seqs = [c.sequence for c in buffer.snapshot()]
    assert 0 in seqs and 1 in seqs  # old chunk도 history에 남음


# ---------------------------------------------------------------------------
# Case D (섹션 10): chunk horizon 완전 경과 -> expired/unusable
# ---------------------------------------------------------------------------


def test_case_d_chunk_horizon_fully_elapsed_is_expired() -> None:
    clock = FakeClock(start=3000.0)
    worker, vla_client, provider, buffer = _make_worker(clock=clock, plans=[ChunkPlan(latency_s=0.05)])
    worker._run_one_iteration()

    chunk = buffer.latest()
    horizon_end = chunk.horizon_end_time_monotonic  # obs_time + chunk_size*spacing (마지막 index=49 다음)
    mid_horizon = chunk.observation_time_monotonic + 25 * chunk.chunk_index_spacing_s  # 아직 여유 있음
    assert chunk.is_expired(mid_horizon) is False
    assert chunk.is_expired(horizon_end) is True
    assert chunk.is_expired(horizon_end + 10.0) is True
    assert buffer.valid_chunks(horizon_end + 10.0) == ()
    assert len(buffer.valid_chunks(mid_horizon)) == 1


# ---------------------------------------------------------------------------
# Case E (섹션 8/10): HTTP 실패 3회 후 성공 -> worker 생존, consecutive_failures reset
# ---------------------------------------------------------------------------


def test_case_e_three_failures_then_success_resets_consecutive_failures() -> None:
    clock = FakeClock(start=4000.0)
    plans = [
        ChunkPlan(ok=False, error_kind="communication", error_message="연결 실패 1"),
        ChunkPlan(ok=False, error_kind="communication", error_message="연결 실패 2"),
        ChunkPlan(raise_exception=True),  # 통신 실패의 또 다른 형태(예외) - 이것도 안 죽어야 함
        ChunkPlan(ok=True, latency_s=0.05),
    ]
    worker, vla_client, provider, buffer = _make_worker(clock=clock, plans=plans)

    worker._run_one_iteration()
    h1 = worker.health_snapshot()
    assert h1.consecutive_failures == 1
    assert h1.last_error is not None
    assert buffer.latest() is None

    worker._run_one_iteration()
    h2 = worker.health_snapshot()
    assert h2.consecutive_failures == 2

    worker._run_one_iteration()  # 예외 발생 케이스 - worker 메서드 자체는 죽지 않고 정상 반환해야 함
    h3 = worker.health_snapshot()
    assert h3.consecutive_failures == 3
    assert "예외" in h3.last_error or "RuntimeError" in h3.last_error

    worker._run_one_iteration()  # 이제 성공
    h4 = worker.health_snapshot()
    assert h4.consecutive_failures == 0  # reset
    assert h4.last_error is None
    assert h4.last_success_time_monotonic is not None
    assert buffer.latest() is not None
    assert h4.total_published == 1
    assert h4.total_requests == 4


# ---------------------------------------------------------------------------
# Case F (섹션 8/10): malformed/NaN chunk -> publish 금지
# ---------------------------------------------------------------------------


def test_case_f_nan_chunk_is_never_published() -> None:
    clock = FakeClock(start=5000.0)
    worker, vla_client, provider, buffer = _make_worker(
        clock=clock, plans=[ChunkPlan(ok=True, latency_s=0.05, inject_nan=True)],
    )

    worker._run_one_iteration()

    assert buffer.latest() is None  # NaN이 섞인 chunk는 publish되지 않음
    h = worker.health_snapshot()
    assert h.consecutive_failures == 1  # 이것도 실패로 집계됨 (데이터 정합성 문제)
    assert h.total_published == 0


def test_observation_capture_failure_does_not_call_predict_chunk() -> None:
    clock = FakeClock(start=5100.0)
    provider = FakeObservationSnapshotProvider(clock=clock, fail=True)
    worker, vla_client, _, buffer = _make_worker(clock=clock, plans=[ChunkPlan()], obs_provider=provider)

    worker._run_one_iteration()

    assert len(vla_client.calls) == 0  # 관측 실패 시 predict_chunk를 아예 시도하지 않음
    h = worker.health_snapshot()
    assert h.consecutive_failures == 1
    assert buffer.latest() is None


# ---------------------------------------------------------------------------
# health_snapshot() 기본 필드
# ---------------------------------------------------------------------------


def test_health_snapshot_initial_state() -> None:
    clock = FakeClock()
    worker, *_ = _make_worker(clock=clock, plans=[ChunkPlan()])
    h = worker.health_snapshot()
    assert h.running is False
    assert h.consecutive_failures == 0
    assert h.last_success_time_monotonic is None
    assert h.last_error is None
    assert h.latest_sequence is None
    assert h.total_requests == 0
    assert h.total_published == 0


def test_sequence_increments_each_iteration() -> None:
    clock = FakeClock()
    worker, vla_client, *_ = _make_worker(clock=clock, plans=[ChunkPlan(latency_s=0.01)])
    worker._run_one_iteration()
    worker._run_one_iteration()
    worker._run_one_iteration()
    assert [c["sequence"] for c in vla_client.calls] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Case G (섹션 9/10): start()/stop() clean shutdown + duplicate start 방지
# ---------------------------------------------------------------------------


def test_case_g_start_then_stop_clean_shutdown() -> None:
    clock = FakeClock()
    worker, *_ = _make_worker(clock=clock, plans=[ChunkPlan(latency_s=0.0)], min_interval_s=0.005)

    worker.start()
    assert worker.is_running() is True
    time.sleep(0.05)  # 실제 background thread가 몇 iteration 돌 시간을 줌
    worker.stop(timeout_s=5.0)
    assert worker.is_running() is False
    assert worker.health_snapshot().running is False


def test_duplicate_start_raises() -> None:
    clock = FakeClock()
    worker, *_ = _make_worker(clock=clock, plans=[ChunkPlan(latency_s=0.0)], min_interval_s=0.01)
    worker.start()
    try:
        with pytest.raises(RuntimeError):
            worker.start()
    finally:
        worker.stop()


def test_start_stop_start_stop_cycle_works() -> None:
    clock = FakeClock()
    worker, *_ = _make_worker(clock=clock, plans=[ChunkPlan(latency_s=0.0)], min_interval_s=0.005)
    worker.start()
    time.sleep(0.02)
    worker.stop()
    assert worker.is_running() is False

    worker.start()  # 재시작 가능해야 함
    time.sleep(0.02)
    worker.stop()
    assert worker.is_running() is False


def test_stop_without_start_is_a_noop() -> None:
    clock = FakeClock()
    worker, *_ = _make_worker(clock=clock, plans=[ChunkPlan()])
    worker.stop()  # 예외 없이 조용히 넘어가야 함
    assert worker.is_running() is False


def test_worker_thread_survives_persistent_failures_real_thread() -> None:
    """실제 background thread에서, 매 요청이 계속 실패해도 스레드가 죽지 않고 계속
    돈다는 것을 실측한다 (exception isolation, 섹션 8/9)."""
    clock = FakeClock()
    always_fail_client = ScriptedVLAChunkClient(clock=clock, plans=[ChunkPlan(raise_exception=True)])
    provider = FakeObservationSnapshotProvider(clock=clock)
    buffer = TrajectoryBuffer()
    worker = AsyncVLAChunkInferenceWorker(
        vla_client=always_fail_client, observation_provider=provider, buffer=buffer, session_id="s1",
        task="t", min_interval_s=0.005, monotonic_fn=time.monotonic,  # 실제 clock (thread 타이밍용)
    )
    worker.start()
    time.sleep(0.08)
    assert worker.is_running() is True  # 계속 실패해도 스레드는 살아있음
    h = worker.health_snapshot()
    assert h.consecutive_failures > 0
    assert h.total_requests > 0
    worker.stop()
    assert worker.is_running() is False


def test_worker_publishes_multiple_chunks_in_real_background_thread() -> None:
    """기본 통합 스모크 - 실제 thread에서 성공 응답이 실제로 buffer에 쌓이는지."""
    clock = FakeClock()
    always_ok_client = ScriptedVLAChunkClient(clock=clock, plans=[ChunkPlan(latency_s=0.0)])
    provider = FakeObservationSnapshotProvider(clock=clock)
    buffer = TrajectoryBuffer()
    worker = AsyncVLAChunkInferenceWorker(
        vla_client=always_ok_client, observation_provider=provider, buffer=buffer, session_id="s1",
        task="t", min_interval_s=0.005, monotonic_fn=time.monotonic,
    )
    worker.start()
    time.sleep(0.1)
    worker.stop()

    h = worker.health_snapshot()
    assert h.total_published >= 1
    assert buffer.latest() is not None
    assert h.consecutive_failures == 0


def test_max_requests_one_stops_after_exactly_one_publish() -> None:
    clock = FakeClock()
    client = ScriptedVLAChunkClient(clock=clock, plans=[ChunkPlan(latency_s=0.0)])
    provider = FakeObservationSnapshotProvider(clock=clock)
    buffer = TrajectoryBuffer()
    worker = AsyncVLAChunkInferenceWorker(
        vla_client=client, observation_provider=provider, buffer=buffer, session_id="single",
        task="t", min_interval_s=0.0, max_requests=1, monotonic_fn=time.monotonic,
    )

    worker.start()
    time.sleep(0.05)
    worker.stop()

    health = worker.health_snapshot()
    assert health.total_requests == 1
    assert health.total_published == 1
    assert len(client.calls) == 1
    assert [chunk.sequence for chunk in buffer.snapshot()] == [0]


@pytest.mark.parametrize("value", [0, -1])
def test_max_requests_rejects_non_positive_values(value: int) -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match="max_requests"):
        AsyncVLAChunkInferenceWorker(
            vla_client=ScriptedVLAChunkClient(clock=clock, plans=[ChunkPlan()]),
            observation_provider=FakeObservationSnapshotProvider(clock=clock),
            buffer=TrajectoryBuffer(), session_id="single", task="t", max_requests=value,
        )


def test_slow_single_response_is_published_but_expired_on_observation_timebase() -> None:
    clock = FakeClock(start=7000.0)
    client = ScriptedVLAChunkClient(clock=clock, plans=[ChunkPlan(latency_s=2.0)])
    provider = FakeObservationSnapshotProvider(clock=clock)
    buffer = TrajectoryBuffer()
    worker = AsyncVLAChunkInferenceWorker(
        vla_client=client, observation_provider=provider, buffer=buffer,
        session_id="trace", task="t", max_requests=1, monotonic_fn=clock,
    )

    worker._run_one_iteration()
    health = worker.health_snapshot()

    assert health.total_requests == 1
    assert health.total_published == 1
    assert buffer.latest().sequence == 0
    assert health.last_observation_capture_time_monotonic == pytest.approx(7000.0)
    assert health.last_request_started_time_monotonic == pytest.approx(7000.0)
    assert health.last_response_received_time_monotonic == pytest.approx(7002.0)
    assert health.last_publication_time_monotonic == pytest.approx(7002.0)
    assert health.last_chunk_end_time_monotonic == pytest.approx(7000.0 + 50.0 / 30.0)
    assert buffer.valid_chunks(clock.now()) == ()


def test_single_diagnostic_rebases_only_effective_chunk_time_to_publication() -> None:
    clock = FakeClock(start=8000.0)
    client = ScriptedVLAChunkClient(clock=clock, plans=[ChunkPlan(latency_s=2.0)])
    provider = FakeObservationSnapshotProvider(clock=clock)
    buffer = TrajectoryBuffer()
    worker = AsyncVLAChunkInferenceWorker(
        vla_client=client, observation_provider=provider, buffer=buffer,
        session_id="trace", task="t", max_requests=1,
        rebase_chunk_to_publication_time=True, monotonic_fn=clock,
    )

    worker._run_one_iteration()
    health = worker.health_snapshot()
    chunk = buffer.latest()

    assert health.total_requests == 1
    assert health.total_published == 1
    assert chunk.sequence == 0
    assert health.last_observation_capture_time_monotonic == pytest.approx(8000.0)
    assert health.last_request_started_time_monotonic == pytest.approx(8000.0)
    assert health.last_response_received_time_monotonic == pytest.approx(8002.0)
    assert health.last_publication_time_monotonic == pytest.approx(8002.0)
    assert chunk.observation_time_monotonic == pytest.approx(8002.0)
    assert chunk.horizon_end_time_monotonic == pytest.approx(8002.0 + 50.0 / 30.0)
    assert buffer.valid_chunks(clock.now()) == (chunk,)
