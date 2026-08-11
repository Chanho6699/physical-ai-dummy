"""Phase C-1B: VLA inference를 robot control loop와 분리하는 백그라운드 worker.

## 루프 (섹션 3)

별도 thread에서 계속 반복한다::

    loop:
        fresh observation snapshot 캡처
        -> vla_client.predict_chunk(...)   (/predict_chunk, Phase C-1A - 변경 없음)
        -> 응답 검증
        -> TimestampedActionChunk 생성
        -> TrajectoryBuffer.publish()
        -> (선택적 min_interval만큼 대기) -> 즉시 다음 iteration

Sleep으로 3Hz 같은 특정 cadence를 강제하지 않는다 - 이전 inference가 끝나는 즉시 다음
fresh observation으로 재시도한다(실측 steady-state ~338ms 기준 자연스럽게 ~3Hz 근방이
나오지만, 이건 강제된 값이 아니라 추론 자체의 소요 시간이 만드는 결과다). CPU/GPU/서버
과부하 방지용 ``min_interval_s``(기본값 0.0 = 제한 없음)는 선택적으로만 제공한다.

## /session/reset을 호출하지 않는 이유 (섹션 5)

``predict_chunk()``는 ``SmolVLAPolicy.predict_action_chunk()``를 쓰는데(Phase C-1A,
``runtime/desktop/vla_server.py``), 이 메서드는 ``select_action()``의 action queue를 전혀
건드리지 않는다 - 매 호출마다 그 순간의 관측으로 항상 fresh 전체 chunk를 새로 추론한다.
"큐를 비워서 fresh하게 만든다"는 ``session_reset()``의 존재 이유(``staged_real_rollout.py``가
매 step 호출하는 것) 자체가 여기엔 없다 - 그래서 이 worker는 ``session_reset()``을 전혀
호출하지 않는다(``staged_real_rollout.py``와의 의도적인 차이).

**전제(주석으로 명확히 남김, 섹션 5 요구사항)**: 이 설계는 Candidate B의 실제 checkpoint
config가 ``n_obs_steps=1``이라는 사실에 의존한다 - 즉 정책이 "지금 이 순간의 관측 하나"만
보고 chunk를 만들고, 여러 과거 관측을 누적해 쓰는 관측 history가 없다(조사 결과,
Phase C-1A 보고서 참고). 만약 향후 다른 checkpoint가 ``n_obs_steps>1``이라면, 매 호출마다
관측 하나만 새로 넣고 이전 관측 history를 공유하는 게 의도한 semantics인지 다시 검토해야
한다 - 이 worker는 그 경우를 대비한 코드가 없다.

## 절대 하지 않는 것 (섹션 12)

이 worker는 Safety Gate를 모르고, follower에 아무것도 write하지 않고, action을 하나도
"선택"하지 않는다(temporal ensemble/interpolation 없음) - ``TrajectoryBuffer``에 chunk를
채워 넣기만 한다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from runtime.laptop.observation_snapshot import ObservationSnapshotProvider
from runtime.laptop.trajectory_buffer import TrajectoryBuffer
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

DEFAULT_MIN_INTERVAL_S = 0.0  # 제한 없음 - 이전 inference 종료 즉시 다음 fresh observation


class ChunkPredictResultProtocol(Protocol):
    """``VLAHttpClient.predict_chunk()``/``InProcessSmolVLAClient.predict_chunk()``가
    공통으로 반환하는 ``ChunkPredictResult``의 duck-typed 부분집합 - worker가 실제로
    쓰는 필드만 명시한다(순환 import 방지 목적도 겸함)."""

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


class VLAChunkClientProtocol(Protocol):
    def predict_chunk(
        self, *, session_id: str, task: str, sequence: int, state: dict[str, float], images: dict[str, object]
    ) -> ChunkPredictResultProtocol: ...


@dataclass(frozen=True)
class WorkerHealthSnapshot:
    """``AsyncVLAChunkInferenceWorker.health_snapshot()``의 반환 타입 (섹션 8 요구사항)."""

    running: bool
    consecutive_failures: int
    last_success_time_monotonic: float | None
    last_error: str | None
    latest_sequence: int | None
    # 참고용 추가 필드(요구사항 외 - 진단에 유용해서 추가, 필수 계약을 어기지 않음).
    total_requests: int
    total_published: int
    total_discarded_stale: int


class AsyncVLAChunkInferenceWorker:
    """별도 thread에서 ``observation_provider.capture()`` -> ``vla_client.predict_chunk()``
    -> ``TrajectoryBuffer.publish()``를 반복하는 백그라운드 worker.

    ``monotonic_fn``은 테스트에서 controllable fake clock을 주입하기 위한 훅이다(섹션 10 -
    "fake clock 또는 controllable timestamps로 검증"). 기본값은 진짜 ``time.monotonic``.
    """

    def __init__(
        self,
        *,
        vla_client: VLAChunkClientProtocol,
        observation_provider: ObservationSnapshotProvider,
        buffer: TrajectoryBuffer,
        session_id: str,
        task: str,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_interval_s < 0:
            raise ValueError(f"min_interval_s는 음수일 수 없습니다: {min_interval_s}")
        self._vla_client = vla_client
        self._observation_provider = observation_provider
        self._buffer = buffer
        self._session_id = session_id
        self._task = task
        self._min_interval_s = min_interval_s
        self._monotonic = monotonic_fn

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._sequence = 0

        # health 상태는 worker thread가 쓰고 임의의 다른 thread(호출자)가 읽으므로 별도
        # lock으로 보호한다 - TrajectoryBuffer의 lock과는 완전히 분리된 lock이고, 이
        # lock을 쥔 채로 buffer/vla_client 등 다른 객체를 절대 호출하지 않는다(섹션 9
        # "buffer lock과 worker lock deadlock 없음" - 두 lock이 서로를 기다릴 경로 자체가
        # 코드에 존재하지 않는다).
        self._health_lock = threading.Lock()
        self._running = False
        self._consecutive_failures = 0
        self._last_success_time_monotonic: float | None = None
        self._last_error: str | None = None
        self._latest_sequence: int | None = None
        self._total_requests = 0
        self._total_published = 0
        self._total_discarded_stale = 0

    # -- lifecycle (섹션 9) -------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("AsyncVLAChunkInferenceWorker가 이미 실행 중입니다 (duplicate start 방지).")
        self._stop_event.clear()
        with self._health_lock:
            self._running = True
        # daemon=True는 "호출자가 stop()을 깜빡했을 때 프로세스 종료를 막지 않는" 최후의
        # 안전장치일 뿐이다 - 정상적인 종료 경로는 항상 stop_event + join()이다(섹션 9
        # "daemon thread에 의존해서 강제 종료하지 말 것" - 여기선 daemon을 종료 *수단*으로
        # 쓰지 않는다는 뜻으로 해석해 stop()을 유일한 정상 종료 경로로 구현했다).
        self._thread = threading.Thread(target=self._run_loop, name="AsyncVLAChunkInferenceWorker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                # join이 timeout 안에 안 끝났다 - 강제로 죽이지 않고(Python thread는 그럴
                # 방법도 없음) 명확한 예외로 알린다. 호출자가 이걸 무시하고 조용히 넘어가지
                # 않게 한다.
                raise TimeoutError(
                    f"AsyncVLAChunkInferenceWorker 스레드가 {timeout_s}s 안에 종료되지 않았습니다."
                )
        with self._health_lock:
            self._running = False
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- 조회 -----------------------------------------------------------------------

    def health_snapshot(self) -> WorkerHealthSnapshot:
        with self._health_lock:
            return WorkerHealthSnapshot(
                running=self._running,
                consecutive_failures=self._consecutive_failures,
                last_success_time_monotonic=self._last_success_time_monotonic,
                last_error=self._last_error,
                latest_sequence=self._latest_sequence,
                total_requests=self._total_requests,
                total_published=self._total_published,
                total_discarded_stale=self._total_discarded_stale,
            )

    # -- 내부 루프 --------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._run_one_iteration()
            if self._min_interval_s > 0:
                # sleep 대신 stop_event.wait()을 써서, 대기 중에도 stop()이 즉시 반응하게
                # 한다(섹션 9 "clean shutdown" - min_interval 도중에도 멈춰야 함).
                self._stop_event.wait(timeout=self._min_interval_s)

    def _run_one_iteration(self) -> None:
        sequence = self._sequence
        self._sequence += 1
        with self._health_lock:
            self._total_requests += 1

        try:
            snapshot = self._observation_provider.capture(sequence=sequence)
        except Exception as exc:  # noqa: BLE001 - worker thread는 절대 죽으면 안 됨(섹션 8/9)
            self._record_failure(f"observation capture 실패: {type(exc).__name__}: {exc}")
            return

        request_started = self._monotonic()
        try:
            result = self._vla_client.predict_chunk(
                session_id=self._session_id, task=snapshot.task, sequence=sequence,
                state=snapshot.state, images=snapshot.images,
            )
        except Exception as exc:  # noqa: BLE001
            self._record_failure(f"predict_chunk 호출 중 예외: {type(exc).__name__}: {exc}")
            return
        response_received = self._monotonic()

        if not result.ok:
            self._record_failure(f"{result.error_kind}: {result.error_message}")
            return

        try:
            chunk = TimestampedActionChunk(
                sequence=sequence,
                session_id=self._session_id,
                observation_time_monotonic=snapshot.capture_monotonic_time,
                request_started_time_monotonic=request_started,
                response_received_time_monotonic=response_received,
                server_received_at=result.server_received_at,
                server_responded_at=result.server_responded_at,
                inference_latency_ms=result.inference_latency_ms,
                chunk_index_spacing_s=result.chunk_index_spacing_s,
                chunk_size=result.chunk_size,
                actions=tuple(result.chunk),
                model_id=result.model_id,
                backend=result.backend,
            )
        except Exception as exc:  # noqa: BLE001 - result.chunk_index_spacing_s가 None인 등
            # ChunkPredictResult.ok=True인데 필수 필드가 비정상인 방어적 케이스까지 포함.
            self._record_failure(f"TimestampedActionChunk 생성 실패: {type(exc).__name__}: {exc}")
            return

        publish_result = self._buffer.publish(chunk)
        if publish_result.accepted:
            self._record_success(sequence)
        elif publish_result.discarded_as_stale_out_of_order:
            # predict_chunk() 자체는 성공했다 - 단지 더 최신 응답이 먼저 도착해서 버려진
            # 것뿐이므로 "추론/통신 실패"가 아니다. consecutive_failures를 건드리지 않고
            # last_success_time은 갱신한다(섹션 8: "last valid chunk는 남겨두되..." 정책과
            # 같은 정신 - inference 자체의 건강 상태를 반영).
            self._record_success(sequence, published=False)
        else:
            # chunk.validate() 실패(이론상 거의 안 일어남 - predict_chunk() 반환 전에 이미
            # validate_action_chunk를 거쳤으므로) - 이것도 통신 실패는 아니지만 데이터
            # 정합성 문제이므로 보수적으로 실패로 기록한다.
            self._record_failure(f"buffer.publish 거부(비정상 데이터): {publish_result.reason}")

    def _record_success(self, sequence: int, *, published: bool = True) -> None:
        with self._health_lock:
            self._consecutive_failures = 0
            self._last_success_time_monotonic = self._monotonic()
            self._last_error = None
            self._latest_sequence = sequence
            if published:
                self._total_published += 1
            else:
                self._total_discarded_stale += 1

    def _record_failure(self, message: str) -> None:
        with self._health_lock:
            self._consecutive_failures += 1
            self._last_error = message
