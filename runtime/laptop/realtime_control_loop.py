"""Phase C-3B: production-style 50~60Hz realtime follower control scheduler.

이번 phase 목표(섹션 17, 사용자 문구 그대로): **"실제 hardware writer만 꽂으면 실행
가능한 60Hz full runtime control architecture를 offline integration으로 완성"**. 이
파일 자체는 어떤 실물 connect/send_action도 호출하지 않는다 - 그건 이 모듈을 쓰는
쪽(테스트/향후 hardware validation phase)이 어떤 ``state_source``/``writer``를 주입하는지에
달려 있다.

# 두 loop의 완전한 분리 (섹션 2)

``AsyncVLAChunkInferenceWorker``(Phase C-1B, 이 phase에서 수정 없음)는 자기 페이스(실측
약 3Hz)로 계속 돌며 ``TrajectoryBuffer``에 fresh chunk를 채운다. 이 파일의
``RealTimeFollowerControlLoop``는 완전히 독립된 thread에서 ``control_hz``(기본 60Hz)로
돈다. 두 thread가 공유하는 건 ``TrajectoryBuffer``(자체 lock으로 보호, 이 파일이 그
lock을 오래 쥐지 않음 - ``snapshot()``/``valid_chunks()``는 즉시 반환) 하나뿐이다.
control loop은 inference 완료를 **절대 blocking wait하지 않는다** - 매 tick
``buffer.valid_chunks()``로 "지금 쓸 수 있는 것"만 훑고 바로 다음 단계로 넘어간다. VLA
latency가 300~400ms여도 motor tick 자체의 cadence는 그 latency와 무관하게 유지된다
(``TrajectoryBuffer``의 lock-free-ish 짧은 critical section이 그걸 가능하게 하는 근거 -
``trajectory_buffer.py`` 참고).

# deadline-based scheduling (섹션 1)

``sleep(period)``를 단순 반복하지 않는다 - ``next_deadline += period``를 유지하며
tick 처리 시간을 deadline에서 제외하고 sleep한다. tick이 deadline을 넘기면(overrun)
기록만 하고, 밀린 정도가 ``max_catchup_periods * period``를 넘으면 **catch-up을
포기하고 지금 시점 기준으로 재동기화**한다(``_MAX_CATCHUP_PERIODS`` 참고) - "밀린 tick을
전부 만회하려고 sleep 없이 연속 실행"하는 무한 catch-up loop를 구조적으로 금지한다.

# Intent quarantine (섹션 8)

Intent Validation이 raw target을 막으면, 그 판정에 기여한 chunk의 ``sequence``들을
``_quarantined_sequences``에 추가한다. 다음 tick부터 ``TrajectoryBuffer.snapshot()``에서
그 sequence들을 제외한 뒤 ``RealTimeControlTargetGenerator.tick()``에 넘긴다 - 같은
위험한 chunk를 60Hz로 계속 재평가하며 BLOCK 로그만 반복하지 않는다. 더 최신
observation에서 나온(=더 큰 sequence 번호) 새 chunk가 도착하면 그 chunk는 quarantine
대상이 아니므로 자동으로 다시 사용 가능해진다(recovery) - 별도 "unquarantine" 로직이
필요 없다(quarantine set은 절대 줄어들지 않지만, 매 tick 새로 만들어지는 chunk의
sequence는 항상 그 시점의 quarantine set보다 커서 애초에 그 안에 없다).

# Hold 정책 (섹션 7) - A(no write) / B(last commanded hold) / C(measured hold) 비교

SO-101은 **position-controlled** 서보다 - ``Goal_Position``을 새로 쓰지 않는 한 서보는
마지막으로 받은 목표 위치를 물리적으로 계속 유지한다(``staged_follower_writer.py``의
``send_action`` 문서 참고 - 이 저장소가 만든 동작이 아니라 Feetech 서보/LeRobot의
표준 position-control 동작). 그래서:

    - **A. NO_WRITE (기본값)**: trajectory가 잠깐 끊겨도 아무것도 새로 쓰지 않는다.
      서보가 이미 마지막 명령 위치를 물리적으로 유지하고 있으므로, 이게 곧 "마지막
      안전 위치 유지"와 **정확히 같은 물리적 결과**를 write 없이 얻는다. 위험이 가장
      적다(새 write 자체가 없으므로 stale/오독 state를 실수로 내보낼 방법이 없다).
    - **B. HOLD_LAST_COMMANDED**: A와 물리적으로 동일한 목표지만, 굳이 같은 값을 다시
      write한다 - A 대비 이점이 없고 write 실패/노이즈 위험만 추가된다. 그래도 "제어
      루프가 살아있다는 것을 주기적으로 재확인하고 싶다"는 운영상의 이유가 있을 수
      있어 옵션으로 남긴다.
    - **C. HOLD_MEASURED**: 이번 tick 실측 현재 위치를 그대로 write한다. 서보가 이미
      거기 있으므로 이론적으로는 무해하지만, state read의 노이즈/지연을 그대로
      goal로 되먹임하는 추가 위험이 있고 A/B 대비 아무 이득이 없다.

**기본값은 A(NO_WRITE)** - 위 이유로 가장 안전하고 가장 단순하다. B/C는 완전성을 위해
구현하되 기본이 아니다. **trajectory extrapolation은 세 정책 어디서도 하지 않는다** -
B/C 둘 다 "이미 알려진 안전한 정적 위치"만 다시 쓸 뿐, 새 VLA raw target을 추정하지
않는다. B/C를 쓸 때도 그 정적 target은 **다시 Final SafetyGate로 재검증한 뒤에만**
write된다(§16 writer invariant를 hold 경로에서도 절대 깨지 않기 위함).

# Writer invariant (섹션 16, 가장 중요)

``writer.write()``는 오직 다음 중 하나일 때만 호출된다:

    1. 이번 tick ``ControlTargetResult.target_valid is True``(=Intent ACCEPT AND
       Final SafetyGate ACCEPT, ``realtime_control_target.py`` 참고) - 주 경로.
    2. hold 정책(B/C)이 활성화돼 있고, hold 후보 action이 ``SafetyGate.evaluate()``로
       **다시** ACCEPT된 경우만 - hold 경로.

둘 다 아니면 이 tick은 write를 절대 시도하지 않는다.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, Sequence

from runtime.laptop.action_adapter import adapt_vla_action
from runtime.laptop.async_chunk_inference_worker import WorkerHealthSnapshot
from runtime.laptop.follower_action_writer import WriteResult
from runtime.laptop.follower_state_source import FollowerStateSnapshot
from runtime.laptop.realtime_control_target import ControlTargetResult, RealTimeControlTargetGenerator
from runtime.laptop.safety_gate import SafetyGate
from runtime.laptop.trajectory_buffer import TrajectoryBuffer
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

DEFAULT_CONTROL_HZ = 60.0
# 밀린 정도가 이 배수(주기 단위)를 넘으면 catch-up을 포기하고 지금 시점으로 재동기화한다
# (섹션 1 "무한 catch-up loop 금지"). 2주기(약 33ms @ 60Hz)면 "한 번의 긴 GC pause/OS
# 스케줄링 지연" 정도는 catch-up 시도를 하고, 그보다 심하면(예: 디버거로 멈췄다 재개)
# 곧바로 포기하고 리셋하는 게 낫다는 판단 - 임의로 크게 잡지 않았다.
DEFAULT_MAX_CATCHUP_PERIODS = 2.0
DEFAULT_TICK_HISTORY_SIZE = 3600  # 60Hz 기준 1분 - 진단 목적, 무한정 쌓지 않음
# health_snapshot().consecutive_failures가 이 이상이면 INFERENCE_DEGRADED로 표시한다
# (여전히 buffer에 usable trajectory가 남아있는 동안은 RUNNING과 기능적으로 동일하게
# 계속 write한다 - 섹션 9 "inference 한 번 실패했다고 motor thread가 죽으면 안 됨").
DEFAULT_INFERENCE_DEGRADED_THRESHOLD = 3


class ControlLoopState(str, Enum):
    """섹션 6 명시 상태 전부."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    NO_TRAJECTORY = "NO_TRAJECTORY"
    STALE_TRAJECTORY = "STALE_TRAJECTORY"
    INTENT_BLOCKED = "INTENT_BLOCKED"
    SAFETY_BLOCKED = "SAFETY_BLOCKED"
    INFERENCE_DEGRADED = "INFERENCE_DEGRADED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAULT = "FAULT"


# write가 이번 tick에 실제로 일어날 수 있는 상태 집합(주 경로) - 그 외 상태는 아래
# hold 정책이 명시적으로 다시 허용하지 않는 한 write하지 않는다.
_WRITABLE_STATES = frozenset({ControlLoopState.RUNNING, ControlLoopState.INFERENCE_DEGRADED})
# hold 정책(B/C)이 적용될 수 있는 상태 - "target이 아예 없거나 stale"한 경우로 한정한다.
# INTENT_BLOCKED/SAFETY_BLOCKED는 "명확한 차단 사유가 있는 tick"이므로 hold로 덮어써
# 애매하게 만들지 않는다(설계 선택 - 아래 hold 처리 코드 참고).
_HOLD_ELIGIBLE_STATES = frozenset({ControlLoopState.NO_TRAJECTORY, ControlLoopState.STALE_TRAJECTORY})


class HoldPolicy(str, Enum):
    """모듈 docstring "Hold 정책" 절 참고. 기본값은 ``NO_WRITE``."""

    NO_WRITE = "NO_WRITE"
    HOLD_LAST_COMMANDED = "HOLD_LAST_COMMANDED"
    HOLD_MEASURED = "HOLD_MEASURED"


class FollowerStateSourceProtocol(Protocol):
    def read(self) -> FollowerStateSnapshot: ...


class FollowerActionWriterProtocol(Protocol):
    def write(self, action_deg: dict[str, float]) -> WriteResult: ...


class ControlLoopError(RuntimeError):
    """생성자 인자 오류 등 - 실행 중 fail-closed 처리와는 별개(그건 예외를 던지지 않고
    FAULT 상태로 기록한다, 아래 ``_do_tick`` 참고)."""


@dataclass(frozen=True)
class ControlTickRecord:
    """한 tick의 전체 진단 기록 (섹션 10 요구사항 필드 전부 + 참고용 추가 필드)."""

    tick_index: int
    scheduled_time_monotonic: float | None  # 이 tick이 원래 맞췄어야 할 deadline (첫 tick은 None)
    actual_start_time_monotonic: float
    dt_s: float | None  # 이전 tick의 actual_start 대비 실제 경과 시간 (첫 tick은 None)
    tick_compute_ms: float
    state_read_ms: float | None
    target_compute_ms: float | None  # generator.tick() 소요 시간 (ensemble+intent+guard+safety 전부 포함)
    write_ms: float | None
    deadline_overrun_ms: float  # max(0, ...) - overrun 없으면 0.0
    intent_decision: str | None
    safety_decision: str | None
    raw_target: dict[str, float] | None
    guarded_target: dict[str, float] | None
    final_target: dict[str, float] | None
    clamp_reasons: tuple[str, ...]
    target_valid: bool
    stop_reason: str | None
    contributing_sequences: tuple[int, ...]
    quarantined_sequences_excluded: tuple[int, ...]  # 이번 tick에서 필터링으로 제외된 sequence
    trajectory_age_s: float | None  # 가장 최근 기여 chunk의 관측 시각 대비 나이
    write_attempted: bool
    write_executed: bool
    write_path: str | None  # "primary" | "hold_last_commanded" | "hold_measured" | None
    state: ControlLoopState
    errors: tuple[str, ...]
    quarantine_before_tick: tuple[int, ...] = ()
    quarantine_after_tick: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["state"] = self.state.value
        return d


@dataclass(frozen=True)
class ControlLoopStats:
    """``compute_loop_stats()``의 반환 타입 (섹션 10 통계 helper 요구사항)."""

    n_ticks: int
    actual_hz: float | None
    median_period_ms: float | None
    p90_period_ms: float | None
    p95_period_ms: float | None
    jitter_ms: float | None  # period의 표준편차
    deadline_miss_count: int
    deadline_miss_rate: float
    max_overrun_ms: float
    resync_count: int


def compute_loop_stats(records: Sequence[ControlTickRecord]) -> ControlLoopStats:
    if not records:
        return ControlLoopStats(
            n_ticks=0, actual_hz=None, median_period_ms=None, p90_period_ms=None, p95_period_ms=None,
            jitter_ms=None, deadline_miss_count=0, deadline_miss_rate=0.0, max_overrun_ms=0.0, resync_count=0,
        )
    dts = [r.dt_s for r in records if r.dt_s is not None]
    periods_ms = [dt * 1000.0 for dt in dts]
    periods_ms_sorted = sorted(periods_ms)

    def _percentile(sorted_vals: list[float], pct: float) -> float | None:
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, max(0, round(pct / 100.0 * (len(sorted_vals) - 1))))
        return sorted_vals[idx]

    actual_hz = None
    if len(records) >= 2:
        span = records[-1].actual_start_time_monotonic - records[0].actual_start_time_monotonic
        if span > 0:
            actual_hz = (len(records) - 1) / span

    overruns = [r.deadline_overrun_ms for r in records]
    miss_count = sum(1 for o in overruns if o > 0)
    return ControlLoopStats(
        n_ticks=len(records),
        actual_hz=actual_hz,
        median_period_ms=(statistics.median(periods_ms) if periods_ms else None),
        p90_period_ms=_percentile(periods_ms_sorted, 90),
        p95_period_ms=_percentile(periods_ms_sorted, 95),
        jitter_ms=(statistics.pstdev(periods_ms) if len(periods_ms) >= 2 else 0.0 if periods_ms else None),
        deadline_miss_count=miss_count,
        deadline_miss_rate=(miss_count / len(records)) if records else 0.0,
        max_overrun_ms=max(overruns) if overruns else 0.0,
        resync_count=sum(1 for r in records for e in r.errors if e.startswith("RESYNC:")),
    )


@dataclass(frozen=True)
class RealTimeFollowerControlLoopConfig:
    control_hz: float = DEFAULT_CONTROL_HZ
    hold_policy: HoldPolicy = HoldPolicy.NO_WRITE
    hold_timeout_s: float = 0.5  # trajectory 끊긴 뒤 이 시간까지만 hold-write를 시도(그 이후는 순수 NO_WRITE로 degrade)
    max_catchup_periods: float = DEFAULT_MAX_CATCHUP_PERIODS
    tick_history_size: int = DEFAULT_TICK_HISTORY_SIZE
    inference_degraded_threshold: int = DEFAULT_INFERENCE_DEGRADED_THRESHOLD
    trajectory_max_age_ms: float | None = None  # None이면 TrajectoryBuffer 기본 정책(청크 자체 horizon 경과 여부)만 사용

    def __post_init__(self) -> None:
        if self.control_hz <= 0:
            raise ControlLoopError(f"control_hz는 양수여야 합니다: {self.control_hz}")
        if self.hold_timeout_s < 0:
            raise ControlLoopError(f"hold_timeout_s는 음수일 수 없습니다: {self.hold_timeout_s}")
        if self.max_catchup_periods <= 0:
            raise ControlLoopError(f"max_catchup_periods는 양수여야 합니다: {self.max_catchup_periods}")
        if self.tick_history_size < 1:
            raise ControlLoopError(f"tick_history_size는 1 이상이어야 합니다: {self.tick_history_size}")


class RealTimeFollowerControlLoop:
    """모듈 docstring 참고. ``start()``/``stop()``으로 백그라운드 thread를 제어한다."""

    def __init__(
        self,
        *,
        generator: RealTimeControlTargetGenerator,
        safety_gate: SafetyGate,
        trajectory_buffer: TrajectoryBuffer,
        state_source: FollowerStateSourceProtocol,
        writer: FollowerActionWriterProtocol,
        config: RealTimeFollowerControlLoopConfig | None = None,
        health_source: Callable[[], WorkerHealthSnapshot] | None = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._generator = generator
        self._safety_gate = safety_gate
        self._trajectory_buffer = trajectory_buffer
        self._state_source = state_source
        self._writer = writer
        self._config = config or RealTimeFollowerControlLoopConfig()
        self._health_source = health_source
        self._monotonic = monotonic_fn
        self._period_s = 1.0 / self._config.control_hz

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._diag_lock = threading.Lock()
        self._state = ControlLoopState.STARTING
        self._tick_history: list[ControlTickRecord] = []
        self._tick_index = 0
        self._last_actual_start: float | None = None

        self._quarantined_sequences: set[int] = set()
        self._last_commanded_action: dict[str, float] | None = None
        self._last_valid_tick_time: float | None = None

    # -- lifecycle --------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise ControlLoopError("RealTimeFollowerControlLoop가 이미 실행 중입니다 (duplicate start 방지).")
        self._stop_event.clear()
        with self._diag_lock:
            self._state = ControlLoopState.STARTING
        self._thread = threading.Thread(target=self._run_loop, name="RealTimeFollowerControlLoop", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        with self._diag_lock:
            if self._state not in (ControlLoopState.STOPPED,):
                self._state = ControlLoopState.STOPPING
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                raise TimeoutError(f"RealTimeFollowerControlLoop 스레드가 {timeout_s}s 안에 종료되지 않았습니다.")
        with self._diag_lock:
            self._state = ControlLoopState.STOPPED
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- 조회 -----------------------------------------------------------------------

    @property
    def state(self) -> ControlLoopState:
        with self._diag_lock:
            return self._state

    def tick_history(self) -> tuple[ControlTickRecord, ...]:
        with self._diag_lock:
            return tuple(self._tick_history)

    def stats(self) -> ControlLoopStats:
        return compute_loop_stats(self.tick_history())

    @property
    def quarantined_sequences(self) -> frozenset[int]:
        with self._diag_lock:
            return frozenset(self._quarantined_sequences)

    # -- 내부: scheduler (섹션 1) -------------------------------------------------------

    def _run_loop(self) -> None:
        period = self._period_s
        # ``deadline``은 "이번 tick이 끝나 있어야 하는 시각"이면서 동시에 "다음 tick이
        # 시작해야 하는 시각"이다(주기적 스케줄에서는 같은 값) - 그래서 아래 루프는
        # (1) 이번 tick을 이 값으로 실행하고, (2) sleep은 반드시 **이 값**까지 하고(다음
        # tick의 deadline으로 미리 늘려놓은 값이 아니라), (3) 깨어난 뒤에야 다음 tick을
        # 위해 값을 period만큼 늘린다 - 순서를 바꾸면 매 tick 한 주기씩 밀린다(직접
        # 검증한 버그 - "sleep 대상"과 "다음 tick에 전달할 deadline"을 같은 변수에서
        # 같은 시점에 갱신하면 한 주기 앞서가 버림).
        deadline = self._monotonic() + period
        while not self._stop_event.is_set():
            self._do_tick(scheduled_time=deadline)
            tick_end = self._monotonic()

            max_lag = self._config.max_catchup_periods * period
            if deadline < tick_end - max_lag:
                # 무한 catch-up 방지: 이미 너무 많이 밀렸다 - sleep을 건너뛰고 지금
                # 시점 기준으로 재동기화한다(밀린 tick들을 몰아서 실행하지 않음).
                self._record_resync()
                next_deadline = tick_end + period
                sleep_s = 0.0
            else:
                next_deadline = deadline + period
                sleep_s = deadline - self._monotonic()

            if sleep_s > 0:
                self._stop_event.wait(timeout=sleep_s)
            deadline = next_deadline
        with self._diag_lock:
            self._state = ControlLoopState.STOPPED

    def _record_resync(self) -> None:
        # 다음 tick 기록에 RESYNC 마커를 남기기 위해 별도 큐 대신, 다음 _do_tick 호출의
        # errors 필드에 직접 넣도록 플래그만 세팅한다.
        self._pending_resync_marker = True

    # -- 내부: 한 tick ------------------------------------------------------------------

    def _do_tick(self, *, scheduled_time: float | None) -> ControlTickRecord:
        tick_index = self._tick_index
        self._tick_index += 1
        actual_start = self._monotonic()
        dt_s = None if self._last_actual_start is None else (actual_start - self._last_actual_start)
        self._last_actual_start = actual_start

        errors: list[str] = []
        if getattr(self, "_pending_resync_marker", False):
            errors.append("RESYNC: 누적 지연이 max_catchup_periods를 넘어 재동기화했습니다.")
            self._pending_resync_marker = False

        fault = False
        current_state: dict[str, float] | None = None
        state_read_ms = None
        t0 = self._monotonic()
        try:
            snapshot = self._state_source.read()
            current_state = snapshot.positions_deg
        except Exception as exc:  # noqa: BLE001 - control loop thread는 절대 죽으면 안 됨
            fault = True
            errors.append(f"state_read 실패: {type(exc).__name__}: {exc}")
        state_read_ms = (self._monotonic() - t0) * 1000.0

        quarantined_snapshot = frozenset()
        result: ControlTargetResult | None = None
        target_compute_ms = None
        excluded_sequences: tuple[int, ...] = ()
        if not fault:
            with self._diag_lock:
                quarantined_snapshot = frozenset(self._quarantined_sequences)
            raw_chunks = self._trajectory_buffer.valid_chunks(actual_start, max_chunk_age_ms=self._config.trajectory_max_age_ms)
            # Quarantine는 live buffer 항목에만 의미가 있다. 퇴출/stale sequence를 계속
            # 보존하면 set이 세션 내내 단조 증가한다.
            live_sequences = {c.sequence for c in raw_chunks}
            with self._diag_lock:
                self._quarantined_sequences.intersection_update(live_sequences)
                quarantined_snapshot = frozenset(self._quarantined_sequences)
            chunks = tuple(c for c in raw_chunks if c.sequence not in quarantined_snapshot)
            excluded_sequences = tuple(c.sequence for c in raw_chunks if c.sequence in quarantined_snapshot)

            t0 = self._monotonic()
            try:
                result = self._generator.tick(chunks=chunks, now_monotonic=actual_start, current_follower_state_deg=current_state)
            except Exception as exc:  # noqa: BLE001
                fault = True
                errors.append(f"generator.tick 실패: {type(exc).__name__}: {exc}")
            target_compute_ms = (self._monotonic() - t0) * 1000.0

        # -- Intent quarantine 갱신 (섹션 8) -----------------------------------------
        # WOULD_CLAMP는 해당 tick에서 fail-closed지만 영구 quarantine하지 않는다.
        # schema/mechanical/gross-step REJECT만 해당 live chunk를 격리한다.
        if result is not None and result.intent_decision == "REJECT":
            with self._diag_lock:
                # At an ensemble handoff the newest contributor is the only new
                # evidence. Older contributors were already accepted, so quarantining
                # all of them creates an artificial empty-buffer outage.
                if result.contributing_sequences:
                    self._quarantined_sequences.add(max(result.contributing_sequences))

        with self._diag_lock:
            quarantine_after_tick = tuple(sorted(self._quarantined_sequences))

        # -- 상태 계산 (섹션 6) -------------------------------------------------------
        state = self._compute_state(fault=fault, result=result)

        # -- write 결정 (섹션 5/7/16) --------------------------------------------------
        write_attempted = False
        write_executed = False
        write_path: str | None = None
        write_ms = None
        action_to_write: dict[str, float] | None = None

        if not fault and result is not None and result.target_valid and result.final_target is not None:
            action_to_write = result.final_target
            write_path = "primary"
        elif not fault and current_state is not None and state in _HOLD_ELIGIBLE_STATES and self._config.hold_policy != HoldPolicy.NO_WRITE:
            candidate, path = self._compute_hold_candidate(current_state=current_state, now=actual_start)
            if candidate is not None:
                revalidated = self._revalidate_for_hold(candidate, current_state)
                if revalidated:
                    action_to_write = candidate
                    write_path = path

        writer_fault_reason: str | None = None
        if action_to_write is not None:
            write_attempted = True
            t0 = self._monotonic()
            try:
                write_result = self._writer.write(action_to_write)
                write_executed = write_result.executed
                if write_executed:
                    with self._diag_lock:
                        self._last_commanded_action = dict(action_to_write)
                        self._last_valid_tick_time = actual_start
                else:
                    errors.append(f"writer 실패(비-예외): {write_result.error}")
            except Exception as exc:  # noqa: BLE001
                fault = True
                state = ControlLoopState.FAULT
                writer_fault_reason = "WRITER_FAULT"
                errors.append(f"writer 예외: {type(exc).__name__}: {exc}")
            write_ms = (self._monotonic() - t0) * 1000.0
        # action_to_write가 None인 나머지 모든 경우(INTENT_BLOCKED/SAFETY_BLOCKED, hold
        # timeout 초과, hold 재검증 실패 등)는 write를 전혀 시도하지 않는다 - 이게 바로
        # writer invariant(섹션 16)의 핵심이다.

        overrun_ms = 0.0
        if scheduled_time is not None:
            overrun_ms = max(0.0, (self._monotonic() - scheduled_time) * 1000.0)

        contributing = result.contributing_sequences if result is not None else ()
        traj_age = None
        if result is not None and contributing:
            chunk_by_seq = {c.sequence: c for c in (self._trajectory_buffer.snapshot())}
            newest = max((chunk_by_seq[s] for s in contributing if s in chunk_by_seq), key=lambda c: c.observation_time_monotonic, default=None)
            if newest is not None:
                traj_age = actual_start - newest.observation_time_monotonic

        record = ControlTickRecord(
            tick_index=tick_index,
            scheduled_time_monotonic=scheduled_time,
            actual_start_time_monotonic=actual_start,
            dt_s=dt_s,
            tick_compute_ms=(self._monotonic() - actual_start) * 1000.0,
            state_read_ms=state_read_ms,
            target_compute_ms=target_compute_ms,
            write_ms=write_ms,
            deadline_overrun_ms=overrun_ms,
            intent_decision=(result.intent_decision if result is not None else None),
            safety_decision=(result.safety_decision if result is not None else None),
            raw_target=(result.raw_ensemble_target if result is not None else None),
            guarded_target=(result.guarded_target if result is not None else None),
            final_target=(result.final_target if result is not None else None),
            clamp_reasons=(result.clamp_reasons if result is not None else ()),
            target_valid=bool(result.target_valid) if result is not None else False,
            stop_reason=(writer_fault_reason or (result.stop_reason if result is not None else ("FAULT" if fault else None))),
            contributing_sequences=contributing,
            quarantined_sequences_excluded=excluded_sequences,
            trajectory_age_s=traj_age,
            write_attempted=write_attempted,
            write_executed=write_executed,
            write_path=write_path,
            quarantine_before_tick=tuple(sorted(quarantined_snapshot)),
            quarantine_after_tick=quarantine_after_tick,
            state=state,
            errors=tuple(errors),
        )

        with self._diag_lock:
            self._tick_history.append(record)
            if len(self._tick_history) > self._config.tick_history_size:
                del self._tick_history[: len(self._tick_history) - self._config.tick_history_size]
            self._state = state

        return record

    # -- 내부: 상태 계산 (섹션 6) --------------------------------------------------------

    def _compute_state(self, *, fault: bool, result: ControlTargetResult | None) -> ControlLoopState:
        if fault:
            return ControlLoopState.FAULT
        if result is None:
            return ControlLoopState.FAULT
        if result.target_valid:
            base = ControlLoopState.RUNNING
        elif result.stop_reason is None:
            base = ControlLoopState.FAULT
        elif result.stop_reason == "NO_TARGET":
            base = ControlLoopState.NO_TRAJECTORY
        elif result.stop_reason == "STALE_TRAJECTORY":
            base = ControlLoopState.STALE_TRAJECTORY
        elif result.stop_reason.startswith("INTENT_"):
            base = ControlLoopState.INTENT_BLOCKED
        elif result.stop_reason.startswith("SAFETY_") or result.stop_reason.startswith("GUARD_INVALID"):
            base = ControlLoopState.SAFETY_BLOCKED
        else:
            base = ControlLoopState.FAULT

        if base == ControlLoopState.RUNNING and self._health_source is not None:
            try:
                health = self._health_source()
            except Exception:  # noqa: BLE001 - 진단 조회 실패가 제어 루프를 절대 방해하면 안 됨
                return base
            if health.consecutive_failures >= self._config.inference_degraded_threshold:
                return ControlLoopState.INFERENCE_DEGRADED
        return base

    # -- 내부: hold 후보 계산/재검증 (섹션 7/16) ----------------------------------------

    def _compute_hold_candidate(self, *, current_state: dict[str, float], now: float) -> tuple[dict[str, float] | None, str | None]:
        with self._diag_lock:
            last_valid = self._last_valid_tick_time
            last_commanded = dict(self._last_commanded_action) if self._last_commanded_action is not None else None
        if last_valid is None:
            return None, None  # 한 번도 유효한 write가 없었다 - hold할 "이전 안전 상태" 자체가 없음(예: 시작 직후)
        if (now - last_valid) > self._config.hold_timeout_s:
            return None, None  # hold timeout 초과 - 순수 NO_WRITE로 degrade
        if self._config.hold_policy == HoldPolicy.HOLD_LAST_COMMANDED:
            if last_commanded is None:
                return None, None
            return last_commanded, "hold_last_commanded"
        if self._config.hold_policy == HoldPolicy.HOLD_MEASURED:
            return dict(current_state), "hold_measured"
        return None, None

    def _revalidate_for_hold(self, candidate: dict[str, float], current_state: dict[str, float]) -> bool:
        """hold 후보를 Final SafetyGate로 다시 검증한다 - writer invariant(섹션 16)를
        hold 경로에서도 절대 깨지 않기 위함(모듈 docstring "Writer invariant" 절)."""
        try:
            adapted = adapt_vla_action(candidate)
        except Exception:  # noqa: BLE001
            return False
        decision = self._safety_gate.evaluate(adapted_action=adapted, current_state_deg=current_state, observation_valid=True)
        return decision.decision == "ACCEPT"
