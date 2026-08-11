#!/usr/bin/env python3
"""Phase C-5: 실제 SO-101 follower에 붙는 최종 realtime pick & drop runtime.

C-4까지 offline/shadow로 검증된 스택을 실물에 연결한다 - 새 architecture를 만들지
않는다:

    AsyncVLAChunkInferenceWorker -> TrajectoryBuffer -> TemporalEnsembler ->
    PolicyIntentValidator -> Motion Guard -> Final SafetyGate ->
    RealTimeFollowerControlLoop -> SO101FollowerActionWriter

이 스크립트가 실제로 새로 만드는 건 딱 두 가지 작은 접합부(hardware layer 자체는 재사용,
"duplicate hardware layer" 아님)뿐이다:

    1. ``RealObservationSnapshotProvider`` - ``RealCameraObservationSource`` +
       ``ReadOnlyRealFollowerStateSource``를 합쳐 ``AsyncVLAChunkInferenceWorker``가
       원하는 ``ObservationSnapshot``을 만든다. ``observation_snapshot.py``의 모듈
       docstring이 이미 "향후 실제 runtime에서는 이 Protocol을 만족하는 클래스가
       camera_source.py/follower_state_source.py를 내부적으로 호출해 만들어질
       것이다(아직 만들지 않음)"로 예고해 둔 바로 그 접합부다.
    2. ``LockedFollowerStateSource`` - 아래 "왜 lock이 필요한가" 절 참고.

## follower 연결 - 왜 connection이 2개(읽기 1 + 쓰기 1)인가

``scripts/run_real_follower_staged_safety_test.py``(이미 실물에서 검증된 선례)가 쓰는
패턴을 그대로 따른다 - ``ReadOnlyRealFollowerStateSource``(읽기 전용, write 메서드
자체가 없음)로 state를 읽고, ``SO101FollowerActionWriter``(``StagedFollowerArmedWriter``를
합성 - Phase C-3B)가 감싼 실제 ``SOFollower``로만 write한다. 새 writer를 만들지 않는다 -
``SO101FollowerActionWriter``가 이미 "ACCEPT-only 판정은 호출자 책임, write 경로는 이
한 곳뿐"이라는 원칙을 갖고 있다(``follower_action_writer.py`` 모듈 docstring).

## 왜 LockedFollowerStateSource가 필요한가 (이 phase에서 직접 발견한 설계 이슈)

기존 코드(``ReadOnlySO101Reader``/``StagedFollowerArmedWriter``) 어디에도 thread-safety
lock이 없다 - 원래 있던 유일한 사용처(``StagedRealRolloutRunner``)는 완전히
순차적(step 하나씩)이라 문제가 없었다. 이번 phase는 다르다: **같은
``ReadOnlyRealFollowerStateSource`` 인스턴스를 두 개의 서로 다른 thread**가 쓴다 -
``AsyncVLAChunkInferenceWorker``(observation capture, ~3Hz)와
``RealTimeFollowerControlLoop``(state read, ~60Hz). serial I/O는 GIL을 release하므로
lock 없이 두 thread가 동시에 같은 시리얼 포트에 read를 시도하면 패킷이 섞여
깨질 수 있다 - 그래서 이 스크립트가 ``threading.Lock``으로 감싼
``LockedFollowerStateSource``를 만들어 두 소비자가 항상 하나의 lock을 거쳐 직렬화되게
한다(connection 개수 자체는 늘리지 않음 - 여전히 읽기 connection 1개).

## 이번 phase의 절대 규칙

- Claude(이 세션)는 실제 로봇을 스스로 실행하지 않는다. 아래에서 하는 검증은 전부
  ``--dry-run`` 또는 완전한 Fake(``tests/test_run_real_pick_drop_realtime_cli.py``)로만
  한다.
- ``configs/safety_gate.yaml``/``configs/follower_safe_mapper.yaml``/``/predict``/
  ``/predict_chunk``/Candidate B checkpoint - 전혀 바꾸지 않는다.
- write는 오직 ``SO101FollowerActionWriter.write()`` 한 곳에서만 일어난다(내부적으로
  ``StagedFollowerArmedWriter.write_action_once()`` -> ``follower.send_action()`` 단
  하나의 경로 - grep으로 재확인 가능).
- Hold policy는 ``NO_WRITE``로 고정한다(Phase C-3B 결론 그대로 - SO-101은
  position-controlled라 안 쓰는 것 자체가 "마지막 안전 위치 유지"와 물리적으로 동일).

## "Full Pick & Drop"에 대한 설계 해석

Candidate B는 "Pick up the cube and drop it into the bin."이라는 하나의 task로
end-to-end 학습됐다 - pick/drop을 별도 phase로 나누는 state machine이 이 저장소
어디에도 없고(이전 phase들에서도 없었다), 이 스크립트도 새로 만들지 않는다. 그래서
"Full Pick & Drop까지 수행"은 곧 "sanity window를 통과한 뒤 재시작 없이 같은 realtime
loop를 ``--max-runtime-s``까지 계속 돌린다"로 구현한다 - 정책 자체가 pick과 drop을
하나의 연속 동작으로 다룬다.

## 실행 예시

    source ~/lerobot/.venv/bin/activate

    # 1) dry-run (하드웨어/네트워크 전혀 접근 안 함 - 계획만 출력)
    python scripts/run_real_pick_drop_realtime.py --dry-run \\
      --hardware-config configs/hardware.local.json \\
      --follower-port /dev/ttyACM0 --follower-id chanho_follower

    # 2) 실제 실행 (로봇 옆에서 직접 지켜보면서)
    python scripts/run_real_pick_drop_realtime.py \\
      --hardware-config configs/hardware.local.json \\
      --follower-port /dev/ttyACM0 --follower-id chanho_follower \\
      --vla-server-url http://100.75.147.72:9200 \\
      --confirm-physically-present
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 예외 클래스/순수 검증 함수만 미리 import - 하드웨어를 건드리지 않으므로 dry-run에서도 안전.
from runtime.laptop.camera_source import CameraSourceError  # noqa: E402
from runtime.laptop.follower_state_source import FollowerStateSourceError  # noqa: E402
from scripts.run_real_follower_staged_safety_test import (  # noqa: E402
    CONFIRMATION_PHRASE,
    StagedSafetyTestError,
    _checkpoint_signature,
    _validate_hardware_config,
    _verify_desktop_checkpoint,
    require_interactive_confirmation,
)

DEFAULT_CANDIDATE_B_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "pick_drop_v3_v4_combined69" / "smolvla_pick_drop_v3_v4_combined69_uniform_fresh"
    / "checkpoints" / "010000" / "pretrained_model"
)
DEFAULT_TASK = "Pick up the cube and drop it into the bin."
DEFAULT_VLA_SERVER_URL = "http://100.75.147.72:9200"
DEFAULT_FOLLOWER_ID = "chanho_follower"
DEFAULT_CONTROL_HZ = 60.0
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "real_pick_drop_realtime_v1"

# -- 타이밍/판정 상수 (전부 근거를 docstring에 남긴다 - 임의 숫자 아님) -------------------
DEFAULT_SANITY_WINDOW_S = 1.5  # 사용자 요구사항 "최초 1~2초" 중간값
DEFAULT_MAX_RUNTIME_S = 180.0  # 3분 - pick&drop 한 사이클에 충분히 넉넉한 상한(오버라이드 가능)
DEFAULT_TRAJECTORY_WAIT_TIMEOUT_S = 10.0  # 최초 usable trajectory 대기 상한 - 이보다 오래 걸리면
# VLA 서버/체크포인트/카메라 중 하나가 비정상이라는 신호로 보고 시작하지 않는다.
FATAL_NONPRODUCTIVE_S = 5.0  # NO_TRAJECTORY/STALE_TRAJECTORY/INTENT_BLOCKED/SAFETY_BLOCKED가
# 이 시간 이상 연속되면(=이유 불문 5초간 아무것도 안 써짐) fatal stop. sanity window(1.5~2s)
# 보다 확실히 크게 잡아 "가끔 한 tick 막히는 정상 동작"을 fatal로 오판하지 않게 한다.
FATAL_CONSECUTIVE_FAULT_TICKS = 10  # ~167ms @ 60Hz - 한 번의 transient glitch는 허용하되
# 지속되는 FAULT(생성기 예외 등)는 즉시 멈춘다.
FATAL_CONSECUTIVE_STATE_READ_FAILURES = 3  # state read는 유독 하드웨어 통신 문제를 직접
# 반영하므로 fault 일반 임계값보다 훨씬 타이트하게 잡는다.
STATUS_PRINT_INTERVAL_S = 2.0
CONTROL_LOOP_POLL_INTERVAL_S = 0.1  # 이 polling은 진단/모니터링용 - 60Hz 자체는 별도 thread


class RealtimeRunError(RuntimeError):
    pass


class StopReason(str, Enum):
    NORMAL_MAX_RUNTIME = "NORMAL_MAX_RUNTIME_REACHED"
    KEYBOARD_INTERRUPT = "KEYBOARD_INTERRUPT"
    SANITY_WINDOW_FATAL = "SANITY_WINDOW_FATAL_NO_WRITE"
    NONPRODUCTIVE_FATAL = "NONPRODUCTIVE_STATE_SUSTAINED_FATAL"
    CONSECUTIVE_FAULT_FATAL = "CONSECUTIVE_FAULT_FATAL"
    WRITER_EXCEPTION_FATAL = "WRITER_EXCEPTION_FATAL"
    STATE_READ_FAILURE_FATAL = "STATE_READ_FAILURE_FATAL"
    INFERENCE_UNRECOVERABLE_FATAL = "INFERENCE_UNRECOVERABLE_FATAL"
    TRAJECTORY_NEVER_USABLE_ABORT = "TRAJECTORY_NEVER_USABLE_ABORT"  # 시작 전 abort(별도 케이스)


_NONPRODUCTIVE_STATE_NAMES = frozenset({"NO_TRAJECTORY", "STALE_TRAJECTORY", "INTENT_BLOCKED", "SAFETY_BLOCKED"})


# ---------------------------------------------------------------------------
# 접합부 1: LockedFollowerStateSource (모듈 docstring "왜 lock이 필요한가" 참고)
# ---------------------------------------------------------------------------


class LockedFollowerStateSource:
    """``FollowerStateSourceProtocol``을 그대로 만족한다 - inner를 재구현하지 않고
    lock만 씌운다."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            return self._inner.read()


# ---------------------------------------------------------------------------
# 접합부 2: RealObservationSnapshotProvider
# ---------------------------------------------------------------------------


@dataclass
class RealObservationSnapshotProvider:
    """``observation_snapshot.py``가 예고한 접합부 - camera_source/state_source를
    그대로 호출만 한다(새 read 경로 없음)."""

    camera_source: object  # RealCameraObservationSource
    state_source: object  # LockedFollowerStateSource(ReadOnlyRealFollowerStateSource)
    task: str
    monotonic_fn: Callable[[], float] = time.monotonic

    def capture(self, *, sequence: int):
        from runtime.laptop.observation_snapshot import ObservationSnapshot

        t0 = self.monotonic_fn()
        frames = self.camera_source.capture_all()
        images = {key: frame.image_rgb for key, frame in frames.items()}
        state_snapshot = self.state_source.read()
        return ObservationSnapshot(
            images=images, state=state_snapshot.positions_deg, task=self.task,
            capture_monotonic_time=t0, sequence=sequence,
        )


# ---------------------------------------------------------------------------
# Recording proxy (진단 전용, C-4와 동일 기법 - shipped generator 재구현 안 함)
# ---------------------------------------------------------------------------


class RecordingGeneratorProxy:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.results: list = []

    def tick(self, **kwargs):
        result = self._inner.tick(**kwargs)
        self.results.append(result)
        return result


# ---------------------------------------------------------------------------
# SessionReport (섹션 10 요구사항 필드)
# ---------------------------------------------------------------------------


@dataclass
class SessionReport:
    stop_reason: str
    runtime_duration_s: float
    control: dict
    inference: dict
    trajectory: dict
    intent: dict
    motion_guard: dict
    final_safety: dict
    writer: dict
    quarantined_sequences: list

    def to_dict(self) -> dict:
        return {
            "stop_reason": self.stop_reason, "runtime_duration_s": self.runtime_duration_s,
            "control": self.control, "inference": self.inference, "trajectory": self.trajectory,
            "intent": self.intent, "motion_guard": self.motion_guard, "final_safety": self.final_safety,
            "writer": self.writer, "quarantined_sequences": self.quarantined_sequences,
        }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct / 100.0 * (len(s) - 1))))
    return s[idx]


# ---------------------------------------------------------------------------
# RealtimeSessionOrchestrator - 실 하드웨어 없이도 테스트 가능한 핵심 로직
# (tests/test_run_real_pick_drop_realtime_cli.py가 전부 Fake로 이 클래스를 검증한다)
# ---------------------------------------------------------------------------


class RealtimeSessionOrchestrator:
    """``main()``이 이미 connect/open까지 끝낸 구성요소를 받아, 시작 절차(6~8)부터
    sanity window/fatal 감시/종료/report를 담당한다. 이 클래스 자체는 하드웨어를 몰라도
    되게 설계했다 - 전부 이미 주입된 협력 객체(Fake 가능)로만 동작한다."""

    def __init__(
        self,
        *,
        observation_provider,
        state_source,
        writer,
        vla_client,
        safety_gate,
        motion_limits,
        session_id: str,
        task: str,
        control_hz: float = DEFAULT_CONTROL_HZ,
        sanity_window_s: float = DEFAULT_SANITY_WINDOW_S,
        max_runtime_s: float = DEFAULT_MAX_RUNTIME_S,
        trajectory_wait_timeout_s: float = DEFAULT_TRAJECTORY_WAIT_TIMEOUT_S,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        from runtime.laptop.async_chunk_inference_worker import AsyncVLAChunkInferenceWorker
        from runtime.laptop.motion_guard import DEFAULT_JOINT_MOTION_LIMITS
        from runtime.laptop.realtime_control_loop import (
            HoldPolicy,
            RealTimeFollowerControlLoop,
            RealTimeFollowerControlLoopConfig,
        )
        from runtime.laptop.realtime_control_target import RealTimeControlTargetGenerator
        from runtime.laptop.temporal_ensemble import TemporalEnsembler
        from runtime.laptop.trajectory_buffer import TrajectoryBuffer

        self._monotonic = monotonic_fn
        self._sleep = sleep_fn
        self._print = print_fn
        self._sanity_window_s = sanity_window_s
        self._max_runtime_s = max_runtime_s
        self._trajectory_wait_timeout_s = trajectory_wait_timeout_s
        self._session_id = session_id
        self._writer = writer

        self._buffer = TrajectoryBuffer(max_chunks=4)
        self._worker = AsyncVLAChunkInferenceWorker(
            vla_client=vla_client, observation_provider=observation_provider, buffer=self._buffer,
            session_id=session_id, task=task, min_interval_s=0.0,
        )
        real_generator = RealTimeControlTargetGenerator(
            ensembler=TemporalEnsembler(half_life_s=0.338, max_contributors=3), safety_gate=safety_gate,
            motion_limits=motion_limits or DEFAULT_JOINT_MOTION_LIMITS, control_hz=control_hz,
        )
        self._gen_proxy = RecordingGeneratorProxy(real_generator)
        # session_write_cap: "무제한"이 아니라 이번 세션 길이에 맞춘 명시적 상한(3배 여유) -
        # follower_action_writer.py의 circuit-breaker 철학을 이 세션 규모에 맞게 재적용.
        self._session_write_cap = max(1000, int(max_runtime_s * control_hz * 3))
        self._loop = RealTimeFollowerControlLoop(
            generator=self._gen_proxy, safety_gate=safety_gate, trajectory_buffer=self._buffer,
            state_source=state_source, writer=writer,
            config=RealTimeFollowerControlLoopConfig(control_hz=control_hz, hold_policy=HoldPolicy.NO_WRITE),
            health_source=self._worker.health_snapshot, monotonic_fn=monotonic_fn,
        )

    # -- 시작 절차 6~8 -----------------------------------------------------------------

    def _wait_for_usable_trajectory(self) -> bool:
        deadline = self._monotonic() + self._trajectory_wait_timeout_s
        while self._monotonic() < deadline:
            if self._buffer.valid_chunks(self._monotonic()):
                return True
            self._sleep(0.05)
        return bool(self._buffer.valid_chunks(self._monotonic()))

    def run(self) -> SessionReport:
        self._print(f"[시작] async inference worker 시작 (session_id={self._session_id})")
        self._worker.start()

        self._print(f"[시작] 최초 usable trajectory 대기 (최대 {self._trajectory_wait_timeout_s:.1f}s)...")
        if not self._wait_for_usable_trajectory():
            self._worker.stop()
            health = self._worker.health_snapshot()
            raise RealtimeRunError(
                f"{self._trajectory_wait_timeout_s:.1f}s 안에 usable trajectory가 생기지 않았습니다 - "
                f"VLA 서버/체크포인트/카메라를 확인하세요 (worker health: "
                f"consecutive_failures={health.consecutive_failures}, last_error={health.last_error!r})."
            )
        self._print("[시작] 최초 trajectory 확보 완료 - realtime control loop 시작")

        self._loop.start()
        start_time = self._monotonic()
        stop_reason = self._run_monitor_loop(start_time)
        report = self._shutdown_and_report(stop_reason, self._monotonic() - start_time)
        return report

    # -- sanity window + fatal 감시 + 주기 상태 출력 ------------------------------------

    def _run_monitor_loop(self, start_time: float) -> StopReason:
        sanity_checked = False
        last_status_print = start_time
        nonproductive_since: float | None = None
        consecutive_fault_ticks = 0
        consecutive_state_read_failures = 0
        last_inference_success = self._worker.health_snapshot().last_success_time_monotonic or start_time

        try:
            while True:
                now = self._monotonic()
                elapsed = now - start_time

                if not sanity_checked and elapsed >= self._sanity_window_s:
                    sanity_checked = True
                    if not self._print_sanity_report(elapsed):
                        return StopReason.SANITY_WINDOW_FATAL
                    self._print("[sanity] 정상 - 재시작 없이 계속 진행합니다.")

                if now - last_status_print >= STATUS_PRINT_INTERVAL_S:
                    self._print_periodic_status(elapsed)
                    last_status_print = now

                records = self._loop.tick_history()
                recent = records[-1] if records else None
                if recent is not None:
                    if recent.state.value == "FAULT":
                        if recent.stop_reason == "WRITER_FAULT":
                            self._print(f"[FATAL] writer 예외 발생: {recent.errors}")
                            return StopReason.WRITER_EXCEPTION_FATAL
                        if any("state_read" in e for e in recent.errors):
                            consecutive_state_read_failures += 1
                            consecutive_fault_ticks = 0
                            if consecutive_state_read_failures >= FATAL_CONSECUTIVE_STATE_READ_FAILURES:
                                self._print(f"[FATAL] state read {consecutive_state_read_failures}회 연속 실패")
                                return StopReason.STATE_READ_FAILURE_FATAL
                        else:
                            consecutive_fault_ticks += 1
                            consecutive_state_read_failures = 0
                            if consecutive_fault_ticks >= FATAL_CONSECUTIVE_FAULT_TICKS:
                                self._print(f"[FATAL] {consecutive_fault_ticks}tick 연속 FAULT")
                                return StopReason.CONSECUTIVE_FAULT_FATAL
                    else:
                        consecutive_fault_ticks = 0
                        consecutive_state_read_failures = 0

                    if recent.state.value in _NONPRODUCTIVE_STATE_NAMES:
                        if nonproductive_since is None:
                            nonproductive_since = now
                        elif now - nonproductive_since >= FATAL_NONPRODUCTIVE_S:
                            self._print(
                                f"[FATAL] {FATAL_NONPRODUCTIVE_S:.1f}s 이상 non-productive 상태 지속"
                                f"(state={recent.state.value})"
                            )
                            return StopReason.NONPRODUCTIVE_FATAL
                    else:
                        nonproductive_since = None

                health = self._worker.health_snapshot()
                if health.last_success_time_monotonic is not None:
                    last_inference_success = health.last_success_time_monotonic
                if now - last_inference_success >= FATAL_NONPRODUCTIVE_S * 3:  # 15s
                    self._print(f"[FATAL] {now - last_inference_success:.1f}s 동안 inference 성공 없음")
                    return StopReason.INFERENCE_UNRECOVERABLE_FATAL

                if elapsed >= self._max_runtime_s:
                    self._print(f"[정상 종료] max_runtime_s({self._max_runtime_s:.1f}s) 도달")
                    return StopReason.NORMAL_MAX_RUNTIME

                self._sleep(CONTROL_LOOP_POLL_INTERVAL_S)
        except KeyboardInterrupt:
            self._print("\n[중단] Ctrl+C 감지")
            return StopReason.KEYBOARD_INTERRUPT

    def _print_sanity_report(self, elapsed: float) -> bool:
        stats = self._loop.stats()
        write_count = self._writer.write_count if hasattr(self._writer, "write_count") else None
        self._print(f"\n--- sanity window ({elapsed:.2f}s) ---")
        self._print(f"  control Hz        = {stats.actual_hz}")
        self._print(f"  jitter(ms)        = {stats.jitter_ms}")
        self._print(f"  deadline misses   = {stats.deadline_miss_count}/{stats.n_ticks}")
        self._print(f"  writer count      = {write_count}")
        results = self._gen_proxy.results
        intent = [r.intent_decision for r in results if r.intent_decision is not None]
        safety = [r.safety_decision for r in results if r.safety_decision is not None]
        self._print(f"  Intent ACCEPT/BLOCK = {intent.count('ACCEPT')}/{len(intent) - intent.count('ACCEPT')} (n={len(intent)})")
        self._print(f"  Final Safety ACCEPT = {safety.count('ACCEPT')}/{len(safety)}")
        ok = bool(write_count and write_count > 0)
        if not ok:
            self._print(f"  [FATAL] sanity window 동안 write가 한 번도 없었습니다.")
        return ok

    def _print_periodic_status(self, elapsed: float) -> None:
        stats = self._loop.stats()
        write_count = self._writer.write_count if hasattr(self._writer, "write_count") else None
        self._print(
            f"[{elapsed:6.1f}s] state={self._loop.state.value:16s} Hz={stats.actual_hz} "
            f"writes={write_count} deadline_miss={stats.deadline_miss_count} "
            f"quarantined={len(self._loop.quarantined_sequences)}"
        )

    # -- 종료 (섹션 8: control loop stop -> worker stop -> follower/camera는 main()이 처리) --

    def _shutdown_and_report(self, stop_reason: StopReason, duration_s: float) -> SessionReport:
        self._print(f"\n[종료 절차] stop_reason={stop_reason.value}")
        try:
            self._loop.stop()
        except Exception as exc:  # noqa: BLE001
            self._print(f"[경고] control loop stop 중 예외: {exc}")
        try:
            self._worker.stop()
        except Exception as exc:  # noqa: BLE001
            self._print(f"[경고] inference worker stop 중 예외: {exc}")

        return self._build_report(stop_reason, duration_s)

    def _build_report(self, stop_reason: StopReason, duration_s: float) -> SessionReport:
        stats = self._loop.stats()
        health = self._worker.health_snapshot()
        tick_history = self._loop.tick_history()
        results = self._gen_proxy.results

        n_results = len(results)
        n_usable = sum(1 for r in results if r.raw_ensemble_target is not None)
        n_no_target = sum(1 for r in results if r.stop_reason == "NO_TARGET")
        n_stale = sum(1 for r in results if r.stop_reason == "STALE_TRAJECTORY")

        intent = [r.intent_decision for r in results if r.intent_decision is not None]
        safety = [r.safety_decision for r in results if r.safety_decision is not None]

        guard_rows = [r for r in results if r.raw_ensemble_target is not None and r.guarded_target is not None]
        joint_names = list(guard_rows[0].raw_ensemble_target.keys()) if guard_rows else []
        motion_guard = {}
        for j in joint_names:
            deltas = [abs(r.raw_ensemble_target[j] - r.guarded_target[j]) for r in guard_rows]
            motion_guard[j] = {
                "activation_rate": (sum(1 for d in deltas if d > 1e-6) / len(deltas)) if deltas else None,
                "tracking_lag_mean": (sum(deltas) / len(deltas)) if deltas else None,
                "tracking_lag_p95": _percentile(deltas, 95),
            }

        writer_count = self._writer.write_count if hasattr(self._writer, "write_count") else None

        report = SessionReport(
            stop_reason=stop_reason.value,
            runtime_duration_s=duration_s,
            control={
                "n_ticks": stats.n_ticks, "actual_hz": stats.actual_hz, "median_period_ms": stats.median_period_ms,
                "p95_period_ms": stats.p95_period_ms, "jitter_ms": stats.jitter_ms,
                "deadline_miss_count": stats.deadline_miss_count, "deadline_miss_rate": stats.deadline_miss_rate,
                "max_overrun_ms": stats.max_overrun_ms, "resync_count": stats.resync_count,
            },
            inference={
                "total_requests": health.total_requests, "total_published": health.total_published,
                "total_discarded_stale": health.total_discarded_stale, "consecutive_failures": health.consecutive_failures,
                "last_error": health.last_error, "latest_sequence": health.latest_sequence,
            },
            trajectory={
                "n_ticks_seen": n_results,
                "usable_fraction": (n_usable / n_results) if n_results else None,
                "no_target_fraction": (n_no_target / n_results) if n_results else None,
                "stale_fraction": (n_stale / n_results) if n_results else None,
            },
            intent={"n": len(intent), "accept": intent.count("ACCEPT"), "would_clamp": intent.count("WOULD_CLAMP"), "reject": intent.count("REJECT")},
            motion_guard=motion_guard,
            final_safety={"n": len(safety), "accept": safety.count("ACCEPT"), "would_clamp": safety.count("WOULD_CLAMP"), "reject": safety.count("REJECT")},
            writer={"write_count": writer_count},
            quarantined_sequences=sorted(self._loop.quarantined_sequences),
        )
        self._print("\n" + json.dumps(report.to_dict(), indent=2, default=str))
        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vla-server-url", default=DEFAULT_VLA_SERVER_URL)
    p.add_argument("--follower-port", required=True)
    p.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    p.add_argument("--follower-calibration-path", default=None)
    p.add_argument("--hardware-config", type=Path, required=True)
    p.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CANDIDATE_B_CHECKPOINT, help="Desktop이 로딩했어야 할 checkpoint(검증용, 로딩하지 않음)")
    p.add_argument("--vla-timeout-s", type=float, default=15.0)
    p.add_argument("--vla-api-token", default=None)
    p.add_argument("--force-checkpoint-mismatch", action="store_true")
    p.add_argument("--sanity-window-s", type=float, default=DEFAULT_SANITY_WINDOW_S)
    p.add_argument("--max-runtime-s", type=float, default=DEFAULT_MAX_RUNTIME_S)
    p.add_argument("--trajectory-wait-timeout-s", type=float, default=DEFAULT_TRAJECTORY_WAIT_TIMEOUT_S)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--dry-run", action="store_true", help="하드웨어/네트워크에 전혀 접근하지 않고 계획만 출력")
    p.add_argument("--confirm-physically-present", action="store_true")
    return p.parse_args(argv)


def print_banner(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("MODE=REAL_PICK_DROP_REALTIME (Phase C-5)")
    print(f"VLA_SERVER_URL={args.vla_server_url}")
    print(f"CHECKPOINT={args.checkpoint} (Desktop이 이 checkpoint를 로딩했는지 /health로 검증됨)")
    print(f"TASK={args.task!r}")
    print(f"FOLLOWER_PORT={args.follower_port}  FOLLOWER_ID={args.follower_id}")
    print(f"CONTROL_HZ={args.control_hz}  MAX_RUNTIME_S={args.max_runtime_s}  SANITY_WINDOW_S={args.sanity_window_s}")
    print("SAFETY_GATE=UNCHANGED · HOLD_POLICY=NO_WRITE · WRITE_PATH=SO101FollowerActionWriter 단 하나")
    print(f"DRY_RUN={args.dry_run}")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if not args.dry_run:
            _validate_hardware_config(args.hardware_config)
    except StagedSafetyTestError as exc:
        print(f"[오류] {exc}")
        return 2

    print_banner(args)

    if args.dry_run:
        print("\n[dry-run] 하드웨어/서버에 전혀 접근하지 않았습니다.")
        print(f"[dry-run] 실제 실행 시 {args.vla_server_url}/health를 먼저 확인하고, "
              f"checkpoint 서명({_checkpoint_signature(args.checkpoint)!r})이 일치하지 않으면 시작하지 않습니다.")
        print("[dry-run] --confirm-physically-present와 실행 중 타이핑 확인이 추가로 필요합니다.")
        return 0

    if not args.confirm_physically_present:
        print("[오류] --dry-run이 아니면 --confirm-physically-present가 필수입니다.")
        return 2

    camera_source = None
    raw_state_source = None
    writer = None
    try:
        require_interactive_confirmation()

        # 여기서부터만 하드웨어/lerobot 관련 모듈을 import한다 (dry-run은 여기 안 옴).
        from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower

        from hardware.safety.staged_follower_writer import StagedFollowerArmedWriter  # noqa: F401 (SO101FollowerActionWriter가 내부에서 씀)
        from runtime.laptop.camera_source import RealCameraObservationSource
        from runtime.laptop.follower_action_writer import SO101FollowerActionWriter
        from runtime.laptop.follower_state_source import ReadOnlyRealFollowerStateSource
        from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
        from runtime.laptop.vla_client import VLAClientConfig, VLAHttpClient

        print(f"[준비] Desktop VLA 서버 health 확인 중: {args.vla_server_url}")
        vla_client = VLAHttpClient(VLAClientConfig(server_url=args.vla_server_url, timeout_s=args.vla_timeout_s, api_token=args.vla_api_token))
        health = vla_client.check_health()
        if not health.ok:
            raise RealtimeRunError(f"Desktop VLA 서버가 정상이 아닙니다 - 시작하지 않습니다. (status={health.status}, round_trip_ms={health.round_trip_ms})")
        print(f"[준비] health OK: backend={health.backend} model_id={health.model_id} device={health.device}")
        _verify_desktop_checkpoint(health=health, expected_checkpoint=args.checkpoint, force=args.force_checkpoint_mismatch)

        camera_source = RealCameraObservationSource.from_hardware_config_path(str(args.hardware_config))
        camera_source.open()
        print("[준비] 카메라 오픈 완료 (workspace/wrist)")

        raw_state_source = ReadOnlyRealFollowerStateSource.from_port(
            port=args.follower_port, follower_id=args.follower_id, calibration_path=args.follower_calibration_path,
        )
        raw_state_source.connect()
        state_source = LockedFollowerStateSource(raw_state_source)
        initial_state = state_source.read()
        print(f"[준비] follower state 읽기 확인: {initial_state.positions_deg}")

        safety_gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
        source_summary = safety_gate.config.source_summary()
        if source_summary.get("uses_calibration_fallback"):
            print("[경고] Safety Gate가 fallback 범위를 쓰고 있습니다 - 실제 캘리브레이션 파일 존재 여부를 다시 확인하세요.")

        follower_config = SO101FollowerConfig(port=args.follower_port, id=args.follower_id, cameras={}, disable_torque_on_disconnect=True)
        follower = SOFollower(follower_config)
        writer = SO101FollowerActionWriter(follower=follower)
        writer.connect()
        print(f"[준비] follower 연결 완료 (write 경로): port={args.follower_port} id={args.follower_id}")

        observation_provider = RealObservationSnapshotProvider(camera_source=camera_source, state_source=state_source, task=args.task)

        orchestrator = RealtimeSessionOrchestrator(
            observation_provider=observation_provider, state_source=state_source, writer=writer, vla_client=vla_client,
            safety_gate=safety_gate, motion_limits=None, session_id="c5-real-pick-drop", task=args.task,
            control_hz=args.control_hz, sanity_window_s=args.sanity_window_s, max_runtime_s=args.max_runtime_s,
            trajectory_wait_timeout_s=args.trajectory_wait_timeout_s,
        )
        report = orchestrator.run()

        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.report_dir / f"session_{int(time.time())}.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"\n리포트 저장: {report_path}")
        return 0

    except (CameraSourceError, FollowerStateSourceError, StagedSafetyTestError, RealtimeRunError) as exc:
        print(f"[오류] {exc}")
        return 2
    finally:
        if writer is not None:
            try:
                writer.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if raw_state_source is not None:
            try:
                raw_state_source.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if camera_source is not None:
            try:
                camera_source.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
