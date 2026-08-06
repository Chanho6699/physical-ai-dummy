"""백그라운드 폴링 루프 + 최신 상태 캐시 + API 응답 조립.

HTTP 요청마다 하드웨어를 다시 읽지 않는다. 서버 시작 시 리더/팔로워 reader를 연결한 뒤,
단일 백그라운드 스레드가 주기적으로(기본 30Hz) 두 팔을 순서대로 읽어 캐시에 저장한다.
``GET /state``, ``GET /health``는 이 캐시를 즉시 반환한다 (요청 시점에 serial I/O를
수행하지 않는다).

읽기가 실패해도 마지막 정상값과 타임스탬프를 보존하고 ``stale``로 표시할 뿐, 거짓 정상값을
만들지 않는다.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from hardware.state_server.readonly_so101_reader import ReadOnlySO101Reader
from hardware.state_server.state_models import (
    MODE_READ_ONLY,
    ArmPollState,
    ArmSample,
    ArmView,
    validate_positions_deg,
)

logger = logging.getLogger(__name__)

# 이벤트 콜백 시그니처: (event, arm_name, consecutive_errors, message)
LogCallback = Callable[[str, str, int, str], None]

# 연결 진행 상황 콜백 시그니처: (label, phase, error_message)
ConnectCallback = Callable[[str, str, str | None], None]


def _noop_log(event: str, arm_name: str, count: int, message: str) -> None:
    return None


@dataclass(frozen=True)
class StateSnapshot:
    timestamp: float
    sequence: int
    mode: str
    leader: ArmView
    follower: ArmView
    difference_deg: dict[str, float]
    warnings: list[str]


@dataclass(frozen=True)
class HealthSnapshot:
    status: str  # "ok" | "degraded"
    mode: str
    leader_connected: bool
    follower_connected: bool
    write_enabled: bool
    timestamp: float
    errors: list[str]


@dataclass
class _ArmRuntime:
    reader: ReadOnlySO101Reader
    poll: ArmPollState = field(default_factory=ArmPollState)


class StatePoller:
    """리더/팔로워 reader를 소유하고, 단일 스레드에서 주기적으로 읽어 캐시한다."""

    def __init__(
        self,
        *,
        leader_reader: ReadOnlySO101Reader,
        follower_reader: ReadOnlySO101Reader,
        rate_hz: float,
        stale_after_ms: float,
        max_read_errors: int,
        on_log: LogCallback | None = None,
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz는 0보다 커야 합니다.")
        if stale_after_ms <= 0:
            raise ValueError("stale_after_ms는 0보다 커야 합니다.")
        if max_read_errors <= 0:
            raise ValueError("max_read_errors는 0보다 커야 합니다.")

        self._rate_hz = rate_hz
        self._stale_after_ms = stale_after_ms
        self._max_read_errors = max_read_errors
        self._on_log = on_log or _noop_log

        self._leader = _ArmRuntime(reader=leader_reader)
        self._follower = _ArmRuntime(reader=follower_reader)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self.sample_count = 0

    # -- 연결/해제 -----------------------------------------------------

    def connect_all(self, *, on_event: ConnectCallback | None = None) -> list[str]:
        """가능한 만큼 연결을 시도하고, 실패한 팔의 한글 오류 메시지 목록을 반환한다.

        하나의 팔이 실패해도 다른 팔 연결은 계속 시도한다 (부분 연결 허용). ``on_event``가
        주어지면 ``(label, phase, error_message)``로 진행 상황을 알린다
        (``phase`` in {"connecting", "connected", "failed"}) - 콘솔 출력은 호출자
        (scripts/run_hardware_state_server.py)가 담당한다.
        """

        errors: list[str] = []
        for runtime, label in ((self._leader, "리더암"), (self._follower, "팔로워암")):
            if on_event:
                on_event(label, "connecting", None)
            try:
                runtime.reader.connect()
                with self._lock:
                    runtime.poll.connected = True
                if on_event:
                    on_event(label, "connected", None)
            except Exception as exc:  # noqa: BLE001 - 하드웨어 연결 실패를 폭넓게 잡아 보고한다
                with self._lock:
                    runtime.poll.connected = False
                    runtime.poll.last_error = str(exc)
                message = str(exc)
                errors.append(f"{label} 연결 실패: {message}")
                logger.warning("%s 연결 실패: %s", label, message)
                if on_event:
                    on_event(label, "failed", message)
        return errors

    def disconnect_all(self) -> list[str]:
        """모든 reader의 연결을 해제한다. 오류가 발생해도 나머지는 계속 시도하고,
        오류 메시지 목록을 반환한다 (종료 중 오류를 숨기지 않기 위함)."""

        errors: list[str] = []
        for runtime, label in ((self._leader, "리더암"), (self._follower, "팔로워암")):
            try:
                runtime.reader.disconnect()
            except Exception as exc:  # noqa: BLE001 - 종료 중 오류를 숨기지 않고 기록한다
                message = f"{label} disconnect 중 오류: {exc}"
                errors.append(message)
                logger.error(message)
            finally:
                with self._lock:
                    runtime.poll.connected = False
        return errors

    # -- 폴링 스레드 -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("StatePoller는 이미 시작되었습니다.")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="so101-state-poller", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        period_s = 1.0 / self._rate_hz
        while not self._stop_event.is_set():
            tick_start = time.monotonic()
            self.poll_once()
            elapsed = time.monotonic() - tick_start
            self._stop_event.wait(max(0.0, period_s - elapsed))

    def poll_once(self) -> None:
        """리더 -> 팔로워 순서로 한 번씩 읽고 시퀀스를 1 증가시킨다. 테스트에서 직접 호출 가능."""

        self._poll_arm("leader", self._leader, "리더암")
        self._poll_arm("follower", self._follower, "팔로워암")
        with self._lock:
            self._sequence += 1
            self.sample_count = self._sequence

    def _poll_arm(self, key: str, runtime: _ArmRuntime, label: str) -> None:
        try:
            if not runtime.reader.is_connected:
                raise RuntimeError(f"{label} 연결이 끊어져 있습니다.")

            positions = runtime.reader.read_positions()
            validate_positions_deg(positions)
            try:
                raw = runtime.reader.read_raw_positions()
            except Exception as raw_exc:  # noqa: BLE001
                # raw tick 읽기 실패는 degree 값 자체는 유효하므로 raw만 None 처리한다.
                logger.debug("%s raw tick 읽기 실패 (state는 유지): %s", label, raw_exc)
                raw = None

            sample = ArmSample(positions_deg=positions, raw_ticks=raw, timestamp=time.time())
            with self._lock:
                had_errors = runtime.poll.consecutive_errors > 0
                runtime.poll.last_good = sample
                runtime.poll.consecutive_errors = 0
                runtime.poll.last_error = None
                runtime.poll.connected = True
            if had_errors:
                self._on_log("recovered", key, 0, "")
        except Exception as exc:  # noqa: BLE001 - 통신 오류/JointReadError를 모두 흡수
            with self._lock:
                runtime.poll.consecutive_errors += 1
                runtime.poll.last_error = str(exc)
                count = runtime.poll.consecutive_errors
            self._on_log("read_error", key, count, str(exc))
            if count == self._max_read_errors:
                self._on_log("degraded_threshold", key, count, str(exc))

    # -- 스냅샷 조립 -----------------------------------------------------

    def _arm_view(self, runtime: _ArmRuntime) -> ArmView:
        with self._lock:
            poll = runtime.poll
            last_good = poll.last_good
            connected = poll.connected
            error_count = poll.consecutive_errors
            last_error = poll.last_error

        if last_good is None:
            return ArmView(
                connected=connected,
                positions_deg=None,
                raw_ticks=None,
                stale=True,
                age_ms=None,
                read_error_count=error_count,
                last_error=last_error,
            )

        age_ms = max(0.0, (time.time() - last_good.timestamp) * 1000.0)
        stale = age_ms > self._stale_after_ms or error_count > 0
        return ArmView(
            connected=connected,
            positions_deg=last_good.positions_deg,
            raw_ticks=last_good.raw_ticks,
            stale=stale,
            age_ms=age_ms,
            read_error_count=error_count,
            last_error=last_error,
        )

    def snapshot(self) -> StateSnapshot:
        leader_view = self._arm_view(self._leader)
        follower_view = self._arm_view(self._follower)

        difference_deg: dict[str, float] = {}
        warnings: list[str] = []

        if leader_view.positions_deg is not None and follower_view.positions_deg is not None:
            difference_deg = {
                joint: leader_view.positions_deg[joint] - follower_view.positions_deg[joint]
                for joint in leader_view.positions_deg
            }
        else:
            warnings.append("리더 또는 팔로워 값이 없어 difference_deg를 계산할 수 없습니다.")

        if leader_view.stale:
            warnings.append("리더암 상태가 stale합니다 (마지막 정상값을 유지 중).")
        if follower_view.stale:
            warnings.append("팔로워암 상태가 stale합니다 (마지막 정상값을 유지 중).")
        if leader_view.read_error_count >= self._max_read_errors:
            warnings.append(f"리더암 state 읽기 {leader_view.read_error_count}회 연속 실패.")
        if follower_view.read_error_count >= self._max_read_errors:
            warnings.append(f"팔로워암 state 읽기 {follower_view.read_error_count}회 연속 실패.")

        with self._lock:
            sequence = self._sequence

        return StateSnapshot(
            timestamp=time.time(),
            sequence=sequence,
            mode=MODE_READ_ONLY,
            leader=leader_view,
            follower=follower_view,
            difference_deg=difference_deg,
            warnings=warnings,
        )

    def health(self) -> HealthSnapshot:
        leader_view = self._arm_view(self._leader)
        follower_view = self._arm_view(self._follower)

        errors: list[str] = []
        if not leader_view.connected:
            errors.append("리더암 연결 실패")
        elif leader_view.read_error_count >= self._max_read_errors:
            errors.append(f"리더암 state 읽기 {leader_view.read_error_count}회 연속 실패")

        if not follower_view.connected:
            errors.append("팔로워암 연결 실패")
        elif follower_view.read_error_count >= self._max_read_errors:
            errors.append(f"팔로워암 state 읽기 {follower_view.read_error_count}회 연속 실패")

        status = "ok" if not errors and not leader_view.stale and not follower_view.stale else "degraded"

        return HealthSnapshot(
            status=status,
            mode=MODE_READ_ONLY,
            leader_connected=leader_view.connected,
            follower_connected=follower_view.connected,
            write_enabled=False,
            timestamp=time.time(),
            errors=errors,
        )
