"""노트북의 SO-101 읽기 전용 상태 서버(GET /health, /state, /calibration)에 접속하는 클라이언트.

이 모듈은 **GET 메서드만** 구현한다. POST/PUT/PATCH/DELETE는 물론
``/action``·``/move``·``/command``·``/teleop`` 같은 제어 경로를 호출하는 메서드는
존재하지 않는다 (요구사항: 노트북에 명령을 보내지 않는다). ``requests.Session``에도
GET 요청만 사용한다 - 코드 리뷰로 바로 확인할 수 있도록 이 파일 안에서 HTTP 메서드
호출은 ``session.get(...)`` 한 곳뿐이다.

원격 서버가 보낸 값은 신뢰할 수 없는 입력으로 취급한다. 다음을 모두 방어적으로 처리한다:

- 연결 실패 / timeout (제한된 횟수만 재시도, 무한 재시도 없음)
- malformed JSON (최상위가 object가 아니거나 파싱 자체가 실패)
- 관절 값 누락 / NaN / Inf (해당 팔의 positions_deg를 무효로 표시하고 이유를 기록할 뿐,
  예외를 던지지 않는다 - 호출자가 "그 팔만 무효" 상태로 안전하게 판단할 수 있게 하기 위함)
- stale 상태, sequence 정지, timestamp 나이 초과는 :class:`StaleGuard` /
  :class:`SequenceWatchdog` 헬퍼로 별도 판정한다 (호출 시점 스냅샷만으로는 "정지"
  여부를 알 수 없기 때문에 시간 경과를 추적하는 별도 상태가 필요하다).

API 토큰은 생성자에만 전달하고, ``repr``/로그/리포트에는 절대 출력하지 않는다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import requests

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

READ_ONLY_MODE = "read_only"


class RemoteStateError(RuntimeError):
    """원격 상태 서버 접속/응답 파싱 실패 (연결 오류, timeout, HTTP 오류, malformed JSON 등)."""


@dataclass(frozen=True)
class RemoteClientConfig:
    """원격 클라이언트 설정. configs/remote_mujoco_diagnostic.yaml의 ``remote`` 섹션과 대응된다."""

    server_url: str
    timeout_ms: float = 500.0
    max_retries: int = 3
    retry_backoff_s: float = 0.05
    api_token: str | None = None

    def __post_init__(self) -> None:
        if not self.server_url:
            raise ValueError("server_url이 비어 있습니다.")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms는 0보다 커야 합니다.")
        if self.max_retries < 1:
            raise ValueError("max_retries는 1 이상이어야 합니다 (무한 재시도는 지원하지 않습니다).")

    def __repr__(self) -> str:  # api_token이 로그/예외 메시지에 노출되지 않도록 직접 정의
        token_state = "설정됨" if self.api_token else "없음"
        return (
            f"RemoteClientConfig(server_url={self.server_url!r}, timeout_ms={self.timeout_ms}, "
            f"max_retries={self.max_retries}, api_token={token_state})"
        )


@dataclass(frozen=True)
class ArmStateView:
    """``/state`` 응답의 leader 또는 follower 한 쪽을 검증/정규화한 결과."""

    connected: bool
    positions_deg: dict[str, float] | None  # None이면 누락/NaN/Inf 등으로 무효
    raw_ticks: dict[str, int] | None
    stale: bool  # 서버가 보고한 stale 플래그 (없으면 age_ms 등으로 보수적으로 True 처리)
    age_ms: float | None
    valid: bool  # positions_deg가 6개 관절 + 유한값 검증을 통과했는지
    invalid_reason: str | None


@dataclass(frozen=True)
class RemoteState:
    """``GET /state`` 응답을 검증한 결과 + 클라이언트가 측정한 네트워크 지연."""

    raw_timestamp: float | None
    sequence: int | None
    mode: str | None
    leader: ArmStateView
    follower: ArmStateView
    difference_deg: dict[str, float]
    warnings: list[str]
    received_at_monotonic: float
    received_at_wall: float
    network_latency_ms: float

    @property
    def state_age_ms(self) -> float | None:
        """비교 가능한 age_ms 중 더 오래된(큰) 값. 클럭 스큐 없이 서버가 직접 계산한 값이라
        신뢰도가 높다. 둘 다 없으면 None (호출자가 network_latency_ms로 대체 판단해야 함)."""
        candidates = [v for v in (self.leader.age_ms, self.follower.age_ms) if v is not None]
        return max(candidates) if candidates else None


@dataclass(frozen=True)
class HealthState:
    status: str | None  # "ok" | "degraded" | None(알 수 없음)
    mode: str | None
    leader_connected: bool
    follower_connected: bool
    write_enabled: bool | None  # None이면 서버 응답에 필드가 없었다는 뜻 - 안전하게 "확인 불가"로 취급
    timestamp: float | None
    errors: list[str]


@dataclass(frozen=True)
class CalibrationEntry:
    homing_offset: int | None
    range_min: int | None
    range_max: int | None


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        fv = float(value)
        return fv if math.isfinite(fv) else None
    return None


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _validate_positions(raw: object) -> tuple[dict[str, float] | None, str | None]:
    if raw is None:
        return None, "positions_deg 필드가 없습니다."
    if not isinstance(raw, dict):
        return None, f"positions_deg가 object가 아닙니다: {type(raw)!r}"

    missing = [joint for joint in JOINT_NAMES if joint not in raw]
    if missing:
        return None, f"관절 값이 누락되었습니다: {missing}"

    result: dict[str, float] = {}
    for joint in JOINT_NAMES:
        value = raw[joint]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"'{joint}' 값이 숫자가 아닙니다: {value!r}"
        fvalue = float(value)
        if math.isnan(fvalue) or math.isinf(fvalue):
            return None, f"'{joint}' 값이 NaN/Inf입니다: {value!r}"
        result[joint] = fvalue
    return result, None


def _safe_int_dict(raw: object) -> dict[str, int] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    result: dict[str, int] = {}
    for joint in JOINT_NAMES:
        if joint not in raw:
            continue
        value = raw[joint]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        result[joint] = int(value)
    return result or None


def _parse_arm(raw: object) -> ArmStateView:
    if not isinstance(raw, dict):
        return ArmStateView(
            connected=False,
            positions_deg=None,
            raw_ticks=None,
            stale=True,
            age_ms=None,
            valid=False,
            invalid_reason="응답에 팔 상태(object)가 없습니다.",
        )

    connected = bool(raw.get("connected", False))
    server_stale = bool(raw.get("stale", False))
    age_ms = _safe_float(raw.get("age_ms"))
    positions, reason = _validate_positions(raw.get("positions_deg"))
    raw_ticks = _safe_int_dict(raw.get("raw_ticks"))

    return ArmStateView(
        connected=connected,
        positions_deg=positions,
        raw_ticks=raw_ticks,
        stale=server_stale,
        age_ms=age_ms,
        valid=positions is not None,
        invalid_reason=reason,
    )


def _validate_difference(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for joint in JOINT_NAMES:
        if joint not in raw:
            continue
        value = raw[joint]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        fvalue = float(value)
        if math.isfinite(fvalue):
            result[joint] = fvalue
    return result


def compute_effective_stale(arm: ArmStateView, stale_after_ms: float) -> bool:
    """서버가 보고한 stale뿐 아니라, 클라이언트 자신의 stale_after_ms 기준으로도 재판정한다.

    age_ms 자체가 없으면(한 번도 정상값을 못 받은 경우 등) 안전하게 stale로 간주한다.
    """
    if arm.stale:
        return True
    if arm.age_ms is None:
        return True
    return arm.age_ms > stale_after_ms


@dataclass
class SequenceWatchdogConfig:
    stall_warn_after_s: float = 2.0
    stall_block_after_s: float = 5.0

    def __post_init__(self) -> None:
        if self.stall_block_after_s < self.stall_warn_after_s:
            raise ValueError("stall_block_after_s는 stall_warn_after_s 이상이어야 합니다.")


class SequenceWatchdog:
    """``sequence`` 값이 시간이 지나도 증가하지 않으면 WARN/BLOCKED를 반환한다.

    단일 스냅샷만으로는 "정지"를 판정할 수 없으므로(증가하는 도중의 한 순간일 수도 있음)
    직전 관측 시각을 들고 있는 상태 객체로 구현한다.
    """

    def __init__(self, config: SequenceWatchdogConfig | None = None) -> None:
        self._config = config or SequenceWatchdogConfig()
        self._last_sequence: int | None = None
        self._last_change_monotonic: float | None = None

    def reset(self) -> None:
        self._last_sequence = None
        self._last_change_monotonic = None

    def observe(self, sequence: int | None, now: float | None = None) -> str:
        """PASS / WARN / BLOCKED 중 하나를 반환한다. sequence가 None이면 WARN(판정 불가)."""
        now = now if now is not None else time.monotonic()
        if sequence is None:
            return "WARN"
        if self._last_sequence is None or sequence != self._last_sequence:
            self._last_sequence = sequence
            self._last_change_monotonic = now
            return "PASS"

        last_change = self._last_change_monotonic if self._last_change_monotonic is not None else now
        stalled_for = now - last_change
        if stalled_for >= self._config.stall_block_after_s:
            return "BLOCKED"
        if stalled_for >= self._config.stall_warn_after_s:
            return "WARN"
        return "PASS"


class RemoteSO101StateClient:
    """``GET /health``, ``GET /state``, ``GET /calibration``만 호출하는 읽기 전용 클라이언트.

    Args:
        config: 서버 URL, timeout, 재시도 횟수, (선택) API 토큰.
        session: 테스트에서 가짜 세션을 주입하기 위한 hook. None이면 ``requests.Session()``을
            새로 만든다. 어떤 경우든 이 클래스는 ``session.get()``만 호출하고, 다른 HTTP
            메서드는 절대 호출하지 않는다.
    """

    def __init__(self, config: RemoteClientConfig, session: requests.Session | None = None) -> None:
        self._config = config
        self._session = session if session is not None else requests.Session()
        if config.api_token:
            self._session.headers["Authorization"] = f"Bearer {config.api_token}"
        self._base_url = config.server_url.rstrip("/")

    def _get(self, path: str) -> dict:
        url = f"{self._base_url}{path}"
        timeout_s = self._config.timeout_ms / 1000.0
        attempts = self._config.max_retries
        last_error: RemoteStateError | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self._session.get(url, timeout=timeout_s)
            except requests.exceptions.Timeout:
                last_error = RemoteStateError(
                    f"{path} 요청이 timeout({self._config.timeout_ms:.0f}ms)되었습니다 "
                    f"(시도 {attempt}/{attempts})."
                )
            except requests.exceptions.ConnectionError as exc:
                last_error = RemoteStateError(f"{path} 연결에 실패했습니다 (시도 {attempt}/{attempts}): {exc}")
            except requests.exceptions.RequestException as exc:
                last_error = RemoteStateError(f"{path} 요청 중 오류가 발생했습니다 (시도 {attempt}/{attempts}): {exc}")
            else:
                if response.status_code != 200:
                    last_error = RemoteStateError(
                        f"{path} 요청이 실패했습니다 (HTTP {response.status_code}, 시도 {attempt}/{attempts})."
                    )
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        last_error = RemoteStateError(f"{path} 응답이 올바른 JSON이 아닙니다: {exc}")
                    else:
                        if not isinstance(data, dict):
                            last_error = RemoteStateError(f"{path} 응답 최상위가 JSON object가 아닙니다.")
                        else:
                            return data

            if attempt < attempts:
                time.sleep(self._config.retry_backoff_s)

        assert last_error is not None
        raise last_error

    def check_health(self) -> HealthState:
        data = self._get("/health")
        errors_raw = data.get("errors")
        errors = [str(e) for e in errors_raw] if isinstance(errors_raw, list) else []
        write_enabled_raw = data.get("write_enabled")
        return HealthState(
            status=data.get("status") if isinstance(data.get("status"), str) else None,
            mode=data.get("mode") if isinstance(data.get("mode"), str) else None,
            leader_connected=bool(data.get("leader_connected", False)),
            follower_connected=bool(data.get("follower_connected", False)),
            write_enabled=write_enabled_raw if isinstance(write_enabled_raw, bool) else None,
            timestamp=_safe_float(data.get("timestamp")),
            errors=errors,
        )

    def get_state(self) -> RemoteState:
        t0 = time.monotonic()
        data = self._get("/state")
        t1 = time.monotonic()

        leader = _parse_arm(data.get("leader"))
        follower = _parse_arm(data.get("follower"))
        difference = _validate_difference(data.get("difference_deg"))

        warnings_raw = data.get("warnings")
        warnings = [str(w) for w in warnings_raw] if isinstance(warnings_raw, list) else []

        return RemoteState(
            raw_timestamp=_safe_float(data.get("timestamp")),
            sequence=_safe_int(data.get("sequence")),
            mode=data.get("mode") if isinstance(data.get("mode"), str) else None,
            leader=leader,
            follower=follower,
            difference_deg=difference,
            warnings=warnings,
            received_at_monotonic=t1,
            received_at_wall=time.time(),
            network_latency_ms=(t1 - t0) * 1000.0,
        )

    def get_calibration(self) -> dict[str, dict[str, CalibrationEntry]]:
        data = self._get("/calibration")
        result: dict[str, dict[str, CalibrationEntry]] = {}
        for arm_key in ("leader", "follower"):
            arm_raw = data.get(arm_key)
            entries: dict[str, CalibrationEntry] = {}
            if isinstance(arm_raw, dict):
                for joint, entry_raw in arm_raw.items():
                    if not isinstance(entry_raw, dict):
                        continue
                    entries[joint] = CalibrationEntry(
                        homing_offset=_safe_int(entry_raw.get("homing_offset")),
                        range_min=_safe_int(entry_raw.get("range_min")),
                        range_max=_safe_int(entry_raw.get("range_max")),
                    )
            result[arm_key] = entries
        return result

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "RemoteSO101StateClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
