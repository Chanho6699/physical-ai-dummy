"""simulation/mujoco/remote_state_client.py 단위 테스트.

실제 네트워크나 실제 노트북 서버 없이, ``requests.Session``과 동일한 인터페이스
(``get(url, timeout=...)``, ``headers``, ``close()``)를 갖는 가짜 세션을 주입해 검증한다.
"""

from __future__ import annotations

import math

import pytest
import requests

from simulation.mujoco.remote_state_client import (
    JOINT_NAMES,
    RemoteClientConfig,
    RemoteSO101StateClient,
    RemoteStateError,
    SequenceWatchdog,
    SequenceWatchdogConfig,
    compute_effective_stale,
)


def _good_positions(offset: float = 0.0) -> dict[str, float]:
    return {name: float(i) + offset for i, name in enumerate(JOINT_NAMES)}


def _good_state_payload(sequence: int = 1) -> dict:
    leader_pos = _good_positions()
    follower_pos = _good_positions(0.5)
    return {
        "timestamp": 1786020000.0,
        "sequence": sequence,
        "mode": "read_only",
        "leader": {
            "connected": True,
            "positions_deg": leader_pos,
            "raw_ticks": {name: 2000 + i for i, name in enumerate(JOINT_NAMES)},
            "stale": False,
            "age_ms": 15.0,
        },
        "follower": {
            "connected": True,
            "positions_deg": follower_pos,
            "raw_ticks": {},
            "stale": False,
            "age_ms": 12.0,
        },
        "difference_deg": {name: leader_pos[name] - follower_pos[name] for name in JOINT_NAMES},
        "warnings": [],
    }


def _good_health_payload() -> dict:
    return {
        "status": "ok",
        "mode": "read_only",
        "leader_connected": True,
        "follower_connected": True,
        "write_enabled": False,
        "timestamp": 1786020000.0,
        "errors": [],
    }


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, data: object = None, json_error: bool = False) -> None:
        self.status_code = status_code
        self._data = data
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("malformed JSON")
        return self._data


class _FakeSession:
    """path suffix -> 응답(또는 응답을 만드는 callable) 매핑으로 동작하는 가짜 세션."""

    def __init__(self, responses: dict | None = None, raises: Exception | None = None) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, float]] = []
        self._responses = responses or {}
        self._raises = raises
        self.closed = False

    def get(self, url: str, timeout: float | None = None):
        self.calls.append((url, timeout))
        if self._raises is not None:
            raise self._raises
        for suffix, response in self._responses.items():
            if url.endswith(suffix):
                return response() if callable(response) else response
        raise AssertionError(f"예상치 못한 URL 요청: {url}")

    def close(self) -> None:
        self.closed = True


def _client(session: _FakeSession, **config_kwargs) -> RemoteSO101StateClient:
    config = RemoteClientConfig(server_url="http://laptop.local:8001", **config_kwargs)
    return RemoteSO101StateClient(config, session=session)


# ---------------------------------------------------------------------------
# 정상 응답
# ---------------------------------------------------------------------------


def test_check_health_normal_response():
    session = _FakeSession({"/health": _FakeResponse(data=_good_health_payload())})
    client = _client(session)
    health = client.check_health()
    assert health.status == "ok"
    assert health.mode == "read_only"
    assert health.leader_connected is True
    assert health.follower_connected is True
    assert health.write_enabled is False
    assert health.errors == []


def test_get_state_normal_response():
    session = _FakeSession({"/state": _FakeResponse(data=_good_state_payload(sequence=42))})
    client = _client(session)
    state = client.get_state()
    assert state.sequence == 42
    assert state.mode == "read_only"
    assert state.leader.valid
    assert state.follower.valid
    assert state.leader.positions_deg["wrist_flex"] == pytest.approx(3.0)
    assert state.leader.raw_ticks["wrist_flex"] == 2003
    assert state.network_latency_ms >= 0.0
    assert state.state_age_ms == 15.0  # leader/follower age_ms 중 더 큰 값


def test_get_calibration_normal_response():
    payload = {
        "leader": {"wrist_flex": {"homing_offset": 10, "range_min": 0, "range_max": 4095}},
        "follower": {"wrist_flex": {"homing_offset": -5, "range_min": 0, "range_max": 4095}},
    }
    session = _FakeSession({"/calibration": _FakeResponse(data=payload)})
    client = _client(session)
    calibration = client.get_calibration()
    assert calibration["leader"]["wrist_flex"].homing_offset == 10
    assert calibration["follower"]["wrist_flex"].range_max == 4095


# ---------------------------------------------------------------------------
# 네트워크/응답 오류
# ---------------------------------------------------------------------------


def test_connection_error_raises_after_limited_retries():
    session = _FakeSession(raises=requests.exceptions.ConnectionError("연결 거부"))
    client = _client(session, max_retries=3, retry_backoff_s=0.0)
    with pytest.raises(RemoteStateError):
        client.get_state()
    assert len(session.calls) == 3  # 무한 재시도가 아니라 정확히 max_retries번만 시도


def test_timeout_raises_remote_state_error():
    session = _FakeSession(raises=requests.exceptions.Timeout("timed out"))
    client = _client(session, max_retries=2, retry_backoff_s=0.0)
    with pytest.raises(RemoteStateError, match="timeout"):
        client.check_health()
    assert len(session.calls) == 2


def test_malformed_json_raises_remote_state_error():
    session = _FakeSession({"/state": _FakeResponse(json_error=True)})
    client = _client(session, max_retries=1)
    with pytest.raises(RemoteStateError, match="JSON"):
        client.get_state()


def test_non_object_json_top_level_raises():
    session = _FakeSession({"/state": _FakeResponse(data=[1, 2, 3])})
    client = _client(session, max_retries=1)
    with pytest.raises(RemoteStateError):
        client.get_state()


def test_http_error_status_raises():
    session = _FakeSession({"/state": _FakeResponse(status_code=500, data={})})
    client = _client(session, max_retries=1)
    with pytest.raises(RemoteStateError, match="500"):
        client.get_state()


def test_max_retries_must_be_at_least_one():
    with pytest.raises(ValueError):
        RemoteClientConfig(server_url="http://x:8001", max_retries=0)


# ---------------------------------------------------------------------------
# 콘텐츠 검증: NaN/Inf, 누락 필드, disconnect, stale
# ---------------------------------------------------------------------------


def test_nan_position_marks_arm_invalid_not_exception():
    payload = _good_state_payload()
    payload["leader"]["positions_deg"]["wrist_flex"] = math.nan
    session = _FakeSession({"/state": _FakeResponse(data=payload)})
    client = _client(session)
    state = client.get_state()  # 예외를 던지지 않고, 무효 표시만 한다
    assert state.leader.valid is False
    assert "wrist_flex" in state.leader.invalid_reason


def test_inf_position_marks_arm_invalid():
    payload = _good_state_payload()
    payload["follower"]["positions_deg"]["gripper"] = math.inf
    session = _FakeSession({"/state": _FakeResponse(data=payload)})
    client = _client(session)
    state = client.get_state()
    assert state.follower.valid is False


def test_missing_joint_marks_arm_invalid():
    payload = _good_state_payload()
    del payload["leader"]["positions_deg"]["gripper"]
    session = _FakeSession({"/state": _FakeResponse(data=payload)})
    client = _client(session)
    state = client.get_state()
    assert state.leader.valid is False
    assert "gripper" in state.leader.invalid_reason


def test_leader_disconnected_reflected_in_state():
    payload = _good_state_payload()
    payload["leader"]["connected"] = False
    session = _FakeSession({"/state": _FakeResponse(data=payload)})
    client = _client(session)
    state = client.get_state()
    assert state.leader.connected is False


def test_follower_disconnected_reflected_in_state():
    payload = _good_state_payload()
    payload["follower"]["connected"] = False
    session = _FakeSession({"/state": _FakeResponse(data=payload)})
    client = _client(session)
    state = client.get_state()
    assert state.follower.connected is False


def test_stale_flag_from_server_is_preserved():
    payload = _good_state_payload()
    payload["leader"]["stale"] = True
    session = _FakeSession({"/state": _FakeResponse(data=payload)})
    client = _client(session)
    state = client.get_state()
    assert state.leader.stale is True
    assert compute_effective_stale(state.leader, stale_after_ms=500.0) is True


def test_compute_effective_stale_uses_age_ms_threshold():
    payload = _good_state_payload()
    payload["leader"]["stale"] = False
    payload["leader"]["age_ms"] = 900.0
    session = _FakeSession({"/state": _FakeResponse(data=payload)})
    client = _client(session)
    state = client.get_state()
    assert compute_effective_stale(state.leader, stale_after_ms=500.0) is True
    assert compute_effective_stale(state.leader, stale_after_ms=1000.0) is False


def test_compute_effective_stale_missing_age_is_conservative():
    payload = _good_state_payload()
    payload["leader"]["age_ms"] = None
    payload["leader"]["stale"] = False
    session = _FakeSession({"/state": _FakeResponse(data=payload)})
    client = _client(session)
    state = client.get_state()
    assert compute_effective_stale(state.leader, stale_after_ms=500.0) is True


def test_mode_not_read_only_is_parsed_as_is():
    payload = _good_state_payload()
    payload["mode"] = "teleop"
    session = _FakeSession({"/state": _FakeResponse(data=payload)})
    client = _client(session)
    state = client.get_state()
    assert state.mode == "teleop"


def test_write_enabled_missing_is_none_not_false():
    payload = _good_health_payload()
    del payload["write_enabled"]
    session = _FakeSession({"/health": _FakeResponse(data=payload)})
    client = _client(session)
    health = client.check_health()
    assert health.write_enabled is None  # "확인 불가" - False로 임의 추정하지 않는다


def test_write_enabled_true_is_parsed_as_true():
    payload = _good_health_payload()
    payload["write_enabled"] = True
    session = _FakeSession({"/health": _FakeResponse(data=payload)})
    client = _client(session)
    health = client.check_health()
    assert health.write_enabled is True


# ---------------------------------------------------------------------------
# sequence watchdog
# ---------------------------------------------------------------------------


def test_sequence_watchdog_pass_while_increasing():
    watchdog = SequenceWatchdog(SequenceWatchdogConfig(stall_warn_after_s=2.0, stall_block_after_s=5.0))
    assert watchdog.observe(1, now=0.0) == "PASS"
    assert watchdog.observe(2, now=0.5) == "PASS"
    assert watchdog.observe(3, now=1.0) == "PASS"


def test_sequence_watchdog_warns_then_blocks_when_stalled():
    watchdog = SequenceWatchdog(SequenceWatchdogConfig(stall_warn_after_s=2.0, stall_block_after_s=5.0))
    assert watchdog.observe(10, now=0.0) == "PASS"
    assert watchdog.observe(10, now=1.0) == "PASS"  # 아직 warn 임계값 전
    assert watchdog.observe(10, now=2.5) == "WARN"
    assert watchdog.observe(10, now=6.0) == "BLOCKED"


def test_sequence_watchdog_recovers_when_sequence_changes_again():
    watchdog = SequenceWatchdog(SequenceWatchdogConfig(stall_warn_after_s=2.0, stall_block_after_s=5.0))
    watchdog.observe(1, now=0.0)
    assert watchdog.observe(1, now=6.0) == "BLOCKED"
    assert watchdog.observe(2, now=6.1) == "PASS"


def test_sequence_watchdog_none_sequence_is_warn():
    watchdog = SequenceWatchdog()
    assert watchdog.observe(None, now=0.0) == "WARN"


# ---------------------------------------------------------------------------
# 토큰 비노출 + GET 전용 확인
# ---------------------------------------------------------------------------


def test_api_token_sent_as_bearer_header_and_not_in_repr():
    session = _FakeSession({"/health": _FakeResponse(data=_good_health_payload())})
    config = RemoteClientConfig(server_url="http://laptop.local:8001", api_token="super-secret-token")
    client = RemoteSO101StateClient(config, session=session)
    client.check_health()
    assert session.headers["Authorization"] == "Bearer super-secret-token"
    assert "super-secret-token" not in repr(config)
    assert "super-secret-token" not in str(config)


def test_no_post_like_http_methods_exist_on_client():
    session = _FakeSession({"/health": _FakeResponse(data=_good_health_payload())})
    client = _client(session)
    for forbidden in ("post", "put", "patch", "delete"):
        assert not hasattr(client, forbidden)


def test_close_closes_underlying_session():
    session = _FakeSession()
    client = _client(session)
    client.close()
    assert session.closed is True


def test_client_as_context_manager_closes_session():
    session = _FakeSession({"/health": _FakeResponse(data=_good_health_payload())})
    config = RemoteClientConfig(server_url="http://laptop.local:8001")
    with RemoteSO101StateClient(config, session=session) as client:
        client.check_health()
    assert session.closed is True
