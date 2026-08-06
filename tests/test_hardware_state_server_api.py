"""hardware/state_server/app.py (FastAPI) 통합 테스트.

실물 하드웨어 없이, 가짜 reader로 채운 StatePoller를 그대로 create_app()에 넣어
HTTP 계층(라우팅/직렬화/인증/금지된 엔드포인트의 부재)만 검증한다. 백그라운드 폴링
스레드는 시작하지 않는다 - /health, /state는 poller의 현재 캐시를 즉시 반환하므로
타이밍에 의존할 필요가 없다.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from hardware.state_server.app import create_app
from hardware.state_server.calibration_loader import to_public_dict
from hardware.state_server.state_models import JOINT_NAMES
from hardware.state_server.state_service import StatePoller

GOOD_LEADER = {joint: float(i) for i, joint in enumerate(JOINT_NAMES)}
GOOD_FOLLOWER = {joint: float(i) + 0.5 for i, joint in enumerate(JOINT_NAMES)}

class _FakeCalibrationEntry:
    def __init__(self, homing_offset: int, range_min: int, range_max: int) -> None:
        self.homing_offset = homing_offset
        self.range_min = range_min
        self.range_max = range_max


def _fake_calibration_entries() -> dict[str, _FakeCalibrationEntry]:
    return {joint: _FakeCalibrationEntry(homing_offset=i, range_min=0, range_max=4095) for i, joint in enumerate(JOINT_NAMES)}


class FakeReader:
    def __init__(self, name: str, position: dict[str, float] | None) -> None:
        self.name = name
        self._position = position
        self._connected = False
        self.disconnect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._position is None:
            raise RuntimeError(f"{self.name} 연결 실패 (시뮬레이션)")
        self._connected = True

    def read_positions(self) -> dict[str, float]:
        if self._position is None:
            raise RuntimeError("읽기 실패")
        return dict(self._position)

    def read_raw_positions(self) -> dict[str, int] | None:
        return None

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False


def _build_client(*, api_token: str | None = None, both_connected: bool = True) -> TestClient:
    leader = FakeReader("leader", GOOD_LEADER)
    follower = FakeReader("follower", GOOD_FOLLOWER if both_connected else None)
    poller = StatePoller(
        leader_reader=leader,
        follower_reader=follower,
        rate_hz=30.0,
        stale_after_ms=500.0,
        max_read_errors=3,
    )
    poller.connect_all()
    poller.poll_once()

    calibration_public = {
        "leader": to_public_dict(_fake_calibration_entries()),
        "follower": to_public_dict(_fake_calibration_entries()),
    }
    app = create_app(poller=poller, calibration_public=calibration_public, api_token=api_token)
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_ok_when_both_arms_connected():
    client = _build_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mode"] == "read_only"
    assert body["leader_connected"] is True
    assert body["follower_connected"] is True
    assert body["write_enabled"] is False
    assert "timestamp" in body


def test_health_degraded_when_follower_disconnected():
    client = _build_client(both_connected=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["follower_connected"] is False
    assert any("팔로워암" in e for e in body["errors"])


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------


def test_state_returns_positions_and_difference():
    client = _build_client()
    resp = client.get("/state")
    assert resp.status_code == 200
    body = resp.json()

    assert body["mode"] == "read_only"
    assert body["sequence"] == 1
    assert body["leader"]["connected"] is True
    assert body["leader"]["positions_deg"] == GOOD_LEADER
    assert body["follower"]["positions_deg"] == GOOD_FOLLOWER
    for joint in JOINT_NAMES:
        assert body["difference_deg"][joint] == GOOD_LEADER[joint] - GOOD_FOLLOWER[joint]


def test_state_shows_stale_and_null_positions_when_arm_never_read():
    client = _build_client(both_connected=False)
    resp = client.get("/state")
    body = resp.json()
    assert body["follower"]["connected"] is False
    assert body["follower"]["positions_deg"] is None
    assert body["follower"]["stale"] is True


# ---------------------------------------------------------------------------
# /calibration
# ---------------------------------------------------------------------------


def test_calibration_returns_leader_and_follower():
    client = _build_client()
    resp = client.get("/calibration")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"leader", "follower"}
    assert set(body["leader"]) == set(JOINT_NAMES)
    assert set(body["leader"]["wrist_flex"]) == {"homing_offset", "range_min", "range_max"}


# ---------------------------------------------------------------------------
# 금지된 제어 API가 존재하지 않는지 검증
# ---------------------------------------------------------------------------


def test_forbidden_control_endpoints_return_404():
    client = _build_client()
    for path in ("/action", "/move", "/command", "/teleop"):
        resp = client.post(path, json={})
        assert resp.status_code == 404, f"{path}는 존재하지 않아야 합니다 (404 예상, got {resp.status_code})"


def test_post_to_read_only_paths_returns_405():
    client = _build_client()
    for path in ("/health", "/state", "/calibration"):
        resp = client.post(path, json={})
        assert resp.status_code == 405, f"{path}는 GET만 허용해야 합니다 (405 예상, got {resp.status_code})"


def test_openapi_exposes_only_get_on_read_only_paths():
    client = _build_client()
    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) == {"/health", "/state", "/calibration"}
    for path, methods in schema["paths"].items():
        assert set(methods) == {"get"}, f"{path}에 GET 이외의 메서드가 노출되어 있습니다: {set(methods)}"


def test_no_body_accepting_routes_exist_besides_docs():
    """request body를 받는 라우트가 없는지 openapi requestBody 필드로 확인한다."""

    client = _build_client()
    schema = client.get("/openapi.json").json()
    for path, methods in schema["paths"].items():
        for _method, operation in methods.items():
            assert "requestBody" not in operation, f"{path}가 request body를 받습니다."


# ---------------------------------------------------------------------------
# api_token 인증
# ---------------------------------------------------------------------------


def test_api_token_rejects_missing_authorization():
    client = _build_client(api_token="secret-token")
    resp = client.get("/health")
    assert resp.status_code == 401


def test_api_token_rejects_wrong_token():
    client = _build_client(api_token="secret-token")
    resp = client.get("/health", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_api_token_accepts_correct_token():
    client = _build_client(api_token="secret-token")
    resp = client.get("/health", headers={"Authorization": "Bearer secret-token"})
    assert resp.status_code == 200


def test_no_api_token_means_no_auth_required():
    client = _build_client(api_token=None)
    resp = client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# CORS: 전체 허용 미들웨어가 추가되지 않았는지 확인
# ---------------------------------------------------------------------------


def test_no_wildcard_cors_headers_on_cross_origin_request():
    client = _build_client()
    resp = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert resp.status_code == 200
    # CORSMiddleware를 추가하지 않았으므로 Access-Control-Allow-Origin이 없어야 한다.
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}
