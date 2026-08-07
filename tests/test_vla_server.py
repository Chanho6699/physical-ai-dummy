"""runtime/desktop/vla_server.py (FastAPI) 통합 테스트 - FakePolicyRunner backend.

실제 SmolVLA 체크포인트/GPU 없이 HTTP 계약(라우팅/스키마/오류 구분)만 검증한다.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY, JOINT_ORDER
from runtime.desktop.vla_server import FakePolicyRunner, PolicyInferenceError, create_app


def _b64_jpeg(shape=(480, 640, 3)) -> str:
    arr = np.zeros(shape, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _good_state() -> dict[str, float]:
    return {joint: float(i) for i, joint in enumerate(JOINT_ORDER)}


def _good_predict_body(**overrides) -> dict:
    body = {
        "session_id": "s1",
        "task": "pick up the cube",
        "sequence": 0,
        "timestamp": 1.0,
        "observation": {
            "state": _good_state(),
            "images": {CAMERA_WORKSPACE_KEY: _b64_jpeg(), CAMERA_WRIST_KEY: _b64_jpeg()},
        },
    }
    body.update(overrides)
    return body


def _client(runner=None) -> TestClient:
    return TestClient(create_app(policy_runner=runner or FakePolicyRunner()))


def test_health_ok() -> None:
    resp = _client().get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend"] == "fake"
    assert data["model_loaded"] is True


def test_health_degraded_when_not_ready() -> None:
    class NotReadyRunner(FakePolicyRunner):
        def is_ready(self) -> bool:
            return False

    resp = _client(NotReadyRunner()).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_session_reset_calls_policy_reset() -> None:
    runner = FakePolicyRunner()
    resp = _client(runner).post("/session/reset", json={"session_id": "s1", "task": "pick"})
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "s1", "ok": True}
    assert runner.reset_calls == [("s1", "pick")]


def test_predict_success_returns_action() -> None:
    resp = _client().post("/predict", json=_good_predict_body())
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["action"]) == set(JOINT_ORDER)
    assert data["backend"] == "fake"
    assert data["inference_latency_ms"] >= 0.0


def test_predict_missing_state_joint_returns_422() -> None:
    body = _good_predict_body()
    del body["observation"]["state"]["gripper"]
    resp = _client().post("/predict", json=body)
    assert resp.status_code == 422


def test_predict_nan_state_returns_422() -> None:
    import json

    body = _good_predict_body()
    body["observation"]["state"]["gripper"] = float("nan")
    # httpx(TestClient)의 json= 헬퍼는 JSON 표준을 엄격히 따라 NaN을 인코딩 단계에서
    # 거부한다 (실제 프로덕션 client인 requests는 stdlib json.dumps 기본값(allow_nan=True)을
    # 써서 "NaN" 토큰을 그대로 보낸다 - runtime/laptop/vla_client.py가 실제로 쓰는 경로).
    # 여기서는 그 실제 wire 동작을 재현하기 위해 raw body를 직접 인코딩해서 보낸다.
    raw = json.dumps(body, allow_nan=True)
    resp = _client().post("/predict", content=raw, headers={"content-type": "application/json"})
    assert resp.status_code == 422


def test_predict_missing_camera_returns_422() -> None:
    body = _good_predict_body()
    del body["observation"]["images"][CAMERA_WRIST_KEY]
    resp = _client().post("/predict", json=body)
    assert resp.status_code == 422


def test_predict_malformed_image_base64_returns_422() -> None:
    body = _good_predict_body()
    body["observation"]["images"][CAMERA_WRIST_KEY] = "not-base64!!"
    resp = _client().post("/predict", json=body)
    assert resp.status_code == 422


def test_predict_inference_failure_returns_500_with_inference_marker() -> None:
    class FailingRunner(FakePolicyRunner):
        def predict(self, *, task, state, images):
            raise PolicyInferenceError("모델이 죽었습니다")

    resp = _client(FailingRunner()).post("/predict", json=_good_predict_body())
    assert resp.status_code == 500
    assert "[inference_error]" in resp.json()["detail"]


def test_predict_malformed_policy_output_returns_500() -> None:
    class BadOutputRunner(FakePolicyRunner):
        def predict(self, *, task, state, images):
            return {"shoulder_pan": float("nan")}  # 관절 5개 누락 + NaN

    resp = _client(BadOutputRunner()).post("/predict", json=_good_predict_body())
    assert resp.status_code == 500


def test_action_ack_is_logged_but_does_not_control_anything() -> None:
    app = create_app(policy_runner=FakePolicyRunner())
    client = TestClient(app)
    resp = client.post("/action/ack", json={"session_id": "s1", "sequence": 0, "executed": True, "backend": "realistic_mujoco"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert app.state.last_acks[0]["session_id"] == "s1"


def test_no_control_endpoints_exposed() -> None:
    client = _client()
    for path in ("/action", "/move", "/command", "/teleop"):
        resp = client.post(path, json={})
        assert resp.status_code == 404


def test_api_token_required_when_configured() -> None:
    app = create_app(policy_runner=FakePolicyRunner(), api_token="secret")
    client = TestClient(app)
    assert client.get("/health").status_code == 401
    resp = client.get("/health", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
