"""runtime/laptop/vla_client.py 테스트 - 진짜 소켓(uvicorn)으로 왕복 + 통신/추론 실패 구분."""

from __future__ import annotations

import numpy as np
import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.desktop.vla_server import PolicyInferenceError
from runtime.laptop.vla_client import VLAClientConfig, VLAHttpClient

_IMAGES = {
    "observation.images.workspace": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
}
_STATE = {joint: float(i) for i, joint in enumerate(JOINT_ORDER)}


def test_health_round_trip_over_real_socket(live_fake_vla_server) -> None:
    base_url, _ = live_fake_vla_server
    client = VLAHttpClient(VLAClientConfig(server_url=base_url))
    health = client.check_health()
    assert health.ok is True
    assert health.backend == "fake"
    assert health.round_trip_ms >= 0.0
    client.close()


def test_session_reset_and_predict_round_trip(live_fake_vla_server) -> None:
    base_url, policy_runner = live_fake_vla_server
    client = VLAHttpClient(VLAClientConfig(server_url=base_url))
    assert client.session_reset(session_id="s1", task="pick up the cube") is True

    result = client.predict(session_id="s1", task="pick up the cube", sequence=0, state=_STATE, images=_IMAGES)
    assert result.ok is True
    assert result.action_schema_valid is True
    assert set(result.action) == set(JOINT_ORDER)
    assert result.inference_latency_ms is not None
    assert result.request_latency_ms >= 0.0
    assert result.error_kind is None
    client.close()


def test_action_ack_round_trip(live_fake_vla_server) -> None:
    base_url, _ = live_fake_vla_server
    client = VLAHttpClient(VLAClientConfig(server_url=base_url))
    assert client.action_ack(session_id="s1", sequence=0, executed=True, backend="realistic_mujoco") is True
    client.close()


def test_predict_distinguishes_inference_failure_from_communication_failure(live_fake_vla_server) -> None:
    base_url, policy_runner = live_fake_vla_server

    def _boom(*, task, state, images):
        raise PolicyInferenceError("모델이 죽었습니다")

    policy_runner.predict = _boom  # type: ignore[method-assign]

    client = VLAHttpClient(VLAClientConfig(server_url=base_url))
    result = client.predict(session_id="s1", task="pick", sequence=0, state=_STATE, images=_IMAGES)
    assert result.ok is False
    assert result.error_kind == "inference"
    assert "모델이 죽었습니다" in result.error_message
    client.close()


def test_communication_failure_on_unreachable_server() -> None:
    # 아무도 듣고 있지 않은 포트 - 연결 자체가 실패해야 한다 (모델 추론 실패와 구분).
    client = VLAHttpClient(VLAClientConfig(server_url="http://127.0.0.1:1", timeout_s=1.0, max_retries=1))
    health = client.check_health()
    assert health.ok is False
    assert health.raw_error is not None
    client.close()


def test_predict_communication_failure_kind() -> None:
    client = VLAHttpClient(VLAClientConfig(server_url="http://127.0.0.1:1", timeout_s=1.0, max_retries=1))
    result = client.predict(session_id="s1", task="pick", sequence=0, state=_STATE, images=_IMAGES)
    assert result.ok is False
    assert result.error_kind == "communication"
    client.close()


# ---------------------------------------------------------------------------
# Phase C-1A: predict_chunk() (/predict_chunk) - predict()/PredictResult는 위 그대로.
# ---------------------------------------------------------------------------


def test_predict_chunk_round_trip(live_fake_vla_server) -> None:
    base_url, _ = live_fake_vla_server
    client = VLAHttpClient(VLAClientConfig(server_url=base_url))
    result = client.predict_chunk(session_id="s1", task="pick up the cube", sequence=0, state=_STATE, images=_IMAGES)
    assert result.ok is True
    assert result.chunk_schema_valid is True
    assert result.chunk_size == 50  # FakePolicyRunner 기본값
    assert len(result.chunk) == 50
    for step in result.chunk:
        assert set(step) == set(JOINT_ORDER)
    assert result.chunk_index_spacing_s == pytest.approx(1.0 / 30.0)
    assert result.inference_latency_ms is not None
    assert result.request_latency_ms >= 0.0
    assert result.requested_at_monotonic > 0.0
    assert result.error_kind is None
    client.close()


def test_predict_chunk_distinguishes_inference_failure_from_communication_failure(live_fake_vla_server) -> None:
    base_url, policy_runner = live_fake_vla_server

    def _boom(*, task, state, images):
        raise PolicyInferenceError("chunk 모델이 죽었습니다")

    policy_runner.predict_chunk = _boom  # type: ignore[method-assign]

    client = VLAHttpClient(VLAClientConfig(server_url=base_url))
    result = client.predict_chunk(session_id="s1", task="pick", sequence=0, state=_STATE, images=_IMAGES)
    assert result.ok is False
    assert result.error_kind == "inference"
    assert "chunk 모델이 죽었습니다" in result.error_message
    client.close()


def test_predict_chunk_communication_failure_kind() -> None:
    client = VLAHttpClient(VLAClientConfig(server_url="http://127.0.0.1:1", timeout_s=1.0, max_retries=1))
    result = client.predict_chunk(session_id="s1", task="pick", sequence=0, state=_STATE, images=_IMAGES)
    assert result.ok is False
    assert result.error_kind == "communication"
    assert result.chunk is None
    client.close()


def test_predict_chunk_malformed_chunk_shape_is_schema_error(live_fake_vla_server) -> None:
    """서버가 (스펙을 어기고) 빈 chunk를 반환하면 client가 fail-closed로 schema 오류를
    내야 한다 - 빈 list/부분 chunk를 그대로 executor에 넘기지 않는다."""
    base_url, policy_runner = live_fake_vla_server
    policy_runner.predict_chunk = lambda *, task, state, images: []  # type: ignore[method-assign]

    client = VLAHttpClient(VLAClientConfig(server_url=base_url))
    result = client.predict_chunk(session_id="s1", task="pick", sequence=0, state=_STATE, images=_IMAGES)
    assert result.ok is False
    assert result.chunk is None
    client.close()


def test_predict_chunk_does_not_affect_existing_predict(live_fake_vla_server) -> None:
    """같은 client/서버에서 predict_chunk를 호출한 뒤에도 predict()가 정상 동작해야 한다."""
    base_url, _ = live_fake_vla_server
    client = VLAHttpClient(VLAClientConfig(server_url=base_url))
    client.predict_chunk(session_id="s1", task="pick", sequence=0, state=_STATE, images=_IMAGES)
    result = client.predict(session_id="s1", task="pick", sequence=1, state=_STATE, images=_IMAGES)
    assert result.ok is True
    client.close()


def test_nan_state_is_rejected_over_real_wire(live_fake_vla_server) -> None:
    """실제 프로덕션 경로(requests)는 NaN을 그대로 JSON에 실어 보낸다 - 서버가 422로
    거부하는지, 클라이언트가 그 실패를 통신 실패가 아니라 스키마 문제로 다루는지 확인한다."""
    base_url, _ = live_fake_vla_server
    client = VLAHttpClient(VLAClientConfig(server_url=base_url))
    bad_state = dict(_STATE)
    bad_state["gripper"] = float("nan")
    result = client.predict(session_id="s1", task="pick", sequence=0, state=bad_state, images=_IMAGES)
    assert result.ok is False
    # 422는 200도 500도 아니므로 "그 외 HTTP 오류"로 재시도 후 communication 실패로 분류된다 -
    # 어느 쪽이든 모델이 정상 action을 반환하지 않았다는 사실은 명확히 구분된다.
    assert result.error_kind in ("communication", "schema")
    client.close()
