"""Laptop -> Desktop SmolVLA 서버 HTTP client.

``simulation/mujoco/remote_state_client.py``와 동일한 defensive parsing 원칙을 따른다:
원격 서버가 보낸 값은 신뢰할 수 없는 입력으로 취급하고, 연결 실패/timeout/malformed
JSON/스키마 불일치를 모두 예외가 아니라(가능한 경우) 구분된 결과로 다룬다. 다만 이
클라이언트는 (그 모듈과 달리) POST도 사용한다 - ``/session/reset``, ``/predict``,
``/action/ack``가 POST 엔드포인트이기 때문이다. 이 클라이언트가 호출하는 HTTP 메서드는
이 파일 안의 ``session.get``/``session.post`` 두 곳뿐이며, 실물 로봇에는 어떤 것도
쓰지 않는다 (이 클라이언트는 Desktop에만 말한다).

환경변수 ``VLA_SERVER_URL``을 기본 서버 URL로 사용한다 (기존 Fake/Real VLA 설계 이력에서
언급된 이름 그대로 재사용 - 새 이름을 만들지 않았다).
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field

import requests

from runtime.common.vla_contract import (
    ACTION_ACK_PATH,
    HEALTH_PATH,
    PREDICT_PATH,
    SESSION_RESET_PATH,
    validate_joint_dict,
)

VLA_SERVER_URL_ENV_VAR = "VLA_SERVER_URL"


class VLAClientError(RuntimeError):
    """통신 실패(연결/timeout/HTTP 오류/malformed JSON) - 모델 추론 실패와 구분된다."""


class VLAInferenceError(RuntimeError):
    """통신은 성공했지만 서버가 모델 추론 실패로 응답한 경우 (HTTP 500 + inference_error 표시)."""


@dataclass(frozen=True)
class VLAClientConfig:
    server_url: str
    timeout_s: float = 15.0
    max_retries: int = 2
    retry_backoff_s: float = 0.2
    api_token: str | None = None

    def __post_init__(self) -> None:
        if not self.server_url:
            raise ValueError("server_url이 비어 있습니다.")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s는 0보다 커야 합니다.")
        if self.max_retries < 1:
            raise ValueError("max_retries는 1 이상이어야 합니다.")

    def __repr__(self) -> str:  # api_token 노출 방지
        token_state = "설정됨" if self.api_token else "없음"
        return f"VLAClientConfig(server_url={self.server_url!r}, timeout_s={self.timeout_s}, api_token={token_state})"

    @staticmethod
    def from_env(*, timeout_s: float = 15.0, api_token: str | None = None) -> "VLAClientConfig":
        url = os.environ.get(VLA_SERVER_URL_ENV_VAR)
        if not url:
            raise ValueError(f"환경변수 {VLA_SERVER_URL_ENV_VAR}가 설정되지 않았습니다.")
        return VLAClientConfig(server_url=url, timeout_s=timeout_s, api_token=api_token)


@dataclass(frozen=True)
class HealthResult:
    ok: bool
    status: str | None
    backend: str | None
    model_loaded: bool | None
    model_id: str | None
    device: str | None
    errors: list[str]
    round_trip_ms: float
    raw_error: str | None = None


@dataclass(frozen=True)
class PredictResult:
    """``/predict`` 왕복 결과. 통신 성공 여부와 action schema 유효성을 분리해서 담는다."""

    ok: bool
    session_id: str
    sequence: int
    action: dict[str, float] | None
    action_schema_valid: bool
    action_invalid_reason: str | None
    model_id: str | None
    backend: str | None
    inference_latency_ms: float | None
    request_latency_ms: float  # 클라이언트가 측정한 왕복(round-trip) 지연
    error_kind: str | None  # None | "communication" | "inference" | "schema"
    error_message: str | None = None


def _encode_image_base64(image_rgb) -> str:
    """HWC uint8 RGB numpy 배열을 base64 JPEG 문자열로 인코딩한다."""
    import io

    import numpy as np
    from PIL import Image

    arr = np.asarray(image_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"이미지 shape이 HWC*3이 아닙니다: {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"이미지 dtype이 uint8이 아닙니다: {arr.dtype}")
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLAHttpClient:
    """``/health``, ``/session/reset``, ``/predict``, ``/action/ack``만 호출하는 client.

    이 클래스는 실물 SO-101 follower를 전혀 참조하지 않는다 - Desktop VLA 서버와만
    통신한다. 실행/제어 관련 메서드(예: ``send_action``)는 존재하지 않는다.
    """

    def __init__(self, config: VLAClientConfig, session: requests.Session | None = None) -> None:
        self._config = config
        self._session = session if session is not None else requests.Session()
        if config.api_token:
            self._session.headers["Authorization"] = f"Bearer {config.api_token}"
        self._base_url = config.server_url.rstrip("/")

    def _request(self, method: str, path: str, *, json_body: dict | None = None) -> tuple[dict | None, float, str | None]:
        """(응답 JSON 또는 None, round_trip_ms, 오류 메시지 또는 None)."""
        url = f"{self._base_url}{path}"
        attempts = self._config.max_retries
        last_error: str | None = None

        for attempt in range(1, attempts + 1):
            t0 = time.monotonic()
            try:
                response = self._session.request(method, url, json=json_body, timeout=self._config.timeout_s)
            except requests.exceptions.Timeout:
                last_error = f"{path} 요청이 timeout({self._config.timeout_s}s)되었습니다 (시도 {attempt}/{attempts})."
            except requests.exceptions.ConnectionError as exc:
                last_error = f"{path} 연결에 실패했습니다 (시도 {attempt}/{attempts}): {exc}"
            except requests.exceptions.RequestException as exc:
                last_error = f"{path} 요청 중 오류가 발생했습니다 (시도 {attempt}/{attempts}): {exc}"
            else:
                round_trip_ms = (time.monotonic() - t0) * 1000.0
                if response.status_code == 500:
                    # 서버가 모델 추론 실패를 알린 경우 - 통신 자체는 성공이므로 재시도하지 않는다.
                    try:
                        detail = response.json().get("detail", "")
                    except ValueError:
                        detail = response.text
                    return None, round_trip_ms, f"__INFERENCE__{detail}"
                if response.status_code != 200:
                    last_error = f"{path} 요청이 실패했습니다 (HTTP {response.status_code}, 시도 {attempt}/{attempts}): {response.text[:200]}"
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        last_error = f"{path} 응답이 올바른 JSON이 아닙니다: {exc}"
                    else:
                        if not isinstance(data, dict):
                            last_error = f"{path} 응답 최상위가 JSON object가 아닙니다."
                        else:
                            return data, round_trip_ms, None

            if attempt < attempts:
                time.sleep(self._config.retry_backoff_s)

        return None, 0.0, last_error

    def check_health(self) -> HealthResult:
        data, round_trip_ms, error = self._request("GET", HEALTH_PATH)
        if data is None:
            return HealthResult(
                ok=False, status=None, backend=None, model_loaded=None, model_id=None, device=None,
                errors=[], round_trip_ms=round_trip_ms, raw_error=(error or "").removeprefix("__INFERENCE__"),
            )
        return HealthResult(
            ok=data.get("status") == "ok",
            status=data.get("status") if isinstance(data.get("status"), str) else None,
            backend=data.get("backend") if isinstance(data.get("backend"), str) else None,
            model_loaded=bool(data.get("model_loaded", False)),
            model_id=data.get("model_id") if isinstance(data.get("model_id"), str) else None,
            device=data.get("device") if isinstance(data.get("device"), str) else None,
            errors=[str(e) for e in data.get("errors", [])] if isinstance(data.get("errors"), list) else [],
            round_trip_ms=round_trip_ms,
        )

    def session_reset(self, *, session_id: str, task: str) -> bool:
        data, _, error = self._request("POST", SESSION_RESET_PATH, json_body={"session_id": session_id, "task": task})
        if data is None:
            raise VLAClientError(error or "session/reset 실패 (원인 불명)")
        return bool(data.get("ok", False))

    def predict(
        self,
        *,
        session_id: str,
        task: str,
        sequence: int,
        state: dict[str, float],
        images: dict[str, "object"],
    ) -> PredictResult:
        """이미지는 ``runtime.common.vla_contract.CAMERA_KEYS``를 key로 하는
        HWC uint8 RGB numpy 배열 dict여야 한다."""
        encoded_images = {key: _encode_image_base64(arr) for key, arr in images.items()}
        body = {
            "session_id": session_id,
            "task": task,
            "sequence": sequence,
            "timestamp": time.time(),
            "observation": {"state": state, "images": encoded_images},
        }
        data, round_trip_ms, error = self._request("POST", PREDICT_PATH, json_body=body)

        if data is None:
            if error and error.startswith("__INFERENCE__"):
                return PredictResult(
                    ok=False, session_id=session_id, sequence=sequence, action=None,
                    action_schema_valid=False, action_invalid_reason=None, model_id=None, backend=None,
                    inference_latency_ms=None, request_latency_ms=round_trip_ms,
                    error_kind="inference", error_message=error.removeprefix("__INFERENCE__"),
                )
            return PredictResult(
                ok=False, session_id=session_id, sequence=sequence, action=None,
                action_schema_valid=False, action_invalid_reason=None, model_id=None, backend=None,
                inference_latency_ms=None, request_latency_ms=round_trip_ms,
                error_kind="communication", error_message=error,
            )

        action, reason = validate_joint_dict(data.get("action"), context="response.action")
        return PredictResult(
            ok=action is not None,
            session_id=str(data.get("session_id", session_id)),
            sequence=int(data.get("sequence", sequence)),
            action=action,
            action_schema_valid=action is not None,
            action_invalid_reason=reason,
            model_id=data.get("model_id") if isinstance(data.get("model_id"), str) else None,
            backend=data.get("backend") if isinstance(data.get("backend"), str) else None,
            inference_latency_ms=data.get("inference_latency_ms"),
            request_latency_ms=round_trip_ms,
            error_kind=None if action is not None else "schema",
            error_message=None if action is not None else reason,
        )

    def action_ack(self, *, session_id: str, sequence: int, executed: bool, backend: str, note: str | None = None) -> bool:
        data, _, error = self._request(
            "POST",
            ACTION_ACK_PATH,
            json_body={"session_id": session_id, "sequence": sequence, "executed": executed, "backend": backend, "note": note},
        )
        if data is None:
            raise VLAClientError(error or "action/ack 실패 (원인 불명)")
        return bool(data.get("ok", False))

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "VLAHttpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
