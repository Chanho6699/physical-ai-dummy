"""Desktop SmolVLA FastAPI 서버 - ``/health``, ``/session/reset``, ``/predict``, ``/action/ack``.

이 모듈은 저장소 조사 결과(``runtime/common/vla_contract.py`` docstring 참고) 새로 설계한
계약을 구현한다. 기존에 physical-ai-dummy 저장소 안에 이 계약을 구현한 코드가 없었다
(최종 보고서 "A/B/C 판정" 참고).

``PolicyRunner``를 교체 가능한 backend로 분리했다 - 동일한 HTTP 계약(라우트/스키마)을
Fake backend(``FakePolicyRunner``, GPU/체크포인트 없이 파이프라인 왕복 검증용)와 Real
backend(``SmolVLAPolicyRunner``, 실제 SmolVLA 체크포인트 추론)가 그대로 공유한다. 이것이
"기존 Fake/Real VLA 공통 HTTP contract" 요구사항을 만족하는 방식이다 - 서버 코드를 두 벌
만들지 않고, 추론 backend만 주입한다.

이 서버는 실물 SO-101에 어떤 write도 하지 않는다. ``/action/ack``도 로깅용 bookkeeping일
뿐 - 이 서버가 어떤 로봇에도 명령을 보내지 않는다 (Shadow Mode에서 실제 실행은 Laptop의
Realistic MuJoCo에서만 일어난다).
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from runtime.common.vla_contract import (
    ACTION_ACK_PATH,
    BACKEND_FAKE,
    BACKEND_SMOLVLA,
    CAMERA_KEYS,
    HEALTH_PATH,
    JOINT_ORDER,
    PREDICT_PATH,
    SCHEMA_VERSION,
    SESSION_RESET_PATH,
    validate_joint_dict,
)

logger = logging.getLogger(__name__)


class PolicyInferenceError(RuntimeError):
    """모델 추론 자체의 실패 (HTTP/통신 실패와 구분 - 섹션 3 요구사항).

    ``/predict``에서 이 예외가 발생하면 HTTP 500 + ``error_kind="inference"``로 응답해
    클라이언트가 "통신은 됐지만 모델이 실패했다"를 구분할 수 있게 한다.
    """


# ---------------------------------------------------------------------------
# PolicyRunner backend 인터페이스
# ---------------------------------------------------------------------------


class PolicyRunner(Protocol):
    """``/predict``가 위임하는 추론 backend. Fake/Real 모두 이 인터페이스를 따른다."""

    backend_name: str
    model_id: str

    def is_ready(self) -> bool: ...

    def reset(self, *, session_id: str, task: str) -> None: ...

    def predict(self, *, task: str, state: dict[str, float], images: dict[str, "object"]) -> dict[str, float]: ...

    def device_label(self) -> str: ...


@dataclass
class FakePolicyRunner:
    """체크포인트/GPU 없이 HTTP 왕복(통신 계약)만 검증하기 위한 backend.

    기본 동작은 "현재 state를 그대로 유지"(identity)다 - 6개 관절 dict를 그대로 반환한다.
    ``joint_offsets``를 주면 테스트에서 action != state를 쉽게 만들 수 있다 (예: Safety Gate
    WOULD_CLAMP 케이스 재현).
    """

    backend_name: str = BACKEND_FAKE
    model_id: str = "fake-identity-v1"
    joint_offsets: dict[str, float] = field(default_factory=dict)
    reset_calls: list[tuple[str, str]] = field(default_factory=list, repr=False)

    def is_ready(self) -> bool:
        return True

    def reset(self, *, session_id: str, task: str) -> None:
        self.reset_calls.append((session_id, task))

    def predict(self, *, task: str, state: dict[str, float], images: dict[str, object]) -> dict[str, float]:
        return {joint: state[joint] + self.joint_offsets.get(joint, 0.0) for joint in JOINT_ORDER}

    def device_label(self) -> str:
        return "cpu (fake)"


class SmolVLAPolicyRunner:
    """실제 SmolVLA 체크포인트를 로딩해 추론하는 backend.

    [미검증] 이 세션에서는 로컬에 파인튜닝된 SmolVLA 체크포인트도, GPU도 없어 이
    backend를 실제 체크포인트로 실행/검증하지 못했다 (최종 보고서 참고). 구현은
    LeRobot의 공식 real-robot/원격 추론 경로(``lerobot.async_inference.policy_server``,
    ``lerobot.policies.factory.get_policy_class``/``make_pre_post_processors``,
    ``lerobot/policies/utils.py``의 ``prepare_observation_for_inference``)를 그대로
    참고해 만들었다 - 정규화/역정규화를 직접 재구현하지 않고 정책이 로딩하는
    preprocessor/postprocessor pipeline에 위임한다.

    관절 순서는 ``JOINT_ORDER``(이 저장소 전역 관례)를 그대로 신뢰한다 - 체크포인트가
    반환하는 action 텐서의 마지막 차원이 6이 아니면(schema mismatch) 추측으로 채우지
    않고 ``PolicyInferenceError``를 던진다 (섹션 4: "schema가 확실하지 않으면 inference를
    진행하지 말고 명확한 diagnostic failure로 처리").
    """

    backend_name = BACKEND_SMOLVLA

    def __init__(self, checkpoint: str, *, policy_type: str = "smolvla", device: str | None = None) -> None:
        self.checkpoint = checkpoint
        self.model_id = checkpoint
        self.policy_type = policy_type
        self._device_arg = device
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._device = None
        self._load_error: str | None = None
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - 로딩 실패를 서버 기동 실패가 아니라 degraded로 다룬다
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("SmolVLA 체크포인트 로딩 실패 (checkpoint=%s)", checkpoint)

    def _load(self) -> None:
        import torch
        from lerobot.policies import get_policy_class, make_pre_post_processors

        policy_cls = get_policy_class(self.policy_type)
        self._policy = policy_cls.from_pretrained(self.checkpoint)
        self._device = torch.device(self._device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._policy.to(self._device)
        self._policy.eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config, pretrained_path=self.checkpoint
        )

    def is_ready(self) -> bool:
        return self._policy is not None and self._load_error is None

    def reset(self, *, session_id: str, task: str) -> None:
        if self._policy is not None:
            self._policy.reset()

    def predict(self, *, task: str, state: dict[str, float], images: dict[str, object]) -> dict[str, float]:
        if not self.is_ready():
            raise PolicyInferenceError(f"정책이 로딩되지 않았습니다: {self._load_error}")
        import numpy as np
        import torch

        try:
            state_arr = np.array([state[j] for j in JOINT_ORDER], dtype=np.float32)
            batch: dict[str, object] = {
                "observation.state": torch.from_numpy(state_arr).unsqueeze(0).to(self._device)
            }
            for key, image_array in images.items():
                img = torch.from_numpy(image_array)
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                img = img.permute(2, 0, 1).contiguous()
                batch[key] = img.unsqueeze(0).to(self._device)
            batch["task"] = [task]

            with torch.inference_mode():
                processed = self._preprocessor(batch)
                raw_action = self._policy.select_action(processed)
                action = self._postprocessor(raw_action)
        except PolicyInferenceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PolicyInferenceError(f"SmolVLA 추론 중 오류: {type(exc).__name__}: {exc}") from exc

        action_flat = action.detach().to("cpu").reshape(-1)
        if action_flat.shape[0] != len(JOINT_ORDER):
            raise PolicyInferenceError(
                f"정책 출력 차원이 예상과 다릅니다: {action_flat.shape[0]} (기대={len(JOINT_ORDER)}). "
                "체크포인트의 action feature 구성을 확인하세요."
            )
        return {joint: float(action_flat[i]) for i, joint in enumerate(JOINT_ORDER)}

    def device_label(self) -> str:
        return str(self._device) if self._device is not None else "unknown"


# ---------------------------------------------------------------------------
# 이미지 디코딩 (base64 jpg -> HWC uint8 numpy)
# ---------------------------------------------------------------------------


class ImageDecodeError(ValueError):
    """observation.images.* base64 페이로드 디코딩 실패."""


def decode_base64_image(data: str, *, context: str):
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ImageDecodeError(f"{context}: base64 디코딩 실패: {exc}") from exc
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ImageDecodeError(f"{context}: 유효한 JPEG 이미지가 아닙니다: {exc}") from exc
    array = np.array(img, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageDecodeError(f"{context}: 이미지 shape이 HWC*3이 아닙니다: {array.shape}")
    return array


# ---------------------------------------------------------------------------
# HTTP 스키마
# ---------------------------------------------------------------------------


class HealthResponseModel(BaseModel):
    status: str
    backend: str
    model_loaded: bool
    model_id: str
    device: str
    schema_version: str
    timestamp: float
    errors: list[str] = []


class SessionResetRequestModel(BaseModel):
    session_id: str
    task: str


class SessionResetResponseModel(BaseModel):
    session_id: str
    ok: bool


class ObservationModel(BaseModel):
    state: dict[str, float]
    images: dict[str, str]  # camera_key -> base64 jpg


class PredictRequestModel(BaseModel):
    session_id: str
    task: str
    sequence: int
    timestamp: float
    observation: ObservationModel


class PredictResponseModel(BaseModel):
    session_id: str
    sequence: int
    action: dict[str, float]
    model_id: str
    backend: str
    inference_latency_ms: float
    server_received_at: float
    server_responded_at: float


class ActionAckRequestModel(BaseModel):
    session_id: str
    sequence: int
    executed: bool
    backend: str
    note: str | None = None


class ActionAckResponseModel(BaseModel):
    ok: bool


API_DESCRIPTION = (
    "Desktop SmolVLA 추론 서버입니다. 실물 SO-101에 어떤 write도 하지 않습니다 - "
    "관측(observation)을 받아 action을 반환하는 순수 추론 API입니다."
)


def create_app(*, policy_runner: PolicyRunner, api_token: str | None = None) -> FastAPI:
    app = FastAPI(title="Desktop SmolVLA VLA Server", description=API_DESCRIPTION, version="0.1.0")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if api_token is None:
            return
        if authorization != f"Bearer {api_token}":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증 토큰이 없거나 올바르지 않습니다.")

    last_acks: list[dict] = []

    @app.get(HEALTH_PATH, response_model=HealthResponseModel, tags=["vla"])
    def get_health(_: None = Depends(require_token)) -> HealthResponseModel:
        ready = policy_runner.is_ready()
        return HealthResponseModel(
            status="ok" if ready else "degraded",
            backend=policy_runner.backend_name,
            model_loaded=ready,
            model_id=policy_runner.model_id,
            device=policy_runner.device_label(),
            schema_version=SCHEMA_VERSION,
            timestamp=time.time(),
            errors=[] if ready else ["policy_runner가 준비되지 않았습니다 (체크포인트 로딩 실패 가능)."],
        )

    @app.post(SESSION_RESET_PATH, response_model=SessionResetResponseModel, tags=["vla"])
    def post_session_reset(
        req: SessionResetRequestModel, _: None = Depends(require_token)
    ) -> SessionResetResponseModel:
        policy_runner.reset(session_id=req.session_id, task=req.task)
        return SessionResetResponseModel(session_id=req.session_id, ok=True)

    @app.post(PREDICT_PATH, response_model=PredictResponseModel, tags=["vla"])
    def post_predict(req: PredictRequestModel, _: None = Depends(require_token)) -> PredictResponseModel:
        received_at = time.time()

        state, reason = validate_joint_dict(req.observation.state, context="observation.state")
        if state is None:
            raise HTTPException(status_code=422, detail=f"observation.state가 올바르지 않습니다: {reason}")

        missing_cams = [k for k in CAMERA_KEYS if k not in req.observation.images]
        if missing_cams:
            raise HTTPException(status_code=422, detail=f"observation.images에 다음 카메라가 없습니다: {missing_cams}")

        images = {}
        try:
            for key in CAMERA_KEYS:
                images[key] = decode_base64_image(req.observation.images[key], context=f"observation.images.{key}")
        except ImageDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        t0 = time.perf_counter()
        try:
            action = policy_runner.predict(task=req.task, state=state, images=images)
        except PolicyInferenceError as exc:
            # 통신(HTTP)은 성공했지만 모델 추론 자체가 실패한 경우 - 500이지만 detail에
            # "inference"임을 명시해 클라이언트가 통신 실패와 구분할 수 있게 한다 (섹션 3).
            raise HTTPException(status_code=500, detail=f"[inference_error] {exc}") from exc
        inference_latency_ms = (time.perf_counter() - t0) * 1000.0

        validated_action, action_reason = validate_joint_dict(action, context="action")
        if validated_action is None:
            raise HTTPException(status_code=500, detail=f"[inference_error] 정책 출력 action이 올바르지 않습니다: {action_reason}")

        return PredictResponseModel(
            session_id=req.session_id,
            sequence=req.sequence,
            action=validated_action,
            model_id=policy_runner.model_id,
            backend=policy_runner.backend_name,
            inference_latency_ms=inference_latency_ms,
            server_received_at=received_at,
            server_responded_at=time.time(),
        )

    @app.post(ACTION_ACK_PATH, response_model=ActionAckResponseModel, tags=["vla"])
    def post_action_ack(req: ActionAckRequestModel, _: None = Depends(require_token)) -> ActionAckResponseModel:
        # 이 서버는 ack를 받아도 어떤 로봇에도 명령을 보내지 않는다 - 로깅용 bookkeeping뿐이다.
        last_acks.append(req.model_dump())
        logger.info(
            "action/ack session=%s sequence=%s executed=%s backend=%s",
            req.session_id,
            req.sequence,
            req.executed,
            req.backend,
        )
        return ActionAckResponseModel(ok=True)

    app.state.last_acks = last_acks  # 테스트에서 조회용
    return app
