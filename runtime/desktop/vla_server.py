"""Desktop SmolVLA FastAPI 서버 - ``/health``, ``/session/reset``, ``/predict``, ``/action/ack``,
``/predict_chunk``.

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

# Phase C-1A (2026-08) - ``/predict_chunk`` (순수 추가, ``/predict``는 무변경)

기존 ``/predict``는 SmolVLA의 action queue(``select_action()``)를 거쳐 chunk[0] 하나만
반환한다 - staged safety 검증 경로가 이미 이걸 신뢰하고 있으므로(``runtime/laptop/
staged_real_rollout.py``, 이 파일에서 절대 건드리지 않음) 그 동작은 그대로 둔다.

Full Pick & Drop용 향후 async executor는 "action 1개"가 아니라 "timestamp가 붙은 전체
50-step chunk"가 필요하다 - 그래서 별도 ``/predict_chunk`` 엔드포인트를 추가한다.
``SmolVLAPolicy.predict_action_chunk()``(이미 public, action queue를 전혀 안 건드림)를
호출해 매번 fresh chunk를 만들고, 그 원본(정규화된) 텐서를 반드시 기존 postprocessor에
통과시켜 degree/percent 절대좌표로 변환한다(단위 변환을 새로 발명하지 않는다 - ``/predict``와
동일한 postprocessor 재사용).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    PREDICT_CHUNK_PATH,
    PREDICT_PATH,
    SCHEMA_VERSION,
    SESSION_RESET_PATH,
    validate_action_chunk,
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
    """``/predict``가 위임하는 추론 backend. Fake/Real 모두 이 인터페이스를 따른다.

    ``chunk_index_spacing_s``/``predict_chunk()``는 Phase C-1A에서 추가된 ``/predict_chunk``
    전용 확장이다 - 기존 ``predict()`` 계약은 그대로다.
    """

    backend_name: str
    model_id: str
    # /predict_chunk 전용: chunk[k]와 chunk[k+1] 사이 시간 간격(초). 이 값을 못 구하면 None -
    # predict_chunk()가 그 시점에 fail-closed로 거부한다(초기화 시점에 서버 전체를 죽이지
    # 않음 - /predict는 이 값과 무관하게 계속 동작해야 하므로).
    chunk_index_spacing_s: float | None
    inference_seed: int | None

    def is_ready(self) -> bool: ...

    def reset(self, *, session_id: str, task: str) -> None: ...

    def predict(self, *, task: str, state: dict[str, float], images: dict[str, "object"]) -> dict[str, float]: ...

    def predict_chunk(
        self, *, task: str, state: dict[str, float], images: dict[str, "object"]
    ) -> list[dict[str, float]]: ...

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
    # /predict_chunk 전용 설정 (기본값은 이 저장소 dataset 관례인 30fps -> 1/30s 간격을
    # 테스트 편의를 위해 명시적으로 못박아 둔 것뿐이다 - SmolVLAPolicyRunner의 실제 구현은
    # 이 값을 하드코딩하지 않고 dataset 메타데이터에서 구한다, 아래 참고).
    chunk_size: int = 50
    chunk_index_spacing_s: float = 1.0 / 30.0
    inference_seed: int | None = None

    def is_ready(self) -> bool:
        return True

    def reset(self, *, session_id: str, task: str) -> None:
        self.reset_calls.append((session_id, task))

    def predict(self, *, task: str, state: dict[str, float], images: dict[str, object]) -> dict[str, float]:
        return {joint: state[joint] + self.joint_offsets.get(joint, 0.0) for joint in JOINT_ORDER}

    def predict_chunk(self, *, task: str, state: dict[str, float], images: dict[str, object]) -> list[dict[str, float]]:
        # k=0..chunk_size-1, chunk[k] = state + joint_offsets*(k+1) - k=0일 때 predict()의
        # 단일 action(state+joint_offsets)과 정확히 같은 값이 되게 해서(k+1에서 k=0 -> *1),
        # 테스트에서 "chunk[0]이 predict()의 action과 같은 semantic"임을 쉽게 확인할 수 있다.
        return [
            {joint: state[joint] + self.joint_offsets.get(joint, 0.0) * (k + 1) for joint in JOINT_ORDER}
            for k in range(self.chunk_size)
        ]

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

    def __init__(
        self,
        checkpoint: str,
        *,
        policy_type: str = "smolvla",
        device: str | None = None,
        dataset_fps: float | None = None,
        inference_seed: int | None = None,
    ) -> None:
        """``dataset_fps``: ``/predict_chunk``의 ``chunk_index_spacing_s`` 산출에 쓰는 학습
        dataset의 fps(예: V3/V4/combined69는 30). 명시적으로 주면 이게 최우선 - 안 주면
        ``_resolve_chunk_index_spacing_s()``가 checkpoint의 ``train_config.json``에 기록된
        학습 dataset root(``dataset.root``)의 ``meta/info.json``에서 best-effort로 읽는다
        (이 checkpoint를 학습한 바로 그 머신/경로에서 서빙할 때만 성공 - 다른 머신이면
        실패할 수 있음, 그 경우 ``chunk_index_spacing_s``는 ``None``으로 남고
        ``predict_chunk()``가 호출 시점에 명확히 fail-closed 에러를 던진다. 30을 어디에도
        암묵적으로 하드코딩하지 않는다 - 요구사항)."""
        self.checkpoint = checkpoint
        self.model_id = checkpoint
        self.policy_type = policy_type
        self._device_arg = device
        self._dataset_fps_override = dataset_fps
        if inference_seed is not None and inference_seed < 0:
            raise ValueError(f"inference_seed must be non-negative, got {inference_seed}")
        self.inference_seed = inference_seed
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._device = None
        self._load_error: str | None = None
        # /predict와 /predict_chunk가 같은 self._policy 객체(내부 action queue 포함)를
        # 공유하므로, "preprocess 직후 policy inference + postprocess" 구간만 이 lock으로
        # 직렬화한다 (섹션 3 요구사항 - lock 범위를 최소로).
        self._lock = threading.Lock()
        self.chunk_index_spacing_s: float | None = None
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - 로딩 실패를 서버 기동 실패가 아니라 degraded로 다룬다
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("SmolVLA 체크포인트 로딩 실패 (checkpoint=%s)", checkpoint)
        else:
            self.chunk_index_spacing_s = self._resolve_chunk_index_spacing_s()

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

    def _resolve_chunk_index_spacing_s(self) -> float | None:
        """1/fps를 구한다. 명시적 ``dataset_fps``가 최우선이고, 없으면 checkpoint 옆
        ``train_config.json``의 ``dataset.root``가 가리키는 실제 dataset의
        ``meta/info.json["fps"]``를 best-effort로 읽는다. 어느 쪽도 안 되면 ``None`` -
        호출자가 이 값 없이 ``/predict_chunk``를 쓰지 못하게 막는다(fail-closed).
        조사 결과: checkpoint 자신의 ``config.json``/``train_config.json`` 최상위에는 fps
        필드가 전혀 없다(`grep` 결과 0건) - fps는 오직 dataset 메타데이터에만 있다."""
        if self._dataset_fps_override is not None:
            if self._dataset_fps_override <= 0:
                return None
            return 1.0 / self._dataset_fps_override
        try:
            train_config_path = Path(self.checkpoint) / "train_config.json"
            if not train_config_path.is_file():
                return None
            train_config = json.loads(train_config_path.read_text())
            dataset_root = train_config.get("dataset", {}).get("root")
            if not dataset_root:
                return None
            info_path = Path(dataset_root) / "meta" / "info.json"
            if not info_path.is_file():
                return None
            info = json.loads(info_path.read_text())
            fps = info.get("fps")
            if fps is None or float(fps) <= 0:
                return None
            return 1.0 / float(fps)
        except Exception:  # noqa: BLE001 - best-effort 추론일 뿐 - 실패하면 조용히 None
            return None

    def is_ready(self) -> bool:
        return self._policy is not None and self._load_error is None

    def reset(self, *, session_id: str, task: str) -> None:
        if self._policy is not None:
            self._policy.reset()

    def _build_batch(self, *, task: str, state: dict[str, float], images: dict[str, object]) -> dict[str, object]:
        """``predict()``/``predict_chunk()`` 공용 관측 전처리 - numpy/tensor 조립만 하고
        정책 객체는 전혀 건드리지 않으므로 lock 밖에서 안전하게 호출할 수 있다."""
        import numpy as np
        import torch

        state_arr = np.array([state[j] for j in JOINT_ORDER], dtype=np.float32)
        batch: dict[str, object] = {"observation.state": torch.from_numpy(state_arr).unsqueeze(0).to(self._device)}
        for key, image_array in images.items():
            img = torch.from_numpy(image_array)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            img = img.permute(2, 0, 1).contiguous()
            batch[key] = img.unsqueeze(0).to(self._device)
        batch["task"] = [task]
        return batch

    def _deterministic_noise(self, *, batch_size: int):
        """Build fixed SmolVLA flow-matching x_1 without mutating global RNG state."""
        if self.inference_seed is None:
            return None
        import torch

        generator = torch.Generator(device=self._device)
        generator.manual_seed(self.inference_seed)
        return torch.randn(
            (batch_size, self._policy.config.chunk_size, self._policy.config.max_action_dim),
            dtype=torch.float32, device=self._device, generator=generator,
        )

    def predict(self, *, task: str, state: dict[str, float], images: dict[str, object]) -> dict[str, float]:
        if not self.is_ready():
            raise PolicyInferenceError(f"정책이 로딩되지 않았습니다: {self._load_error}")
        import torch

        try:
            batch = self._build_batch(task=task, state=state, images=images)

            with self._lock, torch.inference_mode():
                processed = self._preprocessor(batch)
                noise = self._deterministic_noise(batch_size=processed["observation.state"].shape[0])
                raw_action = self._policy.select_action(processed, noise=noise)
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

    def predict_chunk(self, *, task: str, state: dict[str, float], images: dict[str, object]) -> list[dict[str, float]]:
        """전체 SmolVLA action chunk(fresh, unnormalized)를 반환한다.

        ``select_action()``이 아니라 반드시 ``predict_action_chunk()``를 호출한다 - 그래야
        ``/predict``(``select_action`` 기반 action queue)와 무관하게 매번 이 관측에서 새로
        추론한 전체 chunk를 얻는다(action queue를 소비/오염시키지 않음 - 섹션 요구사항)."""
        if not self.is_ready():
            raise PolicyInferenceError(f"정책이 로딩되지 않았습니다: {self._load_error}")
        if self.chunk_index_spacing_s is None:
            raise PolicyInferenceError(
                "chunk_index_spacing_s(=1/dataset_fps)를 알 수 없어 /predict_chunk를 제공할 수 "
                "없습니다 - SmolVLAPolicyRunner(..., dataset_fps=...)로 명시하거나, 이 "
                "checkpoint의 train_config.json이 가리키는 dataset root가 이 머신에서 유효해야 "
                "합니다 (fail-closed: 30 같은 값으로 암묵적 fallback하지 않습니다)."
            )
        import torch

        try:
            batch = self._build_batch(task=task, state=state, images=images)

            with self._lock, torch.inference_mode():
                processed = self._preprocessor(batch)
                noise = self._deterministic_noise(batch_size=processed["observation.state"].shape[0])
                raw_chunk = self._policy.predict_action_chunk(processed, noise=noise)  # select_action()이 아님!
                chunk = self._postprocessor(raw_chunk)
        except PolicyInferenceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PolicyInferenceError(f"SmolVLA chunk 추론 중 오류: {type(exc).__name__}: {exc}") from exc

        chunk_cpu = chunk.detach().to("cpu")
        # chunk_size를 50으로 하드코딩하지 않고 checkpoint의 실제 config에서 가져온다.
        expected_chunk_size = self._policy.config.chunk_size
        expected_action_dim = len(JOINT_ORDER)
        if chunk_cpu.ndim != 3:
            raise PolicyInferenceError(
                f"chunk 텐서가 3차원(batch, chunk_size, action_dim)이 아닙니다: shape={tuple(chunk_cpu.shape)}"
            )
        batch_size, actual_chunk_size, action_dim = chunk_cpu.shape
        if batch_size != 1:
            raise PolicyInferenceError(f"chunk 텐서의 batch 차원이 1이 아닙니다: {batch_size}")
        if actual_chunk_size != expected_chunk_size:
            raise PolicyInferenceError(
                f"chunk 길이가 checkpoint config.chunk_size와 다릅니다: {actual_chunk_size} "
                f"(기대={expected_chunk_size})"
            )
        if action_dim != expected_action_dim:
            raise PolicyInferenceError(
                f"chunk의 action 차원이 예상과 다릅니다: {action_dim} (기대={expected_action_dim}). "
                "체크포인트의 action feature 구성을 확인하세요."
            )

        chunk_list: list[dict[str, float]] = []
        for k in range(actual_chunk_size):
            step = chunk_cpu[0, k]
            chunk_list.append({joint: float(step[i]) for i, joint in enumerate(JOINT_ORDER)})
        return chunk_list

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
    inference_mode: str
    inference_seed: int | None
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
    request_id: str | None = None


class PredictResponseModel(BaseModel):
    session_id: str
    sequence: int
    action: dict[str, float]
    model_id: str
    backend: str
    inference_latency_ms: float
    server_received_at: float
    server_responded_at: float


class PredictChunkResponseModel(BaseModel):
    """``/predict_chunk`` 응답 - ``/predict``의 ``PredictRequestModel``을 요청 body로 그대로
    재사용한다(요구사항). ``chunk``는 postprocessor를 거친 절대좌표(degree/percent_0_100) -
    ``/predict``의 ``action`` 필드와 완전히 같은 단위계다."""

    session_id: str
    sequence: int
    chunk: list[dict[str, float]]  # 길이 = chunk_size. chunk[k] = k번째 미래 시점의 절대좌표 target.
    chunk_index_spacing_s: float  # chunk[k]와 chunk[k+1] 사이 시간 간격(초) = 1/dataset_fps
    chunk_size: int  # len(chunk)와 항상 같음 - 클라이언트가 별도로 세지 않아도 되게 명시
    model_id: str
    backend: str
    inference_latency_ms: float
    server_received_at: float
    server_responded_at: float
    input_diagnostic: dict[str, object] | None = None


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


def create_app(
    *, policy_runner: PolicyRunner, api_token: str | None = None,
    input_diagnostic_dir: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Desktop SmolVLA VLA Server", description=API_DESCRIPTION, version="0.1.0")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        if api_token is None:
            return
        if authorization != f"Bearer {api_token}":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증 토큰이 없거나 올바르지 않습니다.")

    last_acks: list[dict] = []
    diagnostic_lock = threading.Lock()
    diagnostic_path = (
        Path(input_diagnostic_dir) / "server_input_diagnostics.jsonl"
        if input_diagnostic_dir is not None else None
    )
    if diagnostic_path is not None:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)

    def build_input_diagnostic(req: PredictRequestModel, images: dict[str, object]) -> dict[str, object]:
        import numpy as np

        cameras: dict[str, object] = {}
        for key in CAMERA_KEYS:
            jpeg_bytes = base64.b64decode(req.observation.images[key], validate=True)
            arr = np.asarray(images[key])
            cameras[key] = {
                "jpeg_sha256": hashlib.sha256(jpeg_bytes).hexdigest(),
                "decoded_rgb_shape": list(arr.shape),
                "decoded_rgb_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "decoded_rgb_sum": int(arr.astype(np.uint64).sum()),
            }
        return {
            "request_id": req.request_id,
            "session_id": req.session_id,
            "sequence": req.sequence,
            "server_received_at": time.time(),
            "cameras": cameras,
        }

    def persist_input_diagnostic(record: dict[str, object]) -> None:
        if diagnostic_path is None:
            return
        with diagnostic_lock:
            with diagnostic_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @app.get(HEALTH_PATH, response_model=HealthResponseModel, tags=["vla"])
    def get_health(_: None = Depends(require_token)) -> HealthResponseModel:
        ready = policy_runner.is_ready()
        return HealthResponseModel(
            status="ok" if ready else "degraded",
            backend=policy_runner.backend_name,
            model_loaded=ready,
            model_id=policy_runner.model_id,
            device=policy_runner.device_label(),
            inference_mode=("deterministic" if policy_runner.inference_seed is not None else "stochastic"),
            inference_seed=policy_runner.inference_seed,
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

    @app.post(PREDICT_CHUNK_PATH, response_model=PredictChunkResponseModel, tags=["vla"])
    def post_predict_chunk(req: PredictRequestModel, _: None = Depends(require_token)) -> PredictChunkResponseModel:
        """Phase C-1A - 순수 추가 엔드포인트. 위 ``post_predict``와 요청 파싱/에러 처리
        구조는 의도적으로 동일하게 맞췄다(같은 검증 순서/같은 422·500 관례) - 다만 이
        핸들러는 ``post_predict``를 호출하거나 그 코드를 공유하지 않는다(완전히 독립),
        그래서 이 함수 안의 어떤 변경도 ``/predict``에 영향을 줄 수 없다."""
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

        input_diagnostic = (
            build_input_diagnostic(req, images) if req.request_id is not None else None
        )

        t0 = time.perf_counter()
        try:
            chunk = policy_runner.predict_chunk(task=req.task, state=state, images=images)
        except PolicyInferenceError as exc:
            raise HTTPException(status_code=500, detail=f"[inference_error] {exc}") from exc
        inference_latency_ms = (time.perf_counter() - t0) * 1000.0

        validated_chunk, chunk_reason = validate_action_chunk(chunk, context="chunk")
        if validated_chunk is None:
            raise HTTPException(status_code=500, detail=f"[inference_error] 정책 출력 chunk가 올바르지 않습니다: {chunk_reason}")

        spacing_s = policy_runner.chunk_index_spacing_s
        if spacing_s is None or spacing_s <= 0:
            # predict_chunk() 자체가 이미 이 경우 PolicyInferenceError를 던지도록 구현했지만
            # (SmolVLAPolicyRunner), 다른 PolicyRunner 구현이 그 규칙을 안 지킬 가능성까지
            # 방어한다 - 여기서도 fail-closed.
            raise HTTPException(
                status_code=500,
                detail="[inference_error] policy_runner.chunk_index_spacing_s를 알 수 없습니다 (fail-closed).",
            )

        if input_diagnostic is not None:
            input_diagnostic["raw_action_chunk"] = validated_chunk
            persist_input_diagnostic(input_diagnostic)

        return PredictChunkResponseModel(
            session_id=req.session_id,
            sequence=req.sequence,
            chunk=validated_chunk,
            chunk_index_spacing_s=spacing_s,
            chunk_size=len(validated_chunk),
            model_id=policy_runner.model_id,
            backend=policy_runner.backend_name,
            inference_latency_ms=inference_latency_ms,
            server_received_at=received_at,
            server_responded_at=time.time(),
            input_diagnostic=input_diagnostic,
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
