"""Shadow Mode가 별도의 Desktop HTTP 서버 없이, 같은 프로세스 안에서 SmolVLA checkpoint를
직접 로딩해 추론하는 VLA client 어댑터.

# 왜 필요한가

기존 Shadow Mode(``scripts/run_shadow_mode.py``)는 ``--vla-server-url``로 이미 떠 있는
Desktop VLA 서버(``scripts/run_vla_server.py --checkpoint ...``)에만 말할 수 있었다 -
Shadow Mode 자체는 어떤 checkpoint를 쓰는지 몰랐고, "이 checkpoint를 평가하고 싶다"는
의도를 CLI로 명시적으로 표현할 방법이 없었다(Desktop을 별도로 띄우는 사람이 실수로 옛날
checkpoint를 쓰고 있어도 Shadow 쪽에서는 알 방법이 없었다). 이 모듈은 그 gap을 메운다 -
``--checkpoint``를 주면 Shadow 프로세스가 직접 checkpoint를 로딩해 추론하므로, "지금 이
프로세스가 정확히 어떤 checkpoint로 실행 중인지"가 항상 명시적이다.

# 재사용 원칙 (요구사항 3번)

추론 backend 자체는 ``runtime/desktop/vla_server.py``의 ``SmolVLAPolicyRunner``를 그대로
재사용한다 - preprocessor/postprocessor 로딩/추론 경로(``get_policy_class`` ->
``from_pretrained`` -> ``make_pre_post_processors`` -> ``preprocessor(batch) ->
policy.select_action(...) -> postprocessor(...)``)를 새로 만들지 않는다. 카메라 rename도
그 경로에 이미 내장되어 있으므로(체크포인트의 저장된 ``policy_preprocessor.json``이
workspace->camera1/wrist->camera2를 담당) 이 모듈은 별도 rename을 하지 않는다.

# HTTP client와의 관계

``runtime.laptop.vla_client.VLAHttpClient``와 정확히 같은 duck-typed 인터페이스
(``check_health``/``session_reset``/``predict``/``action_ack``/``close``)를 제공한다 -
``ShadowModeRunner``는 생성자로 주입된 client가 HTTP인지 in-process인지 구분하지 않는다.
네트워크 왕복이 없으므로 ``request_latency_ms``는 추론 latency와 사실상 같다.

이 모듈은 실물 SO-101에 어떤 write도 하지 않는다 - ``SmolVLAPolicyRunner``와 마찬가지로
관측을 받아 action을 반환하는 순수 추론 어댑터다.
"""

from __future__ import annotations

import time

from runtime.common.vla_contract import validate_joint_dict
from runtime.desktop.vla_server import PolicyInferenceError, SmolVLAPolicyRunner
from runtime.laptop.vla_client import HealthResult, PredictResult


class InProcessVLAClientError(RuntimeError):
    """체크포인트 로딩 실패 (preflight에서 즉시 드러나야 한다)."""


class InProcessSmolVLAClient:
    """``VLAHttpClient``와 같은 인터페이스로 SmolVLA checkpoint를 in-process 호출한다."""

    def __init__(self, *, checkpoint: str, policy_type: str = "smolvla", device: str | None = None) -> None:
        self.checkpoint = checkpoint
        self._policy_runner = SmolVLAPolicyRunner(checkpoint, policy_type=policy_type, device=device)
        self._base_url = f"in-process:{checkpoint}"
        if not self._policy_runner.is_ready():
            raise InProcessVLAClientError(
                f"SmolVLA checkpoint 로딩에 실패했습니다 ({checkpoint}): {self._policy_runner._load_error}"
            )

    def check_health(self) -> HealthResult:
        t0 = time.monotonic()
        ready = self._policy_runner.is_ready()
        round_trip_ms = (time.monotonic() - t0) * 1000.0
        return HealthResult(
            ok=ready,
            status="ok" if ready else "degraded",
            backend=self._policy_runner.backend_name,
            model_loaded=ready,
            model_id=self._policy_runner.model_id,
            device=self._policy_runner.device_label(),
            errors=[] if ready else [f"in-process 로딩 실패: {self._policy_runner._load_error}"],
            round_trip_ms=round_trip_ms,
        )

    def session_reset(self, *, session_id: str, task: str) -> bool:
        self._policy_runner.reset(session_id=session_id, task=task)
        return True

    def predict(
        self,
        *,
        session_id: str,
        task: str,
        sequence: int,
        state: dict[str, float],
        images: dict[str, object],
    ) -> PredictResult:
        """``images``는 HWC uint8 RGB numpy 배열 dict다(``VLAHttpClient.predict``와 동일 계약) -
        HTTP client처럼 base64 JPEG로 인코딩/디코딩하는 왕복이 없으므로 그대로 넘긴다."""
        t0 = time.monotonic()
        try:
            raw_action = self._policy_runner.predict(task=task, state=state, images=images)
        except PolicyInferenceError as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            return PredictResult(
                ok=False,
                session_id=session_id,
                sequence=sequence,
                action=None,
                action_schema_valid=False,
                action_invalid_reason=None,
                model_id=self._policy_runner.model_id,
                backend=self._policy_runner.backend_name,
                inference_latency_ms=elapsed_ms,
                request_latency_ms=elapsed_ms,
                error_kind="inference",
                error_message=str(exc),
            )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        action, reason = validate_joint_dict(raw_action, context="response.action")
        return PredictResult(
            ok=action is not None,
            session_id=session_id,
            sequence=sequence,
            action=action,
            action_schema_valid=action is not None,
            action_invalid_reason=reason,
            model_id=self._policy_runner.model_id,
            backend=self._policy_runner.backend_name,
            inference_latency_ms=elapsed_ms,
            request_latency_ms=elapsed_ms,
            error_kind=None if action is not None else "schema",
            error_message=None if action is not None else reason,
        )

    def action_ack(self, *, session_id: str, sequence: int, executed: bool, backend: str, note: str | None = None) -> bool:
        # in-process에는 별도 원격 bookkeeping이 없다 - VLAHttpClient와 같은 signature만 유지한다.
        return True

    def close(self) -> None:
        return None

    def __enter__(self) -> "InProcessSmolVLAClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
