"""Shadow Mode v1 오케스트레이터 - single-step (섹션 11, 16).

    real observation 1회 -> SmolVLA inference 1회 -> action 1회 -> Safety ->
    Realistic MuJoCo 실행 -> Validation -> STOP

이 러너는 실물 SO-101 follower에 어떤 write도 하지 않는다. ``state_source``는
``ReadOnlyRealFollowerStateSource``(또는 테스트용 fake)만 주입 가능하고, 이 클래스
자체에는 ``send_action``류의 실행 경로가 존재하지 않는다 - "이 파일을 아무리 잘못 고쳐도
실물에 쓸 방법이 없다"는 섹션 17 원칙을 그대로 따른다.

모든 외부 의존성(카메라/state/VLA client/MuJoCo backend)은 생성자로 주입한다 -
``tests/test_shadow_mode_runner.py``가 fake로 교체해 오프라인으로 전체 파이프라인을
검증할 수 있게 하기 위함이다 (섹션 20).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from runtime.laptop.action_adapter import AdaptedAction, adapt_vla_action
from runtime.laptop.camera_source import CameraFrame
from runtime.laptop.follower_state_source import REAL_FOLLOWER_WRITE_COUNT, FollowerStateSnapshot
from runtime.laptop.mujoco_shadow_backend import MuJoCoBackendError, MuJoCoStepResult, RealisticMuJoCoBackend
from runtime.laptop.observation_builder import BuiltObservation, build_observation
from runtime.laptop.safety_gate import SafetyDecision, SafetyGate
from runtime.laptop.shadow_logger import (
    RESULT_SHADOW_FAIL,
    RESULT_SHADOW_PASS,
    RESULT_SHADOW_PASS_WITH_CLAMP,
    build_report,
    make_session_id,
    write_report,
)
from runtime.laptop.shadow_validator import ShadowValidationResult, validate_mujoco_step
from runtime.laptop.vla_client import PredictResult, VLAHttpClient

BACKEND_LABEL = "realistic_mujoco"
DEFAULT_STATE_STALE_AFTER_MS = 500.0


class CameraSourceProtocol(Protocol):
    def capture_all(self) -> dict[str, CameraFrame]: ...


class FollowerStateSourceProtocol(Protocol):
    def read(self) -> FollowerStateSnapshot: ...


class CameraReadError(RuntimeError):
    pass


class StateReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShadowRunResult:
    session_id: str
    result: str  # SHADOW_PASS | SHADOW_PASS_WITH_CLAMP | SHADOW_FAIL
    reasons: tuple[str, ...]
    report: dict
    report_path: Path
    real_follower_write_count: int


class ShadowModeRunner:
    def __init__(
        self,
        *,
        camera_source: CameraSourceProtocol,
        state_source: FollowerStateSourceProtocol,
        vla_client: VLAHttpClient,
        safety_gate: SafetyGate,
        mujoco_backend: RealisticMuJoCoBackend,
        task: str,
        reports_dir: Path | None = None,
        state_stale_after_ms: float = DEFAULT_STATE_STALE_AFTER_MS,
    ) -> None:
        self._camera_source = camera_source
        self._state_source = state_source
        self._vla_client = vla_client
        self._safety_gate = safety_gate
        self._mujoco_backend = mujoco_backend
        self._task = task
        self._reports_dir = reports_dir
        self._state_stale_after_ms = state_stale_after_ms

    def run_once(self) -> ShadowRunResult:
        session_id = make_session_id()
        sequence = 0
        reasons: list[str] = []

        communication: dict = {
            "server_url": getattr(self._vla_client, "_base_url", None),
            "health_ok": None,
            "request_latency_ms": None,
            "inference_latency_ms": None,
            "round_trip_latency_ms": None,
            "error_kind": None,
            "error_message": None,
        }
        vla_report: dict = {"raw_action": None, "action_schema_valid": None, "model_id": None, "backend": None}
        adapter_report: dict = {"mapped_action": None, "warnings": []}
        safety_report: dict = {"decision": None, "reasons": [], "would_clamp": None, "safe_action": None}
        mujoco_report: dict = {"initial_state": None, "final_state": None, "realistic_layer_diagnostics": None}
        validation_report: dict = {"passed": None, "warnings": [], "failures": [], "checks": {}}

        # -- 1. 실제 observation (카메라 + 팔로워 state, read-only) --------------------
        try:
            camera_frames = self._camera_source.capture_all()
        except Exception as exc:  # noqa: BLE001
            camera_frames = {}
            reasons.append(f"카메라 캡처 실패: {exc}")

        state_snapshot: FollowerStateSnapshot | None = None
        state_stale = False
        state_stale_reason: str | None = None
        try:
            state_snapshot = self._state_source.read()
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"팔로워 state 읽기 실패: {exc}")

        if state_snapshot is not None:
            age_ms = (time.time() - state_snapshot.read_at_wall) * 1000.0
            if age_ms > self._state_stale_after_ms:
                state_stale = True
                state_stale_reason = f"state age {age_ms:.1f}ms > {self._state_stale_after_ms}ms"

        observation_dict = {
            "fixed_camera_valid": False,
            "wrist_camera_valid": False,
            "state_valid": False,
            "schema_valid": False,
            "invalid_reasons": list(reasons),
        }

        built: BuiltObservation | None = None
        if state_snapshot is not None:
            built = build_observation(state_snapshot=state_snapshot, camera_frames=camera_frames)
            observation_dict = {
                "fixed_camera_valid": built.fixed_camera_valid,
                "wrist_camera_valid": built.wrist_camera_valid,
                "state_valid": built.state_valid,
                "schema_valid": built.schema_valid,
                "invalid_reasons": list(built.invalid_reasons),
            }

        if built is None or not built.schema_valid:
            all_reasons = reasons + (list(built.invalid_reasons) if built is not None else [])
            return self._finalize(
                session_id=session_id, task=self._task, observation=observation_dict, communication=communication,
                vla=vla_report, adapter=adapter_report, safety=safety_report, mujoco=mujoco_report,
                validation=validation_report, result=RESULT_SHADOW_FAIL, reasons=all_reasons or ["관측 스키마가 유효하지 않습니다."],
            )

        # -- 2. Desktop SmolVLA와 통신 ------------------------------------------------
        health = self._vla_client.check_health()
        communication["health_ok"] = health.ok
        communication["round_trip_latency_ms"] = health.round_trip_ms
        if not health.ok:
            communication["error_kind"] = "communication"
            communication["error_message"] = health.raw_error or f"health.status={health.status!r}"
            return self._finalize(
                session_id=session_id, task=self._task, observation=observation_dict, communication=communication,
                vla=vla_report, adapter=adapter_report, safety=safety_report, mujoco=mujoco_report,
                validation=validation_report, result=RESULT_SHADOW_FAIL,
                reasons=[f"Desktop VLA 서버에 연결할 수 없습니다: {communication['error_message']}"],
            )

        try:
            self._vla_client.session_reset(session_id=session_id, task=self._task)
        except Exception as exc:  # noqa: BLE001 - session reset 실패는 predict 자체를 막지 않는다 (best-effort)
            communication["session_reset_error"] = str(exc)

        predict: PredictResult = self._vla_client.predict(
            session_id=session_id, task=self._task, sequence=sequence, state=built.state, images=built.images
        )
        communication["request_latency_ms"] = predict.request_latency_ms
        communication["inference_latency_ms"] = predict.inference_latency_ms
        vla_report["raw_action"] = predict.action
        vla_report["action_schema_valid"] = predict.action_schema_valid
        vla_report["model_id"] = predict.model_id
        vla_report["backend"] = predict.backend

        if not predict.ok:
            communication["error_kind"] = predict.error_kind
            communication["error_message"] = predict.error_message
            return self._finalize(
                session_id=session_id, task=self._task, observation=observation_dict, communication=communication,
                vla=vla_report, adapter=adapter_report, safety=safety_report, mujoco=mujoco_report,
                validation=validation_report, result=RESULT_SHADOW_FAIL,
                reasons=[f"VLA {predict.error_kind} 오류: {predict.error_message}"],
            )

        # -- 3. Action Adapter ---------------------------------------------------------
        adapted: AdaptedAction = adapt_vla_action(predict.action)
        adapter_report["mapped_action"] = adapted.command_deg if adapted.valid else None
        adapter_report["warnings"] = list(adapted.warnings)
        if not adapted.valid:
            return self._finalize(
                session_id=session_id, task=self._task, observation=observation_dict, communication=communication,
                vla=vla_report, adapter=adapter_report, safety=safety_report, mujoco=mujoco_report,
                validation=validation_report, result=RESULT_SHADOW_FAIL,
                reasons=[f"Action Adapter 검증 실패: {adapted.invalid_reason}"],
            )

        # -- 4. Safety Gate --------------------------------------------------------------
        decision: SafetyDecision = self._safety_gate.evaluate(
            adapted_action=adapted,
            current_state_deg=built.state,
            observation_valid=built.schema_valid,
            observation_reasons=tuple(built.invalid_reasons),
            state_stale=state_stale,
            state_stale_reason=state_stale_reason,
        )
        safety_report.update(decision.to_dict())

        try:
            self._vla_client.action_ack(
                session_id=session_id, sequence=sequence, executed=(decision.decision != "REJECT"), backend=BACKEND_LABEL
            )
        except Exception as exc:  # noqa: BLE001 - ack 실패는 결과에 영향을 주지 않는다 (best-effort 로깅)
            communication["action_ack_error"] = str(exc)

        if decision.decision == "REJECT":
            return self._finalize(
                session_id=session_id, task=self._task, observation=observation_dict, communication=communication,
                vla=vla_report, adapter=adapter_report, safety=safety_report, mujoco=mujoco_report,
                validation=validation_report, result=RESULT_SHADOW_FAIL,
                reasons=[f"Safety REJECT: {r}" for r in decision.reasons],
            )

        # -- 5. Realistic MuJoCo 실행 (safe_action만, 실물에는 절대 쓰지 않음) --------------
        assert decision.safe_action is not None
        try:
            self._mujoco_backend.sync_initial_state(built.state)
            step_result: MuJoCoStepResult = self._mujoco_backend.execute_single_step(decision.safe_action, now=time.monotonic())
        except MuJoCoBackendError as exc:
            return self._finalize(
                session_id=session_id, task=self._task, observation=observation_dict, communication=communication,
                vla=vla_report, adapter=adapter_report, safety=safety_report, mujoco=mujoco_report,
                validation=validation_report, result=RESULT_SHADOW_FAIL, reasons=[f"MuJoCo 실행 실패: {exc}"],
            )

        mujoco_report = {
            "initial_state": step_result.initial_state_deg,
            "final_state": step_result.final_state_deg,
            "realistic_layer_diagnostics": step_result.realistic_diagnostics,
            "processed_command": step_result.processed_command_deg,
            "mechanical_violations": list(step_result.mechanical_violations),
            "tracking_error_deg": step_result.tracking_error_deg,
            "max_abs_qvel": step_result.max_abs_qvel,
        }

        # -- 6. Shadow Validation ----------------------------------------------------
        validation: ShadowValidationResult = validate_mujoco_step(step_result)
        validation_report = validation.to_dict()

        if not validation.passed:
            return self._finalize(
                session_id=session_id, task=self._task, observation=observation_dict, communication=communication,
                vla=vla_report, adapter=adapter_report, safety=safety_report, mujoco=mujoco_report,
                validation=validation_report, result=RESULT_SHADOW_FAIL,
                reasons=[f"MuJoCo validation 실패: {f}" for f in validation.failures],
            )

        result = RESULT_SHADOW_PASS_WITH_CLAMP if decision.decision == "WOULD_CLAMP" else RESULT_SHADOW_PASS
        result_reasons = list(decision.reasons) if decision.decision == "WOULD_CLAMP" else []
        return self._finalize(
            session_id=session_id, task=self._task, observation=observation_dict, communication=communication,
            vla=vla_report, adapter=adapter_report, safety=safety_report, mujoco=mujoco_report,
            validation=validation_report, result=result, reasons=result_reasons,
        )

    def _finalize(
        self,
        *,
        session_id: str,
        task: str,
        observation: dict,
        communication: dict,
        vla: dict,
        adapter: dict,
        safety: dict,
        mujoco: dict,
        validation: dict,
        result: str,
        reasons: list[str],
    ) -> ShadowRunResult:
        # REAL_SO101_WRITE 불변식: 이 러너는 어떤 write 메서드도 참조하지 않으므로 항상 0이다.
        write_count = REAL_FOLLOWER_WRITE_COUNT
        report = build_report(
            task=task, backend=BACKEND_LABEL, observation=observation, communication=communication, vla=vla,
            adapter=adapter, safety=safety, mujoco=mujoco, validation=validation,
            real_follower_write_count=write_count, result=result, result_reasons=reasons,
        )
        report["session_id"] = session_id
        report_path = write_report(report, reports_dir=self._reports_dir, session_id=session_id)
        return ShadowRunResult(
            session_id=session_id, result=result, reasons=tuple(reasons), report=report, report_path=report_path,
            real_follower_write_count=write_count,
        )
