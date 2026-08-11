"""Real-follower staged safety test 오케스트레이터.

``runtime/laptop/shadow_mode_runner.py::ShadowModeRunner.run_once()``와 정확히 같은 4단계
파이프라인(관측 -> VLA 추론 -> Action Adapter -> Safety Gate)을 그대로 쓰되, MuJoCo 대신
실제 follower에 쓰고(``hardware/safety/staged_follower_writer.StagedFollowerArmedWriter``),
그 파이프라인을 스테이지의 step 수만큼 반복한다.

## 핵심 불변식 (요구사항, 모두 이 파일 하나에서 강제한다)

1. ``decision.decision == "ACCEPT"``인 action만 실제로 write된다. ``WOULD_CLAMP``/``REJECT``는
   (설령 Safety Gate가 계산해 둔 ``safe_action``이 있어도) 절대 ``writer.write_action_once``에
   전달하지 않는다.
2. 첫 번째 non-ACCEPT가 나오는 즉시 stage 전체를 ``ABNORMAL_STOPPED``로 멈춘다 - 남은 step을
   건너뛰고 계속하지 않는다.
3. 매 step마다 write 전 상태(``before_state_deg``, 이번 step 관측에서 읽은 실제 값)와 write
   후 상태(``after_state_deg``, write 후 다시 읽은 실제 값 - 가정하지 않고 항상 재측정)를
   기록하고, 그 차이를 ``delta_deg``에 남긴다. write를 안 했거나 못 했으면 ``after_state_deg``/
   ``delta_deg``는 ``None``이다(측정 안 한 값을 지어내지 않는다).
4. Safety Gate/``configs/safety_gate.yaml`` 임계값은 이 파일에서 전혀 건드리지 않는다 -
   ``SafetyGate.evaluate()``를 그대로 호출만 한다.
5. 이 파일 자체는 stage 개수 제한을 강제하지 않는다 - "몇 step까지 허용할지"(``max_steps``)는
   호출부(``scripts/run_real_follower_staged_safety_test.py``)가 stage별 하드코딩된 상수로
   결정해서 넘긴다(요구사항: 별도 max step 제한, CLI로 그 상수 자체를 못 늘리게).
6. **매 step, predict 직전에 반드시 ``vla_client.session_reset()``을 호출한다** (2026-08
   stale-chunk 조사 근거 - 아래 "receding-horizon 근거" 절 참고). reset이 실패(예외 또는
   ``ok=False``)하면 **그 step의 predict/write를 절대 시도하지 않고 즉시 stage를 중단한다**
   (fail-closed) - "reset이 실패했으니 옛날 chunk라도 쓴다"는 폴백은 없다.

## receding-horizon 근거 (2026-08 SmolVLA action-queue 조사)

별도 조사(이 세션 이전)에서 코드로 확인한 사실: Desktop ``SmolVLAPolicyRunner``(및 그
내부 ``policy`` 객체)는 FastAPI 서버 프로세스가 살아있는 동안 단 한 번만 생성되고,
``policy.select_action()``의 action queue(``deque(maxlen=n_action_steps)``, Candidate B는
``n_action_steps=chunk_size=50``)는 ``/session/reset``이 호출될 때만 비워진다
(``~/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py``의 ``reset()``/
``select_action()``/``_check_get_actions_condition()``). 이 파일이 이전에는 매 step마다
fresh camera/state를 ``/predict``로 보내면서도 ``/session/reset``을 전혀 호출하지 않았기
때문에, queue가 비어있지 않은 한 그 fresh observation은 실제 action 계산에 쓰이지 않고
과거(최대 50 step 전) observation에서 나온 stale ``chunk[k]``가 그대로 반환됐다 - 즉
Stage 2/3(step 수 ≤15 < chunk_size=50)는 사실상 open-loop chunk 실행이었다.

이 모듈은 그 문제를 다음 receding-horizon 시퀀스로 고친다(매 step 반복)::

    obs_t (fresh camera + fresh state)
      -> vla_client.session_reset(...)      # policy queue를 강제로 비움 (fail-closed)
      -> vla_client.predict(...)            # 큐가 비었으므로 반드시 obs_t 기반 새 chunk 추론
      -> chunk[0]만 사용 (predict()가 이미 chunk[0]만 반환함 - select_action()의 계약)
      -> Action Adapter -> Safety Gate -> (ACCEPT면) write
      -> obs_t+1 에서 반복

``shadow_mode_runner.py``가 이미 매 ``run_once()``마다 predict 직전 ``session_reset()``을
호출하는 동일한 패턴을 쓰고 있었다 - 이 모듈은 그 계약을 그대로 재사용한다(새 프로토콜을
만들지 않음). ``VLAHttpClient``/``InProcessSmolVLAClient`` 둘 다 이미 ``session_reset()``을
구현하고 있으므로 client 쪽 코드는 변경하지 않았다.

## 이번 구현의 의도적 범위 제한 (요구사항 명시)

이 모듈은 **staged safety validation(Stage 1/2/3)용 synchronous single-action
receding-horizon** 구현이다 - 매 step reset+predict 왕복 지연을 감수하고 "이 step의
write가 반드시 이 step의 관측에 조건화됐다"는 것을 fail-closed로 보장하는 데만 목적이
있다. Full Pick & Drop 실물 runtime의 최종 구조(비동기 추론, short-horizon 실행,
temporal ensembling, 50~60Hz 모터 보간, velocity/acceleration/jerk limiting)는 이 모듈의
범위가 아니며 별도 후속 작업으로 남겨둔다 - 여기서 과도하게 구현하지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.action_adapter import AdaptedAction, adapt_vla_action
from runtime.laptop.camera_source import CameraFrame
from runtime.laptop.follower_state_source import FollowerStateSnapshot
from runtime.laptop.observation_builder import BuiltObservation, build_observation
from runtime.laptop.safety_gate import SafetyDecision, SafetyGate
from runtime.laptop.vla_client import PredictResult

PASS = "PASS"
ABNORMAL_STOPPED = "ABNORMAL_STOPPED"


class CameraSourceProtocol(Protocol):
    def capture_all(self) -> dict[str, CameraFrame]: ...


class StateSourceProtocol(Protocol):
    def read(self) -> FollowerStateSnapshot: ...


class VLAClientProtocol(Protocol):
    def session_reset(self, *, session_id: str, task: str) -> bool: ...
    def predict(self, *, session_id: str, task: str, sequence: int, state: dict, images: dict) -> PredictResult: ...


class WriterProtocol(Protocol):
    def read_state_deg(self) -> dict[str, float]: ...
    def write_action_once(self, safe_action_deg: dict[str, float]): ...


@dataclass(frozen=True)
class StagedStepResult:
    step: int  # 이 step의 sequence index (session_id="staged-{step}", predict(sequence=step)와 동일)
    before_state_deg: dict[str, float]
    raw_action_deg: dict[str, float] | None
    adapter_valid: bool
    adapter_invalid_reason: str | None
    safety_decision: str | None  # "ACCEPT" | "WOULD_CLAMP" | "REJECT" | None(관측/추론 단계에서 이미 중단)
    safety_reasons: tuple[str, ...]
    would_be_safe_action_deg: dict[str, float] | None  # WOULD_CLAMP/REJECT라도 진단용으로 기록 (절대 write 안 함)
    written: bool
    sent_action_deg: dict[str, float] | None
    after_state_deg: dict[str, float] | None
    delta_deg: dict[str, float] | None
    write_error: str | None
    step_error: str | None  # 관측/reset/추론/adapter 단계 실패 시 사유
    # -- receding-horizon 진단 필드 (2026-08 stale-chunk 수정) --------------------------
    session_reset_ok: bool | None  # None = 이 step에서 reset을 시도조차 못 함(관측 실패 등)
    session_reset_error: str | None
    session_reset_latency_ms: float | None
    predict_inference_latency_ms: float | None  # Desktop 서버가 보고한 순수 추론 시간
    predict_request_latency_ms: float | None  # predict() 왕복 전체(네트워크 포함) 시간
    fresh_inference: bool | None  # True: reset이 이 step에서 predict 직전 성공 -> select_action()이
    # 큐가 비어있음을 보장받은 상태에서 이번 obs_t로 새 chunk를 추론했음(계약상 100% 보장, 관측/재추론
    # 결과가 아니라 reset->predict 순서 자체에서 유도됨). None = reset을 시도조차 못 함.

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "before_state_deg": dict(self.before_state_deg) if self.before_state_deg else None,
            "raw_action_deg": dict(self.raw_action_deg) if self.raw_action_deg else None,
            "adapter_valid": self.adapter_valid,
            "adapter_invalid_reason": self.adapter_invalid_reason,
            "safety_decision": self.safety_decision,
            "safety_reasons": list(self.safety_reasons),
            "would_be_safe_action_deg": dict(self.would_be_safe_action_deg) if self.would_be_safe_action_deg else None,
            "written": self.written,
            "sent_action_deg": dict(self.sent_action_deg) if self.sent_action_deg else None,
            "after_state_deg": dict(self.after_state_deg) if self.after_state_deg else None,
            "delta_deg": dict(self.delta_deg) if self.delta_deg else None,
            "write_error": self.write_error,
            "step_error": self.step_error,
            "session_reset_ok": self.session_reset_ok,
            "session_reset_error": self.session_reset_error,
            "session_reset_latency_ms": self.session_reset_latency_ms,
            "predict_inference_latency_ms": self.predict_inference_latency_ms,
            "predict_request_latency_ms": self.predict_request_latency_ms,
            "fresh_inference": self.fresh_inference,
        }


@dataclass(frozen=True)
class StagedStageResult:
    stage: int
    max_steps: int
    verdict: str  # PASS | ABNORMAL_STOPPED
    stop_reason: str | None
    steps: tuple[StagedStepResult, ...]
    real_follower_write_count: int
    checkpoint: str
    task: str

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "max_steps": self.max_steps,
            "verdict": self.verdict,
            "stop_reason": self.stop_reason,
            "steps": [s.to_dict() for s in self.steps],
            "real_follower_write_count": self.real_follower_write_count,
            "checkpoint": self.checkpoint,
            "task": self.task,
        }


class StagedRealRolloutRunner:
    def __init__(
        self,
        *,
        camera_source: CameraSourceProtocol,
        state_source: StateSourceProtocol,
        vla_client: VLAClientProtocol,
        safety_gate: SafetyGate,
        writer: WriterProtocol,
        task: str,
        checkpoint_label: str,
        post_write_settle_s: float = 0.3,
        sleep_fn=time.sleep,
    ) -> None:
        self._camera_source = camera_source
        self._state_source = state_source
        self._vla_client = vla_client
        self._safety_gate = safety_gate
        self._writer = writer
        self._task = task
        self._checkpoint_label = checkpoint_label
        self._post_write_settle_s = post_write_settle_s
        self._sleep = sleep_fn

    def run_stage(self, *, stage: int, max_steps: int) -> StagedStageResult:
        steps: list[StagedStepResult] = []
        verdict = PASS
        stop_reason: str | None = None

        for i in range(max_steps):
            step, halt, reason = self._run_one_step(i)
            steps.append(step)
            if halt:
                verdict = ABNORMAL_STOPPED
                stop_reason = reason
                break

        return StagedStageResult(
            stage=stage, max_steps=max_steps, verdict=verdict, stop_reason=stop_reason,
            steps=tuple(steps), real_follower_write_count=sum(1 for s in steps if s.written),
            checkpoint=self._checkpoint_label, task=self._task,
        )

    def _run_one_step(self, i: int) -> tuple[StagedStepResult, bool, str | None]:
        def _early_stop(
            reason: str,
            *,
            before: dict | None = None,
            raw: dict | None = None,
            reset_ok: bool | None = None,
            reset_error: str | None = None,
            reset_latency_ms: float | None = None,
        ) -> tuple[StagedStepResult, bool, str]:
            return (
                StagedStepResult(
                    step=i, before_state_deg=before or {}, raw_action_deg=raw, adapter_valid=False,
                    adapter_invalid_reason=None, safety_decision=None, safety_reasons=(),
                    would_be_safe_action_deg=None, written=False, sent_action_deg=None,
                    after_state_deg=None, delta_deg=None, write_error=None, step_error=reason,
                    session_reset_ok=reset_ok, session_reset_error=reset_error,
                    session_reset_latency_ms=reset_latency_ms,
                    predict_inference_latency_ms=None, predict_request_latency_ms=None,
                    fresh_inference=False if reset_ok is False else None,
                ),
                True,
                reason,
            )

        # -- 1. 관측 (read-only) --------------------------------------------------------
        try:
            camera_frames = self._camera_source.capture_all()
            state_snapshot = self._state_source.read()
        except Exception as exc:  # noqa: BLE001
            return _early_stop(f"observation read 실패: {exc}")

        built: BuiltObservation = build_observation(state_snapshot=state_snapshot, camera_frames=camera_frames)
        if not built.schema_valid:
            return _early_stop(f"관측 스키마가 유효하지 않습니다: {built.invalid_reasons}")

        before_state = dict(built.state)
        session_id = f"staged-{i}"

        # -- 2. VLA session reset (receding-horizon 근거: 모듈 docstring 참고) ----------------
        # 이 step의 predict가 반드시 obs_t(방금 읽은 before_state/camera_frames) 기반 fresh
        # chunk[0]이 되도록, predict 직전에 policy action queue를 강제로 비운다.
        # fail-closed: reset이 실패하면(예외든 ok=False든) 절대 predict/write로 진행하지
        # 않는다 - "reset 실패했으니 옛날 chunk라도 쓴다"는 폴백은 만들지 않는다.
        reset_t0 = time.monotonic()
        try:
            reset_ok = self._vla_client.session_reset(session_id=session_id, task=self._task)
        except Exception as exc:  # noqa: BLE001
            reset_latency_ms = (time.monotonic() - reset_t0) * 1000.0
            return _early_stop(
                f"VLA session_reset 실패 (fail-closed - predict/write 진행 안 함): {exc}",
                before=before_state, reset_ok=False, reset_error=str(exc), reset_latency_ms=reset_latency_ms,
            )
        reset_latency_ms = (time.monotonic() - reset_t0) * 1000.0
        if not reset_ok:
            return _early_stop(
                "VLA session_reset이 실패를 반환했습니다(ok=False) - fail-closed, predict/write 진행 안 함",
                before=before_state, reset_ok=False, reset_error="session_reset() returned ok=False",
                reset_latency_ms=reset_latency_ms,
            )

        # -- 3. VLA 추론 (reset 직후이므로 policy queue가 비어있음이 보장됨 -> 이번 obs_t로 -----
        #    새로 추론한 chunk의 chunk[0]만 반환됨. select_action()이 chunk[0] 하나만 꺼내
        #    돌려주는 계약이므로 이 predict() 반환값 자체가 이미 chunk[0]이다.
        predict: PredictResult = self._vla_client.predict(
            session_id=session_id, task=self._task, sequence=i, state=before_state, images=built.images
        )
        if not predict.ok:
            return _early_stop(
                f"VLA {predict.error_kind} 오류: {predict.error_message}", before=before_state,
                reset_ok=True, reset_error=None, reset_latency_ms=reset_latency_ms,
            )
        fresh_inference = True  # 위 reset이 이 predict 직전에 성공했으므로 계약상 100% 보장됨.

        # -- 4. Action Adapter -------------------------------------------------------------
        adapted: AdaptedAction = adapt_vla_action(predict.action)
        if not adapted.valid:
            step = StagedStepResult(
                step=i, before_state_deg=before_state, raw_action_deg=predict.action, adapter_valid=False,
                adapter_invalid_reason=adapted.invalid_reason, safety_decision=None, safety_reasons=(),
                would_be_safe_action_deg=None, written=False, sent_action_deg=None, after_state_deg=None,
                delta_deg=None, write_error=None, step_error=None,
                session_reset_ok=True, session_reset_error=None, session_reset_latency_ms=reset_latency_ms,
                predict_inference_latency_ms=predict.inference_latency_ms,
                predict_request_latency_ms=predict.request_latency_ms, fresh_inference=fresh_inference,
            )
            return step, True, f"Action Adapter 검증 실패: {adapted.invalid_reason}"

        # -- 5. Safety Gate (변경 없음) ------------------------------------------------------
        decision: SafetyDecision = self._safety_gate.evaluate(
            adapted_action=adapted, current_state_deg=before_state, observation_valid=True
        )

        written = False
        sent_action_deg: dict[str, float] | None = None
        after_state_deg: dict[str, float] | None = None
        delta_deg: dict[str, float] | None = None
        write_error: str | None = None

        if decision.decision == "ACCEPT":
            assert decision.safe_action is not None
            result = self._writer.write_action_once(decision.safe_action)
            written = result.executed
            sent_action_deg = result.sent_action_deg
            write_error = result.error
            if written:
                self._sleep(self._post_write_settle_s)
                try:
                    after_state_deg = self._writer.read_state_deg()
                except Exception as exc:  # noqa: BLE001
                    after_state_deg = None
                    write_error = f"{write_error or ''} | readback 실패: {exc}".strip(" |")
                if after_state_deg is not None:
                    delta_deg = {j: after_state_deg[j] - before_state[j] for j in JOINT_ORDER}

        step = StagedStepResult(
            step=i, before_state_deg=before_state, raw_action_deg=predict.action, adapter_valid=True,
            adapter_invalid_reason=None, safety_decision=decision.decision, safety_reasons=decision.reasons,
            would_be_safe_action_deg=decision.safe_action, written=written, sent_action_deg=sent_action_deg,
            after_state_deg=after_state_deg, delta_deg=delta_deg, write_error=write_error, step_error=None,
            session_reset_ok=True, session_reset_error=None, session_reset_latency_ms=reset_latency_ms,
            predict_inference_latency_ms=predict.inference_latency_ms,
            predict_request_latency_ms=predict.request_latency_ms, fresh_inference=fresh_inference,
        )

        if decision.decision != "ACCEPT":
            return step, True, f"step {i}: safety decision={decision.decision} (ACCEPT 아님) - 실제 write 안 함, stage 즉시 중단"
        if not written:
            return step, True, f"step {i}: ACCEPT였지만 write 실패 ({write_error}) - stage 즉시 중단"

        return step, False, None
