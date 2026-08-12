"""Phase C-5 후속: tick-level diagnostic JSONL logger (분석/계측 전용, additive-only).

# 왜 필요한가

C-5 첫/두 번째 real session(``reports/real_pick_drop_realtime_v1/session_1786453529.json``,
``session_1786461563.json``)은 ``SessionReport``라는 **집계 통계**만 남겼다 - 어떤 tick에서
어떤 joint가 얼마의 delta로 Intent/Final Safety를 막았는지, quarantine이 각 sequence를
정확히 언제(어떤 tick, 어떤 사유로) 처음 먹었는지는 재구성이 불가능했다(원본 조사 근거:
세션 스크래치패드 분석). 이 모듈은 **그 다음 real run 한 번만으로** 그 질문에 답할 수
있도록 tick마다 한 줄씩 JSONL을 남긴다.

# 절대 규칙 (요구사항 그대로)

    - Safety threshold(``configs/safety_gate.yaml``)를 읽기만 한다 - 바꾸지 않는다.
    - Motion Guard 파라미터(``motion_guard.py``)를 바꾸지 않는다 - 이 모듈은 그 파일을
      import조차 하지 않는다.
    - Quarantine semantics(``realtime_control_loop.py``)를 바꾸지 않는다 - 그 파일도
      **한 글자도 수정하지 않는다.** quarantine 진행 상황은 ``ControlTickRecord``가 이미
      공개적으로 노출하는 ``intent_decision``/``contributing_sequences``/``tick_index``만
      가지고 **읽기 전용으로 재구성**한다(아래 "quarantine 재구성" 절 - 실제 loop 내부의
      trigger 규칙과 정확히 같은 조건을 그대로 다시 평가할 뿐, 그 결정에 관여하지 않는다).
    - 기존 realtime 동작(control loop, write 판정, quarantine)을 절대 바꾸지 않는다 -
      이 모듈의 어떤 함수도 ``RealTimeFollowerControlLoop``/``RealTimeControlTargetGenerator``/
      ``PolicyIntentValidator``/``SafetyGate``의 코드를 한 줄도 수정하지 않는다. 이미
      shipped된 ``SafetyGate.evaluate()``(순수 함수, side effect 없음)를 **다시 호출**해서
      per-joint 진단 정보(원래 호출부가 버리는 ``SafetyDecision.per_joint``)를 뽑아낼
      뿐이다 - 같은 입력이면 항상 같은 출력이므로 실제 판정에 어떤 영향도 주지 않는다.

# 두 단계 캡처 (thread 분리를 그대로 존중)

``realtime_control_loop.py`` 모듈 docstring이 이미 "60Hz control thread와 ~3Hz inference
thread를 절대 blocking으로 섞지 않는다"는 원칙을 세워뒀다. 이 로거도 같은 원칙을 지킨다:

    1. **generator-stage capture (control loop thread, 매 tick)**: ``DiagnosticCapturingGeneratorProxy.tick()``이
       ``RealTimeControlTargetGenerator.tick()``을 감싸 호출한 직후, raw target/guarded
       target/Intent·Final Safety per-joint 진단을 계산해 **메모리 버퍼에만 append**한다
       (``TickDiagnosticRecorder._pending``, lock으로 보호). 디스크 I/O가 전혀 없으므로
       60Hz tick 타이밍에 영향을 주지 않는다 - 기존 ``RecordingGeneratorProxy``가
       ``self.results.append(result)`` 하나만 하던 것과 같은 무게의 작업(list append +
       순수 dict 계산, ms 미만)이다.
    2. **flush (main/monitor thread, ~10Hz 기존 polling 주기에 편승)**: 이미
       ``RealtimeSessionOrchestrator._run_monitor_loop()``가 100ms마다
       ``self._loop.tick_history()``를 폴링하고 있다(``CONTROL_LOOP_POLL_INTERVAL_S``,
       기존 코드, 안 바꿈) - 그 자리에 ``TickDiagnosticRecorder.drain_and_write(records)``
       호출 하나만 추가한다. 여기서 파일 write+flush(디스크 I/O)가 일어나지만, 이 thread는
       애초에 timing-critical하지 않은 monitor loop다(60Hz motor thread가 아님).

# tick_index 정합성 (생략 없이 매 tick 기록)

``ControlTickRecord``는 fault tick(state read 실패 등)에도 매 ``_do_tick()``마다
빠짐없이 만들어진다(``realtime_control_loop.py`` 확인). 이 로거도 **모든 tick_index를
빠짐없이** 한 줄씩 남긴다 - generator가 아예 호출되지 못한 tick(state read 실패)은
generator-stage 필드가 전부 null인 채로, 그래도 한 줄은 남는다. drain은 ``tick_index``
오름차순으로만 진행하고 이미 쓴 tick_index는 다시 쓰지 않는다(멱등, 여러 번 drain해도
중복 라인 없음).

# quarantine 재구성 (읽기 전용, 새 규칙 발명 안 함)

``realtime_control_loop.py``의 실제 규칙(모듈 docstring "Intent quarantine" 절, 코드
그대로 인용)::

    if result.intent_decision is not None and result.intent_decision != "ACCEPT":
        self._quarantined_sequences.update(result.contributing_sequences)

이 로거는 ``ControlTickRecord.intent_decision``/``.contributing_sequences``만 가지고
**정확히 같은 조건**을 tick_index 오름차순으로 재적용해 "이 tick에서 새로 quarantine된
sequence"와 "이 tick 시작 시점의 누적 quarantine 집합"을 계산한다 - 결정 로직을 다시
"구현"하는 게 아니라, 이미 일어난 결정의 **결과를 그대로 다시 읽는 것**이다(같은 입력 ->
같은 predicate). 세션 종료 시 ``cross_check_final_quarantine()``으로 이 재구성 결과를
``RealTimeFollowerControlLoop.quarantined_sequences``(진짜 authoritative 값, 이미 공개
property)와 대조해 불일치가 있으면 경고를 남긴다 - 로직 drift를 조용히 넘기지 않기 위한
안전장치.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.action_adapter import adapt_vla_action
from runtime.laptop.realtime_control_target import ControlTargetResult
from runtime.laptop.safety_gate import SafetyGate

# realtime_control_loop.ControlTickRecord를 타입 힌트로만 쓴다 - import는 하되 그 모듈의
# 어떤 것도 수정/재구현하지 않는다 (TYPE_CHECKING 불필요할 만큼 가벼운 순수 dataclass).
from runtime.laptop.realtime_control_loop import ControlTickRecord


def _evaluate_stage_diagnostics(
    *, safety_gate: SafetyGate, target_deg: dict[str, float] | None, current_state_deg: dict[str, float] | None,
) -> dict | None:
    """``SafetyGate.evaluate()``를 진단 목적으로 다시 호출해 per-joint 상세를 뽑는다.

    순수 함수 재호출이다 - ``target_deg``/``current_state_deg``가 실제 파이프라인에
    넘어갔던 것과 동일한 값이면 (Intent Validation이든 Final SafetyGate든) 이 호출은
    실제로 이미 내려진 decision과 **항상 같은 결과**를 낸다(``SafetyGate.evaluate()``는
    stateless). 그 decision 자체는 로깅에만 쓰고 버린다 - 실제 판정은 이미
    ``ControlTargetResult``/``ControlTickRecord``에 확정돼 있다.
    """
    if target_deg is None or current_state_deg is None:
        return None
    try:
        adapted = adapt_vla_action(target_deg)
        decision = safety_gate.evaluate(adapted_action=adapted, current_state_deg=current_state_deg, observation_valid=True)
    except Exception as exc:  # noqa: BLE001 - 진단 계산 실패가 절대 control loop에 새어나가면 안 됨
        return {"error": f"{type(exc).__name__}: {exc}"}

    cfg = safety_gate.config
    per_joint: dict[str, dict] = {}
    for joint in JOINT_ORDER:
        rep = decision.per_joint.get(joint)
        current_value = current_state_deg.get(joint)
        raw_value = rep.raw_value if rep is not None else None
        delta = (raw_value - current_value) if (raw_value is not None and current_value is not None) else None
        lo_hi = cfg.joint_range_deg.get(joint)
        per_joint[joint] = {
            "raw_value": raw_value,
            "current_value": current_value,
            "delta_vs_current": delta,
            "safe_value": rep.safe_value if rep is not None else None,
            "clamped": rep.clamped if rep is not None else None,
            "rejected": rep.rejected if rep is not None else None,
            "reasons": list(rep.reasons) if rep is not None else [],
            "excessive_step_threshold_deg": cfg.max_step_deg.get(joint),
            "mechanical_range_deg": list(lo_hi) if lo_hi is not None else None,
        }
    return {
        "decision": decision.decision,
        "reasons": list(decision.reasons),
        "per_joint": per_joint,
    }


@dataclass
class _PendingGeneratorRecord:
    """control-loop thread에서 append된, 아직 ``ControlTickRecord``와 merge 전인 한 tick 분량."""

    now_monotonic: float
    current_follower_state_deg: dict[str, float] | None
    contributing_sequences: tuple[int, ...]
    raw_ensemble_target: dict[str, float] | None
    intent_decision: str | None
    intent_reasons: tuple[str, ...]
    intent_diagnostics: dict | None  # _evaluate_stage_diagnostics(raw_ensemble_target)
    guarded_target: dict[str, float] | None
    final_target: dict[str, float] | None
    clamp_reasons: tuple[str, ...]
    safety_decision: str | None
    safety_reasons: tuple[str, ...]
    safety_diagnostics: dict | None  # _evaluate_stage_diagnostics(guarded_target)
    stop_reason: str | None
    consumed: bool = False


class TickDiagnosticRecorder:
    """control-loop thread(생산자, in-memory append만)와 main/monitor thread(소비자,
    파일 write)를 분리한 tick-level JSONL writer. ``close()``까지 매 tick 하나씩 라인을
    남긴다 - 실패한 tick도 예외를 삼키고 최소 정보만으로 한 줄을 남긴다(누락 없음)."""

    def __init__(self, path: str | Path, *, safety_gate: SafetyGate) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="utf-8")
        self._safety_gate = safety_gate

        self._lock = threading.Lock()
        # now_monotonic(float) -> _PendingGeneratorRecord. generator.tick()은 tick당
        # 최대 한 번만 호출되고(``realtime_control_loop._do_tick`` 확인), 그 tick의
        # ``now_monotonic``은 곧 ``ControlTickRecord.actual_start_time_monotonic``과 같은
        # float 변수에서 나온다(같은 ``actual_start`` 값) - 그래서 이 키로 정확히 1:1 매칭된다.
        self._pending: dict[float, _PendingGeneratorRecord] = {}
        self._next_tick_index = 0
        self._quarantine_accumulator: set[int] = set()
        self._lines_written = 0
        self._closed = False

    # -- 1. control-loop thread에서 매 tick 호출 (in-memory only, I/O 없음) -------------

    def capture_generator_tick(
        self, *, now_monotonic: float, current_follower_state_deg: dict[str, float] | None, result: ControlTargetResult,
    ) -> None:
        try:
            intent_diag = _evaluate_stage_diagnostics(
                safety_gate=self._safety_gate, target_deg=result.raw_ensemble_target,
                current_state_deg=current_follower_state_deg,
            )
            safety_diag = _evaluate_stage_diagnostics(
                safety_gate=self._safety_gate, target_deg=result.guarded_target,
                current_state_deg=current_follower_state_deg,
            )
            pending = _PendingGeneratorRecord(
                now_monotonic=now_monotonic,
                current_follower_state_deg=(dict(current_follower_state_deg) if current_follower_state_deg else None),
                contributing_sequences=tuple(result.contributing_sequences),
                raw_ensemble_target=(dict(result.raw_ensemble_target) if result.raw_ensemble_target else None),
                intent_decision=result.intent_decision,
                intent_reasons=tuple(result.intent_reasons),
                intent_diagnostics=intent_diag,
                guarded_target=(dict(result.guarded_target) if result.guarded_target else None),
                final_target=(dict(result.final_target) if result.final_target else None),
                clamp_reasons=tuple(result.clamp_reasons),
                safety_decision=result.safety_decision,
                safety_reasons=tuple(result.safety_reasons),
                safety_diagnostics=safety_diag,
                stop_reason=result.stop_reason,
            )
        except Exception as exc:  # noqa: BLE001 - 진단 캡처는 control loop을 절대 방해하면 안 됨
            pending = _PendingGeneratorRecord(
                now_monotonic=now_monotonic,
                current_follower_state_deg=(dict(current_follower_state_deg) if current_follower_state_deg else None),
                contributing_sequences=(), raw_ensemble_target=None, intent_decision=None, intent_reasons=(),
                intent_diagnostics={"capture_error": f"{type(exc).__name__}: {exc}"}, guarded_target=None,
                final_target=None, clamp_reasons=(), safety_decision=None, safety_reasons=(), safety_diagnostics=None, stop_reason=None,
            )
        with self._lock:
            self._pending[now_monotonic] = pending

    # -- 2. main/monitor thread에서 주기적으로 호출 (파일 I/O 여기서만) ------------------

    def drain_and_write(self, tick_records: Sequence[ControlTickRecord]) -> int:
        """``tick_index`` 오름차순으로 아직 안 쓴 record만 골라 한 줄씩 flush한다.
        멱등(idempotent) - 여러 번 불러도 중복 라인이 생기지 않는다."""
        if self._closed:
            return 0
        written = 0
        for record in tick_records:
            if record.tick_index < self._next_tick_index:
                continue  # 이미 썼음
            if record.tick_index > self._next_tick_index:
                # gap이 있으면(이론상 발생하지 않음 - tick_index는 gapless) 그 사이는
                # 다음 drain에서 채워질 수 있으니 여기서는 건너뛴다 - 순서를 어기지 않는다.
                continue
            self._write_one(record)
            self._next_tick_index = record.tick_index + 1
            written += 1
        return written

    @staticmethod
    def _per_joint_of(diagnostics: dict | None) -> dict | None:
        """``_evaluate_stage_diagnostics()``의 반환값(``{"decision":..., "reasons":...,
        "per_joint": {...}}``)에서 per-joint 매핑만 뽑는다. 진단 계산 자체가 실패했던
        경우(``{"capture_error": ...}``)는 그 marker를 그대로 보여준다(누락 없이 사유를
        남기기 위함)."""
        if diagnostics is None:
            return None
        if "capture_error" in diagnostics:
            return diagnostics
        return diagnostics.get("per_joint")

    def _write_one(self, record: ControlTickRecord) -> None:
        # -- quarantine 재구성 (읽기 전용, 모듈 docstring "quarantine 재구성" 참고) -----
        quarantined_before_tick = sorted(self._quarantine_accumulator)
        newly_quarantined: list[int] = []
        if record.intent_decision == "REJECT":
            new_seqs = [s for s in record.contributing_sequences if s not in self._quarantine_accumulator]
            if new_seqs:
                newly_quarantined = sorted(new_seqs)
                self._quarantine_accumulator.update(new_seqs)

        with self._lock:
            pending = self._pending.pop(record.actual_start_time_monotonic, None)

        event: dict = {
            "tick_index": record.tick_index,
            "monotonic_ts": record.actual_start_time_monotonic,
            "scheduled_time_monotonic": record.scheduled_time_monotonic,
            "dt_s": record.dt_s,
            "control_state": record.state.value,
            "stop_reason": record.stop_reason,
            "target_valid": record.target_valid,
            "errors": list(record.errors),
            "contributing_sequences": list(record.contributing_sequences),
            "quarantined_sequences_excluded_this_tick": list(record.quarantined_sequences_excluded),
            "quarantine": {
                "before_tick": quarantined_before_tick,
                "newly_quarantined_this_tick": newly_quarantined,
                "after_tick": sorted(self._quarantine_accumulator),
            },
            "current_follower_state_deg": None,
            "raw_ensemble_target": None,
            "raw_target": record.raw_target,
            "intent_decision": record.intent_decision,
            "intent_reasons": [],
            "intent_per_joint": None,
            "guarded_target": None,
            "final_target": record.final_target,
            "clamp_reasons": list(record.clamp_reasons),
            "final_safety_decision": record.safety_decision,
            "final_safety_reasons": [],
            "final_safety_per_joint": None,
            "motion_guard_delta_deg": None,  # raw_ensemble_target - guarded_target, joint별
            "write": {
                "attempted": record.write_attempted,
                "executed": record.write_executed,
                "path": record.write_path,
                "written_target_deg": None,
                "no_write_reason": (None if record.write_executed else (record.stop_reason or record.state.value)),
            },
        }

        if pending is not None:
            event["current_follower_state_deg"] = pending.current_follower_state_deg
            event["raw_ensemble_target"] = pending.raw_ensemble_target
            event["intent_reasons"] = list(pending.intent_reasons)
            event["intent_per_joint"] = self._per_joint_of(pending.intent_diagnostics)
            event["guarded_target"] = pending.guarded_target
            event["final_target"] = pending.final_target
            event["clamp_reasons"] = list(pending.clamp_reasons)
            event["final_safety_reasons"] = list(pending.safety_reasons)
            event["final_safety_per_joint"] = self._per_joint_of(pending.safety_diagnostics)

            if pending.raw_ensemble_target is not None and pending.final_target is not None:
                event["motion_guard_delta_deg"] = {
                    j: pending.raw_ensemble_target[j] - pending.guarded_target[j]
                    for j in JOINT_ORDER
                    if j in pending.raw_ensemble_target and j in pending.guarded_target
                }

            # write invariant(realtime_control_loop.py 섹션 16): primary path의 write는
            # 항상 guarded_target 그대로다 - 그 값을 그대로 다시 보여준다(재계산 아님).
            if record.write_executed and record.write_path == "primary" and pending.guarded_target is not None:
                event["write"]["written_target_deg"] = dict(pending.final_target)
            elif record.write_executed and record.write_path != "primary":
                # hold 경로(HOLD_LAST_COMMANDED/HOLD_MEASURED) - 이 값은
                # realtime_control_loop.py 내부(_compute_hold_candidate)에서만 계산되고
                # 이 로거는 그 코드를 건드리지 않으므로 정확한 written 값을 여기서 다시
                # 구할 수 없다. C-5 실제 세션은 hold_policy=NO_WRITE로 고정되어 있어
                # (scripts/run_real_pick_drop_realtime.py) 이 분기는 정상 운영에서
                # 발생하지 않는다 - 발생 시 path만 기록하고 값은 비워 정직하게 남긴다.
                event["write"]["written_target_deg"] = None
                event["write"]["note"] = f"hold path({record.write_path}) - written value not observable without touching realtime_control_loop.py"

        self._file.write(json.dumps(event, default=str) + "\n")
        self._file.flush()
        self._lines_written += 1

    # -- 3. 세션 종료 시 -----------------------------------------------------------------

    def cross_check_final_quarantine(self, authoritative_quarantined_sequences: frozenset[int]) -> list[str]:
        """재구성한 quarantine 누적 결과를 진짜 값(``RealTimeFollowerControlLoop.quarantined_sequences``,
        공개 property)과 대조한다 - 로직 drift가 있으면 조용히 넘어가지 않고 경고 문자열을
        반환한다(``realtime_control_loop.py``는 여전히 안 건드림 - 비교만 함)."""
        mine = set(self._quarantine_accumulator)
        theirs = set(authoritative_quarantined_sequences)
        if mine == theirs:
            return []
        return [
            f"[진단 경고] JSONL에서 재구성한 quarantine 집합({sorted(mine)})이 실제 loop의 "
            f"quarantined_sequences({sorted(theirs)})와 다릅니다 - drain이 아직 끝나지 않았거나 "
            f"(close() 전 마지막 drain 누락) realtime_control_loop.py의 quarantine trigger 조건이 "
            f"이 모듈의 재구성 로직과 달라졌을 수 있습니다."
        ]

    @property
    def lines_written(self) -> int:
        return self._lines_written

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True


class DiagnosticCapturingGeneratorProxy:
    """``scripts/run_real_pick_drop_realtime.py``의 기존 ``RecordingGeneratorProxy``와
    같은 자리(``generator=`` 인자)에 꽂는 drop-in 대체품 - ``.results`` 리스트를 똑같이
    채워서(``_gen_proxy.results`` 사용하는 기존 sanity-window/report 코드가 전혀 안
    바뀌어도 되게) 하고, 추가로 ``recorder``가 있으면(``None``이면 기존과 100% 동일하게
    동작 - 진단 기능 전면 opt-in) 매 tick 진단 캡처를 덧붙인다."""

    def __init__(self, inner, *, recorder: TickDiagnosticRecorder | None = None) -> None:
        self._inner = inner
        self._recorder = recorder
        self.results: list[ControlTargetResult] = []

    def tick(self, **kwargs):
        result = self._inner.tick(**kwargs)
        self.results.append(result)
        if self._recorder is not None:
            self._recorder.capture_generator_tick(
                now_monotonic=kwargs.get("now_monotonic"),
                current_follower_state_deg=kwargs.get("current_follower_state_deg"),
                result=result,
            )
        return result
