"""runtime/laptop/diagnostic_jsonl_logger.py 검증 (C-5 두 번째 real session 후속 - 분석/계측
전용 additive logging). 전부 fake/offline - 하드웨어 접근 없음.

핵심으로 검증하는 것:
    - Intent/Final Safety 단계 per-joint 진단(delta/threshold/reason)이 실제
      ``SafetyGate.evaluate()`` 판정과 정확히 일치한다(재계산이 실제 판정과 다르면 로그 자체가
      거짓말이 되므로 가장 중요한 불변조건).
    - 매 tick 빠짐없이 한 줄(“최소 다음을 기록” 요구사항) - fault tick(generator 자체가 호출
      안 된 tick)도 누락되지 않는다.
    - drain_and_write는 멱등(여러 번 불러도 중복 라인 없음).
    - quarantine 재구성(``newly_quarantined_this_tick``/``after_tick``)이
      ``realtime_control_loop.py``의 실제 trigger 조건(그 파일 자체는 여기서 전혀 안 씀)과
      같은 predicate로 정확히 누적된다.
    - capture 단계에서 예외가 나도 control-loop 쪽(``DiagnosticCapturingGeneratorProxy.tick()``)은
      절대 죽지 않는다.
"""

from __future__ import annotations

import json

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.diagnostic_jsonl_logger import (
    DiagnosticCapturingGeneratorProxy,
    TickDiagnosticRecorder,
    _evaluate_stage_diagnostics,
)
from runtime.laptop.realtime_control_loop import ControlLoopState, ControlTickRecord
from runtime.laptop.realtime_control_target import ControlTargetResult
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig


def _neutral(v: float = 0.0) -> dict[str, float]:
    return {j: v for j in JOINT_ORDER}


def _gate(*, range_deg: tuple[float, float] = (-1000.0, 1000.0), max_step: float = 5.0) -> SafetyGate:
    return SafetyGate(SafetyGateConfig(
        joint_range_deg={j: range_deg for j in JOINT_ORDER},
        max_step_deg={j: max_step for j in JOINT_ORDER},
    ))


def _tick_record(
    *, tick_index: int, state: ControlLoopState, intent_decision: str | None, safety_decision: str | None,
    contributing_sequences: tuple[int, ...] = (), stop_reason: str | None = None, target_valid: bool = False,
    write_attempted: bool = False, write_executed: bool = False, write_path: str | None = None,
    actual_start: float = 0.0, raw_target: dict | None = None, guarded_target: dict | None = None,
    final_target: dict | None = None, clamp_reasons: tuple[str, ...] = (),
    quarantined_sequences_excluded: tuple[int, ...] = (),
    quarantine_before_tick: tuple[int, ...] = (), quarantine_after_tick: tuple[int, ...] = (),
) -> ControlTickRecord:
    return ControlTickRecord(
        tick_index=tick_index, scheduled_time_monotonic=actual_start, actual_start_time_monotonic=actual_start,
        dt_s=1 / 60.0, tick_compute_ms=1.0, state_read_ms=0.1, target_compute_ms=0.2, write_ms=None,
        deadline_overrun_ms=0.0, intent_decision=intent_decision, safety_decision=safety_decision,
        raw_target=raw_target, guarded_target=guarded_target, final_target=final_target, clamp_reasons=clamp_reasons,
        target_valid=target_valid, stop_reason=stop_reason, contributing_sequences=contributing_sequences,
        quarantined_sequences_excluded=quarantined_sequences_excluded, trajectory_age_s=None, write_attempted=write_attempted,
        write_executed=write_executed, write_path=write_path, state=state, errors=(),
        quarantine_before_tick=quarantine_before_tick, quarantine_after_tick=quarantine_after_tick,
    )


def _result(
    *, raw_target: dict | None, intent_decision: str | None, guarded_target: dict | None = None,
    safety_decision: str | None = None, contributing_sequences: tuple[int, ...] = (),
    stop_reason: str | None = None, target_valid: bool = False, final_target: dict | None = None,
    clamp_reasons: tuple[str, ...] = (), target_lookahead: dict | None = None,
    motion_guard_dt_s: float | None = None, motion_guard_diagnostics: dict | None = None,
) -> ControlTargetResult:
    return ControlTargetResult(
        target_time_monotonic=0.0, raw_ensemble_target=raw_target, intent_decision=intent_decision,
        intent_reasons=(), smoothed_target=guarded_target, guarded_target=guarded_target,
        final_target=final_target, clamp_reasons=clamp_reasons,
        safety_decision=safety_decision, safety_reasons=(), contributing_sequences=contributing_sequences,
        target_valid=target_valid, stop_reason=stop_reason, target_lookahead=target_lookahead,
        motion_guard_dt_s=motion_guard_dt_s, motion_guard_diagnostics=motion_guard_diagnostics,
    )


# ---------------------------------------------------------------------------
# _evaluate_stage_diagnostics - 재계산이 실제 SafetyGate 판정과 일치하는가
# ---------------------------------------------------------------------------


def test_stage_diagnostics_matches_accept():
    gate = _gate(max_step=5.0)
    current = _neutral(0.0)
    target = _neutral(1.0)  # delta=1.0, threshold=5.0 -> ACCEPT
    diag = _evaluate_stage_diagnostics(safety_gate=gate, target_deg=target, current_state_deg=current)
    assert diag["decision"] == "ACCEPT"
    for j in JOINT_ORDER:
        assert diag["per_joint"][j]["delta_vs_current"] == pytest.approx(1.0)
        assert diag["per_joint"][j]["excessive_step_threshold_deg"] == pytest.approx(5.0)
        assert diag["per_joint"][j]["clamped"] is False
        assert diag["per_joint"][j]["rejected"] is False


def test_stage_diagnostics_identifies_a_dangerous_raw_outlier_joint():
    gate = _gate(max_step=5.0)
    current = _neutral(0.0)
    target = _neutral(1.0)
    target["shoulder_pan"] = 20.0  # delta=13.18 > 5.0 -> 이 joint만 clamp돼야 함 (C-5 wrist regression과 동일 패턴)
    diag = _evaluate_stage_diagnostics(safety_gate=gate, target_deg=target, current_state_deg=current)
    assert diag["decision"] == "WOULD_CLAMP"
    assert diag["per_joint"]["shoulder_pan"]["clamped"] is True
    assert diag["per_joint"]["shoulder_pan"]["delta_vs_current"] == pytest.approx(20.0)
    assert any("EXCESSIVE_STEP_CLAMPED" in r for r in diag["per_joint"]["shoulder_pan"]["reasons"])
    # 다른 5개 관절은 전부 정상(clamp 아님) - "어떤 joint가 원인인지" 구분이 핵심 요구사항.
    for j in JOINT_ORDER:
        if j == "shoulder_pan":
            continue
        assert diag["per_joint"][j]["clamped"] is False


def test_stage_diagnostics_gross_violation_is_reject():
    gate = _gate(max_step=5.0)  # GROSS_STEP_MULTIPLIER=5.0 -> gross 경계 25deg
    current = _neutral(0.0)
    target = _neutral(0.0)
    target["elbow_flex"] = 999.0
    diag = _evaluate_stage_diagnostics(safety_gate=gate, target_deg=target, current_state_deg=current)
    assert diag["decision"] == "REJECT"
    assert diag["per_joint"]["elbow_flex"]["rejected"] is True


def test_stage_diagnostics_none_target_returns_none():
    gate = _gate()
    assert _evaluate_stage_diagnostics(safety_gate=gate, target_deg=None, current_state_deg=_neutral()) is None
    assert _evaluate_stage_diagnostics(safety_gate=gate, target_deg=_neutral(), current_state_deg=None) is None


# ---------------------------------------------------------------------------
# DiagnosticCapturingGeneratorProxy - passthrough + capture, control loop을 절대 안 막음
# ---------------------------------------------------------------------------


class _StubGenerator:
    def __init__(self, result: ControlTargetResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def tick(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


def test_proxy_is_transparent_passthrough_when_recorder_is_none():
    result = _result(raw_target=_neutral(1.0), intent_decision="ACCEPT")
    inner = _StubGenerator(result)
    proxy = DiagnosticCapturingGeneratorProxy(inner, recorder=None)
    returned = proxy.tick(now_monotonic=1.0, current_follower_state_deg=_neutral(0.0), chunks=())
    assert returned is result  # 값 변형 없음 - 순수 passthrough
    assert proxy.results == [result]
    assert inner.calls == [{"now_monotonic": 1.0, "current_follower_state_deg": _neutral(0.0), "chunks": ()}]


def test_proxy_tick_survives_diagnostic_computation_blowing_up(tmp_path, monkeypatch):
    """진단 계산 내부(``_evaluate_stage_diagnostics``)가 예상 못한 예외를 던져도
    ``proxy.tick()``은 control loop 값을 그대로 정상 반환해야 한다 - 진단 로깅 실패가
    control loop을 절대 막으면 안 된다는 모듈 docstring의 핵심 불변조건."""
    import runtime.laptop.diagnostic_jsonl_logger as diag_module

    def _boom(**kwargs):
        raise RuntimeError("injected diagnostic computation failure")

    monkeypatch.setattr(diag_module, "_evaluate_stage_diagnostics", _boom)

    gate = _gate()
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)
    result = _result(raw_target=_neutral(1.0), intent_decision="ACCEPT")
    inner = _StubGenerator(result)
    proxy = DiagnosticCapturingGeneratorProxy(inner, recorder=recorder)

    returned = proxy.tick(now_monotonic=1.0, current_follower_state_deg=_neutral(0.0))
    assert returned is result  # control loop이 받는 값은 진단 실패와 무관하게 정상
    assert proxy.results == [result]

    tick = _tick_record(tick_index=0, state=ControlLoopState.RUNNING, intent_decision="ACCEPT",
                         safety_decision=None, actual_start=1.0)
    recorder.drain_and_write([tick])
    recorder.close()
    event = json.loads((tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip())
    assert event["intent_per_joint"] == {"capture_error": "RuntimeError: injected diagnostic computation failure"}


def test_capture_generator_tick_never_raises_on_malformed_input(tmp_path):
    gate = _gate()
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)
    # raw_ensemble_target에 joint가 하나 빠진 malformed dict -> adapt_vla_action이 내부적으로
    # invalid로 처리하거나 KeyError를 던질 수 있는 입력 - 그래도 capture는 죽지 않아야 한다.
    malformed = {j: 0.0 for j in JOINT_ORDER if j != "gripper"}
    result = _result(raw_target=malformed, intent_decision="ACCEPT")
    recorder.capture_generator_tick(now_monotonic=5.0, current_follower_state_deg=_neutral(0.0), result=result)
    tick = _tick_record(tick_index=0, state=ControlLoopState.RUNNING, intent_decision="ACCEPT",
                         safety_decision="ACCEPT", actual_start=5.0)
    recorder.drain_and_write([tick])
    recorder.close()
    lines = (tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    # capture가 실패했더라도 한 줄은 남아야 한다(요구사항: 누락 없음).
    assert event["tick_index"] == 0


# ---------------------------------------------------------------------------
# TickDiagnosticRecorder - end-to-end 한 줄짜리 JSONL 이벤트 검증
# ---------------------------------------------------------------------------


def test_full_tick_accept_and_write_round_trip(tmp_path):
    gate = _gate(max_step=5.0)
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)

    current = _neutral(0.0)
    raw = _neutral(1.0)
    guarded = _neutral(0.5)  # motion guard가 raw를 0.5만큼만 허용했다고 가정
    result = _result(
        raw_target=raw, intent_decision="ACCEPT", guarded_target=guarded, final_target=guarded,
        safety_decision="ACCEPT", contributing_sequences=(3, 4), target_valid=True,
        target_lookahead=_neutral(1.1), motion_guard_dt_s=1 / 60.0,
        motion_guard_diagnostics={
            "deterministic_reset": True, "phase_scale": 0.75, "phase_state": "SLOWDOWN",
            "pre_state": None, "post_state": {"positions": guarded},
        },
    )
    recorder.capture_generator_tick(now_monotonic=10.0, current_follower_state_deg=current, result=result)
    tick = _tick_record(
        tick_index=0, state=ControlLoopState.RUNNING, intent_decision="ACCEPT", safety_decision="ACCEPT",
        contributing_sequences=(3, 4), target_valid=True, write_attempted=True, write_executed=True,
        write_path="primary", actual_start=10.0, raw_target=raw, guarded_target=guarded, final_target=guarded,
    )
    n = recorder.drain_and_write([tick])
    assert n == 1
    recorder.close()

    lines = (tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])

    assert event["control_state"] == "RUNNING"
    assert event["current_follower_state_deg"] == current
    assert event["encoder_state"] == current
    assert event["raw_ensemble_target"] == raw
    assert event["raw_target"] == raw
    assert event["target_lookahead"] == _neutral(1.1)
    assert event["motion_guard_dt_s"] == pytest.approx(1 / 60.0)
    assert event["motion_guard_phase_scale"] == pytest.approx(0.75)
    assert event["motion_guard_phase_state"] == "SLOWDOWN"
    assert event["motion_guard_deterministic_reset"] is True
    assert event["motion_guard_pre_state"] is None
    assert event["motion_guard_post_state"]["positions"] == guarded
    assert event["guarded_target"] == guarded
    assert event["final_target"] == guarded
    assert event["clamp_reasons"] == []
    assert event["contributing_sequences"] == [3, 4]
    assert event["intent_per_joint"]["shoulder_pan"]["delta_vs_current"] == pytest.approx(1.0)
    assert event["final_safety_per_joint"]["shoulder_pan"]["delta_vs_current"] == pytest.approx(0.5)
    # motion guard가 raw->guarded로 얼마나 깎았는지 joint별로 직접 보임 (요구사항 4번:
    # "Motion Guard가 Intent-ACCEPT target을 Final Safety에서 막히는 target으로 만드는지").
    assert event["motion_guard_delta_deg"]["shoulder_pan"] == pytest.approx(0.5)
    assert event["write"]["attempted"] is True
    assert event["write"]["executed"] is True
    assert event["write"]["written_target_deg"] == guarded
    assert event["write"]["no_write_reason"] is None
    # 이 tick은 아무것도 quarantine되지 않음(intent ACCEPT).
    assert event["quarantine"]["newly_quarantined_this_tick"] == []


def test_intent_would_clamp_is_tick_local_fail_closed_without_quarantine(tmp_path):
    """intent_validation.py/realtime_control_target.py의 실제 동작 그대로: Intent가 막으면
    Motion Guard는 호출되지 않는다 - guarded_target=None이어야 하고, 그 tick의
    contributing_sequences가 quarantine에 새로 들어가야 한다(요구사항 1, 6)."""
    gate = _gate(max_step=5.0)
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)

    current = _neutral(0.0)
    raw = _neutral(0.0)
    raw["elbow_flex"] = 20.0  # threshold(5.0)를 넘는 outlier -> Intent WOULD_CLAMP
    result = _result(
        raw_target=raw, intent_decision="WOULD_CLAMP", guarded_target=None, safety_decision=None,
        contributing_sequences=(7,), stop_reason="INTENT_WOULD_CLAMP", target_valid=False,
    )
    recorder.capture_generator_tick(now_monotonic=20.0, current_follower_state_deg=current, result=result)
    tick = _tick_record(
        tick_index=0, state=ControlLoopState.INTENT_BLOCKED, intent_decision="WOULD_CLAMP", safety_decision=None,
        contributing_sequences=(7,), stop_reason="INTENT_WOULD_CLAMP", actual_start=20.0,
    )
    recorder.drain_and_write([tick])
    recorder.close()

    event = json.loads((tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip())
    assert event["intent_decision"] == "WOULD_CLAMP"
    assert event["guarded_target"] is None
    assert event["final_safety_decision"] is None
    # 원인 joint를 바로 지목할 수 있어야 한다 (요구사항 1).
    assert event["intent_per_joint"]["elbow_flex"]["clamped"] is True
    assert event["intent_per_joint"]["elbow_flex"]["delta_vs_current"] == pytest.approx(20.0)
    assert event["write"]["executed"] is False
    assert event["write"]["no_write_reason"] == "INTENT_WOULD_CLAMP"
    assert event["quarantine"]["newly_quarantined_this_tick"] == []
    assert event["quarantine"]["after_tick"] == []


def test_soft_mechanical_endpoint_saturation_writes_final_target(tmp_path):
    gate = _gate(range_deg=(-10.0, 10.0), max_step=100.0)
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)
    current = _neutral(0.0)
    raw = _neutral(1.0)
    guarded = _neutral(1.0)
    guarded["shoulder_pan"] = 12.0
    final = dict(guarded)
    final["shoulder_pan"] = 10.0
    reasons = ("MECHANICAL_LIMIT_CLAMPED: shoulder_pan 12.0 -> 10.0",)
    result = _result(
        raw_target=raw, intent_decision="ACCEPT", guarded_target=guarded, final_target=final,
        clamp_reasons=reasons, safety_decision="WOULD_CLAMP", target_valid=True,
    )
    recorder.capture_generator_tick(now_monotonic=25.0, current_follower_state_deg=current, result=result)
    tick = _tick_record(
        tick_index=0, state=ControlLoopState.RUNNING, intent_decision="ACCEPT",
        safety_decision="WOULD_CLAMP", raw_target=raw, guarded_target=guarded, final_target=final,
        clamp_reasons=reasons, target_valid=True, write_attempted=True, write_executed=True,
        write_path="primary", actual_start=25.0,
    )
    recorder.drain_and_write([tick])
    recorder.close()
    event = json.loads((tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip())
    assert event["guarded_target"] == guarded
    assert event["final_target"] == final
    assert event["clamp_reasons"] == list(reasons)
    assert event["final_safety_decision"] == "WOULD_CLAMP"
    assert event["write"]["attempted"] is True
    assert event["write"]["executed"] is True
    assert event["write"]["written_target_deg"] == final


def test_final_safety_reject_tick_is_distinguishable_from_would_clamp(tmp_path):
    """요구사항 3: Final Safety REJECT 2건의 원인 joint를 WOULD_CLAMP과 구분할 수 있어야 한다."""
    gate = _gate(max_step=5.0)
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)

    current = _neutral(0.0)
    raw = _neutral(0.0)
    guarded = _neutral(0.0)
    guarded["shoulder_lift"] = 999.0  # gross violation -> REJECT
    result = _result(
        raw_target=raw, intent_decision="ACCEPT", guarded_target=guarded, safety_decision="REJECT",
        contributing_sequences=(9,), stop_reason="SAFETY_REJECT", target_valid=False,
    )
    recorder.capture_generator_tick(now_monotonic=30.0, current_follower_state_deg=current, result=result)
    tick = _tick_record(
        tick_index=0, state=ControlLoopState.SAFETY_BLOCKED, intent_decision="ACCEPT", safety_decision="REJECT",
        contributing_sequences=(9,), stop_reason="SAFETY_REJECT", actual_start=30.0,
    )
    recorder.drain_and_write([tick])
    recorder.close()

    event = json.loads((tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip())
    assert event["final_safety_decision"] == "REJECT"
    assert event["final_safety_per_joint"]["shoulder_lift"]["rejected"] is True
    assert event["final_safety_per_joint"]["shoulder_lift"]["clamped"] is False
    # REJECT는 Final SafetyGate 단계에서 나왔지, Intent 단계는 ACCEPT였다는 게 한 줄로 보임.
    assert event["intent_decision"] == "ACCEPT"
    # REJECT도 intent_decision이 ACCEPT였으므로 quarantine 대상이 아니다(quarantine은
    # intent_decision != ACCEPT일 때만 트리거된다는 실제 규칙 그대로 재구성됨).
    assert event["quarantine"]["newly_quarantined_this_tick"] == []


def test_fault_tick_without_generator_call_still_gets_one_line(tmp_path):
    """state_read 실패 등으로 generator.tick()이 아예 호출 안 된 tick도 빠짐없이 한 줄
    남아야 한다(요구사항: "각 control tick에서 최소 다음을 기록")."""
    gate = _gate()
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)
    tick = _tick_record(
        tick_index=0, state=ControlLoopState.FAULT, intent_decision=None, safety_decision=None,
        stop_reason="FAULT", actual_start=40.0,
    )
    tick = ControlTickRecord(**{**tick.__dict__, "errors": ("state_read 실패: RuntimeError: boom",)})
    recorder.drain_and_write([tick])
    recorder.close()
    event = json.loads((tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip())
    assert event["control_state"] == "FAULT"
    assert event["raw_ensemble_target"] is None
    assert event["intent_per_joint"] is None
    assert event["errors"] == ["state_read 실패: RuntimeError: boom"]


def test_drain_is_idempotent_no_duplicate_lines(tmp_path):
    gate = _gate()
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)
    result = _result(raw_target=_neutral(0.0), intent_decision="ACCEPT", guarded_target=_neutral(0.0),
                      safety_decision="ACCEPT", target_valid=True)
    recorder.capture_generator_tick(now_monotonic=1.0, current_follower_state_deg=_neutral(0.0), result=result)
    tick = _tick_record(tick_index=0, state=ControlLoopState.RUNNING, intent_decision="ACCEPT",
                         safety_decision="ACCEPT", target_valid=True, actual_start=1.0)
    recorder.drain_and_write([tick])
    recorder.drain_and_write([tick])  # 같은 record로 다시 drain - 멱등
    recorder.drain_and_write([tick])
    recorder.close()
    lines = (tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_reject_quarantine_accumulates_across_ticks_and_ignores_duplicates(tmp_path):
    gate = _gate()
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)

    def _blocked(tick_index, seqs, ts, before, after):
        result = _result(raw_target=_neutral(99.0), intent_decision="REJECT", contributing_sequences=seqs)
        recorder.capture_generator_tick(now_monotonic=ts, current_follower_state_deg=_neutral(0.0), result=result)
        return _tick_record(tick_index=tick_index, state=ControlLoopState.INTENT_BLOCKED,
                             intent_decision="REJECT", safety_decision=None,
                             contributing_sequences=seqs, stop_reason="INTENT_REJECT", actual_start=ts,
                             quarantine_before_tick=before, quarantine_after_tick=after)

    ticks = [
        _blocked(0, (1, 2), 0.0, (), (2,)),
        _blocked(1, (2, 3), 1.0, (2,), (2, 3)),
        _blocked(2, (3,), 2.0, (3,), (3,)),
    ]
    recorder.drain_and_write(ticks)
    recorder.close()

    events = [json.loads(l) for l in (tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert events[0]["quarantine"]["newly_quarantined_this_tick"] == [2]
    assert events[0]["quarantine"]["after_tick"] == [2]
    assert events[1]["quarantine"]["before_tick"] == [2]
    assert events[1]["quarantine"]["newly_quarantined_this_tick"] == [3]
    assert events[1]["quarantine"]["after_tick"] == [2, 3]
    assert events[2]["quarantine"]["newly_quarantined_this_tick"] == []
    assert events[2]["quarantine"]["after_tick"] == [3]

    assert recorder.cross_check_final_quarantine(frozenset({3})) == []
    assert recorder.cross_check_final_quarantine(frozenset({2, 3})) != []  # drift가 있으면 경고를 낸다


def test_intent_reject_quarantines_contributing_sequence(tmp_path):
    """quarantine 재구성 규칙(``intent_decision not in (None, "ACCEPT")``)이 REJECT에도
    적용되는지 - Intent REJECT는 이번 세션들에서 관측되지 않았지만(reject=0), 재구성 로직이
    ACCEPT만 제외하는 실제 조건과 일치하는지는 별도로 검증해둔다."""
    gate = _gate()
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)
    result = _result(raw_target=_neutral(999.0), intent_decision="REJECT", contributing_sequences=(5,))
    recorder.capture_generator_tick(now_monotonic=0.0, current_follower_state_deg=_neutral(0.0), result=result)
    tick = _tick_record(tick_index=0, state=ControlLoopState.INTENT_BLOCKED, intent_decision="REJECT",
                         safety_decision=None, contributing_sequences=(5,), stop_reason="INTENT_REJECT",
                         actual_start=0.0, quarantine_before_tick=(), quarantine_after_tick=(5,))
    recorder.drain_and_write([tick])
    recorder.close()
    event = json.loads((tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip())
    assert event["quarantine"]["newly_quarantined_this_tick"] == [5]


def test_hold_path_write_is_labeled_but_value_left_null_honestly(tmp_path):
    gate = _gate()
    recorder = TickDiagnosticRecorder(tmp_path / "diag.jsonl", safety_gate=gate)
    result = _result(raw_target=None, intent_decision=None)
    recorder.capture_generator_tick(now_monotonic=0.0, current_follower_state_deg=_neutral(0.0), result=result)
    tick = _tick_record(tick_index=0, state=ControlLoopState.NO_TRAJECTORY, intent_decision=None,
                         safety_decision=None, stop_reason="NO_TARGET", write_attempted=True, write_executed=True,
                         write_path="hold_measured", actual_start=0.0)
    recorder.drain_and_write([tick])
    recorder.close()
    event = json.loads((tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip())
    assert event["write"]["path"] == "hold_measured"
    assert event["write"]["written_target_deg"] is None
    assert "note" in event["write"]
