from __future__ import annotations

import json
from types import SimpleNamespace

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.diagnostic_jsonl_logger import TickDiagnosticRecorder
from runtime.laptop.realtime_control_loop import ControlLoopState, ControlTickRecord
from runtime.laptop.realtime_control_target import RealTimeControlTargetGenerator
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
from scripts.replay_motion_guard_exact import load_events, replay_events


class _HandoffEnsembler:
    def compute_target(self, chunks, target_time):
        del chunks
        elapsed = target_time - 10.0
        action = {
            joint: (index + 1) * 0.2 * elapsed
            for index, joint in enumerate(JOINT_ORDER)
        }
        if elapsed < 0.05:
            sequences = (1,)
        elif elapsed < 0.12:
            sequences = (2, 1)
        else:
            sequences = (2,)
        return SimpleNamespace(action=action, contributing_sequences=sequences)


def _gate() -> SafetyGate:
    return SafetyGate(SafetyGateConfig(
        joint_range_deg={joint: (-1000.0, 1000.0) for joint in JOINT_ORDER},
        max_step_deg={joint: 1000.0 for joint in JOINT_ORDER},
    ))


def _record(index, now, result):
    return ControlTickRecord(
        tick_index=index, scheduled_time_monotonic=now, actual_start_time_monotonic=now,
        dt_s=None if index == 0 else 1 / 60.0, tick_compute_ms=0.0, state_read_ms=0.0,
        target_compute_ms=0.0, write_ms=None, deadline_overrun_ms=0.0,
        intent_decision=result.intent_decision, safety_decision=result.safety_decision,
        raw_target=result.raw_ensemble_target, guarded_target=result.guarded_target,
        final_target=result.final_target, clamp_reasons=result.clamp_reasons,
        target_valid=result.target_valid, stop_reason=result.stop_reason,
        contributing_sequences=result.contributing_sequences,
        quarantined_sequences_excluded=(), trajectory_age_s=None,
        write_attempted=False, write_executed=False, write_path=None,
        state=ControlLoopState.RUNNING, errors=(),
    )


def test_synthetic_runtime_jsonl_exact_replay_including_handoff(tmp_path):
    gate = _gate()
    generator = RealTimeControlTargetGenerator(
        ensembler=_HandoffEnsembler(), safety_gate=gate, control_hz=60.0,
    )
    path = tmp_path / "synthetic_handoff.jsonl"
    recorder = TickDiagnosticRecorder(path, safety_gate=gate)
    encoder = {joint: 0.0 for joint in JOINT_ORDER}
    records = []
    for index in range(12):
        now = 10.0 + index / 60.0
        result = generator.tick(chunks=(object(),), now_monotonic=now,
                                current_follower_state_deg=encoder)
        assert result.target_valid
        recorder.capture_generator_tick(
            now_monotonic=now, current_follower_state_deg=encoder, result=result,
        )
        records.append(_record(index, now, result))
        encoder = dict(result.guarded_target)
    assert recorder.drain_and_write(records) == len(records)
    recorder.close()

    events = load_events(path)
    assert all(event["target_lookahead"] is not None for event in events)
    assert all(event["motion_guard_post_state"] is not None for event in events)
    report = replay_events(events, tolerance=1e-12)
    assert report["replayed_ticks"] == 12
    assert report["handoff_ticks"] == 2
    assert report["mismatch_ticks"] == 0
    assert report["max_error"] == 0.0
    assert report["handoff_max_error"] == 0.0


def test_legacy_target_log_is_rejected_instead_of_approximated():
    legacy = {
        "tick_index": 1,
        "current_follower_state_deg": {joint: 0.0 for joint in JOINT_ORDER},
        "raw_ensemble_target": {joint: 1.0 for joint in JOINT_ORDER},
        "guarded_target": {joint: 0.1 for joint in JOINT_ORDER},
    }
    try:
        replay_events([legacy])
    except ValueError as exc:
        assert "lacks exact replay fields" in str(exc)
    else:
        raise AssertionError("legacy log must not be silently approximated")
