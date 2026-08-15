#!/usr/bin/env python3
"""Offline exact replay of MotionGuard diagnostic JSONL; never opens hardware."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.motion_guard import (
    JointMotionLimits,
    CoordinatedGuardState,
    apply_coordinated_motion_guard,
)

REQUIRED_FIELDS = (
    "current_follower_state_deg", "raw_ensemble_target", "target_lookahead",
    "motion_guard_dt_s", "guarded_target", "motion_guard_deterministic_reset",
    "motion_guard_config",
)


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _max_joint_error(actual: dict[str, float], expected: dict[str, float]) -> float:
    return max(abs(float(actual[j]) - float(expected[j])) for j in JOINT_ORDER)


def replay_events(events: Iterable[dict[str, Any]], *, tolerance: float = 1e-9) -> dict[str, Any]:
    state: CoordinatedGuardState | None = None
    errors: list[dict[str, Any]] = []
    replayed = 0
    handoff_ticks = 0
    previous_sequences: tuple[int, ...] | None = None
    for event in events:
        missing = [field for field in REQUIRED_FIELDS if event.get(field) is None]
        if missing:
            if event.get("raw_ensemble_target") is None or event.get("guarded_target") is None:
                state = None
                previous_sequences = None
                continue
            raise ValueError(f"tick {event.get('tick_index')} lacks exact replay fields: {missing}")
        sequences = tuple(event.get("contributing_sequences") or ())
        is_handoff = previous_sequences is not None and sequences != previous_sequences
        handoff_ticks += int(is_handoff)
        previous_sequences = sequences
        if event["motion_guard_deterministic_reset"]:
            state = None
        config = event["motion_guard_config"]
        limits = {
            joint: JointMotionLimits(**values)
            for joint, values in config["motion_limits"].items()
        }
        guarded, state = apply_coordinated_motion_guard(
            limits_by_joint=limits,
            current_state=event["current_follower_state_deg"],
            target_now=event["raw_ensemble_target"],
            target_lookahead=event["target_lookahead"],
            prev_state=state,
            dt_s=float(event["motion_guard_dt_s"]),
            correction_time_constant_s=float(config["correction_time_constant_s"]),
            tracking_lead_limits=config["tracking_lead_limits"],
        )
        target_error = _max_joint_error(guarded, event["guarded_target"])
        phase_error = abs(state.phase_scale - float(event["motion_guard_phase_scale"]))
        post = event.get("motion_guard_post_state") or {}
        state_error = 0.0
        for name in ("positions", "velocities", "accelerations"):
            recorded = post.get(name)
            if recorded is not None:
                state_error = max(state_error, _max_joint_error(getattr(state, name), recorded))
        worst = max(target_error, phase_error, state_error)
        errors.append({
            "tick_index": event.get("tick_index"), "handoff": is_handoff,
            "guarded_target_max_error": target_error, "phase_scale_error": phase_error,
            "post_state_max_error": state_error, "max_error": worst,
        })
        replayed += 1
    values = [row["max_error"] for row in errors]
    handoff_values = [row["max_error"] for row in errors if row["handoff"]]
    max_error = max(values, default=0.0)
    return {
        "replayed_ticks": replayed, "handoff_ticks": handoff_ticks, "tolerance": tolerance,
        "mean_error": sum(values) / len(values) if values else 0.0, "max_error": max_error,
        "handoff_max_error": max(handoff_values, default=0.0),
        "mismatch_ticks": sum(value > tolerance for value in values),
        "exact_within_tolerance": bool(values) and max_error <= tolerance,
        "worst_ticks": sorted(errors, key=lambda row: row["max_error"], reverse=True)[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    if args.tolerance < 0 or not math.isfinite(args.tolerance):
        parser.error("--tolerance must be finite and non-negative")
    report = replay_events(load_events(args.jsonl), tolerance=args.tolerance)
    print(json.dumps(report, indent=2))
    return 0 if report["exact_within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
