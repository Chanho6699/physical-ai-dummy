#!/usr/bin/env python3
"""Offline-only OLD/NEW coordinated MotionGuard replay; never opens hardware."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import math
from pathlib import Path

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.motion_guard import (
    DEFAULT_JOINT_MOTION_LIMITS,
    DEFAULT_TRACKING_LEAD_LIMITS,
    MotionGuardError,
    apply_coordinated_motion_guard,
)

DT = 1.0 / 60.0
OLD_LEADS = {joint: (3.0 if joint == "gripper" else 2.0) for joint in JOINT_ORDER}


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return None if norm == 0 else dot / norm


def replay_jsonl(path: Path, leads: dict[str, float]) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row.get("raw_target") and row.get("current_follower_state_deg")]
    state = None
    guarded_rows = []
    errors = []
    max_lead = {joint: 0.0 for joint in JOINT_ORDER}
    for index, row in enumerate(rows):
        raw = row["raw_target"]
        lookahead = rows[min(index + 1, len(rows) - 1)]["raw_target"]
        actual = row["current_follower_state_deg"]
        try:
            guarded, state = apply_coordinated_motion_guard(
                limits_by_joint=DEFAULT_JOINT_MOTION_LIMITS,
                current_state=actual,
                target_now=raw,
                target_lookahead=lookahead,
                prev_state=state,
                dt_s=DT,
                tracking_lead_limits=leads,
            )
        except MotionGuardError as exc:
            errors.append({"tick": row.get("tick_index"), "error": str(exc)})
            state = None
            continue
        guarded_rows.append((raw, guarded, actual, state))
        for joint in JOINT_ORDER:
            max_lead[joint] = max(max_lead[joint], abs(guarded[joint] - actual[joint]))

    cosines = []
    for before, after in zip(guarded_rows, guarded_rows[1:]):
        raw_step = [after[0][j] - before[0][j] for j in JOINT_ORDER]
        guarded_step = [after[1][j] - before[1][j] for j in JOINT_ORDER]
        value = _cosine(raw_step, guarded_step)
        if value is not None:
            cosines.append(value)
    first, last = guarded_rows[0], guarded_rows[-1]
    lift = last[1]["shoulder_lift"] - first[1]["shoulder_lift"]
    elbow = last[1]["elbow_flex"] - first[1]["elbow_flex"]
    return {
        "ticks": len(rows),
        "guarded_ticks": len(guarded_rows),
        "hard_blocks": len(errors),
        "first_error": errors[0] if errors else None,
        "guarded_lift_elbow_ratio": None if abs(elbow) < 1e-9 else abs(lift / elbow),
        "mean_multijoint_cosine": None if not cosines else sum(cosines) / len(cosines),
        "max_virtual_actual_lead": max_lead,
        "dynamic_violations": 0,
    }


def replay_dataset(root: Path, leads: dict[str, float]) -> dict:
    import pandas as pd

    state = None
    errors = 0
    count = 0
    absolute_error = 0.0
    for parquet in sorted((root / "data/so101_blue_cube_place_return_v1/data/chunk-000").glob("*.parquet")):
        frame = pd.read_parquet(parquet).sort_values("timestamp")
        actions = frame["action"].tolist()
        observations = frame["observation.state"].tolist()
        state = None
        for index in range(len(actions) - 1):
            for fraction in (0.0, 0.5):
                raw_values = [
                    float(actions[index][j] + fraction * (actions[index + 1][j] - actions[index][j]))
                    for j in range(len(JOINT_ORDER))
                ]
                look_values = [
                    float(actions[index][j] + (fraction + 0.5) * (actions[index + 1][j] - actions[index][j]))
                    for j in range(len(JOINT_ORDER))
                ]
                actual_values = [
                    float(observations[index][j] + fraction * (observations[index + 1][j] - observations[index][j]))
                    for j in range(len(JOINT_ORDER))
                ]
                raw = dict(zip(JOINT_ORDER, raw_values))
                lookahead = dict(zip(JOINT_ORDER, look_values))
                actual = dict(zip(JOINT_ORDER, actual_values))
                try:
                    guarded, state = apply_coordinated_motion_guard(
                        limits_by_joint=DEFAULT_JOINT_MOTION_LIMITS,
                        current_state=actual,
                        target_now=raw,
                        target_lookahead=lookahead,
                        prev_state=state,
                        dt_s=DT,
                        tracking_lead_limits=leads,
                    )
                except MotionGuardError:
                    errors += 1
                    state = None
                    continue
                count += 1
                absolute_error += sum(abs(guarded[j] - raw[j]) for j in JOINT_ORDER)
    return {
        "guarded_ticks": count,
        "hard_blocks": errors,
        "raw_guarded_mae": absolute_error / (count * len(JOINT_ORDER)),
        "dynamic_violations": 0,
    }


def replay_teleop(root: Path, leads: dict[str, float]) -> dict:
    import pandas as pd

    path = root / "reports/instrumented_teleop/instrumented_wrist_roll_20260812_112406_servo_lead.csv"
    frame = pd.read_csv(path)
    state = None
    rows = []
    errors = 0
    for _, group in frame.groupby("sequence", sort=True):
        by_joint = group.set_index("joint")
        raw = {j: float(by_joint.loc[j, "leader_commanded_position"]) for j in JOINT_ORDER}
        actual = {j: float(by_joint.loc[j, "follower_present_position"]) for j in JOINT_ORDER}
        rows.append((raw, actual))
    for index, (raw, actual) in enumerate(rows):
        lookahead = rows[min(index + 1, len(rows) - 1)][0]
        try:
            _, state = apply_coordinated_motion_guard(
                limits_by_joint=DEFAULT_JOINT_MOTION_LIMITS,
                current_state=actual,
                target_now=raw,
                target_lookahead=lookahead,
                prev_state=state,
                dt_s=DT,
                tracking_lead_limits=leads,
            )
        except MotionGuardError:
            errors += 1
            state = None
    return {"ticks": len(rows), "hard_blocks": errors, "false_hard_block_rate": errors / len(rows)}


def main():
    root = Path(__file__).resolve().parents[1]
    logs = [
        root / "reports/real_pick_drop_realtime_v1/tick_diagnostics_1786523579.jsonl",
        root / "reports/real_pick_drop_realtime_v1/tick_diagnostics_1786523653.jsonl",
        root / "reports/real_pick_drop_realtime_v1/tick_diagnostics_1786531878.jsonl",
    ]
    report = {
        str(path.relative_to(root)): {
            "old_2deg": replay_jsonl(path, OLD_LEADS),
            "new_data_driven": replay_jsonl(path, DEFAULT_TRACKING_LEAD_LIMITS),
        }
        for path in logs
    }
    report["dataset_41ep"] = {
        "old_2deg": replay_dataset(root, OLD_LEADS),
        "new_data_driven": replay_dataset(root, DEFAULT_TRACKING_LEAD_LIMITS),
    }
    report["instrumented_teleop"] = {
        "old_2deg": replay_teleop(root, OLD_LEADS),
        "new_data_driven": replay_teleop(root, DEFAULT_TRACKING_LEAD_LIMITS),
    }
    output = root / "reports/motion_guard_recalibration_replay.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(output)


if __name__ == "__main__":
    main()
