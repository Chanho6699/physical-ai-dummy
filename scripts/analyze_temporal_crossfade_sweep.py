#!/usr/bin/env python3
"""Compare phase-continuity fade sweeps without touching hardware."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "gripper")


def _percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _reversal_metrics(
    positions: np.ndarray,
    times: np.ndarray,
    handoff_ticks: np.ndarray,
    fade_s: float,
) -> dict:
    velocity = np.diff(positions) / np.diff(times)
    handoff_times = times[handoff_ticks]
    events: list[tuple[float, bool]] = []
    for index in range(1, len(velocity)):
        if not np.isfinite(velocity[index - 1 : index + 1]).all():
            continue
        if velocity[index - 1] * velocity[index] >= 0.0:
            continue
        amplitude = min(abs(velocity[index - 1]), abs(velocity[index]))
        in_fade = bool(
            np.any(
                (handoff_times <= times[index])
                & (times[index] - handoff_times <= fade_s + 1e-9)
            )
        )
        events.append((amplitude, in_fade))
    duration = float(times[-1] - times[0])
    amplitudes = np.asarray([event[0] for event in events], dtype=float)
    return {
        "reversal_per_s": len(events) / duration,
        "small_reversal_per_s": sum(amp < 5.0 for amp, _ in events) / duration,
        "large_reversal_per_s": sum(amp >= 5.0 for amp, _ in events) / duration,
        "amplitude_p50": _percentile(amplitudes, 50),
        "amplitude_p95": _percentile(amplitudes, 95),
        "energy_per_s": float(np.sum(amplitudes**2) / duration),
        "fade_window_fraction": (
            sum(in_fade for _, in_fade in events) / len(events) if events else 0.0
        ),
    }


def analyze(directory: Path, scale: float) -> dict:
    rows = _read_jsonl(directory / "virtual_targets_60hz.jsonl")
    usable = [
        row
        for row in rows
        if row["temporal_ensemble_interpolated_target"] is not None
        and row["raw_interpolated_target"] is not None
    ]
    lag = np.asarray(
        [
            math.dist(
                list(row["raw_interpolated_target"].values()),
                list(row["temporal_ensemble_interpolated_target"].values()),
            )
            for row in usable
        ]
    )
    times = np.asarray([row["virtual_time_s"] for row in usable])
    handoff_ticks = np.asarray([row["handoff_tick"] for row in usable], dtype=bool)
    chunks = _read_jsonl(directory / "virtual_chunks.jsonl")
    cadence_s = float(np.median(np.diff([row["observation_offset_s"] for row in chunks])))
    analysis = _read_json(directory / "analysis.json")
    report = _read_json(directory / "report.json")
    result = {
        "handoff_mean": analysis["handoff_jump_l2"]["ensemble_interpolated"]["mean"],
        "handoff_p95": analysis["handoff_jump_l2"]["ensemble_interpolated"]["p95"],
        "lag_mean": float(np.mean(lag)),
        "lag_p95": _percentile(lag, 95),
        "usable_fraction": report["usable_target_fraction"],
        "no_target_fraction": report["no_target_fraction"],
        "stale_fraction": report["stale_fraction"],
        "joints": {},
    }
    for joint in JOINTS:
        positions = np.asarray(
            [row["temporal_ensemble_interpolated_target"][joint] for row in usable]
        )
        raw_positions = np.asarray([row["raw_interpolated_target"][joint] for row in usable])
        window = 30  # 0.5 seconds at 60 Hz
        raw_direction = raw_positions[window:] - raw_positions[:-window]
        ensemble_direction = positions[window:] - positions[:-window]
        moving = np.abs(raw_direction) >= 0.25
        direction_agreement = (
            float(np.mean(np.sign(raw_direction[moving]) == np.sign(ensemble_direction[moving])))
            if np.any(moving)
            else None
        )
        stream = analysis["target_stream"][joint]
        jump = analysis["handoff_joint"][joint]["ensemble"]
        result["joints"][joint] = {
            "handoff_mean": jump["mean"],
            "handoff_p95": jump["p95"],
            "reversal": _reversal_metrics(positions, times, handoff_ticks, cadence_s * scale),
            "velocity_variance": stream["velocity_variance"],
            "acceleration_rms": stream["acceleration_rms"],
            "jerk_rms": stream["jerk_rms"],
            "direction_agreement_0p5s": direction_agreement,
            "total_variation_per_s": float(np.sum(np.abs(np.diff(positions))) / (times[-1] - times[0])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    models = {"V1": "v1_7500_seed20260815", "V2": "v2_10000_seed20260815"}
    scales = {"0.25": "025", "0.50": "050", "0.75": "075", "1.00": "100"}
    result = {
        model: {
            scale: analyze(args.replay_root / f"{base}_fade_{tag}", float(scale))
            for scale, tag in scales.items()
        }
        for model, base in models.items()
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
