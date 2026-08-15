from __future__ import annotations

import ast
from pathlib import Path

from scripts import run_readonly_real_shadow as shadow


def test_shadow_runner_has_no_writer_or_safety_imports() -> None:
    path = Path(shadow.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("follower_action_writer", "staged_follower_writer", "motion_guard", "intent", "safety_gate")
    assert not any(token in module for module in imported for token in forbidden)
    assert shadow.WRITER_CREATED is False
    assert shadow.WRITER_CALL_COUNT == 0


def test_parser_defaults_to_by_id_and_seed() -> None:
    args = shadow.parser().parse_args(["replay", "--model-label", "v1-7.5k", "--output-dir", "/tmp/out"])
    assert args.seed == 20260815
    capture = shadow.parser().parse_args(["capture"])
    assert capture.follower_port == shadow.DEFAULT_PORT
    assert 20 <= capture.duration_s <= 30


def test_analysis_reports_temporal_metrics() -> None:
    a = {joint: 0.0 for joint in shadow.JOINT_ORDER}
    b = {joint: 1.0 for joint in shadow.JOINT_ORDER}
    chunks = [{"raw_chunk": [a] * 11}, {"raw_chunk": [b] * 11}]
    ticks = [
        {"tick_offset_s": 0.0, "handoff_tick": False, "raw_interpolated_target": a,
         "temporal_ensemble_interpolated_target": a},
        {"tick_offset_s": 1 / 60, "handoff_tick": True, "raw_interpolated_target": b,
         "temporal_ensemble_interpolated_target": b},
    ]
    observations = [{"state": a}, {"state": b}]
    report = shadow._analyze(chunks, ticks, observations)
    assert report["consecutive_chunk_first_action_jump_l2"]["mean"] > 0
    assert "target_velocity_variance" in report


def test_virtual_replay_known_overlap_and_expiration() -> None:
    from scripts.virtualize_readonly_shadow_replay import simulate, virtual_chunks

    def action(value):
        return {joint: float(value) for joint in shadow.JOINT_ORDER}

    observations = [
        {"sequence": 0, "timestamp": 10.0},
        {"sequence": 1, "timestamp": 10.5},
    ]
    chunks = [
        {
            "sequence": i, "raw_chunk": [action(i)] * 4, "chunk_size": 4,
            "chunk_index_spacing_s": 0.5, "inference_latency_ms": 250.0,
            "model_id": "fixture", "backend": "fixture",
        }
        for i in range(2)
    ]
    timed, _ = virtual_chunks(observations, chunks, "latency-replay")
    assert timed[0]["observation_offset_s"] == 0.0
    assert timed[0]["response_offset_s"] == 0.25
    assert timed[1]["response_offset_s"] == 0.75

    ticks, handoffs, report = simulate(timed, end_s=2.0, hz=20.0, half_life=0.338)
    assert report["usable_target_fraction"] > 0.8
    assert report["stale_fraction"] == 1 / 41  # exact last-sample tick has no future action
    assert report["handoff_count"] == 1
    assert report["contributor_overlap_duration_s"] > 0.7
    assert any(t["contributor_count"] == 2 for t in ticks)
    assert handoffs[0]["contributors_before"] == [0]
    assert handoffs[0]["contributors_after"] == [1, 0]


def test_virtual_replay_zero_compute_publication() -> None:
    from scripts.virtualize_readonly_shadow_replay import virtual_chunks

    joint = {name: 0.0 for name in shadow.JOINT_ORDER}
    observations = [{"sequence": 0, "timestamp": 3.0}]
    chunks = [{"sequence": 0, "raw_chunk": [joint] * 2, "chunk_size": 2,
               "chunk_index_spacing_s": 0.5, "inference_latency_ms": 999.0}]
    timed, end_s = virtual_chunks(observations, chunks, "zero-compute")
    assert end_s == 0.0
    assert timed[0]["response_offset_s"] == 0.0
