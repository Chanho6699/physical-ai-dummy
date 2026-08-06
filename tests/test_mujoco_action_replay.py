from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import build_synthetic_dataset
from simulation.mujoco.dataset_action_replay import ReplayArgs, run_replay


def _args(dataset_root: Path, **overrides) -> ReplayArgs:
    defaults = dict(
        dataset_root=dataset_root,
        episode_index=0,
        speed=1.0,
        mode="headless",
        max_frames=None,
        start_frame=0,
        report_path=None,
        config_path=None,
        quiet=False,
        verbose=False,
        no_color=True,
        dry_run=False,
        continue_on_warning=True,
    )
    defaults.update(overrides)
    return ReplayArgs(**defaults)


def test_dry_run_produces_pass_or_warn_report(tmp_path):
    root = build_synthetic_dataset(tmp_path / "ds", num_frames=15)
    args = _args(root, dry_run=True, report_path=tmp_path / "report.json")
    outcome = run_replay(args)
    assert outcome.final_result in ("PASS", "WARN")
    assert outcome.exit_code == 0
    assert outcome.report["processed_frames"] == 0
    assert outcome.report["dry_run"] is True
    assert outcome.report_path.is_file()


def test_headless_replay_full_episode(tmp_path):
    root = build_synthetic_dataset(tmp_path / "ds", num_frames=20)
    args = _args(root, report_path=tmp_path / "report.json")
    outcome = run_replay(args)
    assert outcome.final_result in ("PASS", "WARN", "BLOCKED")
    assert outcome.report["frame_count"] == 20
    assert outcome.report["processed_frames"] <= 20
    assert outcome.report["joint_names"] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]


def test_headless_partial_replay_max_frames(tmp_path):
    root = build_synthetic_dataset(tmp_path / "ds", num_frames=50)
    args = _args(root, max_frames=10, report_path=tmp_path / "report.json")
    outcome = run_replay(args)
    assert outcome.report["frame_count"] == 10


def test_json_report_is_valid_json_and_matches_terminal_summary(tmp_path):
    root = build_synthetic_dataset(tmp_path / "ds", num_frames=10)
    report_path = tmp_path / "report.json"
    args = _args(root, report_path=report_path)
    outcome = run_replay(args)
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk["final_result"] == outcome.final_result
    assert on_disk == outcome.report


def test_missing_dataset_returns_error(tmp_path):
    args = _args(tmp_path / "no_such_dataset", dry_run=True)
    outcome = run_replay(args)
    assert outcome.exit_code != 0
    assert outcome.final_result == "BLOCKED"


def test_invalid_episode_index_returns_error(tmp_path):
    root = build_synthetic_dataset(tmp_path / "ds", num_frames=5)
    args = _args(root, episode_index=99, dry_run=True)
    outcome = run_replay(args)
    assert outcome.exit_code != 0


def test_quiet_suppresses_header(tmp_path, capsys):
    root = build_synthetic_dataset(tmp_path / "ds", num_frames=5)
    args = _args(root, dry_run=True, quiet=True, report_path=tmp_path / "report.json")
    run_replay(args)
    captured = capsys.readouterr()
    assert "[준비] SO-101 MuJoCo" not in captured.out
    assert "[완료]" in captured.out  # 최종 결과는 quiet에서도 출력되어야 함


def test_no_color_has_no_ansi_escape_codes(tmp_path, capsys):
    root = build_synthetic_dataset(tmp_path / "ds", num_frames=5)
    args = _args(root, dry_run=True, no_color=True, report_path=tmp_path / "report.json")
    run_replay(args)
    captured = capsys.readouterr()
    assert "\033[" not in captured.out


def test_report_path_does_not_overwrite_existing_file(tmp_path):
    root = build_synthetic_dataset(tmp_path / "ds", num_frames=5)
    fixed_path = tmp_path / "fixed_report.json"
    args = _args(root, dry_run=True, report_path=fixed_path)
    outcome1 = run_replay(args)
    outcome2 = run_replay(args)
    assert outcome1.report_path == fixed_path
    assert outcome2.report_path != fixed_path
    assert outcome2.report_path.is_file()


def test_action_out_of_range_causes_blocked(tmp_path):
    values = np.zeros((10, 6), dtype=np.float32)
    values[3, 1] = 999.0  # shoulder_lift: 명백히 관절 range 밖
    root = build_synthetic_dataset(tmp_path / "ds", num_frames=10, action_values=values)
    args = _args(root, report_path=tmp_path / "report.json")
    outcome = run_replay(args)
    assert outcome.final_result == "BLOCKED"
    assert outcome.exit_code == 1
    assert outcome.report["blocked_reason"] is not None


@pytest.mark.slow
def test_real_dataset_smoke_60_frames(real_dataset_root):
    args = _args(real_dataset_root, max_frames=60, report_path=None)
    outcome = run_replay(args)
    assert outcome.report["frame_count"] == 60
    assert outcome.final_result in ("PASS", "WARN", "BLOCKED")
    assert outcome.report_path.is_file()
