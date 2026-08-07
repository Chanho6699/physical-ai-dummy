"""scripts/analyze_instrumented_teleop_runs.py CLI 테스트 - 전부 fake CSV/JSON, offline."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_instrumented_teleop_runs.py"

pytest.importorskip("lerobot", reason="TeleopCycleSample 재사용을 위해 lerobot 환경에서 실행")

from hardware.diagnostics.instrumented_teleop import CSV_FIELDNAMES, TeleopCycleSample


def _load_cli_module():
    module_name = "analyze_instrumented_teleop_runs_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli():
    return _load_cli_module()


def _wrist_roll_deg(raw: float) -> float:
    return (raw - 2047.5) * 360.0 / 4095


ALL_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _sample(loop_index, *, elapsed_sec, present_raw=2023, goal_raw=2021) -> TeleopCycleSample:
    present_deg = _wrist_roll_deg(present_raw)
    goal_deg = _wrist_roll_deg(goal_raw)
    joints = {f"{j}.pos": 0.0 for j in ALL_JOINTS}
    return TeleopCycleSample(
        loop_index=loop_index,
        timestamp_iso="2026-08-07T00:00:00+00:00",
        elapsed_sec=elapsed_sec,
        loop_hz=59.0,
        leader_wrist_roll_deg=0.0,
        leader_wrist_roll_delta_from_start_deg=0.0,
        command_wrist_roll_deg=0.0,
        follower_goal_raw=goal_raw,
        follower_goal_deg=goal_deg,
        follower_present_raw=present_raw,
        follower_present_deg=present_deg,
        goal_present_error_raw=goal_raw - present_raw,
        goal_present_error_deg=goal_deg - present_deg,
        follower_present_delta_from_prev_raw=0,
        follower_present_delta_from_prev_deg=0.0,
        follower_present_delta_from_start_deg=0.0,
        follower_torque_enable=1,
        follower_acceleration=254,
        follower_acceleration_multiplier=1,
        follower_moving=0,
        follower_status=0,
        send_action_executed=True,
        leader_command_all_joints=dict(joints),
        follower_sent_all_joints=dict(joints),
        follower_observation_all_joints=dict(joints),
    )


def _write_run(tmp_path: Path, timestamp: str, n_samples: int = 30) -> None:
    samples = [_sample(i, elapsed_sec=i * 0.02) for i in range(n_samples)]
    csv_path = tmp_path / f"instrumented_wrist_roll_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        for s in samples:
            writer.writerow(s.to_csv_row())
    json_path = tmp_path / f"instrumented_wrist_roll_{timestamp}_report.json"
    report = {
        "stopped_reason": "DURATION_ELAPSED",
        "follower_start_present_raw": 2023,
        "follower_start_present_deg": _wrist_roll_deg(2023),
        "analysis": {
            "sample_count": n_samples,
            "elapsed_sec": (n_samples - 1) * 0.02,
            "actual_loop_hz": 59.0,
            "register_read_error_count": 0,
            "status_ever_nonzero": False,
            "command_to_actual_lag_estimate": "INSUFFICIENT_DATA",
        },
    }
    json_path.write_text(json.dumps(report), encoding="utf-8")


def test_cli_runs_end_to_end_and_writes_outputs(cli, tmp_path, capsys):
    for i in range(6):
        _write_run(tmp_path, f"20260101_0000{i:02d}")

    args = cli.build_arg_parser().parse_args(["--runs-dir", str(tmp_path), "--count", "6"])
    exit_code = cli.run(args)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "direct_register_write_count=0" in out
    assert "hardware_execution_count=0" in out

    json_outputs = list(tmp_path.glob("aggregate_6runs_*.json"))
    md_outputs = list(tmp_path.glob("aggregate_6runs_*.md"))
    assert len(json_outputs) == 1
    assert len(md_outputs) == 1

    report = json.loads(json_outputs[0].read_text(encoding="utf-8"))
    assert report["run_count"] == 6
    assert report["direct_register_write_count"] == 0


def test_cli_refuses_missing_runs_dir(cli, tmp_path, capsys):
    args = cli.build_arg_parser().parse_args(["--runs-dir", str(tmp_path / "does_not_exist")])
    exit_code = cli.run(args)
    assert exit_code == 2


def test_cli_warns_when_fewer_than_requested_runs_available(cli, tmp_path, capsys):
    for i in range(3):
        _write_run(tmp_path, f"20260101_0000{i:02d}")
    args = cli.build_arg_parser().parse_args(["--runs-dir", str(tmp_path), "--count", "6"])
    exit_code = cli.run(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "3개의 유효한 run" in out


def test_cli_does_not_modify_original_files(cli, tmp_path):
    for i in range(6):
        _write_run(tmp_path, f"20260101_0000{i:02d}")
    csv_files_before = {p.name: p.read_bytes() for p in tmp_path.glob("instrumented_wrist_roll_*.csv")}
    json_files_before = {p.name: p.read_bytes() for p in tmp_path.glob("instrumented_wrist_roll_*_report.json")}

    args = cli.build_arg_parser().parse_args(["--runs-dir", str(tmp_path), "--count", "6"])
    cli.run(args)

    for name, content in csv_files_before.items():
        assert (tmp_path / name).read_bytes() == content
    for name, content in json_files_before.items():
        assert (tmp_path / name).read_bytes() == content


def test_cli_never_imports_lerobot_or_hardware_safety():
    import inspect

    module = _load_cli_module()
    source = inspect.getsource(module)
    for forbidden in ("import lerobot", "from lerobot", "FeetechMotorsBus", "hardware.safety.single_joint"):
        assert forbidden not in source
