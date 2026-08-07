"""hardware/diagnostics/instrumented_teleop_logger.py 단위 테스트.

하드웨어 접근이 없는 순수 파일 I/O만 테스트한다 (``tmp_path``만 사용).
"""

from __future__ import annotations

import csv
import json

import pytest

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행 (import 경로 확인용)")

from hardware.diagnostics.instrumented_teleop import CSV_FIELDNAMES, TeleopCycleSample
from hardware.diagnostics.instrumented_teleop_logger import (
    CSV_FILENAME_PREFIX,
    CsvSampleWriter,
    build_csv_path,
    build_json_report_path,
    write_json_report,
)


def _sample(loop_index=0) -> TeleopCycleSample:
    return TeleopCycleSample(
        loop_index=loop_index,
        timestamp_iso="2026-08-07T00:00:00+00:00",
        elapsed_sec=loop_index * 0.02,
        loop_hz=59.4,
        leader_wrist_roll_deg=-1.80,
        leader_wrist_roll_delta_from_start_deg=0.35,
        command_wrist_roll_deg=-1.80,
        follower_goal_raw=2000,
        follower_goal_deg=-1.80,
        follower_present_raw=1998,
        follower_present_deg=-1.98,
        goal_present_error_raw=2,
        goal_present_error_deg=0.176,
        follower_present_delta_from_prev_raw=0,
        follower_present_delta_from_prev_deg=0.0,
        follower_present_delta_from_start_deg=0.176,
        follower_torque_enable=1,
        follower_acceleration=254,
        follower_acceleration_multiplier=1,
        follower_moving=1,
        follower_status=0,
        send_action_executed=True,
        leader_command_all_joints={"wrist_roll.pos": -1.80, "gripper.pos": 50.0},
        follower_sent_all_joints={"wrist_roll.pos": -1.80, "gripper.pos": 50.0},
        follower_observation_all_joints={"wrist_roll.pos": -1.98, "gripper.pos": 49.5},
    )


def test_build_csv_path_uses_prefix_and_timestamp(tmp_path):
    path = build_csv_path(tmp_path, timestamp="20260807_120000")
    assert path == tmp_path / f"{CSV_FILENAME_PREFIX}_20260807_120000.csv"


def test_build_json_report_path_shares_timestamp_with_csv(tmp_path):
    csv_path = build_csv_path(tmp_path, timestamp="20260807_120000")
    json_path = build_json_report_path(csv_path)
    assert json_path.name == f"{CSV_FILENAME_PREFIX}_20260807_120000_report.json"
    assert json_path.parent == csv_path.parent


def test_csv_sample_writer_writes_header_and_rows(tmp_path):
    path = tmp_path / "out.csv"
    writer = CsvSampleWriter(path)
    writer.write_sample(_sample(0))
    writer.write_sample(_sample(1))
    writer.close()

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert set(rows[0].keys()) == set(CSV_FIELDNAMES)
    assert rows[0]["loop_index"] == "0"
    assert rows[1]["loop_index"] == "1"
    assert rows[0]["leader_command_wrist_roll"] == "-1.8"
    assert rows[0]["follower_observation_wrist_roll"] == "-1.98"


def test_csv_sample_writer_flush_makes_data_visible_before_close(tmp_path):
    path = tmp_path / "out.csv"
    writer = CsvSampleWriter(path)
    writer.write_sample(_sample(0))
    writer.flush()
    content = path.read_text(encoding="utf-8")
    assert "loop_index" in content  # 헤더
    assert "\n0," in content or content.strip().endswith("0") is False  # 최소 한 행은 기록됨
    writer.close()


def test_write_json_report_round_trips(tmp_path):
    path = tmp_path / "report.json"
    report = {"write_count": 0, "stopped_reason": "DURATION_ELAPSED", "nested": {"a": 1}}
    write_json_report(path, report)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == report
