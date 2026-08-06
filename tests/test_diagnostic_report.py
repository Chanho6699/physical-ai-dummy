"""simulation/mujoco/diagnostic_report.py 단위 테스트 (CSV/JSON 생성, 경로 결정, 토큰 비노출)."""

from __future__ import annotations

import csv
import json

from simulation.mujoco.diagnostic_report import (
    CSV_FIELDNAMES,
    build_json_summary,
    make_session_id,
    resolve_session_paths,
    write_csv_report,
    write_json_report,
)


def _sample_summary(**overrides) -> dict:
    base = dict(
        server_url="http://127.0.0.1:8001",
        duration_sec=20.0,
        requested_rate_hz=20.0,
        actual_sample_rate_hz=19.8,
        sample_count=396,
        latency_mean_ms=18.4,
        latency_max_ms=74.1,
        stale_count=0,
        timeout_count=0,
        joint_names=["wrist_flex"],
        max_abs_difference={"wrist_flex": 2.6},
        mean_abs_difference={"wrist_flex": 1.1},
        persistent_difference_events=1,
        follower_saturation_events=1,
        sign_mismatch_events=0,
        offset_suspected=[],
        mujoco_blocked_events=3,
        network_pause_events=0,
        warnings=[],
        final_result="WARN",
    )
    base.update(overrides)
    return base


def test_build_json_summary_has_all_required_keys():
    summary = build_json_summary(**_sample_summary())
    required_keys = {
        "server_url",
        "duration_sec",
        "requested_rate_hz",
        "actual_sample_rate_hz",
        "sample_count",
        "latency_mean_ms",
        "latency_max_ms",
        "stale_count",
        "timeout_count",
        "joint_names",
        "max_abs_difference",
        "mean_abs_difference",
        "persistent_difference_events",
        "follower_saturation_events",
        "sign_mismatch_events",
        "offset_suspected",
        "mujoco_blocked_events",
        "network_pause_events",
        "warnings",
        "final_result",
    }
    assert required_keys <= set(summary.keys())


def test_write_json_report_creates_file_with_content(tmp_path):
    summary = build_json_summary(**_sample_summary())
    path = tmp_path / "nested" / "session_x.json"
    write_json_report(path, summary)
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["final_result"] == "WARN"
    assert loaded["mujoco_blocked_events"] == 3


def test_write_json_report_strips_api_token_like_keys(tmp_path):
    summary = build_json_summary(**_sample_summary())
    summary["api_token"] = "super-secret"
    summary["Authorization"] = "Bearer super-secret"
    path = tmp_path / "session.json"
    write_json_report(path, summary)
    raw_text = path.read_text(encoding="utf-8")
    assert "super-secret" not in raw_text
    loaded = json.loads(raw_text)
    assert "api_token" not in loaded
    assert "Authorization" not in loaded


def test_write_csv_report_has_required_columns_in_order(tmp_path):
    rows = [
        {
            "local_timestamp": 1.0,
            "remote_timestamp": 2.0,
            "sequence": 5,
            "network_latency_ms": 10.0,
            "state_age_ms": 15.0,
            "joint_name": "wrist_flex",
            "leader_position_deg": 82.35,
            "follower_position_deg": 79.81,
            "difference_deg": 2.54,
            "leader_raw_tick": 3000,
            "follower_raw_tick": 2990,
            "mujoco_target_deg": 82.35,
            "mujoco_qpos_deg": 81.92,
            "mujoco_limit_margin_deg": 12.65,
            "safety_status": "PASS",
            "event_code": "",
            "blocked_reason": "",
        }
    ]
    path = tmp_path / "session.csv"
    write_csv_report(path, rows)
    assert path.is_file()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == list(CSV_FIELDNAMES)
        data_row = next(reader)
        assert data_row[header.index("joint_name")] == "wrist_flex"
        assert data_row[header.index("safety_status")] == "PASS"


def test_write_csv_report_fills_missing_fields_blank(tmp_path):
    rows = [{"joint_name": "gripper", "safety_status": "WARN"}]  # 나머지 필드 누락
    path = tmp_path / "session.csv"
    write_csv_report(path, rows)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert row["joint_name"] == "gripper"
        assert row["mujoco_target_deg"] == ""


def test_write_csv_report_ignores_extra_fields(tmp_path):
    rows = [{"joint_name": "gripper", "safety_status": "PASS", "not_a_real_column": "x"}]
    path = tmp_path / "session.csv"
    write_csv_report(path, rows)  # extrasaction="ignore"라 예외 없이 통과해야 함
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "not_a_real_column" not in reader.fieldnames


def test_resolve_session_paths_default_uses_reports_dir(tmp_path):
    paths = resolve_session_paths(
        reports_dir=tmp_path / "reports", session_id="20260806_120000", explicit_report_path=None, write_csv=True
    )
    assert paths.json_path == tmp_path / "reports" / "session_20260806_120000.json"
    assert paths.csv_path == tmp_path / "reports" / "session_20260806_120000.csv"
    assert (tmp_path / "reports").is_dir()


def test_resolve_session_paths_without_csv_when_not_recording(tmp_path):
    paths = resolve_session_paths(
        reports_dir=tmp_path / "reports", session_id="s1", explicit_report_path=None, write_csv=False
    )
    assert paths.csv_path is None
    assert paths.json_path is not None


def test_resolve_session_paths_explicit_path_derives_csv_sibling(tmp_path):
    explicit = tmp_path / "custom" / "my_report.json"
    paths = resolve_session_paths(reports_dir=tmp_path / "reports", session_id="s1", explicit_report_path=explicit, write_csv=True)
    assert paths.json_path == explicit
    assert paths.csv_path == explicit.with_suffix(".csv")
    assert explicit.parent.is_dir()


def test_make_session_id_format():
    import datetime

    fixed = datetime.datetime(2026, 8, 6, 12, 34, 56)
    assert make_session_id(fixed) == "20260806_123456"
