"""hardware/diagnostics/instrumented_teleop_aggregate.py 단위 테스트.

전부 fake CSV/JSON fixture(``tmp_path``)로만 검증한다 - 실제 serial/hardware/lerobot 객체는
전혀 쓰지 않는다(``pytest.importorskip("lerobot")``은 ``TeleopCycleSample`` 등 데이터
구조를 재사용하기 위한 것일 뿐, 이 모듈 자체는 lerobot 없이도 import된다 - 별도로
``test_instrumented_teleop_aggregate_importable_without_lerobot``에서 확인한다).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("lerobot", reason="TeleopCycleSample 등 데이터 구조 재사용을 위해 lerobot 환경에서 실행")

from hardware.diagnostics.instrumented_teleop import CSV_FIELDNAMES, TeleopCycleSample
from hardware.diagnostics.instrumented_teleop_aggregate import (
    HIGH_RESPONSE_REGION,
    NO_RESPONSE_REGION,
    QUALITY_OK,
    QUALITY_WARNING,
    TRANSITION_REGION,
    RunBundle,
    assess_run_quality,
    build_aggregate_report,
    classify_deadband_region,
    compute_candidates,
    compute_deadband_aggregate,
    compute_frame_deltas,
    compute_joint_aggregate,
    compute_joint_value_series,
    compute_latency_aggregate,
    compute_run_to_run_stability,
    compute_velocities,
    discover_run_files,
    load_run_bundle,
    load_run_report,
    load_run_samples,
    percentile_summary,
    render_markdown_report,
    row_to_sample,
    select_latest_runs,
)


def _wrist_roll_deg(raw: float) -> float:
    return (raw - 2047.5) * 360.0 / 4095


ALL_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _sample(
    loop_index,
    *,
    elapsed_sec,
    present_raw=2023,
    goal_raw=2021,
    leader_deg=0.0,
    command_deg=0.0,
    status=0,
    torque=1,
    accel=254,
    per_joint_command=None,
    per_joint_observation=None,
    warning_types=(),
    register_read_error=None,
) -> TeleopCycleSample:
    present_deg = _wrist_roll_deg(present_raw)
    goal_deg = _wrist_roll_deg(goal_raw)
    per_joint_command = per_joint_command or {f"{j}.pos": 0.0 for j in ALL_JOINTS}
    per_joint_command["wrist_roll.pos"] = command_deg
    per_joint_observation = per_joint_observation or {f"{j}.pos": 0.0 for j in ALL_JOINTS}
    per_joint_observation["wrist_roll.pos"] = present_deg
    return TeleopCycleSample(
        loop_index=loop_index,
        timestamp_iso="2026-08-07T00:00:00+00:00",
        elapsed_sec=elapsed_sec,
        loop_hz=59.0,
        leader_wrist_roll_deg=leader_deg,
        leader_wrist_roll_delta_from_start_deg=leader_deg,
        command_wrist_roll_deg=command_deg,
        follower_goal_raw=goal_raw,
        follower_goal_deg=goal_deg,
        follower_present_raw=present_raw,
        follower_present_deg=present_deg,
        goal_present_error_raw=goal_raw - present_raw,
        goal_present_error_deg=goal_deg - present_deg,
        follower_present_delta_from_prev_raw=0,
        follower_present_delta_from_prev_deg=0.0,
        follower_present_delta_from_start_deg=present_deg - _wrist_roll_deg(2023),
        follower_torque_enable=torque,
        follower_acceleration=accel,
        follower_acceleration_multiplier=1,
        follower_moving=0,
        follower_status=status,
        send_action_executed=True,
        leader_command_all_joints=per_joint_command,
        follower_sent_all_joints=dict(per_joint_command),
        follower_observation_all_joints=per_joint_observation,
        register_read_error=register_read_error,
        warning_types=warning_types,
    )


def _write_run(
    tmp_path: Path,
    timestamp: str,
    samples: list[TeleopCycleSample],
    *,
    stopped_reason: str = "DURATION_ELAPSED",
    analysis_overrides: dict | None = None,
    write_json: bool = True,
    csv_padding_rows: int = 0,
    follower_start_present_raw: int = 2023,
) -> tuple[Path, Path | None]:
    csv_path = tmp_path / f"instrumented_wrist_roll_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        for s in samples:
            writer.writerow(s.to_csv_row())
        for _ in range(csv_padding_rows):
            f.write("garbage,row,not,enough,columns\n")

    json_path = None
    if write_json:
        json_path = tmp_path / f"instrumented_wrist_roll_{timestamp}_report.json"
        analysis = {
            "sample_count": len(samples),
            "elapsed_sec": samples[-1].elapsed_sec if samples else 0.0,
            "actual_loop_hz": (len(samples) - 1) / samples[-1].elapsed_sec if len(samples) > 1 and samples[-1].elapsed_sec else 0.0,
            "register_read_error_count": sum(1 for s in samples if s.register_read_error is not None),
            "status_ever_nonzero": any((s.follower_status or 0) != 0 for s in samples),
            "command_movement_range_deg": 1.0,
            "follower_movement_range_deg": 1.0,
            "command_to_actual_lag_estimate": "INSUFFICIENT_DATA",
        }
        if analysis_overrides:
            analysis.update(analysis_overrides)
        report = {
            "stopped_reason": stopped_reason,
            "follower_start_present_raw": follower_start_present_raw,
            "follower_start_present_deg": _wrist_roll_deg(follower_start_present_raw),
            "analysis": analysis,
        }
        json_path.write_text(json.dumps(report), encoding="utf-8")

    return csv_path, json_path


# ---------------------------------------------------------------------------
# discover/select
# ---------------------------------------------------------------------------


def test_discover_run_files_matches_and_sorts(tmp_path):
    _write_run(tmp_path, "20260101_000000", [_sample(0, elapsed_sec=0.0)])
    _write_run(tmp_path, "20260101_000100", [_sample(0, elapsed_sec=0.0)])
    (tmp_path / "not_a_run.csv").write_text("x", encoding="utf-8")

    found = discover_run_files(tmp_path)
    assert [ts for ts, _, _ in found] == ["20260101_000000", "20260101_000100"]


def test_select_latest_runs_excludes_tiny_files_and_missing_json(tmp_path):
    _write_run(tmp_path, "20260101_000000", [_sample(0, elapsed_sec=0.0)] * 50)  # 정상 크기
    tiny_csv = tmp_path / "instrumented_wrist_roll_20260101_000100.csv"
    tiny_csv.write_text("loop_index\n", encoding="utf-8")  # dry-run 스타일 빈 파일
    _write_run(tmp_path, "20260101_000200", [_sample(0, elapsed_sec=0.0)] * 50, write_json=False)  # json 없음

    selected = select_latest_runs(tmp_path, count=6, min_csv_bytes=50)
    timestamps = [ts for ts, _, _ in selected]
    assert "20260101_000000" in timestamps
    assert "20260101_000100" not in timestamps  # 너무 작음
    assert "20260101_000200" not in timestamps  # json 없음


def test_select_latest_runs_returns_most_recent_n(tmp_path):
    for i in range(9):
        _write_run(tmp_path, f"20260101_0000{i:02d}", [_sample(0, elapsed_sec=0.0)] * 50)
    selected = select_latest_runs(tmp_path, count=6, min_csv_bytes=50)
    timestamps = [ts for ts, _, _ in selected]
    assert timestamps == [f"20260101_0000{i:02d}" for i in range(3, 9)]


# ---------------------------------------------------------------------------
# row <-> sample
# ---------------------------------------------------------------------------


def test_row_to_sample_roundtrip():
    original = _sample(3, elapsed_sec=1.5, present_raw=2050, goal_raw=2040)
    row = original.to_csv_row()
    # csv.DictWriter가 실제로 만드는 문자열 형태를 흉내낸다.
    row = {k: ("" if v is None else str(v)) for k, v in row.items()}
    restored = row_to_sample(row)
    assert restored is not None
    assert restored.loop_index == 3
    assert restored.follower_present_raw == 2050
    assert restored.follower_goal_raw == 2040
    assert restored.leader_command_all_joints["wrist_roll.pos"] == pytest.approx(0.0)


def test_row_to_sample_missing_required_field_returns_none():
    row = _sample(0, elapsed_sec=0.0).to_csv_row()
    row = {k: ("" if v is None else str(v)) for k, v in row.items()}
    del row["loop_index"]
    assert row_to_sample(row) is None


def test_row_to_sample_garbage_value_returns_none():
    row = _sample(0, elapsed_sec=0.0).to_csv_row()
    row = {k: ("" if v is None else str(v)) for k, v in row.items()}
    row["elapsed_sec"] = "not_a_number"
    assert row_to_sample(row) is None


def test_load_run_samples_counts_malformed_rows(tmp_path):
    samples = [_sample(i, elapsed_sec=i * 0.02) for i in range(5)]
    csv_path, _ = _write_run(tmp_path, "20260101_000000", samples, csv_padding_rows=2)
    loaded, malformed = load_run_samples(csv_path)
    assert len(loaded) == 5
    assert malformed == 2


def test_load_run_report_missing_file_returns_none(tmp_path):
    assert load_run_report(tmp_path / "does_not_exist.json") is None


def test_load_run_report_malformed_json_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_run_report(path) is None


# ---------------------------------------------------------------------------
# run-level 품질 점검
# ---------------------------------------------------------------------------


def test_assess_run_quality_ok_for_normal_run(tmp_path):
    samples = [_sample(i, elapsed_sec=i * 0.02) for i in range(200)]
    csv_path, json_path = _write_run(tmp_path, "20260101_000000", samples, analysis_overrides={"actual_loop_hz": 59.0})
    bundle = load_run_bundle("20260101_000000", csv_path, json_path)
    assert bundle.quality["verdict"] == QUALITY_OK
    assert bundle.quality["reasons"] == []


def test_assess_run_quality_warns_on_abnormal_stopped_reason_and_low_sample_count(tmp_path):
    samples = [_sample(0, elapsed_sec=0.0)]  # sample_count=1, 매우 적음
    csv_path, json_path = _write_run(
        tmp_path, "20260101_000000", samples, stopped_reason="READ_FAILURE", analysis_overrides={"actual_loop_hz": 59.0}
    )
    bundle = load_run_bundle("20260101_000000", csv_path, json_path)
    assert bundle.quality["verdict"] == QUALITY_WARNING
    assert any("READ_FAILURE" in r for r in bundle.quality["reasons"])
    assert any("sample_count" in r for r in bundle.quality["reasons"])


def test_assess_run_quality_warns_on_out_of_range_hz_and_status_nonzero(tmp_path):
    samples = [_sample(i, elapsed_sec=i * 0.02, status=4) for i in range(200)]
    csv_path, json_path = _write_run(
        tmp_path, "20260101_000000", samples, analysis_overrides={"actual_loop_hz": 500.0, "status_ever_nonzero": True}
    )
    bundle = load_run_bundle("20260101_000000", csv_path, json_path)
    assert bundle.quality["verdict"] == QUALITY_WARNING
    assert any("actual_loop_hz" in r for r in bundle.quality["reasons"])
    assert any("Status" in r for r in bundle.quality["reasons"])


def test_assess_run_quality_does_not_auto_exclude_bad_runs(tmp_path):
    """이상 run이어도 자동 제외하지 않고 bundle 자체는 그대로 로드된다."""
    samples = [_sample(0, elapsed_sec=0.0)]
    csv_path, json_path = _write_run(tmp_path, "20260101_000000", samples, stopped_reason="READ_FAILURE")
    bundle = load_run_bundle("20260101_000000", csv_path, json_path)
    assert len(bundle.samples) == 1  # 제외되지 않고 그대로 있음


# ---------------------------------------------------------------------------
# joint 시계열 / frame delta / velocity / percentile
# ---------------------------------------------------------------------------


def test_compute_joint_value_series_extracts_only_present_values():
    samples = [_sample(0, elapsed_sec=0.0), _sample(1, elapsed_sec=0.02)]
    values, timestamps = compute_joint_value_series(samples, "wrist_roll", field_name="follower_observation_all_joints")
    assert len(values) == 2
    assert timestamps == [0.0, 0.02]


def test_compute_frame_deltas():
    assert compute_frame_deltas([1.0, 1.5, 1.2, 2.0]) == pytest.approx([0.5, 0.3, 0.8])
    assert compute_frame_deltas([1.0]) == []


def test_compute_velocities_flags_abnormal_dt():
    values = [0.0, 1.0, 2.0, 3.0]
    timestamps = [0.0, 0.02, 0.02, 1.0]  # index1->2: dt=0(비정상), index2->3: dt=0.98(비정상, > max)
    velocities, flagged = compute_velocities(values, timestamps, min_dt_s=1e-4, max_dt_s=0.5)
    assert flagged == 2
    assert len(velocities) == 1
    assert velocities[0] == pytest.approx(1.0 / 0.02)


def test_percentile_summary_basic():
    summary = percentile_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["min"] == 1.0
    assert summary["max"] == 5.0
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["p50"] == pytest.approx(3.0)
    assert summary["n"] == 5


def test_percentile_summary_empty_returns_none():
    assert percentile_summary([]) is None


# ---------------------------------------------------------------------------
# joint aggregate: wrist_roll(register) vs 기타 joint(command_vs_observation)
# ---------------------------------------------------------------------------


def test_compute_joint_aggregate_wrist_roll_uses_register_error(tmp_path):
    samples = [_sample(i, elapsed_sec=i * 0.02, present_raw=2023, goal_raw=2023 + 10) for i in range(20)]
    csv_path, json_path = _write_run(tmp_path, "20260101_000000", samples)
    bundle = load_run_bundle("20260101_000000", csv_path, json_path)

    agg = compute_joint_aggregate([bundle], "wrist_roll")
    assert agg["tracking_error_source"] == "goal_vs_present_register"
    expected_error = abs(_wrist_roll_deg(2023 + 10) - _wrist_roll_deg(2023))
    assert agg["aggregate"]["tracking_error"]["mean"] == pytest.approx(expected_error)


def test_compute_joint_aggregate_other_joint_uses_command_vs_observation(tmp_path):
    per_cmd = {f"{j}.pos": 0.0 for j in ALL_JOINTS}
    per_obs = {f"{j}.pos": 0.0 for j in ALL_JOINTS}
    per_cmd["shoulder_pan.pos"] = 10.0
    per_obs["shoulder_pan.pos"] = 9.0  # 1도 tracking error
    samples = [_sample(i, elapsed_sec=i * 0.02, per_joint_command=dict(per_cmd), per_joint_observation=dict(per_obs)) for i in range(20)]
    csv_path, json_path = _write_run(tmp_path, "20260101_000000", samples)
    bundle = load_run_bundle("20260101_000000", csv_path, json_path)

    agg = compute_joint_aggregate([bundle], "shoulder_pan")
    assert agg["tracking_error_source"] == "command_vs_observation"
    assert agg["aggregate"]["tracking_error"]["mean"] == pytest.approx(1.0)
    assert agg["aggregate"]["range"]["min"] == pytest.approx(9.0)
    assert agg["aggregate"]["range"]["max"] == pytest.approx(9.0)


def test_compute_joint_aggregate_pools_across_multiple_runs(tmp_path):
    samples1 = [_sample(i, elapsed_sec=i * 0.02, present_raw=2023) for i in range(20)]
    samples2 = [_sample(i, elapsed_sec=i * 0.02, present_raw=2100) for i in range(20)]
    csv1, json1 = _write_run(tmp_path, "20260101_000000", samples1)
    csv2, json2 = _write_run(tmp_path, "20260101_000100", samples2)
    b1 = load_run_bundle("20260101_000000", csv1, json1)
    b2 = load_run_bundle("20260101_000100", csv2, json2)

    agg = compute_joint_aggregate([b1, b2], "wrist_roll")
    assert len(agg["per_run"]) == 2
    # pooled range는 두 run의 값을 모두 포함해야 한다.
    assert agg["aggregate"]["range"]["min"] == pytest.approx(min(_wrist_roll_deg(2023), _wrist_roll_deg(2100)))
    assert agg["aggregate"]["range"]["max"] == pytest.approx(max(_wrist_roll_deg(2023), _wrist_roll_deg(2100)))


# ---------------------------------------------------------------------------
# latency aggregate
# ---------------------------------------------------------------------------


def test_compute_latency_aggregate_insufficient_when_no_run_has_valid_lag(tmp_path):
    samples = [_sample(i, elapsed_sec=i * 0.02) for i in range(20)]
    csv_path, json_path = _write_run(tmp_path, "20260101_000000", samples)
    bundle = load_run_bundle("20260101_000000", csv_path, json_path)
    result = compute_latency_aggregate([bundle])
    assert result["verdict"] == "INSUFFICIENT_DATA"


def test_compute_latency_aggregate_averages_available_runs(tmp_path):
    bundles = []
    for i, lag_ms in enumerate([40.0, 50.0, 60.0]):
        samples = [_sample(j, elapsed_sec=j * 0.02) for j in range(20)]
        csv_path, json_path = _write_run(
            tmp_path,
            f"20260101_00000{i}",
            samples,
            analysis_overrides={
                "command_to_actual_lag_estimate": "OK",
                "command_to_actual_lag_frames": 3,
                "command_to_actual_lag_ms_timestamp_based": lag_ms,
                "command_to_actual_lag_ms_frame_based": lag_ms,
            },
        )
        bundles.append(load_run_bundle(f"20260101_00000{i}", csv_path, json_path))

    result = compute_latency_aggregate(bundles)
    assert result["verdict"] == "AVAILABLE"
    assert result["lag_ms_median"] == pytest.approx(50.0)
    assert result["lag_ms_min"] == pytest.approx(40.0)
    assert result["lag_ms_max"] == pytest.approx(60.0)
    assert result["n_runs_with_valid_lag"] == 3


# ---------------------------------------------------------------------------
# deadband aggregate: 재사용 + 단일 run vs 여러 run 반복성 구분
# ---------------------------------------------------------------------------


def test_compute_deadband_aggregate_sums_buckets_across_runs(tmp_path):
    def _make_run(ts, *, error_ticks, response_raw_delta):
        # error가 고정 크기로 유지되도록 goal이 present를 계속 앞서가게 만든다.
        samples = []
        present = 2023
        for i in range(30):
            present_raw = present + i * response_raw_delta
            samples.append(_sample(i, elapsed_sec=i * 0.02, present_raw=present_raw, goal_raw=present_raw + error_ticks))
        csv_path, json_path = _write_run(tmp_path, ts, samples, follower_start_present_raw=present)
        return load_run_bundle(ts, csv_path, json_path)

    # 3 tick 오차에서 run1은 실제로 반응(같은 방향 이동)하지만, run2/run3은 반응하지 않는다.
    b1 = _make_run("20260101_000000", error_ticks=3, response_raw_delta=2)
    b2 = _make_run("20260101_000100", error_ticks=3, response_raw_delta=0)
    b3 = _make_run("20260101_000200", error_ticks=3, response_raw_delta=0)

    result = compute_deadband_aggregate([b1, b2, b3], lookahead_ms=200.0)
    assert result["verdict"] == "DEADBAND_AGGREGATE_AVAILABLE"
    bucket3 = next(b for b in result["buckets"] if b["abs_goal_present_error_ticks"] == 3)
    # 3개 run 중 1개에서만 반응이 있었다는 것을 per_run_response_fraction으로 구분할 수 있어야 한다.
    assert bucket3["runs_with_any_response"] == 1
    non_zero = [r for r in bucket3["per_run_response_fraction"] if (r["response_fraction"] or 0) > 0]
    assert len(non_zero) == 1
    assert non_zero[0]["timestamp"] == "20260101_000000"


def test_compute_deadband_aggregate_insufficient_when_no_samples(tmp_path):
    samples = [_sample(0, elapsed_sec=0.0)]
    csv_path, json_path = _write_run(tmp_path, "20260101_000000", samples)
    bundle = load_run_bundle("20260101_000000", csv_path, json_path)
    bundle.samples = []  # 강제로 빈 샘플
    result = compute_deadband_aggregate([bundle])
    assert result["verdict"] != "DEADBAND_AGGREGATE_AVAILABLE"


def test_classify_deadband_region_thresholds():
    assert classify_deadband_region(0.0) == NO_RESPONSE_REGION
    assert classify_deadband_region(0.09) == NO_RESPONSE_REGION
    assert classify_deadband_region(0.10) == TRANSITION_REGION
    assert classify_deadband_region(0.5) == TRANSITION_REGION
    assert classify_deadband_region(0.80) == TRANSITION_REGION
    assert classify_deadband_region(0.81) == HIGH_RESPONSE_REGION
    assert classify_deadband_region(1.0) == HIGH_RESPONSE_REGION
    assert classify_deadband_region(None) is None


# ---------------------------------------------------------------------------
# run-to-run stability / candidates
# ---------------------------------------------------------------------------


def test_compute_run_to_run_stability_reports_mean_std_cv(tmp_path):
    bundles = []
    for i, hz in enumerate([58.0, 59.0, 60.0]):
        samples = [_sample(j, elapsed_sec=j * 0.02) for j in range(20)]
        csv_path, json_path = _write_run(tmp_path, f"20260101_00000{i}", samples, analysis_overrides={"actual_loop_hz": hz})
        bundles.append(load_run_bundle(f"20260101_00000{i}", csv_path, json_path))

    joint_aggregates = {"wrist_roll": compute_joint_aggregate(bundles, "wrist_roll")}
    latency = compute_latency_aggregate(bundles)
    stability = compute_run_to_run_stability(bundles, joint_aggregates, latency)
    assert stability["actual_loop_hz"]["mean"] == pytest.approx(59.0)
    assert stability["actual_loop_hz"]["n"] == 3
    assert stability["actual_loop_hz"]["cv"] is not None


def test_compute_candidates_are_labeled_candidate_only(tmp_path):
    samples = [_sample(i, elapsed_sec=i * 0.02) for i in range(20)]
    csv_path, json_path = _write_run(tmp_path, "20260101_000000", samples)
    bundle = load_run_bundle("20260101_000000", csv_path, json_path)
    joint_aggregates = {"wrist_roll": compute_joint_aggregate([bundle], "wrist_roll")}
    deadband = compute_deadband_aggregate([bundle])
    latency = compute_latency_aggregate([bundle])
    candidates = compute_candidates(joint_aggregates, deadband, latency)
    assert candidates["label"] == "CANDIDATE_ONLY"
    assert candidates["joints"]["wrist_roll"]["label"] == "CANDIDATE_ONLY"


# ---------------------------------------------------------------------------
# end-to-end + markdown + 원본 파일 무변경
# ---------------------------------------------------------------------------


def test_build_aggregate_report_end_to_end_with_six_runs(tmp_path):
    bundles = []
    for i in range(6):
        samples = [_sample(j, elapsed_sec=j * 0.02, present_raw=2023 + i) for j in range(50)]
        csv_path, json_path = _write_run(tmp_path, f"20260101_0000{i:02d}", samples, analysis_overrides={"actual_loop_hz": 59.0})
        bundles.append(load_run_bundle(f"20260101_0000{i:02d}", csv_path, json_path))

    report = build_aggregate_report(bundles)
    assert report["run_count"] == 6
    assert set(report["joint_aggregates"].keys()) == set(ALL_JOINTS)
    assert report["direct_register_write_count"] == 0
    assert report["hardware_execution_count"] == 0

    md = render_markdown_report(report)
    assert "Instrumented Teleop 통합 분석" in md
    assert "CANDIDATE_ONLY" in md


def test_original_csv_and_json_unchanged_after_analysis(tmp_path):
    samples = [_sample(i, elapsed_sec=i * 0.02) for i in range(30)]
    csv_path, json_path = _write_run(tmp_path, "20260101_000000", samples)
    csv_before = csv_path.read_bytes()
    json_before = json_path.read_bytes()

    bundle = load_run_bundle("20260101_000000", csv_path, json_path)
    build_aggregate_report([bundle])

    assert csv_path.read_bytes() == csv_before
    assert json_path.read_bytes() == json_before


def test_aggregate_module_importable_without_lerobot():
    # 시스템 python(lerobot 미설치 환경)에서도 aggregate 모듈이 import되는지 확인한다 -
    # 완전 offline(하드웨어/lerobot 무관) 모듈이라는 주장을 실제로 검증한다.
    system_python = Path("/usr/bin/python3")
    if not system_python.is_file():
        pytest.skip("/usr/bin/python3가 없어 건너뜁니다.")
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [str(system_python), "-c", "import hardware.diagnostics.instrumented_teleop_aggregate"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
