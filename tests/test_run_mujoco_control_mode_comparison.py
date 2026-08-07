"""scripts/run_mujoco_control_mode_comparison.py 테스트.

이 스크립트 자체가 hardware-free 벤치마크이므로, 이 테스트도 leader/follower/서버 없이
mujoco만으로 돌아간다. 실행 시간을 짧게 유지하기 위해 num_steps을 작게 준다.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_mujoco_control_mode_comparison.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_mujoco_control_mode_comparison", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclass(전방참조 문자열 annotation) 해석이 sys.modules에서 자기 모듈을 찾으므로,
    # exec 이전에 등록해 둬야 한다 (그렇지 않으면 PassResult 등의 dataclass 정의가 깨진다).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_script_module()


def test_build_synthetic_trajectory_is_deterministic_and_covers_all_joints(mod):
    t1 = mod.build_synthetic_trajectory(50, wrist_roll_dither_deg=0.2)
    t2 = mod.build_synthetic_trajectory(50, wrist_roll_dither_deg=0.2)
    assert t1 == t2
    assert len(t1) == 50
    for step in t1:
        assert set(step) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}


def test_synthetic_trajectory_shoulder_pan_jump_and_wrist_roll_phases(mod):
    trajectory = mod.build_synthetic_trajectory(80, wrist_roll_dither_deg=0.2)
    assert trajectory[0]["shoulder_pan"] == 0.0
    assert trajectory[mod.SHOULDER_PAN_JUMP_STEP]["shoulder_pan"] == mod.SHOULDER_PAN_JUMP_TARGET_DEG
    # wrist_roll deadband 구간: 진폭이 dither_deg를 넘지 않는다.
    for step in trajectory[: mod.WRIST_ROLL_DEADBAND_PHASE_STEPS]:
        assert abs(step["wrist_roll"]) <= 0.2
    # 이후 큰 점프 (transition region 이상).
    assert trajectory[mod.WRIST_ROLL_DEADBAND_PHASE_STEPS]["wrist_roll"] > 1.0


def test_gripper_values_stay_within_percent_semantics(mod):
    trajectory = mod.build_synthetic_trajectory(30, wrist_roll_dither_deg=0.2)
    for step in trajectory:
        assert 0.0 <= step["gripper"] <= 100.0


def test_main_runs_end_to_end_and_writes_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mujoco_control_mode_comparison.py",
            "--num-steps",
            "40",
            "--control-hz",
            "30",
            "--output-dir",
            str(tmp_path),
        ],
    )
    mod = _load_script_module()
    exit_code = mod.main()
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "baseline" in out
    assert "realistic" in out

    comparison_files = list(tmp_path.glob("comparison_*.json"))
    assert len(comparison_files) == 1
    payload = json.loads(comparison_files[0].read_text(encoding="utf-8"))
    assert payload["num_steps"] == 40
    assert set(payload["comparison"]["joints"]) == {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    }
    assert "shoulder_pan_step_jump" in payload["comparison"]
    assert "wrist_roll_deadband_phase" in payload["comparison"]

    diag_json = list(tmp_path.glob("realistic_diagnostics_*.json"))
    diag_csv = list(tmp_path.glob("realistic_diagnostics_*.csv"))
    assert len(diag_json) == 1 and len(diag_csv) == 1


def test_realistic_pass_rate_limits_large_jump_more_than_baseline(tmp_path, monkeypatch):
    """섹션 12 핵심 목적: realistic 쪽 measurement가 profile candidate 방향으로 이동하는지."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_mujoco_control_mode_comparison.py", "--num-steps", "40", "--output-dir", str(tmp_path)],
    )
    mod = _load_script_module()
    mod.main()

    comparison_files = list(tmp_path.glob("comparison_*.json"))
    payload = json.loads(comparison_files[0].read_text(encoding="utf-8"))
    shoulder_pan = payload["comparison"]["joints"]["shoulder_pan"]
    # realistic 쪽 frame_delta max가 baseline보다 훨씬 작아야 한다 (soft limit 적용).
    assert shoulder_pan["realistic"]["frame_delta"]["max"] < shoulder_pan["baseline"]["frame_delta"]["max"]


def test_disable_flags_are_wired_through(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mujoco_control_mode_comparison.py",
            "--num-steps",
            "30",
            "--output-dir",
            str(tmp_path),
            "--disable-latency",
            "--disable-deadband",
            "--disable-rate-limit",
            "--disable-historical-range-diagnostic",
        ],
    )
    mod = _load_script_module()
    exit_code = mod.main()
    assert exit_code == 0

    import csv

    diag_csv = list(tmp_path.glob("realistic_diagnostics_*.csv"))[0]
    with diag_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "diagnostics CSV가 비어 있습니다"
    # 4가지 특성을 모두 끈 realistic pass는 baseline과 동일한 identity 결과여야 한다 -
    # rate_limited/deadband_applied가 어떤 행에도 True로 찍히지 않아야 한다.
    for row in rows:
        assert row["rate_limited"] == "False"
        assert row["deadband_applied"] == "False"


def test_invalid_profile_path_returns_error_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mujoco_control_mode_comparison.py",
            "--num-steps",
            "10",
            "--output-dir",
            str(tmp_path),
            "--profile",
            str(tmp_path / "missing.json"),
        ],
    )
    mod = _load_script_module()
    exit_code = mod.main()
    assert exit_code == 1
    assert "오류" in capsys.readouterr().out
