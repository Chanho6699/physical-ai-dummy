"""simulation/realism/so101_control_profile.py 테스트.

실제 candidate 파일(``configs/generated/so101_control_profile_candidate_v1.json``)을 그대로
로딩해 검증하고, 계약을 어긴 변형 JSON들이 전부 :class:`ControlProfileError`로 거부되는지
확인한다. 하드웨어/시뮬레이터 의존성 없음 (순수 JSON 로딩 테스트).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from simulation.realism.so101_control_profile import (
    DEFAULT_PROFILE_PATH,
    JOINT_NAMES,
    ControlProfileError,
    load_control_profile,
)

REAL_PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "generated" / "so101_control_profile_candidate_v1.json"


def _load_real_raw() -> dict:
    return json.loads(REAL_PROFILE_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 실제 candidate 파일 로딩
# ---------------------------------------------------------------------------


def test_default_profile_path_matches_real_generated_file():
    assert DEFAULT_PROFILE_PATH == REAL_PROFILE_PATH


def test_loads_real_candidate_file_without_error():
    profile = load_control_profile(REAL_PROFILE_PATH)
    assert profile.status == "CANDIDATE_ONLY"
    assert profile.apply_automatically is False
    assert profile.source == "instrumented_teleop_6runs"
    assert profile.run_count == 6
    assert set(profile.joints) == set(JOINT_NAMES)


def test_gripper_unit_is_percent_not_degree():
    profile = load_control_profile(REAL_PROFILE_PATH)
    assert profile.joints["gripper"].unit == "percent_0_100"
    for joint in JOINT_NAMES:
        if joint == "gripper":
            continue
        assert profile.joints[joint].unit == "degree"


def test_wrist_roll_deadband_no_response_upper_matches_5_ticks():
    profile = load_control_profile(REAL_PROFILE_PATH)
    deadband = profile.wrist_roll_deadband
    assert deadband.no_response_upper_tick == 5
    assert deadband.transition_start_tick == 6
    assert deadband.no_response_upper_deg == pytest.approx(0.43956, abs=1e-4)


def test_timing_latency_median_matches_source_json():
    profile = load_control_profile(REAL_PROFILE_PATH)
    assert profile.timing.latency_median_ms == pytest.approx(92.89707204179743)
    assert profile.timing.valid_runs == "4/6"


def test_joint_frame_delta_and_velocity_soft_limits_are_p99_candidates():
    profile = load_control_profile(REAL_PROFILE_PATH)
    shoulder_pan = profile.joints["shoulder_pan"]
    assert shoulder_pan.frame_delta_soft_limit == pytest.approx(1.4065934065934087)
    assert shoulder_pan.velocity_soft_limit == pytest.approx(83.7951516343046)
    assert shoulder_pan.historical_range == pytest.approx((-89.71428571428571, 81.1041758241757))


# ---------------------------------------------------------------------------
# 계약 위반 -> ControlProfileError
# ---------------------------------------------------------------------------


def test_missing_file_raises():
    with pytest.raises(ControlProfileError):
        load_control_profile("/nonexistent/path/profile.json")


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ControlProfileError):
        load_control_profile(path)


def test_non_object_top_level_raises(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ControlProfileError):
        load_control_profile(path)


def test_status_not_candidate_only_raises(tmp_path):
    raw = _load_real_raw()
    raw["status"] = "PRODUCTION"
    with pytest.raises(ControlProfileError, match="status"):
        load_control_profile(_write(tmp_path, raw))


def test_apply_automatically_true_raises(tmp_path):
    raw = _load_real_raw()
    raw["apply_automatically"] = True
    with pytest.raises(ControlProfileError, match="apply_automatically"):
        load_control_profile(_write(tmp_path, raw))


def test_missing_source_raises(tmp_path):
    raw = _load_real_raw()
    del raw["source"]
    with pytest.raises(ControlProfileError, match="source"):
        load_control_profile(_write(tmp_path, raw))


def test_missing_run_count_raises(tmp_path):
    raw = _load_real_raw()
    del raw["run_count"]
    with pytest.raises(ControlProfileError, match="run_count"):
        load_control_profile(_write(tmp_path, raw))


def test_missing_joint_data_raises(tmp_path):
    raw = _load_real_raw()
    del raw["joints"]["wrist_roll"]
    with pytest.raises(ControlProfileError, match="wrist_roll"):
        load_control_profile(_write(tmp_path, raw))


def test_gripper_unit_mislabeled_as_degree_raises(tmp_path):
    raw = _load_real_raw()
    raw["joints"]["gripper"]["unit"] = "degree"
    with pytest.raises(ControlProfileError, match="gripper"):
        load_control_profile(_write(tmp_path, raw))


def test_arm_joint_unit_mislabeled_as_percent_raises(tmp_path):
    raw = _load_real_raw()
    raw["joints"]["shoulder_pan"]["unit"] = "percent_0_100"
    with pytest.raises(ControlProfileError, match="shoulder_pan"):
        load_control_profile(_write(tmp_path, raw))


def test_missing_wrist_roll_deadband_block_raises(tmp_path):
    raw = _load_real_raw()
    del raw["wrist_roll_deadband_analysis"]
    with pytest.raises(ControlProfileError, match="wrist_roll_deadband_analysis"):
        load_control_profile(_write(tmp_path, raw))


def test_missing_timing_block_raises(tmp_path):
    raw = _load_real_raw()
    del raw["timing"]
    with pytest.raises(ControlProfileError, match="timing"):
        load_control_profile(_write(tmp_path, raw))


def test_deep_copy_of_real_raw_is_still_valid(tmp_path):
    """위 negative 테스트들이 원본을 변형해도 서로 영향을 주지 않는지 확인하는 대조군."""
    raw = copy.deepcopy(_load_real_raw())
    profile = load_control_profile(_write(tmp_path, raw))
    assert profile.status == "CANDIDATE_ONLY"
