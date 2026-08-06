"""hardware/state_server/calibration_loader.py 단위 테스트.

하드웨어/LeRobot 없이 순수 JSON 파싱 로직만 검증한다 (``to_motor_calibration_map``만
lerobot을 지연 import하므로 이 파일에서는 사용하지 않는다).
"""

from __future__ import annotations

import json

import pytest

from hardware.state_server.calibration_loader import (
    JOINT_NAMES,
    CalibrationLoadError,
    load_calibration_file,
    to_public_dict,
)

VALID_CALIBRATION = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2021, "range_min": 1571, "range_max": 2640},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1312, "range_min": 845, "range_max": 3092},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 1670, "range_min": 916, "range_max": 3084},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -1419, "range_min": 1019, "range_max": 3225},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": -1426, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 1952, "range_min": 2027, "range_max": 3436},
}


def _write(tmp_path, data) -> str:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_load_valid_calibration_file(tmp_path):
    path = _write(tmp_path, VALID_CALIBRATION)
    entries = load_calibration_file(path)

    assert set(entries) == set(JOINT_NAMES)
    assert entries["wrist_flex"].homing_offset == -1419
    assert entries["wrist_flex"].range_min == 1019
    assert entries["wrist_flex"].range_max == 3225
    assert entries["wrist_flex"].id == 4


def test_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(CalibrationLoadError, match="찾을 수 없습니다"):
        load_calibration_file(missing)


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="파싱 실패"):
        load_calibration_file(path)


def test_missing_joint_raises(tmp_path):
    data = dict(VALID_CALIBRATION)
    del data["gripper"]
    path = _write(tmp_path, data)
    with pytest.raises(CalibrationLoadError, match="gripper"):
        load_calibration_file(path)


def test_non_object_top_level_raises(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="object"):
        load_calibration_file(path)


def test_equal_range_min_max_raises(tmp_path):
    data = json.loads(json.dumps(VALID_CALIBRATION))  # deep copy
    data["wrist_flex"]["range_min"] = 100
    data["wrist_flex"]["range_max"] = 100
    path = _write(tmp_path, data)
    with pytest.raises(CalibrationLoadError, match="range_min"):
        load_calibration_file(path)


def test_non_numeric_field_raises(tmp_path):
    data = json.loads(json.dumps(VALID_CALIBRATION))
    data["wrist_flex"]["homing_offset"] = "not-a-number"
    path = _write(tmp_path, data)
    with pytest.raises(CalibrationLoadError, match="wrist_flex"):
        load_calibration_file(path)


def test_default_drive_mode_when_absent(tmp_path):
    data = json.loads(json.dumps(VALID_CALIBRATION))
    del data["gripper"]["drive_mode"]
    path = _write(tmp_path, data)
    entries = load_calibration_file(path)
    assert entries["gripper"].drive_mode == 0


def test_to_public_dict_excludes_id_and_drive_mode(tmp_path):
    path = _write(tmp_path, VALID_CALIBRATION)
    entries = load_calibration_file(path)
    public = to_public_dict(entries)

    assert set(public) == set(JOINT_NAMES)
    assert public["wrist_flex"] == {"homing_offset": -1419, "range_min": 1019, "range_max": 3225}
    assert "id" not in public["wrist_flex"]
    assert "drive_mode" not in public["wrist_flex"]
