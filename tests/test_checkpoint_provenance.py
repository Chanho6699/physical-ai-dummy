"""runtime/common/checkpoint_provenance.py 테스트 - lerobot/torch 없이도 순수 파일 기반."""

from __future__ import annotations

import json
from pathlib import Path

from runtime.common.checkpoint_provenance import (
    check_train_dataset_provenance,
    infer_checkpoint_step,
    read_checkpoint_train_dataset_root,
    validate_checkpoint_camera_mapping,
)

EXPECTED_MAP = {
    "observation.images.workspace": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


# ---------------------------------------------------------------------------
# infer_checkpoint_step
# ---------------------------------------------------------------------------


def test_infer_checkpoint_step_pretrained_model_dir() -> None:
    assert infer_checkpoint_step(Path("outputs/grid35/x/checkpoints/007500/pretrained_model")) == 7500


def test_infer_checkpoint_step_bare_numeric_dir() -> None:
    assert infer_checkpoint_step(Path("checkpoints/2500")) == 2500


def test_infer_checkpoint_step_non_numeric_is_none() -> None:
    assert infer_checkpoint_step(Path("checkpoints/last/pretrained_model")) is None


# ---------------------------------------------------------------------------
# read_checkpoint_train_dataset_root
# ---------------------------------------------------------------------------


def test_read_checkpoint_train_dataset_root_reads_dataset_root(tmp_path: Path) -> None:
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "train_config.json").write_text(json.dumps({"dataset": {"root": "/x/data/so101_cube_xy_grid35_v1"}}))
    root, error = read_checkpoint_train_dataset_root(ckpt)
    assert root == "/x/data/so101_cube_xy_grid35_v1"
    assert error is None


def test_read_checkpoint_train_dataset_root_missing_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    root, error = read_checkpoint_train_dataset_root(ckpt)
    assert root is None
    assert "train_config.json" in error


def test_read_checkpoint_train_dataset_root_missing_dataset_root_key(tmp_path: Path) -> None:
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "train_config.json").write_text(json.dumps({"dataset": {}}))
    root, error = read_checkpoint_train_dataset_root(ckpt)
    assert root is None
    assert "dataset.root" in error


def test_read_checkpoint_train_dataset_root_malformed_json(tmp_path: Path) -> None:
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "train_config.json").write_text("{not json")
    root, error = read_checkpoint_train_dataset_root(ckpt)
    assert root is None
    assert error is not None


# ---------------------------------------------------------------------------
# check_train_dataset_provenance
# ---------------------------------------------------------------------------


def test_check_train_dataset_provenance_matches_by_basename(tmp_path: Path) -> None:
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "train_config.json").write_text(json.dumps({"dataset": {"root": "/machine_a/data/so101_cube_xy_grid35_v1"}}))
    warning = check_train_dataset_provenance(ckpt, Path("/machine_b/somewhere/data/so101_cube_xy_grid35_v1"))
    assert warning is None


def test_check_train_dataset_provenance_flags_legacy_dataset_mismatch(tmp_path: Path) -> None:
    """과거 20ep pilot dataset으로 학습된 checkpoint를 새 실험이라고 착각하는 것을 막는다."""
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "train_config.json").write_text(json.dumps({"dataset": {"root": "/x/data/so101_cube_train_v6"}}))
    warning = check_train_dataset_provenance(ckpt, Path("/x/data/so101_cube_xy_grid35_v1"))
    assert warning is not None
    assert "so101_cube_train_v6" in warning
    assert "so101_cube_xy_grid35_v1" in warning


# ---------------------------------------------------------------------------
# validate_checkpoint_camera_mapping
# ---------------------------------------------------------------------------


def test_validate_checkpoint_camera_mapping_ok(tmp_path: Path) -> None:
    preprocessor = {
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": dict(EXPECTED_MAP)},
            }
        ]
    }
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(preprocessor))
    assert validate_checkpoint_camera_mapping(tmp_path, EXPECTED_MAP) == []


def test_validate_checkpoint_camera_mapping_mismatch(tmp_path: Path) -> None:
    preprocessor = {
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {"observation.images.workspace": "observation.images.camera3"}},
            }
        ]
    }
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(preprocessor))
    warnings = validate_checkpoint_camera_mapping(tmp_path, EXPECTED_MAP)
    assert len(warnings) >= 1
    assert any("카메라 rename 불일치" in w for w in warnings)


def test_validate_checkpoint_camera_mapping_missing_file(tmp_path: Path) -> None:
    warnings = validate_checkpoint_camera_mapping(tmp_path, EXPECTED_MAP)
    assert len(warnings) == 1
    assert "policy_preprocessor.json" in warnings[0]
