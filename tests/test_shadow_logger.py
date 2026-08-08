"""runtime/laptop/shadow_logger.py 테스트."""

from __future__ import annotations

import json

import numpy as np
import pytest

from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY
from runtime.laptop.shadow_logger import (
    RESULT_SHADOW_PASS,
    build_report,
    make_session_id,
    resolve_camera_frame_paths,
    resolve_report_path,
    save_camera_frames,
    write_report,
)


def _minimal_report(**overrides) -> dict:
    kwargs = dict(
        task="pick up the cube",
        backend="realistic_mujoco",
        observation={"schema_valid": True},
        communication={"health_ok": True},
        vla={"raw_action": {}},
        adapter={"mapped_action": {}},
        safety={"decision": "ACCEPT"},
        mujoco={"initial_state": {}},
        validation={"passed": True},
        real_follower_write_count=0,
        result=RESULT_SHADOW_PASS,
        result_reasons=[],
    )
    kwargs.update(overrides)
    return build_report(**kwargs)


def test_build_report_has_required_top_level_keys() -> None:
    report = _minimal_report()
    for key in ("mode", "backend", "real_robot_write_enabled", "task", "observation", "communication", "vla",
                "adapter", "safety", "mujoco", "hardware", "result"):
        assert key in report
    assert report["mode"] == "SHADOW"
    assert report["real_robot_write_enabled"] is False
    assert report["hardware"]["real_follower_write_count"] == 0


def test_build_report_rejects_nonzero_write_count() -> None:
    with pytest.raises(AssertionError):
        _minimal_report(real_follower_write_count=1)


def test_write_report_creates_json_file(tmp_path) -> None:
    report = _minimal_report()
    session_id = make_session_id()
    path = write_report(report, reports_dir=tmp_path, session_id=session_id)
    assert path == resolve_report_path(tmp_path, session_id)
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["result"] == RESULT_SHADOW_PASS


def test_session_id_format() -> None:
    sid = make_session_id()
    assert len(sid) == 15  # YYYYMMDD_HHMMSS
    assert sid[8] == "_"


# ---------------------------------------------------------------------------
# checkpoint/evaluation_mode/scene_metadata (요구사항 4/7번) - 기본값은 None, 있으면 그대로 담긴다.
# ---------------------------------------------------------------------------


def test_build_report_checkpoint_and_eval_mode_default_to_none() -> None:
    report = _minimal_report()
    assert report["checkpoint"] is None
    assert report["evaluation_mode"] is None
    assert report["scene_metadata"] is None


def test_build_report_carries_checkpoint_metadata() -> None:
    checkpoint = {
        "checkpoint_path": "/abs/outputs/grid35/x/checkpoints/010000/pretrained_model",
        "checkpoint_step": 10000,
        "train_dataset_provenance": {"recorded_train_dataset_root": "/x/data/so101_cube_xy_grid35_v1"},
    }
    report = _minimal_report(checkpoint=checkpoint, evaluation_mode="midpoint-shadow", scene_metadata={"label": "T03"})
    assert report["checkpoint"]["checkpoint_step"] == 10000
    assert report["evaluation_mode"] == "midpoint-shadow"
    assert report["scene_metadata"]["label"] == "T03"


# ---------------------------------------------------------------------------
# save_camera_frames / resolve_camera_frame_paths (요구사항 5번)
# ---------------------------------------------------------------------------


def _fake_image() -> np.ndarray:
    return (np.random.rand(48, 64, 3) * 255).astype(np.uint8)


def test_save_camera_frames_writes_both_images(tmp_path) -> None:
    session_id = "20260101_000000"
    images = {CAMERA_WORKSPACE_KEY: _fake_image(), CAMERA_WRIST_KEY: _fake_image()}
    saved = save_camera_frames(images, reports_dir=tmp_path, session_id=session_id)
    assert set(saved) == {"workspace_path", "wrist_path"}
    for path_str in saved.values():
        from pathlib import Path

        p = Path(path_str)
        assert p.is_file()
        assert p.stat().st_size > 0


def test_save_camera_frames_empty_images_returns_empty_dict(tmp_path) -> None:
    assert save_camera_frames({}, reports_dir=tmp_path, session_id="20260101_000000") == {}


def test_save_camera_frames_partial_images_saves_only_available(tmp_path) -> None:
    images = {CAMERA_WORKSPACE_KEY: _fake_image()}
    saved = save_camera_frames(images, reports_dir=tmp_path, session_id="20260101_000000")
    assert set(saved) == {"workspace_path"}


def test_resolve_camera_frame_paths_creates_per_session_directory(tmp_path) -> None:
    paths = resolve_camera_frame_paths(tmp_path, "20260101_000000")
    assert paths["workspace"].parent == paths["wrist"].parent
    assert paths["workspace"].parent.is_dir()
    assert paths["workspace"].parent.name == "20260101_000000"
