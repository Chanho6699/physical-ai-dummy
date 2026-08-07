"""runtime/laptop/observation_builder.py 테스트 - 학습 데이터셋 schema와의 매핑 검증."""

from __future__ import annotations

import math
import time

import numpy as np

from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY, JOINT_ORDER
from runtime.laptop.camera_source import CameraFrame
from runtime.laptop.follower_state_source import FollowerStateSnapshot
from runtime.laptop.observation_builder import build_observation


def _state(**overrides) -> FollowerStateSnapshot:
    positions = {joint: float(i) for i, joint in enumerate(JOINT_ORDER)}
    positions.update(overrides)
    return FollowerStateSnapshot(positions_deg=positions, read_at_monotonic=0.0, read_at_wall=time.time())


def _frames(**overrides) -> dict[str, CameraFrame]:
    good = np.zeros((480, 640, 3), dtype=np.uint8)
    frames = {
        CAMERA_WORKSPACE_KEY: CameraFrame(image_rgb=good, captured_at_wall=time.time(), width=640, height=480),
        CAMERA_WRIST_KEY: CameraFrame(image_rgb=good, captured_at_wall=time.time(), width=640, height=480),
    }
    frames.update(overrides)
    return frames


def test_valid_observation_is_schema_valid() -> None:
    built = build_observation(state_snapshot=_state(), camera_frames=_frames())
    assert built.schema_valid is True
    assert built.fixed_camera_valid is True
    assert built.wrist_camera_valid is True
    assert built.state_valid is True
    assert set(built.state) == set(JOINT_ORDER)
    assert set(built.images) == {CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY}


def test_missing_wrist_camera_invalidates_schema() -> None:
    frames = _frames()
    del frames[CAMERA_WRIST_KEY]
    built = build_observation(state_snapshot=_state(), camera_frames=frames)
    assert built.schema_valid is False
    assert built.wrist_camera_valid is False
    assert built.fixed_camera_valid is True
    assert any("wrist" in r for r in built.invalid_reasons)


def test_wrong_channel_count_invalidates_camera() -> None:
    frames = _frames()
    frames[CAMERA_WORKSPACE_KEY] = CameraFrame(
        image_rgb=np.zeros((480, 640, 1), dtype=np.uint8), captured_at_wall=time.time(), width=640, height=480
    )
    built = build_observation(state_snapshot=_state(), camera_frames=frames)
    assert built.schema_valid is False
    assert built.fixed_camera_valid is False


def test_nan_state_invalidates_schema() -> None:
    built = build_observation(state_snapshot=_state(gripper=math.nan), camera_frames=_frames())
    assert built.schema_valid is False
    assert built.state_valid is False
    assert built.state == {}


def test_wrong_dtype_image_invalidates_camera() -> None:
    frames = _frames()
    frames[CAMERA_WRIST_KEY] = CameraFrame(
        image_rgb=np.zeros((480, 640, 3), dtype=np.float32), captured_at_wall=time.time(), width=640, height=480
    )
    built = build_observation(state_snapshot=_state(), camera_frames=frames)
    assert built.wrist_camera_valid is False
    assert built.schema_valid is False
