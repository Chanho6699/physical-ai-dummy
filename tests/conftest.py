from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

REAL_DATASET_ROOT = PROJECT_ROOT / "data" / "so101_cube_xy_train_v1"


def build_synthetic_dataset(
    root: Path,
    *,
    num_frames: int = 10,
    fps: int = 30,
    action_names: list[str] | None = None,
    action_values: np.ndarray | None = None,
    frame_index_override: np.ndarray | None = None,
    timestamp_override: np.ndarray | None = None,
    task: str = "테스트용 합성 에피소드.",
) -> Path:
    """dataset_loader가 요구하는 최소 v3.0 형식의 합성 데이터셋을 만든다.

    실제 LeRobot 데이터셋 구조(meta/info.json, meta/episodes/*, data/*)를 그대로 모사하되,
    비디오/카메라는 만들지 않는다 (action replay는 이미지에 의존하지 않음).
    """
    names = action_names or ACTION_NAMES
    dim = len(names)
    action = action_values if action_values is not None else np.zeros((num_frames, dim), dtype=np.float32)

    meta_dir = root / "meta"
    (meta_dir / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "features": {
            "action": {"dtype": "float32", "names": names, "shape": [dim]},
            "observation.state": {"dtype": "float32", "names": names, "shape": [dim]},
        },
        "total_episodes": 1,
        "total_frames": num_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "robot_type": "so_follower",
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    frame_index = frame_index_override if frame_index_override is not None else np.arange(num_frames, dtype=np.int64)
    timestamp = (
        timestamp_override
        if timestamp_override is not None
        else (np.arange(num_frames, dtype=np.float64) / fps)
    )

    data_df = pd.DataFrame(
        {
            "action": list(action.astype(np.float32)),
            "observation.state": list(action.astype(np.float32)),
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": np.zeros(num_frames, dtype=np.int64),
            "index": np.arange(num_frames, dtype=np.int64),
            "task_index": np.zeros(num_frames, dtype=np.int64),
        }
    )
    data_df.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")

    episodes_df = pd.DataFrame(
        {
            "episode_index": [0],
            "tasks": [[task]],
            "length": [num_frames],
            "data/chunk_index": [0],
            "data/file_index": [0],
        }
    )
    episodes_df.to_parquet(meta_dir / "episodes" / "chunk-000" / "file-000.parquet")

    return root


@pytest.fixture
def synthetic_dataset(tmp_path):
    return build_synthetic_dataset(tmp_path / "synthetic_ds")


@pytest.fixture
def real_dataset_root() -> Path:
    if not REAL_DATASET_ROOT.is_dir():
        pytest.skip(f"실제 데이터셋을 찾을 수 없습니다: {REAL_DATASET_ROOT}")
    return REAL_DATASET_ROOT
