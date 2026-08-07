from __future__ import annotations

import json
import socket
import sys
import threading
import time
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


# ---------------------------------------------------------------------------
# Shadow Mode: 진짜 소켓으로 왕복하는 Fake VLA 서버 (uvicorn 백그라운드 스레드).
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live_fake_vla_server():
    """(base_url, FakePolicyRunner) - 실제 loopback 소켓으로 통신하는 Fake VLA 서버.

    TestClient(ASGI in-process transport)가 아니라 진짜 uvicorn + 소켓을 띄운다 -
    ``VLAHttpClient``가 실제 HTTP 왕복 지연/timeout/연결 실패 경로까지 검증할 수 있게
    하기 위함이다 (섹션 3/20 - "통신은 Fake VLA와 실제 Desktop server를 구분해서
    테스트 가능하게").
    """
    import uvicorn

    from runtime.desktop.vla_server import FakePolicyRunner, create_app

    port = _free_port()
    policy_runner = FakePolicyRunner()
    app = create_app(policy_runner=policy_runner)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if getattr(server, "started", False):
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("live_fake_vla_server가 제한 시간 내에 기동하지 못했습니다.")

    try:
        yield f"http://127.0.0.1:{port}", policy_runner
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
