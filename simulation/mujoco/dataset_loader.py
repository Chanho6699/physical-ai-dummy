"""LeRobot v3 데이터셋(action/state/timestamp) 로딩.

이 모듈은 `data/so101_cube_xy_train_v1` 등 codebase_version "v3.0" 형식의 LeRobot
데이터셋을 대상으로 한다 (meta/info.json, meta/episodes/chunk-*/file-*.parquet,
data/chunk-*/file-*.parquet). 이미지는 읽지 않는다 (action replay에 필요하지 않고,
video 디코딩 의존성을 추가하지 않기 위함).

기존 데이터셋 파일은 읽기 전용으로만 다루며 절대 수정하지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class DatasetLoadError(RuntimeError):
    """데이터셋 경로 또는 필수 파일을 찾을 수 없거나 형식이 잘못된 경우."""

    def __init__(self, dataset_root: Path, message: str) -> None:
        self.dataset_root = dataset_root
        super().__init__(message)


class InvalidEpisodeIndexError(RuntimeError):
    """요청한 episode_index가 데이터셋에 존재하지 않는 경우."""

    def __init__(self, requested: int, total_episodes: int) -> None:
        self.requested = requested
        self.total_episodes = total_episodes
        super().__init__(
            f"존재하지 않는 에피소드입니다. 요청={requested}, 허용 범위=0~{total_episodes - 1}"
        )


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    fps: int
    total_episodes: int
    total_frames: int
    action_names: tuple[str, ...]  # 예: ("shoulder_pan.pos", ...), info.json 원문 그대로
    action_dim: int
    state_names: tuple[str, ...]
    state_dim: int
    robot_type: str
    codebase_version: str
    data_path_template: str
    camera_keys: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class EpisodeData:
    episode_index: int
    length: int
    task: str
    fps: int
    joint_names: tuple[str, ...]  # dataset_info.action_names와 동일 (참조용)
    action: np.ndarray  # shape (T, D), dtype float64
    state: np.ndarray  # shape (T, D), dtype float64
    timestamp: np.ndarray  # shape (T,), dtype float64
    frame_index: np.ndarray  # shape (T,), dtype int64


def _require_dir(dataset_root: Path) -> None:
    if not dataset_root.is_dir():
        raise DatasetLoadError(dataset_root, f"데이터셋 경로를 찾을 수 없습니다: {dataset_root}")


def _load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise DatasetLoadError(dataset_root, f"필수 metadata 파일이 없습니다: {info_path}")
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(dataset_root, f"meta/info.json 파싱 실패: {exc}") from exc


def load_dataset_info(dataset_root: str | Path) -> DatasetInfo:
    """meta/info.json을 읽어 DatasetInfo를 만든다. action 차원을 6으로 하드코딩하지 않는다."""
    root = Path(dataset_root).expanduser().resolve()
    _require_dir(root)
    info = _load_info(root)

    features = info.get("features") or {}
    action_feature = features.get("action")
    state_feature = features.get("observation.state")
    if action_feature is None:
        raise DatasetLoadError(root, "meta/info.json에 'action' feature가 없습니다.")

    action_names = tuple(action_feature.get("names") or [])
    action_shape = action_feature.get("shape") or []
    action_dim = int(action_shape[0]) if action_shape else len(action_names)
    if not action_names or action_dim <= 0:
        raise DatasetLoadError(root, "meta/info.json의 action feature 이름/차원을 확인할 수 없습니다.")
    if len(action_names) != action_dim:
        raise DatasetLoadError(
            root,
            f"action feature 이름 개수({len(action_names)})와 선언된 차원({action_dim})이 다릅니다.",
        )

    state_names = tuple((state_feature or {}).get("names") or [])
    state_shape = (state_feature or {}).get("shape") or []
    state_dim = int(state_shape[0]) if state_shape else len(state_names)

    camera_keys = tuple(
        name.removeprefix("observation.images.")
        for name in features
        if name.startswith("observation.images.")
    )

    try:
        return DatasetInfo(
            root=root,
            fps=int(info["fps"]),
            total_episodes=int(info["total_episodes"]),
            total_frames=int(info["total_frames"]),
            action_names=action_names,
            action_dim=action_dim,
            state_names=state_names,
            state_dim=state_dim,
            robot_type=str(info.get("robot_type", "unknown")),
            codebase_version=str(info.get("codebase_version", "unknown")),
            data_path_template=str(info["data_path"]),
            camera_keys=camera_keys,
        )
    except KeyError as exc:
        raise DatasetLoadError(root, f"meta/info.json에 필수 키가 없습니다: {exc}") from exc


def _load_episode_meta_row(dataset_root: Path, episode_index: int, total_episodes: int) -> pd.Series:
    episode_files = sorted(dataset_root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not episode_files:
        raise DatasetLoadError(dataset_root, "meta/episodes/chunk-*/file-*.parquet 파일이 없습니다.")

    frames = [pd.read_parquet(path) for path in episode_files]
    episodes = pd.concat(frames, ignore_index=True)

    matches = episodes[episodes["episode_index"] == episode_index]
    if matches.empty:
        raise InvalidEpisodeIndexError(episode_index, total_episodes)
    return matches.iloc[0]


def load_episode(dataset_root: str | Path, episode_index: int, dataset_info: DatasetInfo | None = None) -> EpisodeData:
    """지정한 에피소드의 action/state/timestamp/frame_index를 읽는다.

    Raises:
        DatasetLoadError: 데이터셋 경로/필수 파일 문제.
        InvalidEpisodeIndexError: episode_index가 존재하지 않는 경우.
    """
    root = Path(dataset_root).expanduser().resolve()
    _require_dir(root)
    info = dataset_info or load_dataset_info(root)

    if episode_index < 0:
        raise InvalidEpisodeIndexError(episode_index, info.total_episodes)

    episode_row = _load_episode_meta_row(root, episode_index, info.total_episodes)

    chunk_index = int(episode_row["data/chunk_index"])
    file_index = int(episode_row["data/file_index"])
    data_path = root / info.data_path_template.format(chunk_index=chunk_index, file_index=file_index)
    if not data_path.is_file():
        raise DatasetLoadError(root, f"데이터 parquet 파일을 찾을 수 없습니다: {data_path}")

    data = pd.read_parquet(data_path)
    episode_rows = data[data["episode_index"] == episode_index].sort_values("frame_index")
    if episode_rows.empty:
        raise DatasetLoadError(
            root, f"에피소드 {episode_index}에 해당하는 행을 {data_path}에서 찾을 수 없습니다."
        )

    action = np.stack(episode_rows["action"].to_numpy()).astype(np.float64)
    state = np.stack(episode_rows["observation.state"].to_numpy()).astype(np.float64)
    timestamp = episode_rows["timestamp"].to_numpy().astype(np.float64)
    frame_index = episode_rows["frame_index"].to_numpy().astype(np.int64)

    tasks = episode_row.get("tasks")
    task = str(tasks[0]) if isinstance(tasks, (list, np.ndarray)) and len(tasks) > 0 else ""

    declared_length = int(episode_row.get("length", len(episode_rows)))
    if declared_length != len(episode_rows):
        raise DatasetLoadError(
            root,
            f"에피소드 {episode_index}: metadata length={declared_length}, "
            f"실제 프레임 수={len(episode_rows)} (불일치).",
        )

    return EpisodeData(
        episode_index=episode_index,
        length=len(episode_rows),
        task=task,
        fps=info.fps,
        joint_names=info.action_names,
        action=action,
        state=state,
        timestamp=timestamp,
        frame_index=frame_index,
    )
