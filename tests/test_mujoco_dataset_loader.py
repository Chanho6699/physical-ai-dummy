from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import ACTION_NAMES, build_synthetic_dataset
from simulation.mujoco.dataset_loader import (
    DatasetLoadError,
    InvalidEpisodeIndexError,
    load_dataset_info,
    load_episode,
)


def test_load_dataset_info_valid(synthetic_dataset: Path):
    info = load_dataset_info(synthetic_dataset)
    assert info.fps == 30
    assert info.total_episodes == 1
    assert info.action_dim == 6
    assert list(info.action_names) == ACTION_NAMES


def test_load_dataset_info_does_not_hardcode_six_dims(tmp_path):
    root = build_synthetic_dataset(
        tmp_path / "four_dim",
        action_names=["a.pos", "b.pos", "c.pos", "d.pos"],
        action_values=np.zeros((5, 4), dtype=np.float32),
        num_frames=5,
    )
    info = load_dataset_info(root)
    assert info.action_dim == 4


def test_missing_dataset_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(DatasetLoadError):
        load_dataset_info(missing)


def test_missing_metadata_file_raises(tmp_path):
    root = tmp_path / "no_meta"
    root.mkdir()
    with pytest.raises(DatasetLoadError):
        load_dataset_info(root)


def test_load_episode_valid(synthetic_dataset: Path):
    info = load_dataset_info(synthetic_dataset)
    episode = load_episode(synthetic_dataset, 0, info)
    assert episode.length == 10
    assert episode.action.shape == (10, 6)
    assert episode.task == "테스트용 합성 에피소드."


def test_load_episode_invalid_index_raises(synthetic_dataset: Path):
    info = load_dataset_info(synthetic_dataset)
    with pytest.raises(InvalidEpisodeIndexError) as excinfo:
        load_episode(synthetic_dataset, 25, info)
    assert excinfo.value.requested == 25
    assert excinfo.value.total_episodes == 1


def test_load_episode_negative_index_raises(synthetic_dataset: Path):
    info = load_dataset_info(synthetic_dataset)
    with pytest.raises(InvalidEpisodeIndexError):
        load_episode(synthetic_dataset, -1, info)


def test_real_dataset_episode_0(real_dataset_root: Path):
    info = load_dataset_info(real_dataset_root)
    assert info.total_episodes == 20
    assert info.action_dim == 6
    episode = load_episode(real_dataset_root, 0, info)
    assert episode.length == 897
    assert episode.action.shape == (897, 6)
    assert np.all(np.diff(episode.frame_index) == 1)
    assert np.all(np.diff(episode.timestamp) > 0)
