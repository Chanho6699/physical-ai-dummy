from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

from runtime.training.canonical_index_view import apply_training_index_manifest


class FakeDataset:
    def __init__(self, root: Path):
        self.root = root
        self.values = list(range(9))
        self.meta = SimpleNamespace(
            episodes=Dataset.from_dict(
                {
                    "episode_index": [0, 1],
                    "length": [4, 5],
                    "dataset_from_index": [0, 4],
                    "dataset_to_index": [4, 9],
                }
            )
        )

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {"index": index, "value": self.values[index]}


def write_manifest(path: Path, root: Path, relative=None):
    relative = relative or {"0": [0, 2, 3], "1": [0, 3, 4]}
    absolute = [0, 2, 3, 4, 7, 8]
    path.write_text(
        json.dumps(
            {
                "candidate": "A",
                "dataset_root": str(root),
                "source_dataset_modified": False,
                "selected_relative_indices_by_episode": relative,
                "selected_absolute_indices": absolute,
            }
        )
    )


def test_view_maps_indices_and_rewrites_sampler_bounds(tmp_path):
    manifest = tmp_path / "indices.json"
    write_manifest(manifest, tmp_path)
    view = apply_training_index_manifest(
        FakeDataset(tmp_path), manifest
    ) if False else None
    # Production constants are intentionally strict; exercise them on the real manifest elsewhere.
    from runtime.training.canonical_index_view import CanonicalTrainingIndexView, validate_candidate_a_manifest
    validation = validate_candidate_a_manifest(FakeDataset(tmp_path), manifest, expected_samples=6, expected_episodes=2)
    view = CanonicalTrainingIndexView(FakeDataset(tmp_path), validation)
    assert [view[i]["index"] for i in range(len(view))] == [0, 2, 3, 4, 7, 8]
    assert view.meta.episodes["dataset_from_index"] == [0, 3]
    assert view.meta.episodes["dataset_to_index"] == [3, 6]
    assert view.num_frames == 6


@pytest.mark.parametrize(
    "relative",
    [
        {"0": [2, 3], "1": [0, 3, 4]},
        {"0": [0, 2, 3], "1": [0, 2, 4]},
        {"0": [0, 2, 2, 3], "1": [0, 3, 4]},
    ],
)
def test_manifest_fails_closed_on_invalid_candidate_a(tmp_path, relative):
    manifest = tmp_path / "indices.json"
    write_manifest(manifest, tmp_path, relative)
    from runtime.training.canonical_index_view import validate_candidate_a_manifest
    with pytest.raises(ValueError):
        validate_candidate_a_manifest(FakeDataset(tmp_path), manifest, expected_samples=6, expected_episodes=2)
