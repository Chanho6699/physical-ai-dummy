"""Fail-closed, read-only index view for a LeRobot training dataset."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ManifestValidation:
    sample_count: int
    episode_count: int
    excluded_count: int
    selected_base_indices: tuple[int, ...]
    selected_relative_by_episode: dict[int, tuple[int, ...]]


def _episode_bounds(dataset: Any) -> tuple[list[int], list[int]]:
    episodes = dataset.meta.episodes
    return list(map(int, episodes["dataset_from_index"])), list(map(int, episodes["dataset_to_index"]))


def validate_candidate_a_manifest(
    dataset: Any,
    manifest_path: str | Path,
    *,
    expected_samples: int = 20_857,
    expected_episodes: int = 41,
) -> ManifestValidation:
    """Validate Candidate A against the loaded dataset without changing either one."""
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("candidate") != "A":
        raise ValueError(f"Expected Candidate A manifest, got {payload.get('candidate')!r}")
    if payload.get("source_dataset_modified") is not False:
        raise ValueError("Manifest does not attest that the source dataset was unmodified")

    manifest_root = Path(payload["dataset_root"]).expanduser().resolve()
    dataset_root = Path(dataset.root).expanduser().resolve()
    if manifest_root != dataset_root:
        raise ValueError(f"Manifest dataset_root {manifest_root} != loaded dataset root {dataset_root}")

    starts, ends = _episode_bounds(dataset)
    if len(starts) != expected_episodes or len(ends) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, loaded {len(starts)}")
    raw_relative = payload.get("selected_relative_indices_by_episode")
    if not isinstance(raw_relative, dict) or set(raw_relative) != {str(i) for i in range(expected_episodes)}:
        raise ValueError("Manifest must contain exactly episodes 0..40")

    selected: list[int] = []
    relative_by_episode: dict[int, tuple[int, ...]] = {}
    for episode, (start, end) in enumerate(zip(starts, ends, strict=True)):
        relative = tuple(int(i) for i in raw_relative[str(episode)])
        length = end - start
        if not relative or relative[0] != 0:
            raise ValueError(f"Episode {episode}: frame 0 is missing")
        if len(relative) != len(set(relative)) or list(relative) != sorted(relative):
            raise ValueError(f"Episode {episode}: duplicate or unsorted indices")
        if relative[0] < 0 or relative[-1] >= length:
            raise ValueError(f"Episode {episode}: out-of-range index for length {length}")
        if len(relative) < 2:
            raise ValueError(f"Episode {episode}: onset/tail is missing")
        onset = relative[1]
        expected_relative = (0, *range(onset, length))
        if relative != expected_relative:
            raise ValueError(
                f"Episode {episode}: Candidate A must be [0] + [onset..end); got a non-tail selection"
            )
        relative_by_episode[episode] = relative
        selected.extend(start + i for i in relative)

    declared_absolute = tuple(int(i) for i in payload.get("selected_absolute_indices", []))
    if tuple(selected) != declared_absolute:
        raise ValueError("Manifest absolute indices do not match its episode-relative indices")
    if len(selected) != expected_samples:
        raise ValueError(f"Expected {expected_samples} selected samples, got {len(selected)}")
    if len(selected) != len(set(selected)):
        raise ValueError("Manifest contains duplicate absolute indices")
    if selected[0] < 0 or selected[-1] >= len(dataset):
        raise ValueError("Manifest contains an out-of-range absolute index")

    return ManifestValidation(
        sample_count=len(selected),
        episode_count=len(starts),
        excluded_count=len(dataset) - len(selected),
        selected_base_indices=tuple(selected),
        selected_relative_by_episode=relative_by_episode,
    )


class CanonicalTrainingIndexView(torch.utils.data.Dataset):
    """Map a contiguous sampler index space onto immutable base-dataset samples.

    The synthetic episode bounds are important: LeRobot's EpisodeAwareSampler reads
    them directly, so a plain ``Subset`` paired with the base metadata would silently
    sample the excluded frames again.
    """

    def __init__(self, dataset: Any, validation: ManifestValidation):
        self.dataset = dataset
        self.indices = validation.selected_base_indices
        self.validation = validation
        self.meta = copy.copy(dataset.meta)

        lengths = [len(validation.selected_relative_by_episode[i]) for i in range(validation.episode_count)]
        ends = np.cumsum(lengths).astype(np.int64).tolist()
        starts = [0, *ends[:-1]]
        episodes = dataset.meta.episodes
        episodes = episodes.remove_columns(["dataset_from_index", "dataset_to_index", "length"])
        episodes = episodes.add_column("length", lengths)
        episodes = episodes.add_column("dataset_from_index", starts)
        episodes = episodes.add_column("dataset_to_index", ends)
        self.meta.episodes = episodes

        self.episodes = None
        self.absolute_to_relative_idx = None
        self.num_frames = len(self.indices)
        self.num_episodes = validation.episode_count

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def apply_training_index_manifest(dataset: Any, manifest_path: str | Path) -> CanonicalTrainingIndexView:
    validation = validate_candidate_a_manifest(dataset, manifest_path)
    return CanonicalTrainingIndexView(dataset, validation)
