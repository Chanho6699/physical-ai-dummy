"""Episode semantic labels stored beside, never inside, a LeRobot dataset schema."""
from __future__ import annotations
import json
import os
from pathlib import Path

LABEL_RELATIVE_PATH = Path("meta/episode_labels.jsonl")
VALID_EPISODE_TYPES = frozenset({"clean", "recovery"})

class EpisodeLabelError(RuntimeError):
    pass

def _label_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root).expanduser().resolve() / LABEL_RELATIVE_PATH

def read_episode_labels(dataset_root: str | Path) -> dict[int, dict]:
    path = _label_path(dataset_root)
    if not path.exists():
        return {}
    labels: dict[int, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            index = int(row["episode_index"])
            episode_type = row["episode_type"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EpisodeLabelError(f"{path}:{line_number}: invalid label row: {exc}") from exc
        if index < 0 or episode_type not in VALID_EPISODE_TYPES or row.get("task_success") is not True:
            raise EpisodeLabelError(f"{path}:{line_number}: invalid label values: {row}")
        if index in labels:
            raise EpisodeLabelError(f"{path}: duplicate episode_index label: {index}")
        labels[index] = {"episode_index": index, "episode_type": episode_type, "task_success": True}
    return labels

def validate_label_alignment(dataset_root: str | Path, dataset_episode_count: int) -> None:
    labels = read_episode_labels(dataset_root)
    expected = set(range(dataset_episode_count))
    actual = set(labels)
    if actual != expected:
        raise EpisodeLabelError(
            f"dataset/label mismatch: episodes={dataset_episode_count}, "
            f"missing_labels={sorted(expected-actual)}, orphan_labels={sorted(actual-expected)}"
        )

def append_episode_label(dataset_root: str | Path, *, episode_index: int, episode_type: str, dataset_episode_count: int) -> Path:
    """Atomically rewrite the manifest, only after LeRobot save_episode succeeds."""
    if episode_type not in VALID_EPISODE_TYPES:
        raise EpisodeLabelError(f"unsupported episode_type: {episode_type}")
    if dataset_episode_count != episode_index + 1:
        raise EpisodeLabelError(f"post-save count mismatch: index={episode_index}, count={dataset_episode_count}")
    labels = read_episode_labels(dataset_root)
    expected_prior = set(range(episode_index))
    if set(labels) != expected_prior:
        raise EpisodeLabelError(
            f"cannot append label {episode_index}: prior labels={sorted(labels)}, expected={sorted(expected_prior)}"
        )
    labels[episode_index] = {"episode_index": episode_index, "episode_type": episode_type, "task_success": True}
    path = _label_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    payload = "".join(json.dumps(labels[i], sort_keys=True) + "\n" for i in sorted(labels))
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path

def get_episode_indices_by_type(dataset_root: str | Path, episode_type: str) -> list[int]:
    labels = read_episode_labels(dataset_root)
    if episode_type == "all":
        return sorted(i for i, row in labels.items() if row["task_success"])
    if episode_type not in VALID_EPISODE_TYPES:
        raise EpisodeLabelError(f"unsupported episode_type: {episode_type}")
    return sorted(i for i, row in labels.items() if row["episode_type"] == episode_type)
