from __future__ import annotations
import json
import pytest
from data_collection.episode_labels import (
    EpisodeLabelError, append_episode_label, get_episode_indices_by_type,
    read_episode_labels, validate_label_alignment,
)
from data_collection.lerobot_record_runner import make_labeled_save_episode

class FakeDataset:
    def __init__(self, root, count=0, fail=False):
        self.root = root
        self.num_episodes = count
        self.fail = fail
        self._pending_episode_type = "clean"

def fake_save(dataset):
    if dataset.fail:
        raise RuntimeError("save failed")
    dataset.num_episodes += 1

def test_clean_recovery_and_subset_indices(tmp_path):
    append_episode_label(tmp_path, episode_index=0, episode_type="clean", dataset_episode_count=1)
    append_episode_label(tmp_path, episode_index=1, episode_type="clean", dataset_episode_count=2)
    append_episode_label(tmp_path, episode_index=2, episode_type="recovery", dataset_episode_count=3)
    assert get_episode_indices_by_type(tmp_path, "clean") == [0, 1]
    assert get_episode_indices_by_type(tmp_path, "recovery") == [2]
    assert get_episode_indices_by_type(tmp_path, "all") == [0, 1, 2]

def test_manifest_schema_and_atomic_temp_cleanup(tmp_path):
    path = append_episode_label(tmp_path, episode_index=0, episode_type="clean", dataset_episode_count=1)
    assert json.loads(path.read_text()) == {
        "episode_index": 0, "episode_type": "clean", "task_success": True,
    }
    assert not path.with_name(path.name + ".tmp").exists()

def test_duplicate_label_is_rejected(tmp_path):
    append_episode_label(tmp_path, episode_index=0, episode_type="clean", dataset_episode_count=1)
    with pytest.raises(EpisodeLabelError):
        append_episode_label(tmp_path, episode_index=0, episode_type="recovery", dataset_episode_count=1)

def test_resume_preserves_labels_and_appends_next_index(tmp_path):
    append_episode_label(tmp_path, episode_index=0, episode_type="clean", dataset_episode_count=1)
    append_episode_label(tmp_path, episode_index=1, episode_type="recovery", dataset_episode_count=2)
    validate_label_alignment(tmp_path, 2)
    append_episode_label(tmp_path, episode_index=2, episode_type="clean", dataset_episode_count=3)
    assert sorted(read_episode_labels(tmp_path)) == [0, 1, 2]

@pytest.mark.parametrize("count,rows", [(2, [0]), (1, [0, 1])])
def test_dataset_manifest_mismatch_detected(tmp_path, count, rows):
    path = tmp_path / "meta/episode_labels.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps({
        "episode_index": i, "episode_type": "clean", "task_success": True,
    }) + "\n" for i in rows))
    with pytest.raises(EpisodeLabelError):
        validate_label_alignment(tmp_path, count)

def test_duplicate_rows_detected(tmp_path):
    path = tmp_path / "meta/episode_labels.jsonl"
    path.parent.mkdir()
    row = json.dumps({"episode_index": 0, "episode_type": "clean", "task_success": True})
    path.write_text(row + "\n" + row + "\n")
    with pytest.raises(EpisodeLabelError, match="duplicate"):
        read_episode_labels(tmp_path)

def test_save_failure_writes_no_label(tmp_path):
    dataset = FakeDataset(tmp_path, fail=True)
    with pytest.raises(RuntimeError):
        make_labeled_save_episode(fake_save)(dataset)
    assert not (tmp_path / "meta/episode_labels.jsonl").exists()

def test_label_written_only_after_successful_save(tmp_path):
    dataset = FakeDataset(tmp_path)
    make_labeled_save_episode(fake_save)(dataset)
    assert dataset.num_episodes == 1
    assert read_episode_labels(tmp_path)[0]["episode_type"] == "clean"

def test_recovery_label_after_successful_save(tmp_path):
    dataset = FakeDataset(tmp_path)
    dataset._pending_episode_type = "recovery"
    make_labeled_save_episode(fake_save)(dataset)
    assert read_episode_labels(tmp_path)[0]["episode_type"] == "recovery"

def test_unlabeled_save_is_fail_closed_before_dataset_mutation(tmp_path):
    dataset = FakeDataset(tmp_path)
    del dataset._pending_episode_type
    with pytest.raises(EpisodeLabelError):
        make_labeled_save_episode(fake_save)(dataset)
    assert dataset.num_episodes == 0
