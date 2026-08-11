from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import simulation.mujoco.primary_replay_rollout as primary_mod
from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY, JOINT_ORDER
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
from simulation.mujoco.pick_drop_eval import ReferenceZones
from simulation.mujoco.primary_replay_rollout import run_primary_replay
from simulation.mujoco.so101_model import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "simulation" / "mujoco" / "assets" / "scene_pick_drop.xml"
SCENES_CONFIG_PATH = PROJECT_ROOT / "configs" / "mujoco_rollout_scenes_v1.json"


class FakeChunkRunner:
    def __init__(self, chunk_size: int = 5) -> None:
        self.chunk_size = chunk_size
        self.reset_count = 0
        self.predict_calls = 0

    def reset(self, *, task: str | None = None) -> None:
        self.reset_count += 1

    def predict_chunk(self, *, state_deg, images, task, seed=None):
        self.predict_calls += 1
        from simulation.mujoco.smolvla_chunk_runner import ChunkPredictResult

        rows = [dict(state_deg) for _ in range(self.chunk_size)]
        return ChunkPredictResult(chunk_deg=rows, chunk_size=self.chunk_size, seed=seed)


class FakeEpisodeMeta(dict):
    pass


class FakeDataset:
    """Duck-types just enough of ``LeRobotDataset`` for ``run_primary_replay``: ``num_episodes``,
    ``meta.episodes[i]`` and ``__getitem__(row)`` returning real-shaped tensors."""

    class _Meta:
        def __init__(self, episodes):
            self.episodes = episodes

    def __init__(self, *, length: int, initial_pose_deg: dict[str, float]):
        self.num_episodes = 1
        self.meta = FakeDataset._Meta([{"length": length, "dataset_from_index": 0}])
        self._length = length
        self._state = torch.tensor([initial_pose_deg[j] for j in JOINT_ORDER], dtype=torch.float32)

    def __getitem__(self, row: int) -> dict:
        img = torch.zeros(3, 8, 8, dtype=torch.float32)
        return {
            "observation.state": self._state.clone(),
            CAMERA_WORKSPACE_KEY: img.clone(),
            CAMERA_WRIST_KEY: img.clone(),
        }


@pytest.fixture(scope="module")
def model():
    if not SCENE_PATH.is_file():
        pytest.skip("scene_pick_drop.xml not generated yet")
    return load_model(SCENE_PATH)


@pytest.fixture(scope="module")
def scenes_config():
    if not SCENES_CONFIG_PATH.is_file():
        pytest.skip("mujoco_rollout_scenes_v1.json not generated yet")
    return json.loads(SCENES_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def safety_gate():
    try:
        return SafetyGate(SafetyGateConfig.from_repo_defaults())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SafetyGate config unavailable: {exc}")


def test_primary_replay_runs_to_episode_exhaustion(monkeypatch, model, scenes_config, safety_gate):
    scene = scenes_config["scenes"][0]
    fake_dataset = FakeDataset(length=23, initial_pose_deg=scene["initial_pose_deg"])
    monkeypatch.setattr(primary_mod, "load_real_episode_dataset", lambda root: fake_dataset)

    zones = ReferenceZones(bin_center_xy=tuple(scene["bin_center_xy"]), bin_inner_half=scenes_config["bin_inner_half"])
    fake_runner = FakeChunkRunner(chunk_size=5)

    result = run_primary_replay(
        chunk_runner=fake_runner, model=model, safety_gate=safety_gate, scene_id=scene["scene_id"],
        dataset_root="fake/dataset", episode_index=0, zones=zones, cube_xy=tuple(scene["cube_xy"]),
        cube_z_init=scene["cube_z_init"], seed=0, chunk_size=5,
    )

    assert result.real_follower_write_count == 0
    assert not result.ended_by_safety_reject
    assert result.ended_reason == "episode_exhausted"
    # 23 frames / chunk_size 5 -> 5 chunk boundaries (0,5,10,15,20), each executing 5 physical
    # steps -> 25 total physics steps.
    assert result.chunk_boundary_frames == [0, 5, 10, 15, 20]
    assert len(result.step_records) == 25
    assert fake_runner.predict_calls == 5
    assert fake_runner.reset_count == 1
