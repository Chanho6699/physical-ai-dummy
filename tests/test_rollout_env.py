from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
from simulation.mujoco.pick_drop_eval import ReferenceZones
from simulation.mujoco.rollout_env import run_synthetic_closed_loop
from simulation.mujoco.so101_model import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "simulation" / "mujoco" / "assets" / "scene_pick_drop.xml"
SCENES_CONFIG_PATH = PROJECT_ROOT / "configs" / "mujoco_rollout_scenes_v1.json"


class FakeChunkRunner:
    """Duck-types ``SmolVLAChunkRunner`` without loading torch/lerobot - returns the current
    state unchanged (robot holds still) so the rollout loop can be exercised end-to-end without
    GPU/checkpoint access."""

    def __init__(self, chunk_size: int = 5) -> None:
        self.chunk_size = chunk_size
        self.reset_count = 0
        self.calls = 0

    def reset(self, *, task: str | None = None) -> None:
        self.reset_count += 1

    def next_queued_action(self, *, state_deg, images, task, seed=None):
        self.calls += 1
        chunk_boundary = (self.calls - 1) % self.chunk_size == 0
        return dict(state_deg), chunk_boundary


@pytest.fixture(scope="module")
def model():
    if not SCENE_PATH.is_file():
        pytest.skip("scene_pick_drop.xml not generated yet (run scripts/generate_mujoco_pick_drop_scene.py)")
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
        pytest.skip(f"SafetyGate config unavailable in this environment: {exc}")


def test_synthetic_closed_loop_runs_to_completion_with_fake_policy(model, scenes_config, safety_gate):
    scene = scenes_config["scenes"][0]
    zones = ReferenceZones(bin_center_xy=tuple(scene["bin_center_xy"]), bin_inner_half=scenes_config["bin_inner_half"])
    fake = FakeChunkRunner(chunk_size=5)

    result = run_synthetic_closed_loop(
        chunk_runner=fake, model=model, safety_gate=safety_gate, scene_id=scene["scene_id"],
        initial_pose_deg=scene["initial_pose_deg"], cube_xy=tuple(scene["cube_xy"]),
        cube_z_init=scene["cube_z_init"], zones=zones, seed=0, max_steps=10,
    )

    assert result.real_follower_write_count == 0
    assert len(result.step_records) == 10
    assert fake.reset_count == 1
    # holding still should never trip excessive-step or gross-range REJECT
    assert not result.ended_by_safety_reject
    assert result.ended_reason == "max_steps_reached"
    assert set(result.safe_command_log[0].keys()) == set(JOINT_ORDER)
    assert result.eval_result.failure.reason in {"failed_approach", "none", "other"}


def test_synthetic_closed_loop_stops_on_safety_reject(model, scenes_config, safety_gate):
    scene = scenes_config["scenes"][0]
    zones = ReferenceZones(bin_center_xy=tuple(scene["bin_center_xy"]), bin_inner_half=scenes_config["bin_inner_half"])

    class WildFakeChunkRunner(FakeChunkRunner):
        def next_queued_action(self, *, state_deg, images, task, seed=None):
            self.calls += 1
            # A wildly out-of-range action should trigger a gross-violation REJECT immediately.
            wild = {j: 999.0 for j in JOINT_ORDER}
            return wild, True

    fake = WildFakeChunkRunner()
    result = run_synthetic_closed_loop(
        chunk_runner=fake, model=model, safety_gate=safety_gate, scene_id=scene["scene_id"],
        initial_pose_deg=scene["initial_pose_deg"], cube_xy=tuple(scene["cube_xy"]),
        cube_z_init=scene["cube_z_init"], zones=zones, seed=0, max_steps=50,
    )
    assert result.ended_by_safety_reject
    assert result.ended_reason == "safety_reject"
    assert result.eval_result.failure.reason == "safety_reject"
    assert result.real_follower_write_count == 0
