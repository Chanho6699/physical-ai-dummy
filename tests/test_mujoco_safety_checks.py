from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from conftest import ACTION_NAMES, build_synthetic_dataset
from simulation.mujoco.action_mapping import build_default_mapping
from simulation.mujoco.dataset_loader import load_dataset_info, load_episode
from simulation.mujoco.safety_checks import (
    DynamicCheckState,
    check_frame_targets,
    check_simulation_state,
    load_safety_config,
    run_static_checks,
)
from simulation.mujoco.so101_model import load_model, make_data


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def config():
    return load_safety_config()


def _load(root, num_frames=10, **kwargs):
    build_synthetic_dataset(root, num_frames=num_frames, **kwargs)
    info = load_dataset_info(root)
    episode = load_episode(root, 0, info)
    mapping = build_default_mapping(list(info.action_names))
    return info, episode, mapping


def test_static_checks_pass_on_clean_data(tmp_path, model):
    info, episode, mapping = _load(tmp_path / "clean")
    events = run_static_checks(episode, info, mapping, model)
    assert not any(e.level == "BLOCKED" for e in events)
    assert any(e.code == "static_checks" and e.level == "PASS" for e in events)


def test_action_shape_mismatch_is_blocked(tmp_path, model):
    info, episode, mapping = _load(tmp_path / "shape")
    bad_episode = dataclasses.replace(episode, action=episode.action[:, :3])
    events = run_static_checks(bad_episode, info, mapping, model)
    assert any(e.level == "BLOCKED" and e.code == "action_shape" for e in events)


def test_nan_action_is_blocked(tmp_path, model):
    values = np.zeros((10, 6), dtype=np.float32)
    values[3, 2] = np.nan
    info, episode, mapping = _load(tmp_path / "nan", action_values=values)
    events = run_static_checks(episode, info, mapping, model)
    blocked = [e for e in events if e.level == "BLOCKED" and e.code == "action_non_finite"]
    assert blocked
    assert blocked[0].frame == 3


def test_inf_action_is_blocked(tmp_path, model):
    values = np.zeros((10, 6), dtype=np.float32)
    values[5, 0] = np.inf
    info, episode, mapping = _load(tmp_path / "inf", action_values=values)
    events = run_static_checks(episode, info, mapping, model)
    assert any(e.level == "BLOCKED" and e.code == "action_non_finite" for e in events)


def test_frame_index_discontinuous_is_blocked(tmp_path, model):
    bad_frame_index = np.arange(10, dtype=np.int64)
    bad_frame_index[5] = 99
    info, episode, mapping = _load(tmp_path / "frame_gap", frame_index_override=bad_frame_index)
    events = run_static_checks(episode, info, mapping, model)
    assert any(e.level == "BLOCKED" and e.code == "frame_index_discontinuous" for e in events)


def test_timestamp_reversed_is_blocked(tmp_path, model):
    bad_ts = np.arange(10, dtype=np.float64) / 30
    bad_ts[4] = bad_ts[3] - 1.0
    info, episode, mapping = _load(tmp_path / "ts_reverse", timestamp_override=bad_ts)
    events = run_static_checks(episode, info, mapping, model)
    assert any(e.level == "BLOCKED" and e.code == "timestamp_reversed" for e in events)


def test_joint_mapping_incomplete_is_blocked(tmp_path, model):
    info, episode, mapping = _load(tmp_path / "map_missing")
    incomplete_mapping = mapping[:-1]  # gripper 매핑 누락
    events = run_static_checks(episode, info, incomplete_mapping, model)
    assert any(e.level == "BLOCKED" and e.code == "joint_mapping_incomplete" for e in events)


def test_actuator_mapping_missing_is_blocked(tmp_path, model):
    info, episode, mapping = _load(tmp_path / "actuator_missing")
    broken = list(mapping)
    broken[0] = dataclasses.replace(broken[0], mujoco_actuator_name="no_such_actuator")
    events = run_static_checks(episode, info, tuple(broken), model)
    assert any(e.level == "BLOCKED" and e.code == "actuator_mapping_missing" for e in events)


def test_dynamic_joint_limit_exceeded_is_blocked(model, config):
    mapping = build_default_mapping(ACTION_NAMES)
    state = DynamicCheckState()
    target = {name: 0.0 for name in [e.mujoco_actuator_name for e in mapping]}
    target["shoulder_lift"] = 999.0  # 명백히 관절 range를 초과하는 값
    events = check_frame_targets(0, target, mapping, model, config, state)
    blocked = [e for e in events if e.level == "BLOCKED" and e.code == "joint_limit"]
    assert blocked
    assert blocked[0].joint == "shoulder_lift"
    assert blocked[0].value == 999.0


def test_dynamic_max_delta_warns(model, config):
    mapping = build_default_mapping(ACTION_NAMES)
    state = DynamicCheckState()
    zero_target = {e.mujoco_actuator_name: 0.0 for e in mapping}
    check_frame_targets(0, zero_target, mapping, model, config, state)  # 첫 프레임은 이전 값 없음
    big_jump = dict(zero_target)
    big_jump["shoulder_pan"] = 1.0  # max_joint_delta_per_frame(0.08)보다 훨씬 큼
    events = check_frame_targets(1, big_jump, mapping, model, config, state)
    assert any(e.level == "WARN" and e.code == "max_delta" for e in events)


def test_dynamic_simulation_nan_is_blocked(model, config):
    mapping = build_default_mapping(ACTION_NAMES)
    data = make_data(model)
    state = DynamicCheckState()
    data.qpos[0] = math.nan
    events = check_simulation_state(0, model, data, mapping, config, state)
    assert any(e.level == "BLOCKED" and e.code == "simulation_nan" for e in events)


def test_load_safety_config_has_required_keys():
    config = load_safety_config()
    assert config.stop_on_joint_limit is True
    assert config.stop_on_nan is True
    assert config.stop_on_collision is True
    assert set(config.max_joint_delta_per_frame.keys()) == {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    }
