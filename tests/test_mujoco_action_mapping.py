from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import ACTION_NAMES
from simulation.mujoco.action_mapping import (
    ActionMappingError,
    build_default_mapping,
    map_action_row,
    map_action_value,
    validate_mapping_against_model,
)
from simulation.mujoco.so101_model import SO101_JOINT_NAMES, load_model


def test_build_default_mapping_order_matches_dataset():
    mapping = build_default_mapping(ACTION_NAMES)
    assert [entry.dataset_name for entry in mapping] == list(SO101_JOINT_NAMES)
    assert [entry.mujoco_joint_name for entry in mapping] == list(SO101_JOINT_NAMES)
    assert [entry.mujoco_actuator_name for entry in mapping] == list(SO101_JOINT_NAMES)
    for i, entry in enumerate(mapping):
        assert entry.dataset_index == i


def test_build_default_mapping_unknown_name_raises():
    with pytest.raises(ActionMappingError):
        build_default_mapping(["not_a_real_joint.pos"] + ACTION_NAMES[1:])


def test_build_default_mapping_duplicate_name_raises():
    names = list(ACTION_NAMES)
    names[1] = names[0]  # 중복
    with pytest.raises(ActionMappingError):
        build_default_mapping(names)


def test_map_action_value_degrees_to_radians():
    mapping = build_default_mapping(ACTION_NAMES)
    entry = mapping[0]
    assert entry.unit == "deg"
    assert math.isclose(map_action_value(180.0, entry), math.pi, rel_tol=1e-6)
    assert math.isclose(map_action_value(0.0, entry), 0.0, abs_tol=1e-9)


def test_map_action_row_returns_all_actuators():
    mapping = build_default_mapping(ACTION_NAMES)
    row = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    result = map_action_row(row, mapping)
    assert set(result.keys()) == set(SO101_JOINT_NAMES)
    assert math.isclose(result["shoulder_pan"], math.radians(10.0), rel_tol=1e-6)


def test_validate_mapping_against_real_model():
    mapping = build_default_mapping(ACTION_NAMES)
    model = load_model()
    validate_mapping_against_model(mapping, model)  # 예외가 없으면 통과
