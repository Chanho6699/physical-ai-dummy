"""runtime/laptop/mujoco_shadow_backend.py 테스트 - 실제 MuJoCo(mujoco 패키지) 사용.

기존 Realistic Control Layer/action_mapping/so101_model을 그대로 재사용하는지, 초기
state sync가 세션 시작 시 1회만 적용되는지를 검증한다.
"""

from __future__ import annotations

import math
import time

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.mujoco_shadow_backend import MuJoCoBackendError, RealisticMuJoCoBackend
from simulation.realism.so101_realistic_control import RealisticControlConfig


def _zero_state() -> dict[str, float]:
    return {j: 0.0 for j in JOINT_ORDER}


@pytest.fixture()
def backend() -> RealisticMuJoCoBackend:
    b = RealisticMuJoCoBackend(control_config=RealisticControlConfig(enable_latency=False))
    b.preflight()
    return b


def test_preflight_populates_joint_limits(backend: RealisticMuJoCoBackend) -> None:
    limits = backend.joint_limits_deg
    assert set(limits) == set(JOINT_ORDER)
    for lo, hi in limits.values():
        assert lo < hi


def test_execute_before_sync_raises(backend: RealisticMuJoCoBackend) -> None:
    with pytest.raises(MuJoCoBackendError):
        backend.execute_single_step(_zero_state(), now=time.monotonic())


def test_execute_before_preflight_raises() -> None:
    b = RealisticMuJoCoBackend()
    with pytest.raises(MuJoCoBackendError):
        b.sync_initial_state(_zero_state())


def test_single_step_moves_toward_command_with_latency_disabled(backend: RealisticMuJoCoBackend) -> None:
    backend.sync_initial_state(_zero_state())
    command = _zero_state()
    command["shoulder_pan"] = 10.0
    result = backend.execute_single_step(command, now=time.monotonic())

    assert result.final_state_finite is True
    assert result.initial_state_deg["shoulder_pan"] == 0.0
    # latency를 껐으므로 이번 프레임에 바로 shoulder_pan 쪽으로 움직여야 한다 (rate-limit은
    # 여전히 켜져 있어 10.0에 정확히 도달하지 않을 수 있으므로 방향/부호만 확인한다).
    assert result.final_state_deg["shoulder_pan"] > 0.0
    assert result.mechanical_violations == ()


def test_single_step_holds_position_with_zero_command(backend: RealisticMuJoCoBackend) -> None:
    initial = _zero_state()
    initial["gripper"] = 20.0
    backend.sync_initial_state(initial)
    result = backend.execute_single_step(initial, now=time.monotonic())
    assert result.final_state_finite is True
    assert math.isclose(result.final_state_deg["gripper"], 20.0, abs_tol=1.0)


def test_sync_initial_state_missing_joint_raises(backend: RealisticMuJoCoBackend) -> None:
    incomplete = _zero_state()
    del incomplete["gripper"]
    with pytest.raises(MuJoCoBackendError):
        backend.sync_initial_state(incomplete)


def test_realistic_diagnostics_cover_all_joints(backend: RealisticMuJoCoBackend) -> None:
    backend.sync_initial_state(_zero_state())
    result = backend.execute_single_step(_zero_state(), now=time.monotonic())
    assert set(result.realistic_diagnostics) == set(JOINT_ORDER)
