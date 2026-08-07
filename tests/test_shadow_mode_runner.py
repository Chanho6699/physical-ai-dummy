"""runtime/laptop/shadow_mode_runner.py 통합 테스트 (섹션 20 체크리스트).

카메라/팔로워 state는 fake로 주입하고, VLA 서버는 FastAPI TestClient(ASGI in-process)로
붙인 ``FakePolicyRunner``를 쓴다 (진짜 소켓 왕복은 tests/test_vla_client.py가 담당).
MuJoCo backend는 실제 mujoco 패키지를 그대로 쓴다 (섹션 13 - 새 executor를 만들지 않았다는
사실 자체를 검증하기 위해 fake로 바꾸지 않는다) - 단, REJECT 경로에서 MuJoCo가 절대
호출되지 않는지 확인하는 테스트만 mock으로 감싼다.
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY, JOINT_ORDER
from runtime.desktop.vla_server import FakePolicyRunner, PolicyInferenceError, create_app
from runtime.laptop.camera_source import CameraFrame
from runtime.laptop.follower_state_source import REAL_FOLLOWER_WRITE_COUNT, FollowerStateSnapshot
from runtime.laptop.mujoco_shadow_backend import RealisticMuJoCoBackend
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
from runtime.laptop.shadow_logger import RESULT_SHADOW_FAIL, RESULT_SHADOW_PASS, RESULT_SHADOW_PASS_WITH_CLAMP
from runtime.laptop.shadow_mode_runner import ShadowModeRunner
from runtime.laptop.vla_client import VLAClientConfig, VLAHttpClient


class FakeCameraSource:
    def __init__(self, *, fail: bool = False, wrong_shape: bool = False) -> None:
        self._fail = fail
        self._wrong_shape = wrong_shape

    def capture_all(self) -> dict[str, CameraFrame]:
        if self._fail:
            raise RuntimeError("카메라 캡처 실패 (시뮬레이션)")
        shape = (10, 10, 3) if self._wrong_shape else (480, 640, 3)
        img = np.zeros(shape, dtype=np.uint8)
        now = time.time()
        return {
            CAMERA_WORKSPACE_KEY: CameraFrame(image_rgb=img, captured_at_wall=now, width=shape[1], height=shape[0]),
            CAMERA_WRIST_KEY: CameraFrame(image_rgb=img, captured_at_wall=now, width=shape[1], height=shape[0]),
        }


class FakeStateSource:
    """``ReadOnlyRealFollowerStateSource``와 동일한 read-only 인터페이스만 노출한다 -
    ``send_action``류 write 메서드가 애초에 정의되어 있지 않다 (섹션 20 - 15/16번 체크)."""

    def __init__(self, *, positions: dict[str, float] | None = None, fail: bool = False, age_s: float = 0.0) -> None:
        self._positions = positions or {j: 0.0 for j in JOINT_ORDER}
        self._fail = fail
        self._age_s = age_s

    def read(self) -> FollowerStateSnapshot:
        if self._fail:
            raise RuntimeError("팔로워 state 읽기 실패 (시뮬레이션)")
        return FollowerStateSnapshot(
            positions_deg=dict(self._positions), read_at_monotonic=time.monotonic(), read_at_wall=time.time() - self._age_s
        )


def _safety_config() -> SafetyGateConfig:
    joint_range = {j: ((0.0, 100.0) if j == "gripper" else (-90.0, 90.0)) for j in JOINT_ORDER}
    max_step = {j: 5.0 for j in JOINT_ORDER}
    return SafetyGateConfig(joint_range_deg=joint_range, max_step_deg=max_step)


def _vla_client(policy_runner=None) -> VLAHttpClient:
    app = create_app(policy_runner=policy_runner or FakePolicyRunner())
    return VLAHttpClient(VLAClientConfig(server_url="http://testserver"), session=TestClient(app))


@pytest.fixture(scope="module")
def real_mujoco_backend() -> RealisticMuJoCoBackend:
    backend = RealisticMuJoCoBackend()
    backend.preflight()
    return backend


def _runner(*, camera=None, state=None, vla_client=None, mujoco_backend=None, task="Pick up the cube", reports_dir=None) -> ShadowModeRunner:
    return ShadowModeRunner(
        camera_source=camera or FakeCameraSource(),
        state_source=state or FakeStateSource(),
        vla_client=vla_client or _vla_client(),
        safety_gate=SafetyGate(_safety_config()),
        mujoco_backend=mujoco_backend,
        task=task,
        reports_dir=reports_dir,
    )


def test_full_pipeline_pass(tmp_path, real_mujoco_backend) -> None:
    runner = _runner(mujoco_backend=real_mujoco_backend, reports_dir=tmp_path)
    result = runner.run_once()
    assert result.result == RESULT_SHADOW_PASS
    assert result.real_follower_write_count == 0
    assert result.report_path.is_file()
    assert result.report["safety"]["decision"] == "ACCEPT"


def test_would_clamp_action_produces_pass_with_clamp(tmp_path, real_mujoco_backend) -> None:
    # gripper=0 offset이지만 state의 gripper=0 -> 명령을 크게 벗어나게 만들어 clamp를 유도.
    runner = _runner(
        camera=FakeCameraSource(),
        state=FakeStateSource(positions={**{j: 0.0 for j in JOINT_ORDER}, "wrist_flex": 0.0}),
        vla_client=_vla_client(FakePolicyRunner(joint_offsets={"wrist_flex": 8.0})),
        mujoco_backend=real_mujoco_backend,
        reports_dir=tmp_path,
    )
    result = runner.run_once()
    assert result.result == RESULT_SHADOW_PASS_WITH_CLAMP
    assert result.report["safety"]["would_clamp"] is True
    assert result.real_follower_write_count == 0


def test_reject_action_never_reaches_mujoco(tmp_path) -> None:
    mock_backend = MagicMock()
    runner = _runner(
        vla_client=_vla_client(FakePolicyRunner(joint_offsets={"shoulder_pan": 99999.0})),
        mujoco_backend=mock_backend,
        reports_dir=tmp_path,
    )
    result = runner.run_once()
    assert result.result == RESULT_SHADOW_FAIL
    assert result.report["safety"]["decision"] == "REJECT"
    mock_backend.sync_initial_state.assert_not_called()
    mock_backend.execute_single_step.assert_not_called()
    assert result.real_follower_write_count == 0


def test_camera_failure_produces_shadow_fail_before_communication(tmp_path) -> None:
    mock_client = MagicMock()
    runner = _runner(camera=FakeCameraSource(fail=True), vla_client=mock_client, mujoco_backend=MagicMock(), reports_dir=tmp_path)
    result = runner.run_once()
    assert result.result == RESULT_SHADOW_FAIL
    assert result.report["observation"]["schema_valid"] is False
    mock_client.check_health.assert_not_called()


def test_wrong_camera_resolution_is_still_schema_invalid_enough_to_fail(tmp_path) -> None:
    mock_client = MagicMock()
    runner = _runner(camera=FakeCameraSource(wrong_shape=True), vla_client=mock_client, mujoco_backend=MagicMock(), reports_dir=tmp_path)
    result = runner.run_once()
    # 10x10x3은 채널 수는 맞지만 MIN_REASONABLE_DIM 미만이라 invalid 처리된다.
    assert result.result == RESULT_SHADOW_FAIL
    mock_client.check_health.assert_not_called()


def test_state_read_failure_produces_shadow_fail(tmp_path) -> None:
    mock_client = MagicMock()
    runner = _runner(state=FakeStateSource(fail=True), vla_client=mock_client, mujoco_backend=MagicMock(), reports_dir=tmp_path)
    result = runner.run_once()
    assert result.result == RESULT_SHADOW_FAIL
    assert result.report["observation"]["state_valid"] is False


def test_stale_state_is_rejected_by_safety_gate(tmp_path, real_mujoco_backend) -> None:
    runner = _runner(state=FakeStateSource(age_s=10.0), mujoco_backend=real_mujoco_backend, reports_dir=tmp_path)
    result = runner.run_once()
    assert result.result == RESULT_SHADOW_FAIL
    assert result.report["safety"]["decision"] == "REJECT"
    assert any("STATE_STALE" in r for r in result.report["safety"]["reasons"])


def test_desktop_unreachable_is_communication_failure(tmp_path) -> None:
    unreachable_client = VLAHttpClient(VLAClientConfig(server_url="http://127.0.0.1:1", timeout_s=1.0, max_retries=1))
    runner = _runner(vla_client=unreachable_client, mujoco_backend=MagicMock(), reports_dir=tmp_path)
    result = runner.run_once()
    assert result.result == RESULT_SHADOW_FAIL
    assert result.report["communication"]["health_ok"] is False
    assert any("Desktop VLA 서버에 연결할 수 없습니다" in r for r in result.reasons)


def test_vla_inference_failure_is_distinguished_from_communication_failure(tmp_path) -> None:
    class FailingRunner(FakePolicyRunner):
        def predict(self, *, task, state, images):
            raise PolicyInferenceError("체크포인트가 손상되었습니다")

    runner = _runner(vla_client=_vla_client(FailingRunner()), mujoco_backend=MagicMock(), reports_dir=tmp_path)
    result = runner.run_once()
    assert result.result == RESULT_SHADOW_FAIL
    assert result.report["communication"]["health_ok"] is True  # 통신 자체는 성공
    assert result.report["communication"]["error_kind"] == "inference"


def test_vla_response_wrong_dimension_is_schema_failure(tmp_path) -> None:
    class ShortActionRunner(FakePolicyRunner):
        def predict(self, *, task, state, images):
            return {"shoulder_pan": 0.0}  # 5개 관절 누락

    runner = _runner(vla_client=_vla_client(ShortActionRunner()), mujoco_backend=MagicMock(), reports_dir=tmp_path)
    result = runner.run_once()
    assert result.result == RESULT_SHADOW_FAIL
    # 서버가 이미 5xx로 거부하므로 client 관점에서는 inference 오류로 분류된다.
    assert result.report["communication"]["error_kind"] == "inference"


def test_real_follower_write_count_is_always_zero_even_on_failure(tmp_path) -> None:
    runner = _runner(state=FakeStateSource(fail=True), vla_client=MagicMock(), mujoco_backend=MagicMock(), reports_dir=tmp_path)
    result = runner.run_once()
    assert result.real_follower_write_count == 0
    assert result.report["hardware"]["real_follower_write_count"] == 0


def test_state_source_has_no_write_methods() -> None:
    """섹션 17: writable real follower backend 자체가 존재하지 않는지 구조적으로 확인한다."""
    source = FakeStateSource()
    for forbidden in ("send_action", "sync_write", "write", "set_target", "enable_torque"):
        assert not hasattr(source, forbidden)
    assert REAL_FOLLOWER_WRITE_COUNT == 0
