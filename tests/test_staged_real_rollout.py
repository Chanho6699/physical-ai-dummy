from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from hardware.safety.staged_follower_writer import JOINT_ORDER, StagedFollowerArmedWriter, WriteAttemptResult
from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY
from runtime.laptop.camera_source import CameraFrame
from runtime.laptop.follower_state_source import FollowerStateSnapshot
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
from runtime.laptop.staged_real_rollout import ABNORMAL_STOPPED, PASS, StagedRealRolloutRunner
from runtime.laptop.vla_client import PredictResult


def _safety_config(*, max_step: float = 5.0) -> SafetyGateConfig:
    joint_range = {j: ((0.0, 100.0) if j == "gripper" else (-90.0, 90.0)) for j in JOINT_ORDER}
    max_step_map = {j: max_step for j in JOINT_ORDER}
    return SafetyGateConfig(joint_range_deg=joint_range, max_step_deg=max_step_map)


def _neutral() -> dict[str, float]:
    return {j: 0.0 for j in JOINT_ORDER}


class FakeCameraSource:
    def capture_all(self) -> dict[str, CameraFrame]:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        now = time.time()
        return {
            CAMERA_WORKSPACE_KEY: CameraFrame(image_rgb=img, captured_at_wall=now, width=640, height=480),
            CAMERA_WRIST_KEY: CameraFrame(image_rgb=img, captured_at_wall=now, width=640, height=480),
        }


class FakeStateSource:
    """실제 시스템에서는 이 read-only state source와 writer(``StagedFollowerArmedWriter``)가
    같은 물리 로봇을 보므로 항상 서로 일치한다 - 테스트에서도 그 성질을 재현하기 위해
    dict를 **참조로 공유**한다(복사하지 않음), 별도 dict를 주면 writer가 write한 뒤에도
    이 state source가 낡은 값을 계속 돌려주는 비현실적인 상황이 생긴다."""

    def __init__(self, positions: dict[str, float]):
        self.positions = positions  # 참조 공유 - follower.state와 같은 객체를 넘겨야 한다

    def read(self) -> FollowerStateSnapshot:
        return FollowerStateSnapshot(
            positions_deg=dict(self.positions), read_at_monotonic=time.monotonic(), read_at_wall=time.time()
        )


@dataclass
class ScriptedVLAClient:
    """매 step마다 미리 정해둔 action을 하나씩 반환한다."""

    actions: list[dict[str, float]]
    calls: list[dict] = field(default_factory=list)

    def predict(self, *, session_id, task, sequence, state, images):
        self.calls.append({"sequence": sequence, "state": dict(state), "task": task})
        action = self.actions[sequence] if sequence < len(self.actions) else self.actions[-1]
        return PredictResult(
            ok=True, session_id=session_id, sequence=sequence, action=dict(action), action_schema_valid=True,
            action_invalid_reason=None, model_id="fake", backend="fake", inference_latency_ms=1.0,
            request_latency_ms=1.0, error_kind=None,
        )


class FakeFollower:
    """staged_follower_writer.py의 FakeFollower와 동일한 duck type, 여기서는 writer 계층
    없이 StagedFollowerArmedWriter를 그대로 통과시켜 write budget 강제까지 함께 검증한다."""

    def __init__(self, initial_state: dict[str, float]):
        self.state = dict(initial_state)
        self.send_action_calls: list[dict] = []

    def connect(self):
        pass

    def get_observation(self):
        return {f"{j}.pos": v for j, v in self.state.items()}

    def send_action(self, action: dict) -> dict:
        self.send_action_calls.append(dict(action))
        for k, v in action.items():
            self.state[k.removesuffix(".pos")] = v
        return dict(action)

    def disconnect(self):
        pass


def _make_runner(actions: list[dict[str, float]], *, initial_state=None, max_write_count=10):
    initial_state = initial_state or _neutral()
    follower = FakeFollower(initial_state)
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=max_write_count)
    writer.connect()
    runner = StagedRealRolloutRunner(
        camera_source=FakeCameraSource(), state_source=FakeStateSource(follower.state),
        vla_client=ScriptedVLAClient(actions=actions), safety_gate=SafetyGate(_safety_config()),
        writer=writer, task="Pick up the cube and drop it into the bin.", checkpoint_label="fake-checkpoint",
        post_write_settle_s=0.0,
    )
    return runner, follower, writer


def test_stage1_single_accept_action_writes_exactly_once_and_passes():
    # small delta well within max_step=5.0 -> ACCEPT
    action = {**_neutral(), "shoulder_pan": 1.0}
    runner, follower, writer = _make_runner([action], max_write_count=1)

    result = runner.run_stage(stage=1, max_steps=1)

    assert result.verdict == PASS
    assert result.real_follower_write_count == 1
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.safety_decision == "ACCEPT"
    assert step.written
    assert step.before_state_deg == _neutral()
    assert step.after_state_deg["shoulder_pan"] == 1.0
    assert step.delta_deg["shoulder_pan"] == 1.0
    assert len(follower.send_action_calls) == 1


def test_reject_action_is_never_written():
    # gross violation -> REJECT (way beyond range and gross step multiplier)
    action = {**_neutral(), "shoulder_pan": 999.0}
    runner, follower, writer = _make_runner([action])

    result = runner.run_stage(stage=1, max_steps=1)

    assert result.verdict == ABNORMAL_STOPPED
    assert result.real_follower_write_count == 0
    assert result.steps[0].safety_decision == "REJECT"
    assert not result.steps[0].written
    assert len(follower.send_action_calls) == 0  # 핵심 불변식: REJECT는 send_action 근처에도 못 감


def test_would_clamp_action_is_never_written():
    # delta of 20deg > max_step(5) but < gross(25) -> WOULD_CLAMP, not REJECT
    action = {**_neutral(), "shoulder_pan": 20.0}
    runner, follower, writer = _make_runner([action])

    result = runner.run_stage(stage=1, max_steps=1)

    assert result.steps[0].safety_decision == "WOULD_CLAMP"
    assert result.verdict == ABNORMAL_STOPPED
    assert not result.steps[0].written
    assert len(follower.send_action_calls) == 0
    # 진단용으로 "would-be" safe_action은 기록하되 실제로 쓰진 않았다.
    assert result.steps[0].would_be_safe_action_deg is not None


def test_stage_halts_immediately_on_first_non_accept_does_not_continue():
    accept_action = {**_neutral(), "shoulder_pan": 1.0}
    reject_action = {**_neutral(), "shoulder_pan": 999.0}
    another_accept = {**_neutral(), "shoulder_pan": 2.0}
    runner, follower, writer = _make_runner([accept_action, reject_action, another_accept], max_write_count=10)

    result = runner.run_stage(stage=2, max_steps=3)

    assert result.verdict == ABNORMAL_STOPPED
    assert len(result.steps) == 2  # 세 번째 step은 시도조차 안 함
    assert result.steps[0].safety_decision == "ACCEPT"
    assert result.steps[1].safety_decision == "REJECT"
    assert len(follower.send_action_calls) == 1  # 첫 step만 write됨


def test_multi_step_stage_all_accept_passes_and_records_every_delta():
    actions = [{**_neutral(), "shoulder_pan": float(i + 1)} for i in range(4)]
    runner, follower, writer = _make_runner(actions, max_write_count=4)

    result = runner.run_stage(stage=2, max_steps=4)

    assert result.verdict == PASS
    assert result.real_follower_write_count == 4
    assert len(follower.send_action_calls) == 4
    for i, step in enumerate(result.steps):
        assert step.written
        assert step.safety_decision == "ACCEPT"
        assert step.delta_deg["shoulder_pan"] == pytest.approx(1.0)  # 매 step 1deg씩 이동


def test_write_budget_exhaustion_mid_stage_halts_stage():
    actions = [{**_neutral(), "shoulder_pan": float(i + 1)} for i in range(3)]
    # budget=2 인데 3 step을 요청 -> 3번째 step에서 budget 소진으로 write 실패, stage 중단.
    runner, follower, writer = _make_runner(actions, max_write_count=2)

    result = runner.run_stage(stage=2, max_steps=3)

    assert result.verdict == ABNORMAL_STOPPED
    assert len(result.steps) == 3
    assert result.steps[0].written and result.steps[1].written
    assert not result.steps[2].written
    assert result.real_follower_write_count == 2
    assert len(follower.send_action_calls) == 2


def test_camera_read_failure_halts_before_any_inference_or_write():
    class FailingCameraSource:
        def capture_all(self):
            raise RuntimeError("카메라 없음 (시뮬레이션)")

    follower = FakeFollower(_neutral())
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=1)
    writer.connect()
    vla = ScriptedVLAClient(actions=[{**_neutral(), "shoulder_pan": 1.0}])
    runner = StagedRealRolloutRunner(
        camera_source=FailingCameraSource(), state_source=FakeStateSource(_neutral()), vla_client=vla,
        safety_gate=SafetyGate(_safety_config()), writer=writer, task="t", checkpoint_label="fake",
    )

    result = runner.run_stage(stage=1, max_steps=1)
    assert result.verdict == ABNORMAL_STOPPED
    assert result.real_follower_write_count == 0
    assert len(vla.calls) == 0  # 관측 실패 시 추론도 시도하지 않는다
    assert len(follower.send_action_calls) == 0
