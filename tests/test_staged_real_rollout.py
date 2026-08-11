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
    """매 step마다 미리 정해둔 action을 하나씩 반환한다.

    ``session_reset()``은 기본적으로 항상 성공(``True``)한다 - ``reset_ok_sequence``를 주면
    step(호출 순서)별로 성공/실패를 스크립트할 수 있다(마지막 값이 그 이후 step에도 반복
    적용됨). ``call_log``는 reset/predict가 실제로 호출된 순서를 그대로 기록한다 - "매 step
    반드시 reset -> predict 순서"를 검증하는 데 쓴다.
    """

    actions: list[dict[str, float]]
    reset_ok_sequence: list[bool] | None = None
    calls: list[dict] = field(default_factory=list)
    reset_calls: list[dict] = field(default_factory=list)
    call_log: list[str] = field(default_factory=list)

    def session_reset(self, *, session_id, task) -> bool:
        idx = len(self.reset_calls)
        self.reset_calls.append({"session_id": session_id, "task": task})
        self.call_log.append(f"reset:{session_id}")
        if self.reset_ok_sequence is None:
            return True
        return self.reset_ok_sequence[idx] if idx < len(self.reset_ok_sequence) else self.reset_ok_sequence[-1]

    def predict(self, *, session_id, task, sequence, state, images):
        self.calls.append({"sequence": sequence, "state": dict(state), "task": task})
        self.call_log.append(f"predict:{session_id}:{sequence}")
        action = self.actions[sequence] if sequence < len(self.actions) else self.actions[-1]
        return PredictResult(
            ok=True, session_id=session_id, sequence=sequence, action=dict(action), action_schema_valid=True,
            action_invalid_reason=None, model_id="fake", backend="fake", inference_latency_ms=1.0,
            request_latency_ms=1.0, error_kind=None,
        )


class RaisingResetVLAClient:
    """session_reset()이 항상 예외를 던지는 client - fail-closed 경로 검증용.

    predict()가 호출되면 그 자체로 fail-closed 위반이므로 즉시 AssertionError를 던진다 -
    "reset 실패했는데 predict가 실행됐다"는 걸 테스트 실패로 직접 드러낸다."""

    def __init__(self) -> None:
        self.reset_calls: list[dict] = []
        self.predict_calls: list[dict] = []

    def session_reset(self, *, session_id, task) -> bool:
        self.reset_calls.append({"session_id": session_id, "task": task})
        raise RuntimeError("session_reset 통신 실패 (시뮬레이션)")

    def predict(self, *, session_id, task, sequence, state, images):
        self.predict_calls.append({"session_id": session_id, "sequence": sequence})
        raise AssertionError("session_reset이 실패했는데 predict가 호출됐다 - fail-closed 위반")


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
    assert len(vla.reset_calls) == 0  # 관측 실패 시 reset도 시도하지 않는다
    assert len(vla.calls) == 0  # 관측 실패 시 추론도 시도하지 않는다
    assert len(follower.send_action_calls) == 0


# ---------------------------------------------------------------------------
# 2026-08 receding-horizon 수정 (SmolVLA 50-step stale action queue 문제) - 매 step
# session_reset -> predict 순서가 강제되는지, reset 실패 시 fail-closed로 멈추는지 검증.
# ---------------------------------------------------------------------------


def test_stage3_ten_steps_calls_session_reset_and_predict_exactly_ten_times_each():
    """Stage 3, steps=10, 전부 ACCEPT면 session_reset 10회 + predict 10회가 나와야 한다 -
    reset이 predict 앞에서 매 step 빠짐없이 호출된다는 요구사항의 핵심 회귀."""
    actions = [{**_neutral(), "shoulder_pan": 1.0} for _ in range(10)]  # 매 step 동일한 작은 delta
    runner, follower, writer = _make_runner(actions, max_write_count=10)

    result = runner.run_stage(stage=3, max_steps=10)

    assert result.verdict == PASS
    assert len(result.steps) == 10
    vla: ScriptedVLAClient = runner._vla_client  # type: ignore[attr-defined]
    assert len(vla.reset_calls) == 10
    assert len(vla.calls) == 10
    for step in result.steps:
        assert step.session_reset_ok is True
        assert step.fresh_inference is True


def test_session_reset_always_called_immediately_before_predict_each_step():
    """호출 순서 자체를 검증한다 - 매 step마다 reset이 그 step의 predict보다 먼저 나와야
    하고, 다른 step의 predict와 뒤섞이면 안 된다."""
    actions = [{**_neutral(), "shoulder_pan": float(i % 2)} for i in range(4)]  # 0/1 deg 반복 (전부 ACCEPT)
    runner, follower, writer = _make_runner(actions, max_write_count=4)

    result = runner.run_stage(stage=2, max_steps=4)

    assert result.verdict == PASS
    vla: ScriptedVLAClient = runner._vla_client  # type: ignore[attr-defined]
    expected = []
    for i in range(4):
        expected.append(f"reset:staged-{i}")
        expected.append(f"predict:staged-{i}:{i}")
    assert vla.call_log == expected


def test_session_reset_exception_is_fail_closed_no_predict_no_write():
    """session_reset이 예외를 던지면 그 step은 predict/write를 절대 시도하지 않고 즉시
    stage가 중단돼야 한다 (fail-closed) - RaisingResetVLAClient의 predict()는 호출되면 그
    자체로 AssertionError를 던지므로, 이 테스트가 통과한다는 것 자체가 predict가 한 번도
    호출되지 않았다는 증거다."""
    follower = FakeFollower(_neutral())
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=5)
    writer.connect()
    vla = RaisingResetVLAClient()
    runner = StagedRealRolloutRunner(
        camera_source=FakeCameraSource(), state_source=FakeStateSource(follower.state), vla_client=vla,
        safety_gate=SafetyGate(_safety_config()), writer=writer, task="t", checkpoint_label="fake",
        post_write_settle_s=0.0,
    )

    result = runner.run_stage(stage=2, max_steps=3)

    assert result.verdict == ABNORMAL_STOPPED
    assert len(result.steps) == 1  # 첫 step에서 즉시 중단, 나머지 step은 시도조차 안 함
    step = result.steps[0]
    assert step.session_reset_ok is False
    assert "session_reset" in step.step_error
    assert step.written is False
    assert step.fresh_inference is False
    assert result.real_follower_write_count == 0
    assert len(vla.reset_calls) == 1
    assert len(vla.predict_calls) == 0  # predict가 단 한 번도 호출되지 않았음을 직접 확인
    assert len(follower.send_action_calls) == 0


def test_session_reset_ok_false_is_also_fail_closed():
    """session_reset이 예외 없이 그냥 ``ok=False``를 반환하는 경우도 동일하게 fail-closed여야
    한다 - 예외 경로와 반환값 경로를 둘 다 커버한다."""
    actions = [{**_neutral(), "shoulder_pan": 1.0}, {**_neutral(), "shoulder_pan": 2.0}]
    runner, follower, writer = _make_runner(actions, max_write_count=5)
    vla: ScriptedVLAClient = runner._vla_client  # type: ignore[attr-defined]
    vla.reset_ok_sequence = [True, False]  # 1번째 step은 성공, 2번째 step부터 실패

    result = runner.run_stage(stage=2, max_steps=3)

    assert result.verdict == ABNORMAL_STOPPED
    assert len(result.steps) == 2
    assert result.steps[0].session_reset_ok is True
    assert result.steps[0].written is True
    assert result.steps[1].session_reset_ok is False
    assert result.steps[1].written is False
    assert result.real_follower_write_count == 1  # 실패 시점 이후로는 write가 전혀 없다
    assert len(vla.calls) == 1  # 2번째 step은 predict까지 가지 않았다
    assert len(follower.send_action_calls) == 1


def test_would_clamp_still_blocks_write_under_new_reset_predict_sequence():
    """reset->predict 시퀀스를 넣어도 WOULD_CLAMP은 여전히 write를 막아야 한다 (기존 불변식
    유지 회귀) - 20deg는 max_step=5.0(WOULD_CLAMP 경계) 초과, gross(25.0) 미만."""
    action = {**_neutral(), "shoulder_pan": 20.0}
    runner, follower, writer = _make_runner([action])
    vla: ScriptedVLAClient = runner._vla_client  # type: ignore[attr-defined]

    result = runner.run_stage(stage=1, max_steps=1)

    assert result.steps[0].safety_decision == "WOULD_CLAMP"
    assert result.steps[0].session_reset_ok is True  # reset/predict 자체는 정상적으로 일어남
    assert result.steps[0].fresh_inference is True
    assert result.verdict == ABNORMAL_STOPPED
    assert not result.steps[0].written
    assert len(follower.send_action_calls) == 0
    assert len(vla.reset_calls) == 1
    assert len(vla.calls) == 1


def test_reject_still_blocks_write_under_new_reset_predict_sequence():
    """REJECT도 마찬가지로 기존 불변식(write 절대 안 함)이 새 시퀀스에서도 유지돼야 한다."""
    action = {**_neutral(), "shoulder_pan": 999.0}
    runner, follower, writer = _make_runner([action])

    result = runner.run_stage(stage=1, max_steps=1)

    assert result.steps[0].safety_decision == "REJECT"
    assert result.steps[0].session_reset_ok is True
    assert result.verdict == ABNORMAL_STOPPED
    assert not result.steps[0].written
    assert len(follower.send_action_calls) == 0


def test_first_non_accept_still_halts_immediately_under_new_sequence():
    """첫 non-ACCEPT에서 즉시 중단하는 기존 불변식이 reset 삽입 후에도 유지되는지 확인 -
    3번째 step은 reset조차 시도되지 않아야 한다."""
    accept_action = {**_neutral(), "shoulder_pan": 1.0}
    reject_action = {**_neutral(), "shoulder_pan": 999.0}
    another_accept = {**_neutral(), "shoulder_pan": 2.0}
    runner, follower, writer = _make_runner([accept_action, reject_action, another_accept], max_write_count=10)
    vla: ScriptedVLAClient = runner._vla_client  # type: ignore[attr-defined]

    result = runner.run_stage(stage=2, max_steps=3)

    assert result.verdict == ABNORMAL_STOPPED
    assert len(result.steps) == 2
    assert len(vla.reset_calls) == 2  # 3번째 step은 reset도 predict도 시도 안 함
    assert len(vla.calls) == 2


# ---------------------------------------------------------------------------
# 2026-08 Phase B: closed-loop latency 계측 필드 (camera/state/safety_gate/write) 회귀.
# 이 필드들은 Fake 경로에서는 항상 매우 작은(거의 0에 가까운) 값만 나온다 - 여기서
# 검증하는 건 "실제 hardware-bound 값"이 아니라 "계측 코드 자체가 매 경로에서 올바르게
# 채워지는지"다 (None이어야 할 때 None인지, 채워져야 할 때 음수가 아닌 float인지).
# ---------------------------------------------------------------------------


def test_latency_fields_populated_on_accept_and_write():
    action = {**_neutral(), "shoulder_pan": 1.0}
    runner, follower, writer = _make_runner([action], max_write_count=1)

    result = runner.run_stage(stage=1, max_steps=1)

    step = result.steps[0]
    assert step.safety_decision == "ACCEPT"
    assert step.written is True
    for field_name in (
        "camera_capture_latency_ms", "state_read_latency_ms", "session_reset_latency_ms",
        "safety_gate_latency_ms", "write_latency_ms",
    ):
        value = getattr(step, field_name)
        assert value is not None, f"{field_name}가 ACCEPT+write 경로에서 None이면 안 됨"
        assert value >= 0.0, f"{field_name}가 음수: {value}"


def test_camera_and_state_latency_recorded_even_when_session_reset_fails():
    """reset이 실패해도(fail-closed로 predict/write는 안 하지만), 이미 성공한 관측 단계의
    camera/state latency는 진단용으로 그대로 남아있어야 한다."""
    follower = FakeFollower(_neutral())
    writer = StagedFollowerArmedWriter(follower=follower, max_write_count=5)
    writer.connect()
    vla = RaisingResetVLAClient()
    runner = StagedRealRolloutRunner(
        camera_source=FakeCameraSource(), state_source=FakeStateSource(follower.state), vla_client=vla,
        safety_gate=SafetyGate(_safety_config()), writer=writer, task="t", checkpoint_label="fake",
        post_write_settle_s=0.0,
    )

    result = runner.run_stage(stage=1, max_steps=1)

    step = result.steps[0]
    assert step.session_reset_ok is False
    assert step.camera_capture_latency_ms is not None
    assert step.camera_capture_latency_ms >= 0.0
    assert step.state_read_latency_ms is not None
    assert step.state_read_latency_ms >= 0.0
    # reset 이후 단계(추론/safety/write)는 아예 시도되지 않았으므로 전부 None이어야 한다.
    assert step.predict_inference_latency_ms is None
    assert step.safety_gate_latency_ms is None
    assert step.write_latency_ms is None


def test_safety_gate_latency_recorded_but_write_latency_none_on_would_clamp():
    """WOULD_CLAMP은 write를 하지 않으므로 write_latency_ms는 None이어야 하지만,
    safety_gate.evaluate() 자체는 호출됐으므로 safety_gate_latency_ms는 채워져야 한다."""
    action = {**_neutral(), "shoulder_pan": 20.0}  # WOULD_CLAMP (max_step=5.0, gross=25.0)
    runner, follower, writer = _make_runner([action])

    result = runner.run_stage(stage=1, max_steps=1)

    step = result.steps[0]
    assert step.safety_decision == "WOULD_CLAMP"
    assert step.safety_gate_latency_ms is not None
    assert step.safety_gate_latency_ms >= 0.0
    assert step.write_latency_ms is None
    assert step.written is False
