"""runtime/laptop/inprocess_vla_client.py 테스트 - GPU/checkpoint 없이 fake backend로 검증.

실제 checkpoint로 하는 end-to-end 검증(진짜 SmolVLAPolicyRunner)은
``tests/test_shadow_mode_runner.py``의 ``@pytest.mark.slow`` 테스트가 담당한다. 여기서는
``InProcessSmolVLAClient``가 ``VLAHttpClient``와 동일한 duck-typed 계약을 지키는지만
``SmolVLAPolicyRunner``를 monkeypatch로 대체해 빠르게 검증한다.
"""

from __future__ import annotations

import pytest

import runtime.laptop.inprocess_vla_client as inprocess_vla_client
from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.inprocess_vla_client import InProcessSmolVLAClient, InProcessVLAClientError


class FakePolicyRunner:
    """SmolVLAPolicyRunner의 공개 인터페이스만 흉내 낸 stub - GPU/체크포인트 불필요."""

    backend_name = "smolvla"

    def __init__(self, checkpoint, *, policy_type="smolvla", device=None, fail_load=False, fail_predict=False):
        self.checkpoint = checkpoint
        self.model_id = checkpoint
        self.reset_calls: list[tuple[str, str]] = []
        self._fail_predict = fail_predict
        self._load_error = "가짜 로딩 실패" if fail_load else None

    def is_ready(self) -> bool:
        return self._load_error is None

    def reset(self, *, session_id: str, task: str) -> None:
        self.reset_calls.append((session_id, task))

    def predict(self, *, task, state, images):
        if self._fail_predict:
            from runtime.desktop.vla_server import PolicyInferenceError

            raise PolicyInferenceError("가짜 추론 실패")
        return {joint: state[joint] for joint in JOINT_ORDER}

    def device_label(self) -> str:
        return "cpu (fake)"


@pytest.fixture(autouse=True)
def _patch_policy_runner(monkeypatch):
    monkeypatch.setattr(inprocess_vla_client, "SmolVLAPolicyRunner", FakePolicyRunner)


def test_construction_raises_when_load_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        inprocess_vla_client,
        "SmolVLAPolicyRunner",
        lambda checkpoint, **kw: FakePolicyRunner(checkpoint, fail_load=True),
    )
    with pytest.raises(InProcessVLAClientError):
        InProcessSmolVLAClient(checkpoint="/fake/checkpoint")


def test_check_health_ok() -> None:
    client = InProcessSmolVLAClient(checkpoint="/fake/checkpoint")
    health = client.check_health()
    assert health.ok is True
    assert health.model_id == "/fake/checkpoint"
    assert health.backend == "smolvla"


def test_session_reset_forwards_to_policy_runner() -> None:
    client = InProcessSmolVLAClient(checkpoint="/fake/checkpoint")
    ok = client.session_reset(session_id="s1", task="pick")
    assert ok is True
    assert client._policy_runner.reset_calls == [("s1", "pick")]


def test_predict_returns_action_matching_state() -> None:
    client = InProcessSmolVLAClient(checkpoint="/fake/checkpoint")
    state = {joint: float(i) for i, joint in enumerate(JOINT_ORDER)}
    images = {"observation.images.workspace": object(), "observation.images.wrist": object()}
    result = client.predict(session_id="s1", task="pick", sequence=0, state=state, images=images)
    assert result.ok is True
    assert result.action == state
    assert result.error_kind is None
    assert result.request_latency_ms >= 0.0


def test_predict_inference_failure_is_reported_as_inference_error(monkeypatch) -> None:
    monkeypatch.setattr(
        inprocess_vla_client,
        "SmolVLAPolicyRunner",
        lambda checkpoint, **kw: FakePolicyRunner(checkpoint, fail_predict=True),
    )
    client = InProcessSmolVLAClient(checkpoint="/fake/checkpoint")
    result = client.predict(
        session_id="s1", task="pick", sequence=0, state={j: 0.0 for j in JOINT_ORDER}, images={}
    )
    assert result.ok is False
    assert result.error_kind == "inference"


def test_action_ack_always_true() -> None:
    client = InProcessSmolVLAClient(checkpoint="/fake/checkpoint")
    assert client.action_ack(session_id="s1", sequence=0, executed=True, backend="realistic_mujoco") is True


def test_close_is_noop_and_context_manager_works() -> None:
    with InProcessSmolVLAClient(checkpoint="/fake/checkpoint") as client:
        assert client.check_health().ok is True


def test_no_write_methods_exist() -> None:
    """섹션 17과 동일한 구조적 불변식 - in-process client도 write 메서드가 없어야 한다."""
    client = InProcessSmolVLAClient(checkpoint="/fake/checkpoint")
    for forbidden in ("send_action", "sync_write", "write", "set_target", "enable_torque"):
        assert not hasattr(client, forbidden)
