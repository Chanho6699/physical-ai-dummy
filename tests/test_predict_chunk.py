"""Phase C-1A: ``/predict_chunk`` HTTP 계약 + ``SmolVLAPolicyRunner.predict_chunk()`` 내부
동작(select_action이 아니라 predict_action_chunk를 쓰는지, postprocessor가 full chunk에
적용되는지, config.chunk_size를 하드코딩하지 않는지, thread-safety) 검증.

기존 ``/predict``/``PolicyRunner.predict()`` 동작은 이 파일에서 전혀 건드리지 않는다 -
``tests/test_vla_server.py``/``tests/test_vla_client.py``/``tests/test_inprocess_vla_client.py``가
여전히 그대로 통과해야 한다(별도로 재확인, 이 파일은 새 코드 경로만 다룬다).

실제 하드웨어/로봇 접근 없음. GPU/실제 checkpoint도 이 파일에서는 쓰지 않는다(offline
smoke는 별도 스크립트 - 최종 보고서 참고) - 여기는 순수 unit/HTTP 계약 테스트다.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY, JOINT_ORDER
from runtime.desktop.vla_server import FakePolicyRunner, PolicyInferenceError, SmolVLAPolicyRunner, create_app

# ---------------------------------------------------------------------------
# HTTP 계약 (섹션 4/8/9-B,H) - FakePolicyRunner backend
# ---------------------------------------------------------------------------


def _b64_jpeg(shape=(480, 640, 3)) -> str:
    arr = np.zeros(shape, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _good_state() -> dict[str, float]:
    return {joint: float(i) for i, joint in enumerate(JOINT_ORDER)}


def _good_predict_body(**overrides) -> dict:
    body = {
        "session_id": "s1",
        "task": "pick up the cube",
        "sequence": 0,
        "timestamp": 1.0,
        "observation": {
            "state": _good_state(),
            "images": {CAMERA_WORKSPACE_KEY: _b64_jpeg(), CAMERA_WRIST_KEY: _b64_jpeg()},
        },
    }
    body.update(overrides)
    return body


def _client(runner=None) -> TestClient:
    return TestClient(create_app(policy_runner=runner or FakePolicyRunner()))


def test_predict_chunk_returns_200_with_full_chunk() -> None:
    resp = _client().post("/predict_chunk", json=_good_predict_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_size"] == 50  # FakePolicyRunner 기본값
    assert len(data["chunk"]) == 50
    for step in data["chunk"]:
        assert set(step) == set(JOINT_ORDER)
    assert data["chunk_index_spacing_s"] > 0.0
    assert data["backend"] == "fake"
    assert data["model_id"]
    assert data["inference_latency_ms"] >= 0.0
    assert data["session_id"] == "s1"
    assert data["sequence"] == 0


def test_predict_chunk_first_action_matches_predict_semantics() -> None:
    """FakePolicyRunner.predict_chunk()는 chunk[0] = state + joint_offsets*1로 설계했다 -
    predict()의 단일 action(state+joint_offsets)과 정확히 같은 값이어야 한다(semantic parity)."""
    runner = FakePolicyRunner(joint_offsets={"shoulder_pan": 2.5})
    client = _client(runner)
    predict_resp = client.post("/predict", json=_good_predict_body()).json()
    chunk_resp = client.post("/predict_chunk", json=_good_predict_body()).json()
    assert chunk_resp["chunk"][0] == predict_resp["action"]


def test_predict_chunk_missing_state_joint_returns_422() -> None:
    body = _good_predict_body()
    del body["observation"]["state"]["gripper"]
    resp = _client().post("/predict_chunk", json=body)
    assert resp.status_code == 422


def test_predict_chunk_missing_camera_returns_422() -> None:
    body = _good_predict_body()
    del body["observation"]["images"][CAMERA_WRIST_KEY]
    resp = _client().post("/predict_chunk", json=body)
    assert resp.status_code == 422


def test_predict_chunk_malformed_image_base64_returns_422() -> None:
    body = _good_predict_body()
    body["observation"]["images"][CAMERA_WRIST_KEY] = "not-base64!!"
    resp = _client().post("/predict_chunk", json=body)
    assert resp.status_code == 422


def test_predict_chunk_inference_failure_returns_500_with_marker() -> None:
    class FailingRunner(FakePolicyRunner):
        def predict_chunk(self, *, task, state, images):
            raise PolicyInferenceError("chunk 추론이 죽었습니다")

    resp = _client(FailingRunner()).post("/predict_chunk", json=_good_predict_body())
    assert resp.status_code == 500
    assert "[inference_error]" in resp.json()["detail"]


def test_predict_chunk_malformed_policy_output_returns_500() -> None:
    class BadOutputRunner(FakePolicyRunner):
        def predict_chunk(self, *, task, state, images):
            return [{"shoulder_pan": float("nan")}]  # 관절 누락 + NaN

    resp = _client(BadOutputRunner()).post("/predict_chunk", json=_good_predict_body())
    assert resp.status_code == 500


def test_predict_chunk_empty_chunk_returns_500() -> None:
    class EmptyChunkRunner(FakePolicyRunner):
        def predict_chunk(self, *, task, state, images):
            return []

    resp = _client(EmptyChunkRunner()).post("/predict_chunk", json=_good_predict_body())
    assert resp.status_code == 500


def test_predict_chunk_missing_spacing_returns_500_fail_closed() -> None:
    """PolicyRunner 구현이 chunk_index_spacing_s를 None으로 두면(방어적 케이스), 라우트가
    직접 fail-closed로 500을 내야 한다 - 스펙을 어기는 backend가 있어도 클라이언트에
    유효하지 않은 spacing이 새어나가면 안 된다. ``chunk_index_spacing_s``는
    ``FakePolicyRunner``의 dataclass 필드이므로 생성자로 바로 override한다(서브클래스에서
    클래스 속성만 다시 선언하면 dataclass가 생성한 ``__init__``이 그 값을 무시하고 부모
    기본값으로 되돌려 놓는다 - 그 함정을 피하기 위해 직접 override)."""
    runner = FakePolicyRunner(chunk_index_spacing_s=None)
    resp = _client(runner).post("/predict_chunk", json=_good_predict_body())
    assert resp.status_code == 500
    assert "chunk_index_spacing_s" in resp.json()["detail"]


def test_predict_unaffected_by_predict_chunk_addition() -> None:
    """같은 app에 /predict_chunk가 추가된 뒤에도 /predict는 완전히 그대로 동작해야 한다."""
    resp = _client().post("/predict", json=_good_predict_body())
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["action"]) == set(JOINT_ORDER)
    assert "chunk" not in data  # /predict 응답 스키마에 chunk 필드가 섞여 들어가지 않음


# ---------------------------------------------------------------------------
# SmolVLAPolicyRunner 내부 동작 (섹션 2/3 요구사항) - 실제 checkpoint 없이, __init__을
# 우회해 controlled fake policy/pre/postprocessor를 주입한다 (GPU 불필요, 빠름).
# ---------------------------------------------------------------------------


class _CountingFakePolicy:
    """select_action()/predict_action_chunk() 호출 횟수만 기록하는 최소 stub."""

    def __init__(self, chunk_size: int = 4, action_dim: int = 6, chunk_size_mismatch: bool = False):
        self.config = SimpleNamespace(chunk_size=chunk_size)
        self._chunk_size = chunk_size
        self._actual_chunk_size = chunk_size + 1 if chunk_size_mismatch else chunk_size
        self._action_dim = action_dim
        self.select_action_calls = 0
        self.predict_action_chunk_calls = 0

    def select_action(self, batch):
        self.select_action_calls += 1
        return torch.zeros(1, self._action_dim)

    def predict_action_chunk(self, batch, noise=None):
        self.predict_action_chunk_calls += 1
        return torch.zeros(1, self._actual_chunk_size, self._action_dim)


def _bare_runner(policy, *, preprocessor=None, postprocessor=None, dataset_fps: float | None = 30.0) -> SmolVLAPolicyRunner:
    """``SmolVLAPolicyRunner.__init__``(실제 checkpoint 로딩)을 우회하고, 내부 상태만
    직접 채운 인스턴스를 만든다 - GPU/실제 checkpoint 없이 predict()/predict_chunk()의
    로직 자체(어떤 policy 메서드를 호출하는지, lock, shape 검증)만 검증하기 위함."""
    runner = SmolVLAPolicyRunner.__new__(SmolVLAPolicyRunner)
    runner.checkpoint = "/fake/checkpoint"
    runner.model_id = "/fake/checkpoint"
    runner.policy_type = "smolvla"
    runner._device_arg = None
    runner._dataset_fps_override = dataset_fps
    runner._policy = policy
    runner._preprocessor = preprocessor if preprocessor is not None else (lambda batch: batch)
    runner._postprocessor = postprocessor if postprocessor is not None else (lambda x: x)
    runner._device = "cpu"
    runner._load_error = None
    runner._lock = threading.Lock()
    runner.chunk_index_spacing_s = (1.0 / dataset_fps) if dataset_fps else None
    return runner


def _state() -> dict[str, float]:
    return {j: 0.0 for j in JOINT_ORDER}


def _images() -> dict[str, np.ndarray]:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return {CAMERA_WORKSPACE_KEY: img, CAMERA_WRIST_KEY: img}


def test_predict_chunk_calls_predict_action_chunk_not_select_action() -> None:
    policy = _CountingFakePolicy(chunk_size=4)
    runner = _bare_runner(policy)
    result = runner.predict_chunk(task="t", state=_state(), images=_images())
    assert policy.predict_action_chunk_calls == 1
    assert policy.select_action_calls == 0
    assert len(result) == 4


def test_predict_still_calls_select_action_not_predict_action_chunk() -> None:
    """회귀 방지 - predict_chunk 추가가 predict()의 내부 호출 경로를 바꾸지 않았는지."""
    policy = _CountingFakePolicy(chunk_size=4)
    runner = _bare_runner(policy)
    runner.predict(task="t", state=_state(), images=_images())
    assert policy.select_action_calls == 1
    assert policy.predict_action_chunk_calls == 0


def test_predict_chunk_uses_config_chunk_size_not_hardcoded_50() -> None:
    """chunk_size=7인 가짜 checkpoint config에서도(50이 아님) 정확히 7개를 반환해야 한다 -
    50을 어디에도 하드코딩하지 않았음을 증명."""
    policy = _CountingFakePolicy(chunk_size=7)
    runner = _bare_runner(policy)
    result = runner.predict_chunk(task="t", state=_state(), images=_images())
    assert len(result) == 7


def test_predict_chunk_rejects_shape_mismatch_vs_declared_config_chunk_size() -> None:
    """policy가 실제로 반환한 chunk 길이가 config.chunk_size와 다르면 fail-closed로
    PolicyInferenceError를 던져야 한다 - 조용히 잘라내거나 늘리지 않는다."""
    policy = _CountingFakePolicy(chunk_size=7, chunk_size_mismatch=True)  # 실제로는 8개 반환
    runner = _bare_runner(policy)
    with pytest.raises(PolicyInferenceError):
        runner.predict_chunk(task="t", state=_state(), images=_images())


def test_predict_chunk_applies_postprocessor_to_full_chunk_tensor() -> None:
    """postprocessor(raw_chunk)가 실제로 (batch, chunk_size, action_dim) 텐서 전체에
    적용되는지, 그 결과가 최종 반환값에 그대로 반영되는지 확인 - 가짜 postprocessor를
    ``x * 2.0 + 1.0``(unnormalize를 흉내낸 선형 변환)으로 주고, raw(0)가 아니라 변환된
    값(1.0)이 나오는지 본다."""
    policy = _CountingFakePolicy(chunk_size=3, action_dim=6)
    postprocessor_calls: list[tuple[int, ...]] = []

    def fake_postprocessor(x: torch.Tensor) -> torch.Tensor:
        postprocessor_calls.append(tuple(x.shape))
        return x * 2.0 + 1.0

    runner = _bare_runner(policy, postprocessor=fake_postprocessor)
    result = runner.predict_chunk(task="t", state=_state(), images=_images())
    assert postprocessor_calls == [(1, 3, 6)]  # full chunk shape로 호출됐음(단일 action shape 아님)
    for step in result:
        for joint in JOINT_ORDER:
            assert step[joint] == pytest.approx(1.0)  # 0*2+1


def test_predict_chunk_output_is_absolute_degree_percent_not_normalized() -> None:
    """postprocessor가 unnormalize한 값(학습 데이터 실측 범위, 예: shoulder_lift -80deg,
    gripper 45%)을 그대로 돌려주는지 - normalize된 [-1,1]류 값으로 우연히 축소/왜곡되지
    않는지 최소 확인."""
    policy = _CountingFakePolicy(chunk_size=2, action_dim=6)

    def fake_postprocessor(x: torch.Tensor) -> torch.Tensor:
        # degree/percent 절대좌표를 흉내낸 값을 강제로 심어서, "postprocessor를 거친 값이
        # 그대로 dict에 반영되는지"만 확인한다 (실제 unnormalize 수식 자체는 §10 offline
        # smoke에서 real checkpoint로 검증).
        out = torch.zeros_like(x)
        values = torch.tensor([12.3, -80.5, 45.6, -10.0, 3.3, 45.0])
        out[:] = values
        return out

    runner = _bare_runner(policy, postprocessor=fake_postprocessor)
    result = runner.predict_chunk(task="t", state=_state(), images=_images())
    for step in result:
        assert step["shoulder_lift"] == pytest.approx(-80.5)
        assert step["gripper"] == pytest.approx(45.0)


def test_predict_chunk_fails_closed_when_dataset_fps_unknown() -> None:
    policy = _CountingFakePolicy(chunk_size=3)
    runner = _bare_runner(policy, dataset_fps=None)
    assert runner.chunk_index_spacing_s is None
    with pytest.raises(PolicyInferenceError, match="chunk_index_spacing_s"):
        runner.predict_chunk(task="t", state=_state(), images=_images())


def test_predict_still_works_when_dataset_fps_unknown() -> None:
    """chunk_index_spacing_s를 못 구해도 /predict(단일 action)는 전혀 영향받지 않아야
    한다 - fps 문제는 /predict_chunk에만 격리돼야 한다."""
    policy = _CountingFakePolicy(chunk_size=3)
    runner = _bare_runner(policy, dataset_fps=None)
    result = runner.predict(task="t", state=_state(), images=_images())
    assert set(result) == set(JOINT_ORDER)


def test_explicit_dataset_fps_yields_correct_spacing() -> None:
    policy = _CountingFakePolicy(chunk_size=3)
    runner = _bare_runner(policy, dataset_fps=30.0)
    assert runner.chunk_index_spacing_s == pytest.approx(1.0 / 30.0)


# ---------------------------------------------------------------------------
# Thread-safety (섹션 3, 9-I 요구사항) - predict()와 predict_chunk()가 같은 self._policy를
# 동시에 두드릴 때 lock으로 직렬화되는지 + deadlock 없이 둘 다 완료되는지.
# ---------------------------------------------------------------------------


class _SlowFakePolicy:
    """select_action()/predict_action_chunk() 각각 짧게 sleep하며 자신의 [시작,종료]
    구간을 기록한다 - 두 스레드가 동시에 이 policy를 두드려도 구간이 겹치지 않아야
    lock이 제대로 걸린 것이다."""

    def __init__(self, sleep_s: float = 0.05):
        self.config = SimpleNamespace(chunk_size=2)
        self._sleep_s = sleep_s
        self.intervals: list[tuple[str, float, float]] = []
        self._record_lock = threading.Lock()  # 테스트 기록용 - 검증 대상 lock과는 별개

    def _run(self, label: str, result: torch.Tensor) -> torch.Tensor:
        t0 = time.monotonic()
        time.sleep(self._sleep_s)
        t1 = time.monotonic()
        with self._record_lock:
            self.intervals.append((label, t0, t1))
        return result

    def select_action(self, batch):
        return self._run("select_action", torch.zeros(1, 6))

    def predict_action_chunk(self, batch, noise=None):
        return self._run("predict_action_chunk", torch.zeros(1, 2, 6))


def test_predict_and_predict_chunk_serialize_access_to_shared_policy() -> None:
    policy = _SlowFakePolicy(sleep_s=0.05)
    runner = _bare_runner(policy)

    results: dict[str, object] = {}
    errors: list[BaseException] = []

    def _call_predict():
        try:
            results["predict"] = runner.predict(task="t", state=_state(), images=_images())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _call_predict_chunk():
        try:
            results["predict_chunk"] = runner.predict_chunk(task="t", state=_state(), images=_images())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=_call_predict)
    t2 = threading.Thread(target=_call_predict_chunk)
    start = time.monotonic()
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    elapsed = time.monotonic() - start

    # deadlock 없음: 둘 다 timeout 안에 끝났어야 한다.
    assert not t1.is_alive(), "predict()가 5초 안에 끝나지 않음 - deadlock 의심"
    assert not t2.is_alive(), "predict_chunk()가 5초 안에 끝나지 않음 - deadlock 의심"
    assert not errors, f"스레드 안에서 예외 발생: {errors}"
    assert "predict" in results and "predict_chunk" in results

    # 직렬화 확인: lock이 없었다면 sleep(0.05s)이 겹쳐 전체가 ~0.05s 안에 끝날 수 있지만,
    # 직렬화되면 두 sleep이 순차적으로 실행되어 최소 2*0.05s=0.1s는 걸려야 한다.
    assert elapsed >= 0.09, f"두 호출이 직렬화되지 않은 것으로 보임 (elapsed={elapsed:.3f}s)"

    assert len(policy.intervals) == 2
    (_, a_start, a_end), (_, b_start, b_end) = policy.intervals

    def _overlaps(s1, e1, s2, e2) -> bool:
        return s1 < e2 and s2 < e1

    assert not _overlaps(a_start, a_end, b_start, b_end), (
        f"lock이 걸리지 않아 두 policy 호출 구간이 겹침: {policy.intervals}"
    )
