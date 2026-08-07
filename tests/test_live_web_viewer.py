"""simulation/mujoco/live_web_viewer.py 테스트.

실제 HTTP 서버를 ephemeral 포트(0)에 띄워 진짜 GET 요청으로 검증한다 (mock으로 HTTP 계층
자체를 대체하면 "브라우저에서 실제로 응답하는지"를 검증하는 의미가 없어지기 때문). 노트북
서버는 ``test_remote_diagnostic.py``와 동일한 패턴의 스크립트 가능한 가짜 클라이언트로
대체한다 - 실제 네트워크는 쓰지 않는다.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

from simulation.mujoco.live_web_viewer import (
    COMMAND_SOURCE_FOLLOWER_SAFE,
    COMMAND_SOURCE_RAW_LEADER,
    LiveWebViewer,
    LiveWebViewerError,
    WebViewerArgs,
    create_http_server,
)
from simulation.mujoco.remote_state_client import JOINT_NAMES, ArmStateView, HealthState, RemoteState, RemoteStateError
from simulation.mujoco.safety_event_tracker import SafetyEventTrackerConfig


def _positions(**overrides) -> dict[str, float]:
    base = {name: 0.0 for name in JOINT_NAMES}
    base.update(overrides)
    return base


def _arm(positions: dict[str, float], *, connected: bool = True, stale: bool = False, age_ms: float = 10.0) -> ArmStateView:
    return ArmStateView(
        connected=connected, positions_deg=dict(positions), raw_ticks=None, stale=stale, age_ms=age_ms, valid=True, invalid_reason=None
    )


def _state(leader: dict[str, float], follower: dict[str, float], *, sequence: int = 1, mode: str = "read_only", leader_kwargs=None) -> RemoteState:
    leader_kwargs = leader_kwargs or {}
    return RemoteState(
        raw_timestamp=1000.0,
        sequence=sequence,
        mode=mode,
        leader=_arm(leader, **leader_kwargs),
        follower=_arm(follower),
        difference_deg={k: leader[k] - follower[k] for k in leader},
        warnings=[],
        received_at_monotonic=time.monotonic(),
        received_at_wall=time.time(),
        network_latency_ms=5.0,
    )


def _health(**overrides) -> HealthState:
    base = dict(
        status="ok", mode="read_only", leader_connected=True, follower_connected=True, write_enabled=False, timestamp=1000.0, errors=[]
    )
    base.update(overrides)
    return HealthState(**base)


class _ScriptedClient:
    """리스트에 준 상태를 순서대로(끝에 도달하면 마지막 값을 반복) 반환하는 가짜 클라이언트."""

    def __init__(self, health: HealthState, states: list) -> None:
        self._health = health
        self._states = states
        self.calls = 0
        self.closed = False

    def check_health(self) -> HealthState:
        return self._health

    def get_state(self) -> RemoteState:
        idx = min(self.calls, len(self._states) - 1)
        item = self._states[idx]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


def _make_viewer(client, *, joints=("wrist_flex",), fps=30.0, rate_hz=50.0, stale_after_ms=200.0, **overrides) -> LiveWebViewer:
    kwargs = dict(
        server_url="http://laptop.local:8001",
        joints=joints,
        host="127.0.0.1",
        port=0,
        fps=fps,
        rate_hz=rate_hz,
        stale_after_ms=stale_after_ms,
        frame_width=64,
        frame_height=48,
    )
    kwargs.update(overrides)
    args = WebViewerArgs(**kwargs)
    return LiveWebViewer(args, client=client)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def test_preflight_raises_on_connection_failure():
    class _FailingClient(_ScriptedClient):
        def check_health(self):
            raise RemoteStateError("연결 실패 (테스트)")

    viewer = _make_viewer(_FailingClient(_health(), [_state(_positions(), _positions())]))
    with pytest.raises(LiveWebViewerError):
        viewer.preflight()


def test_preflight_raises_when_mode_not_read_only():
    viewer = _make_viewer(_ScriptedClient(_health(mode="teleop"), [_state(_positions(), _positions())]))
    with pytest.raises(LiveWebViewerError):
        viewer.preflight()


def test_preflight_raises_when_write_enabled_true():
    viewer = _make_viewer(_ScriptedClient(_health(write_enabled=True), [_state(_positions(), _positions())]))
    with pytest.raises(LiveWebViewerError):
        viewer.preflight()


def test_preflight_succeeds_with_good_health():
    viewer = _make_viewer(_ScriptedClient(_health(), [_state(_positions(), _positions())]))
    viewer.preflight()  # 예외 없이 끝나야 함


# ---------------------------------------------------------------------------
# render 루프 (HTTP 없이 LiveWebViewer 자체만)
# ---------------------------------------------------------------------------


def _start_and_wait_for_frame(viewer: LiveWebViewer, timeout: float = 5.0) -> bytes:
    viewer.preflight()
    viewer.start()
    deadline = time.monotonic() + timeout
    jpeg = None
    while time.monotonic() < deadline:
        jpeg, gen = viewer.frames.snapshot()
        if jpeg is not None:
            break
        time.sleep(0.05)
    assert jpeg is not None, "제한 시간 안에 프레임이 생성되지 않았습니다"
    return jpeg


def test_render_loop_produces_nonblank_jpeg_frame():
    import io

    from PIL import Image
    import numpy as np

    client = _ScriptedClient(_health(), [_state(_positions(wrist_flex=10.0), _positions(wrist_flex=0.0))])
    viewer = _make_viewer(client)
    try:
        jpeg = _start_and_wait_for_frame(viewer)
        arr = np.array(Image.open(io.BytesIO(jpeg)))
        assert arr.shape == (48, 64, 3)
        assert float(arr.std()) > 1.0  # 단색(블랭크)이 아님
    finally:
        viewer.stop()


def test_status_reflects_leader_and_target():
    client = _ScriptedClient(_health(), [_state(_positions(wrist_flex=15.0), _positions(wrist_flex=1.0))])
    viewer = _make_viewer(client)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.3)
        status = viewer.status.to_dict()
        assert status["connected"] is True
        assert status["stale"] is False
        joint = status["joints"]["wrist_flex"]
        assert joint["leader_deg"] == pytest.approx(15.0)
        assert joint["follower_deg"] == pytest.approx(1.0)
        assert joint["mujoco_target_deg"] == pytest.approx(15.0, abs=0.5)
        # 초기 target(0deg)에서 15deg로 한 프레임만에 뛴 것 자체가 max_joint_delta_per_frame
        # WARN을 유발할 수 있다 (safety_checks.py 재사용 - 이 테스트가 새로 정의하지 않음).
        # 여기서 확인하려는 것은 "range 위반이 아니다(BLOCKED 아님)"이지 delta WARN 여부가 아니다.
        assert joint["safety_status"] in ("PASS", "WARN")
    finally:
        viewer.stop()


def test_out_of_range_target_is_blocked_and_previous_target_held():
    states = [
        _state(_positions(wrist_flex=0.0), _positions(wrist_flex=0.0)),
        _state(_positions(wrist_flex=999.0), _positions(wrist_flex=1.0)),  # 명백히 range 초과
    ]
    client = _ScriptedClient(_health(), states)
    viewer = _make_viewer(client, fps=40.0, rate_hz=80.0)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.4)  # 두 번째(BLOCKED) state까지 반영될 시간을 준다
        status = viewer.status.to_dict()
        joint = status["joints"]["wrist_flex"]
        assert joint["safety_status"] == "BLOCKED"
        # target이 999deg 근처로 튀지 않아야 한다 (직전 안전 target 유지, clamp 아님)
        assert joint["mujoco_target_deg"] is None or joint["mujoco_target_deg"] < 150.0
    finally:
        viewer.stop()


def test_stale_snapshot_freezes_target_but_keeps_rendering():
    good = _state(_positions(wrist_flex=5.0), _positions(wrist_flex=0.0))
    stale = _state(_positions(wrist_flex=80.0), _positions(wrist_flex=0.0), leader_kwargs={"stale": True})
    client = _ScriptedClient(_health(), [good, stale])
    viewer = _make_viewer(client, fps=40.0, rate_hz=80.0)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.4)
        status = viewer.status.to_dict()
        assert status["stale"] is True
        joint = status["joints"]["wrist_flex"]
        # stale 상태에서는 leader=80이 와도 target이 5 근처에 머물러야 한다 (갱신 중지)
        assert joint["mujoco_target_deg"] is None or joint["mujoco_target_deg"] < 20.0
        # 렌더링 자체(프레임 생성)는 계속되어야 한다
        jpeg, generation = viewer.frames.snapshot()
        assert jpeg is not None and generation > 0
    finally:
        viewer.stop()


def test_connection_error_marks_disconnected_but_keeps_rendering():
    client = _ScriptedClient(_health(), [_state(_positions(), _positions()), RemoteStateError("네트워크 끊김 (테스트)")])
    viewer = _make_viewer(client, fps=40.0, rate_hz=80.0)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.4)
        # 마지막 프레임은 여전히 생성되어 있어야 한다
        jpeg, _ = viewer.frames.snapshot()
        assert jpeg is not None
    finally:
        viewer.stop()


def test_mode_violation_permanently_stops_updates():
    """network/render 스레드가 분리돼 있으므로(요구사항: 최신 snapshot만 사용), network
    폴링을 render보다 훨씬 느리게 해서 render 루프가 violation 샘플을 놓치지 않게 만든다
    (그렇지 않으면 violation 샘플이 다음 폴링에 덮여 쓰여 관측되지 못할 수 있다 - 이건 "최신
    snapshot만 본다"는 설계상 당연한 특성이지, 이 테스트가 노리는 지점이 아니다)."""
    good = _state(_positions(wrist_flex=1.0), _positions(wrist_flex=1.0))
    violation = _state(_positions(wrist_flex=50.0), _positions(wrist_flex=1.0), mode="teleop")
    recovered = _state(_positions(wrist_flex=1.0), _positions(wrist_flex=1.0))  # mode 복구돼도 fatal은 유지되어야 함
    client = _ScriptedClient(_health(), [good, violation, recovered, recovered])
    viewer = _make_viewer(client, fps=30.0, rate_hz=6.0)
    try:
        _start_and_wait_for_frame(viewer)
        deadline = time.monotonic() + 5.0
        server_status = ""
        while time.monotonic() < deadline:
            server_status = viewer.status.to_dict()["server_status"]
            if "영구 정지" in server_status:
                break
            time.sleep(0.05)
        assert "영구 정지" in server_status
        # mode가 다시 read_only로 복구된 뒤에도(recovered 샘플들) fatal은 풀리지 않아야 한다.
        time.sleep(0.3)
        assert "영구 정지" in viewer.status.to_dict()["server_status"]
    finally:
        viewer.stop()


# ---------------------------------------------------------------------------
# 실제 HTTP 서버 (ephemeral 포트)
# ---------------------------------------------------------------------------


@pytest.fixture()
def running_server():
    client = _ScriptedClient(_health(), [_state(_positions(wrist_flex=7.0), _positions(wrist_flex=0.0))])
    viewer = _make_viewer(client, fps=20.0, rate_hz=40.0)
    viewer.preflight()
    server = create_http_server(viewer)
    port = server.server_address[1]
    import threading

    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    viewer.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and viewer.frames.snapshot()[0] is None:
        time.sleep(0.05)

    yield viewer, port

    server.shutdown()
    server.server_close()
    viewer.stop()
    thread.join(timeout=2.0)


def test_index_html_response(running_server):
    _, port = running_server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
        assert "<html" in body
        assert "/stream.mjpg" in body
        assert "wrist_flex" in body


def test_health_endpoint(running_server):
    _, port = running_server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
        data = json.loads(resp.read())
        assert data["status"] == "ok"


def test_status_endpoint(running_server):
    _, port = running_server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as resp:
        data = json.loads(resp.read())
        assert "server_status" in data
        assert "joints" in data
        assert "wrist_flex" in data["joints"]


def test_single_frame_endpoint_returns_jpeg(running_server):
    _, port = running_server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/frame.jpg", timeout=5) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "image/jpeg"
        body = resp.read()
        assert body[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_mjpeg_stream_returns_multiple_frames(running_server):
    _, port = running_server
    req = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream.mjpg", timeout=5)
    try:
        chunk = req.read(200_000)
    finally:
        req.close()
    assert b"Content-Type: image/jpeg" in chunk
    assert chunk.count(b"--so101frame") >= 2  # 최소 2프레임 이상 (연속 스트리밍 확인)


def test_unknown_path_returns_404(running_server):
    _, port = running_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
    assert exc_info.value.code == 404


def test_client_disconnect_mid_stream_does_not_crash_server(running_server):
    _, port = running_server
    req = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream.mjpg", timeout=5)
    req.read(10_000)
    req.close()  # 브라우저 탭을 닫는 것과 동일 - 서버 프로세스는 계속 동작해야 함

    time.sleep(0.2)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
        assert resp.status == 200


# ---------------------------------------------------------------------------
# 소유권 (client를 주입하면 stop()이 임의로 닫지 않는다)
# ---------------------------------------------------------------------------


def test_injected_client_is_not_closed_by_stop():
    client = _ScriptedClient(_health(), [_state(_positions(), _positions())])
    viewer = _make_viewer(client)
    viewer.preflight()
    viewer.start()
    time.sleep(0.1)
    viewer.stop()
    assert client.closed is False


# ---------------------------------------------------------------------------
# --joint wrist_flex vs --all-joints 회귀 테스트
#
# 재현 보고: "--joint wrist_flex는 리더암을 움직이면 MuJoCo가 실시간 추종하지만,
# --all-joints는 화면 수치는 변해도 렌더링된 로봇이 전혀 움직이지 않는다."
#
# 조사 결과 (코드 감사 + 아래 테스트들로 재현 시도):
#   - args.joints(표시 대상 목록)는 mapping/apply_update/mj_step/data.ctrl 어디에도 영향을
#     주지 않는다 - mapping은 항상 JOINT_NAMES 6개 전체로 고정 생성된다. 따라서 순수
#     코드 경로상으로는 --joint와 --all-joints가 물리 적용에 있어 동일하게 동작해야 한다.
#   - 아래 "6개 관절이 모두 정상 범위 안에서 변할 때"의 회귀 테스트는 현재 코드에서
#     통과한다 (data.ctrl/data.qpos/프레임이 모두 실제로 변함) - 즉 "--all-joints 자체가
#     구조적으로 멈춘다"는 재현은 이 환경에서 되지 않았다.
#   - 대신 재현에 성공한 시나리오는 따로 있다: 리더암의 정지된(사용자가 움직이지 않는)
#     나머지 관절들이 이 MuJoCo 모델의 실제 관절 range를 벗어나 있으면, 그 관절들은
#     (design대로) BLOCKED되어 직전 안전 target(대개 초기값)에 "정지"한다. 사용자가
#     wrist_flex 하나만 움직이며 관찰하면 나머지 5개가 멈춰 있는 게 "로봇 전체가 안 움직인다"
#     로 보일 수 있다 - 이게 실제 코드 버그는 아니지만(각 관절은 서로 독립적으로 정상
#     처리됨), 화면에서 "어떤 관절이 왜 멈췄는지"가 보이지 않아 이렇게 오인하기 매우
#     쉬웠다. 그래서 이 세션에서 최소 수정한 것은: (1) Requested/Applied target을 분리해
#     BLOCKED된 관절은 즉시 눈에 띄게 하고 (2) --debug-control로 실물 환경에서 바로
#     선다이러 진단을 할 수 있게 했다. safety 완화/clamp/range 수정은 하지 않았다.
# ---------------------------------------------------------------------------


class _DynamicClient:
    """호출마다(=매 폴링마다) 다른 leader position을 만들어내는 가짜 클라이언트.

    _ScriptedClient는 고정된 리스트를 반복하므로 "계속 변화하는 리더암"을 표현하기
    번거롭다 - 이 클라이언트는 호출 횟수를 넘겨주는 콜백으로 매번 새 값을 만든다.
    """

    def __init__(self, health: HealthState, position_fn) -> None:
        self._health = health
        self._position_fn = position_fn
        self.calls = 0
        self.closed = False

    def check_health(self) -> HealthState:
        return self._health

    def get_state(self) -> RemoteState:
        self.calls += 1
        positions = self._position_fn(self.calls)
        return _state(positions, {name: 0.0 for name in JOINT_NAMES}, sequence=self.calls)

    def close(self) -> None:
        self.closed = True


def _single_joint_moving_positions(call_index: int) -> dict[str, float]:
    return _positions(wrist_flex=(call_index * 2.0) % 40.0 - 20.0)


def _all_joints_moving_positions(call_index: int) -> dict[str, float]:
    # 6개 관절 전부, 서로 다른 주기로, 모두 이 모델의 정상 range 안에서 움직인다
    # (모두 ±20deg 이내 - 특정 joint가 BLOCKED되어 이 테스트의 결론을 흐리지 않도록).
    return {
        "shoulder_pan": (call_index * 1.3) % 20.0 - 10.0,
        "shoulder_lift": (call_index * 1.7) % 16.0 - 8.0,
        "elbow_flex": (call_index * 0.9) % 12.0 - 6.0,
        "wrist_flex": (call_index * 2.0) % 40.0 - 20.0,
        "wrist_roll": (call_index * 1.1) % 30.0 - 15.0,
        "gripper": (call_index * 0.5) % 10.0 - 5.0,
    }


def _collect_ctrl_qpos_and_frame_samples(viewer: LiveWebViewer, joints: tuple[str, ...], *, duration: float = 3.0, min_samples: int = 6):
    import hashlib

    samples: dict[str, list[tuple[float | None, float | None]]] = {name: [] for name in joints}
    frame_hashes: set[str] = set()
    deadline = time.monotonic() + duration
    count = 0
    while time.monotonic() < deadline and count < min_samples:
        time.sleep(duration / min_samples)
        count += 1
        status = viewer.status.to_dict()
        for name in joints:
            j = status["joints"][name]
            samples[name].append((j["mujoco_target_deg"], j["mujoco_qpos_deg"]))
        jpeg, _ = viewer.frames.snapshot()
        if jpeg is not None:
            frame_hashes.add(hashlib.md5(jpeg).hexdigest())
    return samples, frame_hashes


def test_single_joint_ctrl_qpos_and_frame_change_over_time():
    """--joint wrist_flex 시나리오: data.ctrl/data.qpos/렌더 프레임이 실제로 변해야 한다."""
    client = _DynamicClient(_health(), _single_joint_moving_positions)
    viewer = _make_viewer(client, joints=("wrist_flex",), fps=30.0, rate_hz=60.0)
    try:
        _start_and_wait_for_frame(viewer)
        samples, frame_hashes = _collect_ctrl_qpos_and_frame_samples(viewer, ("wrist_flex",))
        targets = {round(v, 2) for v, _ in samples["wrist_flex"] if v is not None}
        qposes = {round(v, 2) for _, v in samples["wrist_flex"] if v is not None}
        assert len(targets) >= 2, f"wrist_flex의 data.ctrl(target)이 변하지 않았습니다: {samples}"
        assert len(qposes) >= 2, f"wrist_flex의 data.qpos가 변하지 않았습니다: {samples}"
        assert len(frame_hashes) >= 2, "렌더 프레임(픽셀)이 변하지 않았습니다"
    finally:
        viewer.stop()


def test_all_joints_ctrl_qpos_and_frame_change_over_time():
    """--all-joints 회귀 테스트: 숫자만 변하고 data.ctrl/data.qpos가 정지하면 실패해야 한다."""
    client = _DynamicClient(_health(), _all_joints_moving_positions)
    viewer = _make_viewer(client, joints=tuple(JOINT_NAMES), fps=30.0, rate_hz=60.0)
    try:
        _start_and_wait_for_frame(viewer)
        samples, frame_hashes = _collect_ctrl_qpos_and_frame_samples(viewer, tuple(JOINT_NAMES))

        for name in JOINT_NAMES:
            targets = {round(v, 2) for v, _ in samples[name] if v is not None}
            qposes = {round(v, 2) for _, v in samples[name] if v is not None}
            assert len(targets) >= 2, f"[회귀] {name}의 data.ctrl(target)이 --all-joints에서 정지했습니다: {samples[name]}"
            assert len(qposes) >= 2, f"[회귀] {name}의 data.qpos가 --all-joints에서 정지했습니다: {samples[name]}"
        assert len(frame_hashes) >= 2, "[회귀] --all-joints에서 렌더 프레임(픽셀)이 정지했습니다"
    finally:
        viewer.stop()


def test_requested_target_differs_from_applied_when_blocked_all_joints_mode():
    """--all-joints에서 한 관절만 range를 벗어나도 (a) 그 관절만 BLOCKED되고 requested/applied가
    갈라지며 (b) 나머지 관절은 정상적으로 requested == applied를 유지하며 계속 움직여야 한다."""
    states = [
        _state(_positions(wrist_flex=0.0, gripper=0.0), _positions()),
        _state(_positions(wrist_flex=10.0, gripper=999.0), _positions()),  # gripper만 명백히 range 초과
        _state(_positions(wrist_flex=15.0, gripper=999.0), _positions()),
    ]
    client = _ScriptedClient(_health(), states)
    viewer = _make_viewer(client, joints=tuple(JOINT_NAMES), fps=30.0, rate_hz=60.0)
    try:
        _start_and_wait_for_frame(viewer)
        deadline = time.monotonic() + 3.0
        gripper_blocked = False
        status = {}
        while time.monotonic() < deadline:
            status = viewer.status.to_dict()
            if status["joints"]["gripper"]["blocked"]:
                gripper_blocked = True
                break
            time.sleep(0.05)

        assert gripper_blocked, "gripper가 명백한 range 초과에도 BLOCKED로 감지되지 않았습니다"
        gj = status["joints"]["gripper"]
        assert gj["requested_target_deg"] == pytest.approx(999.0, abs=1.0)
        # applied는 999 근처로 튀지 않아야 한다 (clamp 아님, 직전 안전값 유지)
        assert gj["mujoco_target_deg"] is None or gj["mujoco_target_deg"] < 50.0

        # wrist_flex는 gripper가 BLOCKED이어도 영향받지 않고 정상적으로 requested == applied.
        wf = status["joints"]["wrist_flex"]
        assert wf["blocked"] is False
        if wf["requested_target_deg"] is not None and wf["mujoco_target_deg"] is not None:
            assert wf["requested_target_deg"] == pytest.approx(wf["mujoco_target_deg"], abs=0.5)
    finally:
        viewer.stop()


def test_debug_control_prints_diagnostic_block_without_secrets(capsys):
    client = _ScriptedClient(_health(), [_state(_positions(wrist_flex=5.0), _positions())])
    args = WebViewerArgs(
        server_url="http://laptop.local:8001",
        joints=("wrist_flex",),
        host="127.0.0.1",
        port=0,
        fps=20.0,
        rate_hz=40.0,
        frame_width=64,
        frame_height=48,
        api_token="super-secret-token",
        debug_control=True,
    )
    viewer = LiveWebViewer(args, client=client)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(1.3)  # 1초 주기 진단 출력을 최소 1회 기다림
        out = capsys.readouterr().out
        assert "[제어 진단]" in out
        for field in ("selected_joints", "mapped_targets", "blocking_joints", "applied_targets", "data.ctrl", "data.qpos"):
            assert field in out
        assert "super-secret-token" not in out
    finally:
        viewer.stop()


# ---------------------------------------------------------------------------
# WARN/BLOCKED 원인 추적 (safety_event_tracker 통합)
# ---------------------------------------------------------------------------


def test_out_of_range_value_creates_sticky_safety_event():
    # _ScriptedClient는 마지막 state를 계속 반복하므로(clamp), index1(999deg)이 여러 샘플
    # 동안 유지된다 - render 스레드가 그 값을 놓치지 않고 관측할 시간을 준다.
    states = [
        _state(_positions(wrist_flex=0.0), _positions()),
        _state(_positions(wrist_flex=999.0), _positions()),  # 명백히 range 초과 -> BLOCKED, 이후 반복
    ]
    client = _ScriptedClient(_health(), states)
    viewer = _make_viewer(
        client,
        joints=("wrist_flex",),
        fps=30.0,
        rate_hz=60.0,
        safety_event_config=SafetyEventTrackerConfig(clear_after_samples=2, sticky_display_sec=10.0),
    )
    try:
        _start_and_wait_for_frame(viewer)
        deadline = time.monotonic() + 3.0
        found = None
        while time.monotonic() < deadline:
            events = viewer.safety_tracker.recent_events(now_wall=time.time())
            found = next((e for e in events if e["reason_code"] == "JOINT_RANGE_HIGH" and e["joint"] == "wrist_flex"), None)
            if found is not None:
                break
            time.sleep(0.02)
        assert found is not None, "JOINT_RANGE_HIGH sticky 이벤트가 기록되지 않았습니다"
        assert found["severity"] == "BLOCKED"
        assert found["requested_target_deg"] == pytest.approx(999.0, abs=1.0)
        # 한 샘플만 BLOCKED여도 sticky 목록에 남아 있어야 한다 (정상 복귀 이후에도).
        assert found["sample_count"] >= 1
    finally:
        viewer.stop()


def test_status_endpoint_and_events_endpoint_expose_safety_fields(running_server):
    _, port = running_server
    for path in ("/status", "/events"):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert "current_safety" in data
            assert "level" in data["current_safety"]
            assert "recent_safety_events" in data
            assert isinstance(data["recent_safety_events"], list)
            assert "safety_event_counts" in data
            assert set(data["safety_event_counts"].keys()) == {"WARN", "BLOCKED"}


def test_events_endpoint_rejects_non_get(running_server):
    """/events는 GET만 허용한다 - POST 등은 지원하지 않는 메서드로 처리되어야 한다."""
    import http.client

    _, port = running_server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("POST", "/events", body=b"{}")
        resp = conn.getresponse()
        resp.read()
        assert resp.status in (405, 501)  # BaseHTTPRequestHandler는 미구현 메서드에 501을 준다
    finally:
        conn.close()


def test_safety_events_written_to_json_and_csv_on_stop(tmp_path):
    states = [
        _state(_positions(wrist_flex=0.0), _positions()),
        _state(_positions(wrist_flex=999.0), _positions()),
    ]
    client = _ScriptedClient(_health(), states)
    viewer = _make_viewer(
        client,
        joints=("wrist_flex",),
        fps=30.0,
        rate_hz=60.0,
        events_report_dir=tmp_path,
        safety_event_config=SafetyEventTrackerConfig(clear_after_samples=2, sticky_display_sec=10.0),
    )
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.5)
    finally:
        viewer.stop()

    assert viewer.last_safety_report_paths is not None
    json_path, csv_path = viewer.last_safety_report_paths
    assert json_path.is_file()
    assert csv_path.is_file()
    assert json_path.parent == tmp_path

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["event_count"] >= 1
    assert any(e["reason_code"] == "JOINT_RANGE_HIGH" for e in payload["events"])


def test_safety_event_report_never_leaks_api_token(tmp_path):
    states = [
        _state(_positions(wrist_flex=0.0), _positions()),
        _state(_positions(wrist_flex=999.0), _positions()),
    ]
    client = _ScriptedClient(_health(), states)
    viewer = _make_viewer(
        client,
        joints=("wrist_flex",),
        fps=30.0,
        rate_hz=60.0,
        events_report_dir=tmp_path,
        api_token="ultra-secret-token-xyz",
        safety_event_config=SafetyEventTrackerConfig(clear_after_samples=2, sticky_display_sec=10.0),
    )
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.5)
    finally:
        viewer.stop()

    json_path, csv_path = viewer.last_safety_report_paths
    for path in (json_path, csv_path):
        text = path.read_text(encoding="utf-8")
        assert "ultra-secret-token-xyz" not in text


def test_existing_web_viewer_behavior_has_no_regression_with_safety_tracking():
    """safety 이벤트 추적 코드가 기존 requested/applied target·safety_status 계산을 바꾸지 않는지 확인."""
    states = [
        _state(_positions(wrist_flex=0.0), _positions(wrist_flex=0.0)),
        _state(_positions(wrist_flex=999.0), _positions(wrist_flex=1.0)),
    ]
    client = _ScriptedClient(_health(), states)
    viewer = _make_viewer(client, joints=("wrist_flex",), fps=30.0, rate_hz=60.0)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.4)
        status = viewer.status.to_dict()
        joint = status["joints"]["wrist_flex"]
        assert joint["safety_status"] == "BLOCKED"
        assert joint["blocked"] is True
        assert joint["mujoco_target_deg"] is None or joint["mujoco_target_deg"] < 150.0
    finally:
        viewer.stop()


# ---------------------------------------------------------------------------
# --command-source follower-safe (안전 명령 매퍼 통합)
# ---------------------------------------------------------------------------


def test_default_command_source_is_raw_leader():
    args = WebViewerArgs(server_url="http://x:8001")
    assert args.command_source == COMMAND_SOURCE_RAW_LEADER


def test_invalid_command_source_raises():
    with pytest.raises(ValueError):
        WebViewerArgs(server_url="http://x:8001", command_source="teleop-direct")


def test_raw_leader_mode_does_not_populate_follower_safe_fields():
    """raw-leader(기본) 모드에서는 follower-safe 전용 필드가 채워지면 안 된다 (회귀 방지)."""
    states = [_state(_positions(wrist_flex=5.0), _positions(wrist_flex=1.0))]
    client = _ScriptedClient(_health(), states)
    viewer = _make_viewer(client, joints=("wrist_flex",), command_source=COMMAND_SOURCE_RAW_LEADER)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.2)
        status = viewer.status.to_dict()
        assert status["command_source"] == COMMAND_SOURCE_RAW_LEADER
        assert status["hold"] is False
        assert status["intervention_count"] == 0
        joint = status["joints"]["wrist_flex"]
        assert joint["follower_current_deg"] is None
        assert joint["rate_limited"] is False
        assert viewer.follower_mapper is None
    finally:
        viewer.stop()


def test_follower_safe_mode_applies_rate_limited_command_not_instant_jump():
    """리더가 0->80도로 순간 점프해도, MuJoCo actual(qpos)은 즉시 80으로 안 가고 서서히
    수렴해야 한다 (실제 검증 시나리오 1~2번).

    _ScriptedClient는 마지막 state를 "같은 객체"로 반복해 sequence까지 멈춰버려서(실제로는
    SEQUENCE_STALLED로 정확히 잡히는 게 맞지만) 이 테스트의 목적(rate-limit 수렴)과는 다른
    hold를 유발한다 - 그래서 매 호출마다 sequence는 계속 증가하되 리더값은 80으로 고정
    유지되는 _DynamicClient를 쓴다.
    """
    client = _DynamicClient(_health(), lambda call: _positions(wrist_flex=(0.0 if call <= 1 else 80.0)))
    viewer = _make_viewer(
        client,
        joints=("wrist_flex",),
        fps=30.0,
        rate_hz=60.0,
        command_source=COMMAND_SOURCE_FOLLOWER_SAFE,
    )
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.15)  # 짧은 시간만 경과 - rate limit(기본 15deg/s) 상 80도에 도달 불가
        status = viewer.status.to_dict()
        joint = status["joints"]["wrist_flex"]
        assert joint["requested_target_deg"] == pytest.approx(80.0, abs=1.0)  # B단계는 그대로 기록
        assert joint["mujoco_target_deg"] is not None
        assert joint["mujoco_target_deg"] < 30.0  # 아직 80에 훨씬 못 미쳐야 함 (rate limit)
        assert joint["mujoco_qpos_deg"] < 30.0  # MuJoCo 실제 pose도 마찬가지로 서서히만 이동

        # 충분히 기다리면 결국 목표(80도, wrist_flex 안전 range=±84.6도 안)에 수렴해야 한다.
        deadline = time.monotonic() + 6.0
        converged = False
        while time.monotonic() < deadline:
            status = viewer.status.to_dict()
            if status["joints"]["wrist_flex"]["mujoco_qpos_deg"] > 75.0:
                converged = True
                break
            time.sleep(0.05)
        assert converged, "충분한 시간이 지나도 목표(80도)에 수렴하지 못했습니다"
    finally:
        viewer.stop()


def test_follower_safe_status_exposes_raw_mapped_limited_actual_separately():
    states = [_state(_positions(wrist_flex=10.0), _positions(wrist_flex=2.0))]
    client = _ScriptedClient(_health(), states)
    viewer = _make_viewer(client, joints=("wrist_flex",), command_source=COMMAND_SOURCE_FOLLOWER_SAFE)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.2)
        status = viewer.status.to_dict()
        assert status["command_source"] == COMMAND_SOURCE_FOLLOWER_SAFE
        j = status["joints"]["wrist_flex"]
        # raw / follower current / mapped(requested) / limited(applied) / MuJoCo actual - 전부 분리 노출.
        assert j["leader_deg"] == pytest.approx(10.0)
        assert j["follower_current_deg"] == pytest.approx(2.0)
        assert j["requested_target_deg"] is not None  # mapped target (B단계)
        assert j["mujoco_target_deg"] is not None  # limited command (F단계)
        assert j["mujoco_qpos_deg"] is not None  # MuJoCo actual
    finally:
        viewer.stop()


def test_follower_safe_stale_holds_then_resumes_without_jump():
    good = _state(_positions(wrist_flex=5.0), _positions(wrist_flex=5.0))
    stale = _state(_positions(wrist_flex=50.0), _positions(wrist_flex=5.0), leader_kwargs={"stale": True})
    client = _ScriptedClient(_health(), [good, stale])
    viewer = _make_viewer(
        client, joints=("wrist_flex",), fps=30.0, rate_hz=60.0, stale_after_ms=50.0, command_source=COMMAND_SOURCE_FOLLOWER_SAFE
    )
    try:
        _start_and_wait_for_frame(viewer)
        deadline = time.monotonic() + 3.0
        held = False
        while time.monotonic() < deadline:
            status = viewer.status.to_dict()
            if status["hold"] and status["hold_reason"] == "REMOTE_STALE":
                held = True
                break
            time.sleep(0.05)
        assert held, "stale 상태에서 top-level hold=REMOTE_STALE이 감지되지 않았습니다"
        joint = viewer.status.to_dict()["joints"]["wrist_flex"]
        # stale인 동안 목표(50)로 가면 안 되고 직전 안전값(5) 근처에 머물러야 한다.
        assert joint["mujoco_target_deg"] == pytest.approx(5.0, abs=1.0)
    finally:
        viewer.stop()


def test_follower_safe_range_violation_holds_instead_of_clamp():
    client = _ScriptedClient(
        _health(),
        [
            _state(_positions(wrist_flex=0.0), _positions(wrist_flex=0.0)),
            _state(_positions(wrist_flex=150.0), _positions(wrist_flex=0.0)),  # wrist_flex 안전range(±84.6도) 초과
        ],
    )
    viewer = _make_viewer(client, joints=("wrist_flex",), fps=30.0, rate_hz=60.0, command_source=COMMAND_SOURCE_FOLLOWER_SAFE)
    try:
        _start_and_wait_for_frame(viewer)
        deadline = time.monotonic() + 3.0
        held = False
        while time.monotonic() < deadline:
            joint = viewer.status.to_dict()["joints"]["wrist_flex"]
            if joint["range_held"]:
                held = True
                break
            time.sleep(0.05)
        assert held, "range 초과가 range_held로 감지되지 않았습니다"
        joint = viewer.status.to_dict()["joints"]["wrist_flex"]
        assert joint["follower_hold_reason"] == "RANGE_VIOLATION"
        assert joint["safety_status"] == "BLOCKED"
        # clamp라면 range 상한 근처(약 84.6도) 값이 나오겠지만, hold이므로 훨씬 작아야 한다.
        assert joint["mujoco_target_deg"] is not None and joint["mujoco_target_deg"] < 50.0
    finally:
        viewer.stop()


def test_follower_safe_report_written_on_stop(tmp_path):
    client = _ScriptedClient(_health(), [_state(_positions(wrist_flex=10.0), _positions(wrist_flex=0.0))])
    viewer = _make_viewer(
        client,
        joints=("wrist_flex",),
        command_source=COMMAND_SOURCE_FOLLOWER_SAFE,
        follower_safe_report_dir=tmp_path,
    )
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.3)
    finally:
        viewer.stop()

    assert viewer.last_follower_safe_report_paths is not None
    json_path, csv_path = viewer.last_follower_safe_report_paths
    assert json_path.is_file() and json_path.parent == tmp_path
    assert csv_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_samples"] > 0
    assert len(payload["samples"]) > 0
    row = payload["samples"][0]
    for field in (
        "timestamp",
        "remote_sequence",
        "joint",
        "raw_leader_deg",
        "follower_current_deg",
        "mapped_target_deg",
        "limited_command_deg",
        "mujoco_actual_deg",
        "rate_limited",
        "range_held",
        "connection_held",
        "hold_reason",
        "elapsed_sec",
        "max_step_deg",
    ):
        assert field in row


def test_follower_safe_gripper_excluded_from_active_output():
    client = _ScriptedClient(_health(), [_state(_positions(gripper=50.0), _positions(gripper=10.0))])
    viewer = _make_viewer(client, joints=("gripper",), command_source=COMMAND_SOURCE_FOLLOWER_SAFE)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.2)
        joint = viewer.status.to_dict()["joints"]["gripper"]
        assert joint["follower_hold_reason"] == "UNVERIFIED_RANGE"
        assert joint["safety_status"] == "BLOCKED"
    finally:
        viewer.stop()


def test_follower_safe_mode_never_imports_write_capable_client_methods():
    """실물 팔로워 write API가 어디에도 존재하지 않는지 소스 레벨로 확인한다."""
    import inspect

    from simulation.mujoco import follower_safe_mapper
    from simulation.mujoco.remote_state_client import RemoteSO101StateClient

    mapper_source = inspect.getsource(follower_safe_mapper)
    # write_json/write_csv(로컬 리포트 파일 저장)는 허용 - 원격 HTTP 쓰기 메서드만 금지한다.
    for forbidden in ("def post(", "def put(", "def patch(", "def delete(", "session.post", "session.put", "requests.post", "requests.put"):
        assert forbidden not in mapper_source.lower()

    client_methods = {name for name, _ in inspect.getmembers(RemoteSO101StateClient, predicate=inspect.isfunction) if not name.startswith("_")}
    assert client_methods <= {"check_health", "get_state", "get_calibration", "close"}


# ---------------------------------------------------------------------------
# gripper UNVERIFIED_RANGE 격리 조사 (실물 재현 보고 대응)
#
# 보고: "gripper가 UNVERIFIED_RANGE로 hold되면 팔 5개 관절도 MuJoCo에서 안 움직인다."
# 코드 감사 + 재현 결과 이 저장소 코드에는 그런 결합이 없다 - 아래 테스트들이 "5개 관절이
# gripper와 무관하게 실제로 계속 움직인다"는 것을 ctrl(target)/qpos/렌더 프레임 세 가지
# 레벨 모두에서 직접 증명한다.
# ---------------------------------------------------------------------------


def _five_arm_joints_moving_gripper_fixed(call_index: int) -> dict[str, float]:
    return {
        "shoulder_pan": (call_index * 1.3) % 16.0 - 8.0,
        "shoulder_lift": (call_index * 1.7) % 14.0 - 7.0,
        "elbow_flex": (call_index * 0.9) % 10.0 - 5.0,
        "wrist_flex": (call_index * 2.0) % 20.0 - 10.0,
        "wrist_roll": (call_index * 1.1) % 18.0 - 9.0,
        "gripper": 45.0,  # 항상 고정값 (percent) - 이 값 자체와 무관하게 gripper는 항상 hold
    }


def test_gripper_unverified_range_isolated_arm_joints_actively_update():
    """gripper 하나가 UNVERIFIED_RANGE로 hold인 동안, 나머지 5개 관절의 target/qpos/렌더
    프레임이 실제로 계속 변하는지 (ctrl/qpos/프레임 3중으로) 직접 확인한다."""
    client = _DynamicClient(_health(), _five_arm_joints_moving_gripper_fixed)
    arm_joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    viewer = _make_viewer(client, joints=tuple(JOINT_NAMES), fps=30.0, rate_hz=60.0, command_source=COMMAND_SOURCE_FOLLOWER_SAFE)
    try:
        _start_and_wait_for_frame(viewer)
        samples, frame_hashes = _collect_ctrl_qpos_and_frame_samples(viewer, arm_joints, duration=3.0, min_samples=6)

        for name in arm_joints:
            targets = {round(v, 2) for v, _ in samples[name] if v is not None}
            qposes = {round(v, 2) for _, v in samples[name] if v is not None}
            assert len(targets) >= 2, f"[격리 실패 의심] gripper가 hold인 동안 {name}의 target이 정지했습니다: {samples[name]}"
            assert len(qposes) >= 2, f"[격리 실패 의심] gripper가 hold인 동안 {name}의 qpos가 정지했습니다: {samples[name]}"
        assert len(frame_hashes) >= 2, "gripper가 hold인 동안 렌더 프레임이 정지했습니다"

        # gripper 자신은 계속 held여야 한다 (이 자체는 정상 동작 - 격리 확인용).
        status = viewer.status.to_dict()
        assert status["joints"]["gripper"]["follower_hold_reason"] == "UNVERIFIED_RANGE"
        assert status["held_joints"] == {"gripper": "UNVERIFIED_RANGE"}
        assert status["active_joint_count"] == 5
        assert status["global_hold"] is False
    finally:
        viewer.stop()


def test_single_joint_range_violation_isolates_only_that_joint():
    """wrist_flex 하나만 range를 벗어나도 그 관절만 hold되고 나머지는 계속 갱신되어야 한다."""
    client = _DynamicClient(
        _health(),
        lambda call: {
            "shoulder_pan": (call * 1.3) % 16.0 - 8.0,
            "shoulder_lift": (call * 1.7) % 14.0 - 7.0,
            "elbow_flex": (call * 0.9) % 10.0 - 5.0,
            "wrist_flex": 150.0,  # wrist_flex 안전range(±84.6도) 명백히 초과, 고정
            "wrist_roll": (call * 1.1) % 18.0 - 9.0,
            "gripper": 0.0,
        },
    )
    other_joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll")
    viewer = _make_viewer(client, joints=tuple(JOINT_NAMES), fps=30.0, rate_hz=60.0, command_source=COMMAND_SOURCE_FOLLOWER_SAFE)
    try:
        _start_and_wait_for_frame(viewer)
        samples, frame_hashes = _collect_ctrl_qpos_and_frame_samples(viewer, other_joints, duration=3.0, min_samples=6)

        for name in other_joints:
            targets = {round(v, 2) for v, _ in samples[name] if v is not None}
            assert len(targets) >= 2, f"wrist_flex만 RANGE_VIOLATION인데 {name}까지 정지했습니다: {samples[name]}"

        status = viewer.status.to_dict()
        assert status["joints"]["wrist_flex"]["follower_hold_reason"] == "RANGE_VIOLATION"
        assert "wrist_flex" in status["held_joints"]
        assert status["global_hold"] is False  # RANGE_VIOLATION은 절대 전역이 아니다
        for name in other_joints:
            assert name not in status["held_joints"]
    finally:
        viewer.stop()


def test_remote_stale_holds_all_six_joints_globally():
    good = _state(_positions(wrist_flex=5.0), _positions(wrist_flex=5.0))
    stale = _state(_positions(wrist_flex=50.0), _positions(wrist_flex=5.0), leader_kwargs={"stale": True})
    client = _ScriptedClient(_health(), [good, stale])
    viewer = _make_viewer(
        client, joints=tuple(JOINT_NAMES), fps=30.0, rate_hz=60.0, stale_after_ms=50.0, command_source=COMMAND_SOURCE_FOLLOWER_SAFE
    )
    try:
        _start_and_wait_for_frame(viewer)
        deadline = time.monotonic() + 3.0
        status = viewer.status.to_dict()
        while time.monotonic() < deadline and not status["global_hold"]:
            status = viewer.status.to_dict()
            time.sleep(0.05)
        assert status["global_hold"] is True
        assert status["global_hold_reason"] == "REMOTE_STALE"
        assert status["held_joint_count"] == 6  # gripper 포함 6개 전부
        assert status["active_joint_count"] == 0
        assert set(status["held_joints"].keys()) == set(JOINT_NAMES)
    finally:
        viewer.stop()


def test_status_summary_matches_requested_json_schema():
    """요구사항 예시(global_hold/active_joint_count/held_joint_count/held_joints) 그대로 노출되는지."""
    client = _ScriptedClient(_health(), [_state(_positions(gripper=45.0), _positions())])
    viewer = _make_viewer(client, joints=tuple(JOINT_NAMES), command_source=COMMAND_SOURCE_FOLLOWER_SAFE)
    try:
        _start_and_wait_for_frame(viewer)
        time.sleep(0.3)
        status = viewer.status.to_dict()
        for key in ("global_hold", "active_joint_count", "held_joint_count", "held_joints"):
            assert key in status
        assert status["global_hold"] is False
        assert status["active_joint_count"] == 5
        assert status["held_joint_count"] == 1
        assert status["held_joints"] == {"gripper": "UNVERIFIED_RANGE"}
    finally:
        viewer.stop()
