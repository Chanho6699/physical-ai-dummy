"""scripts/run_real_pick_drop_realtime.py 검증 (Phase C-5, 섹션 11 요구사항: "실제 실행
전 dry-run 또는 fake integration test 한 번 수행").

``RealtimeSessionOrchestrator``는 이미 connect/open이 끝난 협력 객체(Fake 가능)만
받도록 설계했다 - 이 파일은 전부 Fake writer/state source/VLA client로 실제
background thread(AsyncVLAChunkInferenceWorker + RealTimeFollowerControlLoop)를 짧게
띄워 검증한다. 실제 하드웨어 접근 없음 - connect()가 실제 포트로 호출되는 경로가 이
테스트 어디에도 없다.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import pytest

import scripts.run_real_pick_drop_realtime as cli
from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.fake_follower_state_source import FakeFollowerStateSource
from runtime.laptop.follower_action_writer import FakeFollowerWriter
from runtime.laptop.observation_snapshot import ObservationSnapshot
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig

TASK = "Pick up the cube and drop it into the bin."


def _neutral(v: float = 0.0) -> dict[str, float]:
    return {j: v for j in JOINT_ORDER}


@dataclass
class _FakeChunkResult:
    ok: bool
    chunk: list[dict[str, float]] | None
    chunk_size: int | None
    chunk_index_spacing_s: float | None
    model_id: str | None
    backend: str | None
    inference_latency_ms: float | None
    server_received_at: float | None
    server_responded_at: float | None
    error_kind: str | None
    error_message: str | None


@dataclass
class ScriptedVLAClient:
    """C-4 stress 스크립트와 동일한 기법 - sequence별로 action/실패를 정밀 제어."""

    action_by_default: dict[str, float] | None = None
    fail: bool = False
    delay_s: float = 0.01
    calls: list[int] = field(default_factory=list)

    def predict_chunk(self, *, session_id, task, sequence, state, images):
        self.calls.append(sequence)
        time.sleep(self.delay_s)
        if self.fail:
            return _FakeChunkResult(
                ok=False, chunk=None, chunk_size=None, chunk_index_spacing_s=None, model_id=None, backend=None,
                inference_latency_ms=None, server_received_at=None, server_responded_at=None,
                error_kind="communication", error_message="injected failure",
            )
        action = self.action_by_default if self.action_by_default is not None else dict(state)
        chunk = [dict(action) for _ in range(50)]
        return _FakeChunkResult(
            ok=True, chunk=chunk, chunk_size=50, chunk_index_spacing_s=1.0 / 30.0,
            model_id="fake", backend="fake", inference_latency_ms=self.delay_s * 1000.0,
            server_received_at=None, server_responded_at=None, error_kind=None, error_message=None,
        )


class FakeObservationProvider:
    def __init__(self, state_source, task: str) -> None:
        self.state_source = state_source
        self.task = task

    def capture(self, *, sequence: int) -> ObservationSnapshot:
        return ObservationSnapshot(
            images={}, state=self.state_source.read().positions_deg, task=self.task,
            capture_monotonic_time=time.monotonic(), sequence=sequence,
        )


class ExplodingWriter:
    """FakeFollowerWriter와 같은 write_count 계약을 유지하되 항상 예외를 던진다."""

    def __init__(self) -> None:
        self.write_count = 0

    def write(self, action_deg):
        self.write_count += 1
        raise RuntimeError("injected writer exception")


class AlwaysFailingStateSource:
    def __init__(self) -> None:
        self.read_count = 0

    def read(self):
        self.read_count += 1
        raise RuntimeError("injected state read failure")


def _make_orchestrator(*, vla_client, state_source, writer, safety_gate=None, **kwargs) -> cli.RealtimeSessionOrchestrator:
    gate = safety_gate or SafetyGate(SafetyGateConfig(
        joint_range_deg={j: (-1000.0, 1000.0) for j in JOINT_ORDER}, max_step_deg={j: 1000.0 for j in JOINT_ORDER},
    ))
    obs_provider = FakeObservationProvider(state_source, TASK)
    return cli.RealtimeSessionOrchestrator(
        observation_provider=obs_provider, state_source=state_source, writer=writer, vla_client=vla_client,
        safety_gate=gate, motion_limits=None, session_id="test-session", task=TASK,
        control_hz=60.0, sanity_window_s=kwargs.pop("sanity_window_s", 0.3),
        max_runtime_s=kwargs.pop("max_runtime_s", 1.0),
        trajectory_wait_timeout_s=kwargs.pop("trajectory_wait_timeout_s", 5.0),
        print_fn=lambda *a, **k: None,  # 테스트 출력 억제
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_normal_run_completes_with_max_runtime_and_writes() -> None:
    state_source = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = FakeFollowerWriter()
    vla_client = ScriptedVLAClient(delay_s=0.01)

    orchestrator = _make_orchestrator(vla_client=vla_client, state_source=state_source, writer=writer, max_runtime_s=1.0, sanity_window_s=0.3)
    report = orchestrator.run()

    assert report.stop_reason == cli.StopReason.NORMAL_MAX_RUNTIME.value
    assert report.writer["write_count"] > 0
    assert report.control["actual_hz"] is not None and report.control["actual_hz"] > 30.0
    assert report.intent["accept"] > 0
    assert report.final_safety["accept"] > 0


def test_trajectory_never_usable_raises_before_loop_starts() -> None:
    state_source = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = FakeFollowerWriter()
    vla_client = ScriptedVLAClient(fail=True, delay_s=0.01)  # 항상 실패 -> usable chunk가 절대 안 생김

    orchestrator = _make_orchestrator(
        vla_client=vla_client, state_source=state_source, writer=writer, trajectory_wait_timeout_s=0.5,
    )
    with pytest.raises(cli.RealtimeRunError, match="usable trajectory"):
        orchestrator.run()
    assert writer.write_count == 0


# ---------------------------------------------------------------------------
# Fatal conditions
# ---------------------------------------------------------------------------


def test_sanity_window_fatal_when_dangerous_target_blocks_every_tick() -> None:
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703
    state_source = FakeFollowerStateSource(initial_state_deg=current)
    writer = FakeFollowerWriter()
    dangerous = dict(current)
    dangerous["wrist_flex"] = 33.6703
    vla_client = ScriptedVLAClient(action_by_default=dangerous, delay_s=0.005)

    gate = SafetyGate(SafetyGateConfig.from_repo_defaults())  # 재캘리브레이션된 실제 threshold
    orchestrator = _make_orchestrator(
        vla_client=vla_client, state_source=state_source, writer=writer, safety_gate=gate,
        sanity_window_s=0.3, max_runtime_s=5.0, trajectory_wait_timeout_s=2.0,
    )
    report = orchestrator.run()

    assert report.stop_reason == cli.StopReason.SANITY_WINDOW_FATAL.value
    assert writer.write_count == 0
    assert report.intent["would_clamp"] > 0


def test_nonproductive_fatal_when_stale_forever(monkeypatch) -> None:
    monkeypatch.setattr(cli, "FATAL_NONPRODUCTIVE_S", 0.3)  # 테스트 속도를 위해 임계값만 축소
    state_source = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = FakeFollowerWriter()

    # 첫 호출은 horizon≈333ms짜리 chunk 하나(bounded wait가 확실히 잡을 수 있을 만큼
    # 넉넉함)를 만들고, 그 뒤로는 계속 실패만 반환한다(더 이상 새 chunk가 안 생김) -
    # 그 chunk의 horizon이 지나면 NO_TARGET이 계속되는 상황을 재현한다.
    @dataclass
    class ShortHorizonClient:
        delay_s: float = 0.01
        called: int = 0

        def predict_chunk(self, *, session_id, task, sequence, state, images):
            self.called += 1
            time.sleep(self.delay_s)
            if self.called > 1:
                # 첫 호출 이후로는 계속 실패만 반환한다(응답이 느려지는 게 아니라 아예 새
                # chunk가 생기지 않는 상황을 재현) - worker.stop()이 절대 안 걸리는 sleep에
                # 막히지 않도록, hang 대신 빠른 실패를 쓴다.
                return _FakeChunkResult(
                    ok=False, chunk=None, chunk_size=None, chunk_index_spacing_s=None, model_id=None, backend=None,
                    inference_latency_ms=None, server_received_at=None, server_responded_at=None,
                    error_kind="communication", error_message="no more chunks (시뮬레이션)",
                )
            chunk = [dict(state) for _ in range(10)]  # chunk_size=10, spacing=1/30 -> horizon≈333ms
            return _FakeChunkResult(
                ok=True, chunk=chunk, chunk_size=10, chunk_index_spacing_s=1.0 / 30.0,
                model_id="fake", backend="fake", inference_latency_ms=self.delay_s * 1000.0,
                server_received_at=None, server_responded_at=None, error_kind=None, error_message=None,
            )

    orchestrator = _make_orchestrator(
        vla_client=ShortHorizonClient(), state_source=state_source, writer=writer,
        sanity_window_s=0.05, max_runtime_s=5.0, trajectory_wait_timeout_s=2.0,
    )
    report = orchestrator.run()
    assert report.stop_reason == cli.StopReason.NONPRODUCTIVE_FATAL.value


def test_writer_exception_fatal_stops_immediately() -> None:
    state_source = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = ExplodingWriter()
    vla_client = ScriptedVLAClient(delay_s=0.01)

    orchestrator = _make_orchestrator(
        vla_client=vla_client, state_source=state_source, writer=writer,
        sanity_window_s=10.0,  # sanity window보다 먼저 writer fatal이 걸리는지 보기 위해 크게
        max_runtime_s=5.0, trajectory_wait_timeout_s=2.0,
    )
    report = orchestrator.run()
    assert report.stop_reason == cli.StopReason.WRITER_EXCEPTION_FATAL.value
    # 감시는 폴링 기반(0.1s 간격)이라 몇 번의 "시도"는 더 있을 수 있지만(각 시도는 매번
    # 예외로 실패 - 즉 하드웨어에 뭔가 나간 적은 한 번도 없다), 무한정 계속되지 않고
    # 빠르게 멈춘다는 게 핵심이다.
    assert 1 <= writer.write_count <= 10


def test_state_read_failure_fatal_after_threshold() -> None:
    state_source = AlwaysFailingStateSource()
    writer = FakeFollowerWriter()

    # observation은 별도 fake state로 정상 공급하되(그래야 chunk 자체는 생성됨),
    # 제어 루프의 state_source만 항상 실패하게 한다.
    obs_state_source = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    obs_provider = FakeObservationProvider(obs_state_source, TASK)
    vla_client = ScriptedVLAClient(delay_s=0.01)
    gate = SafetyGate(SafetyGateConfig(
        joint_range_deg={j: (-1000.0, 1000.0) for j in JOINT_ORDER}, max_step_deg={j: 1000.0 for j in JOINT_ORDER},
    ))
    orchestrator = cli.RealtimeSessionOrchestrator(
        observation_provider=obs_provider, state_source=state_source, writer=writer, vla_client=vla_client,
        safety_gate=gate, motion_limits=None, session_id="test-session", task=TASK,
        control_hz=60.0, sanity_window_s=10.0, max_runtime_s=5.0, trajectory_wait_timeout_s=2.0,
        print_fn=lambda *a, **k: None,
    )
    report = orchestrator.run()
    assert report.stop_reason == cli.StopReason.STATE_READ_FAILURE_FATAL.value
    assert writer.write_count == 0


def test_keyboard_interrupt_is_handled_cleanly() -> None:
    state_source = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = FakeFollowerWriter()
    vla_client = ScriptedVLAClient(delay_s=0.01)

    calls = {"n": 0}

    def sleep_then_interrupt(seconds: float) -> None:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise KeyboardInterrupt()
        time.sleep(seconds)

    orchestrator = _make_orchestrator(
        vla_client=vla_client, state_source=state_source, writer=writer,
        sanity_window_s=10.0, max_runtime_s=100.0, trajectory_wait_timeout_s=2.0,
        sleep_fn=sleep_then_interrupt,
    )
    report = orchestrator.run()
    assert report.stop_reason == cli.StopReason.KEYBOARD_INTERRUPT.value
    # 정상 종료 절차(control loop stop -> worker stop)가 예외 없이 끝났다 - report가 만들어졌다는 것 자체가 증거.


# ---------------------------------------------------------------------------
# 종료 후 clean shutdown 확인
# ---------------------------------------------------------------------------


def test_shutdown_stops_both_threads() -> None:
    state_source = FakeFollowerStateSource(initial_state_deg=_neutral(0.0))
    writer = FakeFollowerWriter()
    vla_client = ScriptedVLAClient(delay_s=0.01)
    orchestrator = _make_orchestrator(vla_client=vla_client, state_source=state_source, writer=writer, max_runtime_s=0.5, sanity_window_s=0.2)
    orchestrator.run()
    assert orchestrator._worker.is_running() is False
    assert orchestrator._loop.is_running() is False


# ---------------------------------------------------------------------------
# Real-mode calibration preflight (pure filesystem tests; no hardware import)
# ---------------------------------------------------------------------------

def _write_fake_follower_calibration(path) -> None:
    import json
    raw_ranges = {
        "shoulder_pan": (1070, 3135), "shoulder_lift": (793, 3238),
        "elbow_flex": (873, 3084), "wrist_flex": (1052, 2977),
        "wrist_roll": (0, 4095), "gripper": (2047, 3496),
    }
    payload = {
        joint: {"id": index, "drive_mode": 0, "homing_offset": 0,
                "range_min": bounds[0], "range_max": bounds[1]}
        for index, (joint, bounds) in enumerate(raw_ranges.items(), start=1)
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_real_mode_actual_calibration_loads_and_resolves_ranges(tmp_path) -> None:
    calibration = tmp_path / "chanho_follower.json"
    _write_fake_follower_calibration(calibration)
    config = cli.resolve_real_safety_config(
        follower_id="chanho_follower", calibration_path=str(calibration),
    )
    assert config.uses_calibration_fallback is False
    assert config.calibration_file_path == str(calibration)
    assert config.joint_range_source["wrist_flex"] == "calibration_file"
    lo, hi = config.joint_range_deg["wrist_flex"]
    expected_hi = (2977 - 1052) / 2 * 360 / 4095
    assert (lo, hi) == pytest.approx((-expected_hi, expected_hi))


def test_real_mode_missing_calibration_fails_before_writer_exists(tmp_path) -> None:
    with pytest.raises(cli.RealtimeRunError, match="actual follower calibration not found"):
        cli.resolve_real_safety_config(
            follower_id="chanho_follower",
            calibration_path=str(tmp_path / "missing.json"),
        )


def test_desktop_default_config_still_allows_fallback(tmp_path) -> None:
    mapper = tmp_path / "mapper.yaml"
    mapper.write_text(
        "calibration_file_path: /definitely/missing.json\n"
        "motor_resolution: 4096\n"
        "fallback_raw_range:\n"
        "  shoulder_pan: {min: 1070, max: 3135}\n"
        "  shoulder_lift: {min: 793, max: 3238}\n"
        "  elbow_flex: {min: 873, max: 3084}\n"
        "  wrist_flex: {min: 1052, max: 2977}\n"
        "  wrist_roll: {min: 0, max: 4095}\n"
        "  gripper: {min: 2047, max: 3496}\n"
        "rate_limit_deg_per_sec:\n"
        "  shoulder_pan: 20\n  shoulder_lift: 15\n  elbow_flex: 20\n"
        "  wrist_flex: 15\n  wrist_roll: 25\n  gripper: 30\n",
        encoding="utf-8",
    )
    config = SafetyGateConfig.from_repo_defaults(follower_safe_mapper_config_path=mapper)
    assert config.uses_calibration_fallback is True
