"""simulation/mujoco/remote_diagnostic.py 통합(orchestrator) 테스트.

실제 노트북 서버나 네트워크 없이, ``RemoteSO101StateClient``와 동일한 인터페이스
(``check_health``/``get_state``/``get_calibration``/``close``)를 갖는 스크립트 가능한
가짜 클라이언트를 ``run_diagnostic(args, client=...)``에 주입해 검증한다. 실제 MuJoCo
모델은 그대로 로딩한다 (기존 test_mujoco_safety_checks.py와 동일한 방식 - 빠르고,
모델 자체를 목으로 대체하면 safety gate 검증의 의미가 없어지기 때문).
"""

from __future__ import annotations

import csv
import json
import threading
import time

import pytest

from simulation.mujoco.diagnostic_analysis import DiagnosticConfig
from simulation.mujoco.remote_diagnostic import NetworkSafetyConfig, RemoteDiagnosticArgs, run_diagnostic
from simulation.mujoco.remote_state_client import JOINT_NAMES, ArmStateView, HealthState, RemoteState, RemoteStateError


def _positions(**overrides) -> dict[str, float]:
    base = {name: 0.0 for name in JOINT_NAMES}
    base.update(overrides)
    return base


def _arm(positions: dict[str, float], *, connected: bool = True, stale: bool = False, age_ms: float = 10.0) -> ArmStateView:
    return ArmStateView(
        connected=connected, positions_deg=dict(positions), raw_ticks=None, stale=stale, age_ms=age_ms, valid=True, invalid_reason=None
    )


def _state(
    leader: dict[str, float],
    follower: dict[str, float],
    *,
    sequence: int = 1,
    mode: str = "read_only",
    leader_kwargs: dict | None = None,
    follower_kwargs: dict | None = None,
) -> RemoteState:
    leader_kwargs = leader_kwargs or {}
    follower_kwargs = follower_kwargs or {}
    return RemoteState(
        raw_timestamp=1000.0,
        sequence=sequence,
        mode=mode,
        leader=_arm(leader, **leader_kwargs),
        follower=_arm(follower, **follower_kwargs),
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
    """리스트에 준 상태를 순서대로(마지막에 도달하면 마지막 항목을 반복) 반환하는 가짜 클라이언트."""

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

    def get_calibration(self) -> dict:
        return {"leader": {}, "follower": {}}

    def close(self) -> None:
        self.closed = True


def _base_args(tmp_path, **overrides) -> RemoteDiagnosticArgs:
    kwargs = dict(
        server_url="http://laptop.local:8001",
        mode="headless",
        joints=("wrist_flex",),
        duration_sec=0.2,
        rate_hz=40.0,
        timeout_ms=200.0,
        stale_after_ms=500.0,
        max_retries=1,
        record=True,
        report_path=tmp_path / "session.json",
        diagnostic_config=DiagnosticConfig(),
        network_safety=NetworkSafetyConfig(),
        quiet=True,
        session_id="test_session",
    )
    kwargs.update(overrides)
    return RemoteDiagnosticArgs(**kwargs)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 정상 MuJoCo 적용
# ---------------------------------------------------------------------------


def test_normal_run_applies_leader_state_and_writes_reports(tmp_path):
    good_state = _state(_positions(wrist_flex=10.0), _positions(wrist_flex=9.5))
    client = _ScriptedClient(_health(), [good_state])
    args = _base_args(tmp_path)

    outcome = run_diagnostic(args, client=client)

    assert outcome.exit_code == 0
    assert outcome.final_result in ("PASS", "WARN")
    assert outcome.json_path is not None and outcome.json_path.is_file()
    assert outcome.csv_path is not None and outcome.csv_path.is_file()
    assert outcome.summary["sample_count"] > 0
    # 클라이언트를 호출자가 주입했으므로 close()는 호출자 책임이다 (run_diagnostic이 임의로
    # 닫지 않는다 - 소유권이 없는 리소스를 정리하면 재사용하는 호출자가 놀랄 수 있음).
    assert client.closed is False

    rows = _read_csv(outcome.csv_path)
    wrist_rows = [r for r in rows if r["joint_name"] == "wrist_flex"]
    assert wrist_rows
    # 리더 값이 그대로 CSV에 기록되었는지 확인
    assert float(wrist_rows[-1]["leader_position_deg"]) == pytest.approx(10.0)
    # 정상 범위 내이므로 MuJoCo target도 리더 값을 따라간다
    assert float(wrist_rows[-1]["mujoco_target_deg"]) == pytest.approx(10.0, abs=0.1)


def test_record_false_skips_csv_but_still_writes_json(tmp_path):
    good_state = _state(_positions(wrist_flex=5.0), _positions(wrist_flex=5.0))
    client = _ScriptedClient(_health(), [good_state])
    args = _base_args(tmp_path, record=False, report_path=tmp_path / "no_csv.json")

    outcome = run_diagnostic(args, client=client)

    assert outcome.csv_path is None
    assert outcome.json_path.is_file()
    assert not (tmp_path / "no_csv.csv").exists()


# ---------------------------------------------------------------------------
# MuJoCo range BLOCKED + 직전 안전값 유지
# ---------------------------------------------------------------------------


def test_out_of_range_leader_value_is_blocked_and_previous_target_held(tmp_path):
    init_state = _state(_positions(wrist_flex=0.0), _positions(wrist_flex=0.0))
    blocked_state = _state(_positions(wrist_flex=999.0), _positions(wrist_flex=1.0))  # 명백히 range 초과
    client = _ScriptedClient(_health(), [init_state, blocked_state])
    args = _base_args(tmp_path, duration_sec=0.15, rate_hz=40.0)

    outcome = run_diagnostic(args, client=client)

    assert outcome.summary["mujoco_blocked_events"] >= 1
    assert outcome.final_result == "WARN"

    rows = _read_csv(outcome.csv_path)
    wrist_rows = [r for r in rows if r["joint_name"] == "wrist_flex"]
    # 마지막 몇 개 샘플은 leader=999였지만, mujoco_target_deg는 절대 999 근처로 튀지 않아야 한다
    # (BLOCKED된 관절은 직전 안전 target을 유지하고, clamp도 하지 않는다).
    for row in wrist_rows[1:]:
        target = float(row["mujoco_target_deg"])
        assert target < 150.0  # 999deg는커녕 실제 range(약 ±95deg) 근처에도 못 미쳐야 함
        assert row["blocked_reason"] != ""


def test_blocked_joint_does_not_prevent_other_joints_from_updating(tmp_path):
    init_state = _state(_positions(), _positions())
    mixed_state = _state(
        _positions(wrist_flex=999.0, shoulder_pan=20.0), _positions(wrist_flex=1.0, shoulder_pan=19.5)
    )
    client = _ScriptedClient(_health(), [init_state, mixed_state])
    args = _base_args(tmp_path, joints=("wrist_flex", "shoulder_pan"), duration_sec=0.15, rate_hz=40.0, record=True)

    outcome = run_diagnostic(args, client=client)

    rows = _read_csv(outcome.csv_path)
    shoulder_rows = [r for r in rows if r["joint_name"] == "shoulder_pan"]
    assert shoulder_rows
    # wrist_flex가 막혀도 shoulder_pan은 정상적으로 리더 값을 따라간다
    assert float(shoulder_rows[-1]["mujoco_target_deg"]) == pytest.approx(20.0, abs=0.5)


# ---------------------------------------------------------------------------
# 준비 단계(preflight) BLOCKED
# ---------------------------------------------------------------------------


def test_preflight_blocked_when_mode_not_read_only(tmp_path):
    client = _ScriptedClient(_health(mode="teleop"), [_state(_positions(), _positions())])
    args = _base_args(tmp_path)
    outcome = run_diagnostic(args, client=client)
    assert outcome.final_result == "BLOCKED"
    assert outcome.exit_code == 1
    assert outcome.json_path is None


def test_preflight_blocked_when_write_enabled_not_false(tmp_path):
    client = _ScriptedClient(_health(write_enabled=True), [_state(_positions(), _positions())])
    outcome = run_diagnostic(_base_args(tmp_path), client=client)
    assert outcome.final_result == "BLOCKED"


def test_preflight_blocked_when_write_enabled_missing(tmp_path):
    client = _ScriptedClient(_health(write_enabled=None), [_state(_positions(), _positions())])
    outcome = run_diagnostic(_base_args(tmp_path), client=client)
    assert outcome.final_result == "BLOCKED"  # 확인 불가 -> 안전하게 중단


def test_preflight_blocked_when_leader_disconnected(tmp_path):
    client = _ScriptedClient(_health(leader_connected=False), [_state(_positions(), _positions())])
    outcome = run_diagnostic(_base_args(tmp_path), client=client)
    assert outcome.final_result == "BLOCKED"


def test_preflight_blocked_when_follower_disconnected(tmp_path):
    client = _ScriptedClient(_health(follower_connected=False), [_state(_positions(), _positions())])
    outcome = run_diagnostic(_base_args(tmp_path), client=client)
    assert outcome.final_result == "BLOCKED"


def test_preflight_blocked_when_initial_state_stale(tmp_path):
    stale_state = _state(_positions(), _positions(), leader_kwargs={"stale": True})
    client = _ScriptedClient(_health(), [stale_state])
    outcome = run_diagnostic(_base_args(tmp_path), client=client)
    assert outcome.final_result == "BLOCKED"


def test_preflight_blocked_when_health_status_unknown(tmp_path):
    client = _ScriptedClient(_health(status="unknown"), [_state(_positions(), _positions())])
    outcome = run_diagnostic(_base_args(tmp_path), client=client)
    assert outcome.final_result == "BLOCKED"


def test_preflight_blocked_on_health_connection_error(tmp_path):
    class _FailingHealthClient(_ScriptedClient):
        def check_health(self):
            raise RemoteStateError("연결 실패")

    client = _FailingHealthClient(_health(), [_state(_positions(), _positions())])
    outcome = run_diagnostic(_base_args(tmp_path), client=client)
    assert outcome.final_result == "BLOCKED"


# ---------------------------------------------------------------------------
# 네트워크 안전 (timeout/stale 발생 시 일시정지, 기본은 자동 재개 안 함)
# ---------------------------------------------------------------------------


def test_network_timeout_during_loop_pauses_and_reports_warn(tmp_path):
    good_state = _state(_positions(wrist_flex=1.0), _positions(wrist_flex=1.0))
    client = _ScriptedClient(_health(), [good_state, RemoteStateError("네트워크 끊김")])
    args = _base_args(tmp_path, duration_sec=0.15, rate_hz=40.0)

    outcome = run_diagnostic(args, client=client)

    assert outcome.summary["timeout_count"] >= 1
    assert outcome.summary["network_pause_events"] >= 1
    assert outcome.final_result == "WARN"


def test_stale_during_loop_pauses_updates(tmp_path):
    good_state = _state(_positions(wrist_flex=1.0), _positions(wrist_flex=1.0))
    stale_state = _state(_positions(wrist_flex=50.0), _positions(wrist_flex=1.0), leader_kwargs={"stale": True})
    client = _ScriptedClient(_health(), [good_state, stale_state])
    args = _base_args(tmp_path, duration_sec=0.15, rate_hz=40.0)

    outcome = run_diagnostic(args, client=client)

    assert outcome.summary["stale_count"] >= 1
    rows = _read_csv(outcome.csv_path)
    wrist_rows = [r for r in rows if r["joint_name"] == "wrist_flex"]
    # stale 동안에는 leader=50이 들어와도 target이 그 값을 따라가면 안 된다 (일시정지 유지)
    for row in wrist_rows[1:]:
        assert float(row["mujoco_target_deg"]) < 10.0


def test_auto_resume_false_stays_paused_after_recovery(tmp_path):
    good = _state(_positions(wrist_flex=1.0), _positions(wrist_flex=1.0))
    bad = RemoteStateError("일시적 오류")
    # good -> bad -> good -> good : 기본값(auto_resume=false)이면 한 번 멈추면 계속 정지 상태.
    client = _ScriptedClient(_health(), [good, bad, good, good])
    args = _base_args(tmp_path, duration_sec=0.2, rate_hz=40.0, network_safety=NetworkSafetyConfig(auto_resume=False))

    outcome = run_diagnostic(args, client=client)
    assert outcome.summary["network_pause_events"] == 1  # 재개하지 않으므로 정지 이벤트가 반복 카운트되지 않음


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------


def test_dry_run_never_touches_client_and_writes_minimal_report(tmp_path):
    args = _base_args(tmp_path, dry_run=True, report_path=tmp_path / "dry.json")
    outcome = run_diagnostic(args)  # client 주입 없음 - dry-run은 애초에 client를 만들지 않는다
    assert outcome.exit_code == 0
    assert outcome.final_result == "PASS"
    assert outcome.summary["sample_count"] == 0
    assert outcome.json_path.is_file()
    assert outcome.csv_path is None


# ---------------------------------------------------------------------------
# 콘솔 출력 모드가 예외를 던지지 않는지 (quiet/verbose/no-color)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"quiet": True, "verbose": False, "no_color": False},
        {"quiet": False, "verbose": True, "no_color": False},
        {"quiet": False, "verbose": False, "no_color": True},
    ],
)
def test_console_output_modes_do_not_crash(tmp_path, overrides, capsys):
    good_state = _state(_positions(wrist_flex=1.0), _positions(wrist_flex=1.0))
    client = _ScriptedClient(_health(), [good_state])
    args = _base_args(tmp_path, **overrides)
    outcome = run_diagnostic(args, client=client)
    assert outcome.exit_code == 0


def test_no_injected_client_creates_and_closes_real_client_on_connection_failure(tmp_path):
    """client를 주입하지 않으면 run_diagnostic이 직접 만들고, 끝나면 스스로 close()한다.

    실제 서버가 없는 주소로 접속을 시도해 (제한된 재시도 후) BLOCKED로 정상 종료되는지도
    함께 확인한다 - 무한 대기/무한 재시도가 없어야 테스트가 빠르게 끝난다.
    """
    args = _base_args(
        tmp_path,
        server_url="http://127.0.0.1:1",  # 아무 것도 듣고 있지 않은 포트
        timeout_ms=100.0,
        max_retries=1,
    )
    outcome = run_diagnostic(args)  # client 주입 없음
    assert outcome.final_result == "BLOCKED"
    assert outcome.exit_code == 1


def test_api_token_never_appears_in_json_report(tmp_path):
    good_state = _state(_positions(wrist_flex=1.0), _positions(wrist_flex=1.0))
    client = _ScriptedClient(_health(), [good_state])
    args = _base_args(tmp_path, api_token="top-secret-token")
    outcome = run_diagnostic(args, client=client)
    assert "top-secret-token" not in outcome.json_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# mode == "offscreen" (WSLg GUI 대체 경로 - GUI 창 없이 PNG/MP4 저장)
# ---------------------------------------------------------------------------


def test_offscreen_mode_saves_png_frames_and_manifest(tmp_path):
    states = [
        _state(_positions(wrist_flex=v), _positions(wrist_flex=0.0), sequence=i)
        for i, v in enumerate([0.0, 5.0, 10.0, 15.0])
    ]
    client = _ScriptedClient(_health(), states)
    frames_dir = tmp_path / "frames"
    args = _base_args(
        tmp_path,
        mode="offscreen",
        duration_sec=0.15,
        rate_hz=40.0,
        record=False,
        offscreen_save_frames_dir=frames_dir,
        offscreen_width=64,
        offscreen_height=48,
    )

    outcome = run_diagnostic(args, client=client)

    assert outcome.exit_code == 0
    saved = sorted(frames_dir.glob("frame_*.png"))
    assert len(saved) >= 1
    manifest = json.loads((frames_dir / "frames_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == len(saved)
    # 실제 리더 관절값(state.sequence)이 프레임 manifest에 그대로 기록되어야 한다 (요구사항: 프레임
    # timestamp/state sequence 기록).
    assert all(entry["remote_sequence"] is not None for entry in manifest)


def test_offscreen_mode_without_output_target_is_blocked(tmp_path):
    """--save-frames/--video-output이 둘 다 없으면 CLI에서 막지만, orchestrator 레벨에서도
    안전하게 BLOCKED로 끝나야 한다 (조용히 아무 것도 안 하고 성공한 척하지 않는다)."""
    client = _ScriptedClient(_health(), [_state(_positions(), _positions())])
    args = _base_args(tmp_path, mode="offscreen")  # save_frames/video 지정 없음
    outcome = run_diagnostic(args, client=client)
    assert outcome.final_result == "BLOCKED"
    assert outcome.exit_code == 1


def test_offscreen_frames_reflect_leader_motion(tmp_path):
    """리더암이 실제로 움직이면 저장된 프레임도 서로 달라야 한다 (정적 블랭크 프레임 반복이 아님)."""
    import numpy as np
    from PIL import Image

    states = [
        _state(_positions(wrist_flex=v), _positions(wrist_flex=0.0), sequence=i)
        for i, v in enumerate([0.0, 30.0, 60.0, 90.0])
    ]
    client = _ScriptedClient(_health(), states)
    frames_dir = tmp_path / "frames"
    args = _base_args(
        tmp_path,
        mode="offscreen",
        # mujoco.Renderer의 첫 프레임은 OpenGL context/shader warm-up 때문에 느릴 수 있어
        # (수십~백여 ms), 최소 2프레임을 확보하도록 넉넉히 잡는다.
        duration_sec=1.0,
        rate_hz=20.0,
        record=False,
        offscreen_save_frames_dir=frames_dir,
        offscreen_width=64,
        offscreen_height=48,
    )
    run_diagnostic(args, client=client)

    saved = sorted(frames_dir.glob("frame_*.png"))
    assert len(saved) >= 2
    first = np.array(Image.open(saved[0]))
    last = np.array(Image.open(saved[-1]))
    assert not np.array_equal(first, last)


# ---------------------------------------------------------------------------
# mode == "gui" 실패 경로 (실제 창을 띄우지 않고 launch_passive를 목으로 대체)
# ---------------------------------------------------------------------------


def test_gui_mode_reports_blocked_when_launch_passive_fails(tmp_path, monkeypatch):
    import mujoco.viewer as mj_viewer

    def _boom(model, data):
        raise RuntimeError("가짜 GLFW 실패 (테스트 전용)")

    monkeypatch.setattr(mj_viewer, "launch_passive", _boom)

    client = _ScriptedClient(_health(), [_state(_positions(), _positions())])
    args = _base_args(tmp_path, mode="gui")
    outcome = run_diagnostic(args, client=client)

    assert outcome.final_result == "BLOCKED"
    assert outcome.exit_code == 1


def test_gui_mode_blocked_when_not_called_from_main_thread(tmp_path):
    """요구사항 4번: GUI는 반드시 main thread에서 생성해야 한다 - 다른 스레드에서 시도하면
    실제 GLFW 호출까지 가지 않고 바로 BLOCKED로 막혀야 한다."""
    client = _ScriptedClient(_health(), [_state(_positions(), _positions())])
    args = _base_args(tmp_path, mode="gui")

    result: dict = {}

    def _run():
        result["outcome"] = run_diagnostic(args, client=client)

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=10)

    assert result["outcome"].final_result == "BLOCKED"
    assert result["outcome"].exit_code == 1
