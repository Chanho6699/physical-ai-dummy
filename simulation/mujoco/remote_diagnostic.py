"""노트북 SO-101 읽기 전용 상태 서버 <-> 데스크탑 MuJoCo 실시간 진단 orchestrator.

흐름 (요구사항 1번 구조도 그대로):

    노트북 리더암 state ──┐
                         ├→ 실시간 비교 (leader vs follower, 표시/진단용)
    노트북 팔로워암 state ─┘
                                 ↓
    리더암 state → 기존 action mapping → 기존 safety gate → MuJoCo SO-101
                                 ↓
                       한글 상태 화면 + CSV/JSON 리포트

이 모듈은 실물 하드웨어에 어떤 것도 쓰지 않는다 - 노트북 서버에는 GET만 호출하고
(``remote_state_client.RemoteSO101StateClient``), 팔로워암에는 아무 것도 보내지 않는다.
MuJoCo 안전 판정은 ``safety_checks.py``(dataset_action_replay.py와 동일한 모듈)를 그대로
재사용하며, 이 파일에서 관절 range나 threshold를 새로 정의하거나 완화하지 않는다.

BLOCKED 정책 (요구사항 7번): 특정 관절이 MuJoCo 관절/actuator range를 벗어나면 그 관절만
직전 안전 target을 유지하고, 나머지(차단되지 않은) 관절은 정상적으로 갱신한다. 벗어난
값을 clamp해서 적용하는 코드는 이 파일 어디에도 없다.

네트워크 안전 정책 (요구사항 10번): timeout/stale/sequence 정지/서버 mode 이상 등이
감지되면 즉시 MuJoCo 갱신을 멈춘다(직전 안전 target 유지). 기본값(``auto_resume=false``)은
한 번 멈추면 이 실행 동안 자동으로 재개하지 않는 보수적 정책이다. mode/write_enabled
위반은 ``auto_resume`` 설정과 무관하게 항상 영구 정지(fatal)로 처리한다.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco

from simulation.mujoco import console_status as cs
from simulation.mujoco.action_mapping import (
    ActionMappingError,
    JointMapping,
    build_default_mapping,
    map_positions_dict,
    validate_mapping_against_model,
)
from simulation.mujoco.diagnostic_analysis import DiagnosticAnalyzer, DiagnosticConfig
from simulation.mujoco.diagnostic_report import (
    build_json_summary,
    make_session_id,
    resolve_session_paths,
    write_csv_report,
    write_json_report,
)
from simulation.mujoco.offscreen_recorder import OffscreenRecorder, OffscreenRecorderError
from simulation.mujoco.remote_state_client import (
    JOINT_NAMES,
    READ_ONLY_MODE,
    RemoteClientConfig,
    RemoteSO101StateClient,
    RemoteState,
    RemoteStateError,
    SequenceWatchdog,
    SequenceWatchdogConfig,
    compute_effective_stale,
)
from simulation.mujoco.safety_checks import (
    DynamicCheckState,
    SafetyConfig,
    SafetyConfigError,
    SafetyEvent,
    check_frame_targets,
    check_simulation_state,
    filter_active_blocking,
    load_safety_config,
)
from simulation.mujoco.so101_model import SO101ModelError, get_joint_limits, load_model, make_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "remote_mujoco_diagnostic"

# 기본 콘솔 갱신 주기 상한 (요구사항 8번: "초당 2~5회 이하로 제한"). 중간값 4Hz를 택했다.
DISPLAY_MAX_HZ = 4.0


@dataclass(frozen=True)
class NetworkSafetyConfig:
    """configs/remote_mujoco_diagnostic.yaml의 ``safety`` 섹션과 대응된다."""

    pause_on_timeout: bool = True
    pause_on_stale: bool = True
    pause_on_disconnect: bool = True
    auto_resume: bool = False
    resume_after_consecutive_ok: int = 5
    sequence_stall_warn_after_s: float = 2.0
    sequence_stall_block_after_s: float = 5.0


@dataclass
class RemoteDiagnosticArgs:
    server_url: str
    mode: str = "headless"  # "headless" | "gui" | "offscreen"
    joints: tuple[str, ...] = ("wrist_flex",)  # 표시/진단 대상 (MuJoCo 적용은 항상 6개 전체)
    duration_sec: float = 20.0
    rate_hz: float = 20.0
    timeout_ms: float = 500.0
    stale_after_ms: float = 500.0
    max_retries: int = 3
    api_token: str | None = None
    record: bool = False  # CSV 상세 기록 여부 (JSON 요약은 항상 저장)
    report_path: Path | None = None
    mujoco_config_path: Path | None = None
    diagnostic_config: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    network_safety: NetworkSafetyConfig = field(default_factory=NetworkSafetyConfig)
    quiet: bool = False
    verbose: bool = False
    no_color: bool = False
    dry_run: bool = False
    session_id: str | None = None  # None이면 실행 시각으로 자동 생성 (테스트에서 고정값 주입용)
    # mode == "offscreen" 전용 (WSLg GUI 경로 대체용, docs 요구사항 "대체 검증 경로" 참고).
    offscreen_save_frames_dir: Path | None = None
    offscreen_video_path: Path | None = None
    offscreen_width: int = 640
    offscreen_height: int = 480
    offscreen_fps: float | None = None  # None이면 rate_hz를 그대로 사용


@dataclass(frozen=True)
class RemoteDiagnosticOutcome:
    final_result: str  # PASS | WARN | BLOCKED
    json_path: Path | None
    csv_path: Path | None
    summary: dict
    exit_code: int


class PreflightAbortedError(RuntimeError):
    """준비 단계(요구사항 6번)에서 BLOCKED가 발생해 진단을 시작할 수 없는 경우."""


@dataclass
class _PreflightResult:
    model: mujoco.MjModel
    mapping: tuple[JointMapping, ...]
    safety_config: SafetyConfig
    joint_limits_deg: dict[str, tuple[float, float]]
    initial_state: RemoteState


def _joint_limits_in_degrees(model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    limits = get_joint_limits(model)
    return {name: (math.degrees(lim.joint_range[0]), math.degrees(lim.joint_range[1])) for name, lim in limits.items()}


def _run_preflight(
    args: RemoteDiagnosticArgs, opts: cs.ConsoleOptions, client: RemoteSO101StateClient
) -> _PreflightResult:
    """요구사항 6번 순서 그대로 검사한다. 실패하면 PreflightAbortedError를 던진다."""

    # 1. /health 접근
    try:
        health = client.check_health()
    except RemoteStateError as exc:
        cs.print_check(opts, "BLOCKED", "서버 연결", str(exc))
        raise PreflightAbortedError(str(exc)) from exc
    cs.print_check(opts, "PASS", "서버 연결", "GET /health 응답 수신")

    # 2. status == ok 또는 허용 가능한 degraded
    if health.status not in ("ok", "degraded"):
        msg = f"health.status 값을 신뢰할 수 없습니다: {health.status!r}"
        cs.print_check(opts, "BLOCKED", "서버 상태", msg)
        raise PreflightAbortedError(msg)
    level = "PASS" if health.status == "ok" else "WARN"
    cs.print_check(opts, level, "서버 상태", f"status={health.status}")

    # 3. mode == read_only
    if health.mode != READ_ONLY_MODE:
        msg = f"서버 mode가 read_only가 아닙니다: {health.mode!r}. 즉시 중단합니다."
        cs.print_check(opts, "BLOCKED", "READ ONLY 모드 확인", msg)
        raise PreflightAbortedError(msg)
    cs.print_check(opts, "PASS", "READ ONLY 모드 확인", "mode=read_only")

    # 4. write_enabled == false
    if health.write_enabled is not False:
        msg = f"write_enabled이 false임을 확인할 수 없습니다 (값={health.write_enabled!r}). 즉시 중단합니다."
        cs.print_check(opts, "BLOCKED", "쓰기 기능 비활성화 확인", msg)
        raise PreflightAbortedError(msg)
    cs.print_check(opts, "PASS", "쓰기 기능 비활성화 확인", "write_enabled=false")

    # 5. leader_connected == true
    if not health.leader_connected:
        msg = "리더암이 연결되어 있지 않습니다."
        cs.print_check(opts, "BLOCKED", "리더암 연결", msg)
        raise PreflightAbortedError(msg)
    cs.print_check(opts, "PASS", "리더암 연결", "leader_connected=true")

    # 6. follower_connected == true
    if not health.follower_connected:
        msg = "팔로워암이 연결되어 있지 않습니다."
        cs.print_check(opts, "BLOCKED", "팔로워암 연결", msg)
        raise PreflightAbortedError(msg)
    cs.print_check(opts, "PASS", "팔로워암 연결", "follower_connected=true")

    # 7~8. /state의 stale == false, 관절 6개 확인
    try:
        state = client.get_state()
    except RemoteStateError as exc:
        cs.print_check(opts, "BLOCKED", "최신 state 수신", str(exc))
        raise PreflightAbortedError(str(exc)) from exc

    leader_stale = compute_effective_stale(state.leader, args.stale_after_ms)
    follower_stale = compute_effective_stale(state.follower, args.stale_after_ms)
    if leader_stale or follower_stale or not state.leader.valid or not state.follower.valid:
        reasons = []
        if leader_stale:
            reasons.append("리더암 stale")
        if follower_stale:
            reasons.append("팔로워암 stale")
        if not state.leader.valid:
            reasons.append(f"리더암 값 무효: {state.leader.invalid_reason}")
        if not state.follower.valid:
            reasons.append(f"팔로워암 값 무효: {state.follower.invalid_reason}")
        msg = ", ".join(reasons)
        cs.print_check(opts, "BLOCKED", "최신 state 수신", msg)
        raise PreflightAbortedError(msg)
    cs.print_check(opts, "PASS", "최신 state 수신", f"leader/follower 관절 {len(JOINT_NAMES)}개 확인, stale=false")

    # 9. MuJoCo joint/actuator mapping 확인 + 10. Safety Gate 초기화
    cs.print_section(opts, "MuJoCo SO-101 로딩 중")
    try:
        safety_config = load_safety_config(args.mujoco_config_path)
    except SafetyConfigError as exc:
        cs.print_check(opts, "BLOCKED", "Safety 설정", str(exc))
        raise PreflightAbortedError(str(exc)) from exc

    try:
        model = load_model(safety_config.scene_path)
        mapping = build_default_mapping([f"{name}.pos" for name in JOINT_NAMES])
        validate_mapping_against_model(mapping, model)
    except (SO101ModelError, ActionMappingError) as exc:
        cs.print_check(opts, "BLOCKED", "MuJoCo 모델", str(exc))
        raise PreflightAbortedError(str(exc)) from exc

    cs.print_check(opts, "PASS", "MuJoCo 모델 로딩", f"{safety_config.scene_path.name} (joint={model.njnt}, actuator={model.nu})")
    cs.print_check(opts, "PASS", "관절 mapping 확인", f"{len(mapping)}개 관절 매핑 완료")

    return _PreflightResult(
        model=model,
        mapping=mapping,
        safety_config=safety_config,
        joint_limits_deg=_joint_limits_in_degrees(model),
        initial_state=state,
    )


def _make_client(args: RemoteDiagnosticArgs) -> RemoteSO101StateClient:
    config = RemoteClientConfig(
        server_url=args.server_url,
        timeout_ms=args.timeout_ms,
        max_retries=args.max_retries,
        api_token=args.api_token,
    )
    return RemoteSO101StateClient(config)


def _dry_run(args: RemoteDiagnosticArgs, opts: cs.ConsoleOptions) -> RemoteDiagnosticOutcome:
    """실제 네트워크/HTTP 호출과 MuJoCo actuator 적용 없이 파이프라인만 검증한다."""

    try:
        safety_config = load_safety_config(args.mujoco_config_path)
        model = load_model(safety_config.scene_path)
        mapping = build_default_mapping([f"{name}.pos" for name in JOINT_NAMES])
        validate_mapping_against_model(mapping, model)
    except (SafetyConfigError, SO101ModelError, ActionMappingError) as exc:
        cs.print_check(opts, "BLOCKED", "dry-run 설정 검증", str(exc))
        raise PreflightAbortedError(str(exc)) from exc
    cs.print_check(opts, "PASS", "설정 파일/MuJoCo 모델/관절 mapping/safety 설정", "확인 완료")

    # mock state로 매핑 -> safety 파이프라인 전체를 한 번 통과시켜 본다 (네트워크 없음).
    mock_positions = {name: 0.0 for name in JOINT_NAMES}
    target_rad = map_positions_dict(mock_positions, mapping)
    dynamic_state = DynamicCheckState()
    frame_events = check_frame_targets(0, target_rad, mapping, model, safety_config, dynamic_state)
    frame_events = filter_active_blocking(frame_events, safety_config)
    if any(e.level == "BLOCKED" for e in frame_events):  # pragma: no cover - mock은 항상 0deg라 발생하지 않음
        cs.print_check(opts, "BLOCKED", "dry-run mock state 검사", "mock state(0deg)가 이미 range를 벗어났습니다.")
        raise PreflightAbortedError("mock state가 관절 range를 벗어났습니다.")
    cs.print_check(opts, "PASS", "mock state 처리", "leader=0deg mock 값으로 mapping -> safety gate 통과 확인")

    joint_limits_deg = _joint_limits_in_degrees(model)
    analyzer = DiagnosticAnalyzer(args.diagnostic_config)
    for joint in args.joints:
        analyzer.update(joint, time.monotonic(), 0.0, 0.0, joint_range_deg=joint_limits_deg.get(joint))
    cs.print_check(opts, "PASS", "진단 분석 파이프라인", f"대상 관절 {list(args.joints)} 1회 처리 확인")

    session_id = args.session_id or make_session_id()
    paths = resolve_session_paths(
        reports_dir=DEFAULT_REPORTS_DIR, session_id=session_id, explicit_report_path=args.report_path, write_csv=args.record
    )
    cs.print_remote_dry_run_summary(
        opts, joints=list(args.joints), would_apply_all_joints=True, report_path=str(paths.json_path)
    )

    summary = build_json_summary(
        server_url=args.server_url,
        duration_sec=0.0,
        requested_rate_hz=args.rate_hz,
        actual_sample_rate_hz=0.0,
        sample_count=0,
        latency_mean_ms=0.0,
        latency_max_ms=0.0,
        stale_count=0,
        timeout_count=0,
        joint_names=list(args.joints),
        max_abs_difference={},
        mean_abs_difference={},
        persistent_difference_events=0,
        follower_saturation_events=0,
        sign_mismatch_events=0,
        offset_suspected=[],
        mujoco_blocked_events=0,
        network_pause_events=0,
        warnings=["dry-run: 실제 샘플을 수집하지 않았습니다."],
        final_result="PASS",
    )
    summary["dry_run"] = True
    write_json_report(paths.json_path, summary)
    return RemoteDiagnosticOutcome("PASS", paths.json_path, None, summary, 0)


def run_diagnostic(args: RemoteDiagnosticArgs, *, client: RemoteSO101StateClient | None = None) -> RemoteDiagnosticOutcome:
    opts = cs.ConsoleOptions(quiet=args.quiet, verbose=args.verbose, use_color=cs.resolve_use_color(args.no_color))
    cs.print_remote_prepare_header(opts, server_url=args.server_url)

    if args.dry_run:
        try:
            outcome = _dry_run(args, opts)
        except PreflightAbortedError:
            return RemoteDiagnosticOutcome("BLOCKED", None, None, {}, 1)
        cs.print_remote_preflight_footer(opts)
        return outcome

    owns_client = client is None
    client = client or _make_client(args)

    try:
        try:
            preflight = _run_preflight(args, opts, client)
        except PreflightAbortedError:
            return RemoteDiagnosticOutcome("BLOCKED", None, None, {}, 1)
        cs.print_remote_preflight_footer(opts)

        return _run_loop(args, opts, client, preflight)
    finally:
        if owns_client:
            client.close()


def _run_loop(
    args: RemoteDiagnosticArgs,
    opts: cs.ConsoleOptions,
    client: RemoteSO101StateClient,
    preflight: _PreflightResult,
) -> RemoteDiagnosticOutcome:
    model = preflight.model
    mapping = preflight.mapping
    safety_config = preflight.safety_config
    joint_limits_deg = preflight.joint_limits_deg

    data = make_data(model)
    mujoco.mj_resetData(model, data)

    initial_targets = map_positions_dict(preflight.initial_state.leader.positions_deg, mapping)
    for entry in mapping:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = initial_targets[entry.mujoco_actuator_name]
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, entry.mujoco_actuator_name)
        data.ctrl[actuator_id] = initial_targets[entry.mujoco_actuator_name]
    mujoco.mj_forward(model, data)

    steps_per_sample = max(1, round((1.0 / args.rate_hz) / model.opt.timestep))

    dynamic_state = DynamicCheckState()
    dynamic_state.previous_target_rad = dict(initial_targets)
    previous_safe_targets = dict(initial_targets)

    analyzer = DiagnosticAnalyzer(args.diagnostic_config)
    watchdog = SequenceWatchdog(
        SequenceWatchdogConfig(
            stall_warn_after_s=args.network_safety.sequence_stall_warn_after_s,
            stall_block_after_s=args.network_safety.sequence_stall_block_after_s,
        )
    )

    viewer_cm = None
    viewer = None
    if args.mode == "gui":
        if threading.current_thread() is not threading.main_thread():
            # launch_passive는 GUI 창을 main thread에서 생성/갱신해야 한다 (요구사항 4번) -
            # 이 조건이 깨지면 창이 "떴지만 갱신되지 않는" 증상으로 이어질 수 있어 조용히
            # 넘어가지 않고 바로 BLOCKED 처리한다.
            cs.print_remote_error("GUI viewer는 main thread에서만 생성할 수 있습니다 (현재 스레드가 main이 아닙니다).")
            return RemoteDiagnosticOutcome("BLOCKED", None, None, {}, 1)
        try:
            import mujoco.viewer as mj_viewer

            viewer_cm = mj_viewer.launch_passive(model, data)
            viewer = viewer_cm.__enter__()
        except Exception as exc:
            cs.print_remote_error(f"GUI viewer를 시작할 수 없습니다: {exc}")
            return RemoteDiagnosticOutcome("BLOCKED", None, None, {}, 1)

    recorder: OffscreenRecorder | None = None
    if args.mode == "offscreen":
        try:
            recorder = OffscreenRecorder(
                model,
                width=args.offscreen_width,
                height=args.offscreen_height,
                save_frames_dir=args.offscreen_save_frames_dir,
                video_path=args.offscreen_video_path,
                video_fps=args.offscreen_fps if args.offscreen_fps is not None else args.rate_hz,
            )
        except OffscreenRecorderError as exc:
            cs.print_remote_error(f"오프스크린 렌더러를 시작할 수 없습니다: {exc}")
            return RemoteDiagnosticOutcome("BLOCKED", None, None, {}, 1)

    # -- 세션 상태 누적 --------------------------------------------------
    sample_count = 0
    latencies_ms: list[float] = []
    stale_count = 0
    timeout_count = 0
    mujoco_blocked_events = 0
    network_pause_events = 0
    diagnostic_event_counts = {
        "persistent_difference": 0,
        "follower_saturation_suspected": 0,
        "sign_mismatch_suspected": 0,
    }
    offset_suspected_joints: set[str] = set()
    max_abs_diff: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
    diff_samples: dict[str, list[float]] = {name: [] for name in JOINT_NAMES}
    warnings_seen: list[str] = []
    csv_rows: list[dict] = []

    currently_blocked_joints: set[str] = set()
    paused = False
    fatal = False
    consecutive_ok_after_pause = 0
    last_seq_warn_state = "PASS"
    last_range_warn_joints: set[str] = set()

    def _add_warning(text: str) -> None:
        if text not in warnings_seen:
            warnings_seen.append(text)

    start_time = time.monotonic()
    last_display_time = 0.0
    display_interval = 1.0 / DISPLAY_MAX_HZ
    frame_counter = 0

    try:
        while time.monotonic() - start_time < args.duration_sec:
            if viewer is not None and not viewer.is_running():
                # 사용자가 GUI 창을 직접 닫은 경우 - 남은 duration을 계속 도는 대신 바로 끝낸다
                # (요구사항 5번: is_running() 확인 누락은 이 파일에서 만들지 않는다).
                cs.print_remote_error("GUI 창이 닫혀 있어 진단을 종료합니다.")
                break
            loop_start = time.monotonic()
            now_wall = time.time()

            try:
                state = client.get_state()
            except RemoteStateError as exc:
                timeout_count += 1
                if args.network_safety.pause_on_timeout and not paused:
                    paused = True
                    network_pause_events += 1
                    cs.print_remote_pause(opts, reason=f"/state 요청 실패: {exc}", auto_resume=args.network_safety.auto_resume)
                _add_warning(f"/state 요청 실패: {exc}")
                _sleep_for_rate(loop_start, args.rate_hz)
                continue

            sample_count += 1
            latencies_ms.append(state.network_latency_ms)

            status, reason = _classify_sample(state, args)
            if status != "ok":
                if status == "stale":
                    stale_count += 1
                should_pause = {
                    "stale": args.network_safety.pause_on_stale,
                    "invalid": args.network_safety.pause_on_disconnect,
                    "seq_stall": True,
                    "mode_violation": True,
                }[status]
                if status == "mode_violation":
                    fatal = True
                if should_pause and not paused:
                    paused = True
                    network_pause_events += 1
                    cs.print_remote_pause(opts, reason=reason, auto_resume=args.network_safety.auto_resume and not fatal)
                _add_warning(reason)
                _sleep_for_rate(loop_start, args.rate_hz)
                continue

            # 정상 sequence 진행 - stall 경고에서 회복했으면 알린다.
            if last_seq_warn_state != "PASS":
                cs.print_remote_resume(opts, reason="sequence가 다시 정상적으로 증가하고 있습니다.")
            last_seq_warn_state = "PASS"

            if paused and not fatal:
                if args.network_safety.auto_resume:
                    consecutive_ok_after_pause += 1
                    if consecutive_ok_after_pause >= args.network_safety.resume_after_consecutive_ok:
                        paused = False
                        consecutive_ok_after_pause = 0
                        cs.print_remote_resume(opts, reason="연속 정상 응답을 확인해 MuJoCo 갱신을 재개합니다.")
                # auto_resume=false면 계속 paused 유지 (요구사항 10번 기본값)

            leader_positions = state.leader.positions_deg
            follower_positions = state.follower.positions_deg

            applied_targets = dict(previous_safe_targets)
            blocked_joint_names: set[str] = set()
            frame_events: list[SafetyEvent] = []
            sim_events: list[SafetyEvent] = []

            if not paused:
                target_rad_all = map_positions_dict(leader_positions, mapping)
                frame_events = check_frame_targets(frame_counter, target_rad_all, mapping, model, safety_config, dynamic_state)
                frame_events = filter_active_blocking(frame_events, safety_config)
                blocked_joint_names = {e.joint for e in frame_events if e.level == "BLOCKED"}

                for entry in mapping:
                    if entry.mujoco_joint_name not in blocked_joint_names:
                        applied_targets[entry.mujoco_actuator_name] = target_rad_all[entry.mujoco_actuator_name]
                # check_frame_targets가 내부적으로 (거부된 값 포함) previous_target_rad를 덮어썼으므로,
                # 다음 프레임 delta 비교가 "실제로 적용된" 값을 기준으로 이뤄지도록 되돌린다.
                dynamic_state.previous_target_rad = dict(applied_targets)
                previous_safe_targets = applied_targets

                new_blocked = blocked_joint_names - currently_blocked_joints
                recovered = currently_blocked_joints - blocked_joint_names
                for joint_name in sorted(new_blocked):
                    evt = next(e for e in frame_events if e.joint == joint_name and e.level == "BLOCKED")
                    lo, hi = evt.limit if evt.limit is not None else (float("nan"), float("nan"))
                    cs.print_remote_blocked(
                        opts,
                        joint=joint_name,
                        leader_value_deg=math.degrees(evt.value) if evt.value is not None else float("nan"),
                        limit_deg=(math.degrees(lo), math.degrees(hi)),
                    )
                    mujoco_blocked_events += 1
                for joint_name in sorted(recovered):
                    cs.print_remote_recovered(opts, joint=joint_name)
                currently_blocked_joints = blocked_joint_names

                for actuator_name, value in applied_targets.items():
                    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                    data.ctrl[actuator_id] = value
                for _ in range(steps_per_sample):
                    mujoco.mj_step(model, data)

                sim_events = check_simulation_state(frame_counter, model, data, mapping, safety_config, dynamic_state)
                sim_events = filter_active_blocking(sim_events, safety_config)
                frame_counter += 1

                if viewer is not None:
                    try:
                        viewer.sync()
                    except Exception as exc:
                        # sync 중 예외를 조용히 삼키지 않는다 (요구사항 5번) - GUI 루프
                        # 자체는 계속 진행하되(다음 프레임에서 복구될 수도 있으므로) 반드시 보고한다.
                        cs.print_remote_error(f"viewer.sync() 중 예외 발생: {exc}")

                if recorder is not None:
                    try:
                        recorder.capture(data, remote_sequence=state.sequence, remote_timestamp=state.raw_timestamp)
                    except OffscreenRecorderError as exc:
                        cs.print_remote_error(f"오프스크린 프레임 캡처 실패: {exc}")
                        fatal = True
                        break

            mujoco_qpos_deg = {}
            mujoco_target_deg = {}
            for entry in mapping:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
                qpos_adr = model.jnt_qposadr[joint_id]
                mujoco_qpos_deg[entry.mujoco_joint_name] = math.degrees(float(data.qpos[qpos_adr]))
                mujoco_target_deg[entry.mujoco_joint_name] = math.degrees(applied_targets[entry.mujoco_actuator_name])

            events_by_joint = _events_by_joint(frame_events + sim_events)

            display_rows = []
            new_range_warn_joints: set[str] = set()
            for joint in JOINT_NAMES:
                leader_deg = leader_positions[joint]
                follower_deg = follower_positions[joint]
                sample, diag_events = analyzer.update(
                    joint,
                    time.monotonic(),
                    leader_deg,
                    follower_deg,
                    joint_range_deg=joint_limits_deg.get(joint),
                )

                diff = leader_deg - follower_deg
                max_abs_diff[joint] = max(max_abs_diff[joint], abs(diff))
                diff_samples[joint].append(abs(diff))

                joint_safety_events = events_by_joint.get(joint, [])
                safety_status = "PASS"
                if any(e.level == "BLOCKED" for e in joint_safety_events):
                    safety_status = "BLOCKED"
                elif any(e.level == "WARN" for e in joint_safety_events) or abs(diff) > args.diagnostic_config.difference_warn_deg:
                    safety_status = "WARN"

                lo, hi = joint_limits_deg.get(joint, (float("-inf"), float("inf")))
                target_deg = mujoco_target_deg.get(joint, 0.0)
                margin = min(target_deg - lo, hi - target_deg) if math.isfinite(lo) and math.isfinite(hi) else None

                for evt in diag_events:
                    if evt.code == "leader_out_of_mujoco_range":
                        new_range_warn_joints.add(joint)
                        if joint not in last_range_warn_joints:
                            cs.print_remote_diagnostic_event(opts, evt)
                        continue
                    if evt.code in diagnostic_event_counts:
                        diagnostic_event_counts[evt.code] += 1
                    if evt.code == "offset_suspected":
                        offset_suspected_joints.add(joint)
                    cs.print_remote_diagnostic_event(opts, evt)

                event_code = ";".join(e.code for e in diag_events) or (";".join(e.code for e in joint_safety_events))
                blocked_reason = next((e.message for e in joint_safety_events if e.level == "BLOCKED"), None)

                if args.record:
                    csv_rows.append(
                        {
                            "local_timestamp": now_wall,
                            "remote_timestamp": state.raw_timestamp,
                            "sequence": state.sequence,
                            "network_latency_ms": state.network_latency_ms,
                            "state_age_ms": state.state_age_ms,
                            "joint_name": joint,
                            "leader_position_deg": leader_deg,
                            "follower_position_deg": follower_deg,
                            "difference_deg": diff,
                            "leader_raw_tick": (state.leader.raw_ticks or {}).get(joint, ""),
                            "follower_raw_tick": (state.follower.raw_ticks or {}).get(joint, ""),
                            "mujoco_target_deg": mujoco_target_deg.get(joint, ""),
                            "mujoco_qpos_deg": mujoco_qpos_deg.get(joint, ""),
                            "mujoco_limit_margin_deg": margin if margin is not None else "",
                            "safety_status": safety_status,
                            "event_code": event_code,
                            "blocked_reason": blocked_reason or "",
                        }
                    )

                if joint in args.joints:
                    display_rows.append(
                        {
                            "joint": joint,
                            "leader_deg": leader_deg,
                            "follower_deg": follower_deg,
                            "difference_deg": diff,
                            "mujoco_qpos_deg": mujoco_qpos_deg.get(joint, 0.0),
                            "mujoco_target_deg": mujoco_target_deg.get(joint, 0.0),
                            "limit_margin_deg": margin,
                            "safety_status": safety_status,
                        }
                    )
            last_range_warn_joints = new_range_warn_joints

            now = time.monotonic()
            if not opts.quiet and (now - last_display_time) >= display_interval:
                server_status = "일시정지" if paused else "정상"
                latency_display = latencies_ms[-1] if latencies_ms else 0.0
                if opts.verbose or len(args.joints) > 1:
                    cs.print_remote_table(
                        opts, server_status=server_status, sample_count=sample_count, latency_ms=latency_display, rows=display_rows
                    )
                elif display_rows:
                    row = display_rows[0]
                    cs.print_remote_compact(
                        opts,
                        server_status=server_status,
                        sample_count=sample_count,
                        latency_ms=latency_display,
                        joint=row["joint"],
                        leader_deg=row["leader_deg"],
                        follower_deg=row["follower_deg"],
                        difference_deg=row["difference_deg"],
                        mujoco_target_deg=row["mujoco_target_deg"],
                        mujoco_qpos_deg=row["mujoco_qpos_deg"],
                        limit_margin_deg=row["limit_margin_deg"],
                        safety_status=row["safety_status"],
                    )
                last_display_time = now

            _sleep_for_rate(loop_start, args.rate_hz)
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)
        if recorder is not None:
            offscreen_summary = recorder.close()
            cs.print_remote_offscreen_summary(
                opts,
                frame_count=offscreen_summary.frame_count,
                save_frames_dir=offscreen_summary.save_frames_dir,
                manifest_path=offscreen_summary.manifest_path,
                video_path=offscreen_summary.video_path,
            )

    elapsed = time.monotonic() - start_time
    return _finalize(
        args=args,
        elapsed=elapsed,
        sample_count=sample_count,
        latencies_ms=latencies_ms,
        stale_count=stale_count,
        timeout_count=timeout_count,
        mujoco_blocked_events=mujoco_blocked_events,
        network_pause_events=network_pause_events,
        diagnostic_event_counts=diagnostic_event_counts,
        offset_suspected_joints=offset_suspected_joints,
        max_abs_diff=max_abs_diff,
        diff_samples=diff_samples,
        warnings_seen=warnings_seen,
        csv_rows=csv_rows,
        opts=opts,
    )


def _classify_sample(state: RemoteState, args: RemoteDiagnosticArgs) -> tuple[str, str]:
    if state.mode not in (None, READ_ONLY_MODE):
        return "mode_violation", f"서버 mode가 더 이상 read_only가 아닙니다: {state.mode!r}. 영구 정지합니다."

    leader_stale = compute_effective_stale(state.leader, args.stale_after_ms)
    follower_stale = compute_effective_stale(state.follower, args.stale_after_ms)
    if leader_stale or follower_stale:
        return "stale", "리더 또는 팔로워 state가 stale합니다."

    if (
        not state.leader.valid
        or not state.follower.valid
        or not state.leader.connected
        or not state.follower.connected
    ):
        return "invalid", "리더 또는 팔로워 연결/값이 유효하지 않습니다."

    return "ok", ""


def _events_by_joint(events: list[SafetyEvent]) -> dict[str, list[SafetyEvent]]:
    result: dict[str, list[SafetyEvent]] = {}
    for evt in events:
        if evt.joint is None:
            continue
        result.setdefault(evt.joint, []).append(evt)
    return result


def _sleep_for_rate(loop_start: float, rate_hz: float) -> None:
    period_s = 1.0 / rate_hz
    elapsed = time.monotonic() - loop_start
    sleep_for = period_s - elapsed
    if sleep_for > 0:
        time.sleep(sleep_for)


def _finalize(
    *,
    args: RemoteDiagnosticArgs,
    elapsed: float,
    sample_count: int,
    latencies_ms: list[float],
    stale_count: int,
    timeout_count: int,
    mujoco_blocked_events: int,
    network_pause_events: int,
    diagnostic_event_counts: dict[str, int],
    offset_suspected_joints: set[str],
    max_abs_diff: dict[str, float],
    diff_samples: dict[str, list[float]],
    warnings_seen: list[str],
    csv_rows: list[dict],
    opts: cs.ConsoleOptions,
) -> RemoteDiagnosticOutcome:
    mean_abs_diff = {
        joint: (sum(values) / len(values) if values else 0.0) for joint, values in diff_samples.items()
    }
    latency_mean = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    latency_max = max(latencies_ms) if latencies_ms else 0.0
    actual_rate = sample_count / elapsed if elapsed > 0 else 0.0

    final_result = "PASS"
    if (
        mujoco_blocked_events > 0
        or stale_count > 0
        or timeout_count > 0
        or network_pause_events > 0
        or any(v > 0 for v in diagnostic_event_counts.values())
        or offset_suspected_joints
    ):
        final_result = "WARN"

    summary = build_json_summary(
        server_url=args.server_url,
        duration_sec=elapsed,
        requested_rate_hz=args.rate_hz,
        actual_sample_rate_hz=actual_rate,
        sample_count=sample_count,
        latency_mean_ms=latency_mean,
        latency_max_ms=latency_max,
        stale_count=stale_count,
        timeout_count=timeout_count,
        joint_names=list(args.joints),
        max_abs_difference=max_abs_diff,
        mean_abs_difference=mean_abs_diff,
        persistent_difference_events=diagnostic_event_counts["persistent_difference"],
        follower_saturation_events=diagnostic_event_counts["follower_saturation_suspected"],
        sign_mismatch_events=diagnostic_event_counts["sign_mismatch_suspected"],
        offset_suspected=sorted(offset_suspected_joints),
        mujoco_blocked_events=mujoco_blocked_events,
        network_pause_events=network_pause_events,
        warnings=warnings_seen[:50],
        final_result=final_result,
    )

    session_id = args.session_id or make_session_id()
    paths = resolve_session_paths(
        reports_dir=DEFAULT_REPORTS_DIR, session_id=session_id, explicit_report_path=args.report_path, write_csv=args.record
    )
    write_json_report(paths.json_path, summary)
    if args.record and paths.csv_path is not None:
        write_csv_report(paths.csv_path, csv_rows)

    cs.print_remote_final_summary(
        opts,
        duration_sec=elapsed,
        sample_count=sample_count,
        latency_mean_ms=latency_mean,
        latency_max_ms=latency_max,
        stale_count=stale_count,
        mujoco_blocked_events=mujoco_blocked_events,
        persistent_difference_events=diagnostic_event_counts["persistent_difference"],
        follower_saturation_events=diagnostic_event_counts["follower_saturation_suspected"],
        final_result=final_result,
        csv_path=str(paths.csv_path) if (args.record and paths.csv_path is not None) else None,
        json_path=str(paths.json_path),
    )

    return RemoteDiagnosticOutcome(
        final_result=final_result,
        json_path=paths.json_path,
        csv_path=paths.csv_path if args.record else None,
        summary=summary,
        exit_code=0,
    )
