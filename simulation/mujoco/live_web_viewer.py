"""노트북 SO-101 상태 -> MuJoCo -> 브라우저 실시간 MJPEG 스트리밍 뷰어.

WSLg `mujoco.viewer` 창이 Windows에서 정상적으로 보이지 않는 문제(``docs/remote_mujoco_diagnostic.md``
10절)를 우회하기 위한 대체 경로다. ``mujoco.viewer``는 이 파일에서 아예 import하지 않는다 -
대신 이미 offscreen 경로에서 검증된 ``mujoco.Renderer``로 프레임을 만들고, 표준 라이브러리
``http.server``만으로 MJPEG(``multipart/x-mixed-replace``)를 스트리밍한다. 이 방식은 Chrome이
``<img>`` 태그에서 네이티브로 지원하며 추가 의존성이 필요 없다.

스레드 구조 (요구사항 4번 - 네트워크 polling과 렌더링 분리):

    [network 스레드]  RemoteSO101StateClient.get_state() 반복 폴링
                       -> _SharedSnapshot(스레드 세이프 최신 상태 1개만 보관)
    [render 스레드]    _SharedSnapshot 최신값 읽기 -> mapping/safety gate 통과
                       -> mj_step -> mujoco.Renderer -> JPEG 인코딩
                       -> _FrameBuffer(스레드 세이프 최신 JPEG 1장만 보관)
                       -> _StatusBoard 갱신 (지연/시퀀스/관절값/safety/fps)
    [HTTP 서버 스레드(들)] ThreadingHTTPServer - 연결마다 스레드 1개.
                       /stream.mjpg 는 _FrameBuffer를 반복 polling만 하고, mujoco 객체는
                       절대 건드리지 않는다 (mujoco.MjModel/MjData/Renderer는 render 스레드
                       전용 - GL context를 여러 스레드에서 동시에 건드리지 않기 위함).

이 모듈은 노트북에 GET만 보내고(``remote_state_client.py`` 그대로 재사용), 팔로워암에는
어떤 명령도 쓰지 않는다. BLOCKED 관절은 ``safety_checks.py``를 그대로 재사용해 직전 안전
target을 유지한다 - 이 파일에서 range/threshold를 새로 정의하지 않는다.
"""

from __future__ import annotations

import io
import json
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco

from simulation.mujoco.action_mapping import (
    ActionMappingError,
    JointMapping,
    build_default_mapping,
    map_positions_dict,
    validate_mapping_against_model,
)
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
from simulation.mujoco.safety_event_tracker import (
    CONNECTION_WIDE_JOINT,
    SafetyEventTracker,
    SafetyEventTrackerConfig,
    SafetyIssue,
    classify_frame_event_reason,
    make_event_session_id,
    pick_most_severe_event,
    resolve_safety_event_paths,
    write_safety_events_csv,
    write_safety_events_json,
)
from simulation.mujoco.follower_safe_mapper import (
    ConnectionHoldInput,
    FollowerSafeMapper,
    FollowerSafeMapperConfig,
    FollowerSafeMapperError,
    FollowerSafeRecorder,
    load_config_yaml as load_follower_safe_mapper_config_yaml,
    load_follower_calibration,
    make_report_session_id,
    print_follower_safe_debug,
    resolve_follower_safe_report_paths,
    summarize_hold,
)
from simulation.mujoco.so101_model import SO101ModelError, get_joint_limits, load_model, make_data
from simulation.realism.so101_control_profile import (
    DEFAULT_PROFILE_PATH,
    ControlProfileError,
    load_control_profile,
)
from simulation.realism.so101_realistic_control import RealisticControlConfig, RealisticControlLayer

MJPEG_BOUNDARY = "so101frame"
COMMAND_SOURCE_RAW_LEADER = "raw-leader"
COMMAND_SOURCE_FOLLOWER_SAFE = "follower-safe"
VALID_COMMAND_SOURCES = (COMMAND_SOURCE_RAW_LEADER, COMMAND_SOURCE_FOLLOWER_SAFE)
DEFAULT_EVENTS_REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "remote_mujoco_diagnostic"

# "Realistic SO-101 Control Layer" (simulation/realism/*) 선택 옵션 - 기본값은 항상
# baseline(=기존 동작 그대로, 이 레이어를 생성조차 하지 않음)이다. 실제 SO-101에는 어떤
# 영향도 없다 (이 뷰어 자체가 팔로워에 쓰기를 하지 않는다 - 모듈 docstring 참고).
CONTROL_MODE_BASELINE = "baseline"
CONTROL_MODE_REALISTIC = "realistic"
VALID_CONTROL_MODES = (CONTROL_MODE_BASELINE, CONTROL_MODE_REALISTIC)


class LiveWebViewerError(RuntimeError):
    """preflight(모델/서버 연결) 실패 - 항상 한글 메시지로 호출자가 그대로 출력할 수 있게 한다."""


@dataclass
class WebViewerArgs:
    server_url: str
    joints: tuple[str, ...] = ("wrist_flex",)  # 화면/상태 표시 대상 (MuJoCo 적용은 항상 6개 전체)
    host: str = "0.0.0.0"
    port: int = 8080
    fps: float = 20.0  # 렌더링 목표 FPS
    rate_hz: float = 20.0  # 네트워크 폴링 주기 (렌더링과 분리됨)
    timeout_ms: float = 500.0
    stale_after_ms: float = 500.0
    max_retries: int = 3
    api_token: str | None = None
    mujoco_config_path: Path | None = None
    frame_width: int = 640
    frame_height: int = 480
    jpeg_quality: int = 80
    debug_control: bool = False  # 1초마다 [제어 진단] 블록을 stdout에 출력 (토큰/인증정보/원격 응답 원문은 출력 안 함)

    # sequence stall 판정 - remote_diagnostic.py의 NetworkSafetyConfig와 동일한 기본값을 그대로
    # 재사용한다 (새 threshold 아님, configs/remote_mujoco_diagnostic.yaml의 safety 섹션 재사용).
    sequence_stall_warn_after_s: float = 2.0
    sequence_stall_block_after_s: float = 5.0

    # WARN/BLOCKED 원인 추적 - 표시/기록용 설정이지 safety 판정 threshold가 아니다.
    safety_event_config: SafetyEventTrackerConfig = field(default_factory=SafetyEventTrackerConfig)
    events_report_dir: Path | None = None  # None이면 reports/remote_mujoco_diagnostic 사용

    # 실제 팔로워에 보낼 "예정"인 안전 명령 매퍼 - MuJoCo에만 적용된다 (실물 쓰기 없음).
    # "raw-leader"(기본, 기존 동작)면 이 필드들은 쓰이지 않는다.
    command_source: str = COMMAND_SOURCE_RAW_LEADER
    safe_mapper_config_path: Path | None = None  # None이면 configs/follower_safe_mapper.yaml
    follower_safe_report_dir: Path | None = None  # None이면 reports/remote_mujoco_diagnostic

    # "Realistic SO-101 Control Layer" (섹션: control_mode). baseline(기본)이면 이 필드들은
    # 전혀 쓰이지 않고 기존 raw-leader 경로가 그대로 동작한다 (레이어를 생성조차 하지 않음
    # - 기존 동작과 완전히 동일해야 함). command_source=="follower-safe"에는 아직 적용하지
    # 않는다 (섹션 9 - 두 메커니즘의 목적이 다르므로 이번 v1에서는 raw-leader 경로에만 붙인다).
    control_mode: str = CONTROL_MODE_BASELINE
    realism_profile_path: Path | None = None  # None이면 DEFAULT_PROFILE_PATH (candidate v1)
    realism_enable_latency: bool = True
    realism_enable_deadband: bool = True
    realism_enable_rate_limit: bool = True
    realism_enable_historical_range_diagnostic: bool = True

    def __post_init__(self) -> None:
        if self.command_source not in VALID_COMMAND_SOURCES:
            raise ValueError(f"command_source는 {VALID_COMMAND_SOURCES} 중 하나여야 합니다: {self.command_source!r}")
        if self.control_mode not in VALID_CONTROL_MODES:
            raise ValueError(f"control_mode는 {VALID_CONTROL_MODES} 중 하나여야 합니다: {self.control_mode!r}")


# ---------------------------------------------------------------------------
# 스레드 세이프 공유 상태
# ---------------------------------------------------------------------------


class _SharedSnapshot:
    """network 스레드가 쓰고 render 스레드가 읽는 "최신 state 1개"만 보관한다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: RemoteState | None = None
        self._error: str | None = None
        self._received_monotonic: float | None = None

    def set_state(self, state: RemoteState) -> None:
        with self._lock:
            self._state = state
            self._error = None
            self._received_monotonic = time.monotonic()

    def set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def get(self) -> tuple[RemoteState | None, str | None, float | None]:
        """(state, last_error, 마지막 성공 수신 이후 경과 초)."""
        with self._lock:
            age = time.monotonic() - self._received_monotonic if self._received_monotonic is not None else None
            return self._state, self._error, age


class _FrameBuffer:
    """render 스레드가 쓰고 HTTP 스레드(들)가 읽는 "최신 JPEG 1장"만 보관한다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._generation = 0

    def publish(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg_bytes
            self._generation += 1

    def snapshot(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._generation


@dataclass
class _JointStatus:
    leader_deg: float | None = None
    follower_deg: float | None = None
    # requested_target_deg: 이번 프레임에 리더 값을 그대로 mapping만 거친 "계산된" target
    # (safety gate 통과 전). mujoco_target_deg: 실제로 data.ctrl에 쓰인 "적용된" target
    # (BLOCKED면 직전 안전값 그대로, requested와 달라질 수 있다). 요구사항: 이 둘을 웹
    # 화면에서 명확히 구분해서 보여준다 - BLOCKED로 이전 값이 유지되는 경우를 바로 알 수 있게.
    requested_target_deg: float | None = None
    mujoco_target_deg: float | None = None  # == applied target (data.ctrl에 실제로 쓰인 값)
    mujoco_qpos_deg: float | None = None
    limit_margin_deg: float | None = None
    safety_status: str = "PASS"
    blocked: bool = False  # 이번 프레임에 이 관절이 BLOCKED라 requested != applied인지

    # command_source == "follower-safe"에서만 채워지는 필드들 (요구사항 9번).
    follower_current_deg: float | None = None
    rate_limited: bool = False
    range_held: bool = False
    connection_held: bool = False
    follower_hold_reason: str | None = None


class _StatusBoard:
    """render 스레드가 갱신하고 /status 핸들러가 읽는 표시용 상태 (요구사항 2번 필드 전부)."""

    def __init__(self, joints: tuple[str, ...], *, command_source: str = COMMAND_SOURCE_RAW_LEADER) -> None:
        self._lock = threading.Lock()
        self._joints = joints
        self._command_source = command_source
        self._server_status = "연결 대기 중"
        self._connected = False
        self._sequence: int | None = None
        self._network_latency_ms: float | None = None
        self._stale = False
        self._fps = 0.0
        self._last_error: str | None = None
        self._joint_status: dict[str, _JointStatus] = {name: _JointStatus() for name in joints}
        self._frame_count = 0
        self._follower_safe_hold = False
        self._follower_safe_hold_reason: str | None = None
        self._follower_safe_intervention_count = 0
        self._active_joint_count = 0
        self._held_joint_count = 0
        self._held_joints: dict[str, str] = {}

    def update(
        self,
        *,
        server_status: str,
        connected: bool,
        sequence: int | None,
        network_latency_ms: float | None,
        stale: bool,
        fps: float,
        last_error: str | None,
        joint_status: dict[str, _JointStatus],
        follower_safe_hold: bool = False,
        follower_safe_hold_reason: str | None = None,
        follower_safe_intervention_count: int = 0,
        active_joint_count: int = 0,
        held_joint_count: int = 0,
        held_joints: dict[str, str] | None = None,
    ) -> None:
        with self._lock:
            self._server_status = server_status
            self._connected = connected
            self._sequence = sequence
            self._network_latency_ms = network_latency_ms
            self._stale = stale
            self._fps = fps
            self._last_error = last_error
            self._joint_status = joint_status
            self._frame_count += 1
            self._follower_safe_hold = follower_safe_hold
            self._follower_safe_hold_reason = follower_safe_hold_reason
            self._follower_safe_intervention_count = follower_safe_intervention_count
            self._active_joint_count = active_joint_count
            self._held_joint_count = held_joint_count
            self._held_joints = held_joints or {}

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "server_status": self._server_status,
                "connected": self._connected,
                "sequence": self._sequence,
                "network_latency_ms": self._network_latency_ms,
                "stale": self._stale,
                "fps": round(self._fps, 1),
                "last_error": self._last_error,
                "frame_count": self._frame_count,
                "command_source": self._command_source,
                "hold": self._follower_safe_hold,
                "hold_reason": self._follower_safe_hold_reason,
                "intervention_count": self._follower_safe_intervention_count,
                # 요구사항 스키마 그대로 (global_hold는 hold의 별칭 - 관절 하나만의
                # UNVERIFIED_RANGE/RANGE_VIOLATION 등으로는 절대 true가 되지 않는다).
                "global_hold": self._follower_safe_hold,
                "global_hold_reason": self._follower_safe_hold_reason,
                "active_joint_count": self._active_joint_count,
                "held_joint_count": self._held_joint_count,
                "held_joints": dict(self._held_joints),
                "joints": {
                    name: {
                        "leader_deg": js.leader_deg,
                        "follower_deg": js.follower_deg,
                        "follower_current_deg": js.follower_current_deg,
                        "requested_target_deg": js.requested_target_deg,
                        "mujoco_target_deg": js.mujoco_target_deg,
                        "mujoco_qpos_deg": js.mujoco_qpos_deg,
                        "limit_margin_deg": js.limit_margin_deg,
                        "safety_status": js.safety_status,
                        "blocked": js.blocked,
                        "rate_limited": js.rate_limited,
                        "range_held": js.range_held,
                        "connection_held": js.connection_held,
                        "follower_hold_reason": js.follower_hold_reason,
                    }
                    for name, js in self._joint_status.items()
                },
                "timestamp": time.time(),
            }


# ---------------------------------------------------------------------------
# 메인 오케스트레이터
# ---------------------------------------------------------------------------


class LiveWebViewer:
    """네트워크 폴링 스레드 + MuJoCo 렌더 스레드를 관리한다. HTTP 서버는 별도(run_*.py)에서 붙인다."""

    def __init__(self, args: WebViewerArgs, *, client: RemoteSO101StateClient | None = None) -> None:
        self.args = args
        self._owns_client = client is None
        self._client = client or RemoteSO101StateClient(
            RemoteClientConfig(
                server_url=args.server_url,
                timeout_ms=args.timeout_ms,
                max_retries=args.max_retries,
                api_token=args.api_token,
            )
        )
        self._snapshot = _SharedSnapshot()
        self.frames = _FrameBuffer()
        self.status = _StatusBoard(args.joints, command_source=args.command_source)
        self.safety_tracker = SafetyEventTracker(args.safety_event_config)
        self.last_safety_report_paths: tuple[Path, Path] | None = None

        self._stop_event = threading.Event()
        self._network_thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None

        self._model: mujoco.MjModel | None = None
        self._mapping: tuple[JointMapping, ...] | None = None
        self._safety_config: SafetyConfig | None = None
        self._joint_limits_deg: dict[str, tuple[float, float]] = {}

        # command_source == "follower-safe"에서만 채워진다 (preflight 참고).
        self.follower_mapper: FollowerSafeMapper | None = None
        self._follower_recorder: FollowerSafeRecorder | None = None
        self.last_follower_safe_report_paths: tuple[Path, Path] | None = None

        # control_mode == "realistic"에서만 채워진다 (preflight 참고) - baseline은 None으로
        # 유지되어 render loop가 기존 raw-leader 경로를 그대로 탄다.
        self.realistic_control: RealisticControlLayer | None = None
        self.last_realism_diagnostics: dict[str, object] = {}

        self.preflight_log: list[str] = []  # (level, label, message) 대신 사람이 읽을 문자열만 축적

    # -- preflight -----------------------------------------------------

    def preflight(self) -> None:
        """서버 연결 + MuJoCo 모델/mapping/safety 로딩. 실패하면 LiveWebViewerError."""
        try:
            health = self._client.check_health()
        except RemoteStateError as exc:
            raise LiveWebViewerError(f"노트북 서버에 연결할 수 없습니다: {exc}") from exc

        if health.status not in ("ok", "degraded"):
            raise LiveWebViewerError(f"health.status 값을 신뢰할 수 없습니다: {health.status!r}")
        if health.mode != READ_ONLY_MODE:
            raise LiveWebViewerError(f"서버 mode가 read_only가 아닙니다: {health.mode!r}. 중단합니다.")
        if health.write_enabled is not False:
            raise LiveWebViewerError(f"write_enabled=false를 확인할 수 없습니다 (값={health.write_enabled!r}).")
        if not health.leader_connected:
            raise LiveWebViewerError("리더암이 연결되어 있지 않습니다.")
        if not health.follower_connected:
            raise LiveWebViewerError("팔로워암이 연결되어 있지 않습니다.")

        try:
            safety_config = load_safety_config(self.args.mujoco_config_path)
            model = load_model(safety_config.scene_path)
            mapping = build_default_mapping([f"{name}.pos" for name in JOINT_NAMES])
            validate_mapping_against_model(mapping, model)
        except (SafetyConfigError, SO101ModelError, ActionMappingError) as exc:
            raise LiveWebViewerError(f"MuJoCo 모델/safety 설정 로딩 실패: {exc}") from exc

        self._model = model
        self._mapping = mapping
        self._safety_config = safety_config
        limits = get_joint_limits(model)
        self._joint_limits_deg = {
            name: (math.degrees(lim.joint_range[0]), math.degrees(lim.joint_range[1])) for name, lim in limits.items()
        }

        if self.args.command_source == COMMAND_SOURCE_FOLLOWER_SAFE:
            self._setup_follower_safe_mapper()

        if self.args.control_mode == CONTROL_MODE_REALISTIC:
            self._setup_realistic_control()

    def _setup_realistic_control(self) -> None:
        profile_path = self.args.realism_profile_path or DEFAULT_PROFILE_PATH
        try:
            profile = load_control_profile(profile_path)
        except ControlProfileError as exc:
            raise LiveWebViewerError(f"realistic control profile 로딩 실패 ({profile_path}): {exc}") from exc
        config = RealisticControlConfig(
            enable_latency=self.args.realism_enable_latency,
            enable_deadband=self.args.realism_enable_deadband,
            enable_rate_limit=self.args.realism_enable_rate_limit,
            enable_historical_range_diagnostic=self.args.realism_enable_historical_range_diagnostic,
        )
        self.realistic_control = RealisticControlLayer(profile, config)

    def _apply_realistic_control(
        self,
        target_rad_all: dict[str, float],
        mapping: tuple[JointMapping, ...],
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[dict[str, float], dict[str, object]]:
        """raw command(target_rad_all, rad)를 Realistic Control Layer에 통과시킨다.

        섹션 3 원칙대로: rad -> (기존 action_mapping과 동일한 scale의) degree/percent
        변환 -> RealisticControlLayer.process() -> degree/percent -> rad 역변환.
        새 스케일을 만들지 않는다 - math.degrees/radians는 action_mapping.py가 쓰는
        scale=pi/180의 정확한 역변환이라 gripper(percent가 "degree"로 잘못 붙어 있는
        기존 관례 그대로)에도 그대로 재사용할 수 있다 (simulation/mujoco/action_mapping.py
        모듈 docstring 참고 - 이 파일이 그 관례를 새로 만들지 않았다).
        """
        assert self.realistic_control is not None
        desired_native = {entry.mujoco_actuator_name: math.degrees(target_rad_all[entry.mujoco_actuator_name]) for entry in mapping}

        simulated_actual_native: dict[str, float] = {}
        for entry in mapping:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
            qpos_adr = model.jnt_qposadr[joint_id]
            simulated_actual_native[entry.mujoco_joint_name] = math.degrees(float(data.qpos[qpos_adr]))

        result = self.realistic_control.process(desired_native, now=time.monotonic(), simulated_actual=simulated_actual_native)

        processed_rad = {name: math.radians(value) for name, value in result.processed_action.items()}
        diagnostics_dict = {joint: d.__dict__ for joint, d in result.diagnostics.items()}
        return processed_rad, diagnostics_dict

    def _setup_follower_safe_mapper(self) -> None:
        config_path = self.args.safe_mapper_config_path or (
            Path(__file__).resolve().parents[2] / "configs" / "follower_safe_mapper.yaml"
        )
        try:
            mapper_config = load_follower_safe_mapper_config_yaml(config_path)
            calibrations = load_follower_calibration(
                calibration_file_path=mapper_config.calibration_file_path,
                fallback_raw_range=mapper_config.fallback_raw_range,
                motor_resolution=mapper_config.motor_resolution,
            )
        except FollowerSafeMapperError as exc:
            raise LiveWebViewerError(f"follower-safe 매퍼 설정/캘리브레이션 로딩 실패: {exc}") from exc
        self.follower_mapper = FollowerSafeMapper(mapper_config, calibrations)
        self._follower_recorder = FollowerSafeRecorder()

    # -- 시작/종료 -------------------------------------------------------

    def start(self) -> None:
        if self._model is None:
            raise LiveWebViewerError("preflight()를 먼저 호출해야 합니다.")
        self._stop_event.clear()
        self._network_thread = threading.Thread(target=self._network_loop, name="web-viewer-network", daemon=True)
        self._render_thread = threading.Thread(target=self._render_loop, name="web-viewer-render", daemon=True)
        self._network_thread.start()
        self._render_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._network_thread is not None:
            self._network_thread.join(timeout=2.0)
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
        # render 스레드의 finally에서 이미 finalize()가 불렸겠지만, 렌더 루프가 한 번도 돌지
        # 못하고 끝난 경우(예: renderer 생성 실패) 등을 대비해 안전하게 다시 호출한다 - 활성
        # 이벤트가 없으면 no-op이다.
        self.safety_tracker.finalize(now_wall=time.time())
        self._write_safety_events_report()
        if self._follower_recorder is not None:
            self._write_follower_safe_report()
        if self._owns_client:
            self._client.close()

    def _write_safety_events_report(self) -> tuple[Path, Path]:
        events = self.safety_tracker.all_events()
        reports_dir = self.args.events_report_dir or DEFAULT_EVENTS_REPORTS_DIR
        session_id = make_event_session_id()
        json_path, csv_path = resolve_safety_event_paths(reports_dir, session_id)
        write_safety_events_json(json_path, events)
        write_safety_events_csv(csv_path, events)
        self.last_safety_report_paths = (json_path, csv_path)
        return json_path, csv_path

    def _write_follower_safe_report(self) -> tuple[Path, Path]:
        assert self._follower_recorder is not None
        reports_dir = self.args.follower_safe_report_dir or DEFAULT_EVENTS_REPORTS_DIR
        session_id = make_report_session_id()
        json_path, csv_path = resolve_follower_safe_report_paths(reports_dir, session_id)
        self._follower_recorder.write_json(json_path)
        self._follower_recorder.write_csv(csv_path)
        self.last_follower_safe_report_paths = (json_path, csv_path)
        return json_path, csv_path

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def realism_status_payload(self) -> dict:
        """``/status``가 노출하는 Realistic Control Layer 진단 스냅샷.

        control_mode=="baseline"이면 항상 빈 diagnostics(레이어를 생성조차 하지 않았으므로
        - 회귀 없음)를 반환한다."""
        return {
            "control_mode": self.args.control_mode,
            "realism_diagnostics": dict(self.last_realism_diagnostics),
        }

    def safety_status_payload(self) -> dict:
        """``/status``와 ``/events``가 공유하는 safety 이벤트 스냅샷."""
        now = time.time()
        return {
            "current_safety": self.safety_tracker.current_safety(),
            "recent_safety_events": self.safety_tracker.recent_events(now_wall=now),
            "safety_event_counts": self.safety_tracker.event_counts(now_wall=now),
        }

    # -- network 스레드 ---------------------------------------------------

    def _network_loop(self) -> None:
        period_s = 1.0 / self.args.rate_hz
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                state = self._client.get_state()
                self._snapshot.set_state(state)
            except RemoteStateError as exc:
                self._snapshot.set_error(str(exc))
            elapsed = time.monotonic() - loop_start
            self._stop_event.wait(max(0.0, period_s - elapsed))

    # -- render 스레드 -----------------------------------------------------

    def _render_loop(self) -> None:
        assert self._model is not None and self._mapping is not None and self._safety_config is not None
        model = self._model
        mapping = self._mapping
        safety_config = self._safety_config

        data = make_data(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        try:
            renderer = mujoco.Renderer(model, height=self.args.frame_height, width=self.args.frame_width)
        except Exception as exc:  # pragma: no cover - 렌더러 생성 실패는 환경 문제
            self._snapshot.set_error(f"오프스크린 렌더러 생성 실패: {exc}")
            self.status.update(
                server_status=f"오류: 렌더러 생성 실패 ({exc})",
                connected=False,
                sequence=None,
                network_latency_ms=None,
                stale=True,
                fps=0.0,
                last_error=str(exc),
                joint_status={name: _JointStatus() for name in self.args.joints},
            )
            return

        steps_per_frame = max(1, round((1.0 / self.args.fps) / model.opt.timestep))

        dynamic_state = DynamicCheckState()
        previous_safe_targets: dict[str, float] = {entry.mujoco_actuator_name: 0.0 for entry in mapping}
        dynamic_state.previous_target_rad = dict(previous_safe_targets)
        currently_blocked_joints: set[str] = set()
        fatal = False
        sequence_watchdog = SequenceWatchdog(
            SequenceWatchdogConfig(
                stall_warn_after_s=self.args.sequence_stall_warn_after_s,
                stall_block_after_s=self.args.sequence_stall_block_after_s,
            )
        )

        frame_interval = 1.0 / self.args.fps
        fps_window_start = time.monotonic()
        fps_window_count = 0
        measured_fps = 0.0
        frame_counter = 0
        last_debug_print = 0.0

        try:
            while not self._stop_event.is_set():
                loop_start = time.monotonic()

                state, last_error, age_s = self._snapshot.get()
                stale = True
                connected = False
                sequence = None
                latency_ms = None

                if state is not None:
                    connected = True
                    sequence = state.sequence
                    latency_ms = state.network_latency_ms
                    leader_stale = compute_effective_stale(state.leader, self.args.stale_after_ms)
                    follower_stale = compute_effective_stale(state.follower, self.args.stale_after_ms)
                    age_stale = age_s is not None and age_s * 1000.0 > self.args.stale_after_ms
                    stale = leader_stale or follower_stale or age_stale
                    if state.mode not in (None, READ_ONLY_MODE):
                        fatal = True
                        last_error = f"서버 mode가 더 이상 read_only가 아닙니다: {state.mode!r}. 갱신을 영구 중지합니다."

                apply_update = state is not None and not stale and not fatal and state.leader.valid and state.follower.valid

                # requested_target_rad_all: 리더 값을 mapping만 거친 "계산된" target (safety gate
                # 이전). stale/paused 중에도 "지금 리더가 어디 있는지"는 계속 보여주기 위해
                # apply_update 여부와 무관하게 leader 값이 있으면 항상 계산한다 - 요구사항 3번
                # (표시되는 target이 계산 직후 값인지 실제 적용값인지 명확히 구분).
                requested_target_rad_all: dict[str, float] | None = None
                missing_joint_name: str | None = None
                if state is not None and state.leader.positions_deg:
                    try:
                        requested_target_rad_all = map_positions_dict(state.leader.positions_deg, mapping)
                    except KeyError as exc:
                        missing_joint_name = str(exc.args[0]) if exc.args else str(exc)
                        last_error = f"리더 관절값에 예상한 관절이 없습니다: {missing_joint_name}"
                        requested_target_rad_all = None

                invalid_reason: str | None = None
                if state is not None and (not state.leader.valid or not state.follower.valid):
                    invalid_reason = state.leader.invalid_reason or state.follower.invalid_reason

                seq_watchdog_status = sequence_watchdog.observe(sequence, time.monotonic()) if state is not None else "PASS"
                mode_violation = state is not None and state.mode not in (None, READ_ONLY_MODE)

                applied_targets = dict(previous_safe_targets)
                frame_events: list[SafetyEvent] = []
                sim_events: list[SafetyEvent] = []
                follower_mapper_results: dict[str, object] | None = None

                if self.args.command_source == COMMAND_SOURCE_FOLLOWER_SAFE:
                    assert self.follower_mapper is not None and self._follower_recorder is not None
                    connection_hold = ConnectionHoldInput(
                        remote_stale=stale,
                        sequence_stalled=seq_watchdog_status in ("WARN", "BLOCKED"),
                        mode_not_read_only=mode_violation,
                        connection_lost=not connected,
                    )
                    follower_mapper_results = self.follower_mapper.step(
                        now=time.time(),
                        leader_positions_deg=state.leader.positions_deg if state is not None else None,
                        follower_positions_deg=state.follower.positions_deg if state is not None else None,
                        connection_hold=connection_hold,
                    )
                    # F단계(limited_command_deg)만 MuJoCo에 적용한다 - 요구사항 8번.
                    for entry in mapping:
                        sample = follower_mapper_results.get(entry.mujoco_joint_name)
                        if sample is not None and sample.limited_command_deg is not None:
                            applied_targets[entry.mujoco_actuator_name] = math.radians(sample.limited_command_deg)
                    previous_safe_targets = applied_targets

                    for actuator_name, value in applied_targets.items():
                        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                        data.ctrl[actuator_id] = value
                    for _ in range(steps_per_frame):
                        mujoco.mj_step(model, data)
                    frame_counter += 1

                    mujoco_actual_deg = {}
                    for entry in mapping:
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
                        qpos_adr = model.jnt_qposadr[joint_id]
                        mujoco_actual_deg[entry.mujoco_joint_name] = math.degrees(float(data.qpos[qpos_adr]))
                    self._follower_recorder.record(
                        now_wall=time.time(),
                        remote_sequence=sequence,
                        samples=follower_mapper_results,
                        follower_current_deg=state.follower.positions_deg if state is not None else None,
                        mujoco_actual_deg=mujoco_actual_deg,
                    )
                elif apply_update and requested_target_rad_all is not None:
                    target_rad_all = requested_target_rad_all
                    if self.realistic_control is not None:
                        target_rad_all, self.last_realism_diagnostics = self._apply_realistic_control(
                            target_rad_all, mapping, model, data
                        )
                    frame_events = check_frame_targets(frame_counter, target_rad_all, mapping, model, safety_config, dynamic_state)
                    frame_events = filter_active_blocking(frame_events, safety_config)
                    blocked_joint_names = {e.joint for e in frame_events if e.level == "BLOCKED"}

                    for entry in mapping:
                        if entry.mujoco_joint_name not in blocked_joint_names:
                            applied_targets[entry.mujoco_actuator_name] = target_rad_all[entry.mujoco_actuator_name]
                    dynamic_state.previous_target_rad = dict(applied_targets)
                    previous_safe_targets = applied_targets
                    currently_blocked_joints = blocked_joint_names

                    for actuator_name, value in applied_targets.items():
                        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                        data.ctrl[actuator_id] = value
                    for _ in range(steps_per_frame):
                        mujoco.mj_step(model, data)

                    sim_events = check_simulation_state(frame_counter, model, data, mapping, safety_config, dynamic_state)
                    sim_events = filter_active_blocking(sim_events, safety_config)
                    frame_counter += 1
                # apply_update=False(또는 mapping 실패)면 target/step 없이 마지막 pose를 그대로
                # 렌더링만 한다 (요구사항: 오래된 snapshot은 표시하되 MuJoCo target 갱신은 중지).

                renderer.update_scene(data)
                frame = renderer.render()
                jpeg_bytes = _encode_jpeg(frame, quality=self.args.jpeg_quality)
                self.frames.publish(jpeg_bytes)

                events_by_joint: dict[str, list[SafetyEvent]] = {}
                for evt in frame_events + sim_events:
                    if evt.joint is not None:
                        events_by_joint.setdefault(evt.joint, []).append(evt)

                issues = _build_safety_issues(
                    mapping=mapping,
                    joint_limits_deg=self._joint_limits_deg,
                    requested_target_rad_all=requested_target_rad_all,
                    applied_targets=applied_targets,
                    events_by_joint=events_by_joint,
                    currently_blocked_joints=currently_blocked_joints,
                    near_limit_margin_deg=self.args.safety_event_config.near_limit_margin_deg,
                    stale=stale,
                    connected=connected,
                    remote_age_ms=(age_s * 1000.0 if age_s is not None else None),
                    missing_joint_name=missing_joint_name,
                    invalid_reason=invalid_reason,
                    seq_watchdog_status=seq_watchdog_status,
                    mode_violation=mode_violation,
                )
                self.safety_tracker.observe(now_wall=time.time(), remote_sequence=sequence, issues=issues)

                joint_status: dict[str, _JointStatus] = {}
                for joint in self.args.joints:
                    js = _JointStatus()
                    if state is not None:
                        js.leader_deg = state.leader.positions_deg.get(joint) if state.leader.positions_deg else None
                        js.follower_deg = state.follower.positions_deg.get(joint) if state.follower.positions_deg else None
                    entry = next((e for e in mapping if e.mujoco_joint_name == joint), None)
                    if entry is not None:
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
                        qpos_adr = model.jnt_qposadr[joint_id]
                        js.mujoco_qpos_deg = math.degrees(float(data.qpos[qpos_adr]))
                        if requested_target_rad_all is not None:
                            js.requested_target_deg = math.degrees(requested_target_rad_all[entry.mujoco_actuator_name])
                        target_rad = applied_targets.get(entry.mujoco_actuator_name)
                        if target_rad is not None:
                            js.mujoco_target_deg = math.degrees(target_rad)  # 실제 data.ctrl에 쓰인 값
                            lo, hi = self._joint_limits_deg.get(joint, (float("-inf"), float("inf")))
                            if math.isfinite(lo) and math.isfinite(hi):
                                js.limit_margin_deg = min(js.mujoco_target_deg - lo, hi - js.mujoco_target_deg)
                    js.blocked = joint in currently_blocked_joints
                    joint_evts = events_by_joint.get(joint, [])
                    if any(e.level == "BLOCKED" for e in joint_evts) or js.blocked:
                        js.safety_status = "BLOCKED"
                    elif any(e.level == "WARN" for e in joint_evts):
                        js.safety_status = "WARN"

                    if follower_mapper_results is not None:
                        sample = follower_mapper_results.get(joint)
                        if sample is not None:
                            js.follower_current_deg = state.follower.positions_deg.get(joint) if (state is not None and state.follower.positions_deg) else None
                            js.requested_target_deg = sample.mapped_follower_deg  # B단계(좌표 변환) 결과
                            js.mujoco_target_deg = sample.limited_command_deg  # F단계(최종 적용) 결과
                            js.rate_limited = sample.rate_limited
                            js.range_held = sample.range_held
                            js.connection_held = sample.connection_held
                            js.follower_hold_reason = sample.hold_reason
                            js.blocked = sample.hold
                            if sample.range_min_deg is not None and sample.range_max_deg is not None and sample.limited_command_deg is not None:
                                js.limit_margin_deg = min(
                                    sample.limited_command_deg - sample.range_min_deg, sample.range_max_deg - sample.limited_command_deg
                                )
                            else:
                                js.limit_margin_deg = None
                            js.safety_status = "BLOCKED" if sample.hold else ("WARN" if sample.rate_limited else "PASS")
                    joint_status[joint] = js

                fps_window_count += 1
                now = time.monotonic()
                if now - fps_window_start >= 1.0:
                    measured_fps = fps_window_count / (now - fps_window_start)
                    fps_window_count = 0
                    fps_window_start = now

                if self.args.debug_control and now - last_debug_print >= 1.0:
                    last_debug_print = now
                    if follower_mapper_results is not None:
                        print_follower_safe_debug(follower_mapper_results)
                    else:
                        _print_control_debug(
                            model=model,
                            data=data,
                            mapping=mapping,
                            selected_joints=self.args.joints,
                            requested_target_rad_all=requested_target_rad_all,
                            applied_targets=applied_targets,
                            blocking_joints=currently_blocked_joints,
                            apply_update=apply_update,
                            stale=stale,
                            fatal=fatal,
                            connected=connected,
                        )

                if fatal:
                    server_status = f"영구 정지: {last_error}"
                elif not connected:
                    server_status = f"연결 끊김{f' ({last_error})' if last_error else ''}"
                elif stale:
                    server_status = "일시정지 (stale)"
                else:
                    server_status = "정상"

                follower_safe_intervention_count = 0
                hold_summary = {"global_hold": False, "global_hold_reason": None, "active_joint_count": 0, "held_joint_count": 0, "held_joints": {}}
                if follower_mapper_results is not None:
                    assert self.follower_mapper is not None
                    follower_safe_intervention_count = self.follower_mapper.intervention_count
                    # global_hold은 ARM_WIDE_HOLD_REASONS(REMOTE_STALE/SEQUENCE_STALLED/
                    # MODE_NOT_READ_ONLY/CONNECTION_LOST)만 본다 - UNVERIFIED_RANGE(예: gripper)
                    # 처럼 관절 하나만의 문제는 절대 global_hold로 승격되지 않는다 (요구사항).
                    hold_summary = summarize_hold(follower_mapper_results)

                self.status.update(
                    server_status=server_status,
                    connected=connected,
                    sequence=sequence,
                    network_latency_ms=latency_ms,
                    stale=stale,
                    fps=measured_fps,
                    last_error=last_error,
                    joint_status=joint_status,
                    follower_safe_hold=hold_summary["global_hold"],
                    follower_safe_hold_reason=hold_summary["global_hold_reason"],
                    follower_safe_intervention_count=follower_safe_intervention_count,
                    active_joint_count=hold_summary["active_joint_count"],
                    held_joint_count=hold_summary["held_joint_count"],
                    held_joints=hold_summary["held_joints"],
                )

                elapsed = time.monotonic() - loop_start
                self._stop_event.wait(max(0.0, frame_interval - elapsed))
        finally:
            self.safety_tracker.finalize(now_wall=time.time())
            renderer.close()


def _build_safety_issues(
    *,
    mapping: tuple[JointMapping, ...],
    joint_limits_deg: dict[str, tuple[float, float]],
    requested_target_rad_all: dict[str, float] | None,
    applied_targets: dict[str, float],
    events_by_joint: dict[str, list[SafetyEvent]],
    currently_blocked_joints: set[str],
    near_limit_margin_deg: float,
    stale: bool,
    connected: bool,
    remote_age_ms: float | None,
    missing_joint_name: str | None,
    invalid_reason: str | None,
    seq_watchdog_status: str,
    mode_violation: bool,
) -> list[SafetyIssue]:
    """이번 프레임에 관측된 문제들을 SafetyIssue 목록으로 만든다 (판정 자체는 새로 하지 않음).

    관절별(JOINT_RANGE_LOW/HIGH, FRAME_DELTA_HIGH, INVALID_VALUE(simulation_nan), NEAR_JOINT_LIMIT)
    +연결 전체(REMOTE_STALE, SEQUENCE_STALLED, MODE_NOT_READ_ONLY, INVALID_VALUE, MISSING_JOINT)를
    합쳐서 반환한다. 한 관절에 이번 프레임 여러 SafetyEvent가 겹치면 더 심각한 것 하나만 쓴다.
    """
    issues: list[SafetyIssue] = []

    for entry in mapping:
        joint = entry.mujoco_joint_name
        requested_rad = requested_target_rad_all.get(entry.mujoco_actuator_name) if requested_target_rad_all else None
        applied_rad = applied_targets.get(entry.mujoco_actuator_name)
        requested_deg = math.degrees(requested_rad) if requested_rad is not None else None
        applied_deg = math.degrees(applied_rad) if applied_rad is not None else None

        lo, hi = joint_limits_deg.get(joint, (float("-inf"), float("inf")))
        joint_min_deg = lo if math.isfinite(lo) else None
        joint_max_deg = hi if math.isfinite(hi) else None
        margin_deg = None
        if applied_deg is not None and joint_min_deg is not None and joint_max_deg is not None:
            margin_deg = min(applied_deg - joint_min_deg, joint_max_deg - applied_deg)
        delta_deg = None
        if requested_deg is not None and applied_deg is not None:
            delta_deg = requested_deg - applied_deg

        picked = pick_most_severe_event(events_by_joint.get(joint, []))
        severity: str | None = None
        reason_code: str | None = None
        if picked is not None:
            severity = picked.level
            reason_code = classify_frame_event_reason(picked)
        elif joint in currently_blocked_joints:
            # 이번 프레임엔 새 SafetyEvent가 안 났지만(예: apply_update=False라 check_frame_targets가
            # 이번엔 안 돌았음) 여전히 직전 target을 유지 중인 BLOCKED 관절 - 원인을 이번 프레임
            # 정보만으로 확정할 수 없으므로 UNKNOWN_SAFETY_REASON.
            severity = "BLOCKED"
            reason_code = "UNKNOWN_SAFETY_REASON"
        elif margin_deg is not None and margin_deg < near_limit_margin_deg:
            severity = "WARN"
            reason_code = "NEAR_JOINT_LIMIT"

        issues.append(
            SafetyIssue(
                joint=joint,
                severity=severity,
                reason_code=reason_code,
                requested_target_deg=requested_deg,
                applied_target_deg=applied_deg,
                joint_min_deg=joint_min_deg,
                joint_max_deg=joint_max_deg,
                margin_deg=margin_deg,
                delta_deg=delta_deg,
                remote_age_ms=remote_age_ms,
                stale=stale,
            )
        )

    if missing_joint_name is not None:
        issues.append(
            SafetyIssue(joint=missing_joint_name, severity="WARN", reason_code="MISSING_JOINT", remote_age_ms=remote_age_ms, stale=stale)
        )
    if invalid_reason is not None:
        issues.append(
            SafetyIssue(joint=CONNECTION_WIDE_JOINT, severity="WARN", reason_code="INVALID_VALUE", remote_age_ms=remote_age_ms, stale=stale)
        )
    if stale and connected:
        issues.append(
            SafetyIssue(joint=CONNECTION_WIDE_JOINT, severity="WARN", reason_code="REMOTE_STALE", remote_age_ms=remote_age_ms, stale=stale)
        )
    if seq_watchdog_status in ("WARN", "BLOCKED"):
        issues.append(
            SafetyIssue(
                joint=CONNECTION_WIDE_JOINT, severity=seq_watchdog_status, reason_code="SEQUENCE_STALLED", remote_age_ms=remote_age_ms, stale=stale
            )
        )
    if mode_violation:
        issues.append(
            SafetyIssue(joint=CONNECTION_WIDE_JOINT, severity="BLOCKED", reason_code="MODE_NOT_READ_ONLY", remote_age_ms=remote_age_ms, stale=stale)
        )

    return issues


def _print_control_debug(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapping: tuple[JointMapping, ...],
    selected_joints: tuple[str, ...],
    requested_target_rad_all: dict[str, float] | None,
    applied_targets: dict[str, float],
    blocking_joints: set[str],
    apply_update: bool,
    stale: bool,
    fatal: bool,
    connected: bool,
) -> None:
    """1초마다 [제어 진단] 블록을 stdout에 출력한다 (``--debug-control``).

    토큰/인증정보/원격 응답 원문은 출력하지 않는다 - 여기서 다루는 값은 이미 mapping을
    거친 관절 이름 -> 각도(deg)/라디안 딕셔너리뿐이다.
    """
    mapped_targets_deg = (
        {entry.mujoco_joint_name: round(math.degrees(requested_target_rad_all[entry.mujoco_actuator_name]), 3) for entry in mapping}
        if requested_target_rad_all is not None
        else {}
    )
    applied_targets_deg = {
        entry.mujoco_joint_name: round(math.degrees(applied_targets[entry.mujoco_actuator_name]), 3)
        for entry in mapping
        if entry.mujoco_actuator_name in applied_targets
    }
    ctrl_deg: dict[str, float] = {}
    qpos_deg: dict[str, float] = {}
    for entry in mapping:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, entry.mujoco_actuator_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
        qpos_adr = model.jnt_qposadr[joint_id]
        ctrl_deg[entry.mujoco_joint_name] = round(math.degrees(float(data.ctrl[actuator_id])), 3)
        qpos_deg[entry.mujoco_joint_name] = round(math.degrees(float(data.qpos[qpos_adr])), 3)

    print("[제어 진단]")
    print(f"  selected_joints = {list(selected_joints)}")
    print(f"  mapped_targets(deg) = {mapped_targets_deg}")
    print(f"  blocking_joints = {sorted(blocking_joints)}")
    print(f"  applied_targets(deg) = {applied_targets_deg}")
    print(f"  data.ctrl(deg) = {ctrl_deg}")
    print(f"  data.qpos(deg) = {qpos_deg}")
    print(f"  apply_update={apply_update} stale={stale} fatal={fatal} connected={connected}", flush=True)


def _encode_jpeg(frame, *, quality: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTTP 서버 (표준 라이브러리 http.server만 사용 - 추가 의존성 없음)
# ---------------------------------------------------------------------------

_INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>SO-101 MuJoCo 실시간 뷰어</title>
<style>
  body {{ background:#111; color:#eee; font-family: -apple-system, "Malgun Gothic", sans-serif; margin:0; padding:16px; }}
  h1 {{ font-size:18px; margin:0 0 12px; }}
  .layout {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .video {{ flex: 0 0 auto; }}
  .video img {{ width:{width}px; height:{height}px; background:#000; border:1px solid #444; display:block; }}
  table {{ border-collapse: collapse; min-width:420px; }}
  td, th {{ border:1px solid #333; padding:4px 8px; font-size:13px; text-align:left; }}
  th {{ background:#222; }}
  .PASS {{ color:#4caf50; font-weight:bold; }}
  .WARN {{ color:#ffb300; font-weight:bold; }}
  .BLOCKED {{ color:#f44336; font-weight:bold; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:12px; }}
  #server_status {{ background:#333; }}
  .muted {{ color:#888; font-size:12px; }}
  tr.blocked-row {{ background:#3a1414; }}
  .blocked-note {{ color:#f44336; font-size:11px; }}
  .safety-summary {{ display:flex; gap:24px; align-items:baseline; margin:8px 0; font-size:14px; }}
  .events {{ max-height:420px; overflow-y:auto; min-width:340px; }}
  .event-card {{ border-left:4px solid #666; background:#1b1b1b; padding:6px 10px; margin-bottom:6px; font-size:12px; border-radius:2px; }}
  .event-card.BLOCKED {{ border-left-color:#f44336; }}
  .event-card.WARN {{ border-left-color:#ffb300; }}
  .event-head {{ font-weight:bold; }}
  .event-head.BLOCKED {{ color:#f44336; }}
  .event-head.WARN {{ color:#ffb300; }}
  .event-body {{ color:#ddd; margin:2px 0; }}
  .event-meta, .event-time {{ color:#888; font-size:11px; }}
  .command-source-banner {{ display:flex; flex-wrap:wrap; gap:24px; align-items:baseline; margin:4px 0; font-size:14px; padding:6px 10px; background:#1b1b1b; border-radius:4px; }}
  .hold-true {{ color:#f44336; font-weight:bold; }}
  .hold-false {{ color:#4caf50; font-weight:bold; }}
  .isolation-note {{ margin:4px 0 12px; font-size:12px; color:#888; }}
  .isolation-note.has-held {{ color:#ffb300; }}
  .joint-hold-list {{ margin:4px 0 12px; font-size:12px; }}
  .joint-hold-list li {{ color:#f44336; }}
</style>
</head>
<body>
<h1>SO-101 MuJoCo 실시간 뷰어 (읽기 전용 - 팔로워암 제어 안 함)</h1>
<div class="command-source-banner">
  <span>Command source: <b id="command_source">{command_source}</b></span>
  <span>Global Hold: <b id="top_hold" class="hold-false">false</b></span>
  <span id="top_hold_reason" class="muted"></span>
  <span>Intervention count: <b id="intervention_count">0</b></span>
</div>
<p class="isolation-note" id="isolation_note"></p>
<ul class="joint-hold-list" id="joint_hold_list"></ul>
<div class="layout">
  <div class="video">
    <img src="/stream.mjpg" alt="MuJoCo live stream">
    <p class="muted">서버: {server_url} | 관절: {joints}</p>
  </div>
  <div>
    <table>
      <tr><th>서버 상태</th><td id="server_status">-</td></tr>
      <tr><th>sequence</th><td id="sequence">-</td></tr>
      <tr><th>네트워크 지연</th><td id="latency">-</td></tr>
      <tr><th>stale</th><td id="stale">-</td></tr>
      <tr><th>렌더 FPS</th><td id="fps">-</td></tr>
    </table>
    <table id="joint_table">
      <tr>
        <th>관절</th><th>Leader raw</th><th>Follower current</th><th>Requested/Mapped</th>
        <th>Applied/Limited</th><th>MuJoCo actual</th><th>margin</th><th>safety</th>
      </tr>
    </table>
    <p class="blocked-note">BLOCKED(빨간 배경) 관절은 Requested와 Applied target이 달라집니다 - 직전 안전 target을 그대로 유지 중이라는 뜻입니다.
    follower-safe 모드에서는 각각 raw leader / follower 현재 위치 / B단계(좌표 변환) target / F단계(rate-limit+hold 반영) 최종 명령을 뜻합니다.</p>
  </div>
  <div>
    <div class="safety-summary">
      <span>현재 상태: <b id="current_safety_level" class="PASS">-</b></span>
      <span id="event_counts" class="muted">최근 이벤트: -</span>
    </div>
    <div id="events_list" class="events"></div>
  </div>
</div>
<script>
function fmtTime(epochSec) {{
  if (epochSec == null) return '-';
  const d = new Date(epochSec * 1000);
  return d.toTimeString().slice(0, 8) + '.' + String(d.getMilliseconds()).padStart(3, '0');
}}
function eventSummaryLine(e) {{
  const deg = (v) => (v == null ? '?' : v.toFixed(1));
  switch (e.reason_code) {{
    case 'JOINT_RANGE_HIGH': {{
      const over = (e.requested_target_deg != null && e.joint_max_deg != null) ? (e.requested_target_deg - e.joint_max_deg).toFixed(1) : '?';
      return `target ${{deg(e.requested_target_deg)}}° > max ${{deg(e.joint_max_deg)}}° (초과 ${{over}}°)`;
    }}
    case 'JOINT_RANGE_LOW': {{
      const over = (e.requested_target_deg != null && e.joint_min_deg != null) ? (e.joint_min_deg - e.requested_target_deg).toFixed(1) : '?';
      return `target ${{deg(e.requested_target_deg)}}° < min ${{deg(e.joint_min_deg)}}° (초과 ${{over}}°)`;
    }}
    case 'NEAR_JOINT_LIMIT':
      return `limit margin ${{deg(e.margin_deg)}}°`;
    case 'FRAME_DELTA_HIGH':
      return `프레임간 변화량 ${{deg(e.delta_deg)}}° (임계값 초과)`;
    case 'REMOTE_STALE':
      return `remote age ${{e.remote_age_ms != null ? e.remote_age_ms.toFixed(0) : '?'}} ms (stale)`;
    case 'SEQUENCE_STALLED':
      return 'remote sequence가 갱신되지 않음';
    case 'INVALID_VALUE':
      return '리더/팔로워 값이 유효하지 않음 (NaN/누락 등)';
    case 'MISSING_JOINT':
      return '리더 관절 데이터에 이 관절이 없음';
    case 'MODE_NOT_READ_ONLY':
      return '서버 mode가 read_only가 아님';
    default:
      return '원인 미확정 (UNKNOWN_SAFETY_REASON)';
  }}
}}
async function poll() {{
  try {{
    const res = await fetch('/status', {{cache: 'no-store'}});
    const data = await res.json();
    document.getElementById('server_status').textContent = data.server_status;
    document.getElementById('sequence').textContent = data.sequence ?? '-';
    document.getElementById('latency').textContent = data.network_latency_ms != null ? data.network_latency_ms.toFixed(1) + ' ms' : '-';
    document.getElementById('stale').textContent = data.stale ? 'STALE' : 'fresh';
    document.getElementById('fps').textContent = data.fps.toFixed(1);

    document.getElementById('command_source').textContent = data.command_source;
    const topHoldEl = document.getElementById('top_hold');
    topHoldEl.textContent = data.hold ? 'true' : 'false';
    topHoldEl.className = data.hold ? 'hold-true' : 'hold-false';
    document.getElementById('top_hold_reason').textContent = data.global_hold_reason ? `(${{data.global_hold_reason}})` : '';
    document.getElementById('intervention_count').textContent = data.intervention_count ?? 0;

    const heldJoints = data.held_joints || {{}};
    const heldCount = data.held_joint_count ?? Object.keys(heldJoints).length;
    const activeCount = data.active_joint_count ?? (6 - heldCount);
    const noteEl = document.getElementById('isolation_note');
    if (data.command_source === 'follower-safe') {{
      if (data.global_hold) {{
        noteEl.className = 'isolation-note has-held';
        noteEl.textContent = `전체 제어 정지 (global hold: ${{data.global_hold_reason}}) - 6개 관절 전부 직전 안전값 유지 중`;
      }} else if (heldCount > 0) {{
        noteEl.className = 'isolation-note has-held';
        noteEl.textContent = `전체 제어 정지 아님 - ${{heldCount}}개 관절 격리됨, ${{activeCount}}개 관절 활성`;
      }} else {{
        noteEl.className = 'isolation-note';
        noteEl.textContent = `전체 ${{activeCount}}개 관절 정상 추종 중`;
      }}
    }} else {{
      noteEl.textContent = '';
    }}
    const holdList = document.getElementById('joint_hold_list');
    holdList.innerHTML = '';
    for (const [joint, reason] of Object.entries(heldJoints)) {{
      const li = document.createElement('li');
      li.textContent = `${{joint}}: ${{reason}}`;
      holdList.appendChild(li);
    }}

    const table = document.getElementById('joint_table');
    table.querySelectorAll('tr.joint-row').forEach(el => el.remove());
    for (const [name, j] of Object.entries(data.joints)) {{
      const tr = document.createElement('tr');
      tr.className = 'joint-row' + (j.blocked ? ' blocked-row' : '');
      const fmt = (v) => (v == null ? '-' : v.toFixed(2));
      const safetyExtra = j.blocked ? ` (${{j.follower_hold_reason || '이전값 유지'}})` : (j.rate_limited ? ' (rate-limited)' : '');
      tr.innerHTML = `<td>${{name}}</td><td>${{fmt(j.leader_deg)}}</td><td>${{fmt(j.follower_current_deg ?? j.follower_deg)}}</td>` +
        `<td>${{fmt(j.requested_target_deg)}}</td><td>${{fmt(j.mujoco_target_deg)}}</td>` +
        `<td>${{fmt(j.mujoco_qpos_deg)}}</td><td>${{fmt(j.limit_margin_deg)}}</td>` +
        `<td class="${{j.safety_status}}">${{j.safety_status}}${{safetyExtra}}</td>`;
      table.appendChild(tr);
    }}

    const level = data.current_safety.level;
    const levelEl = document.getElementById('current_safety_level');
    levelEl.textContent = level;
    levelEl.className = level;
    const counts = data.safety_event_counts;
    document.getElementById('event_counts').textContent = `최근 이벤트: BLOCKED ${{counts.BLOCKED}}건, WARN ${{counts.WARN}}건`;

    const list = document.getElementById('events_list');
    list.innerHTML = '';
    for (const e of data.recent_safety_events) {{
      const card = document.createElement('div');
      card.className = 'event-card ' + e.severity;
      const sampleWord = e.sample_count > 1 ? 'samples' : 'sample';
      card.innerHTML = `<div class="event-head ${{e.severity}}">[${{e.severity}}] ${{e.joint}}</div>` +
        `<div class="event-body">${{eventSummaryLine(e)}}</div>` +
        `<div class="event-meta">${{e.sample_count}} ${{sampleWord}} / ${{e.duration_ms.toFixed(0)}} ms${{e.active ? ' (진행 중)' : ''}}</div>` +
        `<div class="event-time">${{fmtTime(e.ended_at)}}</div>`;
      list.appendChild(card);
    }}
  }} catch (e) {{
    document.getElementById('server_status').textContent = '상태 조회 실패: ' + e;
  }}
  setTimeout(poll, 500);
}}
poll();
</script>
</body>
</html>
"""


def _make_handler_class(viewer: LiveWebViewer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SO101MuJoCoWebViewer/1.0"

        def log_message(self, fmt, *args):  # 요청마다 stderr에 찍지 않는다 (한글 상태 출력과 분리)
            pass

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._serve_index()
            elif self.path == "/stream.mjpg":
                self._serve_mjpeg()
            elif self.path == "/frame.jpg":
                self._serve_single_frame()
            elif self.path == "/status":
                self._serve_status()
            elif self.path == "/events":
                self._serve_events()
            elif self.path == "/health":
                self._serve_health()
            else:
                self.send_error(404, "Not Found")

        def _serve_index(self):
            html = _INDEX_HTML_TEMPLATE.format(
                width=viewer.args.frame_width,
                height=viewer.args.frame_height,
                server_url=viewer.args.server_url,
                joints=", ".join(viewer.args.joints),
                command_source=viewer.args.command_source,
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_status(self):
            payload = viewer.status.to_dict()
            payload.update(viewer.safety_status_payload())
            payload.update(viewer.realism_status_payload())
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_events(self):
            # 읽기 전용 (GET) - safety 이벤트 이력만 노출한다. 인증정보/원격 응답 원문은
            # 애초에 SafetyEventRecord에 담기지 않는다.
            body = json.dumps(viewer.safety_status_payload(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_health(self):
            body = json.dumps({"status": "ok", "stopped": viewer.stopped}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_single_frame(self):
            jpeg, _ = viewer.frames.snapshot()
            if jpeg is None:
                self.send_error(503, "아직 렌더링된 프레임이 없습니다")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)

        def _serve_mjpeg(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}")
            self.end_headers()

            last_generation = -1
            min_interval = 1.0 / max(viewer.args.fps, 1.0)
            try:
                # 브라우저가 탭을 닫아 연결이 끊겨도(BrokenPipe/ConnectionReset) 여기서만
                # 조용히 끝내고, 렌더/네트워크 스레드는 계속 동작한다 (요구사항 4번 마지막 항목).
                while not viewer.stopped:
                    jpeg, generation = viewer.frames.snapshot()
                    if jpeg is None or generation == last_generation:
                        time.sleep(min_interval)
                        continue
                    last_generation = generation
                    self.wfile.write(f"--{MJPEG_BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(min_interval)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return Handler


def create_http_server(viewer: LiveWebViewer) -> ThreadingHTTPServer:
    handler_cls = _make_handler_class(viewer)
    server = ThreadingHTTPServer((viewer.args.host, viewer.args.port), handler_cls)
    server.daemon_threads = True
    return server


def detect_local_ip() -> str | None:
    """Windows 브라우저에서 접속할 WSL IP 추정 (외부로 패킷을 실제로 보내지 않는 UDP 트릭)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()
