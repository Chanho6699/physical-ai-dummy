#!/usr/bin/env python3
"""Instrumented Teleop Diagnostic - 정상 LeRobot SO101Leader/SO101Follower teleoperation을
그대로 실행하면서 wrist_roll 중심으로 leader command -> follower Goal/Present 상태를
**passive하게(개입 없이)** 계측한다.

**이 스크립트는 follower를 실제로 움직인다 - 그리고 그 움직임을 절대 방해하지 않는다.**
follower로 나가는 유일한 write 경로는 ``lerobot.robots.so_follower.SO101Follower``의 정상
``connect()``(내부적으로 ``configure()``)/``send_action()``/``disconnect()``뿐이다 - 이
스크립트도, ``hardware/diagnostics/instrumented_teleop.py``도 ``FeetechMotorsBus.write()``/
``sync_write()``/``enable_torque()``/``disable_torque()``를 직접 호출하지 않는다. 자세한
조사 근거는 ``hardware/diagnostics/instrumented_teleop.py`` 모듈 docstring 참고.

## Warning 문턱값 (전부 "감지 후 기록"만 한다 - command를 막거나 바꾸지 않는다)

- ``--command-delta-max-deg``(기본 2.0): follower 시작 위치 기준 이 값을 넘는 wrist_roll
  command가 감지되면 ``WARNING_LARGE_COMMAND_DELTA``를 기록한다. **``send_action()``은
  그대로 호출된다.**
- ``--position-jump-max-deg``(기본 3.0): 한 cycle 안에서 Present_Position이 이 값보다 크게
  튀면 ``WARNING_POSITION_JUMP``를 기록한다.
- ``--direction-mismatch-tolerance-deg``(기본 0.176° ≈ 2 tick): command와 실측 이동 방향이
  이 tolerance를 넘어서고도 불일치하면 ``WARNING_DIRECTION_MISMATCH``를 기록한다.
- ``--tracking-error-max-deg``(기본 1.0): Goal-Present 오차가 이 값을 넘으면
  ``WARNING_LARGE_TRACKING_ERROR``를 기록한다.
- ``--low-loop-rate-hz``(기본 20.0): 한 cycle의 순간 loop rate가 이 값보다 낮으면
  ``WARNING_LOW_LOOP_RATE``를 기록한다.
- ``Status``가 0이 아니면 ``WARNING_STATUS_NONZERO``를 기록한다.

이 프로그램이 실제로 멈추는 경우는 셋뿐이다: 목표 duration 경과(``DURATION_ELAPSED``),
사용자 Ctrl+C(``KEYBOARD_INTERRUPT``), 그리고 정상 teleop 자체가 더 이상 진행할 수 없는 진짜
실패(``get_observation``/``get_action``/프로세서/``send_action`` 예외 - ``READ_FAILURE``).
계측(레지스터 read) 자체의 실패는 루프를 멈추지 않는다.

## Timing 버그 수정 (60Hz 요청 -> 실측 89.67Hz였던 원인)

``run_instrumented_teleop_loop``가 ``sleep_fn``을 명시적으로 넘기지 않으면 기본값이 아무것도
하지 않는 함수였다 - 그래서 fps 제한이 전혀 걸리지 않고 실제 작업 시간만큼의 속도로
루프가 돌았다. 이제 기본값이 ``None``이면 ``lerobot.utils.robot_utils.precise_sleep``
(``lerobot_teleoperate.teleop_loop()``가 실제로 쓰는 것과 동일한 함수)을 지연 import해서
쓴다 - 이 스크립트는 아무것도 바꾸지 않아도 자동으로 고쳐졌다(호출부에서 ``sleep_fn``을
넘기지 않았기 때문). 자세한 근거는 ``hardware/diagnostics/instrumented_teleop.py``의
``run_instrumented_teleop_loop`` docstring 참고.

## causal deadband/motion-onset 분석 (``--deadband-lookahead-ms``/``--motion-response-noise-threshold-ticks``)

이전 버전의 deadband 분석은 "세션 시작 위치에서 present가 얼마나 멀어졌는가"만 봐서, 한 번
움직인 뒤에는 현재 에러가 0이어도 계속 "움직임 있음"으로 잘못 집계됐다. 지금은 각 샘플의
에러 관측 시점 "직후"(``--deadband-lookahead-ms``, 기본 100ms - 60Hz 재측정 전 잠정값) 안의
실제 present 변화만 인과적으로 본다. 자세한 근거는 ``hardware/diagnostics/
instrumented_teleop.py``의 ``compute_deadband_summary``/``compute_motion_onset_analysis``
docstring 참고.

## 실행 예시

설정만 검증(하드웨어 연결 없음, follower/leader에 어떤 write도 발생하지 않음):
    python scripts/run_instrumented_teleop.py --dry-run

실제 실행(follower가 실제로 움직인다 - 사용자가 직접 실행할 것):
    python scripts/run_instrumented_teleop.py --duration-sec 15 --fps 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.diagnostics.instrumented_teleop import (
    DEFAULT_DEADBAND_LOOKAHEAD_MS,
    DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS,
    WarningThresholds,
    WristRollRegisterInstrument,
    compute_run_analysis,
    run_instrumented_teleop_loop,
)
from hardware.diagnostics.instrumented_teleop_console import TerminalDashboard, render_dashboard_lines
from hardware.diagnostics.instrumented_teleop_logger import (
    CsvSampleWriter,
    build_csv_path,
    build_json_report_path,
    write_json_report,
)
from hardware.safety import calibration_resolution as calres
from hardware.safety import port_conflict as pc
from hardware.state_server.calibration_loader import CalibrationLoadError, load_calibration_file

LEADER_CALIBRATION_PATH_TEMPLATE = "~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/{id}.json"
DEFAULT_LEADER_ID_FALLBACK = "chanho_leader"

TARGET_JOINT = "wrist_roll"
DEFAULT_DURATION_SEC = 15.0
DEFAULT_FPS = 60
DEFAULT_DASHBOARD_INTERVAL_S = 0.15  # 약 6.7Hz - 요구사항 5~10Hz 범위 안
DEFAULT_NUM_READ_RETRIES = 2
REPORTS_DIR_RELATIVE = Path("reports") / "instrumented_teleop"


class RefusalError(RuntimeError):
    """설정/안전 정책 위반으로 실행을 거부해야 할 때 (하드웨어 오류와 구분)."""


# ---------------------------------------------------------------------------
# 리더 포트/calibration 경로 해석 (scripts/run_shadow_teleop_diagnostic.py와 동일한 패턴)
# ---------------------------------------------------------------------------


def _load_local_config(project_root: Path) -> dict | None:
    path = project_root / "configs" / "hardware.local.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_leader_port(*, cli_port: str | None, project_root: Path) -> tuple[str, str]:
    if cli_port:
        return cli_port, "cli"
    local = _load_local_config(project_root)
    if local:
        port = (local.get("teleop") or {}).get("port")
        if port:
            return port, "local_config"
    raise RefusalError(
        "리더 포트를 확정할 수 없습니다. --leader-port를 지정하거나 "
        "configs/hardware.local.json의 teleop.port를 설정하세요."
    )


def resolve_leader_id(*, cli_id: str | None, project_root: Path) -> tuple[str, str]:
    if cli_id:
        return cli_id, "cli"
    local = _load_local_config(project_root)
    if local:
        leader_id = (local.get("teleop") or {}).get("id")
        if leader_id:
            return leader_id, "local_config"
    return DEFAULT_LEADER_ID_FALLBACK, "default_fallback"


def resolve_leader_calibration_path(*, leader_id: str) -> Path:
    return Path(LEADER_CALIBRATION_PATH_TEMPLATE.format(id=leader_id)).expanduser()


@dataclass
class ResolvedConfig:
    leader_port: str
    leader_port_source: str
    leader_id: str
    leader_id_source: str
    leader_calibration_path: Path
    follower_port: str
    follower_port_source: str
    follower_id: str
    follower_calibration_path: Path
    follower_calibration_source: str


def resolve_config(args: argparse.Namespace) -> ResolvedConfig:
    leader_port, leader_port_source = resolve_leader_port(cli_port=args.leader_port, project_root=PROJECT_ROOT)
    leader_id, leader_id_source = resolve_leader_id(cli_id=args.leader_id, project_root=PROJECT_ROOT)
    leader_calibration_path = resolve_leader_calibration_path(leader_id=leader_id)

    follower_port, follower_port_source = calres.resolve_port(cli_port=args.follower_port, project_root=PROJECT_ROOT)
    follower_calibration_path, follower_calibration_source = calres.resolve_calibration_path(
        cli_calibration_path=None,
        cli_calibration_id=args.follower_id,
        project_root=PROJECT_ROOT,
        allow_default_fallback=True,
    )
    # calres.resolve_calibration_path는 id에서 follower_id를 못 뽑으므로 local_config를 다시 확인한다.
    follower_id = args.follower_id
    if follower_id is None:
        local = _load_local_config(PROJECT_ROOT)
        follower_id = ((local or {}).get("robot") or {}).get("id") or "chanho_follower"

    return ResolvedConfig(
        leader_port=leader_port,
        leader_port_source=leader_port_source,
        leader_id=leader_id,
        leader_id_source=leader_id_source,
        leader_calibration_path=leader_calibration_path,
        follower_port=follower_port,
        follower_port_source=follower_port_source,
        follower_id=follower_id,
        follower_calibration_path=follower_calibration_path,
        follower_calibration_source=follower_calibration_source,
    )


# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SO-101 Instrumented Teleop Diagnostic - 정상 LeRobot teleoperation을 그대로 실행하며 "
            "wrist_roll 중심으로 계측한다. follower가 실제로 움직인다."
        ),
    )
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC, help=f"최대 실행 시간(초) (기본 {DEFAULT_DURATION_SEC:g})")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"목표 loop 주파수 (기본 {DEFAULT_FPS})")
    parser.add_argument("--leader-port", default=None, help="지정하지 않으면 configs/hardware.local.json의 teleop.port를 사용한다.")
    parser.add_argument("--follower-port", default=None, help="지정하지 않으면 configs/hardware.local.json의 robot.port를 사용한다.")
    parser.add_argument("--leader-id", default=None, help="리더 calibration id (기본: configs/hardware.local.json의 teleop.id, 없으면 chanho_leader).")
    parser.add_argument("--follower-id", default=None, help="팔로워 calibration id (기본: configs/hardware.local.json의 robot.id, 없으면 chanho_follower).")
    parser.add_argument(
        "--command-delta-max-deg",
        type=float,
        default=WarningThresholds().command_delta_max_deg,
        help=(
            "warning 문턱값(감지만, command는 그대로 전달됨): follower 시작 위치 기준 이 값을 넘는 "
            f"wrist_roll command가 감지되면 WARNING_LARGE_COMMAND_DELTA 기록 (기본 {WarningThresholds().command_delta_max_deg:g})."
        ),
    )
    parser.add_argument(
        "--position-jump-max-deg",
        type=float,
        default=WarningThresholds().position_jump_max_deg,
        help=f"warning 문턱값: 한 cycle 최대 Present_Position 변화(도) (기본 {WarningThresholds().position_jump_max_deg:g}).",
    )
    parser.add_argument(
        "--direction-mismatch-tolerance-deg",
        type=float,
        default=WarningThresholds().direction_mismatch_noise_tolerance_deg,
        help=f"warning 문턱값: 방향 불일치 판정 노이즈 tolerance(도) (기본 {WarningThresholds().direction_mismatch_noise_tolerance_deg:g}).",
    )
    parser.add_argument(
        "--tracking-error-max-deg",
        type=float,
        default=WarningThresholds().tracking_error_max_deg,
        help=f"warning 문턱값: Goal-Present 오차(도) (기본 {WarningThresholds().tracking_error_max_deg:g}).",
    )
    parser.add_argument(
        "--low-loop-rate-hz",
        type=float,
        default=WarningThresholds().low_loop_rate_hz,
        help=f"warning 문턱값: 이 값보다 loop rate가 낮은 cycle을 기록 (기본 {WarningThresholds().low_loop_rate_hz:g}).",
    )
    parser.add_argument(
        "--deadband-lookahead-ms",
        type=float,
        default=DEFAULT_DEADBAND_LOOKAHEAD_MS,
        help=(
            "causal deadband/motion-onset 분석 전용: error 관측 시점 이후 이 시간(ms) 안의 present "
            f"변화까지를 '반응'으로 인정한다 (기본 {DEFAULT_DEADBAND_LOOKAHEAD_MS:g}ms - 60Hz 재측정 전 "
            "잠정값, 확정 threshold 아님)."
        ),
    )
    parser.add_argument(
        "--motion-response-noise-threshold-ticks",
        type=int,
        default=DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS,
        help=(
            "causal deadband/motion-onset 분석 전용: 이 값(tick) 미만의 present 변화는 read 노이즈로 "
            f"보고 '반응 없음'으로 판정한다 (기본 {DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS} tick)."
        ),
    )
    parser.add_argument("--accel-refresh-interval-s", type=float, default=1.0, help="Acceleration/Acceleration_Multiplier 재조회 최소 간격(초) (기본 1.0).")
    parser.add_argument("--num-read-retries", type=int, default=DEFAULT_NUM_READ_RETRIES, help=f"계측 레지스터 read 재시도 횟수 (기본 {DEFAULT_NUM_READ_RETRIES}).")
    parser.add_argument("--dashboard-interval-s", type=float, default=DEFAULT_DASHBOARD_INTERVAL_S, help=f"터미널 대시보드 갱신 최소 간격(초) (기본 {DEFAULT_DASHBOARD_INTERVAL_S:g}).")
    parser.add_argument("--no-dashboard", action="store_true", help="터미널 대시보드 출력을 끈다 (CSV/분석은 그대로 수행).")
    parser.add_argument("--csv-dir", default=None, help=f"CSV/JSON 저장 디렉터리 (기본 {REPORTS_DIR_RELATIVE}/)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="포트/calibration 설정만 검증하고 종료한다 - connect()를 전혀 호출하지 않으므로 follower/leader에 어떤 write도 발생하지 않는다.",
    )
    return parser


# ---------------------------------------------------------------------------
# dry-run: 하드웨어에 전혀 연결하지 않는다
# ---------------------------------------------------------------------------


def run_dry_run(args: argparse.Namespace, *, stdout=None) -> int:
    # 기본값을 함수 정의 시점에 sys.stdout으로 고정하지 않는다 - pytest capsys처럼 호출
    #시점에 sys.stdout이 바뀌어 있는 상황(테스트)에서도 실제로 바뀐 스트림에 출력해야 하기
    # 때문에, 호출 시점에 다시 조회한다.
    stdout = stdout if stdout is not None else sys.stdout

    def _p(msg: str) -> None:
        print(msg, file=stdout)

    _p("=== Instrumented Teleop Diagnostic: DRY RUN (연결 없음, write 0회) ===")
    try:
        config = resolve_config(args)
    except (calres.ResolutionError, RefusalError) as exc:
        _p(f"거부: {exc}")
        return 2

    _p(f"leader_port={config.leader_port} (source={config.leader_port_source})")
    _p(f"leader_id={config.leader_id} (source={config.leader_id_source})")
    _p(f"leader_calibration_path={config.leader_calibration_path}")
    _p(f"follower_port={config.follower_port} (source={config.follower_port_source})")
    _p(f"follower_id={config.follower_id}")
    _p(f"follower_calibration_path={config.follower_calibration_path} (source={config.follower_calibration_source})")

    _p("\n--- 포트 점유 검사 ---")
    leader_conflict = pc.check_port_conflict(config.leader_port)
    follower_conflict = pc.check_port_conflict(config.follower_port)
    _p(f"leader: busy={leader_conflict.busy} busy_confirmed={leader_conflict.busy_confirmed}")
    _p(f"follower: busy={follower_conflict.busy} busy_confirmed={follower_conflict.busy_confirmed}")
    ports_ok = not (leader_conflict.busy or follower_conflict.busy)

    _p("\n--- calibration 파일 검증 ---")
    calibration_ok = True
    for label, path in (("leader", config.leader_calibration_path), ("follower", config.follower_calibration_path)):
        try:
            entries = load_calibration_file(path)
            entry = entries[TARGET_JOINT]
            _p(f"{label}: OK (wrist_roll range=[{entry.range_min}, {entry.range_max}], homing_offset={entry.homing_offset})")
        except CalibrationLoadError as exc:
            _p(f"{label}: 실패 - {exc}")
            calibration_ok = False

    _p("\n--- warning 문턱값 (감지만 - command에는 영향 없음) ---")
    _p(f"command_delta_max_deg=±{args.command_delta_max_deg:g}")
    _p(f"position_jump_max_deg=±{args.position_jump_max_deg:g}")
    _p(f"direction_mismatch_tolerance_deg=±{args.direction_mismatch_tolerance_deg:g}")
    _p(f"tracking_error_max_deg=±{args.tracking_error_max_deg:g}")
    _p(f"low_loop_rate_hz={args.low_loop_rate_hz:g}")

    _p("\n--- 실행 계획 ---")
    _p(f"fps={args.fps} duration_sec={args.duration_sec:g}")
    _p("connect() 호출 없음 - 이 dry-run은 leader/follower에 어떤 write도 수행하지 않았습니다. write_count=0")

    _p("\n실제 실행 명령 (follower가 실제로 움직입니다 - 사용자가 직접 실행):")
    _p(
        f"  {sys.executable} {Path(__file__).resolve()} "
        f"--duration-sec {args.duration_sec:g} --fps {args.fps}"
    )

    return 0 if (ports_ok and calibration_ok) else 1


# ---------------------------------------------------------------------------
# 실제 실행
# ---------------------------------------------------------------------------


def run(
    args: argparse.Namespace,
    *,
    leader_factory=None,
    follower_factory=None,
    processors_factory=None,
    stdout=None,
) -> int:
    """CLI 본체. ``leader_factory``/``follower_factory``/``processors_factory``는 테스트가
    실물 ``SO101Leader``/``SO101Follower``/``make_default_processors``를 가짜로 바꿔치기할 수
    있게 하는 주입 지점이다 - 기본값은 실제 lerobot 객체를 생성한다(지연 import).

    ``stdout``의 기본값을 함수 정의 시점에 ``sys.stdout``으로 고정하지 않는다 - pytest
    ``capsys``처럼 호출 시점에 ``sys.stdout``이 교체되어 있는 상황에서도 실제로 교체된
    스트림에 출력되도록, 호출 시점에 다시 조회한다.
    """
    stdout = stdout if stdout is not None else sys.stdout

    if args.dry_run:
        return run_dry_run(args, stdout=stdout)

    def _p(msg: str) -> None:
        print(msg, file=stdout)

    try:
        config = resolve_config(args)
    except (calres.ResolutionError, RefusalError) as exc:
        _p(f"거부: {exc}")
        return 2

    _p("Instrumented Teleop Diagnostic - 정상 LeRobot teleoperation 경로 사용 (follower가 실제로 움직입니다)")
    _p(f"leader_port={config.leader_port} (source={config.leader_port_source}) id={config.leader_id}")
    _p(f"follower_port={config.follower_port} (source={config.follower_port_source}) id={config.follower_id}")

    _p("\n=== 포트 점유 검사 ===")
    leader_conflict = pc.check_port_conflict(config.leader_port)
    follower_conflict = pc.check_port_conflict(config.follower_port)
    _p(f"leader: busy={leader_conflict.busy} busy_confirmed={leader_conflict.busy_confirmed}")
    _p(f"follower: busy={follower_conflict.busy} busy_confirmed={follower_conflict.busy_confirmed}")
    if leader_conflict.busy or follower_conflict.busy:
        _p("거부: 리더 또는 팔로워 포트가 점유 중(또는 판정 불가)이라 연결을 시도하지 않습니다.")
        return 2

    _p("\n=== calibration 파일 사전 점검 (interactive calibrate() 방지) ===")
    # SOFollower/SOLeader.connect()는 calibration 파일이 없거나 모터 실제값과 다르면 인터랙티브
    # calibrate()(input() 대기)를 실행한다 - 비대화형 실행에서 멈추는 것을 막기 위해 파일
    # 존재/구조를 여기서 먼저 확인한다. 실제 값 로딩/적용은 SOFollower/SOLeader 자신이 한다(이
    # 결과는 진입 가능 여부 판단에만 쓴다).
    for label, path in (("leader", config.leader_calibration_path), ("follower", config.follower_calibration_path)):
        try:
            load_calibration_file(path)
        except CalibrationLoadError as exc:
            _p(f"거부: {label} calibration 파일 점검 실패: {exc}")
            return 2
    _p("calibration 파일 존재/구조 확인 완료.")

    if leader_factory is None or follower_factory is None or processors_factory is None:
        from lerobot.processor import make_default_processors
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        if leader_factory is None:
            leader_factory = lambda: SO101Leader(SO101LeaderConfig(port=config.leader_port, id=config.leader_id))  # noqa: E731
        if follower_factory is None:
            follower_factory = lambda: SO101Follower(SO101FollowerConfig(port=config.follower_port, id=config.follower_id))  # noqa: E731
        if processors_factory is None:
            processors_factory = make_default_processors

    leader = leader_factory()
    follower = follower_factory()
    teleop_action_processor, robot_action_processor, _robot_observation_processor = processors_factory()

    csv_dir = Path(args.csv_dir).expanduser() if args.csv_dir else (PROJECT_ROOT / REPORTS_DIR_RELATIVE)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = build_csv_path(csv_dir)
    json_path = build_json_report_path(csv_path)

    _p("\n=== 연결 (정상 LeRobot connect()/configure() - torque enable/Acceleration write 포함) ===")
    leader_connected = False
    follower_connected = False
    result = None
    csv_writer = CsvSampleWriter(csv_path)
    dashboard = None if args.no_dashboard else TerminalDashboard(stream=stdout)
    last_dashboard_at = 0.0

    try:
        try:
            leader.connect()
            leader_connected = True
            _p(f"리더 연결 성공 (port={config.leader_port})")
        except (ConnectionError, RuntimeError) as exc:
            _p(f"거부: 리더 연결 실패: {exc}")
            return 1

        try:
            follower.connect()
            follower_connected = True
            _p(f"팔로워 연결 성공 (port={config.follower_port}) - configure() 완료(Torque_Enable/Acceleration write 포함)")
        except (ConnectionError, RuntimeError) as exc:
            _p(f"거부: 팔로워 연결 실패: {exc}")
            return 1

        instrument = WristRollRegisterInstrument(bus=follower.bus, num_read_retries=args.num_read_retries)
        wrist_roll_calibration = follower.calibration[TARGET_JOINT]
        follower_calibration_range = (wrist_roll_calibration.range_min, wrist_roll_calibration.range_max)

        warning_thresholds = WarningThresholds(
            command_delta_max_deg=args.command_delta_max_deg,
            direction_mismatch_noise_tolerance_deg=args.direction_mismatch_tolerance_deg,
            position_jump_max_deg=args.position_jump_max_deg,
            tracking_error_max_deg=args.tracking_error_max_deg,
            low_loop_rate_hz=args.low_loop_rate_hz,
        )

        warning_count = 0

        def _on_warning(_event) -> None:
            nonlocal warning_count
            warning_count += 1

        def _on_sample(sample) -> None:
            csv_writer.write_sample(sample)
            nonlocal last_dashboard_at
            now = time.monotonic()
            if dashboard is not None and (now - last_dashboard_at) >= args.dashboard_interval_s:
                dashboard.update(
                    render_dashboard_lines(sample, loop_hz=sample.loop_hz, total_warning_count=warning_count)
                )
                last_dashboard_at = now

        _p("\n=== 계측 시작 (passive monitoring - command에 개입하지 않음. Ctrl+C로 언제든 중단 가능) ===")
        result = run_instrumented_teleop_loop(
            leader=leader,
            follower=follower,
            instrument=instrument,
            follower_calibration_range=follower_calibration_range,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            fps=args.fps,
            duration_sec=args.duration_sec,
            warning_thresholds=warning_thresholds,
            accel_refresh_interval_s=args.accel_refresh_interval_s,
            on_sample=_on_sample,
            on_warning=_on_warning,
        )
    finally:
        try:
            if follower_connected:
                follower.disconnect()
        except Exception as exc:  # noqa: BLE001
            _p(f"경고: 팔로워 disconnect 중 오류: {exc}")
        try:
            if leader_connected:
                leader.disconnect()
        except Exception as exc:  # noqa: BLE001
            _p(f"경고: 리더 disconnect 중 오류: {exc}")
        csv_writer.close()

    if result is None:
        return 1

    _p("\n=== 종료 ===")
    _p(f"stopped_reason={result.stopped_reason}")
    if result.error is not None:
        _p(f"error={result.error}")
    _p(f"samples_collected={len(result.samples)}")
    _p(f"total_warning_count={len(result.warnings)}")
    _p(f"CSV 저장 위치: {csv_path}")

    analysis = compute_run_analysis(
        result,
        deadband_lookahead_ms=args.deadband_lookahead_ms,
        motion_response_noise_threshold_ticks=args.motion_response_noise_threshold_ticks,
    )
    _p("\n=== 분석 ===")
    _p(json.dumps(analysis, indent=2, ensure_ascii=False, default=str))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leader_port_source": config.leader_port_source,
        "follower_port_source": config.follower_port_source,
        "leader_id": config.leader_id,
        "follower_id": config.follower_id,
        "csv_path": str(csv_path),
        "stopped_reason": result.stopped_reason,
        "initial_snapshot": result.initial_snapshot.to_dict() if result.initial_snapshot else None,
        "follower_start_present_raw": result.follower_start_present_raw,
        "follower_start_present_deg": result.follower_start_present_deg,
        "warning_thresholds": {
            "command_delta_max_deg": warning_thresholds.command_delta_max_deg,
            "position_jump_max_deg": warning_thresholds.position_jump_max_deg,
            "direction_mismatch_noise_tolerance_deg": warning_thresholds.direction_mismatch_noise_tolerance_deg,
            "tracking_error_max_deg": warning_thresholds.tracking_error_max_deg,
            "low_loop_rate_hz": warning_thresholds.low_loop_rate_hz,
        },
        "warnings": [w.to_dict() for w in result.warnings],
        "analysis": analysis,
        "direct_register_write_count": 0,
    }
    write_json_report(json_path, report)
    _p(f"\nJSON 리포트 저장: {json_path}")
    _p(
        "\ndirect_register_write_count=0 (이 스크립트는 SOFollower.send_action()/정상 "
        "connect()-configure() 외에 어떤 servo write도 직접 수행하지 않았습니다. warning은 전부 "
        "기록만 했고 command를 막거나 바꾸지 않았습니다.)"
    )

    return 0 if result.stopped_reason in ("DURATION_ELAPSED", "KEYBOARD_INTERRUPT") else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
