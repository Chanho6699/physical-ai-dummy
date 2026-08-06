#!/usr/bin/env python3
"""SO-101 리더암/팔로워암 읽기 전용 관절 상태 서버 실행 CLI.

노트북에 연결된 SO-101 리더암·팔로워암의 현재 관절값을 읽어 HTTP API(GET /health,
GET /state, GET /calibration)로 제공한다. 이 서버는 실물에 어떤 명령도 쓰지 않는다
(send_action / sync_write / torque enable 등 없음). 자세한 설명은
docs/hardware_state_server.md, hardware/state_server/readonly_so101_reader.py 참고.

실행 예시:
    python scripts/run_hardware_state_server.py \\
        --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14029966-if00 \\
        --follower-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113538-if00 \\
        --leader-id chanho_leader \\
        --follower-id chanho_follower

하드웨어 연결 없이 설정만 확인:
    python scripts/run_hardware_state_server.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.state_server import console_status as cs
from hardware.state_server.app import create_app
from hardware.state_server.calibration_loader import (
    CalibrationLoadError,
    load_calibration_file,
    to_motor_calibration_map,
    to_public_dict,
)
from hardware.state_server.readonly_so101_reader import ReadOnlySO101Reader
from hardware.state_server.state_service import StatePoller

# LeRobot이 실제로 사용하는 표준 캘리브레이션 캐시 경로 (id로 파일명이 결정된다).
DEFAULT_LEADER_CALIBRATION_TEMPLATE = (
    "~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/{id}.json"
)
DEFAULT_FOLLOWER_CALIBRATION_TEMPLATE = (
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/{id}.json"
)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_RATE_HZ = 30.0
DEFAULT_STALE_AFTER_MS = 500.0
DEFAULT_MAX_READ_ERRORS = 3
STATUS_INTERVAL_S = 5.0
API_TOKEN_ENV_VAR = "SO101_STATE_SERVER_TOKEN"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "hardware_state_server.yaml"


class ConfigLoadError(RuntimeError):
    """설정 YAML을 읽거나 해석할 수 없을 때 발생한다."""

# (yaml 섹션, yaml 키) -> argparse 대상(dest) 이름. configs/hardware_state_server.yaml 참고.
_CONFIG_KEY_MAP: dict[tuple[str, str], str] = {
    ("server", "host"): "host",
    ("server", "port"): "port",
    ("devices", "leader_id"): "leader_id",
    ("devices", "follower_id"): "follower_id",
    ("devices", "leader_port"): "leader_port",
    ("devices", "follower_port"): "follower_port",
    ("devices", "leader_calibration_path"): "leader_calibration_path",
    ("devices", "follower_calibration_path"): "follower_calibration_path",
    ("polling", "rate_hz"): "rate_hz",
    ("polling", "stale_after_ms"): "stale_after_ms",
    ("polling", "max_read_errors"): "max_read_errors",
    ("security", "api_token"): "api_token",
    ("console", "quiet"): "quiet",
    ("console", "verbose"): "verbose",
    ("console", "no_color"): "no_color",
}


def _load_yaml_defaults(path: Path) -> dict[str, object]:
    """설정 YAML을 읽어 argparse 기본값 dict로 변환한다.

    파일이 없으면 조용히 빈 dict를 반환한다 (YAML은 선택 사항 - CLI 플래그만으로도
    완전히 동작해야 한다). 값이 null(None)인 항목은 무시해 argparse 하드코딩 기본값이
    유지되게 한다. 이 함수가 반환한 값은 ``set_defaults()``로만 적용되므로, 사용자가
    실제로 넘긴 CLI 플래그가 항상 우선한다.
    """

    if not path.is_file():
        return {}

    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigLoadError(f"설정 YAML 최상위는 object여야 합니다: {path}")

    defaults: dict[str, object] = {}
    for (section, key), arg_name in _CONFIG_KEY_MAP.items():
        section_value = raw.get(section)
        if not isinstance(section_value, dict):
            continue
        value = section_value.get(key)
        if value is not None:
            defaults[arg_name] = value
    return defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SO-101 리더암/팔로워암 읽기 전용 관절 상태 서버를 실행합니다.",
    )
    parser.add_argument("--leader-port", default=None, help="리더암 serial 포트 (--dry-run이 아니면 필수)")
    parser.add_argument("--follower-port", default=None, help="팔로워암 serial 포트 (--dry-run이 아니면 필수)")
    parser.add_argument("--leader-id", default="chanho_leader", help="리더암 id (캘리브레이션 파일명에 사용)")
    parser.add_argument("--follower-id", default="chanho_follower", help="팔로워암 id")
    parser.add_argument(
        "--leader-calibration-path",
        default=None,
        help="리더암 캘리브레이션 JSON 경로 (기본: LeRobot 표준 캐시 경로에서 --leader-id로 찾음)",
    )
    parser.add_argument(
        "--follower-calibration-path",
        default=None,
        help="팔로워암 캘리브레이션 JSON 경로 (기본: LeRobot 표준 캐시 경로에서 --follower-id로 찾음)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"바인드 host (기본: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"바인드 port (기본: {DEFAULT_PORT})")
    parser.add_argument(
        "--rate-hz", type=float, default=DEFAULT_RATE_HZ, help=f"관절 상태 읽기 주기 Hz (기본: {DEFAULT_RATE_HZ:g})"
    )
    parser.add_argument(
        "--stale-after-ms",
        type=float,
        default=DEFAULT_STALE_AFTER_MS,
        help=f"마지막 정상값을 stale로 표시하기 시작하는 경과 시간 ms (기본: {DEFAULT_STALE_AFTER_MS:g})",
    )
    parser.add_argument(
        "--max-read-errors",
        type=int,
        default=DEFAULT_MAX_READ_ERRORS,
        help=f"이 횟수만큼 연속 읽기 실패 시 health를 degraded로 전환 (기본: {DEFAULT_MAX_READ_ERRORS})",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help=f"설정하면 모든 GET 요청에 Authorization: Bearer <token> 필요 (환경변수 {API_TOKEN_ENV_VAR}로도 지정 가능)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "설정 YAML 경로 (기본: configs/hardware_state_server.yaml, 없으면 무시됨). "
            "여기 값은 기본값으로만 쓰이고, 실제로 넘긴 CLI 플래그가 항상 우선한다."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="하드웨어에 연결하지 않고 설정/캘리브레이션만 검증")
    parser.add_argument("--quiet", action="store_true", help="주기 상태 출력과 PASS 단계 출력을 억제 (오류는 항상 표시)")
    parser.add_argument("--verbose", action="store_true", help="설정 세부값과 추가 진단 정보를 출력")
    parser.add_argument("--no-color", action="store_true", help="ANSI 색상 출력을 끔")
    return parser


def _resolve_calibration_path(explicit: str | None, template: str, id_: str) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(template.format(id=id_)).expanduser()


def main() -> int:
    parser = build_parser()
    # 1차 파스: --config 값만 뽑아낸다 (다른 플래그의 기본값에는 아직 영향 없음).
    preliminary_args, _ = parser.parse_known_args()
    try:
        yaml_defaults = _load_yaml_defaults(Path(preliminary_args.config).expanduser())
    except ConfigLoadError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1
    if yaml_defaults:
        parser.set_defaults(**yaml_defaults)
    # 2차 파스: yaml 기본값이 적용된 상태로 실제 CLI 인자를 최종 해석한다 (CLI가 항상 우선).
    args = parser.parse_args()

    if args.quiet and args.verbose:
        print("[오류] --quiet와 --verbose는 동시에 사용할 수 없습니다.", file=sys.stderr)
        return 2

    opts = cs.ConsoleOptions(
        quiet=args.quiet,
        verbose=args.verbose,
        use_color=cs.resolve_use_color(args.no_color),
    )

    api_token = args.api_token or os.environ.get(API_TOKEN_ENV_VAR) or None

    cs.print_startup_banner(
        opts, leader_id=args.leader_id, follower_id=args.follower_id, rate_hz=args.rate_hz
    )

    leader_cal_path = _resolve_calibration_path(
        args.leader_calibration_path, DEFAULT_LEADER_CALIBRATION_TEMPLATE, args.leader_id
    )
    follower_cal_path = _resolve_calibration_path(
        args.follower_calibration_path, DEFAULT_FOLLOWER_CALIBRATION_TEMPLATE, args.follower_id
    )

    try:
        leader_entries = load_calibration_file(leader_cal_path)
        follower_entries = load_calibration_file(follower_cal_path)
    except CalibrationLoadError as exc:
        cs.print_step(opts, "ERROR", str(exc))
        return 1
    cs.print_step(opts, "PASS", "캘리브레이션 파일 확인")

    calibration_public = {
        "leader": to_public_dict(leader_entries),
        "follower": to_public_dict(follower_entries),
    }

    if args.dry_run:
        cs.print_step(opts, "PASS", "dry-run: 설정 검증 완료 (하드웨어에 연결하지 않았습니다)")
        if args.verbose:
            print(f"[dry-run] leader_port={args.leader_port!r} follower_port={args.follower_port!r}")
            print(f"[dry-run] host={args.host} port={args.port} rate_hz={args.rate_hz:g}")
            print(f"[dry-run] stale_after_ms={args.stale_after_ms:g} max_read_errors={args.max_read_errors}")
            print(f"[dry-run] api_token={'설정됨' if api_token else '없음'}")
            print(f"[dry-run] leader_calibration_path={leader_cal_path}")
            print(f"[dry-run] follower_calibration_path={follower_cal_path}")
        return 0

    if not args.leader_port or not args.follower_port:
        cs.print_step(opts, "ERROR", "--dry-run이 아니면 --leader-port와 --follower-port가 모두 필요합니다.")
        return 2

    leader_reader = ReadOnlySO101Reader(
        name="리더암",
        port=args.leader_port,
        calibration=to_motor_calibration_map(leader_entries),
    )
    follower_reader = ReadOnlySO101Reader(
        name="팔로워암",
        port=args.follower_port,
        calibration=to_motor_calibration_map(follower_entries),
    )

    def _on_poller_log(event: str, arm_key: str, count: int, message: str) -> None:
        label = "리더암" if arm_key == "leader" else "팔로워암"
        if event == "degraded_threshold":
            cs.print_step(opts, "WARN", f"{label} state 읽기 {count}회 연속 실패")
            cs.print_action("마지막 정상값을 stale 상태로 제공합니다.")
        elif event == "recovered":
            cs.print_step(opts, "PASS", f"{label} 연결 복구")

    poller = StatePoller(
        leader_reader=leader_reader,
        follower_reader=follower_reader,
        rate_hz=args.rate_hz,
        stale_after_ms=args.stale_after_ms,
        max_read_errors=args.max_read_errors,
        on_log=_on_poller_log,
    )

    def _on_connect_event(label: str, phase: str, error_message: str | None) -> None:
        if phase == "connecting":
            if not opts.quiet:
                print(f"[연결] {label} 연결 중...")
        elif phase == "connected":
            cs.print_step(opts, "PASS", f"{label} 읽기 연결 완료")
        elif phase == "failed":
            cs.print_step(opts, "ERROR", f"{label} 연결 실패: {error_message}")

    poller.connect_all(on_event=_on_connect_event)

    poller.start()
    cs.print_step(opts, "PASS", "상태 읽기 시작")
    cs.print_server_ready(opts, host=args.host, port=args.port)

    status_printer = cs.PeriodicStatusPrinter(opts, interval_s=STATUS_INTERVAL_S)
    stop_status_event = threading.Event()

    def _status_loop() -> None:
        while not stop_status_event.wait(1.0):
            health = poller.health()
            state = poller.snapshot()
            read_errors = state.leader.read_error_count + state.follower.read_error_count
            status_printer.maybe_print(
                samples=poller.sample_count,
                leader_ok=health.leader_connected and not state.leader.stale,
                follower_ok=health.follower_connected and not state.follower.stale,
                read_errors=read_errors,
            )

    status_thread = threading.Thread(target=_status_loop, name="so101-status-printer", daemon=True)
    status_thread.start()

    app = create_app(poller=poller, calibration_public=calibration_public, api_token=api_token)

    import uvicorn

    uvicorn_config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level="warning" if opts.quiet else "info"
    )
    server = uvicorn.Server(uvicorn_config)

    def _reraise_as_keyboard_interrupt(signum: int, frame: object) -> None:
        # uvicorn.Server는 graceful shutdown을 마친 뒤, 원래 있던 시그널 핸들러를 복원하고
        # signal.raise_signal()로 같은 시그널을 "다시" 보낸다 (프로세스 관리자가 올바른
        # 종료 코드를 볼 수 있게 하려는 의도). SIGINT는 Python 기본 핸들러가
        # KeyboardInterrupt를 던져 우리 try/except까지 전파되지만, SIGTERM의 OS 기본
        # 동작은 그냥 프로세스를 즉시 죽이는 것이라 이 finally 블록(폴링 스레드 정지,
        # serial 연결 해제)이 실행되지 못한다. 그래서 SIGTERM에도 KeyboardInterrupt를
        # 던지는 핸들러를 미리 등록해, 재전달된 시그널이 SIGINT와 동일한 경로로 정상
        # 종료되도록 한다.
        raise KeyboardInterrupt

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _reraise_as_keyboard_interrupt)

    try:
        server.run()
    except KeyboardInterrupt:
        # uvicorn.Server.run()은 SIGINT/SIGTERM을 받으면 내부적으로 이미 정상 종료
        # (graceful shutdown: 접속 종료, "Finished server process" 로그)를 완료한 뒤,
        # 원래 시그널 핸들러를 복원하고 같은 시그널을 다시 보낸다(signal.raise_signal).
        # SIGINT는 Python 기본 핸들러가, SIGTERM은 위에서 등록한 핸들러가 각각
        # KeyboardInterrupt를 던지므로 여기서 흡수하고 아래 정리 코드를 실행한다.
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        stop_status_event.set()
        cs.print_shutdown("서버 종료 처리 중...")
        poller.stop()
        for error in poller.disconnect_all():
            cs.print_step(opts, "ERROR", error)
        cs.print_shutdown("완료")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
