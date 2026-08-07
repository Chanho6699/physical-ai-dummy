#!/usr/bin/env python3
"""노트북 SO-101 상태 서버 -> 데스크탑 MuJoCo -> Windows 브라우저 실시간 MJPEG 뷰어.

WSLg의 `mujoco.viewer` 창이 Windows에서 정상적으로 보이지 않는 문제를 우회하기 위한
대체 실행 경로다 (자세한 조사 내용은 docs/remote_mujoco_diagnostic.md 10절 참고).
`mujoco.viewer`는 이 스크립트/모듈 어디에서도 사용하지 않는다.

기존 `run_remote_mujoco_diagnostic.py`와 마찬가지로 노트북에는 GET만 보내고, 팔로워암에는
어떤 명령도 쓰지 않는다.

실행 예 (기존 raw-leader 모드, 기본값 - 리더 값을 그대로 MuJoCo safety gate에 통과시킴):
    python scripts/run_remote_mujoco_web_viewer.py \\
        --server-url http://100.x.x.x:8001 --joint wrist_flex --host 0.0.0.0 --port 8080 --fps 20

    python scripts/run_remote_mujoco_web_viewer.py \\
        --server-url http://100.x.x.x:8001 --all-joints --host 0.0.0.0 --port 8080 --fps 20

실행 예 (신규 follower-safe 모드 - "실제 팔로워에 보낼 예정"인 안전 명령만 MuJoCo에 적용,
실물 팔로워에는 여전히 아무 것도 쓰지 않는다):
    python scripts/run_remote_mujoco_web_viewer.py \\
        --server-url http://100.x.x.x:8001 --all-joints \\
        --command-source follower-safe --safe-mapper-config configs/follower_safe_mapper.yaml \\
        --host 0.0.0.0 --port 8080 --fps 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.mujoco import console_status as cs
from simulation.mujoco.live_web_viewer import (
    COMMAND_SOURCE_RAW_LEADER,
    VALID_COMMAND_SOURCES,
    LiveWebViewer,
    LiveWebViewerError,
    WebViewerArgs,
    create_http_server,
    detect_local_ip,
)
from simulation.mujoco.remote_state_client import JOINT_NAMES
from simulation.mujoco.safety_event_tracker import SafetyEventTrackerConfig

API_TOKEN_ENV_VAR = "SO101_STATE_SERVER_TOKEN"
BAR = "=" * 68
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "remote_mujoco_diagnostic.yaml"


def _load_yaml_defaults(path: Path) -> dict:
    """configs/remote_mujoco_diagnostic.yaml에서 이 스크립트가 재사용할 값만 뽑는다.

    ``safety.sequence_stall_warn_after_s``/``block_after_s``는 기존 원격 진단(remote_diagnostic.py)이
    쓰는 것과 같은 값을 그대로 재사용한다 - 여기서 새로 정의하지 않는다. ``safety_event_tracking``은
    표시/기록 전용 설정(요구사항)이지 safety 판정 threshold가 아니다.
    """
    if not path.is_file():
        return {}
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}

    defaults: dict = {}
    safety = raw.get("safety")
    if isinstance(safety, dict):
        if safety.get("sequence_stall_warn_after_s") is not None:
            defaults["sequence_stall_warn_after_s"] = float(safety["sequence_stall_warn_after_s"])
        if safety.get("sequence_stall_block_after_s") is not None:
            defaults["sequence_stall_block_after_s"] = float(safety["sequence_stall_block_after_s"])

    tracking = raw.get("safety_event_tracking")
    if isinstance(tracking, dict):
        if tracking.get("clear_after_samples") is not None:
            defaults["clear_after_samples"] = int(tracking["clear_after_samples"])
        if tracking.get("sticky_display_sec") is not None:
            defaults["sticky_display_sec"] = float(tracking["sticky_display_sec"])
        if tracking.get("near_limit_margin_deg") is not None:
            defaults["near_limit_margin_deg"] = float(tracking["near_limit_margin_deg"])
    return defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="노트북 SO-101 상태 서버와 데스크탑 MuJoCo를 연결해 Windows 브라우저에서 실시간으로 봅니다.",
    )
    parser.add_argument("--server-url", required=True, help="노트북 상태 서버 URL (예: http://100.x.x.x:8001)")

    joint_group = parser.add_mutually_exclusive_group()
    joint_group.add_argument(
        "--joint", action="append", choices=JOINT_NAMES, help="표시할 관절 (여러 번 지정 가능, 기본값: wrist_flex)"
    )
    joint_group.add_argument("--all-joints", action="store_true", help="6개 관절 전체를 표시")

    parser.add_argument("--host", default="0.0.0.0", help="HTTP 서버 bind host (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP 서버 포트 (기본: 8080)")
    parser.add_argument("--fps", type=float, default=20.0, help="렌더링 목표 FPS (기본: 20, 권장 15~30)")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="노트북 폴링 주기 Hz (렌더링과 분리됨, 기본: 20)")
    parser.add_argument("--timeout-ms", type=float, default=500.0, help="HTTP timeout ms (기본: 500)")
    parser.add_argument("--stale-after-ms", type=float, default=500.0, help="이 나이(ms)를 넘으면 stale로 간주 (기본: 500)")
    parser.add_argument("--max-retries", type=int, default=3, help="요청 실패 시 재시도 최대 횟수 (기본: 3)")
    parser.add_argument("--api-token", default=None, help=f"노트북 서버 API 토큰 (환경변수 {API_TOKEN_ENV_VAR}로도 지정 가능)")
    parser.add_argument("--mujoco-config", type=Path, default=None, help="MuJoCo safety 설정 YAML 경로")
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG 품질 1~95 (기본: 80)")
    parser.add_argument(
        "--debug-control",
        action="store_true",
        help="1초마다 [제어 진단](selected_joints/mapped_targets/blocking_joints/applied_targets/data.ctrl/data.qpos)을 stdout에 출력",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="원격 진단 설정 YAML 경로 (safety_event_tracking 등)")
    parser.add_argument(
        "--clear-after-samples", type=int, default=None, help="같은 관절/원인이 이 샘플 수만큼 연속 정상이어야 이벤트를 닫음 (기본: 설정 파일, 보통 3)"
    )
    parser.add_argument(
        "--sticky-display-sec", type=float, default=None, help="이벤트 종료 후 화면에 더 남겨둘 시간(초) (기본: 설정 파일, 보통 10)"
    )
    parser.add_argument(
        "--near-limit-margin-deg", type=float, default=None, help="NEAR_JOINT_LIMIT 표시 threshold(도, safety 판정과 무관) (기본: 설정 파일, 보통 5)"
    )
    parser.add_argument("--sequence-stall-warn-after-s", type=float, default=None, help="sequence 정지 WARN 판정 시간(초)")
    parser.add_argument("--sequence-stall-block-after-s", type=float, default=None, help="sequence 정지 BLOCKED 판정 시간(초)")
    parser.add_argument(
        "--events-report-dir", type=Path, default=None, help="safety_events_<timestamp>.json/.csv 저장 디렉터리 (기본: reports/remote_mujoco_diagnostic)"
    )
    parser.add_argument(
        "--command-source",
        choices=VALID_COMMAND_SOURCES,
        default=COMMAND_SOURCE_RAW_LEADER,
        help="MuJoCo에 적용할 명령의 출처. raw-leader(기본, 기존 동작)=리더 값을 safety gate만 거쳐 그대로 적용. "
        "follower-safe=실제 팔로워에 보낼 예정인 안전 명령(매핑+range+rate-limit+hold)만 적용 (실물 쓰기는 여전히 없음).",
    )
    parser.add_argument(
        "--safe-mapper-config", type=Path, default=None, help="follower-safe 모드 설정 YAML (기본: configs/follower_safe_mapper.yaml)"
    )
    parser.add_argument(
        "--follower-safe-report-dir",
        type=Path,
        default=None,
        help="follower_safe_mapper_<timestamp>.json/.csv 저장 디렉터리 (기본: reports/remote_mujoco_diagnostic)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    yaml_defaults = _load_yaml_defaults(Path(args.config).expanduser())

    joints = tuple(JOINT_NAMES) if args.all_joints else tuple(args.joint or ["wrist_flex"])
    api_token = args.api_token or os.environ.get(API_TOKEN_ENV_VAR) or None

    def _pick(cli_value, yaml_key, default):
        if cli_value is not None:
            return cli_value
        return yaml_defaults.get(yaml_key, default)

    tracker_defaults = SafetyEventTrackerConfig()
    safety_event_config = SafetyEventTrackerConfig(
        clear_after_samples=_pick(args.clear_after_samples, "clear_after_samples", tracker_defaults.clear_after_samples),
        sticky_display_sec=_pick(args.sticky_display_sec, "sticky_display_sec", tracker_defaults.sticky_display_sec),
        near_limit_margin_deg=_pick(args.near_limit_margin_deg, "near_limit_margin_deg", tracker_defaults.near_limit_margin_deg),
    )

    web_args = WebViewerArgs(
        server_url=args.server_url,
        joints=joints,
        host=args.host,
        port=args.port,
        fps=args.fps,
        rate_hz=args.rate_hz,
        timeout_ms=args.timeout_ms,
        stale_after_ms=args.stale_after_ms,
        max_retries=args.max_retries,
        api_token=api_token,
        mujoco_config_path=args.mujoco_config,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        jpeg_quality=args.jpeg_quality,
        debug_control=args.debug_control,
        sequence_stall_warn_after_s=_pick(args.sequence_stall_warn_after_s, "sequence_stall_warn_after_s", 2.0),
        sequence_stall_block_after_s=_pick(args.sequence_stall_block_after_s, "sequence_stall_block_after_s", 5.0),
        safety_event_config=safety_event_config,
        events_report_dir=args.events_report_dir,
        command_source=args.command_source,
        safe_mapper_config_path=args.safe_mapper_config,
        follower_safe_report_dir=args.follower_safe_report_dir,
    )

    print(BAR)
    print("[준비] SO-101 MuJoCo 실시간 웹 뷰어")
    print(BAR)
    print(f"[서버] {args.server_url}")
    print(f"[관절] {list(joints)}")
    print(f"[command-source] {args.command_source}")
    if args.command_source == "follower-safe":
        print(f"[안내] follower-safe 매퍼 설정: {args.safe_mapper_config or 'configs/follower_safe_mapper.yaml (기본값)'}")
        print("[안내] MuJoCo에는 '실제 팔로워에 보낼 예정'인 안전 명령(limited_command_deg)만 적용됩니다 - 실물 팔로워 쓰기는 없습니다.")

    viewer = LiveWebViewer(web_args)
    try:
        viewer.preflight()
    except LiveWebViewerError as exc:
        cs.print_remote_error(str(exc))
        return 1
    print("[통과] 노트북 서버 연결 + MuJoCo 모델/safety 설정 로딩 완료")
    if viewer.follower_mapper is not None:
        for joint, cal in viewer.follower_mapper.calibrations.items():
            if cal.verified:
                print(f"[캘리브레이션] {joint}: [{cal.range_min_deg:.1f}, {cal.range_max_deg:.1f}] deg (source={cal.source})")
            else:
                print(f"[캘리브레이션] {joint}: UNVERIFIED_RANGE - 출력 후보에서 제외됨")

    try:
        server = create_http_server(viewer)
    except OSError as exc:
        cs.print_remote_error(f"HTTP 서버를 시작할 수 없습니다 ({args.host}:{args.port}): {exc}")
        return 1

    viewer.start()

    print(BAR)
    print("[웹 뷰어] Windows 브라우저에서 여세요:")
    print(f"http://localhost:{args.port}")
    wsl_ip = detect_local_ip()
    if wsl_ip:
        print(f"[대체] localhost forwarding이 안 되면: http://{wsl_ip}:{args.port}")
    else:
        print("[대체] localhost forwarding이 안 되면 WSL 안에서 `hostname -I`로 IP를 확인해 http://<IP>:8080으로 접속하세요.")
    print("[대체] 또는 Windows PowerShell에서: wsl hostname -I")
    print(BAR)
    print("[안내] 브라우저 탭을 닫아도 이 프로세스는 계속 동작합니다 (재접속 가능). 완전히 끝내려면 Ctrl+C.")
    print(BAR)

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[중단] Ctrl+C로 종료합니다.")
    finally:
        server.server_close()
        viewer.stop()
    print("[완료] 웹 뷰어를 종료했습니다.")
    if viewer.last_safety_report_paths is not None:
        json_path, csv_path = viewer.last_safety_report_paths
        print(f"[Safety 이벤트 JSON] {json_path}")
        print(f"[Safety 이벤트 CSV] {csv_path}")
    if viewer.last_follower_safe_report_paths is not None:
        json_path, csv_path = viewer.last_follower_safe_report_paths
        print(f"[Follower-safe 매퍼 JSON] {json_path}")
        print(f"[Follower-safe 매퍼 CSV] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
