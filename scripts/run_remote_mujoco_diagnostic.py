#!/usr/bin/env python3
"""노트북 SO-101 읽기 전용 상태 서버 <-> 데스크탑 MuJoCo 실시간 진단 CLI.

노트북에서 실행 중인 읽기 전용 상태 서버(GET /health, /state, /calibration)에 접속해
리더암 관절값을 기존 action mapping/safety gate를 거쳐 MuJoCo SO-101에 반영하고,
리더-팔로워 차이를 실시간으로 비교/진단한다. 노트북에는 GET 요청만 보내며, 팔로워암에는
어떤 명령도 전송하지 않는다. 자세한 설명은 docs/remote_mujoco_diagnostic.md 참고.

실행 예시:
    python scripts/run_remote_mujoco_diagnostic.py \\
        --server-url http://100.x.x.x:8001 --headless --joint wrist_flex --duration 20

    python scripts/run_remote_mujoco_diagnostic.py \\
        --server-url http://100.x.x.x:8001 --headless --all-joints --duration 30

설정/CLI/MuJoCo 로딩/관절 mapping/safety 설정만 확인하고 네트워크 호출 없이 끝내려면:
    python scripts/run_remote_mujoco_diagnostic.py --server-url http://127.0.0.1:8001 --dry-run

WSLg GUI 창이 제대로 렌더링되지 않을 때의 대체 경로 (GUI 없이 PNG/MP4로 저장):
    python scripts/run_remote_mujoco_diagnostic.py \\
        --server-url http://100.x.x.x:8001 --offscreen --joint wrist_flex --duration 10 \\
        --save-frames reports/remote_mujoco_diagnostic/frames

    python scripts/run_remote_mujoco_diagnostic.py \\
        --server-url http://100.x.x.x:8001 --offscreen --joint wrist_flex --duration 10 \\
        --video-output reports/remote_mujoco_diagnostic/wrist_flex.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.mujoco.diagnostic_analysis import DiagnosticConfig
from simulation.mujoco.remote_diagnostic import NetworkSafetyConfig, RemoteDiagnosticArgs, run_diagnostic
from simulation.mujoco.remote_state_client import JOINT_NAMES

API_TOKEN_ENV_VAR = "SO101_STATE_SERVER_TOKEN"  # 노트북 서버 실행 시 사용한 것과 같은 값이어야 한다.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "remote_mujoco_diagnostic.yaml"

_CONFIG_KEY_MAP: dict[tuple[str, str], str] = {
    ("remote", "server_url"): "server_url",
    ("remote", "timeout_ms"): "timeout_ms",
    ("remote", "stale_after_ms"): "stale_after_ms",
    ("remote", "max_retries"): "max_retries",
    ("remote", "rate_hz"): "rate_hz",
    ("console", "quiet"): "quiet",
    ("console", "verbose"): "verbose",
    ("console", "no_color"): "no_color",
}


class ConfigLoadError(RuntimeError):
    pass


def _load_yaml_defaults(path: Path) -> tuple[dict[str, object], dict]:
    if not path.is_file():
        return {}, {}
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
    return defaults, raw


def _diagnostic_config_from_yaml(raw: dict) -> DiagnosticConfig:
    section = raw.get("diagnostic")
    if not isinstance(section, dict):
        return DiagnosticConfig()
    defaults = DiagnosticConfig()
    kwargs = {}
    for field_name in defaults.__dataclass_fields__:
        if field_name in section and section[field_name] is not None:
            kwargs[field_name] = type(getattr(defaults, field_name))(section[field_name])
    return DiagnosticConfig(**kwargs)


def _network_safety_from_yaml(raw: dict) -> NetworkSafetyConfig:
    section = raw.get("safety")
    if not isinstance(section, dict):
        return NetworkSafetyConfig()
    defaults = NetworkSafetyConfig()
    kwargs = {}
    for field_name in defaults.__dataclass_fields__:
        if field_name in section and section[field_name] is not None:
            kwargs[field_name] = type(getattr(defaults, field_name))(section[field_name])
    return NetworkSafetyConfig(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="노트북 SO-101 읽기 전용 상태 서버와 데스크탑 MuJoCo를 실시간으로 연결해 진단합니다.",
    )
    parser.add_argument("--server-url", default=None, help="노트북 상태 서버 URL (예: http://100.x.x.x:8001)")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", help="MuJoCo GUI viewer로 실행")
    mode_group.add_argument("--headless", action="store_true", help="GUI 없이 실행 (기본값)")
    mode_group.add_argument(
        "--offscreen",
        action="store_true",
        help="GUI 창 없이 mujoco.Renderer로 프레임을 렌더링해 PNG/MP4로 저장 (WSLg GUI 대체 경로, --save-frames/--video-output과 함께 사용)",
    )

    offscreen_group = parser.add_argument_group("offscreen 옵션 (--offscreen과 함께 사용)")
    offscreen_group.add_argument("--save-frames", type=Path, default=None, help="렌더링한 프레임을 PNG로 저장할 디렉터리")
    offscreen_group.add_argument("--video-output", type=Path, default=None, help="렌더링한 프레임을 이어붙여 저장할 MP4 경로")
    offscreen_group.add_argument("--offscreen-width", type=int, default=640, help="오프스크린 렌더링 프레임 폭(px) (기본: 640)")
    offscreen_group.add_argument("--offscreen-height", type=int, default=480, help="오프스크린 렌더링 프레임 높이(px) (기본: 480)")
    offscreen_group.add_argument(
        "--offscreen-fps", type=float, default=None, help="MP4 저장 fps (기본: --rate-hz와 동일)"
    )

    joint_group = parser.add_mutually_exclusive_group()
    joint_group.add_argument(
        "--joint",
        action="append",
        choices=JOINT_NAMES,
        help="표시/진단할 관절 (여러 번 지정 가능, 기본값: wrist_flex)",
    )
    joint_group.add_argument("--all-joints", action="store_true", help="6개 관절 전체를 표시/진단")

    parser.add_argument("--duration", type=float, default=20.0, help="실행 시간(초) (기본: 20.0)")
    parser.add_argument("--rate-hz", type=float, default=None, help="폴링 주기 Hz (기본: 설정 파일 값, 보통 20)")
    parser.add_argument("--timeout-ms", type=float, default=None, help="HTTP timeout ms (기본: 설정 파일 값, 보통 500)")
    parser.add_argument("--stale-after-ms", type=float, default=None, help="이 나이(ms)를 넘으면 stale로 간주 (기본: 500)")
    parser.add_argument("--max-retries", type=int, default=None, help="요청 실패 시 재시도 최대 횟수 (기본: 3, 무한 재시도 없음)")
    parser.add_argument(
        "--api-token", default=None, help=f"노트북 서버 API 토큰 (환경변수 {API_TOKEN_ENV_VAR}로도 지정 가능)"
    )
    parser.add_argument("--record", action="store_true", help="샘플별 CSV 상세 기록을 저장합니다 (기본은 JSON 요약만 저장)")
    parser.add_argument("--report-path", type=Path, default=None, help="JSON 리포트 저장 경로 (CSV는 같은 이름에 .csv)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="원격 진단 설정 YAML 경로")
    parser.add_argument(
        "--mujoco-config", type=Path, default=None, help="MuJoCo safety 설정 YAML 경로 (기본: configs/mujoco_so101.yaml)"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="네트워크 호출/MuJoCo actuator 적용 없이 설정과 파이프라인만 검사"
    )
    parser.add_argument("--quiet", action="store_true", help="최종 결과와 오류만 출력")
    parser.add_argument("--verbose", action="store_true", help="전체 관절을 표 형식으로 출력")
    parser.add_argument("--no-color", action="store_true", help="ANSI 색상 출력을 끔")
    return parser


def main() -> int:
    parser = build_parser()
    preliminary_args, _ = parser.parse_known_args()
    try:
        yaml_defaults, raw_yaml = _load_yaml_defaults(Path(preliminary_args.config).expanduser())
    except ConfigLoadError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1
    if yaml_defaults:
        parser.set_defaults(**yaml_defaults)
    args = parser.parse_args()

    if args.quiet and args.verbose:
        print("[오류] --quiet와 --verbose는 동시에 사용할 수 없습니다.", file=sys.stderr)
        return 2

    if not args.server_url:
        print("[오류] --server-url이 필요합니다 (또는 설정 파일 remote.server_url).", file=sys.stderr)
        return 2

    if args.offscreen and not args.save_frames and not args.video_output:
        print("[오류] --offscreen을 사용하려면 --save-frames 또는 --video-output 중 최소 하나를 지정해야 합니다.", file=sys.stderr)
        return 2
    if (args.save_frames or args.video_output) and not args.offscreen:
        print("[오류] --save-frames/--video-output은 --offscreen과 함께 사용해야 합니다.", file=sys.stderr)
        return 2

    joints = tuple(JOINT_NAMES) if args.all_joints else tuple(args.joint or ["wrist_flex"])

    api_token = args.api_token or os.environ.get(API_TOKEN_ENV_VAR) or None

    diagnostic_config = _diagnostic_config_from_yaml(raw_yaml)
    network_safety = _network_safety_from_yaml(raw_yaml)

    mode = "gui" if args.gui else ("offscreen" if args.offscreen else "headless")

    diag_args = RemoteDiagnosticArgs(
        server_url=args.server_url,
        mode=mode,
        joints=joints,
        duration_sec=args.duration,
        rate_hz=args.rate_hz if args.rate_hz is not None else 20.0,
        timeout_ms=args.timeout_ms if args.timeout_ms is not None else 500.0,
        stale_after_ms=args.stale_after_ms if args.stale_after_ms is not None else 500.0,
        max_retries=args.max_retries if args.max_retries is not None else 3,
        api_token=api_token,
        record=args.record,
        report_path=args.report_path,
        mujoco_config_path=args.mujoco_config,
        diagnostic_config=diagnostic_config,
        network_safety=network_safety,
        quiet=args.quiet,
        verbose=args.verbose,
        no_color=args.no_color,
        dry_run=args.dry_run,
        offscreen_save_frames_dir=args.save_frames,
        offscreen_video_path=args.video_output,
        offscreen_width=args.offscreen_width,
        offscreen_height=args.offscreen_height,
        offscreen_fps=args.offscreen_fps,
    )

    outcome = run_diagnostic(diag_args)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
