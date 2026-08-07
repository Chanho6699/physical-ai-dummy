#!/usr/bin/env python3
"""Shadow Teleop Diagnostic - 리더+팔로워 wrist_roll을 동시에 read-only로 관측한다.

leader arm을 손으로 움직이면서 leader wrist_roll 상태와 follower wrist_roll 상태를 동시에
읽고, 터미널 대시보드로 실시간 표시하며 CSV로 기록한다. **follower Goal_Position write는
이 스크립트 어디에서도 하지 않는다** - 리더를 움직여도 팔로워는 따라 움직이지 않는다(이번
단계는 관찰 전용이다). 이 파일에는 ``armed``에 해당하는 모드/분기가 아예 없다 - 팔로워
쪽으로 나가는 유일한 통신은 read-only 레지스터 read뿐이다.

## 실시간 안전장치

``--follower-move-abort-threshold-deg``(기본
``hardware.safety.shadow_teleop_diagnostic.DEFAULT_FOLLOWER_MOVE_ABORT_THRESHOLD_DEG``,
약 0.5°)를 넘는 follower 이동이 감지되면 즉시 샘플링을 멈추고 원인 조사를 위한 경고를
출력한다(섹션 10: "만약 follower가 움직이면 즉시 실험을 중단하고 원인을 보고하세요").

## 재사용

포트/calibration 경로 해석은 ``hardware/safety/calibration_resolution.py``(팔로워)와 이
파일 안의 대칭 함수(리더, ``configs/hardware.local.json``의 ``teleop`` 절 + LeRobot leader
표준 캐시 경로)를 쓴다. 포트 점유 검사는 ``hardware/safety/port_conflict.py``를, 실제
read/샘플링/분석은 ``hardware/safety/shadow_teleop_diagnostic.py``를, 대시보드 출력은
``hardware/safety/shadow_teleop_console.py``를 그대로 재사용한다.

실행 예시(기본 20초):
    python scripts/run_shadow_teleop_diagnostic.py

실행 예시(포트/캘리브레이션 직접 지정, 30초):
    python scripts/run_shadow_teleop_diagnostic.py \\
        --duration-sec 30 \\
        --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14029966-if00 \\
        --follower-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113538-if00

Ctrl+C로 언제든 중단할 수 있다 - 중단 시에도 follower write/torque 변경 없이 안전하게
disconnect하고 CSV를 flush한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.safety import calibration_resolution as calres
from hardware.safety import port_conflict as pc
from hardware.safety.shadow_teleop_console import TerminalDashboard, render_dashboard_lines
from hardware.safety.shadow_teleop_diagnostic import (
    CSV_FIELDNAMES,
    DEFAULT_FOLLOWER_MOVE_ABORT_THRESHOLD_DEG,
    FollowerWristRollStateReader,
    LeaderWristRollReader,
    ShadowTeleopSampler,
    WristRollCalibration,
    build_command_delta_reference_table,
    compute_run_analysis,
    run_sampling_loop,
)
from hardware.state_server.calibration_loader import CalibrationLoadError, load_calibration_file

# LeRobot이 실제로 사용하는 표준 리더 캘리브레이션 캐시 경로
# (scripts/run_hardware_state_server.py의 DEFAULT_LEADER_CALIBRATION_TEMPLATE와 동일).
LEADER_CALIBRATION_PATH_TEMPLATE = "~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/{id}.json"
DEFAULT_LEADER_ID_FALLBACK = "chanho_leader"

TARGET_JOINT = "wrist_roll"
DEFAULT_DURATION_SEC = 20.0
DEFAULT_DASHBOARD_INTERVAL_S = 0.15  # 약 6.7Hz - 요구사항 5~10Hz 범위 안
DEFAULT_NUM_READ_RETRIES = 2
REPORTS_DIR_RELATIVE = Path("reports") / "shadow_teleop"


class RefusalError(RuntimeError):
    """설정/안전 정책 위반으로 실행을 거부해야 할 때 (하드웨어 오류와 구분)."""


# ---------------------------------------------------------------------------
# 리더 포트/calibration 경로 해석 (calibration_resolution.py의 팔로워 해석과 대칭)
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


def resolve_leader_calibration_path(
    *, cli_calibration_path: str | None, cli_calibration_id: str | None, project_root: Path
) -> tuple[Path, str]:
    if cli_calibration_path:
        return Path(cli_calibration_path).expanduser(), "cli_path"
    if cli_calibration_id:
        return Path(LEADER_CALIBRATION_PATH_TEMPLATE.format(id=cli_calibration_id)).expanduser(), "cli_id"

    local = _load_local_config(project_root)
    if local:
        teleop = local.get("teleop") or {}
        explicit_path = teleop.get("calibration_path")
        if explicit_path:
            return Path(explicit_path).expanduser(), "local_config"
        local_id = teleop.get("id")
        if local_id:
            return Path(LEADER_CALIBRATION_PATH_TEMPLATE.format(id=local_id)).expanduser(), "local_config"

    return Path(LEADER_CALIBRATION_PATH_TEMPLATE.format(id=DEFAULT_LEADER_ID_FALLBACK)).expanduser(), "default_fallback"


@dataclass
class ResolvedShadowConfig:
    leader_port: str
    leader_port_source: str
    leader_calibration_path: Path
    leader_calibration_source: str
    follower_port: str
    follower_port_source: str
    follower_calibration_path: Path
    follower_calibration_source: str


def resolve_config(args: argparse.Namespace) -> ResolvedShadowConfig:
    leader_port, leader_port_source = resolve_leader_port(cli_port=args.leader_port, project_root=PROJECT_ROOT)
    leader_cal_path, leader_cal_source = resolve_leader_calibration_path(
        cli_calibration_path=args.leader_calibration_path,
        cli_calibration_id=args.leader_calibration_id,
        project_root=PROJECT_ROOT,
    )
    follower_port, follower_port_source = calres.resolve_port(cli_port=args.follower_port, project_root=PROJECT_ROOT)
    follower_cal_path, follower_cal_source = calres.resolve_calibration_path(
        cli_calibration_path=args.follower_calibration_path,
        cli_calibration_id=args.follower_calibration_id,
        project_root=PROJECT_ROOT,
        allow_default_fallback=True,
    )
    return ResolvedShadowConfig(
        leader_port=leader_port,
        leader_port_source=leader_port_source,
        leader_calibration_path=leader_cal_path,
        leader_calibration_source=leader_cal_source,
        follower_port=follower_port,
        follower_port_source=follower_port_source,
        follower_calibration_path=follower_cal_path,
        follower_calibration_source=follower_cal_source,
    )


# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SO-101 리더+팔로워 wrist_roll Shadow Teleop Diagnostic (read-only, follower write 없음).",
    )
    parser.add_argument(
        "--duration-sec", type=float, default=DEFAULT_DURATION_SEC, help=f"최대 실행 시간(초) (기본 {DEFAULT_DURATION_SEC:g})"
    )
    parser.add_argument("--leader-port", default=None, help="지정하지 않으면 configs/hardware.local.json의 teleop.port를 사용한다.")
    parser.add_argument("--follower-port", default=None, help="지정하지 않으면 configs/hardware.local.json의 robot.port를 사용한다.")
    parser.add_argument("--leader-calibration-path", default=None, help="리더 calibration JSON 파일 경로.")
    parser.add_argument("--follower-calibration-path", default=None, help="팔로워 calibration JSON 파일 경로.")
    parser.add_argument("--leader-calibration-id", default=None, help="LeRobot 표준 캐시 경로에 넣을 리더 id.")
    parser.add_argument("--follower-calibration-id", default=None, help="LeRobot 표준 캐시 경로에 넣을 팔로워 id.")
    parser.add_argument(
        "--accel-refresh-interval-s",
        type=float,
        default=1.0,
        help="팔로워 Acceleration/Acceleration_Multiplier를 다시 읽는 최소 간격(초) (기본 1.0).",
    )
    parser.add_argument(
        "--dashboard-interval-s",
        type=float,
        default=DEFAULT_DASHBOARD_INTERVAL_S,
        help=f"터미널 대시보드 갱신 최소 간격(초) (기본 {DEFAULT_DASHBOARD_INTERVAL_S:g}, 약 5~10Hz 범위).",
    )
    parser.add_argument(
        "--follower-move-abort-threshold-deg",
        type=float,
        default=DEFAULT_FOLLOWER_MOVE_ABORT_THRESHOLD_DEG,
        help=(
            "follower_present_delta_from_start_deg의 절대값이 이 값을 넘으면 즉시 중단한다 "
            f"(기본 {DEFAULT_FOLLOWER_MOVE_ABORT_THRESHOLD_DEG:g}°). 0 이하로 지정하면 이 안전장치를 끈다."
        ),
    )
    parser.add_argument(
        "--num-read-retries", type=int, default=DEFAULT_NUM_READ_RETRIES, help=f"레지스터 read 재시도 횟수 (기본 {DEFAULT_NUM_READ_RETRIES})"
    )
    parser.add_argument("--csv-dir", default=None, help=f"CSV 저장 디렉터리 (기본 {REPORTS_DIR_RELATIVE}/)")
    parser.add_argument("--json-report", action="store_true", help="분석 요약을 CSV와 같은 디렉터리에 JSON으로도 저장한다.")
    parser.add_argument("--no-dashboard", action="store_true", help="터미널 대시보드 출력을 끈다 (CSV 기록/분석은 그대로 수행).")
    return parser


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _load_wrist_roll_calibration(path: Path, *, label: str) -> WristRollCalibration:
    entries = load_calibration_file(path)
    entry = entries[TARGET_JOINT]
    print(f"{label} calibration 로드 성공 (motor_id={entry.id}, range=[{entry.range_min}, {entry.range_max}], path_source 확인됨)")
    return WristRollCalibration(
        motor_id=entry.id,
        drive_mode=entry.drive_mode,
        homing_offset=entry.homing_offset,
        range_min=entry.range_min,
        range_max=entry.range_max,
    )


def run(
    args: argparse.Namespace,
    *,
    leader_reader_factory=LeaderWristRollReader,
    follower_reader_factory=FollowerWristRollStateReader,
    stdout=sys.stdout,
) -> int:
    """CLI 본체 - 테스트가 reader factory를 가짜로 바꿔치기할 수 있도록 분리했다."""

    try:
        config = resolve_config(args)
    except calres.ResolutionError as exc:
        print(f"거부: {exc}")
        return 2
    except RefusalError as exc:
        print(f"거부: {exc}")
        return 2

    print("Shadow Teleop Diagnostic - READ ONLY (follower write 없음, 실물 write 0회 보장)")
    print(f"leader_port={config.leader_port} (source={config.leader_port_source})")
    print(f"follower_port={config.follower_port} (source={config.follower_port_source})")

    _print_section("포트 점유 검사")
    leader_conflict = pc.check_port_conflict(config.leader_port)
    follower_conflict = pc.check_port_conflict(config.follower_port)
    print(f"leader: busy={leader_conflict.busy} busy_confirmed={leader_conflict.busy_confirmed}")
    print(f"follower: busy={follower_conflict.busy} busy_confirmed={follower_conflict.busy_confirmed}")
    for note in (*leader_conflict.notes, *follower_conflict.notes):
        print(f"  - {note}")
    if leader_conflict.busy or follower_conflict.busy:
        print("거부: 리더 또는 팔로워 포트가 점유 중(또는 판정 불가)이라 연결을 시도하지 않습니다. write_count=0")
        return 2

    _print_section("calibration 로딩")
    try:
        leader_calibration = _load_wrist_roll_calibration(config.leader_calibration_path, label="리더")
        follower_calibration = _load_wrist_roll_calibration(config.follower_calibration_path, label="팔로워")
    except CalibrationLoadError as exc:
        print(f"거부: calibration 로드 실패: {exc}. write_count=0")
        return 2

    leader_reader = leader_reader_factory(
        port=config.leader_port, calibration=leader_calibration, num_read_retries=args.num_read_retries
    )
    follower_reader = follower_reader_factory(
        port=config.follower_port, calibration=follower_calibration, num_read_retries=args.num_read_retries
    )

    csv_dir = Path(args.csv_dir).expanduser() if args.csv_dir else (PROJECT_ROOT / REPORTS_DIR_RELATIVE)
    csv_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = csv_dir / f"shadow_wrist_roll_{timestamp}.csv"

    _print_section("연결 (read-only)")
    leader_connected = False
    follower_connected = False
    result = None
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(csv_file, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()

        try:
            leader_reader.connect()
            leader_connected = True
            print(f"리더 연결 성공 (port={config.leader_port})")
        except (ConnectionError, RuntimeError) as exc:
            print(f"거부: 리더 연결 실패: {exc}. write_count=0")
            return 1

        try:
            follower_reader.connect()
            follower_connected = True
            print(f"팔로워 연결 성공 (port={config.follower_port})")
        except (ConnectionError, RuntimeError) as exc:
            print(f"거부: 팔로워 연결 실패: {exc}. write_count=0")
            return 1

        sampler = ShadowTeleopSampler(
            leader_reader=leader_reader,
            follower_reader=follower_reader,
            leader_calibration=leader_calibration,
            follower_calibration=follower_calibration,
            accel_refresh_interval_s=args.accel_refresh_interval_s,
        )

        dashboard = None if args.no_dashboard else TerminalDashboard(stream=stdout)
        last_dashboard_at = 0.0
        loop_start = time.monotonic()

        _print_section("샘플링 시작 (Ctrl+C로 언제든 중단 가능)")

        def _on_sample(sample) -> None:
            writer.writerow(sample.to_csv_row())
            nonlocal last_dashboard_at
            now = time.monotonic()
            if dashboard is not None and (now - last_dashboard_at) >= args.dashboard_interval_s:
                elapsed = max(sample.elapsed_sec, 1e-9)
                read_rate_hz = (sample.sample_index + 1) / elapsed
                dashboard.update(
                    render_dashboard_lines(
                        sample, read_rate_hz=read_rate_hz, write_count=0, sample_count=sample.sample_index + 1
                    )
                )
                last_dashboard_at = now

        abort_threshold = args.follower_move_abort_threshold_deg if args.follower_move_abort_threshold_deg > 0 else None
        result = run_sampling_loop(
            sampler,
            duration_sec=args.duration_sec,
            on_sample=_on_sample,
            follower_move_abort_threshold_deg=abort_threshold,
        )
        elapsed_wall = time.monotonic() - loop_start
    finally:
        # 순서와 무관하게 둘 다 시도한다 - 하나가 실패해도 나머지 disconnect는 반드시 시도된다.
        try:
            if follower_connected:
                follower_reader.disconnect()
        except Exception as exc:  # noqa: BLE001 - 종료 중 오류로 결과 자체를 잃지 않는다
            print(f"경고: 팔로워 disconnect 중 오류: {exc}")
        try:
            if leader_connected:
                leader_reader.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"경고: 리더 disconnect 중 오류: {exc}")
        csv_file.flush()
        csv_file.close()

    if result is None:
        return 1

    _print_section("종료")
    print(f"stopped_reason={result.stopped_reason}")
    if result.error is not None:
        print(f"error={result.error}")
    print(f"samples_collected={len(result.samples)} elapsed_wall_sec={elapsed_wall:.2f}")
    print(f"CSV 저장 위치: {csv_path}")

    analysis = compute_run_analysis(result.samples)
    reference_table = build_command_delta_reference_table(
        leader_calibration=leader_calibration, follower_calibration=follower_calibration
    )

    _print_section("분석 (섹션 11)")
    print(json.dumps(analysis, indent=2, ensure_ascii=False, default=str))
    if analysis.get("sample_count", 0) > 0:
        if analysis["leader_moved_while_follower_fixed"]:
            print("확인: 리더가 움직이는 동안 follower Present_Position은 고정되어 있었습니다 (예상된 정상 결과).")
        elif result.stopped_reason == "follower_moved_unexpectedly":
            print("경고: follower가 실제로 움직인 것으로 보입니다 - 원인 조사가 필요합니다 (아래 error 참고).")
        else:
            print(
                "참고: 리더 이동 범위가 판정 문턱값 이하이거나 follower_present_deg_range가 문턱값을 넘어 "
                "'리더 이동/follower 고정' 조건을 명확히 만족하지 않았습니다 - 위 raw 수치를 직접 확인하세요."
            )

    _print_section("섹션 12: 향후 restricted teleop용 tick<->degree 참고표 (계산만, write 없음)")
    print(json.dumps(reference_table, indent=2, ensure_ascii=False, default=str))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leader_port_source": config.leader_port_source,
        "follower_port_source": config.follower_port_source,
        "leader_calibration_source": config.leader_calibration_source,
        "follower_calibration_source": config.follower_calibration_source,
        "leader_motor_id": leader_calibration.motor_id,
        "follower_motor_id": follower_calibration.motor_id,
        "csv_path": str(csv_path),
        "stopped_reason": result.stopped_reason,
        "analysis": analysis,
        "command_delta_reference_table": reference_table,
        "write_count": 0,
    }
    if args.json_report:
        json_path = csv_dir / f"shadow_wrist_roll_{timestamp}_report.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nJSON 리포트 저장: {json_path}")

    print("\nwrite_count=0 (이 스크립트는 follower/leader에 어떤 write도 수행하지 않았습니다.)")

    return 0 if result.stopped_reason in ("duration_elapsed", "keyboard_interrupt") else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
