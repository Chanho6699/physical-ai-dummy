#!/usr/bin/env python3
"""SO-101 팔로워 wrist_roll 단일 관절 안전 시험 CLI.

일곱 모드를 지원한다:

- ``--mode inspect-only``: 포트/calibration 점검 + (연결 가능하면) wrist_roll 현재
  위치를 읽기만 한다.
- ``--mode dry-run``: inspect-only에 더해 ``--direction``으로 지정한 방향으로 0.1°
  단위, 최대 1°까지의 이동 계획을 **계산**하고 각 스텝의 PASS/BLOCKED만 보여준다.
  실제로 그 계획을 실행하지 않는다.
- ``--mode register-diagnostic``: wrist_roll(motor id 5)의 ``Torque_Enable``/
  ``Goal_Position``/``Present_Position``/``Moving``(+ 설치된 control table에 존재하는
  선택 레지스터)을 읽기 전용으로 스냅샷하고, ``TORQUE_DISABLED``/``GOAL_NOT_LATCHED``/
  ``COMMAND_LATCHED_BUT_NO_MOTION``/``MOTOR_STILL_MOVING``/``FAULT_OR_PROTECTION``/
  ``UNKNOWN`` 중 하나로 판정한다. 어떤 write도 하지 않는다 - armed 실행 후 예상과 다른
  결과(예: NO_MOTION)가 나왔을 때 원인을 조사하기 위한 순수 진단 모드다. 자세한 근거는
  ``hardware/safety/single_joint_register_diagnostic.py`` 모듈 docstring 참고.
- ``--mode servo-parameter-diagnostic``: wrist_roll(motor id 5)의 STS3215 **설정**
  레지스터(``CW_Dead_Zone``/``CCW_Dead_Zone``/``Minimum_Startup_Force``/
  ``Operating_Mode``/``Acceleration``/``Goal_Velocity``/``Torque_Limit``/PID 등)를
  읽기 전용으로 스냅샷하고, ``DEAD_ZONE_LIKELY``/``STARTUP_FORCE_THRESHOLD_LIKELY``/
  ``CONTROL_MODE_RESTRICTION``/``VELOCITY_OR_ACCELERATION_RESTRICTION``/
  ``TORQUE_LIMIT_RESTRICTION``/``NO_CONFIGURATION_CAUSE_FOUND``/``UNKNOWN`` 중
  하나 이상으로 판정한다. 1~2 tick Goal_Position 명령이 latch는 되지만 실제로
  움직이지 않는(``COMMAND_LATCHED_BUT_NO_MOTION``) 현상의 **설정상 원인**을 조사하기
  위한 순수 read-only 진단이며, 다음에 시도해볼 만한 단발 후보(최대 2개, write 없이
  계산만)도 함께 보여준다. 자세한 근거는
  ``hardware/safety/single_joint_servo_parameter_diagnostic.py`` 모듈 docstring 참고.
- ``--mode all-joint-parameter-diagnostic``: wrist_roll뿐 아니라 follower 6개 관절
  (``shoulder_pan``/``shoulder_lift``/``elbow_flex``/``wrist_flex``/``wrist_roll``/
  ``gripper``) 전부의 ``Acceleration``/``Maximum_Acceleration``/``Torque_Enable``/
  ``Operating_Mode``/``Goal_Position``/``Present_Position``/``Moving``/``Status``(+
  가능하면 dead-zone/startup-force/torque-limit)를 읽기 전용으로 비교한다.
  wrist_roll의 ``Acceleration=0``이 wrist_roll만의 개별 문제인지, 6개 모터 공통
  상태인지 ``ALL_ACCELERATION_ZERO``/``WRIST_ROLL_ONLY_ZERO``/
  ``MIXED_ACCELERATION_STATE``/``ALL_ACCELERATION_CONFIGURED``/``READ_INCOMPLETE``로
  판정한다. ``SOFollower.connect()``/``configure()``는 쓰지 않는다(그 경로가 실제로
  Acceleration을 다시 254로 write해버려서, 지금 관찰하려는 "있는 그대로의 SRAM 상태"
  자체를 지워버린다). 자세한 근거는
  ``hardware/safety/all_joint_parameter_diagnostic.py`` 모듈 docstring 참고.
- ``--mode armed``: wrist_roll(motor id 5)에 정확히 **한 번의 0.1° 이동**을 실제로
  시도한다. 최소 두 개의 별도 확인 플래그(``--i-have-read-the-safety-plan``,
  ``--confirm-single-write``)와, 사용자가 현재 위치를 직접 확인했다는 근거
  (``--expected-start-raw``/``--expected-start-deg``, 허용 오차 내 일치)가 전부
  있어야만 write를 시도한다. 하나라도 없거나 write 직전 재검사(포트 점유/calibration/
  이음매 회피/raw delta 크기 등)에서 하나라도 실패하면 write 없이 BLOCKED로
  종료한다. 실제 write 로직은 ``hardware/safety/single_joint_writer.py``
  (``SingleJointArmedWriter``/``execute_single_armed_write``) 참고.
- ``--mode acceleration-write``: wrist_roll(motor id 5)의 ``Acceleration`` 레지스터
  **하나만** 현재값(기본 기대 0)에서 254로 정확히 한 번 write한다 - 다른 레지스터/값을
  쓸 수 있는 범용 writer가 아니다(``write_acceleration_once()``는 register 이름/값을
  인자로 받지 않고 전부 하드코딩됨). 최소 두 개의 별도 확인 플래그
  (``--i-understand-this-changes-servo-state``, ``--confirm-acceleration-write``)와
  ``--expected-current-acceleration``(실측과 다르면 BLOCKED)이 전부 있어야만 write를
  시도한다. write 성공 여부와 무관하게 readback까지만 하고 끝난다 - 같은 실행에서
  ``Goal_Position``을 절대 건드리지 않는다(자동 이동 테스트 없음). 실제 write 로직은
  ``hardware/safety/single_joint_parameter_writer.py``
  (``SingleJointParameterWriter``/``execute_single_parameter_write``) 참고.

**Claude Code는 ``--mode armed``와 ``--mode acceleration-write``를 절대 실행하지
않는다** - 코드/정적 검사/fake bus 단위 테스트까지만 수행한다. 아래 실행 예시들은
사람이 나중에 직접 실행할 때를 위한 문서용 예시일 뿐이다. ``--mode inspect-only``/
``--mode dry-run``/``--mode register-diagnostic``/``--mode servo-parameter-diagnostic``/
``--mode all-joint-parameter-diagnostic``은 순수 읽기 전용이라 Claude Code가 실제로
실행해도 된다.

자세한 조사 근거는 ``hardware/safety/single_joint_test_planner.py``,
``hardware/safety/single_joint_writer.py``,
``hardware/safety/single_joint_register_diagnostic.py``,
``hardware/safety/single_joint_servo_parameter_diagnostic.py``,
``hardware/safety/all_joint_parameter_diagnostic.py``,
``hardware/safety/single_joint_parameter_writer.py``,
``hardware/state_server/readonly_so101_reader.py``의 모듈 docstring을 참고한다.

실행 예시(연결 없이 config만 확인):
    python scripts/run_single_joint_hardware_test.py --mode inspect-only

실행 예시(dry-run, positive 방향):
    python scripts/run_single_joint_hardware_test.py --mode dry-run --direction positive \\
        --json-report

실행 예시(register-diagnostic, 이전 armed 실행의 NO_MOTION 원인 조사):
    python scripts/run_single_joint_hardware_test.py \\
        --mode register-diagnostic \\
        --expected-start-raw 2023 \\
        --expected-goal-raw 2024 \\
        --json-report

실행 예시(servo-parameter-diagnostic, 1~2 tick NO_MOTION의 설정상 원인 조사):
    python scripts/run_single_joint_hardware_test.py \\
        --mode servo-parameter-diagnostic \\
        --expected-start-raw 2023 \\
        --expected-goal-raw 2021 \\
        --json-report

실행 예시(all-joint-parameter-diagnostic, wrist_roll Acceleration=0이 개별/전체 문제인지 비교):
    python scripts/run_single_joint_hardware_test.py \\
        --mode all-joint-parameter-diagnostic \\
        --json-report

armed 실행 예시(문서용 - Claude Code는 실행하지 않는다):
    python scripts/run_single_joint_hardware_test.py \\
        --mode armed \\
        --direction positive \\
        --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113538-if00 \\
        --calibration-path ~/.cache/huggingface/lerobot/calibration/robots/so_follower/chanho_follower.json \\
        --max-delta-deg 0.1 \\
        --step-size-deg 0.1 \\
        --expected-start-raw 2023 \\
        --i-have-read-the-safety-plan \\
        --confirm-single-write

acceleration-write 실행 예시(문서용 - Claude Code는 실행하지 않는다):
    python scripts/run_single_joint_hardware_test.py \\
        --mode acceleration-write \\
        --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113538-if00 \\
        --calibration-path ~/.cache/huggingface/lerobot/calibration/robots/so_follower/chanho_follower.json \\
        --expected-current-acceleration 0 \\
        --i-understand-this-changes-servo-state \\
        --confirm-acceleration-write
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.safety import calibration_resolution as calres
from hardware.safety import port_conflict as pc
from hardware.safety.single_joint_bus import WristRollCalibration
from hardware.safety.single_joint_test_planner import (
    DEFAULT_CALIBRATION_MARGIN_DEG,
    DEFAULT_MOTOR_RESOLUTION,
    DEFAULT_STEP_SIZE_DEG,
    DEFAULT_TOTAL_DELTA_DEG,
    EXPECTED_START_RAW_TOLERANCE_TICKS,
    MAX_ALLOWED_STEP_SIZE_DEG,
    MAX_ALLOWED_TOTAL_DELTA_DEG,
    REQUIRED_ARMED_STEP_SIZE_DEG,
    REQUIRED_ARMED_TOTAL_DELTA_DEG,
    TARGET_JOINT,
    PlannerConfigError,
    build_dry_run_plan,
    raw_to_degrees,
)
from hardware.state_server.calibration_loader import CalibrationLoadError, load_calibration_file

# lerobot(so_follower.py)이 wrist_roll에 실제로 배정하는 motor id
# (hardware/state_server/readonly_so101_reader.py._STS3215_MOTOR_IDS와 동일한 값을
# 재확인용으로 여기서도 명시한다). calibration 파일의 id가 이 값과 다르면 즉시 거부한다.
EXPECTED_WRIST_ROLL_MOTOR_ID = 5

TELEOP_SAFE_RANGES_RELATIVE_PATH = Path("configs") / "generated" / "teleop_safe_ranges.json"
REPORTS_DIR = PROJECT_ROOT / "reports" / "hardware_single_joint_dry_run"

MODES = (
    "inspect-only",
    "dry-run",
    "armed",
    "register-diagnostic",
    "servo-parameter-diagnostic",
    "all-joint-parameter-diagnostic",
    "acceleration-write",
)


class RefusalError(RuntimeError):
    """설정/안전 정책 위반으로 실행을 거부해야 할 때 (하드웨어 오류와 구분)."""


# ---------------------------------------------------------------------------
# 보조 정보 로딩 (읽기 전용)
# ---------------------------------------------------------------------------


def _load_teleop_status(project_root: Path) -> dict:
    path = project_root / TELEOP_SAFE_RANGES_RELATIVE_PATH
    if not path.is_file():
        return {
            "available": False,
            "status": None,
            "status_detail": None,
            "note": f"{path}가 없습니다 - teleop 상태를 알 수 없습니다.",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        joint_data = (data.get("joints") or {}).get(TARGET_JOINT) or {}
        return {
            "available": True,
            "status": joint_data.get("status"),
            "status_detail": joint_data.get("status_detail"),
            "note": None,
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "status": None,
            "status_detail": None,
            "note": f"{path} 파싱 실패: {exc}",
        }


# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SO-101 팔로워 wrist_roll 단일 관절 안전 시험 도구 (inspect-only / dry-run만 실행 가능).",
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--direction",
        choices=("positive", "negative"),
        default=None,
        help="dry-run/armed에서 필수. 자동으로 반대 방향으로 전환하지 않는다.",
    )
    parser.add_argument("--port", default=None, help="지정하지 않으면 configs/hardware.local.json의 robot.port를 사용한다.")
    parser.add_argument("--calibration-path", default=None, help="calibration JSON 파일 절대/상대 경로.")
    parser.add_argument(
        "--calibration-id",
        default=None,
        help="LeRobot 표준 캐시 경로 템플릿에 넣을 follower id (--calibration-path보다 우선순위 낮음).",
    )
    parser.add_argument(
        "--max-delta-deg",
        type=float,
        default=DEFAULT_TOTAL_DELTA_DEG,
        help=f"최대 총 이동량(도). {MAX_ALLOWED_TOTAL_DELTA_DEG}° 초과 요청은 거부된다.",
    )
    parser.add_argument(
        "--step-size-deg",
        type=float,
        default=DEFAULT_STEP_SIZE_DEG,
        help=f"스텝 크기(도). {MAX_ALLOWED_STEP_SIZE_DEG}° 초과 요청은 거부된다.",
    )
    parser.add_argument(
        "--margin-deg",
        type=float,
        default=DEFAULT_CALIBRATION_MARGIN_DEG,
        help="calibration 양 끝에서 안쪽으로 확보할 margin(도).",
    )
    parser.add_argument(
        "--start-deg-override",
        type=float,
        default=None,
        help=(
            "시연/오프라인 테스트 전용: 실제 하드웨어에 연결하지 않고 이 시작 각도로 계획만 계산한다. "
            "지정하면 --start-raw-override도 함께 지정해야 한다."
        ),
    )
    parser.add_argument("--start-raw-override", type=int, default=None, help="--start-deg-override와 함께 사용.")
    parser.add_argument("--json-report", action="store_true", help=f"결과를 {REPORTS_DIR}/에 JSON으로 저장한다.")
    parser.add_argument(
        "--i-have-read-the-safety-plan",
        action="store_true",
        help="armed 모드 확인 플래그 1/2. 이것 하나만으로는 write가 실행되지 않는다.",
    )
    parser.add_argument(
        "--confirm-single-write",
        action="store_true",
        help="armed 모드 확인 플래그 2/2. 정확히 한 번의 0.1° write에 동의한다는 의미.",
    )
    parser.add_argument(
        "--expected-start-raw",
        type=int,
        default=None,
        help=(
            "armed 전용: 사용자가 직접 확인한 현재 raw tick. 실측값과의 차이가 "
            f"±{EXPECTED_START_RAW_TOLERANCE_TICKS} tick을 넘어서면 BLOCKED."
        ),
    )
    parser.add_argument(
        "--expected-start-deg",
        type=float,
        default=None,
        help="armed 전용: 사용자가 직접 확인한 현재 각도(도). --expected-start-raw와 함께 또는 대신 사용 가능.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=None,
        help="armed 전용: write 후 readback 전 대기 시간(초). 기본값은 0.3~0.5초 사이 상수(writer 모듈 참고).",
    )
    parser.add_argument(
        "--expected-goal-raw",
        type=int,
        default=None,
        help="register-diagnostic 전용: 이전 armed 실행에서 write했다고 알고 있는 목표 raw tick. Goal_Position latch 여부 비교에 사용.",
    )
    parser.add_argument(
        "--i-understand-this-changes-servo-state",
        action="store_true",
        help="acceleration-write 확인 플래그 1/2. 이것 하나만으로는 write가 실행되지 않는다.",
    )
    parser.add_argument(
        "--confirm-acceleration-write",
        action="store_true",
        help="acceleration-write 확인 플래그 2/2. Acceleration을 254로 바꾸는 것에 동의한다는 의미.",
    )
    parser.add_argument(
        "--expected-current-acceleration",
        type=int,
        default=None,
        help="acceleration-write 필수: 사용자가 직접 확인한 현재 Acceleration 값(기본 0). 실측과 다르면 BLOCKED.",
    )
    return parser


# ---------------------------------------------------------------------------
# 보고서 조립
# ---------------------------------------------------------------------------


@dataclass
class ResolvedConfig:
    port: str
    port_source: str
    calibration_path: Path
    calibration_source: str


def _resolve_config(args: argparse.Namespace, *, allow_default_fallback: bool) -> ResolvedConfig:
    port, port_source = calres.resolve_port(cli_port=args.port, project_root=PROJECT_ROOT)
    calibration_path, calibration_source = calres.resolve_calibration_path(
        cli_calibration_path=args.calibration_path,
        cli_calibration_id=args.calibration_id,
        project_root=PROJECT_ROOT,
        allow_default_fallback=allow_default_fallback,
    )
    return ResolvedConfig(
        port=port, port_source=port_source, calibration_path=calibration_path, calibration_source=calibration_source
    )


def _build_base_report(*, mode: str, config: ResolvedConfig) -> dict:
    teleop = _load_teleop_status(PROJECT_ROOT)
    return {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # 리포트에는 포트의 basename만 남긴다(요구사항: "불필요한 serial 상세정보"를
        # 넣지 않는다 - 전체 절대경로/USB 상세경로는 화면 출력에서만 사람이 확인한다).
        "serial_port_basename": Path(config.port).name,
        "serial_port_source": config.port_source,
        "calibration_path_source": config.calibration_source,
        # 실제 파일시스템 절대 경로는 사용자 홈 디렉터리를 노출할 수 있으므로 리포트에는
        # source(출처)만 남기고, 화면 출력에서만 사람이 직접 확인하도록 분리한다.
        "target_joint": TARGET_JOINT,
        "teleop_status": teleop["status"],
        "teleop_status_detail": teleop["status_detail"],
        "historical_range_applied": False,
        "historical_range_reason": (
            "wrist_roll teleop 상태가 MARGIN_COLLAPSED이거나 확인되지 않아 역사적 텔레옵 범위를 "
            "적용하지 않습니다 (신뢰 가능한 역사 범위가 생성되지 않음). calibration 내부 margin만 사용합니다."
        ),
        "write_count": 0,
    }


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _run_inspect_only(args: argparse.Namespace, config: ResolvedConfig) -> dict:
    report = _build_base_report(mode="inspect-only", config=config)

    _print_section("포트 점유 검사")
    conflict = pc.check_port_conflict(config.port)
    print(f"port_exists={conflict.port_exists} busy={conflict.busy} busy_confirmed={conflict.busy_confirmed}")
    for note in conflict.notes:
        print(f"  - {note}")
    for proc in conflict.holder_processes:
        print(f"  점유 프로세스: pid={proc.pid} command={proc.command} args={proc.args}")
    report["port_conflict"] = conflict.to_dict()

    _print_section("calibration 로딩")
    try:
        entries = load_calibration_file(config.calibration_path)
    except CalibrationLoadError as exc:
        print(f"calibration 로드 실패: {exc}")
        report["calibration_loaded"] = False
        report["calibration_error"] = str(exc)
        report["connected"] = False
        report["write_count"] = 0
        return report

    entry = entries[TARGET_JOINT]
    print(f"calibration 로드 성공 (motor_id={entry.id}, homing_offset={entry.homing_offset}, "
          f"range=[{entry.range_min}, {entry.range_max}])")
    report["calibration_loaded"] = True
    report["motor_id"] = entry.id
    report["drive_mode"] = entry.drive_mode
    report["homing_offset"] = entry.homing_offset
    report["calibration_raw_min"] = entry.range_min
    report["calibration_raw_max"] = entry.range_max
    report["is_full_turn"] = entry.range_min == 0 and entry.range_max == DEFAULT_MOTOR_RESOLUTION - 1
    report["unit"] = "degree"

    if entry.id != EXPECTED_WRIST_ROLL_MOTOR_ID:
        print(
            f"경고: calibration의 wrist_roll motor id({entry.id})가 예상값"
            f"({EXPECTED_WRIST_ROLL_MOTOR_ID})과 다릅니다 - 연결하지 않습니다."
        )
        report["connected"] = False
        report["motor_id_mismatch"] = True
        report["connect_skipped_reason"] = "motor_id_mismatch"
        return report
    report["motor_id_mismatch"] = False

    _print_section("wrist_roll 연결 및 읽기")
    if conflict.busy:
        print("포트가 점유 중(또는 판정 불가)이라 연결을 시도하지 않습니다. config 검사까지만 수행했습니다.")
        report["connected"] = False
        report["connect_skipped_reason"] = "port_busy_or_unknown"
        return report

    from hardware.safety.single_joint_hardware_inspector import InspectorError, SingleJointInspector, WristRollCalibration

    calibration = WristRollCalibration(
        motor_id=entry.id,
        drive_mode=entry.drive_mode,
        homing_offset=entry.homing_offset,
        range_min=entry.range_min,
        range_max=entry.range_max,
    )
    inspector = SingleJointInspector(port=config.port, calibration=calibration)
    try:
        inspector.connect()
        raw = inspector.read_raw()
        deg = inspector.read_degrees()
        print(f"연결 성공. raw={raw} normalized_deg={deg:.4f}")
        report["connected"] = True
        report["raw_position"] = raw
        report["normalized_position_deg"] = deg
    except (ConnectionError, RuntimeError, InspectorError) as exc:
        print(f"연결/읽기 실패: {exc}")
        report["connected"] = False
        report["connect_error"] = str(exc)
    finally:
        try:
            inspector.disconnect()
        except Exception:  # noqa: BLE001 - 종료 중 오류로 인스펙트 결과 자체를 실패시키지 않는다
            pass

    return report


def _run_dry_run(args: argparse.Namespace, config: ResolvedConfig) -> dict:
    if args.direction is None:
        raise RefusalError("--mode dry-run에는 --direction(positive|negative)이 필수입니다.")

    if args.step_size_deg > MAX_ALLOWED_STEP_SIZE_DEG or args.step_size_deg <= 0:
        raise RefusalError(
            f"--step-size-deg는 0보다 크고 {MAX_ALLOWED_STEP_SIZE_DEG}° 이하여야 합니다: {args.step_size_deg}"
        )
    if args.max_delta_deg > MAX_ALLOWED_TOTAL_DELTA_DEG or args.max_delta_deg <= 0:
        raise RefusalError(
            f"--max-delta-deg는 0보다 크고 {MAX_ALLOWED_TOTAL_DELTA_DEG}° 이하여야 합니다: {args.max_delta_deg}"
        )

    inspect_report = _run_inspect_only(args, config)
    report = _build_base_report(mode="dry-run", config=config)
    report.update(
        {k: v for k, v in inspect_report.items() if k not in ("mode",)}
    )  # inspect-only 항목 전체를 dry-run 리포트에 포함

    _print_section("dry-run 이동 계획")

    if args.start_deg_override is not None or args.start_raw_override is not None:
        if args.start_deg_override is None or args.start_raw_override is None:
            raise RefusalError("--start-deg-override와 --start-raw-override는 함께 지정해야 합니다.")
        start_deg = args.start_deg_override
        start_raw = args.start_raw_override
        report["start_source"] = "cli_override_no_hardware_connection"
        print(f"(시연/오프라인 모드) 하드웨어 연결 없이 override된 시작값 사용: raw={start_raw} deg={start_deg}")
    elif report.get("connected"):
        start_deg = report["normalized_position_deg"]
        start_raw = report["raw_position"]
        report["start_source"] = "live_hardware_read"
    else:
        reason = report.get("connect_error") or report.get("connect_skipped_reason") or report.get("calibration_error") or "알 수 없음"
        print(f"시작 위치를 읽을 수 없어 dry-run 계획을 계산할 수 없습니다 (사유: {reason}).")
        report["final_verdict"] = "BLOCKED"
        report["block_reasons"] = [f"시작 위치를 확보하지 못했습니다: {reason}"]
        report["planned_targets"] = []
        return report

    try:
        plan = build_dry_run_plan(
            start_deg=start_deg,
            start_raw=start_raw,
            direction=args.direction,
            range_min=report["calibration_raw_min"],
            range_max=report["calibration_raw_max"],
            motor_resolution=DEFAULT_MOTOR_RESOLUTION,
            margin_deg=args.margin_deg,
            requested_total_delta_deg=args.max_delta_deg,
            step_size_deg=args.step_size_deg,
        )
    except PlannerConfigError as exc:
        raise RefusalError(str(exc)) from exc

    plan_dict = plan.to_dict()
    report.update(plan_dict)
    report["direction"] = args.direction

    print(f"direction={plan.direction} start_deg={plan.start_deg:.4f} start_raw={plan.start_raw}")
    print(
        f"calibration inner-safe range(deg)=[{plan.calibration_range.inner_deg_min:.3f}, "
        f"{plan.calibration_range.inner_deg_max:.3f}] (margin={plan.calibration_range.margin_deg}°, "
        f"is_full_turn={plan.calibration_range.is_full_turn})"
    )
    print(f"teleop_status={report['teleop_status']} historical_range_applied={report['historical_range_applied']}")
    for step in plan.steps:
        print(
            f"  step {step.index:>2}: target_deg={step.target_deg:+.2f} target_raw={step.target_raw} "
            f"calibration={step.calibration_check:8s} local_1deg={step.local_range_check:8s} verdict={step.verdict}"
        )
    print(f"final_verdict={plan.final_verdict}")
    for reason in plan.block_reasons:
        print(f"  block reason: {reason}")

    return report


def _run_register_diagnostic(args: argparse.Namespace, config: ResolvedConfig) -> dict:
    """wrist_roll(motor id 5) 레지스터를 읽기 전용으로 스냅샷하고 원인을 진단한다.

    허용 흐름(요구사항 3번): 포트 존재/점유 확인 -> calibration 읽기 -> wrist_roll
    하나만 아는 ``FeetechMotorsBus`` 생성 -> ``connect()`` -> 레지스터 read ->
    ``disconnect(disable_torque=False)``. ``SOFollower.connect()``는 쓰지 않는다(그
    경로는 configure write를 발생시킨다 - 모듈 조사 근거는
    ``hardware/state_server/readonly_so101_reader.py`` docstring 참고). 이 함수는
    어떤 write도 호출하지 않는다.
    """
    from hardware.safety.single_joint_register_diagnostic import (
        HARDWARE_ERROR_STATUS_NOT_AVAILABLE_NOTE,
        MOVING_STATUS_NOT_AVAILABLE_NOTE,
        DiagnosticVerdict,
        RegisterDiagnosticInspector,
        classify_diagnostic,
    )

    report = _build_base_report(mode="register-diagnostic", config=config)
    report["moving_status_note"] = MOVING_STATUS_NOT_AVAILABLE_NOTE
    report["hardware_error_status_note"] = HARDWARE_ERROR_STATUS_NOT_AVAILABLE_NOTE
    print(f"mode={report['mode']} target_joint={TARGET_JOINT}")

    _print_section("포트 점유 검사")
    conflict = pc.check_port_conflict(config.port)
    print(f"port_exists={conflict.port_exists} port_busy={conflict.busy} busy_confirmed={conflict.busy_confirmed}")
    for note in conflict.notes:
        print(f"  - {note}")
    for proc in conflict.holder_processes:
        print(f"  점유 프로세스: pid={proc.pid} command={proc.command} args={proc.args}")
    report["port_conflict"] = conflict.to_dict()

    _print_section("calibration 로딩")
    try:
        entries = load_calibration_file(config.calibration_path)
    except CalibrationLoadError as exc:
        print(f"calibration 로드 실패: {exc}")
        report["calibration_loaded"] = False
        report["calibration_error"] = str(exc)
        report["connected"] = False
        report["diagnostic_verdict"] = DiagnosticVerdict.UNKNOWN
        report["write_count"] = 0
        return report

    entry = entries[TARGET_JOINT]
    print(f"calibration 로드 성공 (motor_id={entry.id}, range=[{entry.range_min}, {entry.range_max}])")
    report["calibration_loaded"] = True
    report["motor_id"] = entry.id

    if entry.id != EXPECTED_WRIST_ROLL_MOTOR_ID:
        print(
            f"경고: calibration의 wrist_roll motor id({entry.id})가 예상값"
            f"({EXPECTED_WRIST_ROLL_MOTOR_ID})과 다릅니다 - 연결하지 않습니다."
        )
        report["connected"] = False
        report["diagnostic_verdict"] = DiagnosticVerdict.UNKNOWN
        report["write_count"] = 0
        return report

    calibration = WristRollCalibration(
        motor_id=entry.id,
        drive_mode=entry.drive_mode,
        homing_offset=entry.homing_offset,
        range_min=entry.range_min,
        range_max=entry.range_max,
    )

    _print_section("wrist_roll 레지스터 read-only 진단")
    if conflict.busy:
        print("포트가 점유 중(또는 판정 불가)이라 연결을 시도하지 않습니다. config 검사까지만 수행했습니다.")
        report["connected"] = False
        report["diagnostic_verdict"] = DiagnosticVerdict.UNKNOWN
        report["write_count"] = 0
        return report

    inspector = RegisterDiagnosticInspector(port=config.port, calibration=calibration)
    snapshot = None
    try:
        inspector.connect()
        snapshot = inspector.read_snapshot()
        report["connected"] = True
    except (ConnectionError, RuntimeError) as exc:
        print(f"연결/읽기 실패: {exc}")
        report["connected"] = False
        report["connect_error"] = str(exc)
    finally:
        try:
            inspector.disconnect()
        except Exception:  # noqa: BLE001 - 종료 중 오류로 진단 결과 자체를 잃지 않는다
            pass

    if snapshot is None:
        report["diagnostic_verdict"] = DiagnosticVerdict.UNKNOWN
        report["write_count"] = 0
        return report

    report["registers"] = snapshot.to_dict()
    report["expected_start_raw"] = args.expected_start_raw
    report["expected_goal_raw"] = args.expected_goal_raw

    goal_latched = None
    if args.expected_goal_raw is not None and snapshot.goal_position_raw is not None:
        goal_latched = snapshot.goal_position_raw == args.expected_goal_raw
    report["goal_latched"] = goal_latched

    goal_present_delta = None
    if snapshot.goal_position_raw is not None and snapshot.present_position_raw is not None:
        goal_present_delta = snapshot.goal_position_raw - snapshot.present_position_raw
    report["goal_present_delta"] = goal_present_delta

    verdict, reasons = classify_diagnostic(
        snapshot=snapshot,
        expected_start_raw=args.expected_start_raw,
        expected_goal_raw=args.expected_goal_raw,
    )
    report["diagnostic_verdict"] = verdict
    report["diagnostic_reasons"] = list(reasons)
    report["write_count"] = 0

    print(
        f"Torque_Enable={snapshot.torque_enable} Goal_Position={snapshot.goal_position_raw} "
        f"Present_Position={snapshot.present_position_raw} Moving={snapshot.moving}"
    )
    print("Moving_Status=NOT_AVAILABLE_IN_INSTALLED_TABLE (STS3215 control table에 없음 - 'Moving'만 존재)")
    print(
        f"optional registers: Present_Load={snapshot.present_load} Present_Current={snapshot.present_current} "
        f"Present_Velocity={snapshot.present_velocity} Present_Voltage={snapshot.present_voltage} "
        f"Present_Temperature={snapshot.present_temperature} Status={snapshot.status_raw}"
    )
    if snapshot.read_errors:
        print(f"read_errors={snapshot.read_errors}")
    print(f"expected_start_raw={args.expected_start_raw} expected_goal_raw={args.expected_goal_raw}")
    print(f"goal_latched={goal_latched} goal_present_delta={goal_present_delta}")
    print(f"diagnostic_verdict={verdict}")
    for reason in reasons:
        print(f"  reason: {reason}")
    print("write_count=0")

    return report


def _run_servo_parameter_diagnostic(args: argparse.Namespace, config: ResolvedConfig) -> dict:
    """wrist_roll(motor id 5) STS3215 설정 레지스터를 read-only로 진단한다.

    허용 흐름(요구사항 2번): 포트 존재/점유 확인 -> calibration 읽기 -> wrist_roll
    하나만 아는 ``FeetechMotorsBus`` 생성 -> ``connect()`` -> 레지스터 read ->
    ``disconnect(disable_torque=False)``. ``SOFollower.connect()``는 쓰지 않는다.
    이 함수는 어떤 write도 호출하지 않는다.
    """
    from hardware.safety.single_joint_servo_parameter_diagnostic import (
        ServoParameterDiagnosticInspector,
        classify_servo_parameters,
        compute_next_step_candidates,
    )

    report = _build_base_report(mode="servo-parameter-diagnostic", config=config)
    print(f"mode={report['mode']} target_joint={TARGET_JOINT}")

    _print_section("포트 점유 검사")
    conflict = pc.check_port_conflict(config.port)
    print(f"port_exists={conflict.port_exists} port_busy={conflict.busy} busy_confirmed={conflict.busy_confirmed}")
    for note in conflict.notes:
        print(f"  - {note}")
    report["port_conflict"] = conflict.to_dict()

    _print_section("calibration 로딩")
    try:
        entries = load_calibration_file(config.calibration_path)
    except CalibrationLoadError as exc:
        print(f"calibration 로드 실패: {exc}")
        report["calibration_loaded"] = False
        report["calibration_error"] = str(exc)
        report["connected"] = False
        report["write_count"] = 0
        return report

    entry = entries[TARGET_JOINT]
    print(f"calibration 로드 성공 (motor_id={entry.id}, range=[{entry.range_min}, {entry.range_max}])")
    report["calibration_loaded"] = True
    report["motor_id"] = entry.id

    if entry.id != EXPECTED_WRIST_ROLL_MOTOR_ID:
        print(
            f"경고: calibration의 wrist_roll motor id({entry.id})가 예상값"
            f"({EXPECTED_WRIST_ROLL_MOTOR_ID})과 다릅니다 - 연결하지 않습니다."
        )
        report["connected"] = False
        report["write_count"] = 0
        return report

    calibration = WristRollCalibration(
        motor_id=entry.id,
        drive_mode=entry.drive_mode,
        homing_offset=entry.homing_offset,
        range_min=entry.range_min,
        range_max=entry.range_max,
    )

    _print_section("wrist_roll 설정 레지스터 read-only 진단")
    if conflict.busy:
        print("포트가 점유 중(또는 판정 불가)이라 연결을 시도하지 않습니다. config 검사까지만 수행했습니다.")
        report["connected"] = False
        report["write_count"] = 0
        return report

    inspector = ServoParameterDiagnosticInspector(port=config.port, calibration=calibration)
    snapshot = None
    try:
        inspector.connect()
        snapshot = inspector.read_snapshot()
        report["connected"] = True
    except (ConnectionError, RuntimeError) as exc:
        print(f"연결/읽기 실패: {exc}")
        report["connected"] = False
        report["connect_error"] = str(exc)
    finally:
        try:
            inspector.disconnect()
        except Exception:  # noqa: BLE001 - 종료 중 오류로 진단 결과 자체를 잃지 않는다
            pass

    report["write_count"] = 0
    if snapshot is None:
        return report

    report["registers"] = snapshot.to_dict()

    print(
        f"Torque_Enable={snapshot.torque_enable} Goal_Position={snapshot.goal_position_raw} "
        f"Present_Position={snapshot.present_position_raw} Moving={snapshot.moving} Status={snapshot.status_raw}"
    )
    print(
        f"CW_Dead_Zone={snapshot.cw_dead_zone} CCW_Dead_Zone={snapshot.ccw_dead_zone} "
        f"Minimum_Startup_Force={snapshot.minimum_startup_force} Operating_Mode={snapshot.operating_mode}"
    )
    print(
        f"Acceleration={snapshot.acceleration} Maximum_Acceleration={snapshot.maximum_acceleration} "
        f"Goal_Velocity={snapshot.goal_velocity} Maximum_Velocity_Limit={snapshot.maximum_velocity_limit} "
        f"Moving_Velocity_Threshold={snapshot.moving_velocity_threshold}"
    )
    print(f"Torque_Limit={snapshot.torque_limit} Max_Torque_Limit={snapshot.max_torque_limit}")
    print(
        f"P_Coefficient={snapshot.p_coefficient} I_Coefficient={snapshot.i_coefficient} "
        f"D_Coefficient={snapshot.d_coefficient} Lock={snapshot.lock} Angular_Resolution={snapshot.angular_resolution}"
    )
    if snapshot.unavailable_registers:
        print(f"unavailable_registers(설치된 control table에 없음)={list(snapshot.unavailable_registers)}")
    if snapshot.read_errors:
        print(f"read_errors={snapshot.read_errors}")

    # expected 값과 비교 (요구사항 3번 - 이전 negative armed 실행 때문에 Goal_Position이
    # 2021일 수도 있다는 가능성을 추측 없이 실측으로 확인한다).
    report["expected_start_raw"] = args.expected_start_raw
    report["expected_goal_raw"] = args.expected_goal_raw
    if args.expected_goal_raw is not None and snapshot.goal_position_raw is not None:
        report["goal_latched"] = snapshot.goal_position_raw == args.expected_goal_raw
    if snapshot.goal_position_raw is not None and snapshot.present_position_raw is not None:
        report["goal_present_delta"] = snapshot.goal_position_raw - snapshot.present_position_raw
    print(f"expected_start_raw={args.expected_start_raw} expected_goal_raw={args.expected_goal_raw} "
          f"goal_latched={report.get('goal_latched')} goal_present_delta={report.get('goal_present_delta')}")

    _print_section("원인 판정")
    verdicts, reasons = classify_servo_parameters(snapshot)
    report["verdicts"] = list(verdicts)
    report["verdict_reasons"] = list(reasons)
    print(f"verdicts={list(verdicts)}")
    for reason in reasons:
        print(f"  reason: {reason}")

    _print_section("다음 단발 후보 (계산만, write 없음)")
    if snapshot.present_position_raw is not None:
        start_raw = snapshot.present_position_raw
        start_deg = raw_to_degrees(start_raw, range_min=entry.range_min, range_max=entry.range_max)
        candidates = compute_next_step_candidates(
            start_deg=start_deg,
            start_raw=start_raw,
            range_min=entry.range_min,
            range_max=entry.range_max,
            margin_deg=args.margin_deg,
        )
        report["next_step_candidates"] = [c.to_dict() for c in candidates]
        for c in candidates:
            print(
                f"  candidate direction={c.direction} tick={c.tick_count} requested_delta_deg={c.requested_delta_deg:.4f} "
                f"expected_raw_delta={c.expected_raw_delta} expected_target_raw={c.expected_target_raw} "
                f"safe_candidate={c.within_calibration_inner_range and c.within_physical_range and c.under_max_degree}"
            )
    else:
        print("Present_Position을 읽지 못해 다음 후보를 계산할 수 없습니다.")
        report["next_step_candidates"] = []

    print("write_count=0")
    return report


_ALL_JOINT_TABLE_HEADER = (
    f"{'joint':<14}{'id':>4}{'Accel':>8}{'MaxAccel':>10}{'Torque':>8}{'Mode':>6}"
    f"{'Goal':>7}{'Present':>9}{'Moving':>8}{'Status':>8}"
)


def _run_all_joint_parameter_diagnostic(args: argparse.Namespace, config: ResolvedConfig) -> dict:
    """follower 6개 관절 전체의 Acceleration 등 설정 레지스터를 read-only로 비교한다.

    허용 흐름(요구사항 3번): 포트 존재/점유 확인 -> follower calibration(6개 관절)
    읽기 -> 6개 모터를 등록한 ``FeetechMotorsBus`` 생성 -> ``connect()`` -> 레지스터
    read -> ``disconnect(disable_torque=False)``. ``SOFollower.connect()``/
    ``configure()``는 쓰지 않는다. 이 함수는 어떤 write도 호출하지 않는다.
    """
    from hardware.safety.all_joint_parameter_diagnostic import (
        JOINT_NAMES,
        AllJointParameterDiagnosticInspector,
        classify_acceleration_state,
        find_stuck_joints,
    )

    report = _build_base_report(mode="all-joint-parameter-diagnostic", config=config)
    report["target_joint"] = "all"  # 이 모드는 6개 관절 전부를 다룬다 - base report의 wrist_roll 전용 필드를 덮어쓴다
    print(f"mode={report['mode']} target_joints={list(JOINT_NAMES)}")

    _print_section("포트 점유 검사")
    conflict = pc.check_port_conflict(config.port)
    print(f"port_exists={conflict.port_exists} port_busy={conflict.busy} busy_confirmed={conflict.busy_confirmed}")
    for note in conflict.notes:
        print(f"  - {note}")
    report["port_conflict"] = conflict.to_dict()

    _print_section("follower calibration 로딩 (6개 관절)")
    try:
        entries = load_calibration_file(config.calibration_path)
    except CalibrationLoadError as exc:
        print(f"calibration 로드 실패: {exc}")
        report["calibration_loaded"] = False
        report["calibration_error"] = str(exc)
        report["connected"] = False
        report["write_count"] = 0
        return report

    print(f"calibration 로드 성공: { {joint: entries[joint].id for joint in JOINT_NAMES} }")
    report["calibration_loaded"] = True
    report["motor_ids"] = {joint: entries[joint].id for joint in JOINT_NAMES}

    _print_section("6개 관절 레지스터 read-only 진단")
    if conflict.busy:
        print("포트가 점유 중(또는 판정 불가)이라 연결을 시도하지 않습니다. config 검사까지만 수행했습니다.")
        report["connected"] = False
        report["write_count"] = 0
        return report

    inspector = AllJointParameterDiagnosticInspector(port=config.port, calibration=entries)
    snapshots = None
    try:
        inspector.connect()
        snapshots = inspector.read_all_snapshots()
        report["connected"] = True
    except (ConnectionError, RuntimeError) as exc:
        print(f"연결/읽기 실패: {exc}")
        report["connected"] = False
        report["connect_error"] = str(exc)
    finally:
        try:
            inspector.disconnect()
        except Exception:  # noqa: BLE001 - 종료 중 오류로 진단 결과 자체를 잃지 않는다
            pass

    report["write_count"] = 0
    if snapshots is None:
        return report

    report["joints"] = {joint: snap.to_dict() for joint, snap in snapshots.items()}

    print(_ALL_JOINT_TABLE_HEADER)
    for joint in JOINT_NAMES:
        s = snapshots[joint]
        fmt = lambda v: "?" if v is None else str(v)  # noqa: E731
        print(
            f"{joint:<14}{fmt(s.motor_id):>4}{fmt(s.acceleration):>8}{fmt(s.maximum_acceleration):>10}"
            f"{fmt(s.torque_enable):>8}{fmt(s.operating_mode):>6}{fmt(s.goal_position_raw):>7}"
            f"{fmt(s.present_position_raw):>9}{fmt(s.moving):>8}{fmt(s.status_raw):>8}"
        )

    verdict, reasons = classify_acceleration_state(snapshots)
    report["acceleration_verdict"] = verdict
    report["acceleration_verdict_reasons"] = list(reasons)

    stuck_joints = find_stuck_joints(snapshots)
    report["stuck_joints"] = list(stuck_joints)  # Torque_Enable=1, Goal!=Present, Moving=0인 관절들

    _print_section("판정")
    print(f"acceleration_verdict={verdict}")
    for reason in reasons:
        print(f"  reason: {reason}")
    print(f"stuck_joints(Torque_Enable=1, Goal!=Present, Moving=0)={list(stuck_joints)}")
    print("write_count=0")

    return report


def _run_acceleration_write(args: argparse.Namespace, config: ResolvedConfig) -> int:
    """acceleration-write 모드: wrist_roll(motor id 5)의 ``Acceleration``만 정확히 한 번 254로 write한다.

    **Claude Code는 이 함수를 호출하지 않는다** (구현/테스트 중 어떤 경로로도 도달하지
    않아야 한다 - 사람이 CLI에서 ``--mode acceleration-write``를 직접 실행할 때만
    도달한다). 이 함수가 호출하는 하위 함수들(``execute_single_parameter_write`` 등)은
    fake bus로 별도 단위 테스트되어 있다 - 이 함수 자체의 실행 경로만 테스트되지 않는다.

    요구사항 9번: 이 함수는 write + readback까지만 하고 끝난다 - 같은 실행에서
    Goal_Position을 절대 건드리지 않는다(자동 이동 테스트 없음).
    """
    _print_section("acceleration-write 모드 (wrist_roll Acceleration 0→254 단발 write)")

    # -- CLI 레벨 검증: 여기서 막히면 하드웨어에 전혀 접근하지 않는다 -----------------
    confirmation_flags = {
        "i_understand_this_changes_servo_state": bool(args.i_understand_this_changes_servo_state),
        "confirm_acceleration_write": bool(args.confirm_acceleration_write),
    }
    missing_flags = [name for name, value in confirmation_flags.items() if not value]
    if missing_flags:
        print(f"거부: 확인 플래그가 부족합니다: {missing_flags} (둘 다 있어야 진입 가능).")
        return 2

    if args.expected_current_acceleration is None:
        print("거부: --expected-current-acceleration을 지정해야 합니다 (기본 기대값은 0).")
        return 2

    # -- write 직전 재검사: 포트 점유 + calibration 존재를 여기서 "다시" 확인한다 -----
    _print_section("write 직전 재검사: 포트 점유")
    conflict = pc.check_port_conflict(config.port)
    print(f"port_exists={conflict.port_exists} busy={conflict.busy} busy_confirmed={conflict.busy_confirmed}")
    if conflict.busy:
        print("거부: 포트가 점유 중(또는 판정 불가)이라 write를 시도하지 않습니다.")
        return 2

    _print_section("write 직전 재검사: calibration")
    try:
        entries = load_calibration_file(config.calibration_path)
    except CalibrationLoadError as exc:
        print(f"거부: calibration 로드 실패: {exc}")
        return 2

    entry = entries[TARGET_JOINT]
    if entry.id != EXPECTED_WRIST_ROLL_MOTOR_ID:
        print(
            f"거부: calibration의 wrist_roll motor id({entry.id})가 예상값"
            f"({EXPECTED_WRIST_ROLL_MOTOR_ID})과 다릅니다."
        )
        return 2
    print(f"calibration 확인됨 (motor_id={entry.id}).")

    calibration = WristRollCalibration(
        motor_id=entry.id,
        drive_mode=entry.drive_mode,
        homing_offset=entry.homing_offset,
        range_min=entry.range_min,
        range_max=entry.range_max,
    )

    from hardware.safety.single_joint_parameter_writer import (
        SingleJointParameterWriter,
        execute_single_parameter_write,
    )

    _print_section("연결 + 단발 Acceleration write 실행")
    writer = SingleJointParameterWriter(port=config.port, calibration=calibration)
    result = None
    try:
        writer.connect()
        result = execute_single_parameter_write(
            writer=writer,
            expected_current_acceleration=args.expected_current_acceleration,
            confirmation_flags=confirmation_flags,
        )
    except (ConnectionError, RuntimeError) as exc:
        print(f"연결 실패: {exc}")
        return 1
    finally:
        try:
            writer.disconnect()
        except Exception:  # noqa: BLE001 - 종료 중 오류로 결과 자체를 잃지 않는다
            pass

    report = _build_base_report(mode="acceleration-write", config=config)
    report.update(result.to_dict())
    report["write_count"] = result.write_count_after

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nfinal_verdict={result.final_verdict} write_count={result.write_count_after}")
    print("Acceleration write + readback까지만 수행했습니다 - 이 실행에서는 Goal_Position을 전혀 건드리지 않았습니다.")

    if args.json_report:
        path = _save_json_report(report, mode=args.mode)
        print(f"\nJSON 리포트 저장: {path}")

    return 0 if result.final_verdict == "PASS" else 1


def _run_armed(args: argparse.Namespace, config: ResolvedConfig) -> int:
    """armed 모드: wrist_roll(motor id 5)에 정확히 한 번의 0.1° write를 시도한다.

    **Claude Code는 이 함수를 호출하지 않는다** (구현/테스트 중 어떤 경로로도 도달하지
    않아야 한다 - 사람이 CLI에서 ``--mode armed``를 직접 실행할 때만 도달한다). 이
    함수가 호출하는 하위 함수들(``execute_single_armed_write`` 등)은 fake bus로 별도
    단위 테스트되어 있다 - 이 함수 자체의 실행 경로만 테스트되지 않는다.
    """
    _print_section("armed 모드 (wrist_roll 0.1° 단발 이동)")

    # -- CLI 레벨 검증: 여기서 막히면 하드웨어에 전혀 접근하지 않는다 -----------------
    if args.start_deg_override is not None or args.start_raw_override is not None:
        print("거부: armed 모드에서는 --start-deg-override/--start-raw-override를 사용할 수 없습니다.")
        return 2

    if args.direction is None:
        print("거부: --mode armed에는 --direction(positive|negative)이 필수입니다. 자동 방향 선택은 없습니다.")
        return 2

    if (
        abs(args.max_delta_deg - REQUIRED_ARMED_TOTAL_DELTA_DEG) > 1e-9
        or abs(args.step_size_deg - REQUIRED_ARMED_STEP_SIZE_DEG) > 1e-9
    ):
        print(
            f"거부: 첫 armed 실행은 --max-delta-deg와 --step-size-deg가 정확히 "
            f"{REQUIRED_ARMED_TOTAL_DELTA_DEG}이어야 합니다 "
            f"(받은 값: max_delta_deg={args.max_delta_deg}, step_size_deg={args.step_size_deg})."
        )
        return 2

    confirmation_flags = {
        "i_have_read_the_safety_plan": bool(args.i_have_read_the_safety_plan),
        "confirm_single_write": bool(args.confirm_single_write),
    }
    missing_flags = [name for name, value in confirmation_flags.items() if not value]
    if missing_flags:
        print(f"거부: 확인 플래그가 부족합니다: {missing_flags} (둘 다 있어야 진입 가능).")
        return 2

    if args.expected_start_raw is None and args.expected_start_deg is None:
        print("거부: --expected-start-raw 또는 --expected-start-deg 중 최소 하나를 지정해야 합니다.")
        return 2

    # -- write 직전 재검사: 포트 점유 + calibration 존재를 여기서 "다시" 확인한다 -----
    # (inspect-only 단계에서 이미 확인했더라도, armed는 그 결과를 재사용하지 않는다 -
    # 그 사이 다른 프로세스가 포트를 잡았거나 calibration 파일이 바뀌었을 수 있다.)
    _print_section("write 직전 재검사: 포트 점유")
    conflict = pc.check_port_conflict(config.port)
    print(f"port_exists={conflict.port_exists} busy={conflict.busy} busy_confirmed={conflict.busy_confirmed}")
    for note in conflict.notes:
        print(f"  - {note}")
    if conflict.busy:
        print("거부: 포트가 점유 중(또는 판정 불가)이라 write를 시도하지 않습니다.")
        return 2

    _print_section("write 직전 재검사: calibration")
    try:
        entries = load_calibration_file(config.calibration_path)
    except CalibrationLoadError as exc:
        print(f"거부: calibration 로드 실패: {exc}")
        return 2

    entry = entries[TARGET_JOINT]
    if entry.id != EXPECTED_WRIST_ROLL_MOTOR_ID:
        print(
            f"거부: calibration의 wrist_roll motor id({entry.id})가 예상값"
            f"({EXPECTED_WRIST_ROLL_MOTOR_ID})과 다릅니다."
        )
        return 2
    print(f"calibration 확인됨 (motor_id={entry.id}, range=[{entry.range_min}, {entry.range_max}]).")

    calibration = WristRollCalibration(
        motor_id=entry.id,
        drive_mode=entry.drive_mode,
        homing_offset=entry.homing_offset,
        range_min=entry.range_min,
        range_max=entry.range_max,
    )

    from hardware.safety.single_joint_writer import (
        DEFAULT_POST_WRITE_WAIT_S,
        SingleJointArmedWriter,
        execute_single_armed_write,
    )

    wait_seconds = args.wait_seconds if args.wait_seconds is not None else DEFAULT_POST_WRITE_WAIT_S

    _print_section("연결 + 단발 write 실행")
    writer = SingleJointArmedWriter(port=config.port, calibration=calibration)
    result = None
    try:
        writer.connect()
        result = execute_single_armed_write(
            writer=writer,
            direction=args.direction,
            calibration=calibration,
            expected_start_raw=args.expected_start_raw,
            expected_start_deg=args.expected_start_deg,
            confirmation_flags=confirmation_flags,
            margin_deg=args.margin_deg,
            post_write_wait_s=wait_seconds,
        )
    except (ConnectionError, RuntimeError) as exc:
        print(f"연결 실패: {exc}")
        return 1
    finally:
        try:
            writer.disconnect()
        except Exception:  # noqa: BLE001 - 종료 중 오류로 결과 자체를 잃지 않는다
            pass

    report = _build_base_report(mode="armed", config=config)
    report.update(result.to_dict())
    report["write_count"] = result.write_count_after

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nfinal_verdict={result.final_verdict} write_count={result.write_count_after}")

    if args.json_report:
        path = _save_json_report(report, mode=args.mode)
        print(f"\nJSON 리포트 저장: {path}")

    return 0 if result.final_verdict == "PASS" else 1


# ---------------------------------------------------------------------------
# 리포트 저장 + 진입점
# ---------------------------------------------------------------------------


def _save_json_report(report: dict, *, mode: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"single_joint_{mode.replace('-', '_')}_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "armed":
        try:
            config = _resolve_config(args, allow_default_fallback=False)
        except calres.ResolutionError as exc:
            print(f"거부: {exc}")
            return 2
        return _run_armed(args, config)

    if args.mode == "acceleration-write":
        try:
            config = _resolve_config(args, allow_default_fallback=False)
        except calres.ResolutionError as exc:
            print(f"거부: {exc}")
            return 2
        return _run_acceleration_write(args, config)

    try:
        config = _resolve_config(args, allow_default_fallback=True)
    except calres.ResolutionError as exc:
        print(f"거부: {exc}")
        return 2

    print(f"mode={args.mode} port={config.port} (source={config.port_source}) "
          f"calibration_source={config.calibration_source}")

    try:
        if args.mode == "inspect-only":
            report = _run_inspect_only(args, config)
        elif args.mode == "register-diagnostic":
            report = _run_register_diagnostic(args, config)
        elif args.mode == "servo-parameter-diagnostic":
            report = _run_servo_parameter_diagnostic(args, config)
        elif args.mode == "all-joint-parameter-diagnostic":
            report = _run_all_joint_parameter_diagnostic(args, config)
        else:  # dry-run
            report = _run_dry_run(args, config)
    except RefusalError as exc:
        print(f"거부: {exc}")
        return 2

    if args.json_report:
        path = _save_json_report(report, mode=args.mode)
        print(f"\nJSON 리포트 저장: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
