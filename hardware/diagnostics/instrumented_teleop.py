"""Instrumented Teleop Diagnostic - 정상 LeRobot SO101Leader/SO101Follower teleoperation을
그대로 사용하면서 wrist_roll 중심으로 leader command -> follower Goal/Present 상태를
**passive하게(개입 없이)** 계측한다.

## 핵심 원칙 (이번 개정의 방향)

**이 모듈은 기존 LeRobot teleop의 control path에 절대 개입하지 않는다.** leader가 만든
command는 어떤 경우에도 수정/clamp/차단되지 않고 그대로 ``follower.send_action()``에
전달된다 - 이것이 ``tests/test_instrumented_teleop.py``의
``test_instrumented_command_sequence_matches_baseline_exactly``가 검증하는 핵심 계약이다.

과거 버전(이전 세션)에는 ``SAFETY_STOP_COMMAND_DELTA``/``SAFETY_STOP_DIRECTION_MISMATCH``/
``SAFETY_STOP_POSITION_JUMP``/``SAFETY_STOP_STATUS_NONZERO``라는 이름으로 (1) command가
follower 시작 위치 기준 ±2°를 넘으면 ``send_action()`` 자체를 호출하지 않고, (2) 방향 불일치/
급격한 position jump/``Status``≠0이 감지되면 루프를 강제 종료하는 **능동 개입** 로직이 있었다.
이번 개정에서 그 네 가지를 전부 **passive warning 이벤트**(``WARNING_LARGE_COMMAND_DELTA``/
``WARNING_DIRECTION_MISMATCH``/``WARNING_POSITION_JUMP``/``WARNING_STATUS_NONZERO``, 추가로
``WARNING_LARGE_TRACKING_ERROR``/``WARNING_LOW_LOOP_RATE``)로 바꿨다 - **감지는 하되 절대
command를 막거나 수정하거나 루프를 끝내지 않는다.** 이 루프가 실제로 멈추는 경우는 다음
셋뿐이다: 목표 duration 경과(``DURATION_ELAPSED``), 사용자 Ctrl+C(``KEYBOARD_INTERRUPT``),
그리고 정상 teleop 자체가 더 이상 진행할 수 없는 진짜 실패
(``get_observation``/``get_action``/프로세서/``send_action`` 예외 - ``READ_FAILURE``).
**계측(레지스터 read) 자체의 실패는 더 이상 루프를 멈추지 않는다** - 그 cycle의
``register_read_error``만 기록하고 다음 cycle로 계속 진행한다(leader->follower 제어 경로는
이 계측 read와 완전히 독립적이므로).

follower로 나가는 유일한 write 경로는 여전히 ``lerobot.robots.so_follower.SOFollower``의
정상 ``connect()``(내부적으로 ``configure()`` 호출)/``send_action()``/``disconnect()``뿐이다.
이 모듈은 ``FeetechMotorsBus.write``/``sync_write``/``enable_torque``/``disable_torque``를
직접 호출하지 않는다 - ``SOFollower``가 이미 연 ``follower.bus``를 재사용해 ``bus.read()``
(순수 읽기)만 추가로 호출한다.

## 조사 근거 (설치된 lerobot에서 직접 확인 - ``~/lerobot/src/lerobot``, 추측 없음)

``robots/so_follower/so_follower.py`` / ``teleoperators/so_leader/so_leader.py`` /
``motors/feetech/feetech.py`` / ``motors/motors_bus.py`` /
``scripts/lerobot_teleoperate.py`` / ``processor/factory.py``를 다시 읽고 재확인:

1. **``SOFollower.connect()``**: ``bus.connect()`` -> (calibration 일치 시 ``calibrate()``
   스킵) -> 카메라 connect(빈 dict면 아무것도 안 함) -> ``self.configure()``.
2. **``SOFollower.configure()``**: ``with self.bus.torque_disabled(): configure_motors() +
   Operating_Mode/P·I·D_Coefficient write`` - ``torque_disabled()``는 ``disable_torque()``
   후 ``finally: enable_torque()``를 무조건 실행한다.
3. **``SOFollower.send_action(action)``**: ``.pos`` 접미사 키만 골라 ``goal_pos`` dict를
   만들고, ``max_relative_target``이 ``None``(기본값, 이 저장소가 바꾸지 않음)이면 present를
   다시 읽지 않고 곧바로 ``self.bus.sync_write("Goal_Position", goal_pos)`` **한 번만**
   호출한다. 이 모듈은 이 호출 앞에 어떤 조건도 추가하지 않는다.
4. **``SOLeader.get_action()``**: ``bus.sync_read("Present_Position", ...)`` 한 번.
5. **``lerobot_teleoperate.teleop_loop()``**: 매 cycle
   ``robot.get_observation()`` -> ``teleop.get_action()`` -> ``teleop_action_processor``
   -> ``robot_action_processor`` -> ``robot.send_action(...)`` ->
   ``precise_sleep(max(1/fps - dt, 0))`` 순서. ``make_default_processors()``가 SO-101
   기본 경로에서 반환하는 두 액션 프로세서는 ``IdentityProcessorStep()`` 하나뿐이라
   pass-through다. **이 모듈의 루프는 이 다섯 단계의 순서와 인자를 그대로 재현하며, 그
   사이에 어떤 조건 분기도 끼워 넣지 않는다** - 계측(레지스터 read/샘플 기록/warning 판정)은
   전부 ``send_action()`` 호출이 끝난 *뒤에* 일어난다.

이 모듈이 "추가"하는 것: (1) 매 cycle wrist_roll 레지스터 read(``Goal_Position``/
``Present_Position``/``Torque_Enable``/``Moving``/``Status`` + 저속 캐시되는
``Acceleration``/``Acceleration_Multiplier``), (2) 그 read 결과로부터 순수하게 계산되는
passive warning 이벤트, (3) CSV/대시보드 기록용 데이터 구조. write에 해당하는 메서드
(``write``/``sync_write``/``enable_torque``/``disable_torque``)는 이 모듈 어디에도 호출되지
않는다(``tests/test_instrumented_teleop.py``의 소스 감사로 재확인).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hardware.safety.shadow_teleop_diagnostic import (
    FOLLOWER_ACCEL_REGISTERS,
    FOLLOWER_STATE_REGISTERS,
    FollowerAccelSnapshot,
    FollowerStateSnapshot,
)
from hardware.safety.single_joint_test_planner import DEFAULT_MOTOR_RESOLUTION, raw_to_degrees

__all__ = [
    "JOINT_ORDER",
    "TARGET_JOINT",
    "InstrumentedTeleopError",
    "FullRegisterSnapshot",
    "WristRollRegisterInstrument",
    "WarningThresholds",
    "DEFAULT_COMMAND_DELTA_WARNING_DEG",
    "DIRECTION_MISMATCH_NOISE_TOLERANCE_DEG",
    "POSITION_JUMP_WARNING_DEG",
    "TRACKING_ERROR_WARNING_DEG",
    "LOW_LOOP_RATE_WARNING_HZ",
    "MOTION_ONSET_THRESHOLD_RAW_TICKS",
    "DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS",
    "DEFAULT_DEADBAND_LOOKAHEAD_MS",
    "DEFAULT_STATIONARY_WINDOW_SAMPLES",
    "MOTION_ONSET_INSUFFICIENT_DATA",
    "MOTION_ONSET_FOUND",
    "NO_RESPONSE",
    "RESPONSE",
    "OPPOSITE_MOTION",
    "check_command_delta_guard",
    "check_direction_mismatch",
    "check_position_jump",
    "check_tracking_error",
    "check_low_loop_rate",
    "classify_causal_response",
    "STOP_DURATION_ELAPSED",
    "STOP_KEYBOARD_INTERRUPT",
    "STOP_READ_FAILURE",
    "WARNING_LARGE_COMMAND_DELTA",
    "WARNING_DIRECTION_MISMATCH",
    "WARNING_POSITION_JUMP",
    "WARNING_STATUS_NONZERO",
    "WARNING_LARGE_TRACKING_ERROR",
    "WARNING_LOW_LOOP_RATE",
    "WarningEvent",
    "TeleopCycleSample",
    "TeleopRunResult",
    "CSV_FIELDNAMES",
    "run_instrumented_teleop_loop",
    "compute_run_analysis",
    "compute_deadband_summary",
    "compute_motion_onset_analysis",
    "INSUFFICIENT_FOR_DEADBAND_ESTIMATE",
    "INSUFFICIENT_DATA",
    "DEFAULT_ACCEL_REFRESH_INTERVAL_S",
]

JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
TARGET_JOINT = "wrist_roll"

DEFAULT_ACCEL_REFRESH_INTERVAL_S = 1.0  # Acceleration/Acceleration_Multiplier는 SRAM이라도 configure() 이후
# 이 프로그램이 직접 바꾸지 않으므로 매 cycle 다시 읽을 필요가 없다.

INSUFFICIENT_FOR_DEADBAND_ESTIMATE = "INSUFFICIENT_FOR_DEADBAND_ESTIMATE"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# --- warning 문턱값 기본값 (전부 "감지만" 한다 - 넘어도 command는 그대로 전달된다) -------------
DEFAULT_COMMAND_DELTA_WARNING_DEG = 2.0
# 근거: 1 tick(~0.0879°)는 과거 armed 실험에서 NO_MOTION으로 관측된 범위(+1/-2 tick)와 겹쳐
# encoder read 노이즈와 실제 미세 이동을 구분할 수 없다 - 2 tick(~0.176°)을 노이즈 문턱값으로
# 채택한다. 실측 튜닝값이 아니라 보수적 기본값이다.
DIRECTION_MISMATCH_NOISE_TOLERANCE_DEG = 0.176
# 근거: 예전 command-delta 가드 상한(2°)의 1.5배 - 정상적인 느린 수동 조작이라면 한 cycle에서
# 그 정도로 튀지 않는다는 보수적 참고값(실측 튜닝 아님, 감지 전용).
POSITION_JUMP_WARNING_DEG = 3.0
# 근거: 정상 추종 중 발생할 수 있는 순간 지연을 감안한 관대한 기본값 - "이 정도 오차가 나면
# 나중에 VLA deadband/threshold 설계 시 다시 봐야 한다"는 표시일 뿐, 확정값이 아니다.
TRACKING_ERROR_WARNING_DEG = 1.0
# 근거: 대상 fps의 절반 - 통신 지연/계측 오버헤드로 loop가 눈에 띄게 느려졌음을 알리는 참고값.
LOW_LOOP_RATE_WARNING_HZ = 20.0

MOTION_ONSET_THRESHOLD_RAW_TICKS = 2  # (레거시 참고용 - causal 분석에서는 아래 상수를 쓴다)

# --- causal deadband/motion-onset 분석 문턱값 -----------------------------------------------
# 근거: 두 값 모두 이 프로젝트가 이미 여러 곳(direction-mismatch 노이즈 tolerance, 과거 armed
# 실험의 NO_MOTION 관측 범위 +1/-2 tick)에서 채택한 2 tick(~0.176°)과 동일하다 - 1 tick은 read
# 노이즈/quantization과 구분되지 않는다는 것이 반복적으로 확인됐기 때문에, "실제 반응"으로
# 인정하는 최소 present 변화량도 동일하게 2 tick으로 맞춘다. 실측 캘리브레이션 값이 아니라
# 분석용 보수적 기본값이다.
DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS = 2

# 근거: servo 응답 latency(정확한 값은 60Hz 재측정 이후 확정 예정 - 89Hz 실행에서 관측된
# ~55ms를 확정값으로 쓰지 않는다)를 감안해 "에러 발생 직후 1 frame"보다 넉넉한 시간 창을
# 준다. 요구사항이 제시한 50~100ms 범위의 상단을 채택했다 - 너무 좁으면 실제 반응을 놓치고,
# 너무 넓으면 다음 명령의 영향과 섞인다는 trade-off를 감안한 보수적 시작값이다.
DEFAULT_DEADBAND_LOOKAHEAD_MS = 100.0

# motion onset "정지" 판정에 쓰는 직전 샘플 개수 - 60Hz 기준 약 80ms(50~100ms lookahead와
# 같은 자릿수), fps가 달라도 "몇 개의 최근 샘플"이라는 정의를 그대로 쓴다(간단함 우선).
DEFAULT_STATIONARY_WINDOW_SAMPLES = 5

MOTION_ONSET_INSUFFICIENT_DATA = "MOTION_ONSET_INSUFFICIENT_DATA"
MOTION_ONSET_FOUND = "MOTION_ONSET_FOUND"

NO_RESPONSE = "NO_RESPONSE"
RESPONSE = "RESPONSE"
OPPOSITE_MOTION = "OPPOSITE_MOTION"

# --- 종료 사유 (이 셋만 루프를 실제로 멈춘다) ------------------------------------------------
STOP_DURATION_ELAPSED = "DURATION_ELAPSED"
STOP_KEYBOARD_INTERRUPT = "KEYBOARD_INTERRUPT"
STOP_READ_FAILURE = "READ_FAILURE"  # 정상 teleop 자체가 더 이상 진행 불가능한 진짜 실패

# --- warning 이벤트 종류 (전부 기록만 함 - 절대 command/루프에 영향 없음) ---------------------
WARNING_LARGE_COMMAND_DELTA = "WARNING_LARGE_COMMAND_DELTA"
WARNING_DIRECTION_MISMATCH = "WARNING_DIRECTION_MISMATCH"
WARNING_POSITION_JUMP = "WARNING_POSITION_JUMP"
WARNING_STATUS_NONZERO = "WARNING_STATUS_NONZERO"
WARNING_LARGE_TRACKING_ERROR = "WARNING_LARGE_TRACKING_ERROR"
WARNING_LOW_LOOP_RATE = "WARNING_LOW_LOOP_RATE"


class InstrumentedTeleopError(RuntimeError):
    """이 모듈 자체의 사용/구성 오류이거나, action dict에 wrist_roll이 없는 등 예상치 못한 상태."""


# ---------------------------------------------------------------------------
# follower.bus를 재사용하는 read-only 계측기 (새 FeetechMotorsBus를 만들지 않는다)
# ---------------------------------------------------------------------------

_INITIAL_SNAPSHOT_REGISTERS: tuple[str, ...] = (
    "Torque_Enable",
    "Operating_Mode",
    "Goal_Position",
    "Present_Position",
    "Moving",
    "Status",
) + FOLLOWER_ACCEL_REGISTERS


@dataclass(frozen=True)
class FullRegisterSnapshot:
    """connect/configure 직후, 실제 움직임 시작 전에 읽는 wrist_roll 전체 레지스터 (참고용)."""

    torque_enable: int | None
    operating_mode: int | None
    goal_position_raw: int | None
    present_position_raw: int | None
    moving: int | None
    status_raw: int | None
    acceleration: int | None
    acceleration_multiplier: int | None
    read_errors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "Torque_Enable": self.torque_enable,
            "Operating_Mode": self.operating_mode,
            "Goal_Position": self.goal_position_raw,
            "Present_Position": self.present_position_raw,
            "Moving": self.moving,
            "Status": self.status_raw,
            "Acceleration": self.acceleration,
            "Acceleration_Multiplier": self.acceleration_multiplier,
            "read_errors": dict(self.read_errors),
        }


class WristRollRegisterInstrument:
    """``SOFollower``가 이미 연 ``bus``를 그대로 받아 wrist_roll 레지스터를 read-only로 계측한다.

    새 포트/버스를 만들지 않는다 - 생성자가 ``bus`` 객체 자체를 받는다. 공개 메서드는
    ``read_state``/``read_accel``/``read_full_snapshot``뿐이고, write에 해당하는 메서드는
    하나도 없다.
    """

    def __init__(self, *, bus: Any, num_read_retries: int = 2) -> None:
        self._bus = bus
        self._num_read_retries = num_read_retries

    def _read_registers(self, names: tuple[str, ...]) -> tuple[dict[str, int | None], dict[str, str]]:
        values: dict[str, int | None] = {}
        errors: dict[str, str] = {}
        for register in names:
            try:
                raw = self._bus.read(register, TARGET_JOINT, normalize=False, num_retry=self._num_read_retries)
                values[register] = int(raw)
            except Exception as exc:  # noqa: BLE001 - 통신 오류를 폭넓게 잡아 None+사유로 남긴다
                values[register] = None
                errors[register] = str(exc)
        return values, errors

    def read_state(self) -> FollowerStateSnapshot:
        values, errors = self._read_registers(FOLLOWER_STATE_REGISTERS)
        return FollowerStateSnapshot(
            goal_raw=values.get("Goal_Position"),
            present_raw=values.get("Present_Position"),
            torque_enable=values.get("Torque_Enable"),
            moving=values.get("Moving"),
            status_raw=values.get("Status"),
            read_errors=errors,
        )

    def read_accel(self) -> FollowerAccelSnapshot:
        values, errors = self._read_registers(FOLLOWER_ACCEL_REGISTERS)
        return FollowerAccelSnapshot(
            acceleration=values.get("Acceleration"),
            acceleration_multiplier=values.get("Acceleration_Multiplier "),
            read_errors=errors,
        )

    def read_full_snapshot(self) -> FullRegisterSnapshot:
        values, errors = self._read_registers(_INITIAL_SNAPSHOT_REGISTERS)
        return FullRegisterSnapshot(
            torque_enable=values.get("Torque_Enable"),
            operating_mode=values.get("Operating_Mode"),
            goal_position_raw=values.get("Goal_Position"),
            present_position_raw=values.get("Present_Position"),
            moving=values.get("Moving"),
            status_raw=values.get("Status"),
            acceleration=values.get("Acceleration"),
            acceleration_multiplier=values.get("Acceleration_Multiplier "),
            read_errors=errors,
        )


# ---------------------------------------------------------------------------
# warning 판정 함수 (순수 함수 - 하드웨어 접근 없음, 절대 command를 바꾸지 않는다)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarningThresholds:
    """모든 필드는 "감지" 문턱값이다 - 이 값을 넘어도 command/루프는 전혀 영향받지 않는다."""

    command_delta_max_deg: float = DEFAULT_COMMAND_DELTA_WARNING_DEG
    direction_mismatch_noise_tolerance_deg: float = DIRECTION_MISMATCH_NOISE_TOLERANCE_DEG
    position_jump_max_deg: float = POSITION_JUMP_WARNING_DEG
    tracking_error_max_deg: float = TRACKING_ERROR_WARNING_DEG
    low_loop_rate_hz: float = LOW_LOOP_RATE_WARNING_HZ


def check_command_delta_guard(
    *, command_wrist_roll_deg: float, follower_start_present_deg: float, max_delta_deg: float
) -> bool:
    """``True``면 문턱값 이내(정상), ``False``면 초과(경고 대상) - **command를 막지 않는다**."""
    return abs(command_wrist_roll_deg - follower_start_present_deg) <= max_delta_deg


def check_direction_mismatch(
    *, command_delta_from_start_deg: float, present_delta_from_start_deg: float, noise_tolerance_deg: float
) -> bool:
    """두 델타 모두 노이즈 문턱값을 넘어야 방향을 비교한다 (``True``=문제 없음)."""
    if abs(command_delta_from_start_deg) <= noise_tolerance_deg:
        return True
    if abs(present_delta_from_start_deg) <= noise_tolerance_deg:
        return True
    command_sign = 1 if command_delta_from_start_deg > 0 else -1
    present_sign = 1 if present_delta_from_start_deg > 0 else -1
    return command_sign == present_sign


def check_position_jump(*, present_delta_from_prev_deg: float | None, max_jump_deg: float) -> bool:
    """이전 값이 없으면(첫 cycle) 판단 불가하므로 ``True``(정상) 취급."""
    if present_delta_from_prev_deg is None:
        return True
    return abs(present_delta_from_prev_deg) <= max_jump_deg


def check_tracking_error(*, goal_present_error_deg: float | None, max_error_deg: float) -> bool:
    if goal_present_error_deg is None:
        return True
    return abs(goal_present_error_deg) <= max_error_deg


def check_low_loop_rate(*, loop_hz: float, min_hz: float) -> bool:
    return loop_hz >= min_hz


def classify_causal_response(
    *, error_raw: int, present_delta_raw: int, noise_threshold_ticks: int = DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS
) -> str:
    """"이 시점의 Goal-Present 오차(``error_raw``) 직후, lookahead window 안에서 실제로
    같은 방향으로 유의미하게 움직였는가"를 판정한다 (causal deadband/motion-onset 분석의
    핵심 판정 함수, 순수 계산).

    - ``present_delta_raw``의 절대값이 ``noise_threshold_ticks`` 미만이면 무조건
      :data:`NO_RESPONSE` (read 노이즈/quantization과 구분되지 않는 움직임은 "응답"으로
      인정하지 않는다).
    - ``error_raw == 0``이면 애초에 "기대되는 방향"이 없으므로(에러가 없는데 움직였다면
      그 원인을 이 에러로 돌릴 수 없다) 문턱값을 넘는 움직임이 있어도 :data:`NO_RESPONSE`로
      본다 - "이 에러가 이 움직임을 유발했다"고 주장할 인과관계 근거가 없기 때문이다.
    - 그 외에는 ``error_raw``와 ``present_delta_raw``의 부호가 같으면 :data:`RESPONSE`,
      다르면 :data:`OPPOSITE_MOTION`.
    """
    if abs(present_delta_raw) < noise_threshold_ticks:
        return NO_RESPONSE
    if error_raw == 0:
        return NO_RESPONSE
    error_sign = 1 if error_raw > 0 else -1
    delta_sign = 1 if present_delta_raw > 0 else -1
    return RESPONSE if error_sign == delta_sign else OPPOSITE_MOTION


# ---------------------------------------------------------------------------
# warning 이벤트 (섹션 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarningEvent:
    event_type: str
    timestamp_iso: str
    loop_index: int
    joint: str
    leader_value: float | None
    command_value: float | None
    goal_value: float | None
    present_value: float | None
    error_value: float | None
    threshold: float | None

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp_iso,
            "loop_index": self.loop_index,
            "joint": self.joint,
            "leader_value": self.leader_value,
            "command_value": self.command_value,
            "goal_value": self.goal_value,
            "present_value": self.present_value,
            "error_value": self.error_value,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# 샘플 + 결과
# ---------------------------------------------------------------------------

CSV_FIELDNAMES: tuple[str, ...] = (
    "loop_index",
    "timestamp",
    "elapsed_sec",
    "loop_hz",
    "leader_wrist_roll_deg",
    "leader_wrist_roll_delta_from_start_deg",
    "command_wrist_roll_deg",
    "follower_goal_raw",
    "follower_goal_deg",
    "follower_present_raw",
    "follower_present_deg",
    "goal_present_error_raw",
    "goal_present_error_deg",
    "follower_present_delta_from_prev_raw",
    "follower_present_delta_from_prev_deg",
    "follower_present_delta_from_start_deg",
    "follower_torque_enable",
    "follower_acceleration",
    "follower_acceleration_multiplier",
    "follower_moving",
    "follower_status",
    "send_action_executed",
    "register_read_error",
    "warning_types",
) + tuple(f"leader_command_{joint}" for joint in JOINT_ORDER) + tuple(
    f"follower_sent_{joint}" for joint in JOINT_ORDER
) + tuple(f"follower_observation_{joint}" for joint in JOINT_ORDER)


@dataclass(frozen=True)
class TeleopCycleSample:
    """한 teleop cycle의 계측 결과. CSV 한 행, 대시보드 한 프레임에 대응한다.

    이 샘플이 존재한다는 것 자체가 ``send_action()``이 이번 cycle에 정상 호출됐다는 뜻이다
    (passive 모드에서는 command가 절대 차단되지 않으므로 ``send_action_executed``는 항상
    ``True``다 - 필드는 CSV 스키마 안정성을 위해 유지한다).
    """

    loop_index: int
    timestamp_iso: str
    elapsed_sec: float
    loop_hz: float

    leader_wrist_roll_deg: float
    leader_wrist_roll_delta_from_start_deg: float

    command_wrist_roll_deg: float

    follower_goal_raw: int | None
    follower_goal_deg: float | None
    follower_present_raw: int | None
    follower_present_deg: float | None

    goal_present_error_raw: int | None
    goal_present_error_deg: float | None

    follower_present_delta_from_prev_raw: int | None
    follower_present_delta_from_prev_deg: float | None
    follower_present_delta_from_start_deg: float | None

    follower_torque_enable: int | None
    follower_acceleration: int | None
    follower_acceleration_multiplier: int | None
    follower_moving: int | None
    follower_status: int | None

    send_action_executed: bool

    leader_command_all_joints: dict[str, float]
    follower_sent_all_joints: dict[str, float]
    follower_observation_all_joints: dict[str, float]

    register_read_error: str | None = None
    warning_types: tuple[str, ...] = ()

    def to_csv_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "loop_index": self.loop_index,
            "timestamp": self.timestamp_iso,
            "elapsed_sec": self.elapsed_sec,
            "loop_hz": self.loop_hz,
            "leader_wrist_roll_deg": self.leader_wrist_roll_deg,
            "leader_wrist_roll_delta_from_start_deg": self.leader_wrist_roll_delta_from_start_deg,
            "command_wrist_roll_deg": self.command_wrist_roll_deg,
            "follower_goal_raw": self.follower_goal_raw,
            "follower_goal_deg": self.follower_goal_deg,
            "follower_present_raw": self.follower_present_raw,
            "follower_present_deg": self.follower_present_deg,
            "goal_present_error_raw": self.goal_present_error_raw,
            "goal_present_error_deg": self.goal_present_error_deg,
            "follower_present_delta_from_prev_raw": self.follower_present_delta_from_prev_raw,
            "follower_present_delta_from_prev_deg": self.follower_present_delta_from_prev_deg,
            "follower_present_delta_from_start_deg": self.follower_present_delta_from_start_deg,
            "follower_torque_enable": self.follower_torque_enable,
            "follower_acceleration": self.follower_acceleration,
            "follower_acceleration_multiplier": self.follower_acceleration_multiplier,
            "follower_moving": self.follower_moving,
            "follower_status": self.follower_status,
            "send_action_executed": self.send_action_executed,
            "register_read_error": self.register_read_error,
            "warning_types": ";".join(self.warning_types),
        }
        for joint in JOINT_ORDER:
            row[f"leader_command_{joint}"] = self.leader_command_all_joints.get(f"{joint}.pos")
            row[f"follower_sent_{joint}"] = self.follower_sent_all_joints.get(f"{joint}.pos")
            row[f"follower_observation_{joint}"] = self.follower_observation_all_joints.get(f"{joint}.pos")
        return row


@dataclass
class TeleopRunResult:
    samples: list[TeleopCycleSample]
    stopped_reason: str
    error: Exception | None = None
    initial_snapshot: FullRegisterSnapshot | None = None
    follower_start_present_raw: int | None = None
    follower_start_present_deg: float | None = None
    warnings: list[WarningEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 메인 루프 (teleop_loop()과 완전히 동일한 control-path 순서 - 계측은 send_action 이후에만)
# ---------------------------------------------------------------------------


def run_instrumented_teleop_loop(
    *,
    leader: Any,
    follower: Any,
    instrument: WristRollRegisterInstrument,
    follower_calibration_range: tuple[int, int],
    teleop_action_processor: Callable[[tuple[dict, dict]], dict],
    robot_action_processor: Callable[[tuple[dict, dict]], dict],
    fps: int,
    duration_sec: float | None,
    warning_thresholds: WarningThresholds = WarningThresholds(),
    accel_refresh_interval_s: float = DEFAULT_ACCEL_REFRESH_INTERVAL_S,
    on_sample: Callable[[TeleopCycleSample], None] | None = None,
    on_warning: Callable[[WarningEvent], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep_fn: Callable[[float], None] | None = None,
    motor_resolution: int = DEFAULT_MOTOR_RESOLUTION,
) -> TeleopRunResult:
    """``lerobot_teleoperate.teleop_loop()``과 완전히 동일한 control-path 순서
    (``get_observation`` -> ``get_action`` -> 프로세서 -> ``send_action`` -> fps sleep)를
    재현한다. **이 다섯 단계 사이에는 어떤 조건 분기도 없다** - ``send_action()``은 이
    다섯 단계에 도달한 모든 cycle에서 무조건 호출된다.

    계측(레지스터 read, warning 판정, 샘플 기록)은 전부 ``send_action()`` 호출이 끝난
    *뒤에* 일어나고, 그 계측 자체가 실패하거나 warning을 발생시켜도 루프는 절대 멈추거나
    다음 cycle의 command에 영향을 주지 않는다. 루프가 실제로 멈추는 경우는: 목표 duration
    경과, 사용자 Ctrl+C, 그리고 ``get_observation``/``get_action``/프로세서/``send_action``
    자체의 예외(정상 teleop이 더 이상 진행할 수 없는 진짜 실패) 셋뿐이다.

    ``leader``/``follower``는 이미 ``connect()``된 상태여야 한다 - 이 함수는 connect/
    disconnect를 하지 않는다(호출부 책임).

    ``sleep_fn``: 기본값(``None``)이면 ``lerobot.utils.robot_utils.precise_sleep``을 지연
    import해서 그대로 사용한다 - ``lerobot_teleoperate.teleop_loop()``가 실제로 쓰는 sleep
    구현과 동일하다(재구현하지 않는다). **이전 버전의 버그**: 기본값이 아무것도 하지 않는
    ``lambda _: None``이었던 탓에, 실물 실행에서 ``sleep_fn``을 명시적으로 넘기지 않으면
    fps 제한이 전혀 걸리지 않고 루프가 실제 작업 시간(레지스터 read 등)만큼의 속도로
    폭주했다 - 60Hz 요청에 실측 89.67Hz가 나온 정확한 원인이다. 테스트는 실제 sleep을
    피하기 위해 이 인자에 ``lambda _: None``(또는 기록용 spy)을 명시적으로 넘겨야 한다.
    """

    if sleep_fn is None:
        from lerobot.utils.robot_utils import precise_sleep as sleep_fn  # noqa: PLC0415

    range_min, range_max = follower_calibration_range

    def _follower_deg(raw: int) -> float:
        return raw_to_degrees(raw, range_min=range_min, range_max=range_max, motor_resolution=motor_resolution)

    # -- 초기 스냅샷 (참고용) - 실패해도 teleop 자체를 막지 않는다 --------------------------
    # 계측이 아직 시작도 안 한 시점이라 "command를 막는다"는 개념 자체가 없지만, 원칙을
    # 일관되게 유지하기 위해 이 read가 실패해도 루프 진입을 거부하지 않는다 - 단지 그 이후
    # follower_start_present_deg 기준 delta/warning 판정 일부를 계산할 수 없을 뿐이다.
    initial_read_error: str | None = None
    try:
        initial_snapshot = instrument.read_full_snapshot()
    except Exception as exc:  # noqa: BLE001
        initial_snapshot = FullRegisterSnapshot(
            torque_enable=None,
            operating_mode=None,
            goal_position_raw=None,
            present_position_raw=None,
            moving=None,
            status_raw=None,
            acceleration=None,
            acceleration_multiplier=None,
            read_errors={"read_full_snapshot": str(exc)},
        )
        initial_read_error = str(exc)

    follower_start_present_raw = initial_snapshot.present_position_raw
    follower_start_present_deg = (
        _follower_deg(follower_start_present_raw) if follower_start_present_raw is not None else None
    )

    samples: list[TeleopCycleSample] = []
    warnings: list[WarningEvent] = []
    start = clock()
    loop_index = 0
    leader_start_deg: float | None = None
    prev_present_raw: int | None = follower_start_present_raw
    prev_present_deg: float | None = follower_start_present_deg

    cached_accel: FollowerAccelSnapshot = FollowerAccelSnapshot(
        acceleration=initial_snapshot.acceleration,
        acceleration_multiplier=initial_snapshot.acceleration_multiplier,
        read_errors=({"initial": initial_read_error} if initial_read_error else {}),
    )
    cached_accel_at = clock()

    def _emit_warning(event: WarningEvent) -> None:
        warnings.append(event)
        if on_warning is not None:
            on_warning(event)

    def _result(stopped_reason: str, *, error: Exception | None = None) -> TeleopRunResult:
        return TeleopRunResult(
            samples=samples,
            stopped_reason=stopped_reason,
            error=error,
            initial_snapshot=initial_snapshot,
            follower_start_present_raw=follower_start_present_raw,
            follower_start_present_deg=follower_start_present_deg,
            warnings=warnings,
        )

    try:
        while True:
            if duration_sec is not None and (clock() - start) >= duration_sec:
                return _result(STOP_DURATION_ELAPSED)

            loop_start = clock()

            # === 정상 LeRobot control path (teleop_loop()과 동일 순서, 개입 없음) =========
            try:
                obs = follower.get_observation()
                raw_action = leader.get_action()
            except Exception as exc:  # noqa: BLE001 - 진짜 통신 실패 -> 정상 teleop도 못 함
                return _result(STOP_READ_FAILURE, error=exc)

            try:
                teleop_action = teleop_action_processor((raw_action, obs))
                robot_action_to_send = robot_action_processor((teleop_action, obs))
            except Exception as exc:  # noqa: BLE001
                return _result(STOP_READ_FAILURE, error=exc)

            leader_wrist_roll_deg = raw_action.get(f"{TARGET_JOINT}.pos")
            command_wrist_roll_deg = robot_action_to_send.get(f"{TARGET_JOINT}.pos")
            if leader_wrist_roll_deg is None or command_wrist_roll_deg is None:
                return _result(
                    STOP_READ_FAILURE,
                    error=InstrumentedTeleopError(f"action dict에 '{TARGET_JOINT}.pos'가 없습니다."),
                )

            if leader_start_deg is None:
                leader_start_deg = leader_wrist_roll_deg

            try:
                sent_action = follower.send_action(robot_action_to_send)
            except Exception as exc:  # noqa: BLE001 - 진짜 통신 실패
                return _result(STOP_READ_FAILURE, error=exc)
            # === control path 끝 - 여기부터는 전부 read-only 계측이다 =====================

            now = clock()
            wall_now_iso = wall_clock().isoformat()

            state: FollowerStateSnapshot
            register_read_error: str | None
            try:
                state = instrument.read_state()
                register_read_error = None
                if state.read_errors:
                    register_read_error = "; ".join(f"{k}: {v}" for k, v in state.read_errors.items())
            except Exception as exc:  # noqa: BLE001 - 계측 실패는 절대 루프를 멈추지 않는다
                state = FollowerStateSnapshot(
                    goal_raw=None, present_raw=None, torque_enable=None, moving=None, status_raw=None
                )
                register_read_error = str(exc)

            if (now - cached_accel_at) >= accel_refresh_interval_s:
                try:
                    cached_accel = instrument.read_accel()
                except Exception as exc:  # noqa: BLE001 - accel도 선택 정보라 실패해도 진행
                    cached_accel = FollowerAccelSnapshot(
                        acceleration=None, acceleration_multiplier=None, read_errors={"accel_read": str(exc)}
                    )
                cached_accel_at = now

            present_raw = state.present_raw
            present_deg = _follower_deg(present_raw) if present_raw is not None else None
            goal_raw = state.goal_raw
            goal_deg = _follower_deg(goal_raw) if goal_raw is not None else None

            present_delta_from_prev_raw = (
                present_raw - prev_present_raw if present_raw is not None and prev_present_raw is not None else None
            )
            present_delta_from_prev_deg = (
                present_deg - prev_present_deg if present_deg is not None and prev_present_deg is not None else None
            )
            present_delta_from_start_deg = (
                present_deg - follower_start_present_deg
                if present_deg is not None and follower_start_present_deg is not None
                else None
            )

            goal_present_error_raw = goal_raw - present_raw if goal_raw is not None and present_raw is not None else None
            goal_present_error_deg = goal_deg - present_deg if goal_deg is not None and present_deg is not None else None

            loop_dt = clock() - loop_start
            loop_hz = (1.0 / loop_dt) if loop_dt > 0 else 0.0

            # -- warning 판정 (전부 감지만 - command/루프에 어떤 영향도 주지 않는다) ----------
            triggered: list[str] = []

            command_delta_from_start = (
                command_wrist_roll_deg - follower_start_present_deg
                if follower_start_present_deg is not None
                else None
            )
            if command_delta_from_start is not None and not check_command_delta_guard(
                command_wrist_roll_deg=command_wrist_roll_deg,
                follower_start_present_deg=follower_start_present_deg,
                max_delta_deg=warning_thresholds.command_delta_max_deg,
            ):
                triggered.append(WARNING_LARGE_COMMAND_DELTA)
                _emit_warning(
                    WarningEvent(
                        event_type=WARNING_LARGE_COMMAND_DELTA,
                        timestamp_iso=wall_now_iso,
                        loop_index=loop_index,
                        joint=TARGET_JOINT,
                        leader_value=leader_wrist_roll_deg,
                        command_value=command_wrist_roll_deg,
                        goal_value=goal_deg,
                        present_value=present_deg,
                        error_value=command_delta_from_start,
                        threshold=warning_thresholds.command_delta_max_deg,
                    )
                )

            if present_delta_from_start_deg is not None and command_delta_from_start is not None and not check_direction_mismatch(
                command_delta_from_start_deg=command_delta_from_start,
                present_delta_from_start_deg=present_delta_from_start_deg,
                noise_tolerance_deg=warning_thresholds.direction_mismatch_noise_tolerance_deg,
            ):
                triggered.append(WARNING_DIRECTION_MISMATCH)
                _emit_warning(
                    WarningEvent(
                        event_type=WARNING_DIRECTION_MISMATCH,
                        timestamp_iso=wall_now_iso,
                        loop_index=loop_index,
                        joint=TARGET_JOINT,
                        leader_value=leader_wrist_roll_deg,
                        command_value=command_delta_from_start,
                        goal_value=goal_deg,
                        present_value=present_delta_from_start_deg,
                        error_value=None,
                        threshold=warning_thresholds.direction_mismatch_noise_tolerance_deg,
                    )
                )

            if not check_position_jump(
                present_delta_from_prev_deg=present_delta_from_prev_deg,
                max_jump_deg=warning_thresholds.position_jump_max_deg,
            ):
                triggered.append(WARNING_POSITION_JUMP)
                _emit_warning(
                    WarningEvent(
                        event_type=WARNING_POSITION_JUMP,
                        timestamp_iso=wall_now_iso,
                        loop_index=loop_index,
                        joint=TARGET_JOINT,
                        leader_value=leader_wrist_roll_deg,
                        command_value=command_wrist_roll_deg,
                        goal_value=goal_deg,
                        present_value=present_delta_from_prev_deg,
                        error_value=present_delta_from_prev_deg,
                        threshold=warning_thresholds.position_jump_max_deg,
                    )
                )

            if state.status_raw is not None and state.status_raw != 0:
                triggered.append(WARNING_STATUS_NONZERO)
                _emit_warning(
                    WarningEvent(
                        event_type=WARNING_STATUS_NONZERO,
                        timestamp_iso=wall_now_iso,
                        loop_index=loop_index,
                        joint=TARGET_JOINT,
                        leader_value=leader_wrist_roll_deg,
                        command_value=command_wrist_roll_deg,
                        goal_value=goal_deg,
                        present_value=present_deg,
                        error_value=float(state.status_raw),
                        threshold=0.0,
                    )
                )

            if not check_tracking_error(
                goal_present_error_deg=goal_present_error_deg, max_error_deg=warning_thresholds.tracking_error_max_deg
            ):
                triggered.append(WARNING_LARGE_TRACKING_ERROR)
                _emit_warning(
                    WarningEvent(
                        event_type=WARNING_LARGE_TRACKING_ERROR,
                        timestamp_iso=wall_now_iso,
                        loop_index=loop_index,
                        joint=TARGET_JOINT,
                        leader_value=leader_wrist_roll_deg,
                        command_value=command_wrist_roll_deg,
                        goal_value=goal_deg,
                        present_value=present_deg,
                        error_value=goal_present_error_deg,
                        threshold=warning_thresholds.tracking_error_max_deg,
                    )
                )

            if not check_low_loop_rate(loop_hz=loop_hz, min_hz=warning_thresholds.low_loop_rate_hz):
                triggered.append(WARNING_LOW_LOOP_RATE)
                _emit_warning(
                    WarningEvent(
                        event_type=WARNING_LOW_LOOP_RATE,
                        timestamp_iso=wall_now_iso,
                        loop_index=loop_index,
                        joint=TARGET_JOINT,
                        leader_value=None,
                        command_value=None,
                        goal_value=None,
                        present_value=None,
                        error_value=loop_hz,
                        threshold=warning_thresholds.low_loop_rate_hz,
                    )
                )

            sample = TeleopCycleSample(
                loop_index=loop_index,
                timestamp_iso=wall_now_iso,
                elapsed_sec=loop_start - start,
                loop_hz=loop_hz,
                leader_wrist_roll_deg=leader_wrist_roll_deg,
                leader_wrist_roll_delta_from_start_deg=leader_wrist_roll_deg - leader_start_deg,
                command_wrist_roll_deg=command_wrist_roll_deg,
                follower_goal_raw=goal_raw,
                follower_goal_deg=goal_deg,
                follower_present_raw=present_raw,
                follower_present_deg=present_deg,
                goal_present_error_raw=goal_present_error_raw,
                goal_present_error_deg=goal_present_error_deg,
                follower_present_delta_from_prev_raw=present_delta_from_prev_raw,
                follower_present_delta_from_prev_deg=present_delta_from_prev_deg,
                follower_present_delta_from_start_deg=present_delta_from_start_deg,
                follower_torque_enable=state.torque_enable,
                follower_acceleration=cached_accel.acceleration,
                follower_acceleration_multiplier=cached_accel.acceleration_multiplier,
                follower_moving=state.moving,
                follower_status=state.status_raw,
                send_action_executed=True,
                leader_command_all_joints=dict(raw_action),
                follower_sent_all_joints=dict(sent_action),
                follower_observation_all_joints=dict(obs),
                register_read_error=register_read_error,
                warning_types=tuple(triggered),
            )
            samples.append(sample)
            loop_index += 1

            if on_sample is not None:
                on_sample(sample)

            prev_present_raw = present_raw if present_raw is not None else prev_present_raw
            prev_present_deg = present_deg if present_deg is not None else prev_present_deg

            dt_s = clock() - loop_start
            sleep_fn(max(1.0 / fps - dt_s, 0.0))
    except KeyboardInterrupt:
        return _result(STOP_KEYBOARD_INTERRUPT)


# ---------------------------------------------------------------------------
# 분석 (섹션 10) - 순수 계산
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """선형 보간 percentile (numpy 없이) - ``sorted_values``는 이미 정렬돼 있어야 한다."""
    if not sorted_values:
        raise ValueError("빈 리스트의 percentile은 계산할 수 없습니다.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac


def _estimate_command_to_actual_lag_frames(
    command_deltas: list[float], present_deltas: list[float], *, max_lag_frames: int = 30, min_samples: int = 40
) -> int | None:
    """command 변화가 실제 follower 변화보다 몇 frame 앞서는지 추정한다 (best-effort, 순수 계산).

    데이터가 부족하거나(샘플 수/움직임 모두) 유의미한 상관을 찾지 못하면 ``None``을 반환한다 -
    호출부는 이 경우 ``INSUFFICIENT_DATA``로 보고해야 한다(임의로 확정하지 않음).
    """
    n = len(command_deltas)
    if n < min_samples or n != len(present_deltas):
        return None

    cmd_mean = sum(command_deltas) / n
    pres_mean = sum(present_deltas) / n
    cmd = [c - cmd_mean for c in command_deltas]
    pres = [p - pres_mean for p in present_deltas]

    cmd_energy = sum(c * c for c in cmd)
    pres_energy = sum(p * p for p in pres)
    if cmd_energy < 1e-9 or pres_energy < 1e-9:
        return None  # 움직임이 사실상 없어 상관을 계산할 수 없다

    best_lag: int | None = None
    best_score = 0.0
    for lag in range(0, min(max_lag_frames, n - 1) + 1):
        score = sum(pres[t] * cmd[t - lag] for t in range(lag, n))
        if score > best_score:
            best_score = score
            best_lag = lag

    return best_lag


def compute_run_analysis(
    result: TeleopRunResult,
    *,
    deadband_lookahead_ms: float = DEFAULT_DEADBAND_LOOKAHEAD_MS,
    motion_response_noise_threshold_ticks: int = DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS,
    stationary_window_samples: int = DEFAULT_STATIONARY_WINDOW_SAMPLES,
) -> dict[str, Any]:
    samples = result.samples
    warning_counts: dict[str, int] = {}
    for w in result.warnings:
        warning_counts[w.event_type] = warning_counts.get(w.event_type, 0) + 1

    if not samples:
        return {
            "sample_count": 0,
            "stopped_reason": result.stopped_reason,
            "warning_counts": warning_counts,
            "total_warning_count": len(result.warnings),
        }

    n = len(samples)
    elapsed = samples[-1].elapsed_sec - samples[0].elapsed_sec
    actual_loop_hz = (n - 1) / elapsed if elapsed > 0 else 0.0

    leader_degs = [s.leader_wrist_roll_deg for s in samples]
    command_degs = [s.command_wrist_roll_deg for s in samples]
    present_degs = [s.follower_present_deg for s in samples if s.follower_present_deg is not None]
    goal_present_error_raws = [s.goal_present_error_raw for s in samples if s.goal_present_error_raw is not None]
    goal_present_error_degs = [s.goal_present_error_deg for s in samples if s.goal_present_error_deg is not None]
    abs_error_degs = sorted(abs(e) for e in goal_present_error_degs)
    frame_deltas_deg = [
        abs(s.follower_present_delta_from_prev_deg)
        for s in samples
        if s.follower_present_delta_from_prev_deg is not None
    ]

    accel_values = {s.follower_acceleration for s in samples if s.follower_acceleration is not None}
    torque_values = {s.follower_torque_enable for s in samples if s.follower_torque_enable is not None}
    status_ever_nonzero = any((s.follower_status or 0) != 0 for s in samples)
    register_read_error_count = sum(1 for s in samples if s.register_read_error is not None)

    analysis: dict[str, Any] = {
        "sample_count": n,
        "elapsed_sec": elapsed,
        "actual_loop_hz": actual_loop_hz,
        "stopped_reason": result.stopped_reason,
        "configure_initial_torque_enable": result.initial_snapshot.torque_enable if result.initial_snapshot else None,
        "configure_initial_operating_mode": result.initial_snapshot.operating_mode if result.initial_snapshot else None,
        "configure_initial_acceleration": result.initial_snapshot.acceleration if result.initial_snapshot else None,
        "configure_initial_status": result.initial_snapshot.status_raw if result.initial_snapshot else None,
        "leader_movement_range_deg": (max(leader_degs) - min(leader_degs)) if leader_degs else None,
        "command_movement_range_deg": (max(command_degs) - min(command_degs)) if command_degs else None,
        "follower_movement_range_deg": (max(present_degs) - min(present_degs)) if present_degs else None,
        "goal_present_error_raw_min": min(goal_present_error_raws) if goal_present_error_raws else None,
        "goal_present_error_raw_max": max(goal_present_error_raws) if goal_present_error_raws else None,
        "goal_present_error_deg_min": min(goal_present_error_degs) if goal_present_error_degs else None,
        "goal_present_error_deg_max": max(goal_present_error_degs) if goal_present_error_degs else None,
        "max_follower_lag_deg": max(abs_error_degs) if abs_error_degs else None,
        "mae_tracking_error_deg": (sum(abs_error_degs) / len(abs_error_degs)) if abs_error_degs else None,
        "tracking_error_p50_deg": _percentile(abs_error_degs, 50) if abs_error_degs else None,
        "tracking_error_p90_deg": _percentile(abs_error_degs, 90) if abs_error_degs else None,
        "tracking_error_p95_deg": _percentile(abs_error_degs, 95) if abs_error_degs else None,
        "tracking_error_p99_deg": _percentile(abs_error_degs, 99) if abs_error_degs else None,
        "max_frame_to_frame_delta_deg": max(frame_deltas_deg) if frame_deltas_deg else None,
        "mean_frame_to_frame_delta_deg": (sum(frame_deltas_deg) / len(frame_deltas_deg)) if frame_deltas_deg else None,
        "estimated_max_velocity_deg_per_s": (
            max(frame_deltas_deg) * actual_loop_hz if frame_deltas_deg and actual_loop_hz > 0 else None
        ),
        "estimated_mean_velocity_deg_per_s": (
            (sum(frame_deltas_deg) / len(frame_deltas_deg)) * actual_loop_hz
            if frame_deltas_deg and actual_loop_hz > 0
            else None
        ),
        "acceleration_values_seen": sorted(accel_values),
        "torque_enable_values_seen": sorted(torque_values),
        "status_ever_nonzero": status_ever_nonzero,
        "register_read_error_count": register_read_error_count,
        "warning_counts": warning_counts,
        "total_warning_count": len(result.warnings),
        "write_count_direct_register_writes": 0,
    }

    # -- command -> actual lag 추정 (best-effort, 부족하면 INSUFFICIENT_DATA) --------------
    # 알고리즘: 평균을 뺀 command/present 변화량 시퀀스에 대해, 0..max_lag_frames 범위의 각
    # lag 후보마다 present[t]와 command[t-lag]의 내적(covariance-like score)을 계산하고,
    # 그 score가 최대인 lag를 채택한다(간이 cross-correlation, numpy 없이 순수 파이썬).
    command_deltas = [s.command_wrist_roll_deg - command_degs[0] for s in samples]
    present_deltas_for_lag = [
        (s.follower_present_deg - present_degs[0]) if s.follower_present_deg is not None else 0.0 for s in samples
    ]
    lag_frames = _estimate_command_to_actual_lag_frames(command_deltas, present_deltas_for_lag)
    if lag_frames is None:
        analysis["command_to_actual_lag_estimate"] = INSUFFICIENT_DATA
    else:
        analysis["command_to_actual_lag_frames"] = lag_frames
        # frame 기반: 평균 loop_hz로 환산 (loop rate가 일정하다고 가정).
        analysis["command_to_actual_lag_ms_frame_based"] = (
            (lag_frames / actual_loop_hz * 1000.0) if actual_loop_hz > 0 else None
        )
        # timestamp 기반: 실제 elapsed_sec 차이를 lag_frames만큼 떨어진 샘플 쌍 전부에서 평균
        # 낸다 - loop rate가 흔들렸다면 frame 기반 값과 달라질 수 있으므로 둘 다 보고해서
        # 서로 검증할 수 있게 한다.
        ts_diffs = [samples[i + lag_frames].elapsed_sec - samples[i].elapsed_sec for i in range(n - lag_frames)]
        analysis["command_to_actual_lag_ms_timestamp_based"] = (
            (sum(ts_diffs) / len(ts_diffs)) * 1000.0 if ts_diffs else None
        )

    # -- causal motion onset (섹션 10) -------------------------------------------------------
    analysis["motion_onset"] = compute_motion_onset_analysis(
        result,
        lookahead_ms=deadband_lookahead_ms,
        noise_threshold_ticks=motion_response_noise_threshold_ticks,
        stationary_window_samples=stationary_window_samples,
    )

    # -- causal deadband (섹션 4~9) -----------------------------------------------------------
    analysis["deadband_summary"] = compute_deadband_summary(
        result, lookahead_ms=deadband_lookahead_ms, noise_threshold_ticks=motion_response_noise_threshold_ticks
    )

    return analysis


def _find_lookahead_end_index(samples: list[TeleopCycleSample], start_index: int, lookahead_s: float, j_hint: int) -> tuple[int | None, int]:
    """``samples[start_index]``의 timestamp로부터 ``lookahead_s`` 이내에 있는 마지막 샘플의
    인덱스를 찾는다. ``j_hint``는 이전 호출에서 반환된 다음 탐색 시작점(단조 증가 - 전체
    호출에서 amortized O(n)을 보장하기 위한 two-pointer 기법)이다.

    Returns:
        (end_index 또는 None(lookahead 안에 유효한 다음 샘플이 없음), 다음 호출에 넘길 j_hint)
    """
    target_elapsed = samples[start_index].elapsed_sec + lookahead_s
    j = max(j_hint, start_index + 1)
    while j < len(samples) and samples[j].elapsed_sec <= target_elapsed:
        j += 1
    end_index = j - 1
    if end_index <= start_index:
        return None, j
    return end_index, j


def compute_deadband_summary(
    result: TeleopRunResult,
    *,
    lookahead_ms: float = DEFAULT_DEADBAND_LOOKAHEAD_MS,
    noise_threshold_ticks: int = DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS,
    min_samples_per_bucket: int = 3,
) -> dict[str, Any]:
    """**Causal** deadband 분석 (섹션 4~9): |goal-present error raw tick| 구간별로, 그
    "에러가 관측된 시점 이후" ``lookahead_ms`` 안에서 실제로 같은 방향으로 반응했는지를
    ``classify_causal_response``로 판정해서 집계한다.

    이전 버전과의 결정적 차이(섹션 4의 버그 수정): 이전에는 "현재 present가 세션 시작
    위치에서 얼마나 멀어졌는가"만 봤기 때문에, follower가 한 번이라도 움직인 뒤에는 그 이후
    모든 샘플이(현재 에러가 0이든 크든 상관없이) "움직임 있음"으로 잘못 집계됐다 - 그래서
    0/1/2 tick 구간까지도 "100% motion observed"라는 해석 불가능한 결과가 나왔다. 이번
    버전은 각 샘플의 에러 발생 "직후"만 인과적으로 본다.

    데이터가 부족하면 무리하게 결론내지 않고 ``INSUFFICIENT_FOR_DEADBAND_ESTIMATE``를 반환한다.
    """
    samples = result.samples
    if not samples:
        return {"verdict": INSUFFICIENT_FOR_DEADBAND_ESTIMATE, "reason": "샘플이 없습니다.", "buckets": []}

    lookahead_s = lookahead_ms / 1000.0
    buckets: dict[int, list[str]] = {}
    j_hint = 0
    for i, s in enumerate(samples):
        if s.goal_present_error_raw is None or s.follower_present_raw is None:
            continue
        end_index, j_hint = _find_lookahead_end_index(samples, i, lookahead_s, j_hint)
        if end_index is None or samples[end_index].follower_present_raw is None:
            continue  # lookahead window 안에 유효한 present read가 없음 - 이 샘플은 건너뛴다
        present_delta_raw = samples[end_index].follower_present_raw - s.follower_present_raw
        label = classify_causal_response(
            error_raw=s.goal_present_error_raw, present_delta_raw=present_delta_raw, noise_threshold_ticks=noise_threshold_ticks
        )
        bucket_key = min(abs(s.goal_present_error_raw), 6)  # 6 이상은 "6+"로 묶는다
        buckets.setdefault(bucket_key, []).append(label)

    if not buckets:
        return {
            "verdict": INSUFFICIENT_FOR_DEADBAND_ESTIMATE,
            "reason": "lookahead 안에서 판정 가능한 (error, 이후 present) 쌍이 없습니다.",
            "buckets": [],
        }

    bucket_rows = []
    any_bucket_has_enough_samples = False
    for bucket_key in sorted(buckets):
        labels = buckets[bucket_key]
        sample_count = len(labels)
        if sample_count >= min_samples_per_bucket:
            any_bucket_has_enough_samples = True
        response_count = labels.count(RESPONSE)
        no_response_count = labels.count(NO_RESPONSE)
        opposite_motion_count = labels.count(OPPOSITE_MOTION)
        bucket_rows.append(
            {
                "abs_goal_present_error_ticks": "6+" if bucket_key == 6 else bucket_key,
                "sample_count": sample_count,
                "response_count": response_count,
                "no_response_count": no_response_count,
                "opposite_motion_count": opposite_motion_count,
                "response_fraction": (response_count / sample_count) if sample_count else None,
            }
        )

    if not any_bucket_has_enough_samples:
        return {
            "verdict": INSUFFICIENT_FOR_DEADBAND_ESTIMATE,
            "reason": (
                f"모든 tick 구간의 샘플 수가 최소 기준({min_samples_per_bucket})에 못 미칩니다 - "
                "deadband 추정을 위한 데이터가 부족합니다."
            ),
            "lookahead_ms": lookahead_ms,
            "noise_threshold_ticks": noise_threshold_ticks,
            "buckets": bucket_rows,
        }

    first_response_bucket = next((row for row in bucket_rows if row["response_count"] > 0), None)
    if first_response_bucket is None:
        return {
            "verdict": INSUFFICIENT_FOR_DEADBAND_ESTIMATE,
            "reason": "관측된 모든 tick 구간에서 RESPONSE가 한 번도 감지되지 않았습니다.",
            "lookahead_ms": lookahead_ms,
            "noise_threshold_ticks": noise_threshold_ticks,
            "buckets": bucket_rows,
        }

    return {
        "verdict": "DEADBAND_ESTIMATE_AVAILABLE",
        "first_response_bucket_abs_ticks": first_response_bucket["abs_goal_present_error_ticks"],
        "lookahead_ms": lookahead_ms,
        "noise_threshold_ticks": noise_threshold_ticks,
        "buckets": bucket_rows,
    }


def compute_motion_onset_analysis(
    result: TeleopRunResult,
    *,
    lookahead_ms: float = DEFAULT_DEADBAND_LOOKAHEAD_MS,
    noise_threshold_ticks: int = DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS,
    stationary_window_samples: int = DEFAULT_STATIONARY_WINDOW_SAMPLES,
) -> dict[str, Any]:
    """섹션 10: **정지 상태에서 새로운 command error가 발생한 뒤, follower가 실제
    same-direction motion을 시작한 최초 이벤트**를 찾는다 (causal, 순수 계산).

    세 조건을 모두 만족하는 첫 샘플을 찾는다:

    1. **정지**: 직전 ``stationary_window_samples``개 샘플 동안
       ``follower_present_delta_from_prev_raw``의 절대값이 전부 ``noise_threshold_ticks``
       미만 (이미 움직이고 있던 follower를 "새로운 onset"으로 잘못 잡지 않기 위함).
    2. **새로운 에러 발생**: 그 직전 window 동안은 ``goal_present_error_raw``의 절대값이
       전부 ``noise_threshold_ticks`` 미만이었는데(=거의 goal==present였는데), 지금 샘플에서
       처음으로 그 문턱값을 넘음 (Goal==Present인 상태를 움직임의 원인으로 오판하지 않기
       위함 - ``classify_causal_response``의 ``error_raw == 0`` 처리와 같은 원칙).
    3. **실제 반응**: ``lookahead_ms`` 안에서 ``classify_causal_response``가 :data:`RESPONSE`.

    세 조건을 모두 만족하는 샘플이 하나도 없으면 (억지로 값을 만들지 않고)
    :data:`MOTION_ONSET_INSUFFICIENT_DATA`를 반환한다.
    """
    samples = result.samples
    if len(samples) < stationary_window_samples + 2:
        return {
            "verdict": MOTION_ONSET_INSUFFICIENT_DATA,
            "reason": f"샘플 수({len(samples)})가 정지 판정 window({stationary_window_samples}) + 2보다 적습니다.",
        }

    lookahead_s = lookahead_ms / 1000.0
    j_hint = 0
    for i in range(stationary_window_samples, len(samples)):
        s = samples[i]
        if s.goal_present_error_raw is None or s.follower_present_raw is None:
            continue

        window = samples[i - stationary_window_samples : i]

        # 조건 1: 직전 window 동안 거의 정지 상태였는가.
        if any(
            w.follower_present_delta_from_prev_raw is None or abs(w.follower_present_delta_from_prev_raw) >= noise_threshold_ticks
            for w in window
        ):
            continue

        # 조건 2: 직전 window의 에러는 작았는데(거의 goal==present), 지금 새로 문턱값을 넘었는가.
        prior_errors = [w.goal_present_error_raw for w in window if w.goal_present_error_raw is not None]
        if len(prior_errors) < len(window):  # window 안에 에러를 못 읽은 샘플이 있으면 판단 불가
            continue
        if any(abs(e) >= noise_threshold_ticks for e in prior_errors):
            continue  # 이미 에러가 있었으면 "새로 발생"이 아니다
        if abs(s.goal_present_error_raw) < noise_threshold_ticks:
            continue  # 아직 유의미한 에러가 아니다

        # 조건 3: lookahead 안에서 실제로 같은 방향으로 반응했는가.
        end_index, j_hint = _find_lookahead_end_index(samples, i, lookahead_s, j_hint)
        if end_index is None or samples[end_index].follower_present_raw is None:
            continue
        present_delta_raw = samples[end_index].follower_present_raw - s.follower_present_raw
        label = classify_causal_response(
            error_raw=s.goal_present_error_raw, present_delta_raw=present_delta_raw, noise_threshold_ticks=noise_threshold_ticks
        )
        if label != RESPONSE:
            continue

        return {
            "verdict": MOTION_ONSET_FOUND,
            "loop_index": s.loop_index,
            "elapsed_sec": s.elapsed_sec,
            "goal_present_error_raw_at_onset": s.goal_present_error_raw,
            "goal_present_error_deg_at_onset": s.goal_present_error_deg,
            "command_delta_from_start_deg_at_onset": (
                (s.command_wrist_roll_deg - result.follower_start_present_deg)
                if result.follower_start_present_deg is not None
                else None
            ),
            "leader_delta_from_start_deg_at_onset": s.leader_wrist_roll_delta_from_start_deg,
            "present_delta_in_lookahead_raw_ticks": present_delta_raw,
            "lookahead_ms": lookahead_ms,
            "noise_threshold_ticks": noise_threshold_ticks,
            "stationary_window_samples": stationary_window_samples,
        }

    return {
        "verdict": MOTION_ONSET_INSUFFICIENT_DATA,
        "reason": (
            "정지 -> 새 에러 발생 -> lookahead 안 same-direction 반응, 세 조건을 모두 만족하는 "
            "이벤트를 찾지 못했습니다."
        ),
        "lookahead_ms": lookahead_ms,
        "noise_threshold_ticks": noise_threshold_ticks,
        "stationary_window_samples": stationary_window_samples,
    }
