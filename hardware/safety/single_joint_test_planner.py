"""wrist_roll 단일 관절 안전 시험 계획 - 순수 계산 전용 모듈.

이 모듈은 어떤 하드웨어 통신도, 파일/네트워크 접근도 하지 않는다 (serial/lerobot
import 없음). "현재 raw tick이 X, 방향이 Y다"라는 숫자만 받아서 이동 계획(목표
각도/raw tick, calibration 내부 범위 통과 여부, 1° 로컬 한계 통과 여부,
스텝별/전체 PASS-BLOCKED 판정)을 계산할 뿐이다. 실제 연결/읽기는
``hardware/safety/single_joint_hardware_inspector.py``가, write는 (이번 단계에서
의도적으로 구현되지 않은) ``hardware/safety/single_joint_writer.py``가 담당한다.

이 파일에는 write에 해당하는 함수/메서드가 하나도 없다.

## wrist_roll full-turn 처리 방식 (조사 근거)

``~/lerobot/src/lerobot/motors/motors_bus.py``의 ``MotorsBus._normalize``/
``_unnormalize``를 직접 읽고 확인한 내용:

- ``DEGREES`` 정규화 공식은 ``degree = (raw - mid) * 360 / max_res``
  (``mid=(range_min+range_max)/2``, ``max_res=motor_resolution-1=4095``)이며,
  **이 공식은 raw 값을 calibration range로 clamp하지 않는다** - ``_normalize``에서
  ``RANGE_M100_100``/``RANGE_0_100`` 분기만 ``bounded_val``(min/max로 clamp된 값)을
  쓰고, ``DEGREES`` 분기는 원본 ``val``을 그대로 쓴다 (motors_bus.py 854~881행).
  ``_unnormalize``의 ``DEGREES`` 분기도 대칭적으로 clamp가 없다 (883~911행).
- ``~/lerobot/src/lerobot/robots/so_follower/so_follower.py``의 ``calibrate()``는
  wrist_roll을 ``full_turn_motor``로 명시하고 ``record_ranges_of_motion()``을 거치지
  않은 채 ``range_min=0``, ``range_max=4095``를 그대로 대입한다 (135~143행). 실제
  ``chanho_follower.json`` 캘리브레이션 파일도 정확히 이 값이다.
- Feetech STS3215 컨트롤 테이블의 ``Present_Position``은 2바이트 sign-magnitude
  인코딩(부호 비트 인덱스 15, ``tables.py`` ``STS_SMS_SERIES_ENCODINGS_TABLE``)이지만,
  ``Operating_Mode=POSITION``(단일 회전 모드 - ``SOFollower.calibrate()``가 설정하는
  값)일 때는 실측 raw tick이 물리적으로 0~4095(``MODEL_RESOLUTION["sts3215"]-1``)
  범위 안에서만 보고된다. 즉 raw=0과 raw=4095는 각도 표현상 -180°/+180°로 정반대처럼
  보이지만, 실제로는 서로 인접한 물리적 위치(이음매/seam)다.
- ``Homing_Offset``은 서보 펌웨어에 write되어 있어 Present_Position raw 값에 이미
  반영된 상태로 읽힌다 (``simulation/mujoco/follower_safe_mapper.py``에서 동일하게
  확인된 사실). 그래서 이 모듈도 raw tick에서 homing_offset을 다시 빼지 않는다 -
  raw tick을 calibration의 range_min/range_max와 함께 그대로 위 공식에 넣는다.

이 이음매 때문에 "min <= value <= max" 같은 단순 선형 비교로는 이음매 근처에서의 실제
안전성을 보장할 수 없다 (예: 물리적으로 0.2° 움직였을 뿐인데 각도 표현상 -179.9°에서
+179.9°로 "점프"한 것처럼 보일 수 있다). 확실하지 않은 wrap 계산을 시도하는 대신, 이
모듈은 다음처럼 **이음매 자체를 보수적으로 회피**한다:

1. calibration 내부 안전 구간을 ``(-180+margin, 180-margin)``의 **선형** 구간으로만
   정의한다 (모듈러/wrap 계산을 전혀 하지 않는다).
2. 시작 위치가 이 내부 구간 밖(즉 이음매 부근, ``|deg| > 180-margin``)이면 계획
   전체를 BLOCKED 처리하고, 이동 방향 계산 자체를 시도하지 않는다 - 이음매 근처에서는
   "안전한 방향"이라는 개념 자체를 확정할 수 없기 때문이다.
3. 목표 각도가 ``[-180, 180]``을 벗어나거나 내부 구간을 벗어나면 그 스텝(및 전체
   계획)을 BLOCKED 처리한다 - wrap을 가정해서 반대편 raw tick으로 넘어가는 계산을
   하지 않는다.

margin 기본값은 15°다(calibration 양 끝에서 안쪽으로 15°, 요구사항 6번). "이 조사에서
확신이 서지 않는 부분은 armed를 구현하지 않는다"는 원칙에 따라, wrap 자체를 다루는
로직은 이 저장소에 없다 - armed writer가 실제로 필요해지면 그때 실물로 이음매 근처
동작을 재검증한 뒤 구현해야 한다.

``configs/generated/teleop_safe_ranges.json``의 wrist_roll 상태가 ``MARGIN_COLLAPSED``
이므로(margin 적용 후 안전 범위가 역전/소멸) 이 모듈은 그 파일의 역사적 텔레옵 범위를
전혀 사용하지 않는다 - 15° calibration margin이 유일한 안전 여유다. teleop 상태 문자열
자체는 (여기서 계산하지 않고) 호출부가 리포트에 그대로 실어서 "historical range
applied: false"와 함께 보여준다.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_JOINT = "wrist_roll"
MOTOR_MODEL = "sts3215"

# lerobot MODEL_RESOLUTION["sts3215"] (~/lerobot/src/lerobot/motors/feetech/tables.py).
DEFAULT_MOTOR_RESOLUTION = 4096

# calibration 양 끝에서 안쪽으로 확보하는 margin (요구사항 6번 기본 후보).
DEFAULT_CALIBRATION_MARGIN_DEG = 15.0

# 요구사항 3번(dry-run): 최대 이동량 1°, 단계 크기 0.1°. 이 두 값은 "상한"이며, 호출부가
# 이보다 더 보수적인(작은) 값을 요청하는 것은 허용하되 초과 요청은 거부한다.
MAX_ALLOWED_TOTAL_DELTA_DEG = 1.0
MAX_ALLOWED_STEP_SIZE_DEG = 0.1
DEFAULT_STEP_SIZE_DEG = 0.1
DEFAULT_TOTAL_DELTA_DEG = 1.0
DEFAULT_NUM_STEPS = 10  # 1.0 / 0.1

DIRECTIONS: tuple[str, ...] = ("positive", "negative")

PASS = "PASS"
BLOCKED = "BLOCKED"

# --- armed 단발(0.1°) 이동 전용 상수 --------------------------------------
#
# armed의 첫 실행은 dry-run처럼 "최대 1°까지 여러 스텝"이 아니라, 정확히 0.1° 한 번만
# 허용한다 (요구사항 3번). STS3215는 12bit(4096) 분해능이라 0.1°는 정수 tick으로
# 반올림된다: 0.1° * (4095/360) ≈ 1.1375 tick -> 반올림하면 1 tick. 안전하게 "최소
# 1 tick, 최대 2 tick"까지만 허용하고, 0 tick(반올림되어 사라짐)이나 3 tick 이상(예상보다
# 큰 이동)은 거부한다.
REQUIRED_ARMED_STEP_SIZE_DEG = 0.1
REQUIRED_ARMED_TOTAL_DELTA_DEG = 0.1
MIN_COMMAND_RAW_DELTA_TICKS = 1
MAX_COMMAND_RAW_DELTA_TICKS = 2

# expected-start 확인 허용 오차 (요구사항 4번 권장값).
EXPECTED_START_RAW_TOLERANCE_TICKS = 2
EXPECTED_START_DEG_TOLERANCE_DEG = 0.25

# readback 판정 기준 (요구사항 7~8번). 0.1° 명령(raw delta 1~2 tick)에 대해 readback
# 절대 이동량이 4 tick을 넘으면 "예상보다 과도한 이동"으로 본다 - 명령 자체(최대
# 2 tick)의 두 배 여유를 준 보수적 상한이다.
MAX_READBACK_ABS_RAW_DELTA_TICKS = 4

READBACK_PASS = "PASS"
READBACK_NO_MOTION = "NO_MOTION"
READBACK_DIRECTION_MISMATCH = "DIRECTION_MISMATCH"
READBACK_OVERSHOOT = "OVERSHOOT"
READBACK_FAILED = "READBACK_FAILED"
WRITE_FAILED = "WRITE_FAILED"


class PlannerConfigError(ValueError):
    """계획 파라미터 자체가 안전 정책(1°/0.1° 상한, direction 등)을 벗어났을 때 발생."""


@dataclass(frozen=True)
class CalibrationRangeDeg:
    """calibration raw range를 degree로 변환하고 margin을 적용한 결과."""

    raw_min: int
    raw_max: int
    motor_resolution: int
    margin_deg: float
    full_deg_min: float  # calibration 전체 범위 (margin 적용 전)
    full_deg_max: float
    inner_deg_min: float  # margin 적용된 안전 구간 (이음매 회피)
    inner_deg_max: float
    is_full_turn: bool  # raw_min==0 and raw_max==motor_resolution-1 (wrist_roll 특성)


@dataclass(frozen=True)
class PlanStep:
    index: int  # 1..N (N=요청한 총 이동량/step_size)
    delta_deg: float  # start 기준 부호 있는 누적 delta
    target_deg: float
    target_raw: int
    calibration_check: str  # PASS | BLOCKED - calibration 내부 안전 구간 통과 여부
    local_range_check: str  # PASS | BLOCKED - "시작 위치 기준 1° 이내" 통과 여부
    verdict: str  # PASS | BLOCKED - 이 스텝의 최종 판정

    def to_dict(self) -> dict:
        return {
            "step": self.index,
            "delta_deg": self.delta_deg,
            "target_deg": self.target_deg,
            "target_raw": self.target_raw,
            "calibration_check": self.calibration_check,
            "local_range_check": self.local_range_check,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class SingleJointPlan:
    joint: str
    direction: str
    start_deg: float
    start_raw: int
    requested_total_delta_deg: float
    step_size_deg: float
    calibration_range: CalibrationRangeDeg
    steps: tuple[PlanStep, ...]
    final_verdict: str  # PASS | BLOCKED
    block_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "joint": self.joint,
            "direction": self.direction,
            "start_position_deg": self.start_deg,
            "start_position_raw": self.start_raw,
            "requested_delta_deg": self.requested_total_delta_deg,
            "step_size_deg": self.step_size_deg,
            "calibration_range": {
                "raw_min": self.calibration_range.raw_min,
                "raw_max": self.calibration_range.raw_max,
                "motor_resolution": self.calibration_range.motor_resolution,
                "margin_deg": self.calibration_range.margin_deg,
                "full_range_deg": [self.calibration_range.full_deg_min, self.calibration_range.full_deg_max],
                "inner_safe_range_deg": [
                    self.calibration_range.inner_deg_min,
                    self.calibration_range.inner_deg_max,
                ],
                "is_full_turn": self.calibration_range.is_full_turn,
            },
            "planned_targets": [step.to_dict() for step in self.steps],
            "final_verdict": self.final_verdict,
            "block_reasons": list(self.block_reasons),
            "write_count": 0,
        }


# ---------------------------------------------------------------------------
# raw tick <-> degree 변환 (lerobot MotorsBus._normalize/_unnormalize와 동일한 공식)
# ---------------------------------------------------------------------------


def raw_to_degrees(
    raw: float, *, range_min: int, range_max: int, motor_resolution: int = DEFAULT_MOTOR_RESOLUTION
) -> float:
    """lerobot ``MotorsBus._normalize``의 DEGREES 분기와 동일 (clamp 없음)."""
    if range_min == range_max:
        raise PlannerConfigError("range_min과 range_max가 같습니다 (잘못된 calibration).")
    mid = (range_min + range_max) / 2.0
    max_res = motor_resolution - 1
    return (raw - mid) * 360.0 / max_res


def degrees_to_raw(
    deg: float, *, range_min: int, range_max: int, motor_resolution: int = DEFAULT_MOTOR_RESOLUTION
) -> int:
    """lerobot ``MotorsBus._unnormalize``의 DEGREES 분기와 동일 (clamp 없음).

    dry-run 계획에서 raw tick을 "예측"만 하기 위한 순수 계산이다 - 이 함수는 어떤
    write도 수행하지 않는다 (이름 그대로 int를 반환할 뿐).
    """
    if range_min == range_max:
        raise PlannerConfigError("range_min과 range_max가 같습니다 (잘못된 calibration).")
    mid = (range_min + range_max) / 2.0
    max_res = motor_resolution - 1
    return int(deg * max_res / 360.0 + mid)


def compute_calibration_range_deg(
    *,
    range_min: int,
    range_max: int,
    motor_resolution: int = DEFAULT_MOTOR_RESOLUTION,
    margin_deg: float = DEFAULT_CALIBRATION_MARGIN_DEG,
) -> CalibrationRangeDeg:
    if margin_deg < 0:
        raise PlannerConfigError(f"margin_deg는 0 이상이어야 합니다: {margin_deg}")

    deg_at_min = raw_to_degrees(range_min, range_min=range_min, range_max=range_max, motor_resolution=motor_resolution)
    deg_at_max = raw_to_degrees(range_max, range_min=range_min, range_max=range_max, motor_resolution=motor_resolution)
    full_lo, full_hi = (deg_at_min, deg_at_max) if deg_at_min <= deg_at_max else (deg_at_max, deg_at_min)

    is_full_turn = range_min == 0 and range_max == motor_resolution - 1

    return CalibrationRangeDeg(
        raw_min=range_min,
        raw_max=range_max,
        motor_resolution=motor_resolution,
        margin_deg=margin_deg,
        full_deg_min=full_lo,
        full_deg_max=full_hi,
        inner_deg_min=full_lo + margin_deg,
        inner_deg_max=full_hi - margin_deg,
        is_full_turn=is_full_turn,
    )


# ---------------------------------------------------------------------------
# dry-run 계획
# ---------------------------------------------------------------------------


def _validate_plan_inputs(*, direction: str, requested_total_delta_deg: float, step_size_deg: float) -> None:
    if direction not in DIRECTIONS:
        raise PlannerConfigError(f"direction은 {DIRECTIONS} 중 하나여야 합니다: {direction!r}")
    if not (0 < step_size_deg <= MAX_ALLOWED_STEP_SIZE_DEG):
        raise PlannerConfigError(
            f"step_size_deg는 0보다 크고 {MAX_ALLOWED_STEP_SIZE_DEG}° 이하여야 합니다: {step_size_deg}"
        )
    if not (0 < requested_total_delta_deg <= MAX_ALLOWED_TOTAL_DELTA_DEG):
        raise PlannerConfigError(
            f"requested_total_delta_deg는 0보다 크고 {MAX_ALLOWED_TOTAL_DELTA_DEG}° 이하여야 합니다: "
            f"{requested_total_delta_deg}"
        )
    steps_f = requested_total_delta_deg / step_size_deg
    if abs(steps_f - round(steps_f)) > 1e-6:
        raise PlannerConfigError(
            f"requested_total_delta_deg({requested_total_delta_deg})가 "
            f"step_size_deg({step_size_deg})로 나누어 떨어지지 않습니다."
        )


def build_dry_run_plan(
    *,
    start_deg: float,
    start_raw: int,
    direction: str,
    range_min: int,
    range_max: int,
    motor_resolution: int = DEFAULT_MOTOR_RESOLUTION,
    margin_deg: float = DEFAULT_CALIBRATION_MARGIN_DEG,
    requested_total_delta_deg: float = DEFAULT_TOTAL_DELTA_DEG,
    step_size_deg: float = DEFAULT_STEP_SIZE_DEG,
) -> SingleJointPlan:
    """wrist_roll dry-run 이동 계획을 계산한다 (실제 write 없음, 순수 계산).

    한 방향(``direction``)으로만 계산한다 - 계획이 도중에 BLOCKED되어도 반대 방향으로
    자동 전환하지 않는다 (요구사항 7번). 중간에 calibration 내부 안전 구간을 벗어나는
    스텝이 하나라도 있으면 ``final_verdict``는 전체 계획에 대해 BLOCKED가 된다
    (요구사항 7번 "전체 계획을 BLOCKED 처리").
    """
    _validate_plan_inputs(
        direction=direction, requested_total_delta_deg=requested_total_delta_deg, step_size_deg=step_size_deg
    )

    cal_range = compute_calibration_range_deg(
        range_min=range_min, range_max=range_max, motor_resolution=motor_resolution, margin_deg=margin_deg
    )

    block_reasons: list[str] = []
    sign = 1.0 if direction == "positive" else -1.0
    n_steps = round(requested_total_delta_deg / step_size_deg)

    start_outside_inner_band = not (cal_range.inner_deg_min <= start_deg <= cal_range.inner_deg_max)
    if start_outside_inner_band:
        block_reasons.append(
            "시작 위치가 calibration 내부 안전 구간(이음매로부터 margin 확보) 밖에 있어 "
            "안전한 이동 방향을 계산할 수 없습니다 - 전체 계획을 BLOCKED 처리합니다."
        )

    steps: list[PlanStep] = []
    for i in range(1, n_steps + 1):
        delta = sign * step_size_deg * i
        target_deg = start_deg + delta
        target_raw = degrees_to_raw(
            target_deg, range_min=range_min, range_max=range_max, motor_resolution=motor_resolution
        )

        within_physical_bounds = -180.0 <= target_deg <= 180.0
        within_inner_band = cal_range.inner_deg_min <= target_deg <= cal_range.inner_deg_max
        calibration_ok = (not start_outside_inner_band) and within_physical_bounds and within_inner_band
        local_ok = abs(delta) <= requested_total_delta_deg + 1e-9

        calibration_check = PASS if calibration_ok else BLOCKED
        local_range_check = PASS if local_ok else BLOCKED
        verdict = PASS if (calibration_check == PASS and local_range_check == PASS) else BLOCKED

        steps.append(
            PlanStep(
                index=i,
                delta_deg=delta,
                target_deg=target_deg,
                target_raw=target_raw,
                calibration_check=calibration_check,
                local_range_check=local_range_check,
                verdict=verdict,
            )
        )

    any_step_blocked = any(s.verdict == BLOCKED for s in steps)
    if any_step_blocked and not start_outside_inner_band:
        block_reasons.append(
            "하나 이상의 계획된 스텝이 calibration 내부 안전 구간 또는 1° 로컬 한계를 벗어났습니다."
        )

    final_verdict = BLOCKED if (start_outside_inner_band or any_step_blocked) else PASS

    return SingleJointPlan(
        joint=TARGET_JOINT,
        direction=direction,
        start_deg=start_deg,
        start_raw=start_raw,
        requested_total_delta_deg=requested_total_delta_deg,
        step_size_deg=step_size_deg,
        calibration_range=cal_range,
        steps=tuple(steps),
        final_verdict=final_verdict,
        block_reasons=tuple(block_reasons),
    )


# ---------------------------------------------------------------------------
# armed 단발(0.1°) 이동 계획 - 순수 계산 (write 없음)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmedCommandPlan:
    """armed writer가 실제로 write하기 *전에* 확인해야 할 모든 조건의 계산 결과.

    이 데이터클래스 자체는 write를 수행하지 않는다 - ``hardware/safety/single_joint_writer.py``
    가 이 계획의 ``final_verdict``가 ``PASS``일 때만 실제 write를 시도한다.
    """

    joint: str
    direction: str
    start_deg: float
    start_raw: int
    target_deg: float
    target_raw: int
    command_raw_delta: int  # target_raw - start_raw (부호 있음)
    requested_delta_deg: float
    step_size_deg: float
    calibration_range: CalibrationRangeDeg
    checks: dict  # 체크 이름 -> PASS/BLOCKED
    final_verdict: str
    block_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "joint": self.joint,
            "direction": self.direction,
            "start_deg": self.start_deg,
            "start_raw": self.start_raw,
            "target_deg": self.target_deg,
            "target_raw": self.target_raw,
            "command_raw_delta": self.command_raw_delta,
            "requested_delta_deg": self.requested_delta_deg,
            "step_size_deg": self.step_size_deg,
            "calibration_range": {
                "raw_min": self.calibration_range.raw_min,
                "raw_max": self.calibration_range.raw_max,
                "motor_resolution": self.calibration_range.motor_resolution,
                "margin_deg": self.calibration_range.margin_deg,
                "full_range_deg": [self.calibration_range.full_deg_min, self.calibration_range.full_deg_max],
                "inner_safe_range_deg": [
                    self.calibration_range.inner_deg_min,
                    self.calibration_range.inner_deg_max,
                ],
                "is_full_turn": self.calibration_range.is_full_turn,
            },
            "checks": dict(self.checks),
            "final_verdict": self.final_verdict,
            "block_reasons": list(self.block_reasons),
        }


def build_armed_single_step_plan(
    *,
    start_deg: float,
    start_raw: int,
    direction: str,
    range_min: int,
    range_max: int,
    motor_resolution: int = DEFAULT_MOTOR_RESOLUTION,
    margin_deg: float = DEFAULT_CALIBRATION_MARGIN_DEG,
    requested_delta_deg: float = REQUIRED_ARMED_TOTAL_DELTA_DEG,
    step_size_deg: float = REQUIRED_ARMED_STEP_SIZE_DEG,
) -> ArmedCommandPlan:
    """armed 첫 실행 전용: 정확히 0.1° 한 스텝짜리 계획을 계산한다 (write 없음).

    dry-run의 ``build_dry_run_plan``과 달리 스텝이 하나뿐이고, raw delta가 1~2 tick
    범위 안인지까지 확인한다(요구사항 3번). ``requested_delta_deg``/``step_size_deg``가
    정확히 0.1이 아니면 즉시 :class:`PlannerConfigError`를 던진다 - "1° 초과"가 아니라
    "0.1이 아니면 무조건 거부"라는 더 엄격한 armed 전용 규칙이다.
    """
    if direction not in DIRECTIONS:
        raise PlannerConfigError(f"direction은 {DIRECTIONS} 중 하나여야 합니다: {direction!r}")
    if abs(requested_delta_deg - REQUIRED_ARMED_TOTAL_DELTA_DEG) > 1e-9 or abs(
        step_size_deg - REQUIRED_ARMED_STEP_SIZE_DEG
    ) > 1e-9:
        raise PlannerConfigError(
            f"armed 첫 실행은 정확히 {REQUIRED_ARMED_TOTAL_DELTA_DEG}° 단발 이동만 허용합니다 "
            f"(요청: requested_delta_deg={requested_delta_deg}, step_size_deg={step_size_deg})."
        )

    cal_range = compute_calibration_range_deg(
        range_min=range_min, range_max=range_max, motor_resolution=motor_resolution, margin_deg=margin_deg
    )

    checks: dict[str, str] = {}
    block_reasons: list[str] = []

    sign = 1.0 if direction == "positive" else -1.0
    delta_deg = sign * step_size_deg
    target_deg = start_deg + delta_deg
    target_raw = degrees_to_raw(
        target_deg, range_min=range_min, range_max=range_max, motor_resolution=motor_resolution
    )
    command_raw_delta = target_raw - start_raw

    # 1) 시작 위치가 이음매 회피 안전 구간 안인가
    start_ok = cal_range.inner_deg_min <= start_deg <= cal_range.inner_deg_max
    checks["seam_avoidance_start_check"] = PASS if start_ok else BLOCKED
    if not start_ok:
        block_reasons.append(
            f"시작 위치({start_deg:.3f}°)가 이음매 회피 안전 구간 "
            f"[{cal_range.inner_deg_min:.3f}, {cal_range.inner_deg_max:.3f}] 밖입니다."
        )

    # 2) 목표 위치가 calibration 내부 안전 구간 + 물리적 [-180,180] 안인가 (시작이 밖이면
    #    방향 계산 자체가 무의미하므로 무조건 BLOCKED로 처리한다).
    target_within_physical = -180.0 <= target_deg <= 180.0
    target_within_inner_band = cal_range.inner_deg_min <= target_deg <= cal_range.inner_deg_max
    calibration_ok = start_ok and target_within_physical and target_within_inner_band
    checks["calibration_check"] = PASS if calibration_ok else BLOCKED
    if not calibration_ok and start_ok:
        block_reasons.append(
            f"목표 위치({target_deg:.3f}°)가 calibration 내부 안전 구간 또는 물리적 범위를 벗어납니다."
        )

    # 3) 목표 raw tick이 0~motor_resolution-1 안인가 (물리적 존재 범위)
    target_raw_in_bounds = 0 <= target_raw <= motor_resolution - 1
    checks["target_raw_bounds_check"] = PASS if target_raw_in_bounds else BLOCKED
    if not target_raw_in_bounds:
        block_reasons.append(f"목표 raw tick({target_raw})이 [0, {motor_resolution - 1}] 범위를 벗어납니다.")

    # 4) 명령 raw delta가 1~2 tick 안인가 (0 tick=반올림 소멸, 3 tick 이상=과도한 이동)
    raw_delta_abs = abs(command_raw_delta)
    raw_delta_ok = MIN_COMMAND_RAW_DELTA_TICKS <= raw_delta_abs <= MAX_COMMAND_RAW_DELTA_TICKS
    checks["command_raw_delta_check"] = PASS if raw_delta_ok else BLOCKED
    if not raw_delta_ok:
        if raw_delta_abs == 0:
            block_reasons.append(
                f"요청한 {step_size_deg}°가 raw tick 0으로 반올림되어(변화 없음) 명령을 거부합니다."
            )
        else:
            block_reasons.append(
                f"명령 raw delta({command_raw_delta})의 절대값이 허용 범위 "
                f"[{MIN_COMMAND_RAW_DELTA_TICKS}, {MAX_COMMAND_RAW_DELTA_TICKS}] tick을 벗어납니다."
            )

    final_verdict = PASS if all(v == PASS for v in checks.values()) else BLOCKED

    return ArmedCommandPlan(
        joint=TARGET_JOINT,
        direction=direction,
        start_deg=start_deg,
        start_raw=start_raw,
        target_deg=target_deg,
        target_raw=target_raw,
        command_raw_delta=command_raw_delta,
        requested_delta_deg=requested_delta_deg,
        step_size_deg=step_size_deg,
        calibration_range=cal_range,
        checks=checks,
        final_verdict=final_verdict,
        block_reasons=tuple(block_reasons),
    )


def check_expected_start_matches(
    *,
    measured_raw: int,
    measured_deg: float,
    expected_raw: int | None,
    expected_deg: float | None,
    raw_tolerance_ticks: int = EXPECTED_START_RAW_TOLERANCE_TICKS,
    deg_tolerance_deg: float = EXPECTED_START_DEG_TOLERANCE_DEG,
) -> tuple[str, str | None]:
    """사용자가 CLI로 입력한 "현재 위치를 직접 확인했다"는 값과 실측값을 비교한다.

    ``expected_raw``/``expected_deg`` 중 최소 하나는 반드시 있어야 한다 - 둘 다 없으면
    무조건 BLOCKED (요구사항 4번: "두 확인 중 하나만 있거나 expected start가 없으면
    write를 절대 하지 않는다"). 둘 다 있으면 둘 다 허용 오차 안이어야 PASS다.

    Returns:
        (PASS | BLOCKED, 실패 사유 또는 None)
    """
    if expected_raw is None and expected_deg is None:
        return BLOCKED, (
            "--expected-start-raw/--expected-start-deg 중 어느 것도 제공되지 않았습니다 - "
            "사용자가 현재 위치를 직접 확인했다는 근거가 없어 write를 거부합니다."
        )

    if expected_raw is not None:
        raw_diff = abs(measured_raw - expected_raw)
        if raw_diff > raw_tolerance_ticks:
            return BLOCKED, (
                f"실측 raw({measured_raw})와 --expected-start-raw({expected_raw})의 차이"
                f"({raw_diff} tick)가 허용 오차(±{raw_tolerance_ticks} tick)를 초과합니다."
            )

    if expected_deg is not None:
        deg_diff = abs(measured_deg - expected_deg)
        if deg_diff > deg_tolerance_deg:
            return BLOCKED, (
                f"실측 각도({measured_deg:.4f}°)와 --expected-start-deg({expected_deg}°)의 차이"
                f"({deg_diff:.4f}°)가 허용 오차(±{deg_tolerance_deg}°)를 초과합니다."
            )

    return PASS, None


def classify_readback(*, direction: str, command_raw_delta: int, actual_raw_delta: int) -> str:
    """write 후 readback한 raw delta를 명령과 비교해 판정한다 (요구사항 7~8번).

    - 변화가 0 tick: ``NO_MOTION``
    - 명령과 반대 부호로 움직임: ``DIRECTION_MISMATCH``
    - 절대 이동량이 :data:`MAX_READBACK_ABS_RAW_DELTA_TICKS` 초과: ``OVERSHOOT``
    - 그 외(방향 일치 + 허용량 이내): ``PASS``

    이 함수는 어떤 추가 write/재시도도 트리거하지 않는다 - 판정 문자열만 반환한다.
    """
    if actual_raw_delta == 0:
        return READBACK_NO_MOTION

    command_sign = 1 if command_raw_delta > 0 else -1 if command_raw_delta < 0 else 0
    actual_sign = 1 if actual_raw_delta > 0 else -1

    if command_sign != 0 and actual_sign != command_sign:
        return READBACK_DIRECTION_MISMATCH

    if abs(actual_raw_delta) > MAX_READBACK_ABS_RAW_DELTA_TICKS:
        return READBACK_OVERSHOOT

    return READBACK_PASS
