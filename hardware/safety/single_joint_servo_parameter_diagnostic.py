"""wrist_roll(motor id 5) STS3215 설정 레지스터 read-only 진단.

1~2 tick(≈0.09~0.18°) ``Goal_Position`` 명령이 ``Goal_Position``에는 latch되지만
``Present_Position``이 전혀 움직이지 않는(``COMMAND_LATCHED_BUT_NO_MOTION``) 현상의
설정(configuration)상 원인을 조사하기 위한 순수 read-only 진단 모듈이다. write는 전혀
하지 않는다.

## 조사 근거 (설치된 lerobot/Feetech SDK에서 확인, 추측 없음)

``~/lerobot/src/lerobot/motors/feetech/tables.py``의 ``STS_SMS_SERIES_CONTROL_TABLE``
(sts3215가 쓰는 테이블)에서 요구사항 1번이 요청한 개념에 대응하는 **실제 register
이름**을 확인했다::

    "CW_Dead_Zone": (26, 1)                # EPROM
    "CCW_Dead_Zone": (27, 1)                # EPROM
    "Minimum_Startup_Force": (24, 2)        # EPROM
    "Operating_Mode": (33, 1)               # EPROM
    "Acceleration": (41, 1)                 # SRAM (휘발성 - 전원이 끊기면 초기화)
    "Maximum_Acceleration": (85, 1)         # Factory
    "Goal_Velocity": (46, 2)                # SRAM (휘발성)
    "Maximum_Velocity_Limit": (84, 1)       # Factory
    "Moving_Velocity_Threshold": (80, 1)    # Factory (tables.py에 "Moving_Velocity"와
                                             #   동일 주소로 중복 정의됨 - 이 모듈은 더
                                             #   설명적인 이름을 쓴다)
    "Torque_Limit": (48, 2)                 # SRAM (휘발성)
    "Max_Torque_Limit": (16, 2)             # EPROM
    "P_Coefficient": (21, 1)                # EPROM (위치 루프 P)
    "I_Coefficient": (23, 1)                # EPROM (위치 루프 I)
    "D_Coefficient": (22, 1)                # EPROM (위치 루프 D)
    "Lock": (55, 1)                         # SRAM
    "Angular_Resolution": (30, 1)           # EPROM

**요구사항이 예시로 든 "Position P/I/D"**는 ``P_Coefficient``/``I_Coefficient``/
``D_Coefficient``로 확인했다 - 속도 루프 PID(``Velocity_closed_loop_P/I_..._coefficient``,
37/39번지)도 테이블에 있지만 이번 진단은 위치 제어 경로에 집중하므로 위치 루프 PID만
읽는다.

### Operating_Mode 값의 의미는 코드로 확정됨 (``feetech.py`` ``OperatingMode`` Enum)

    class OperatingMode(Enum):
        POSITION = 0   # position servo mode
        VELOCITY = 1   # constant speed mode (파라미터 0x2e로 제어, bit15=방향)
        PWM = 2        # PWM open-loop speed regulation (파라미터 0x2c로 제어, bit11=방향)
        STEP = 3       # step servo mode (파라미터 0x2a로 진행량 표시, bit15=방향)

``SOFollower.configure()``(``bus.torque_disabled()`` 안에서 ``Operating_Mode``를
``POSITION.value``로 write)가 정상 경로에서 이 값을 0으로 맞춘다 - 하지만 이번 진단과
이전 armed 실행 모두 ``SOFollower.connect()``/``configure()``를 거치지 않았으므로,
현재 실제 값이 0인지 이 진단으로 직접 확인해야 한다(추측 금지).

### Acceleration/Goal_Velocity/Torque_Limit/Lock 등은 "SRAM"(휘발성) 레지스터 - 코드로 확인됨

``configure_motors()`` (``feetech.py`` 209~217행)가 실제로 다음을 수행한다::

    self.write("Maximum_Acceleration", motor, maximum_acceleration)  # 기본 254
    self.write("Acceleration", motor, acceleration)                  # 기본 254
    # 주석: "Set 'Maximum_Acceleration' to 254 to speedup acceleration and
    #        deceleration of the motors."

즉 이 저장소의 정상 데이터 수집 경로(``SOFollower.connect()`` 경유)는
``Acceleration``/``Maximum_Acceleration``을 254(빠름)로 맞춰 왔을 가능성이 높다. 하지만
``Acceleration``/``Goal_Velocity``/``Torque_Limit``/``Lock``은 ``tables.py``의 "# SRAM"
주석 구획에 있는 **휘발성** 레지스터라, 서보 전원이 한 번이라도 끊겼다면 그 값들이
초기화됐을 수 있다 - 이 저장소의 armed writer/register-diagnostic 모두
``configure_motors()``를 호출하지 않으므로(의도적으로, torque/PID/가속도를 바꾸지
않기 위해), 현재 SRAM 값이 무엇인지는 추측하지 않고 이번 진단에서 직접 읽는다.

### 중요: ``Acceleration == 0``은 "가속 없음/비활성"이 아니라 별도 정의된 특수값이다 (코드로 확인됨)

``tables.py`` 100행에 다음 주석이 그대로 있다(다음 라운드 조사에서 재확인)::

    "Acceleration_Multiplier ": (86, 1),  # Acceleration multiplier in effect when acceleration is 0

**주의**: 이 키 문자열은 끝에 공백이 하나 포함되어 있다(``"Acceleration_Multiplier "``,
lerobot 원본 코드의 오탈자로 보이나 이 저장소가 임의로 고칠 수 없는 실제 dict key다) -
``bus.read()``로 이 레지스터를 읽으려면 공백까지 정확히 일치하는 문자열을 써야 한다.

이 주석은 **"Acceleration=0일 때는 Acceleration_Multiplier(86번지) 값이 대신
적용된다"**는 것을 코드 수준에서 확정해 준다. 즉 ``Acceleration=0``은 고장/무효값이
아니라 "다른 레지스터로 제어를 위임한다"는 **정의된 대체 모드**일 가능성이 높다 -
"0=가속 불가능=NO_MOTION의 직접 원인"이라는 단순한 결론을 code 근거로 반박한다. 다만
``Acceleration_Multiplier`` 자체의 정확한 스케일/단위 역시 코드에 문서화되어 있지 않고,
그 값이 얼마인지도 이번 조사에서 실측하기 전까지는 알 수 없었다 - 이 모듈은 그래서 이
레지스터도 함께 읽어 참고 정보로 보고한다(``ServoParameterSnapshot.acceleration_multiplier``).

``disable_torque()``/``enable_torque()``(feetech.py 291~304행)가 각각 ``Lock``을
0/1로 write하는 것도 확인했다 - ``Lock``이 이번 진단에서 무엇을 의미하는지 참고용으로만
보고한다(Goal_Position은 이미 latch가 확인되었으므로 Lock이 그 자체를 막고 있지는
않다는 것은 이미 알고 있다).

### 단위/스케일이 code-level로 확정되지 않은 레지스터

``CW_Dead_Zone``/``CCW_Dead_Zone``/``Minimum_Startup_Force``/``P_Coefficient``/
``I_Coefficient``/``D_Coefficient``/``Angular_Resolution``/``Torque_Limit``/
``Max_Torque_Limit``/``Moving_Velocity_Threshold``/``Maximum_Velocity_Limit``은
``tables.py``에 주소/바이트수만 정의되어 있고, 이 값이 정확히 어떤 물리 단위(예: 몇
tick당 1 dead-zone 단위인지, force가 0~1000 스케일인지 등)로 해석되는지 디코딩하는
코드가 이 저장소/설치된 SDK 어디에도 없다. 그래서 이 모듈은 이 레지스터들의 **raw
값만** 보고하고, 단위/스케일은 ``UNKNOWN``으로 명시한다 - 판정 함수도 "0이 아니다/이다"
같은 raw 값 자체의 특징만으로 ``LIKELY`` 라벨을 매길 뿐, 정확한 tick 환산으로 확정하지
않는다.

이 모듈에는 write에 해당하는 함수/메서드가 하나도 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hardware.safety.single_joint_bus import WristRollCalibration, build_wrist_roll_bus
from hardware.safety.single_joint_test_planner import (
    DEFAULT_CALIBRATION_MARGIN_DEG,
    DEFAULT_MOTOR_RESOLUTION,
    MOTOR_MODEL,
    TARGET_JOINT,
    compute_calibration_range_deg,
    raw_to_degrees,
)

__all__ = [
    "WristRollCalibration",
    "STATE_REGISTERS",
    "PARAMETER_REGISTERS",
    "ACCELERATION_MULTIPLIER_REGISTER_NAME",
    "REGISTER_HYPOTHESIS_MAP",
    "ServoParameterSnapshot",
    "ServoParameterVerdict",
    "classify_servo_parameters",
    "NextStepCandidate",
    "compute_next_step_candidates",
    "ServoParameterDiagnosticInspector",
]

# 요구사항 3번: 같은 실행에서 다시 읽는 "현재 상태" 레지스터 (모두 STS_SMS_SERIES_CONTROL_TABLE에 실재).
STATE_REGISTERS: tuple[str, ...] = ("Torque_Enable", "Goal_Position", "Present_Position", "Moving", "Status")

# 요구사항 1번이 요청한 개념에 대응하는 실제 설정 레지스터 (전부 tables.py에 실재 확인됨).
PARAMETER_REGISTERS: tuple[str, ...] = (
    "CW_Dead_Zone",
    "CCW_Dead_Zone",
    "Minimum_Startup_Force",
    "Operating_Mode",
    "Acceleration",
    "Maximum_Acceleration",
    "Goal_Velocity",
    "Maximum_Velocity_Limit",
    "Moving_Velocity_Threshold",
    "Torque_Limit",
    "Max_Torque_Limit",
    "P_Coefficient",
    "I_Coefficient",
    "D_Coefficient",
    "Lock",
    "Angular_Resolution",
    # tables.py 100행 원문 그대로 (끝에 공백 포함 - lerobot 원본의 dict key 오탈자, 이
    # 저장소가 임의로 고칠 수 없다). 주석: "Acceleration multiplier in effect when
    # acceleration is 0" - Acceleration=0일 때 이 레지스터가 대신 적용된다는 것이 코드로
    # 확인된 유일한 문서다.
    "Acceleration_Multiplier ",
)

ACCELERATION_MULTIPLIER_REGISTER_NAME = "Acceleration_Multiplier "


class ServoParameterVerdict:
    DEAD_ZONE_LIKELY = "DEAD_ZONE_LIKELY"
    STARTUP_FORCE_THRESHOLD_LIKELY = "STARTUP_FORCE_THRESHOLD_LIKELY"
    CONTROL_MODE_RESTRICTION = "CONTROL_MODE_RESTRICTION"
    VELOCITY_OR_ACCELERATION_RESTRICTION = "VELOCITY_OR_ACCELERATION_RESTRICTION"
    TORQUE_LIMIT_RESTRICTION = "TORQUE_LIMIT_RESTRICTION"
    NO_CONFIGURATION_CAUSE_FOUND = "NO_CONFIGURATION_CAUSE_FOUND"
    UNKNOWN = "UNKNOWN"

    ALL = (
        DEAD_ZONE_LIKELY,
        STARTUP_FORCE_THRESHOLD_LIKELY,
        CONTROL_MODE_RESTRICTION,
        VELOCITY_OR_ACCELERATION_RESTRICTION,
        TORQUE_LIMIT_RESTRICTION,
        NO_CONFIGURATION_CAUSE_FOUND,
        UNKNOWN,
    )


# 각 레지스터가 어떤 가설(verdict)의 근거로 쓰이는지 - 리포트/테스트에서 참조용.
REGISTER_HYPOTHESIS_MAP: dict[str, str] = {
    "CW_Dead_Zone": ServoParameterVerdict.DEAD_ZONE_LIKELY,
    "CCW_Dead_Zone": ServoParameterVerdict.DEAD_ZONE_LIKELY,
    "Minimum_Startup_Force": ServoParameterVerdict.STARTUP_FORCE_THRESHOLD_LIKELY,
    "Operating_Mode": ServoParameterVerdict.CONTROL_MODE_RESTRICTION,
    "Acceleration": ServoParameterVerdict.VELOCITY_OR_ACCELERATION_RESTRICTION,
    "Maximum_Acceleration": ServoParameterVerdict.VELOCITY_OR_ACCELERATION_RESTRICTION,
    "Goal_Velocity": ServoParameterVerdict.VELOCITY_OR_ACCELERATION_RESTRICTION,
    "Maximum_Velocity_Limit": ServoParameterVerdict.VELOCITY_OR_ACCELERATION_RESTRICTION,
    "Moving_Velocity_Threshold": ServoParameterVerdict.VELOCITY_OR_ACCELERATION_RESTRICTION,
    "Torque_Limit": ServoParameterVerdict.TORQUE_LIMIT_RESTRICTION,
    "Max_Torque_Limit": ServoParameterVerdict.TORQUE_LIMIT_RESTRICTION,
    # 나머지(P/I/D_Coefficient, Lock, Angular_Resolution)는 특정 verdict에 직접 대응하지
    # 않는 정보성(informational) 레지스터다 - 참고 보고만 하고 판정에 단독으로 쓰지 않는다.
}

# lerobot feetech.py의 OperatingMode Enum과 동일 (코드로 확정된 값).
OPERATING_MODE_POSITION = 0


@dataclass(frozen=True)
class ServoParameterSnapshot:
    """한 시점에 읽은 wrist_roll 상태 + 설정 레지스터 값들 (전부 raw)."""

    # -- 상태 (요구사항 3번, 다시 읽음) --
    torque_enable: int | None
    goal_position_raw: int | None
    present_position_raw: int | None
    moving: int | None
    status_raw: int | None

    # -- 설정 파라미터 (요구사항 1~2번) --
    cw_dead_zone: int | None = None
    ccw_dead_zone: int | None = None
    minimum_startup_force: int | None = None
    operating_mode: int | None = None
    acceleration: int | None = None
    maximum_acceleration: int | None = None
    goal_velocity: int | None = None
    maximum_velocity_limit: int | None = None
    moving_velocity_threshold: int | None = None
    torque_limit: int | None = None
    max_torque_limit: int | None = None
    p_coefficient: int | None = None
    i_coefficient: int | None = None
    d_coefficient: int | None = None
    lock: int | None = None
    angular_resolution: int | None = None
    acceleration_multiplier: int | None = None  # Acceleration==0일 때 대신 적용되는 값 (tables.py 주석 근거)

    read_errors: dict = field(default_factory=dict)
    unavailable_registers: tuple[str, ...] = ()  # 설치된 control table에 없어 시도조차 안 한 이름

    def to_dict(self) -> dict:
        return {
            "state": {
                "Torque_Enable": self.torque_enable,
                "Goal_Position": self.goal_position_raw,
                "Present_Position": self.present_position_raw,
                "Moving": self.moving,
                "Status": self.status_raw,
            },
            "parameters": {
                "CW_Dead_Zone": self.cw_dead_zone,
                "CCW_Dead_Zone": self.ccw_dead_zone,
                "Minimum_Startup_Force": self.minimum_startup_force,
                "Operating_Mode": self.operating_mode,
                "Acceleration": self.acceleration,
                "Maximum_Acceleration": self.maximum_acceleration,
                "Goal_Velocity": self.goal_velocity,
                "Maximum_Velocity_Limit": self.maximum_velocity_limit,
                "Moving_Velocity_Threshold": self.moving_velocity_threshold,
                "Torque_Limit": self.torque_limit,
                "Max_Torque_Limit": self.max_torque_limit,
                "P_Coefficient": self.p_coefficient,
                "I_Coefficient": self.i_coefficient,
                "D_Coefficient": self.d_coefficient,
                "Lock": self.lock,
                "Angular_Resolution": self.angular_resolution,
                "Acceleration_Multiplier": self.acceleration_multiplier,
            },
            "units_confirmed_by_code": {
                "Operating_Mode": "lerobot feetech.py OperatingMode Enum (POSITION=0/VELOCITY=1/PWM=2/STEP=3)",
                "Torque_Enable": "lerobot feetech.py TorqueMode Enum (ENABLED=1/DISABLED=0)",
                "Acceleration=0": (
                    "tables.py 100행 주석 'Acceleration multiplier in effect when acceleration is 0' - "
                    "Acceleration=0은 무효값이 아니라 Acceleration_Multiplier가 대신 적용되는 정의된 상태."
                ),
            },
            "units_unknown_note": (
                "CW_Dead_Zone/CCW_Dead_Zone/Minimum_Startup_Force/P_Coefficient/I_Coefficient/"
                "D_Coefficient/Angular_Resolution/Torque_Limit/Max_Torque_Limit/"
                "Moving_Velocity_Threshold/Maximum_Velocity_Limit의 정확한 물리 단위/스케일은 "
                "설치된 lerobot/Feetech SDK 코드에 디코딩 로직이 없어 raw 값만 신뢰할 수 있습니다."
            ),
            "read_errors": dict(self.read_errors),
            "unavailable_registers": list(self.unavailable_registers),
        }


def classify_servo_parameters(snapshot: ServoParameterSnapshot) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """읽은 스냅샷만으로 판정한다 - 순수 계산, 하드웨어/파일 접근 없음.

    요구사항 4번대로 여러 ``LIKELY`` 라벨이 동시에 나올 수 있다(단일 확정 원인을
    강제하지 않는다). 근거가 raw 값의 "0 여부"뿐이라 전부 ``LIKELY`` 등급이며, 이
    함수는 어떤 것도 ``CONFIRMED``로 격상하지 않는다(요구사항: "원인을 과하게 확정하지
    말 것").
    """

    verdicts: list[str] = []
    reasons: list[str] = []

    required_state = (snapshot.torque_enable, snapshot.goal_position_raw, snapshot.present_position_raw, snapshot.moving)
    if any(v is None for v in required_state):
        reasons.append("필수 상태 레지스터(Torque_Enable/Goal_Position/Present_Position/Moving) 일부를 읽지 못했습니다.")
        return (ServoParameterVerdict.UNKNOWN,), tuple(reasons)

    # Operating_Mode: 코드로 확정된 의미 (OperatingMode Enum).
    if snapshot.operating_mode is not None and snapshot.operating_mode != OPERATING_MODE_POSITION:
        verdicts.append(ServoParameterVerdict.CONTROL_MODE_RESTRICTION)
        reasons.append(
            f"Operating_Mode={snapshot.operating_mode} (POSITION=0이 아님 - lerobot OperatingMode Enum 기준 "
            "확정된 의미). position 제어 모드가 아니면 Goal_Position write가 예상대로 해석되지 않을 수 있습니다."
        )
    elif snapshot.operating_mode is None:
        reasons.append("Operating_Mode를 읽지 못해 CONTROL_MODE_RESTRICTION 여부를 판단할 수 없습니다.")

    # Dead zone: 단위 미확인, raw != 0이면 LIKELY.
    dead_zone_values = [v for v in (snapshot.cw_dead_zone, snapshot.ccw_dead_zone) if v is not None]
    if dead_zone_values and any(v > 0 for v in dead_zone_values):
        verdicts.append(ServoParameterVerdict.DEAD_ZONE_LIKELY)
        reasons.append(
            f"CW_Dead_Zone={snapshot.cw_dead_zone}, CCW_Dead_Zone={snapshot.ccw_dead_zone} (0이 아님) - "
            "정확한 tick 환산 계수는 코드로 확인되지 않았으나, 0이 아닌 데드존 설정이 존재함을 시사합니다."
        )

    # Minimum_Startup_Force: 단위 미확인, raw != 0이면 LIKELY.
    if snapshot.minimum_startup_force is not None and snapshot.minimum_startup_force > 0:
        verdicts.append(ServoParameterVerdict.STARTUP_FORCE_THRESHOLD_LIKELY)
        reasons.append(
            f"Minimum_Startup_Force={snapshot.minimum_startup_force} (0이 아님) - 정확한 단위는 코드로 "
            "확인되지 않았으나, 작은 위치 오차에서 구동을 시작하지 않을 가능성을 시사합니다."
        )

    # 가속도: SRAM(휘발성) 레지스터 - 0이면 LIKELY (configure_motors()가 정상적으로
    # Acceleration/Maximum_Acceleration=254를 설정하는 코드 경로가 확인되었으므로, 그
    # 경로를 타지 않은 이번 세션에서 0이라면 유의미한 신호다). Goal_Velocity는
    # OperatingMode.VELOCITY 모드에서만 쓰이는 값(코드로 확인됨)이라 POSITION 모드에서는
    # 0이 정상일 수 있으므로 이 판정에서 제외하고 참고 정보로만 보고한다.
    accel_values = {
        "Acceleration": snapshot.acceleration,
        "Maximum_Acceleration": snapshot.maximum_acceleration,
    }
    zero_accel = {name: v for name, v in accel_values.items() if v is not None and v == 0}
    if zero_accel:
        verdicts.append(ServoParameterVerdict.VELOCITY_OR_ACCELERATION_RESTRICTION)
        multiplier_note = (
            f"참고로 Acceleration_Multiplier={snapshot.acceleration_multiplier}입니다."
            if snapshot.acceleration_multiplier is not None
            else "Acceleration_Multiplier는 이번에 읽지 못했습니다."
        )
        reasons.append(
            f"다음 가속도 레지스터가 0입니다: {list(zero_accel)}. 이들은 SRAM(휘발성) 레지스터라 "
            "전원이 끊기면 초기화되며, 이 저장소의 configure_motors()(SOFollower 정상 경로)가 "
            "Acceleration/Maximum_Acceleration을 254로 설정하는 코드가 확인되었지만 이번 armed/진단 "
            "실행은 그 경로를 타지 않았습니다. 단, tables.py 주석(\"Acceleration multiplier in effect "
            "when acceleration is 0\")에 따르면 Acceleration=0 자체는 무효값이 아니라 "
            "Acceleration_Multiplier가 대신 적용되는 정의된 상태이므로 '0=가속 불가'로 단정하지 않습니다. "
            f"{multiplier_note} (Goal_Velocity/Maximum_Velocity_Limit은 OperatingMode.VELOCITY 전용일 "
            "수 있어 이 판정에서 제외하고 참고 정보로만 보고합니다.)"
        )

    # 토크 제한: 단위 미확인이지만 0은 명백히 비정상.
    torque_limit_values = {"Torque_Limit": snapshot.torque_limit, "Max_Torque_Limit": snapshot.max_torque_limit}
    zero_torque = {name: v for name, v in torque_limit_values.items() if v is not None and v == 0}
    if zero_torque:
        verdicts.append(ServoParameterVerdict.TORQUE_LIMIT_RESTRICTION)
        reasons.append(f"다음 토크 제한 레지스터가 0입니다: {list(zero_torque)} - 비정상적으로 낮은 토크 제한입니다.")

    if not verdicts:
        verdicts.append(ServoParameterVerdict.NO_CONFIGURATION_CAUSE_FOUND)
        reasons.append(
            "읽은 설정 레지스터에서 뚜렷한 이상(0값 등)이 발견되지 않았습니다 - 이 진단이 확인한 범위 안에서는 "
            "1~2 tick NO_MOTION을 직접 설명할 configuration 근거가 없습니다."
        )

    return tuple(verdicts), tuple(reasons)


# ---------------------------------------------------------------------------
# 다음 단발 후보 계산 (순수 계산, write 없음) - 요구사항 5번
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NextStepCandidate:
    direction: str
    tick_count: int
    requested_delta_deg: float
    expected_raw_delta: int
    expected_target_raw: int
    expected_target_deg: float
    within_calibration_inner_range: bool
    within_physical_range: bool
    under_max_degree: bool
    rationale: str

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "tick_count": self.tick_count,
            "requested_delta_deg": self.requested_delta_deg,
            "expected_raw_delta": self.expected_raw_delta,
            "expected_target_raw": self.expected_target_raw,
            "expected_target_deg": self.expected_target_deg,
            "within_calibration_inner_range": self.within_calibration_inner_range,
            "within_physical_range": self.within_physical_range,
            "under_max_degree": self.under_max_degree,
            "safe_candidate": self.within_calibration_inner_range and self.within_physical_range and self.under_max_degree,
            "rationale": self.rationale,
        }


def compute_next_step_candidates(
    *,
    start_deg: float,
    start_raw: int,
    range_min: int,
    range_max: int,
    motor_resolution: int = DEFAULT_MOTOR_RESOLUTION,
    margin_deg: float = DEFAULT_CALIBRATION_MARGIN_DEG,
    candidate_specs: tuple[tuple[str, int], ...] = (("positive", 3), ("negative", 3)),
    max_degree_delta: float = 0.5,
) -> tuple[NextStepCandidate, ...]:
    """실제 register 결과(현재 raw/deg)를 바탕으로 다음 단발 후보를 계산한다 (write 없음).

    각 후보는 raw tick 개수(``candidate_specs``)로 직접 지정한다 - "3 tick"이라는
    표현 자체가 정수 raw 단위라 반올림 불확실성 없이 정확히 그 tick만큼 이동을
    요청한다는 것을 보장하기 위함이다(도(degree) 요청값으로 역산하면 0.1°가 1~2
    tick으로 반올림됐던 것과 같은 문제가 재발할 수 있다).

    기본값(positive 3 tick, negative 3 tick)은 "이미 시도한 최대 크기(2 tick)보다
    큰 최소 미검증 크기를 양쪽 방향에서 동일하게 확인한다"는 근거를 따른 것이며,
    호출부가 실제 레지스터 조사 결과(예: dead zone 값)에 근거해 다른 tick 수를
    넘기면 그 값을 그대로 쓴다 - 이 함수 자체가 3/4 tick을 강제하지 않는다.
    """

    cal_range = compute_calibration_range_deg(
        range_min=range_min, range_max=range_max, motor_resolution=motor_resolution, margin_deg=margin_deg
    )

    candidates: list[NextStepCandidate] = []
    for direction, tick_count in candidate_specs:
        sign = 1 if direction == "positive" else -1
        target_raw = start_raw + sign * tick_count
        target_deg = raw_to_degrees(
            target_raw, range_min=range_min, range_max=range_max, motor_resolution=motor_resolution
        )
        requested_delta_deg = target_deg - start_deg

        within_physical = -180.0 <= target_deg <= 180.0
        within_inner = cal_range.inner_deg_min <= target_deg <= cal_range.inner_deg_max
        under_max_degree = abs(requested_delta_deg) < max_degree_delta

        candidates.append(
            NextStepCandidate(
                direction=direction,
                tick_count=tick_count,
                requested_delta_deg=requested_delta_deg,
                expected_raw_delta=target_raw - start_raw,
                expected_target_raw=target_raw,
                expected_target_deg=target_deg,
                within_calibration_inner_range=within_inner,
                within_physical_range=within_physical,
                under_max_degree=under_max_degree,
                rationale=(
                    f"이전에 시도한 최대 크기(2 tick)보다 큰 최소 미검증 크기({tick_count} tick, "
                    f"direction={direction})입니다 - 방향별 threshold 대칭성을 확인하기 위한 최소한의 다음 실험."
                ),
            )
        )

    return tuple(candidates)


# ---------------------------------------------------------------------------
# 하드웨어 read-only 접근 (write 메서드 없음)
# ---------------------------------------------------------------------------


class ServoParameterDiagnosticInspector:
    """wrist_roll(motor id 5)의 상태 + 설정 레지스터를 읽기 전용으로 스냅샷한다.

    허용 흐름(요구사항 2번): 포트 존재/점유 확인은 호출부(CLI) 책임이고, 이 클래스는
    4~6단계(``FeetechMotorsBus`` 생성 -> ``connect()`` -> read -> ``disconnect``)만
    담당한다. 공개 메서드는 ``connect``/``read_snapshot``/``disconnect``/
    ``is_connected``뿐이고, write에 해당하는 메서드는 하나도 없다.
    """

    def __init__(self, *, port: str, calibration: WristRollCalibration, num_read_retries: int = 2) -> None:
        self._num_read_retries = num_read_retries
        self.calibration = calibration
        self.port = port
        self._bus = build_wrist_roll_bus(port=port, calibration=calibration)

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def connect(self) -> None:
        """직렬 포트를 열고 wrist_roll 모터만 ping/펌웨어 확인한다. 쓰기 없음."""
        if self._bus.is_connected:
            return
        self._bus.connect()

    def _available_registers(self, names: tuple[str, ...]) -> set[str]:
        table = getattr(self._bus, "model_ctrl_table", {}).get(MOTOR_MODEL, {})
        return {name for name in names if name in table}

    def read_snapshot(self) -> ServoParameterSnapshot:
        """상태 5개 + 설치된 control table에 존재하는 설정 레지스터를 전부 read한다.

        레지스터 하나의 read가 실패해도 나머지는 계속 시도한다. 이 메서드는 어떤
        write도 호출하지 않는다(``bus.write``/``sync_write``/``enable_torque``/
        ``disable_torque``가 이 파일 어디에서도 참조되지 않음).
        """

        values: dict[str, int | None] = {}
        errors: dict[str, str] = {}

        available_state = self._available_registers(STATE_REGISTERS)
        available_params = self._available_registers(PARAMETER_REGISTERS)
        unavailable = tuple(
            sorted((set(STATE_REGISTERS) | set(PARAMETER_REGISTERS)) - available_state - available_params)
        )

        for register in list(available_state) + list(available_params):
            try:
                raw = self._bus.read(register, TARGET_JOINT, normalize=False, num_retry=self._num_read_retries)
                values[register] = int(raw)
            except Exception as exc:  # noqa: BLE001 - 통신 오류를 폭넓게 잡아 None+사유로 남긴다
                values[register] = None
                errors[register] = str(exc)

        return ServoParameterSnapshot(
            torque_enable=values.get("Torque_Enable"),
            goal_position_raw=values.get("Goal_Position"),
            present_position_raw=values.get("Present_Position"),
            moving=values.get("Moving"),
            status_raw=values.get("Status"),
            cw_dead_zone=values.get("CW_Dead_Zone"),
            ccw_dead_zone=values.get("CCW_Dead_Zone"),
            minimum_startup_force=values.get("Minimum_Startup_Force"),
            operating_mode=values.get("Operating_Mode"),
            acceleration=values.get("Acceleration"),
            maximum_acceleration=values.get("Maximum_Acceleration"),
            goal_velocity=values.get("Goal_Velocity"),
            maximum_velocity_limit=values.get("Maximum_Velocity_Limit"),
            moving_velocity_threshold=values.get("Moving_Velocity_Threshold"),
            torque_limit=values.get("Torque_Limit"),
            max_torque_limit=values.get("Max_Torque_Limit"),
            p_coefficient=values.get("P_Coefficient"),
            i_coefficient=values.get("I_Coefficient"),
            d_coefficient=values.get("D_Coefficient"),
            lock=values.get("Lock"),
            angular_resolution=values.get("Angular_Resolution"),
            acceleration_multiplier=values.get(ACCELERATION_MULTIPLIER_REGISTER_NAME),
            read_errors=errors,
            unavailable_registers=unavailable,
        )

    def disconnect(self) -> None:
        """포트를 닫는다. torque 상태를 바꾸는 write는 절대 수행하지 않는다."""
        if not self._bus.is_connected:
            return
        self._bus.disconnect(disable_torque=False)
