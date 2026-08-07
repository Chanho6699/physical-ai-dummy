"""wrist_roll(motor id 5) 레지스터 read-only 진단 - armed write 후 NO_MOTION 원인 파악용.

## 조사 근거 (설치된 lerobot/Feetech 코드에서 확인, 추측 없음)

``~/lerobot/src/lerobot/motors/feetech/tables.py``의 ``STS_SMS_SERIES_CONTROL_TABLE``
(``sts3215``가 쓰는 테이블)에 실제로 정의된 레지스터만 읽는다::

    "Torque_Enable": (40, 1)
    "Goal_Position": (42, 2)          # NORMALIZED_DATA에 포함 (degree <-> raw 변환 대상)
    "Present_Position": (56, 2)       # read-only, NORMALIZED_DATA에 포함
    "Moving": (66, 1)                 # read-only
    "Present_Load": (60, 2)           # read-only, sign-magnitude 인코딩(10bit)
    "Present_Current": (69, 2)        # read-only, 부호 인코딩 없음(unsigned 그대로)
    "Present_Velocity": (58, 2)       # read-only, sign-magnitude 인코딩(15bit)
    "Present_Voltage": (62, 1)        # read-only
    "Present_Temperature": (63, 1)    # read-only
    "Status": (65, 1)                 # read-only

**요구사항에서 언급한 ``Moving_Status``와 ``Hardware_Error_Status``는 설치된
``STS_SMS_SERIES_CONTROL_TABLE``에 존재하지 않는다** (Dynamixel 쪽 테이블
``~/lerobot/src/lerobot/motors/dynamixel/tables.py``에만 ``Hardware_Error_Status``가
있고, Feetech/STS3215에는 없다 - 추측으로 만들어내지 않았다):

- ``Moving_Status``: 없음. ``Moving``(66번지, 1바이트, 0/1)만 존재한다. 이 모듈은
  ``Moving_Status``를 ``NOT_AVAILABLE_IN_INSTALLED_TABLE``로 명시하고 실제로 read를
  시도하지 않는다.
- ``Hardware_Error_Status``: 없음. 가장 근접한 것은 ``Status``(65번지, 1바이트,
  read-only)인데, lerobot 소스 어디에도 이 바이트의 비트별 의미를 디코딩하는 코드가
  없다(``_decode_sign``의 인코딩 테이블에도 없음 - 부호 없는 순수 raw byte로 반환됨).
  그래서 이 모듈은 ``Status``를 읽어서 원시값만 보고하고, "0이 아니면 잠재적 이상"
  정도로만 취급한다 - 특정 비트가 과전류/과열/저전압 중 무엇을 뜻하는지는 확정하지
  않는다(요구사항 4번: "정확한 비트 해석은 근거가 있을 때만").

``bus.read(data_name, motor, *, normalize=False, num_retry=...)``
(``~/lerobot/src/lerobot/motors/motors_bus.py`` ``MotorsBus.read``)는 모터 하나의
레지스터 하나를 읽는 순수 read 명령이다 - 내부적으로 ``self._read(...)``(시리얼 read
패킷)만 호출하고 ``write``/``_write``는 절대 호출하지 않는다. ``normalize=False``로
고정해서 ``Goal_Position``/``Present_Position``도 raw tick 그대로 받는다(이전 armed
실행 리포트의 raw 값과 직접 비교하기 위함).

이 모듈에는 write에 해당하는 함수/메서드가 하나도 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hardware.safety.single_joint_bus import WristRollCalibration, build_wrist_roll_bus
from hardware.safety.single_joint_test_planner import MOTOR_MODEL, TARGET_JOINT

__all__ = [
    "WristRollCalibration",
    "RegisterSnapshot",
    "RegisterDiagnosticInspector",
    "DiagnosticVerdict",
    "classify_diagnostic",
    "REQUIRED_REGISTERS",
    "OPTIONAL_REGISTERS",
    "MOVING_STATUS_NOT_AVAILABLE_NOTE",
    "HARDWARE_ERROR_STATUS_NOT_AVAILABLE_NOTE",
]

# 설치된 STS_SMS_SERIES_CONTROL_TABLE에 실제로 존재하는 이름만 사용한다 (위 docstring 근거).
REQUIRED_REGISTERS: tuple[str, ...] = ("Torque_Enable", "Goal_Position", "Present_Position", "Moving")
OPTIONAL_REGISTERS: tuple[str, ...] = (
    "Present_Load",
    "Present_Current",
    "Present_Velocity",
    "Present_Voltage",
    "Present_Temperature",
    "Status",
)

MOVING_STATUS_NOT_AVAILABLE_NOTE = (
    "'Moving_Status' 레지스터는 설치된 lerobot STS_SMS_SERIES_CONTROL_TABLE(sts3215)에 "
    "정의되어 있지 않습니다 - 'Moving'(0/1)만 존재합니다. read를 시도하지 않았습니다."
)
HARDWARE_ERROR_STATUS_NOT_AVAILABLE_NOTE = (
    "'Hardware_Error_Status' 레지스터는 Feetech/STS3215 control table에 없습니다 "
    "(Dynamixel 전용). 가장 근접한 'Status' 레지스터를 대신 읽었으나, lerobot 소스에 "
    "비트별 의미를 디코딩하는 코드가 없어 원시값만 보고합니다."
)


class DiagnosticVerdict:
    """요구사항 4번의 6개 판정 라벨. 클래스로 묶어 이름 충돌/오탐지 감사를 쉽게 한다."""

    TORQUE_DISABLED = "TORQUE_DISABLED"
    GOAL_NOT_LATCHED = "GOAL_NOT_LATCHED"
    COMMAND_LATCHED_BUT_NO_MOTION = "COMMAND_LATCHED_BUT_NO_MOTION"
    MOTOR_STILL_MOVING = "MOTOR_STILL_MOVING"
    FAULT_OR_PROTECTION = "FAULT_OR_PROTECTION"
    UNKNOWN = "UNKNOWN"

    ALL = (
        TORQUE_DISABLED,
        GOAL_NOT_LATCHED,
        COMMAND_LATCHED_BUT_NO_MOTION,
        MOTOR_STILL_MOVING,
        FAULT_OR_PROTECTION,
        UNKNOWN,
    )


@dataclass(frozen=True)
class RegisterSnapshot:
    """한 시점에 읽은 wrist_roll 레지스터 값들 (전부 raw, degree 변환 없음)."""

    torque_enable: int | None
    goal_position_raw: int | None
    present_position_raw: int | None
    moving: int | None
    present_load: int | None = None
    present_current: int | None = None
    present_velocity: int | None = None
    present_voltage: int | None = None
    present_temperature: int | None = None
    status_raw: int | None = None
    read_errors: dict = field(default_factory=dict)  # register 이름 -> 오류 메시지

    def to_dict(self) -> dict:
        return {
            "Torque_Enable": self.torque_enable,
            "Goal_Position": self.goal_position_raw,
            "Present_Position": self.present_position_raw,
            "Moving": self.moving,
            "Moving_Status": "NOT_AVAILABLE_IN_INSTALLED_TABLE",
            "Present_Load": self.present_load,
            "Present_Current": self.present_current,
            "Present_Velocity": self.present_velocity,
            "Present_Voltage": self.present_voltage,
            "Present_Temperature": self.present_temperature,
            "Status": self.status_raw,
            "read_errors": dict(self.read_errors),
        }


def classify_diagnostic(
    *,
    snapshot: RegisterSnapshot,
    expected_start_raw: int | None,
    expected_goal_raw: int | None,
    goal_latch_tolerance_ticks: int = 0,
    present_goal_deadband_ticks: int = 0,
) -> tuple[str, tuple[str, ...]]:
    """읽은 스냅샷만으로 판정한다 - 순수 계산, 하드웨어/파일 접근 없음.

    우선순위(코드 내 판단 근거, 요구사항 4번의 예시 조건을 그대로 구현하되 겹치는
    상황을 명확히 하기 위해 이 모듈이 정한 순서):

    1. 필수 레지스터(Torque_Enable/Goal_Position/Present_Position/Moving) 중 하나라도
       읽기 실패(``None``)면 ``UNKNOWN``.
    2. ``Status``가 읽혔고 0이 아니면 ``FAULT_OR_PROTECTION`` - 하드웨어 오류가 있으면
       torque 꺼짐/goal 불일치/미동작을 전부 설명할 수 있는 근본 원인이라 최우선으로 본다.
    3. ``Torque_Enable == 0``이면 ``TORQUE_DISABLED``.
    4. ``expected_goal_raw``가 주어졌고 ``Goal_Position``이 그 값과
       ``goal_latch_tolerance_ticks``를 넘게 다르면 ``GOAL_NOT_LATCHED``.
    5. ``Moving != 0``이면 ``MOTOR_STILL_MOVING`` (자동 재확인 없이 상태만 보고).
    6. 여기까지 왔다면 torque는 켜져 있고, goal은 latch되어 있고(또는 확인 불가), 아직
       움직이는 중도 아니다. ``Present_Position``과 ``Goal_Position``의 차이가
       ``present_goal_deadband_ticks``를 넘으면 ``COMMAND_LATCHED_BUT_NO_MOTION``.
    7. 그 외(예: goal==present, 즉 이미 목표에 도달한 것으로 보임)는 ``UNKNOWN``으로
       두고 사유에 명시한다 - 이 도구는 "동작 실패를 설명하는 것"이 목적이라 성공
       사례에 대한 별도 라벨을 만들지 않는다.
    """

    reasons: list[str] = []

    missing = [
        name
        for name, value in (
            ("Torque_Enable", snapshot.torque_enable),
            ("Goal_Position", snapshot.goal_position_raw),
            ("Present_Position", snapshot.present_position_raw),
            ("Moving", snapshot.moving),
        )
        if value is None
    ]
    if missing:
        reasons.append(f"필수 레지스터를 읽지 못했습니다: {missing}")
        return DiagnosticVerdict.UNKNOWN, tuple(reasons)

    if snapshot.status_raw is not None and snapshot.status_raw != 0:
        reasons.append(
            f"Status 레지스터가 0이 아닙니다(raw={snapshot.status_raw}) - 하드웨어 오류/보호 상태일 "
            "가능성이 있습니다 (비트별 의미는 확정하지 않음)."
        )
        return DiagnosticVerdict.FAULT_OR_PROTECTION, tuple(reasons)

    if snapshot.torque_enable == 0:
        reasons.append("Torque_Enable == 0 - 모터가 힘을 내지 않는 상태입니다.")
        return DiagnosticVerdict.TORQUE_DISABLED, tuple(reasons)

    if expected_goal_raw is not None:
        goal_diff = abs(snapshot.goal_position_raw - expected_goal_raw)
        if goal_diff > goal_latch_tolerance_ticks:
            reasons.append(
                f"Goal_Position(raw={snapshot.goal_position_raw})이 expected_goal_raw"
                f"({expected_goal_raw})와 {goal_diff} tick 차이납니다(허용 오차 "
                f"{goal_latch_tolerance_ticks} tick)."
            )
            return DiagnosticVerdict.GOAL_NOT_LATCHED, tuple(reasons)

    if snapshot.moving != 0:
        reasons.append(f"Moving == {snapshot.moving} (0이 아님) - 아직 이동 중일 수 있습니다.")
        return DiagnosticVerdict.MOTOR_STILL_MOVING, tuple(reasons)

    goal_present_delta = snapshot.goal_position_raw - snapshot.present_position_raw
    if abs(goal_present_delta) > present_goal_deadband_ticks:
        reasons.append(
            f"Torque_Enable=1, Goal_Position={snapshot.goal_position_raw}, "
            f"Present_Position={snapshot.present_position_raw}(차이 {goal_present_delta} tick), "
            "Moving=0 - 명령은 저장됐지만 모터가 반응하지 않았습니다."
        )
        return DiagnosticVerdict.COMMAND_LATCHED_BUT_NO_MOTION, tuple(reasons)

    reasons.append(
        "Goal_Position과 Present_Position이 사실상 같고(이미 목표에 도달한 것으로 보임) "
        "다른 이상 신호가 없습니다 - 이 도구가 정의한 실패 라벨에 해당하지 않습니다."
    )
    return DiagnosticVerdict.UNKNOWN, tuple(reasons)


class RegisterDiagnosticInspector:
    """wrist_roll(motor id 5) 레지스터를 읽기 전용으로 스냅샷하는 진단 전용 접근자.

    ``hardware/safety/single_joint_hardware_inspector.py``의 ``SingleJointInspector``와
    마찬가지로 wrist_roll 하나만 아는 버스를 쓰지만, ``Present_Position`` 하나가 아니라
    이 모듈이 정의한 진단용 레지스터 세트를 전부 읽는다는 점이 다르다. 공개 메서드는
    ``connect``/``read_snapshot``/``disconnect``/``is_connected``뿐이고, write에 해당하는
    메서드는 하나도 없다.
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

    def _available_optional_registers(self) -> set[str]:
        """설치된 control table에 실제로 존재하는 선택 레지스터만 골라낸다 (추측 방지)."""
        table = getattr(self._bus, "model_ctrl_table", {}).get(MOTOR_MODEL, {})
        return {name for name in OPTIONAL_REGISTERS if name in table}

    def read_snapshot(self) -> RegisterSnapshot:
        """필수 레지스터 4개 + 설치된 table에 존재하는 선택 레지스터를 전부 read한다.

        레지스터 하나의 read가 실패해도 나머지는 계속 시도한다 - 실패한 레지스터는
        ``None``으로 남고 ``read_errors``에 사유가 기록된다. 이 메서드는 어떤 write도
        호출하지 않는다(``bus.write``/``sync_write``/``enable_torque``/``disable_torque``
        전부 이 파일에서 참조되지 않음).
        """

        values: dict[str, int | None] = {}
        errors: dict[str, str] = {}

        registers_to_read = list(REQUIRED_REGISTERS) + sorted(self._available_optional_registers())
        for register in registers_to_read:
            try:
                raw = self._bus.read(register, TARGET_JOINT, normalize=False, num_retry=self._num_read_retries)
                values[register] = int(raw)
            except Exception as exc:  # noqa: BLE001 - 통신 오류를 폭넓게 잡아 None+사유로 남긴다
                values[register] = None
                errors[register] = str(exc)

        return RegisterSnapshot(
            torque_enable=values.get("Torque_Enable"),
            goal_position_raw=values.get("Goal_Position"),
            present_position_raw=values.get("Present_Position"),
            moving=values.get("Moving"),
            present_load=values.get("Present_Load"),
            present_current=values.get("Present_Current"),
            present_velocity=values.get("Present_Velocity"),
            present_voltage=values.get("Present_Voltage"),
            present_temperature=values.get("Present_Temperature"),
            status_raw=values.get("Status"),
            read_errors=errors,
        )

    def disconnect(self) -> None:
        """포트를 닫는다. torque 상태를 바꾸는 write는 절대 수행하지 않는다."""
        if not self._bus.is_connected:
            return
        self._bus.disconnect(disable_torque=False)
