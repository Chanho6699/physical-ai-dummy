"""follower 6개 관절 전체 STS3215 설정/상태 레지스터 read-only 비교 진단.

``hardware/safety/single_joint_servo_parameter_diagnostic.py``가 확인한
wrist_roll의 ``Acceleration=0``이 wrist_roll 하나만의 개별 문제인지, follower 6개
모터에 공통된 상태(SRAM 휘발성 레지스터가 전원 재인가로 전체 초기화된 것)인지 판단하기
위한 순수 read-only 진단이다. 어떤 write도 하지 않는다.

## joint 이름 -> motor id 매핑 (calibration 파일에서 확인, 추측 없음)

``hardware/state_server/calibration_loader.py``의 ``load_calibration_file``을 그대로
재사용해서 ``~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json``을
읽는다 - 이 로더는 이미 6개 관절이 전부 있는지, 각 항목에 ``id``가 있는지 검증하는
로직을 갖고 있다. 실제 확인된 매핑(``chanho_follower.json``, 이번 조사에서 재확인)::

    shoulder_pan=1, shoulder_lift=2, elbow_flex=3, wrist_flex=4, wrist_roll=5, gripper=6

이 순서/번호는 ``hardware/state_server/readonly_so101_reader.py``의 ``JOINT_ORDER``/
``_STS3215_MOTOR_IDS``와 동일하다 - 이 모듈도 관절 순서를
``hardware.state_server.calibration_loader.JOINT_NAMES``를 그대로 따른다.

## 읽는 레지스터 (전부 STS_SMS_SERIES_CONTROL_TABLE에 실재 확인됨, 새 이름 추측 없음)

``hardware/safety/single_joint_register_diagnostic.py``/
``hardware/safety/single_joint_servo_parameter_diagnostic.py``에서 이미 조사한 것과
동일한 레지스터 이름을 재사용한다: ``Torque_Enable``/``Goal_Position``/
``Present_Position``/``Moving``/``Status``/``Acceleration``/``Maximum_Acceleration``/
``Operating_Mode``/``CW_Dead_Zone``/``CCW_Dead_Zone``/``Minimum_Startup_Force``/
``Torque_Limit``.

## SOFollower.connect()/configure()를 쓰지 않는 이유

``SOFollower.connect()``는 무조건 ``configure()``를 호출하고, ``configure()``는
``bus.configure_motors()``(``Maximum_Acceleration``/``Acceleration``/
``Return_Delay_Time`` write) + ``Operating_Mode``/``P``/``I``/``D_Coefficient`` write를
수행한다(``hardware/state_server/readonly_so101_reader.py`` 모듈 docstring에서 이미
상세히 조사됨). 이번 진단은 **현재 SRAM 상태를 있는 그대로** 읽어야 하므로(만약
configure()를 거치면 Acceleration이 254로 다시 write되어 "0인지 아닌지"를 더 이상 관찰할
수 없다), 6개 모터를 등록한 ``FeetechMotorsBus``를 직접 만들어 read-only로만 접근한다.

이 모듈에는 write에 해당하는 함수/메서드가 하나도 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hardware.state_server.calibration_loader import JOINT_NAMES, MotorCalibrationEntry

__all__ = [
    "JOINT_NAMES",
    "STATE_REGISTERS",
    "PARAMETER_REGISTERS",
    "REQUIRED_REGISTERS_FOR_VERDICT",
    "JointRegisterSnapshot",
    "AllJointVerdict",
    "classify_acceleration_state",
    "find_stuck_joints",
    "AllJointParameterDiagnosticInspector",
]

MOTOR_MODEL = "sts3215"

# 요구사항 2번: 필수로 읽는 상태 레지스터.
STATE_REGISTERS: tuple[str, ...] = ("Torque_Enable", "Operating_Mode", "Goal_Position", "Present_Position", "Moving", "Status")

# 요구사항 2번: "가능하면" 같이 읽는 설정 레지스터. Acceleration/Maximum_Acceleration이
# 이번 조사의 핵심이라 STATE_REGISTERS와 별개로 항상 우선 시도한다.
PARAMETER_REGISTERS: tuple[str, ...] = (
    "Acceleration",
    "Maximum_Acceleration",
    "CW_Dead_Zone",
    "CCW_Dead_Zone",
    "Minimum_Startup_Force",
    "Torque_Limit",
)

# 판정(verdict)에 필요한 최소 레지스터 - 이 중 하나라도 특정 관절에서 None이면
# READ_INCOMPLETE로 처리한다 (요구사항 5번 "필수 motor/register 일부를 읽지 못함").
REQUIRED_REGISTERS_FOR_VERDICT: tuple[str, ...] = (
    "Torque_Enable",
    "Operating_Mode",
    "Goal_Position",
    "Present_Position",
    "Moving",
    "Acceleration",
    "Maximum_Acceleration",
)


class AllJointVerdict:
    ALL_ACCELERATION_ZERO = "ALL_ACCELERATION_ZERO"
    WRIST_ROLL_ONLY_ZERO = "WRIST_ROLL_ONLY_ZERO"
    MIXED_ACCELERATION_STATE = "MIXED_ACCELERATION_STATE"
    ALL_ACCELERATION_CONFIGURED = "ALL_ACCELERATION_CONFIGURED"
    READ_INCOMPLETE = "READ_INCOMPLETE"

    ALL = (
        ALL_ACCELERATION_ZERO,
        WRIST_ROLL_ONLY_ZERO,
        MIXED_ACCELERATION_STATE,
        ALL_ACCELERATION_CONFIGURED,
        READ_INCOMPLETE,
    )


@dataclass(frozen=True)
class JointRegisterSnapshot:
    joint: str
    motor_id: int
    torque_enable: int | None
    operating_mode: int | None
    goal_position_raw: int | None
    present_position_raw: int | None
    moving: int | None
    status_raw: int | None
    acceleration: int | None
    maximum_acceleration: int | None
    cw_dead_zone: int | None = None
    ccw_dead_zone: int | None = None
    minimum_startup_force: int | None = None
    torque_limit: int | None = None
    read_errors: dict = field(default_factory=dict)
    unavailable_registers: tuple[str, ...] = ()

    @property
    def goal_present_delta(self) -> int | None:
        if self.goal_position_raw is None or self.present_position_raw is None:
            return None
        return self.goal_position_raw - self.present_position_raw

    def to_dict(self) -> dict:
        return {
            "joint": self.joint,
            "motor_id": self.motor_id,
            "Torque_Enable": self.torque_enable,
            "Operating_Mode": self.operating_mode,
            "Goal_Position": self.goal_position_raw,
            "Present_Position": self.present_position_raw,
            "goal_present_delta": self.goal_present_delta,
            "Moving": self.moving,
            "Status": self.status_raw,
            "Acceleration": self.acceleration,
            "Maximum_Acceleration": self.maximum_acceleration,
            "CW_Dead_Zone": self.cw_dead_zone,
            "CCW_Dead_Zone": self.ccw_dead_zone,
            "Minimum_Startup_Force": self.minimum_startup_force,
            "Torque_Limit": self.torque_limit,
            "read_errors": dict(self.read_errors),
            "unavailable_registers": list(self.unavailable_registers),
        }


def classify_acceleration_state(
    snapshots: dict[str, JointRegisterSnapshot],
) -> tuple[str, tuple[str, ...]]:
    """읽은 스냅샷만으로 판정한다 - 순수 계산, 하드웨어/파일 접근 없음."""

    reasons: list[str] = []

    incomplete_joints = [
        joint
        for joint, snap in snapshots.items()
        if any(getattr(snap, _snapshot_attr(reg)) is None for reg in REQUIRED_REGISTERS_FOR_VERDICT)
    ]
    if incomplete_joints:
        for joint in incomplete_joints:
            snap = snapshots[joint]
            missing = [reg for reg in REQUIRED_REGISTERS_FOR_VERDICT if getattr(snap, _snapshot_attr(reg)) is None]
            reasons.append(f"{joint}: 필수 레지스터를 읽지 못했습니다: {missing} (errors={snap.read_errors})")
        return AllJointVerdict.READ_INCOMPLETE, tuple(reasons)

    accel_by_joint = {joint: snap.acceleration for joint, snap in snapshots.items()}
    zero_joints = sorted(joint for joint, v in accel_by_joint.items() if v == 0)
    nonzero_joints = sorted(joint for joint, v in accel_by_joint.items() if v != 0)

    if len(zero_joints) == len(accel_by_joint):
        reasons.append(f"모든 관절({zero_joints})의 Acceleration이 0입니다 - SRAM 전체 초기화 가능성.")
        return AllJointVerdict.ALL_ACCELERATION_ZERO, tuple(reasons)

    if zero_joints == ["wrist_roll"]:
        reasons.append(
            f"wrist_roll만 Acceleration=0이고 나머지({nonzero_joints})는 0이 아닙니다 - "
            "wrist_roll 개별 설정 이상 가능성."
        )
        return AllJointVerdict.WRIST_ROLL_ONLY_ZERO, tuple(reasons)

    if zero_joints and nonzero_joints:
        reasons.append(f"0인 관절={zero_joints}, 0이 아닌 관절={nonzero_joints} - 관절별 configure 상태가 섞여 있습니다.")
        return AllJointVerdict.MIXED_ACCELERATION_STATE, tuple(reasons)

    reasons.append(f"모든 관절의 Acceleration이 0이 아닙니다: {accel_by_joint} - Acceleration=0 가설은 기각됩니다.")
    return AllJointVerdict.ALL_ACCELERATION_CONFIGURED, tuple(reasons)


def _snapshot_attr(register_name: str) -> str:
    """레지스터 이름(예: 'Torque_Enable') -> JointRegisterSnapshot 필드 이름."""
    mapping = {
        "Torque_Enable": "torque_enable",
        "Operating_Mode": "operating_mode",
        "Goal_Position": "goal_position_raw",
        "Present_Position": "present_position_raw",
        "Moving": "moving",
        "Status": "status_raw",
        "Acceleration": "acceleration",
        "Maximum_Acceleration": "maximum_acceleration",
    }
    return mapping[register_name]


def find_stuck_joints(snapshots: dict[str, JointRegisterSnapshot]) -> tuple[str, ...]:
    """요구사항 6번: Torque_Enable=1, Goal!=Present, Moving=0인 관절만 골라낸다.

    wrist_roll의 "명령은 latch됐지만 안 움직인" 패턴이 다른 관절에도 나타나는지
    확인하기 위한 순수 계산이다. 이 함수는 해당 관절을 움직이지 않는다 - 이름만 보고한다.
    """
    stuck: list[str] = []
    for joint, snap in snapshots.items():
        delta = snap.goal_present_delta
        if snap.torque_enable == 1 and delta is not None and delta != 0 and snap.moving == 0:
            stuck.append(joint)
    return tuple(stuck)


class AllJointParameterDiagnosticInspector:
    """follower 6개 관절 전체를 읽기 전용으로 스냅샷하는 진단 전용 접근자.

    허용 흐름(요구사항 3번): 포트 존재/점유 확인은 호출부(CLI) 책임이고, 이 클래스는
    4~6단계(``FeetechMotorsBus`` 생성 -> ``connect()`` -> read -> ``disconnect``)만
    담당한다. ``SOFollower``/``SOLeader``는 쓰지 않는다. 공개 메서드는
    ``connect``/``read_all_snapshots``/``disconnect``/``is_connected``뿐이고, write에
    해당하는 메서드는 하나도 없다.
    """

    def __init__(
        self, *, port: str, calibration: dict[str, MotorCalibrationEntry], num_read_retries: int = 2
    ) -> None:
        missing = [joint for joint in JOINT_NAMES if joint not in calibration]
        if missing:
            raise ValueError(f"calibration에 다음 관절이 없습니다: {missing}")

        self._num_read_retries = num_read_retries
        self.calibration = calibration
        self.port = port

        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        motors = {}
        motor_calibration = {}
        for joint in JOINT_NAMES:
            entry = calibration[joint]
            # gripper는 이 저장소 전체에서 RANGE_0_100 정규화를 쓴다
            # (hardware/state_server/readonly_so101_reader.py 조사 근거) - 이 진단은 항상
            # normalize=False로 raw만 읽으므로 정규화 모드 자체가 결과에 영향을 주지는
            # 않지만, calibration 객체의 일관성을 위해 프로젝트 관례를 그대로 따른다.
            norm_mode = MotorNormMode.RANGE_0_100 if joint == "gripper" else MotorNormMode.DEGREES
            motors[joint] = Motor(entry.id, MOTOR_MODEL, norm_mode)
            motor_calibration[joint] = MotorCalibration(
                id=entry.id,
                drive_mode=entry.drive_mode,
                homing_offset=entry.homing_offset,
                range_min=entry.range_min,
                range_max=entry.range_max,
            )

        self._bus = FeetechMotorsBus(port=port, motors=motors, calibration=motor_calibration)

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def connect(self) -> None:
        """직렬 포트를 열고 6개 모터 ping/펌웨어 확인한다. 쓰기 없음."""
        if self._bus.is_connected:
            return
        self._bus.connect()

    def _available_registers(self, names: tuple[str, ...]) -> set[str]:
        table = getattr(self._bus, "model_ctrl_table", {}).get(MOTOR_MODEL, {})
        return {name for name in names if name in table}

    def read_all_snapshots(self) -> dict[str, JointRegisterSnapshot]:
        """6개 관절 전부에 대해 상태 + 설정 레지스터를 read한다. write는 절대 호출하지 않는다."""

        available_state = self._available_registers(STATE_REGISTERS)
        available_params = self._available_registers(PARAMETER_REGISTERS)
        unavailable = tuple(sorted((set(STATE_REGISTERS) | set(PARAMETER_REGISTERS)) - available_state - available_params))

        snapshots: dict[str, JointRegisterSnapshot] = {}
        for joint in JOINT_NAMES:
            entry = self.calibration[joint]
            values: dict[str, int | None] = {}
            errors: dict[str, str] = {}
            for register in list(available_state) + sorted(available_params):
                try:
                    raw = self._bus.read(register, joint, normalize=False, num_retry=self._num_read_retries)
                    values[register] = int(raw)
                except Exception as exc:  # noqa: BLE001 - 통신 오류를 폭넓게 잡아 None+사유로 남긴다
                    values[register] = None
                    errors[register] = str(exc)

            snapshots[joint] = JointRegisterSnapshot(
                joint=joint,
                motor_id=entry.id,
                torque_enable=values.get("Torque_Enable"),
                operating_mode=values.get("Operating_Mode"),
                goal_position_raw=values.get("Goal_Position"),
                present_position_raw=values.get("Present_Position"),
                moving=values.get("Moving"),
                status_raw=values.get("Status"),
                acceleration=values.get("Acceleration"),
                maximum_acceleration=values.get("Maximum_Acceleration"),
                cw_dead_zone=values.get("CW_Dead_Zone"),
                ccw_dead_zone=values.get("CCW_Dead_Zone"),
                minimum_startup_force=values.get("Minimum_Startup_Force"),
                torque_limit=values.get("Torque_Limit"),
                read_errors=errors,
                unavailable_registers=unavailable,
            )

        return snapshots

    def disconnect(self) -> None:
        """포트를 닫는다. torque 상태를 바꾸는 write는 절대 수행하지 않는다."""
        if not self._bus.is_connected:
            return
        self._bus.disconnect(disable_torque=False)
