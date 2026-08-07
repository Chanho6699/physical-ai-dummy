"""wrist_roll 전용 최소 ``FeetechMotorsBus`` 생성 - inspector/writer가 공유하는 내부 헬퍼.

이 모듈 자체는 read/write 어느 쪽 "정책"도 정의하지 않는다 - 오직 "포트+calibration으로
wrist_roll 모터(motor id 5) 하나만 아는 ``FeetechMotorsBus`` 인스턴스를 만든다"는
boilerplate만 담당한다. 무엇을 허용할지는 각각
``hardware/safety/single_joint_hardware_inspector.py``(읽기 전용 - write 메서드 자체가
없음)와 ``hardware/safety/single_joint_writer.py``(평생 최대 1회 ``Goal_Position``
write)가 별도로 강제한다.

lerobot import는 ``build_wrist_roll_bus`` 안에서 지연 수행한다 - 이 모듈 자체는 lerobot
없이도 import할 수 있어야 한다(``WristRollCalibration``은 순수 dataclass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hardware.safety.single_joint_test_planner import MOTOR_MODEL, TARGET_JOINT


@dataclass(frozen=True)
class WristRollCalibration:
    motor_id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def build_wrist_roll_bus(*, port: str, calibration: WristRollCalibration) -> Any:
    """wrist_roll(motor id 5) 하나만 등록된 ``FeetechMotorsBus``를 만든다.

    이 함수는 통신을 열지 않는다(``FeetechMotorsBus`` 생성자 자체가 통신을 열지 않음 -
    ``hardware/state_server/readonly_so101_reader.py`` 조사 결과와 동일). 다른 5개
    관절(gripper 포함)은 이 버스의 ``motors`` dict에 아예 존재하지 않는다.
    """

    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {TARGET_JOINT: Motor(calibration.motor_id, MOTOR_MODEL, MotorNormMode.DEGREES)}
    motor_calibration = {
        TARGET_JOINT: MotorCalibration(
            id=calibration.motor_id,
            drive_mode=calibration.drive_mode,
            homing_offset=calibration.homing_offset,
            range_min=calibration.range_min,
            range_max=calibration.range_max,
        )
    }
    return FeetechMotorsBus(port=port, motors=motors, calibration=motor_calibration)
