"""wrist_roll 단일 관절 전용 읽기 전용 하드웨어 조사기.

``hardware/state_server/readonly_so101_reader.py``의 조사 결과를 그대로 재사용한다 -
``FeetechMotorsBus.connect()``는 각 모터에 ping + 펌웨어 버전 read만 하고(write 없음),
``sync_read``는 순수 읽기 명령이라 안전하게 반복 호출할 수 있다
(``~/lerobot/src/lerobot/motors/motors_bus.py`` 조사 근거는
``hardware/safety/single_joint_test_planner.py`` 모듈 docstring 참고).

``ReadOnlySO101Reader``와 다른 점: 이 클래스는 **wrist_roll 모터 하나만** 등록한
``FeetechMotorsBus``를 만든다(``hardware/safety/single_joint_bus.py`` 공유 헬퍼 사용).
따라서 다른 5개 관절(gripper 포함)에는 ping이든 read든 어떤 명령도 절대 보내지 않는다 -
버스 자체가 그 모터들의 존재를 모른다.

이 모듈에 write 메서드는 하나도 없다 (공개 API는 ``connect``/``read_raw``/
``read_degrees``/``disconnect``/``is_connected``뿐 - ``tests/test_single_joint_hardware_inspector.py``
에서 이름 기반으로도 감사한다). 실제 write(단발 armed writer)는
``hardware/safety/single_joint_writer.py``가 별도로, 완전히 분리된 클래스로 담당한다.
"""

from __future__ import annotations

from hardware.safety.single_joint_bus import WristRollCalibration, build_wrist_roll_bus
from hardware.safety.single_joint_test_planner import TARGET_JOINT

__all__ = ["InspectorError", "WristRollCalibration", "SingleJointInspector"]


class InspectorError(RuntimeError):
    """이 inspector 자체의 사용/구성 오류 (하드웨어 통신 오류와 구분)."""


class SingleJointInspector:
    """wrist_roll 하나만 아는 read-only 버스 래퍼.

    lerobot import는 ``build_wrist_roll_bus`` 안에서 지연 수행한다 - 이 모듈 자체는
    lerobot 없이도 import할 수 있어야 하고(예: CLI의 ``--mode inspect-only``
    도움말/설정 검증 단계에서는 아직 필요 없음), 이 클래스를 실제로 생성하는 시점에만
    lerobot이 필요하다.
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

    def read_raw(self) -> int:
        """wrist_roll의 원시 encoder tick(0~4095)을 읽는다. 순수 읽기 명령."""
        raw = self._bus.sync_read(
            "Present_Position", [TARGET_JOINT], normalize=False, num_retry=self._num_read_retries
        )
        return int(raw[TARGET_JOINT])

    def read_degrees(self) -> float:
        """wrist_roll의 정규화된 각도(degree)를 읽는다. 순수 읽기 명령."""
        val = self._bus.sync_read(
            "Present_Position", [TARGET_JOINT], normalize=True, num_retry=self._num_read_retries
        )
        return float(val[TARGET_JOINT])

    def disconnect(self) -> None:
        """포트를 닫는다. torque 상태를 바꾸는 write는 절대 수행하지 않는다."""
        if not self._bus.is_connected:
            return
        self._bus.disconnect(disable_torque=False)
