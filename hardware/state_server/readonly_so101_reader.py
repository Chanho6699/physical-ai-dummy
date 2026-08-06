"""SO-101(리더/팔로워 공용) 관절 상태를 읽기 전용으로 조회하는 reader.

# 왜 lerobot.robots.so_follower.SOFollower / lerobot.teleoperators.so_leader.SOLeader를
# 그대로 쓰지 않는가 (소스 조사 결과)

``~/lerobot/src/lerobot/robots/so_follower/so_follower.py``와
``~/lerobot/src/lerobot/teleoperators/so_leader/so_leader.py``를 직접 읽고 확인한 내용:

1. ``SOFollower.connect()`` / ``SOLeader.connect()``는 내부에서 무조건
   ``self.configure()``를 호출한다. ``configure()``는 ``bus.torque_disabled()`` 컨텍스트
   안에서이긴 하지만 ``bus.write("Operating_Mode", ...)``, ``P/I/D_Coefficient``,
   (follower의 경우) ``Max_Torque_Limit``/``Protection_Current``/``Overload_Torque``까지
   모터 레지스터에 **write()를 실행한다**. 즉 connect() 한 번으로 이미 "읽기 전용"이
   깨진다.
2. 모터에 저장된 캘리브레이션이 캘리브레이션 파일과 다르면(``is_calibrated == False``)
   ``connect()``가 ``self.calibrate()``까지 실행한다. ``calibrate()``는
   ``bus.disable_torque()``, ``input()``으로 사용자에게 팔을 움직이라고 요구,
   ``set_half_turn_homings()``(Homing_Offset을 새로 write), ``record_ranges_of_motion()``,
   ``write_calibration()``을 수행할 수 있다 - 읽기 전용 서버에서는 절대 발생해선 안 되는
   경로다.
3. ``SOLeader``에는 별도로 ``enable_torque()``/``disable_torque()`` 공개 메서드도 있다.

따라서 이 reader는 로봇/텔레오퍼레이터 클래스를 우회하고,
``lerobot.motors.feetech.FeetechMotorsBus``를 직접 사용해 다음 경로만 호출한다.

# 실제 사용하는 read path

- ``FeetechMotorsBus(port=..., motors=..., calibration=...)`` 생성자: 통신을 열지 않는다
  (직렬 포트 객체만 만든다).
- ``bus.connect()`` (``motors_bus.py`` ``SerialMotorsBus.connect`` ->
  ``FeetechMotorsBus._handshake``): 포트를 열고, 각 모터에 ``ping()``,
  ``_assert_same_firmware()``(펌웨어 버전 read)만 수행한다. **write 없음.**
- ``bus.sync_read("Present_Position", ...)``: ``GroupSyncRead`` 기반 읽기 명령. 모터에
  값을 쓰지 않는다. ``normalize=True``(기본값)이면 LeRobot SO-101 설정과 동일한 방식으로
  단위를 변환한다(몸통 5개 관절은 degree, gripper는 0~100 - 아래 설명 참고).
  ``normalize=False``로 호출하면 원시 tick(0~4095) 그대로 반환한다 - 이 역시 순수 read라
  raw tick도 안전하게 지원할 수 있다.
- ``bus.disconnect(disable_torque=False)``: 포트만 닫는다. 기본 인자
  ``disable_torque=True``로 두면 ``disable_torque()``가 ``Torque_Enable``/``Lock``
  레지스터에 ``write()``를 실행하므로(``feetech.py`` 참고) **반드시 False로 고정한다**.
  (torque를 끄는 것 자체는 실물에 해롭지 않지만, "쓰기 명령을 절대 보내지 않는다"는
  이 서버의 원칙을 예외 없이 지키기 위해 write를 발생시키는 경로 자체를 제거했다.)

이 reader가 절대 호출하지 않는 메서드: ``write``, ``sync_write``, ``enable_torque``,
``disable_torque``, ``write_calibration``, ``reset_calibration``, ``set_half_turn_homings``,
``calibrate``, ``configure``, ``setup_motor``, ``send_action``, ``send_feedback``,
``teleop_step``. 이 클래스는 그런 메서드를 공개(expose)하지도 않는다
(``tests/test_readonly_so101_reader.py`` 참고).

# gripper 단위에 대한 주의

LeRobot의 SO-101 설정(``so_follower.py``/``so_leader.py``)은 ``use_degrees=True``여도
gripper만은 항상 ``MotorNormMode.RANGE_0_100``(0~100)으로 정규화한다 - degree가 아니다.
이 reader도 동일한 설정을 그대로 재현했으므로, ``positions_deg["gripper"]``는 실제로는
"열림 비율(0~100)"이며 각도(degree)가 아니다. API 응답 키 이름은 스펙에 맞춰
``positions_deg``로 유지하되, 이 사실은 ``docs/hardware_state_server.md``에 명시한다.
"""

from __future__ import annotations

import logging

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

logger = logging.getLogger(__name__)

JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# lerobot SOFollowerConfig/SOLeaderConfig의 num_read_retries 기본값과 동일하게 맞춘다.
DEFAULT_NUM_READ_RETRIES = 2

_STS3215_MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


class ReadOnlyReaderError(RuntimeError):
    """이 reader 자체의 사용/구성 오류 (하드웨어 통신 오류와 구분)."""


def _build_motor_map(use_degrees: bool) -> dict[str, Motor]:
    body_mode = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    motors: dict[str, Motor] = {}
    for joint, motor_id in _STS3215_MOTOR_IDS.items():
        # gripper는 lerobot SO-101 설정과 동일하게 항상 0~100 정규화 (docstring 참고).
        norm_mode = MotorNormMode.RANGE_0_100 if joint == "gripper" else body_mode
        motors[joint] = Motor(motor_id, "sts3215", norm_mode)
    return motors


class ReadOnlySO101Reader:
    """하나의 SO-101 팔(리더 또는 팔로워) 버스에 대한 읽기 전용 접근자.

    공개 메서드는 ``connect``/``read_positions``/``read_raw_positions``/``disconnect``와
    ``is_connected`` 프로퍼티뿐이다. 쓰기 메서드는 존재하지 않는다.
    """

    def __init__(
        self,
        *,
        name: str,
        port: str,
        calibration: dict[str, MotorCalibration],
        use_degrees: bool = True,
        num_read_retries: int = DEFAULT_NUM_READ_RETRIES,
    ) -> None:
        missing = [joint for joint in JOINT_ORDER if joint not in calibration]
        if missing:
            raise ReadOnlyReaderError(f"'{name}' 캘리브레이션에 관절이 없습니다: {missing}")

        self.name = name
        self._num_read_retries = num_read_retries
        self._bus = FeetechMotorsBus(
            port=port,
            motors=_build_motor_map(use_degrees=use_degrees),
            calibration=calibration,
        )

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def connect(self) -> None:
        """직렬 포트를 열고 모터를 ping/펌웨어 확인한다. 쓰기 없음."""

        if self._bus.is_connected:
            return
        self._bus.connect()
        logger.info("%s: read-only bus connected (port=%s)", self.name, self._bus.port)

    def read_positions(self) -> dict[str, float]:
        """6개 관절의 정규화된 위치를 반환한다 (몸통 5개=degree, gripper=0~100)."""

        raw = self._bus.sync_read(
            "Present_Position",
            list(JOINT_ORDER),
            num_retry=self._num_read_retries,
        )
        return {joint: float(raw[joint]) for joint in JOINT_ORDER}

    def read_raw_positions(self) -> dict[str, int] | None:
        """6개 관절의 원시 encoder tick(0~4095)을 반환한다.

        ``sync_read(..., normalize=False)``는 순수 읽기 명령이라 안전하게 지원할 수 있다.
        읽기 자체가 실패하면(통신 오류) 예외가 그대로 전파된다 - 호출자가 재시도/오류
        카운트를 판단한다.
        """

        raw = self._bus.sync_read(
            "Present_Position",
            list(JOINT_ORDER),
            normalize=False,
            num_retry=self._num_read_retries,
        )
        return {joint: int(raw[joint]) for joint in JOINT_ORDER}

    def disconnect(self) -> None:
        """포트를 닫는다. torque 상태를 바꾸는 write는 절대 수행하지 않는다."""

        if not self._bus.is_connected:
            return
        self._bus.disconnect(disable_torque=False)
        logger.info("%s: read-only bus disconnected", self.name)
