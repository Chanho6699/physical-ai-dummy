"""wrist_roll(motor id 5) ``Acceleration`` 단일 파라미터 write 도구 - 설계/mock 전용.

**이 모듈이 허용하는 유일한 변경은 ``Acceleration``: 현재값 0 → target 254 하나뿐이다.**
다른 레지스터, 다른 값, 다른 관절을 쓸 수 있는 범용 register writer가 아니다 -
``write_acceleration_once()``는 register 이름/motor 이름/값을 인자로 받지 않는다(전부
모듈 상수로 하드코딩). 이 파일에는 ``Goal_Position``/``Torque_Enable``/
``Operating_Mode`` 등을 write하는 코드 경로가 전혀 없다.

## 왜 이 write가 "합리적인 A/B 테스트"로 판단되었는가 (근거는 이 모듈이 아니라
## ``hardware/safety/single_joint_servo_parameter_diagnostic.py``의 실측 결과 + 아래
## 조사 근거를 종합한 것 - 자세한 조사 내용은 그 모듈의 docstring과 이번 작업의 최종
## 보고를 참고)

- ``tables.py``의 ``Acceleration_Multiplier`` 주석("Acceleration multiplier in effect
  when acceleration is 0")에 따르면 ``Acceleration=0``은 무효값이 아니라 정의된
  대체 상태다 - 그래서 "0=고장"이라고 확정하지 않는다.
- 하지만 실제로 이 저장소의 정상 데이터 수집 경로(``configure_motors()``, 코드로 확인됨)
  는 항상 ``Acceleration=254``를 쓴다 - 즉 254는 이 로봇이 실제로 정상 동작했던
  "검증된" 값이다. 0(현재 상태)과 254(과거 정상 상태) 중 어느 쪽이 NO_MOTION의 원인인지
  결정하려면 실제로 값을 바꿔보는 것 외에는 read-only로 확정할 방법이 없다 - 그래서
  "원인 확정"이 아니라 "합리적인 A/B 테스트"로 판단했다.

## write budget (평생 최대 1회) - ``hardware/safety/single_joint_writer.py``의
## ``WriteBudget``을 그대로 재사용한다 (동일한 정책을 두 곳에 따로 구현하지 않는다).

## 요구사항 9번: 이 모듈은 write 후 readback까지만 하고 끝난다 - 같은 실행에서
## Goal_Position을 절대 건드리지 않는다(자동 이동 테스트 없음). ``execute_single_parameter_write``
## 함수 안 어디에도 ``Goal_Position``을 write하는 코드가 없다 (읽기는 참고용 컨텍스트로
## 허용 - ``read_context()``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hardware.safety.single_joint_bus import WristRollCalibration, build_wrist_roll_bus
from hardware.safety.single_joint_test_planner import (
    BLOCKED,
    PASS,
    READBACK_FAILED,
    TARGET_JOINT,
    WRITE_FAILED,
)
from hardware.safety.single_joint_writer import MAX_WRITE_COUNT as _SHARED_MAX_WRITE_COUNT
from hardware.safety.single_joint_writer import WriteBudget, WriteBudgetExceededError

__all__ = [
    "WristRollCalibration",
    "ALLOWED_REGISTER_NAME",
    "FORBIDDEN_REGISTER_NAMES",
    "EXPECTED_MOTOR_ID",
    "DEFAULT_EXPECTED_CURRENT_ACCELERATION",
    "TARGET_ACCELERATION_VALUE",
    "MAX_WRITE_COUNT",
    "REQUIRED_CONFIRMATION_FLAGS",
    "READBACK_MISMATCH",
    "SingleJointParameterWriter",
    "ParameterWriteResult",
    "execute_single_parameter_write",
    "WriteBudget",
    "WriteBudgetExceededError",
]

# 이 writer가 write할 수 있는 유일한 레지스터. 다른 이름은 코드 어디에도 등장하지 않는다.
ALLOWED_REGISTER_NAME = "Acceleration"

# 요구사항 6번: 이 이름들이 write 대상으로 코드에 전달될 수 없다 - 참고/감사(audit) 용도.
# (write_acceleration_once()가 register 이름 인자를 아예 받지 않으므로 구조적으로도
# 불가능하지만, 정적 검사 테스트에서 이 목록을 근거로 소스에 등장하지 않는지 확인한다.)
FORBIDDEN_REGISTER_NAMES = frozenset(
    {
        "Torque_Enable",
        "Goal_Position",
        "Operating_Mode",
        "P_Coefficient",
        "I_Coefficient",
        "D_Coefficient",
        "CW_Dead_Zone",
        "CCW_Dead_Zone",
        "Minimum_Startup_Force",
        "Maximum_Acceleration",
        "Torque_Limit",
        "Max_Torque_Limit",
        "Lock",
        "Homing_Offset",
    }
)

EXPECTED_MOTOR_ID = 5  # wrist_roll (hardware/state_server/readonly_so101_reader.py와 동일)
DEFAULT_EXPECTED_CURRENT_ACCELERATION = 0
TARGET_ACCELERATION_VALUE = 254  # configure_motors()의 기본값과 동일 (코드로 확인됨)
MAX_WRITE_COUNT = _SHARED_MAX_WRITE_COUNT  # 1

REQUIRED_CONFIRMATION_FLAGS: tuple[str, ...] = (
    "i_understand_this_changes_servo_state",
    "confirm_acceleration_write",
)

READBACK_MISMATCH = "READBACK_MISMATCH"


class _ParameterWriterLike(Protocol):
    """``execute_single_parameter_write``가 요구하는 최소 인터페이스 (fake bus 테스트용)."""

    @property
    def write_count(self) -> int: ...
    @property
    def calibration(self) -> WristRollCalibration: ...
    def read_acceleration(self) -> int: ...
    def read_context(self) -> dict: ...
    def write_acceleration_once(self) -> None: ...


class SingleJointParameterWriter:
    """wrist_roll(motor id 5)의 ``Acceleration`` 하나에 대해 평생 최대 1회만 write를 허용하는 래퍼.

    공개 API는 ``connect``/``read_acceleration``/``read_context``/
    ``write_acceleration_once``/``disconnect``/``is_connected``/``write_count``/
    ``calibration``뿐이다. ``write_acceleration_once()``는 register 이름이나 값을
    인자로 받지 않는다 - 항상 ``Acceleration``에 ``TARGET_ACCELERATION_VALUE``(254)만
    쓴다.
    """

    def __init__(self, *, port: str, calibration: WristRollCalibration, num_read_retries: int = 2) -> None:
        self._num_read_retries = num_read_retries
        self.calibration = calibration
        self.port = port
        self._bus = build_wrist_roll_bus(port=port, calibration=calibration)
        self._budget = WriteBudget(max_write_count=MAX_WRITE_COUNT)

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    @property
    def write_count(self) -> int:
        return self._budget.write_count

    def connect(self) -> None:
        """직렬 포트를 열고 wrist_roll 모터만 ping/펌웨어 확인한다. 쓰기 없음."""
        if self._bus.is_connected:
            return
        self._bus.connect()

    def read_acceleration(self) -> int:
        """현재 ``Acceleration`` raw 값을 읽는다. 순수 읽기 명령."""
        return int(
            self._bus.read(ALLOWED_REGISTER_NAME, TARGET_JOINT, normalize=False, num_retry=self._num_read_retries)
        )

    def read_context(self) -> dict:
        """요구사항 5번(5단계): Torque_Enable/Goal_Position/Present_Position을 참고용으로
        read-only 기록한다 - 이 값들은 write 여부 판정에 쓰이지 않고 리포트에만 남는다.
        읽기 실패는 예외를 전파하지 않고 ``None``으로 남긴다(컨텍스트 로깅이 write 여부를
        막아서는 안 되므로).
        """
        context: dict[str, int | None] = {}
        for name in ("Torque_Enable", "Goal_Position", "Present_Position"):
            try:
                context[name] = int(self._bus.read(name, TARGET_JOINT, normalize=False, num_retry=self._num_read_retries))
            except Exception:  # noqa: BLE001 - 컨텍스트 로깅 실패가 전체 흐름을 막지 않는다
                context[name] = None
        return context

    def write_acceleration_once(self) -> None:
        """``Acceleration``에 정확히 한 번, 254를 write한다. 다른 레지스터/값은 불가능하다.

        두 번째 호출은 ``WriteBudgetExceededError`` - 실제 ``bus.write()``에는 도달하지도
        않는다. ``num_retry``를 노출하지 않고 항상 0으로 고정한다(재시도 금지).
        """
        self._budget.consume()  # 여기서 예외가 나면 아래 self._bus.write는 절대 실행되지 않는다
        self._bus.write(ALLOWED_REGISTER_NAME, TARGET_JOINT, TARGET_ACCELERATION_VALUE, normalize=False, num_retry=0)

    def disconnect(self) -> None:
        """포트를 닫는다. torque 상태를 바꾸는 write는 절대 수행하지 않는다."""
        if not self._bus.is_connected:
            return
        self._bus.disconnect(disable_torque=False)


@dataclass
class ParameterWriteResult:
    register: str = ALLOWED_REGISTER_NAME
    joint: str = TARGET_JOINT
    motor_id: int | None = None

    expected_current_acceleration: int = DEFAULT_EXPECTED_CURRENT_ACCELERATION
    measured_current_acceleration: int | None = None
    target_acceleration: int = TARGET_ACCELERATION_VALUE

    context: dict = field(default_factory=dict)  # Torque_Enable/Goal_Position/Present_Position (참고용)
    confirmation_flags: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    write_count_before: int = 0
    write_count_after: int = 0
    write_executed: bool = False

    readback_acceleration: int | None = None
    readback_matches_target: bool | None = None

    final_verdict: str = BLOCKED
    block_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "register": self.register,
            "joint": self.joint,
            "motor_id": self.motor_id,
            "expected_current_acceleration": self.expected_current_acceleration,
            "measured_current_acceleration": self.measured_current_acceleration,
            "target_acceleration": self.target_acceleration,
            "context": dict(self.context),
            "confirmation_flags": dict(self.confirmation_flags),
            "checks": dict(self.checks),
            "write_count_before": self.write_count_before,
            "write_count_after": self.write_count_after,
            "write_executed": self.write_executed,
            "readback_acceleration": self.readback_acceleration,
            "readback_matches_target": self.readback_matches_target,
            "final_verdict": self.final_verdict,
            "block_reasons": list(self.block_reasons),
        }


def execute_single_parameter_write(
    *,
    writer: _ParameterWriterLike,
    expected_current_acceleration: int,
    confirmation_flags: dict,
) -> ParameterWriteResult:
    """wrist_roll의 ``Acceleration``을 정확히 한 번, 254로 write한다 - 모든 사전조건을
    통과했을 때만.

    요구사항 5번의 10단계 흐름 중 3~9단계를 구현한다(1~2단계 포트 점유/연결은 호출부
    책임). 사전조건 중 하나라도 실패하면 write를 절대 시도하지 않고
    ``final_verdict="BLOCKED"``로 즉시 반환한다. write가 끝나면 readback까지만 하고
    반환한다 - 이 함수는 ``Goal_Position``을 절대 건드리지 않는다(요구사항 9번).
    """
    checks: dict[str, str] = {}
    block_reasons: list[str] = []
    write_count_before = writer.write_count
    motor_id = writer.calibration.motor_id

    def _blocked(**extra: Any) -> ParameterWriteResult:
        return ParameterWriteResult(
            motor_id=motor_id,
            expected_current_acceleration=expected_current_acceleration,
            confirmation_flags=dict(confirmation_flags),
            checks=dict(checks),
            write_count_before=write_count_before,
            write_count_after=writer.write_count,
            write_executed=False,
            final_verdict=BLOCKED,
            block_reasons=tuple(block_reasons),
            **extra,
        )

    # 0) motor id가 wrist_roll(5)인가.
    if motor_id != EXPECTED_MOTOR_ID:
        checks["motor_id_check"] = BLOCKED
        block_reasons.append(f"motor id({motor_id})가 wrist_roll({EXPECTED_MOTOR_ID})이 아닙니다.")
        return _blocked()
    checks["motor_id_check"] = PASS

    # 1) 확인 플래그 두 개 모두 있어야 한다 (요구사항 8번).
    missing_flags = [name for name in REQUIRED_CONFIRMATION_FLAGS if not confirmation_flags.get(name)]
    checks["confirmation_flags_check"] = BLOCKED if missing_flags else PASS
    if missing_flags:
        block_reasons.append(f"확인 플래그가 부족합니다: {missing_flags}")
        return _blocked()

    # 2) write budget이 아직 남아 있는가.
    if write_count_before != 0:
        checks["write_budget_check"] = BLOCKED
        block_reasons.append(f"write_count가 이미 {write_count_before}입니다 - 추가 write를 거부합니다.")
        return _blocked()
    checks["write_budget_check"] = PASS

    # 3) 현재 Acceleration read.
    try:
        measured = writer.read_acceleration()
    except Exception as exc:  # noqa: BLE001
        checks["current_read_check"] = BLOCKED
        block_reasons.append(f"현재 Acceleration을 읽을 수 없습니다: {exc}")
        return _blocked()
    checks["current_read_check"] = PASS

    # 4) 현재값이 정확히 expected 값인가.
    if measured != expected_current_acceleration:
        checks["expected_current_match_check"] = BLOCKED
        block_reasons.append(
            f"실측 Acceleration({measured})이 --expected-current-acceleration"
            f"({expected_current_acceleration})와 다릅니다."
        )
        return _blocked(measured_current_acceleration=measured)
    checks["expected_current_match_check"] = PASS

    # 5) Torque_Enable/Goal_Position/Present_Position을 참고용으로 read-only 기록한다.
    context = writer.read_context()

    base_fields = dict(
        motor_id=motor_id,
        expected_current_acceleration=expected_current_acceleration,
        measured_current_acceleration=measured,
        context=context,
        confirmation_flags=dict(confirmation_flags),
        checks=dict(checks),
        write_count_before=write_count_before,
    )

    # -- 여기까지 전부 PASS - 이제서야 단 한 번 write를 시도한다. -----------------------
    write_executed = False
    write_error: str | None = None
    try:
        writer.write_acceleration_once()
        write_executed = True
    except Exception as exc:  # noqa: BLE001 - WriteBudgetExceededError를 포함해 폭넓게 잡는다
        write_error = str(exc)

    write_count_after = writer.write_count
    base_fields["write_count_after"] = write_count_after

    if not write_executed:
        block_reasons.append(f"write 시도가 실패했습니다: {write_error}")
        return ParameterWriteResult(**base_fields, write_executed=False, final_verdict=WRITE_FAILED, block_reasons=tuple(block_reasons))

    # 6) readback.
    try:
        readback = writer.read_acceleration()
    except Exception as exc:  # noqa: BLE001
        block_reasons.append(f"readback 실패: {exc}")
        return ParameterWriteResult(
            **base_fields, write_executed=True, final_verdict=READBACK_FAILED, block_reasons=tuple(block_reasons)
        )

    matches = readback == TARGET_ACCELERATION_VALUE
    if not matches:
        block_reasons.append(f"readback({readback})이 target({TARGET_ACCELERATION_VALUE})과 다릅니다.")

    return ParameterWriteResult(
        **base_fields,
        write_executed=True,
        readback_acceleration=readback,
        readback_matches_target=matches,
        final_verdict=PASS if matches else READBACK_MISMATCH,
        block_reasons=tuple(block_reasons),
    )
