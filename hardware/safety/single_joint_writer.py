"""wrist_roll(motor id 5) armed writer - 정확히 한 번의 ``Goal_Position`` write + readback.

## 실제 단일 모터 write API (설치된 lerobot에서 확인, 추측 없음)

``~/lerobot/src/lerobot/motors/motors_bus.py``의 ``MotorsBus.write``::

    @check_if_not_connected
    def write(
        self, data_name: str, motor: str, value: Value, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        ...
        int_value = self._unnormalize({id_: value})[id_]   # normalize=True일 때만
        int_value = self._encode_sign(data_name, {id_: int_value})[id_]
        self._write(addr, length, id_, int_value, num_retry=num_retry, ...)   # 실제 시리얼 write

인자 순서는 ``(data_name, motor, value)`` (모두 위치 인자로 넘길 수 있음, 이 모듈은
가독성을 위해 위치 인자로 호출한다). ``write()``는 ``sync_write()``와 달리 **모터
하나**에만 쓰고, 응답 패킷을 기다려 성공 여부를 확인한다(더 느리지만 더 신뢰할 수
있는 경로 - docstring: "typically be used when configuring motors"). 이 writer는
``sync_write``를 이 파일 어디에서도 참조하지 않는다 (요구사항: "전체 bus sync_write
금지" - 애초에 우리 버스에는 모터가 wrist_roll 하나뿐이라 "전체"라는 개념도 없지만,
``write()``만 쓴다는 것을 이름 기반 감사로도 확인할 수 있게 했다).

``Goal_Position``(주소 42, 2바이트, ``NORMALIZED_DATA``에 포함)은
``Torque_Enable``(주소 40)/``Operating_Mode``(주소 33)/``P_Coefficient``(주소 55와
근처) 등과 완전히 별개 레지스터다 (``~/lerobot/src/lerobot/motors/feetech/tables.py``
``STS_SMS_SERIES_CONTROL_TABLE``). 이 writer는 그 레지스터들에 절대 접근하지 않는다 -
``enable_torque``/``disable_torque``/``configure_motors``/``write_calibration``/
``set_half_turn_homings``를 import도 하지 않는다(아래 감사 테스트 참고). torque가 이미
꺼져 있는 상태라면 이 write는 실제로 모터를 움직이지 않을 수 있는데, 이 writer는 그
상태를 readback으로 감지만 하고(``NO_MOTION``) 절대 torque를 켜지 않는다.

``bus.disconnect(disable_torque=False)``로 고정한다 - ``hardware/state_server/
readonly_so101_reader.py``와 동일한 이유로, disconnect 자체가 write를 발생시키는
경로(``disable_torque=True``의 기본 동작)를 예외 없이 차단한다.

## motor id 5만 write하는 방식

이 writer가 만드는 ``FeetechMotorsBus``(``hardware/safety/single_joint_bus.py``의
``build_wrist_roll_bus``)는 애초에 motor id 5(wrist_roll) 하나만 등록한다. ``write()``
호출도 항상 하드코딩된 ``TARGET_JOINT``("wrist_roll") 딱 하나만 motor 인자로 준다 -
다른 관절 이름을 받는 파라미터 자체가 없다. gripper를 포함한 나머지 5개 관절은 이
모듈의 어떤 함수/메서드에도 등장하지 않는다.

## write budget (평생 최대 1회)

``WriteBudget``이 ``write_count``를 들고 있다가, write 시도 직전에 ``consume()``으로
현재 카운트를 확인한다. 이미 예산을 다 썼으면 실제 ``bus.write()``를 호출하기도 전에
``WriteBudgetExceededError``를 던진다 - 즉 두 번째 시도는 시리얼 패킷 자체가 나가지
않는다. ``consume()``은 성공 여부와 무관하게(설령 이후 ``bus.write()``가 통신 오류로
실패하더라도) 카운트를 이미 올려 두므로, "패킷이 실제로 도달했는지 불확실한 상태"에서도
재시도를 원천적으로 막는다(요구사항: "retry write 금지").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from hardware.safety.single_joint_bus import WristRollCalibration, build_wrist_roll_bus
from hardware.safety.single_joint_test_planner import (
    BLOCKED,
    DEFAULT_CALIBRATION_MARGIN_DEG,
    PASS,
    READBACK_FAILED,
    READBACK_PASS,
    REQUIRED_ARMED_TOTAL_DELTA_DEG,
    TARGET_JOINT,
    WRITE_FAILED,
    ArmedCommandPlan,
    PlannerConfigError,
    build_armed_single_step_plan,
    check_expected_start_matches,
    classify_readback,
)

__all__ = [
    "WristRollCalibration",
    "WriteBudget",
    "WriteBudgetExceededError",
    "SingleJointArmedWriter",
    "ArmedWriteResult",
    "execute_single_armed_write",
    "REQUIRED_CONFIRMATION_FLAGS",
    "DEFAULT_POST_WRITE_WAIT_S",
]

MAX_WRITE_COUNT = 1

# 요구사항 7번: write 후 대기(자동 반복 제어 없음, 딱 한 번만 기다렸다가 readback).
DEFAULT_POST_WRITE_WAIT_S = 0.4

# armed 모드가 요구하는 두 개의 별도 확인 플래그 (요구사항 4번).
REQUIRED_CONFIRMATION_FLAGS: tuple[str, ...] = ("i_have_read_the_safety_plan", "confirm_single_write")


class WriteBudgetExceededError(RuntimeError):
    """write budget(평생 최대 1회)을 초과해서 write를 시도했을 때."""


class WriteBudget:
    """write 횟수를 추적하고 강제하는 아주 작은 카운터. reset 메서드가 없다."""

    def __init__(self, max_write_count: int = MAX_WRITE_COUNT) -> None:
        self._max = max_write_count
        self._count = 0

    @property
    def write_count(self) -> int:
        return self._count

    def consume(self) -> None:
        """write 직전에 호출한다. 예산이 없으면 예외를 던지고, 있으면 카운트를 올린다."""
        if self._count >= self._max:
            raise WriteBudgetExceededError(
                f"write budget 초과: 이미 {self._count}회 write했고 최대 {self._max}회까지만 허용됩니다."
            )
        self._count += 1


class SingleJointArmedWriter:
    """wrist_roll(motor id 5) 하나에 대해 평생 최대 1회만 ``Goal_Position`` write를 허용하는 래퍼.

    이 클래스가 아는 레지스터 이름은 "Goal_Position"(write 1회 한정)과
    "Present_Position"(read, 횟수 제한 없음) 딱 둘뿐이다. ``enable_torque``/
    ``disable_torque``/``configure``/``write_calibration``/``set_half_turn_homings``
    같은 메서드는 이 클래스에 정의되어 있지 않다.
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
        """직렬 포트를 열고 wrist_roll 모터만 ping/펌웨어 확인한다. 이 메서드는 쓰지 않는다."""
        if self._bus.is_connected:
            return
        self._bus.connect()

    def read_raw(self) -> int:
        raw = self._bus.sync_read(
            "Present_Position", [TARGET_JOINT], normalize=False, num_retry=self._num_read_retries
        )
        return int(raw[TARGET_JOINT])

    def read_degrees(self) -> float:
        val = self._bus.sync_read(
            "Present_Position", [TARGET_JOINT], normalize=True, num_retry=self._num_read_retries
        )
        return float(val[TARGET_JOINT])

    def write_goal_position_once(self, *, target_deg: float) -> None:
        """정확히 한 번만 허용되는 단일 write. motor id 5(wrist_roll)의 ``Goal_Position``에만 쓴다.

        두 번째 호출은 ``WriteBudgetExceededError`` - 실제 ``bus.write()``에는 도달하지도
        않는다. ``num_retry``를 노출하지 않고 항상 0으로 고정한다(재시도 금지).
        """
        self._budget.consume()  # 여기서 예외가 나면 아래 self._bus.write는 절대 실행되지 않는다
        self._bus.write("Goal_Position", TARGET_JOINT, target_deg, normalize=True, num_retry=0)

    def disconnect(self) -> None:
        """포트를 닫는다. torque 상태를 바꾸는 write는 절대 수행하지 않는다."""
        if not self._bus.is_connected:
            return
        self._bus.disconnect(disable_torque=False)


class _WriterLike(Protocol):
    """``execute_single_armed_write``가 요구하는 최소 인터페이스 (fake bus 테스트용 duck type)."""

    @property
    def write_count(self) -> int: ...
    def read_raw(self) -> int: ...
    def read_degrees(self) -> float: ...
    def write_goal_position_once(self, *, target_deg: float) -> None: ...


@dataclass
class ArmedWriteResult:
    """섹션 10에서 요구하는 armed 실행 결과 전체."""

    mode: str = "armed"
    target_joint: str = TARGET_JOINT
    motor_id: int | None = None
    direction: str | None = None

    measured_start_raw: int | None = None
    measured_start_deg: float | None = None
    expected_start_raw: int | None = None
    expected_start_deg: float | None = None

    target_raw: int | None = None
    target_deg: float | None = None
    requested_delta_deg: float = REQUIRED_ARMED_TOTAL_DELTA_DEG
    command_raw_delta: int | None = None

    checks: dict = field(default_factory=dict)
    confirmation_flags: dict = field(default_factory=dict)

    write_count_before: int = 0
    write_count_after: int = 0
    write_executed: bool = False

    readback_raw: int | None = None
    readback_deg: float | None = None
    actual_raw_delta: int | None = None
    actual_deg_delta: float | None = None
    readback_verdict: str | None = None

    final_verdict: str = BLOCKED
    block_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "target_joint": self.target_joint,
            "motor_id": self.motor_id,
            "direction": self.direction,
            "measured_start_raw": self.measured_start_raw,
            "measured_start_deg": self.measured_start_deg,
            "expected_start_raw": self.expected_start_raw,
            "expected_start_deg": self.expected_start_deg,
            "target_raw": self.target_raw,
            "target_deg": self.target_deg,
            "requested_delta_deg": self.requested_delta_deg,
            "command_raw_delta": self.command_raw_delta,
            "checks": dict(self.checks),
            "confirmation_flags": dict(self.confirmation_flags),
            "write_count_before": self.write_count_before,
            "write_count_after": self.write_count_after,
            "write_executed": self.write_executed,
            "readback_raw": self.readback_raw,
            "readback_deg": self.readback_deg,
            "actual_raw_delta": self.actual_raw_delta,
            "actual_deg_delta": self.actual_deg_delta,
            "readback_verdict": self.readback_verdict,
            "final_verdict": self.final_verdict,
            "block_reasons": list(self.block_reasons),
        }


def execute_single_armed_write(
    *,
    writer: _WriterLike,
    direction: str,
    calibration: WristRollCalibration,
    expected_start_raw: int | None,
    expected_start_deg: float | None,
    confirmation_flags: dict,
    margin_deg: float = DEFAULT_CALIBRATION_MARGIN_DEG,
    post_write_wait_s: float = DEFAULT_POST_WRITE_WAIT_S,
    sleep_fn: Any = time.sleep,
) -> ArmedWriteResult:
    """wrist_roll에 정확히 0.1° 한 스텝을 시도한다 - 모든 사전조건을 통과했을 때만.

    호출부(CLI)가 이미 포트 점유/calibration 파일 존재는 재확인했다고 가정한다(요구사항
    5번의 그 두 항목). 이 함수는 나머지(관절/모터 id, calibration/이음매 범위, expected
    start 일치, raw delta 크기, write budget)를 전부 다시 확인한 뒤에만 ``writer``의
    단일 write 메서드를 호출한다. 사전조건 중 하나라도 실패하면 write를 절대 시도하지
    않고 ``final_verdict="BLOCKED"``로 즉시 반환한다.
    """
    checks: dict[str, str] = {}
    block_reasons: list[str] = []
    write_count_before = writer.write_count

    def _blocked(**extra: Any) -> ArmedWriteResult:
        return ArmedWriteResult(
            motor_id=calibration.motor_id,
            direction=direction,
            expected_start_raw=expected_start_raw,
            expected_start_deg=expected_start_deg,
            requested_delta_deg=REQUIRED_ARMED_TOTAL_DELTA_DEG,
            checks=dict(checks),
            confirmation_flags=dict(confirmation_flags),
            write_count_before=write_count_before,
            write_count_after=writer.write_count,
            write_executed=False,
            final_verdict=BLOCKED,
            block_reasons=tuple(block_reasons),
            **extra,
        )

    # 1) 확인 플래그 두 개 모두 있어야 한다.
    missing_flags = [name for name in REQUIRED_CONFIRMATION_FLAGS if not confirmation_flags.get(name)]
    checks["confirmation_flags_check"] = BLOCKED if missing_flags else PASS
    if missing_flags:
        block_reasons.append(f"확인 플래그가 부족합니다: {missing_flags}")
        return _blocked()

    # 2) expected start가 최소 하나는 있어야 한다 (없으면 실측 여부와 무관하게 즉시 거부).
    if expected_start_raw is None and expected_start_deg is None:
        checks["expected_start_provided_check"] = BLOCKED
        block_reasons.append("--expected-start-raw/--expected-start-deg 중 어느 것도 제공되지 않았습니다.")
        return _blocked()
    checks["expected_start_provided_check"] = PASS

    # 3) write budget이 아직 남아 있는가 (write 전에 write_count==0 확인 - 요구사항 6번).
    if write_count_before != 0:
        checks["write_budget_check"] = BLOCKED
        block_reasons.append(f"write_count가 이미 {write_count_before}입니다 - 추가 write를 거부합니다.")
        return _blocked()
    checks["write_budget_check"] = PASS

    # 4) motor id가 5(wrist_roll)인가 (calibration 자체 검증 - CLI가 이미 확인했어도 재확인).
    if calibration.motor_id != 5:
        checks["motor_id_check"] = BLOCKED
        block_reasons.append(f"motor id({calibration.motor_id})가 wrist_roll(5)이 아닙니다.")
        return _blocked()
    checks["motor_id_check"] = PASS

    # 5) 현재 raw/deg read 성공 여부 (write 직전 재검사의 핵심 - 실측값 없이는 아무것도 못 한다).
    try:
        measured_raw = writer.read_raw()
        measured_deg = writer.read_degrees()
    except Exception as exc:  # noqa: BLE001 - 통신 오류를 폭넓게 잡아 BLOCKED로 변환한다
        checks["initial_read_check"] = BLOCKED
        block_reasons.append(f"현재 위치를 읽을 수 없습니다: {exc}")
        return _blocked()
    checks["initial_read_check"] = PASS

    # 6) expected start와 실측값이 허용 오차 안에서 일치하는가.
    expected_check, expected_reason = check_expected_start_matches(
        measured_raw=measured_raw,
        measured_deg=measured_deg,
        expected_raw=expected_start_raw,
        expected_deg=expected_start_deg,
    )
    checks["expected_start_match_check"] = expected_check
    if expected_check == BLOCKED:
        block_reasons.append(expected_reason or "expected start가 실측값과 일치하지 않습니다.")
        return _blocked(measured_start_raw=measured_raw, measured_start_deg=measured_deg)

    # 7) 단발 이동 계획 계산 - 이음매 회피/calibration/raw delta 크기를 전부 포함한다.
    try:
        plan: ArmedCommandPlan = build_armed_single_step_plan(
            start_deg=measured_deg,
            start_raw=measured_raw,
            direction=direction,
            range_min=calibration.range_min,
            range_max=calibration.range_max,
            margin_deg=margin_deg,
        )
    except PlannerConfigError as exc:
        checks["plan_check"] = BLOCKED
        block_reasons.append(str(exc))
        return _blocked(measured_start_raw=measured_raw, measured_start_deg=measured_deg)

    checks.update(plan.checks)
    if plan.final_verdict == BLOCKED:
        block_reasons.extend(plan.block_reasons)
        return _blocked(
            measured_start_raw=measured_raw,
            measured_start_deg=measured_deg,
            target_raw=plan.target_raw,
            target_deg=plan.target_deg,
            command_raw_delta=plan.command_raw_delta,
        )

    # -- 여기까지 전부 PASS - 이제서야 단 한 번 write를 시도한다. -----------------------
    write_executed = False
    write_error: str | None = None
    try:
        writer.write_goal_position_once(target_deg=plan.target_deg)
        write_executed = True
    except Exception as exc:  # noqa: BLE001 - WriteBudgetExceededError를 포함해 폭넓게 잡는다
        write_error = str(exc)

    write_count_after = writer.write_count

    base_fields = dict(
        motor_id=calibration.motor_id,
        direction=direction,
        measured_start_raw=measured_raw,
        measured_start_deg=measured_deg,
        expected_start_raw=expected_start_raw,
        expected_start_deg=expected_start_deg,
        target_raw=plan.target_raw,
        target_deg=plan.target_deg,
        command_raw_delta=plan.command_raw_delta,
        checks=dict(checks),
        confirmation_flags=dict(confirmation_flags),
        write_count_before=write_count_before,
        write_count_after=write_count_after,
    )

    if not write_executed:
        block_reasons.append(f"write 시도가 실패했습니다: {write_error}")
        return ArmedWriteResult(
            **base_fields,
            write_executed=False,
            readback_verdict=None,
            final_verdict=WRITE_FAILED,
            block_reasons=tuple(block_reasons),
        )

    # 요구사항 7번: write 후 짧게 대기한다. 자동 반복/재시도는 절대 하지 않는다 - 이
    # sleep 한 번 뒤 readback을 한 번만 시도하고 끝난다.
    sleep_fn(post_write_wait_s)

    try:
        readback_raw = writer.read_raw()
        readback_deg = writer.read_degrees()
    except Exception as exc:  # noqa: BLE001
        block_reasons.append(f"readback 실패: {exc}")
        return ArmedWriteResult(
            **base_fields,
            write_executed=True,
            readback_verdict=READBACK_FAILED,
            final_verdict=READBACK_FAILED,
            block_reasons=tuple(block_reasons),
        )

    actual_raw_delta = readback_raw - measured_raw
    actual_deg_delta = readback_deg - measured_deg
    readback_verdict = classify_readback(
        direction=direction, command_raw_delta=plan.command_raw_delta, actual_raw_delta=actual_raw_delta
    )

    if readback_verdict != READBACK_PASS:
        block_reasons.append(f"readback 판정: {readback_verdict}")

    return ArmedWriteResult(
        **base_fields,
        write_executed=True,
        readback_raw=readback_raw,
        readback_deg=readback_deg,
        actual_raw_delta=actual_raw_delta,
        actual_deg_delta=actual_deg_delta,
        readback_verdict=readback_verdict,
        final_verdict=readback_verdict,
        block_reasons=tuple(block_reasons),
    )
