"""Shadow Teleop Diagnostic - 리더 wrist_roll + 팔로워 wrist_roll을 동시에 read-only로 관측.

## 목적

leader arm을 손으로 움직이면서 leader wrist_roll 상태와 follower wrist_roll 상태를 **동시에
읽기만** 한다. follower ``Goal_Position`` write는 이 모듈 어디에서도 하지 않는다 - 리더를
움직여도 팔로워가 따라 움직여서는 안 된다(이번 단계는 "관찰"이지 "restricted teleop"이 아니다).

## 재사용/근거

- 버스 생성: ``hardware/safety/single_joint_bus.py``의 ``build_wrist_roll_bus``를 그대로
  재사용한다 - motor id 5(wrist_roll) 하나만 등록한 ``FeetechMotorsBus``를 만들고, 생성자
  자체는 통신을 열지 않는다(``hardware/state_server/readonly_so101_reader.py`` 조사 근거와
  동일).
- read 경로: ``hardware/safety/single_joint_hardware_inspector.py``(``SingleJointInspector``)와
  ``hardware/safety/single_joint_register_diagnostic.py``(``RegisterDiagnosticInspector``)가
  이미 확인한 것과 동일한 API만 쓴다 - ``bus.connect()``(ping+펌웨어 read, write 없음),
  ``bus.sync_read(..., normalize=...)``/``bus.read(data_name, motor, normalize=False, ...)``
  (순수 read), ``bus.disconnect(disable_torque=False)``(포트만 닫음, torque write 없음).
- 각도 변환: ``hardware/safety/single_joint_test_planner.py``의 ``raw_to_degrees``를 그대로
  재사용한다 - lerobot ``MotorsBus._normalize``의 DEGREES 공식과 동일(코드로 확인됨,
  clamp 없음). 리더/팔로워 calibration이 다를 수 있으므로 **각자의 calibration**으로 변환한
  뒤에만 비교한다(raw tick을 직접 빼지 않는다).
- 리더/팔로워 모두 실제 calibration이 full-turn(``range_min=0``, ``range_max=4095``)이라
  ``raw_to_degrees``의 스케일(``360/(motor_resolution-1)``)이 range_min/range_max와 무관하게
  고정된다(offset(``mid``)만 range에 의존) - 그래서 "몇 tick이 몇 도인가"는 두 팔에서 동일할
  것으로 예상되지만, 이 모듈은 그 사실에 의존해 지름길을 타지 않고 각 팔의 실제 calibration을
  ``degrees_per_tick_for_calibration``으로 각각 계산해서 보고한다(섹션 12).

## 이 모듈이 절대 호출하지 않는 것

``write``/``sync_write``/``enable_torque``/``disable_torque``/``configure``/``calibrate``/
``write_calibration``/``set_half_turn_homings``/``send_action``/``send_feedback``/
``teleop_step``. 어느 클래스도 이런 이름의 공개 메서드를 갖지 않는다
(``tests/test_shadow_teleop_diagnostic.py``에서 이름 기반 감사 + 가짜 버스로 재확인).
추가로 이 모듈이 만드는 모든 버스는 ``_install_write_guard``로 ``write``/``sync_write``/
``enable_torque``/``disable_torque`` 메서드 자체를 "호출되면 즉시 예외"로 바꿔치기한다 -
향후 실수로 write 호출 코드가 추가되더라도 실제 시리얼 write 패킷이 나가기 전에 여기서
막는다(defense-in-depth, ``hardware/safety/single_joint_writer.py``의 ``WriteBudget``과
같은 정신).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from hardware.safety.single_joint_bus import WristRollCalibration, build_wrist_roll_bus
from hardware.safety.single_joint_test_planner import (
    DEFAULT_MOTOR_RESOLUTION,
    TARGET_JOINT,
    raw_to_degrees,
)

__all__ = [
    "WristRollCalibration",
    "WriteGuardTriggeredError",
    "ShadowTeleopReaderError",
    "ShadowTeleopReadError",
    "FollowerMovedUnexpectedlyError",
    "LeaderWristRollReader",
    "FollowerWristRollStateReader",
    "FollowerStateSnapshot",
    "FollowerAccelSnapshot",
    "ShadowSample",
    "CSV_FIELDNAMES",
    "ShadowTeleopSampler",
    "SamplingRunResult",
    "run_sampling_loop",
    "FOLLOWER_STATE_REGISTERS",
    "FOLLOWER_ACCEL_REGISTERS",
    "DEFAULT_ACCEL_REFRESH_INTERVAL_S",
    "DEFAULT_FOLLOWER_MOVE_ABORT_THRESHOLD_DEG",
    "FOLLOWER_FIXED_THRESHOLD_DEG",
    "compute_run_analysis",
    "degrees_per_tick_for_calibration",
    "build_command_delta_reference_table",
]

# 설치된 STS_SMS_SERIES_CONTROL_TABLE(sts3215)에 실재하는 이름만 사용한다 - 근거는
# hardware/safety/single_joint_register_diagnostic.py / single_joint_servo_parameter_diagnostic.py
# 모듈 docstring 참고. 마지막 항목은 원본 dict key 오탈자(끝에 공백)를 그대로 쓴다.
FOLLOWER_STATE_REGISTERS: tuple[str, ...] = ("Goal_Position", "Present_Position", "Torque_Enable", "Moving", "Status")
FOLLOWER_ACCEL_REGISTERS: tuple[str, ...] = ("Acceleration", "Acceleration_Multiplier ")

# 가속도 레지스터는 이 세션에서 절대 write되지 않으므로(SRAM이라도) 매 샘플 다시 읽을
# 필요가 없다 - 매 초 한 번만 갱신해서 고속 필수 레지스터(Goal/Present/Torque/Moving/Status +
# 리더 Present_Position)에 read 예산을 몰아준다.
DEFAULT_ACCEL_REFRESH_INTERVAL_S = 1.0

# follower_present_deg의 관측 범위가 이 값 이하면 "사실상 고정"으로 판단한다 - STS3215
# 1 tick ≈ 0.088°(섹션 12 계산과 동일 스케일)의 절반 정도로, 노이즈 없는 read-only 관측에서
# 실제 이동과 read jitter를 구분하기 위한 보수적 문턱값이다(실측 튜닝값은 아님).
FOLLOWER_FIXED_THRESHOLD_DEG = 0.05

# 실시간 안전장치(섹션 10): "리더를 움직여도 팔로워가 움직이면 즉시 실험을 중단한다"를
# 코드로 강제한다. follower_present_delta_from_start_deg의 절대값이 이 문턱값을 넘으면
# run_sampling_loop가 즉시 멈춘다. FOLLOWER_FIXED_THRESHOLD_DEG(사후 분석용, 0.05°)보다
# 훨씬 크게 잡아(약 5~6 tick) read jitter/backlash로 인한 오탐 중단을 피하면서도, 실제
# 유의미한 움직임은 빠르게 감지한다 - 실측 튜닝값은 아니며 보수적 기본값이다.
DEFAULT_FOLLOWER_MOVE_ABORT_THRESHOLD_DEG = 0.5


class WriteGuardTriggeredError(RuntimeError):
    """이 모듈이 만든 버스에서 write 계열 메서드 호출이 시도되면 던져진다."""


class ShadowTeleopReaderError(RuntimeError):
    """reader 자체의 사용/구성 오류 (하드웨어 통신 오류와 구분)."""


class ShadowTeleopReadError(RuntimeError):
    """샘플링 중 leader/follower 필수 레지스터 read가 실패했을 때 - 호출부는 안전 종료해야 한다."""


class FollowerMovedUnexpectedlyError(RuntimeError):
    """follower_present_delta_from_start_deg가 안전 문턱값을 초과했을 때 - 즉시 중단 신호."""


def _install_write_guard(bus: Any) -> None:
    """bus의 write 계열 메서드를 전부 "호출되면 즉시 예외"로 바꿔치기한다 (defense-in-depth).

    이 diagnostic 코드 경로 자체가 이 메서드들을 호출하지 않는다는 것은 소스 검사/단위
    테스트로 이미 보장하지만, 향후 실수로 호출 코드가 추가되더라도 실제 시리얼 write
    패킷이 나가기 전에 여기서 막는다. ``disconnect(disable_torque=False)``는 이 메서드들을
    호출하지 않으므로(항상 False로 고정) 정상 disconnect는 이 가드의 영향을 받지 않는다.
    """

    def _forbidden(name: str) -> Callable[..., None]:
        def _raise(*args: Any, **kwargs: Any) -> None:
            raise WriteGuardTriggeredError(
                f"Shadow Teleop Diagnostic 버스에서 '{name}()' 호출이 시도되었습니다 - 차단합니다."
            )

        return _raise

    for name in ("write", "sync_write", "enable_torque", "disable_torque"):
        setattr(bus, name, _forbidden(name))


# ---------------------------------------------------------------------------
# 리더 read-only reader (SingleJointInspector와 동일 패턴 - 별도 클래스로 분리해
# 리더/팔로워 혼동을 코드 레벨에서부터 방지한다)
# ---------------------------------------------------------------------------


class LeaderWristRollReader:
    """리더 wrist_roll(motor id는 리더 calibration에서 확인) read-only reader.

    공개 메서드는 ``connect``/``read_raw``/``disconnect``/``is_connected``뿐이다. write에
    해당하는 메서드는 하나도 없다.
    """

    def __init__(self, *, port: str, calibration: WristRollCalibration, num_read_retries: int = 2) -> None:
        self._num_read_retries = num_read_retries
        self.calibration = calibration
        self.port = port
        self._bus = build_wrist_roll_bus(port=port, calibration=calibration)
        _install_write_guard(self._bus)

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def connect(self) -> None:
        """직렬 포트를 열고 wrist_roll 모터만 ping/펌웨어 확인한다. 쓰기 없음."""
        if self._bus.is_connected:
            return
        self._bus.connect()

    def read_raw(self) -> int:
        """리더 wrist_roll의 원시 encoder tick(0~4095)을 읽는다. 순수 읽기 명령."""
        raw = self._bus.sync_read(
            "Present_Position", [TARGET_JOINT], normalize=False, num_retry=self._num_read_retries
        )
        return int(raw[TARGET_JOINT])

    def disconnect(self) -> None:
        """포트를 닫는다. torque 상태를 바꾸는 write는 절대 수행하지 않는다."""
        if not self._bus.is_connected:
            return
        self._bus.disconnect(disable_torque=False)


# ---------------------------------------------------------------------------
# 팔로워 read-only 상태 reader (필요한 레지스터만 최소로 읽어 read rate를 확보한다)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FollowerStateSnapshot:
    """한 시점에 읽은 팔로워 wrist_roll 상태 레지스터 (전부 raw)."""

    goal_raw: int | None
    present_raw: int | None
    torque_enable: int | None
    moving: int | None
    status_raw: int | None
    read_errors: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FollowerAccelSnapshot:
    """한 시점에 읽은 팔로워 wrist_roll 가속도 관련 레지스터 (전부 raw)."""

    acceleration: int | None
    acceleration_multiplier: int | None
    read_errors: dict = field(default_factory=dict)


class FollowerWristRollStateReader:
    """팔로워 wrist_roll(motor id 5) read-only 상태 reader - Shadow Teleop 전용.

    ``hardware/safety/single_joint_register_diagnostic.py``와 달리 필요한 레지스터만
    (``FOLLOWER_STATE_REGISTERS``/``FOLLOWER_ACCEL_REGISTERS``) 개별 ``bus.read()``로
    읽는다 - Present_Load/Present_Current 등 진단에 불필요한 레지스터를 매 샘플 읽지 않아
    read rate를 높게 유지한다. 공개 메서드는 ``connect``/``read_state``/``read_accel``/
    ``disconnect``/``is_connected``뿐이고, write에 해당하는 메서드는 하나도 없다.
    """

    def __init__(self, *, port: str, calibration: WristRollCalibration, num_read_retries: int = 2) -> None:
        self._num_read_retries = num_read_retries
        self.calibration = calibration
        self.port = port
        self._bus = build_wrist_roll_bus(port=port, calibration=calibration)
        _install_write_guard(self._bus)

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def connect(self) -> None:
        """직렬 포트를 열고 wrist_roll 모터만 ping/펌웨어 확인한다. 쓰기 없음."""
        if self._bus.is_connected:
            return
        self._bus.connect()

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
        """Goal_Position/Present_Position/Torque_Enable/Moving/Status를 read한다.

        레지스터 하나의 read가 실패해도 나머지는 계속 시도한다 - 실패한 레지스터는
        ``None``으로 남고 ``read_errors``에 사유가 기록된다.
        """
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
        """Acceleration/Acceleration_Multiplier를 read한다 (참고 정보, 저속 갱신 대상)."""
        values, errors = self._read_registers(FOLLOWER_ACCEL_REGISTERS)
        return FollowerAccelSnapshot(
            acceleration=values.get("Acceleration"),
            acceleration_multiplier=values.get("Acceleration_Multiplier "),
            read_errors=errors,
        )

    def disconnect(self) -> None:
        """포트를 닫는다. torque 상태를 바꾸는 write는 절대 수행하지 않는다."""
        if not self._bus.is_connected:
            return
        self._bus.disconnect(disable_torque=False)


# ---------------------------------------------------------------------------
# 샘플 + 샘플러
# ---------------------------------------------------------------------------

CSV_FIELDNAMES: tuple[str, ...] = (
    "sample_index",
    "timestamp",
    "elapsed_sec",
    "leader_wrist_roll_raw",
    "leader_wrist_roll_deg",
    "follower_goal_raw",
    "follower_goal_deg",
    "follower_present_raw",
    "follower_present_deg",
    "goal_present_error_raw",
    "goal_present_error_deg",
    "leader_vs_follower_present_deg",
    "follower_acceleration",
    "follower_acceleration_multiplier",
    "follower_torque_enable",
    "follower_moving",
    "follower_status",
    "leader_delta_from_start_deg",
    "follower_present_delta_from_start_deg",
)


@dataclass(frozen=True)
class ShadowSample:
    """한 시점의 리더+팔로워 관측 결과. CSV 한 행, 대시보드 한 프레임에 대응한다."""

    sample_index: int
    timestamp_iso: str
    elapsed_sec: float

    leader_wrist_roll_raw: int
    leader_wrist_roll_deg: float
    leader_delta_from_start_deg: float

    follower_goal_raw: int
    follower_goal_deg: float
    follower_present_raw: int
    follower_present_deg: float
    follower_present_delta_from_start_deg: float

    goal_present_error_raw: int
    goal_present_error_deg: float

    leader_vs_follower_present_deg: float

    follower_acceleration: int | None
    follower_acceleration_multiplier: int | None
    follower_torque_enable: int | None
    follower_moving: int | None
    follower_status: int | None

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "timestamp": self.timestamp_iso,
            "elapsed_sec": self.elapsed_sec,
            "leader_wrist_roll_raw": self.leader_wrist_roll_raw,
            "leader_wrist_roll_deg": self.leader_wrist_roll_deg,
            "follower_goal_raw": self.follower_goal_raw,
            "follower_goal_deg": self.follower_goal_deg,
            "follower_present_raw": self.follower_present_raw,
            "follower_present_deg": self.follower_present_deg,
            "goal_present_error_raw": self.goal_present_error_raw,
            "goal_present_error_deg": self.goal_present_error_deg,
            "leader_vs_follower_present_deg": self.leader_vs_follower_present_deg,
            "follower_acceleration": self.follower_acceleration,
            "follower_acceleration_multiplier": self.follower_acceleration_multiplier,
            "follower_torque_enable": self.follower_torque_enable,
            "follower_moving": self.follower_moving,
            "follower_status": self.follower_status,
            "leader_delta_from_start_deg": self.leader_delta_from_start_deg,
            "follower_present_delta_from_start_deg": self.follower_present_delta_from_start_deg,
        }


class ShadowTeleopSampler:
    """리더 reader + 팔로워 reader를 묶어 한 번에 ``ShadowSample`` 하나를 만든다.

    write는 절대 하지 않는다 - ``leader_reader``/``follower_reader``가 이미 read-only이고,
    이 클래스는 그 반환값을 조합/변환만 한다(순수 계산 + 시각/시퀀스 번호 부여).
    """

    def __init__(
        self,
        *,
        leader_reader: LeaderWristRollReader,
        follower_reader: FollowerWristRollStateReader,
        leader_calibration: WristRollCalibration,
        follower_calibration: WristRollCalibration,
        accel_refresh_interval_s: float = DEFAULT_ACCEL_REFRESH_INTERVAL_S,
        motor_resolution: int = DEFAULT_MOTOR_RESOLUTION,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._leader = leader_reader
        self._follower = follower_reader
        self._leader_calibration = leader_calibration
        self._follower_calibration = follower_calibration
        self._accel_refresh_interval_s = accel_refresh_interval_s
        self._motor_resolution = motor_resolution
        self._clock = clock
        self._wall_clock = wall_clock

        self._start_monotonic: float | None = None
        self._leader_start_deg: float | None = None
        self._follower_start_present_deg: float | None = None
        self._follower_start_goal_deg: float | None = None

        self._cached_accel: FollowerAccelSnapshot | None = None
        self._cached_accel_read_monotonic: float | None = None

        self._sample_count = 0

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def leader_start_deg(self) -> float | None:
        return self._leader_start_deg

    @property
    def follower_start_present_deg(self) -> float | None:
        return self._follower_start_present_deg

    @property
    def follower_start_goal_deg(self) -> float | None:
        return self._follower_start_goal_deg

    def _leader_deg(self, raw: int) -> float:
        return raw_to_degrees(
            raw,
            range_min=self._leader_calibration.range_min,
            range_max=self._leader_calibration.range_max,
            motor_resolution=self._motor_resolution,
        )

    def _follower_deg(self, raw: int) -> float:
        return raw_to_degrees(
            raw,
            range_min=self._follower_calibration.range_min,
            range_max=self._follower_calibration.range_max,
            motor_resolution=self._motor_resolution,
        )

    def sample(self) -> ShadowSample:
        """리더+팔로워를 한 번씩 read해서 ``ShadowSample`` 하나를 만든다.

        필수 레지스터(리더 Present_Position, 팔로워 Goal_Position/Present_Position) read가
        실패하면 ``ShadowTeleopReadError``를 던진다 - 호출부(``run_sampling_loop``)는 이
        예외를 받으면 재시도하지 않고 안전 종료해야 한다. Torque_Enable/Moving/Status/
        Acceleration/Acceleration_Multiplier는 선택 정보라 read 실패 시 해당 필드만
        ``None``으로 남기고 계속 진행한다.
        """
        now_mono = self._clock()
        if self._start_monotonic is None:
            self._start_monotonic = now_mono
        elapsed = now_mono - self._start_monotonic

        try:
            leader_raw = self._leader.read_raw()
        except Exception as exc:  # noqa: BLE001 - 통신 오류를 폭넓게 잡아 안전 종료 신호로 변환
            raise ShadowTeleopReadError(f"리더 wrist_roll read 실패: {exc}") from exc
        leader_deg = self._leader_deg(leader_raw)

        try:
            follower_state = self._follower.read_state()
        except Exception as exc:  # noqa: BLE001
            raise ShadowTeleopReadError(f"팔로워 wrist_roll 상태 read 실패: {exc}") from exc

        if follower_state.present_raw is None or follower_state.goal_raw is None:
            raise ShadowTeleopReadError(
                f"팔로워 필수 레지스터(Goal_Position/Present_Position) read 실패: {follower_state.read_errors}"
            )

        follower_present_deg = self._follower_deg(follower_state.present_raw)
        follower_goal_deg = self._follower_deg(follower_state.goal_raw)

        if (
            self._cached_accel is None
            or self._cached_accel_read_monotonic is None
            or (now_mono - self._cached_accel_read_monotonic) >= self._accel_refresh_interval_s
        ):
            try:
                self._cached_accel = self._follower.read_accel()
            except Exception as exc:  # noqa: BLE001 - accel은 선택 정보라 실패해도 진행하되 사유는 남긴다
                self._cached_accel = FollowerAccelSnapshot(
                    acceleration=None, acceleration_multiplier=None, read_errors={"accel_read": str(exc)}
                )
            self._cached_accel_read_monotonic = now_mono

        if self._leader_start_deg is None:
            self._leader_start_deg = leader_deg
            self._follower_start_present_deg = follower_present_deg
            self._follower_start_goal_deg = follower_goal_deg

        sample = ShadowSample(
            sample_index=self._sample_count,
            timestamp_iso=self._wall_clock().isoformat(),
            elapsed_sec=elapsed,
            leader_wrist_roll_raw=leader_raw,
            leader_wrist_roll_deg=leader_deg,
            leader_delta_from_start_deg=leader_deg - self._leader_start_deg,
            follower_goal_raw=follower_state.goal_raw,
            follower_goal_deg=follower_goal_deg,
            follower_present_raw=follower_state.present_raw,
            follower_present_deg=follower_present_deg,
            follower_present_delta_from_start_deg=follower_present_deg - self._follower_start_present_deg,
            goal_present_error_raw=follower_state.goal_raw - follower_state.present_raw,
            goal_present_error_deg=follower_goal_deg - follower_present_deg,
            leader_vs_follower_present_deg=leader_deg - follower_present_deg,
            follower_acceleration=self._cached_accel.acceleration,
            follower_acceleration_multiplier=self._cached_accel.acceleration_multiplier,
            follower_torque_enable=follower_state.torque_enable,
            follower_moving=follower_state.moving,
            follower_status=follower_state.status_raw,
        )
        self._sample_count += 1
        return sample


# ---------------------------------------------------------------------------
# 샘플링 루프 (duration/Ctrl+C/read 실패 - 안전 종료를 테스트 가능한 형태로 분리)
# ---------------------------------------------------------------------------


@dataclass
class SamplingRunResult:
    samples: list[ShadowSample]
    stopped_reason: str  # "duration_elapsed" | "keyboard_interrupt" | "read_error"
    error: Exception | None = None


def run_sampling_loop(
    sampler: ShadowTeleopSampler,
    *,
    duration_sec: float | None,
    on_sample: Callable[[ShadowSample], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    follower_move_abort_threshold_deg: float | None = None,
) -> SamplingRunResult:
    """샘플링 루프 본체 - 하드웨어 연결/disconnect는 호출부(CLI) 책임이다.

    네 가지 방식으로만 멈춘다: 요청한 시간 경과(``duration_elapsed``), 사용자
    Ctrl+C(``keyboard_interrupt``), read 실패(``read_error`` - ``sampler.sample()``이
    ``ShadowTeleopReadError``를 던졌을 때, 재시도하지 않고 즉시 멈춘다), 또는
    ``follower_move_abort_threshold_deg``가 주어졌고 ``follower_present_delta_from_start_deg``의
    절대값이 그 문턱값을 넘었을 때(``follower_moved_unexpectedly`` - 섹션 10 안전장치: "리더를
    움직여도 팔로워가 움직이면 즉시 실험을 중단한다"). 네 경우 모두 지금까지 모은 ``samples``를
    그대로 반환한다 - 예외를 호출부까지 전파하지 않는다(호출부가 disconnect/CSV flush를 항상
    실행할 수 있게 하기 위함). ``follower_moved_unexpectedly``로 멈춘 마지막 샘플도
    ``on_sample``에 전달된 뒤 멈춘다(CSV/대시보드에 그 샘플까지 기록되어야 원인 분석이
    가능하다).
    """
    samples: list[ShadowSample] = []
    start = clock()
    try:
        while True:
            if duration_sec is not None and (clock() - start) >= duration_sec:
                return SamplingRunResult(samples=samples, stopped_reason="duration_elapsed")
            try:
                sample = sampler.sample()
            except ShadowTeleopReadError as exc:
                return SamplingRunResult(samples=samples, stopped_reason="read_error", error=exc)
            samples.append(sample)

            move_triggered = (
                follower_move_abort_threshold_deg is not None
                and abs(sample.follower_present_delta_from_start_deg) > follower_move_abort_threshold_deg
            )

            if on_sample is not None:
                on_sample(sample)

            if move_triggered:
                error = FollowerMovedUnexpectedlyError(
                    "follower_present_delta_from_start_deg="
                    f"{sample.follower_present_delta_from_start_deg:.4f}°가 안전 문턱값 "
                    f"±{follower_move_abort_threshold_deg}°를 초과했습니다 - follower가 실제로 움직였을 "
                    "가능성이 있어 즉시 중단합니다."
                )
                return SamplingRunResult(samples=samples, stopped_reason="follower_moved_unexpectedly", error=error)
    except KeyboardInterrupt:
        return SamplingRunResult(samples=samples, stopped_reason="keyboard_interrupt")


# ---------------------------------------------------------------------------
# 분석 (섹션 11) - 순수 계산, 하드웨어/파일 접근 없음
# ---------------------------------------------------------------------------


def compute_run_analysis(samples: list[ShadowSample]) -> dict[str, Any]:
    """수집된 샘플로 섹션 11이 요구하는 요약 통계를 계산한다."""

    if not samples:
        return {"sample_count": 0}

    n = len(samples)
    elapsed = samples[-1].elapsed_sec - samples[0].elapsed_sec
    actual_rate_hz = (n - 1) / elapsed if elapsed > 0 else 0.0

    leader_degs = [s.leader_wrist_roll_deg for s in samples]
    leader_deltas = [s.leader_delta_from_start_deg for s in samples]
    follower_present_degs = [s.follower_present_deg for s in samples]
    goal_present_errors_deg = [s.goal_present_error_deg for s in samples]
    accel_values = {s.follower_acceleration for s in samples if s.follower_acceleration is not None}
    torque_values = {s.follower_torque_enable for s in samples if s.follower_torque_enable is not None}
    moving_ever_nonzero = any((s.follower_moving or 0) != 0 for s in samples)
    status_ever_nonzero = any((s.follower_status or 0) != 0 for s in samples)

    follower_present_range_deg = max(follower_present_degs) - min(follower_present_degs)
    leader_range_deg = max(leader_degs) - min(leader_degs)
    leader_moved = leader_range_deg > FOLLOWER_FIXED_THRESHOLD_DEG
    follower_fixed = follower_present_range_deg <= FOLLOWER_FIXED_THRESHOLD_DEG

    return {
        "sample_count": n,
        "elapsed_sec": elapsed,
        "actual_sample_rate_hz": actual_rate_hz,
        "leader_wrist_roll_deg_min": min(leader_degs),
        "leader_wrist_roll_deg_max": max(leader_degs),
        "leader_range_deg": leader_range_deg,
        "leader_delta_deg_max_abs": max(abs(d) for d in leader_deltas),
        "follower_present_deg_min": min(follower_present_degs),
        "follower_present_deg_max": max(follower_present_degs),
        "follower_present_deg_range": follower_present_range_deg,
        "goal_present_error_deg_min": min(goal_present_errors_deg),
        "goal_present_error_deg_max": max(goal_present_errors_deg),
        "acceleration_values_seen": sorted(accel_values),
        "acceleration_changed": len(accel_values) > 1,
        "torque_enable_values_seen": sorted(torque_values),
        "torque_enable_changed": len(torque_values) > 1,
        "moving_ever_nonzero": moving_ever_nonzero,
        "status_ever_nonzero": status_ever_nonzero,
        "leader_moved_while_follower_fixed": leader_moved and follower_fixed,
        "follower_fixed_threshold_deg": FOLLOWER_FIXED_THRESHOLD_DEG,
        "write_count": 0,
    }


# ---------------------------------------------------------------------------
# 섹션 12: 향후 restricted teleop 실험용 tick<->degree 참고표 (계산만, write 없음)
# ---------------------------------------------------------------------------


def degrees_per_tick_for_calibration(
    calibration: WristRollCalibration, *, motor_resolution: int = DEFAULT_MOTOR_RESOLUTION
) -> float:
    """이 calibration에서 tick 1개가 몇 도인지 실제로 두 인접 tick을 변환해서 계산한다.

    ``raw_to_degrees``의 DEGREES 공식(``(raw-mid)*360/(motor_resolution-1)``)은
    range_min/range_max가 offset(``mid``)에만 영향을 주고 스케일 자체에는 영향을 주지 않는다
    (즉 이론상 모든 calibration에서 동일한 값이 나온다) - 하지만 이 함수는 그 사실을
    가정하지 않고, 실제로 두 인접 tick(0, 1)을 이 calibration으로 변환해 차이를 계산한다
    (요구사항: 값이 얼마인지는 always 계산으로 확인, 추측 금지).
    """
    deg_at_0 = raw_to_degrees(
        0, range_min=calibration.range_min, range_max=calibration.range_max, motor_resolution=motor_resolution
    )
    deg_at_1 = raw_to_degrees(
        1, range_min=calibration.range_min, range_max=calibration.range_max, motor_resolution=motor_resolution
    )
    return deg_at_1 - deg_at_0


_REFERENCE_DEGREE_TARGETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0)
_REFERENCE_TICK_TARGETS: tuple[int, ...] = (1, 2, 3, 4, 5)


def build_command_delta_reference_table(
    *,
    leader_calibration: WristRollCalibration,
    follower_calibration: WristRollCalibration,
    motor_resolution: int = DEFAULT_MOTOR_RESOLUTION,
) -> dict[str, Any]:
    """섹션 12 요구사항: 실제 Goal_Position write 없이 tick<->degree 환산표만 계산한다."""

    leader_deg_per_tick = degrees_per_tick_for_calibration(leader_calibration, motor_resolution=motor_resolution)
    follower_deg_per_tick = degrees_per_tick_for_calibration(follower_calibration, motor_resolution=motor_resolution)

    def _table(deg_per_tick: float) -> dict[str, Any]:
        return {
            "deg_per_tick": deg_per_tick,
            "degrees_to_ticks": {f"{d:g}_deg_in_ticks": d / deg_per_tick for d in _REFERENCE_DEGREE_TARGETS},
            "ticks_to_degrees": {f"{t}_tick_in_deg": t * deg_per_tick for t in _REFERENCE_TICK_TARGETS},
        }

    return {
        "leader": _table(leader_deg_per_tick),
        "follower": _table(follower_deg_per_tick),
        "leader_and_follower_scale_equal": abs(leader_deg_per_tick - follower_deg_per_tick) < 1e-9,
        "note": (
            "이 표는 raw_to_degrees의 DEGREES 공식을 계산만 한 참고값이다. 실제 Goal_Position "
            "write는 이 모듈 어디에서도 수행하지 않았다 - 향후 restricted teleop 실험에서 "
            "최소 command delta 후보를 고를 때 참고하라는 목적일 뿐이다."
        ),
    }
