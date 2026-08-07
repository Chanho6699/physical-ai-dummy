"""리더암 → "실제 팔로워에 보낼 예정인" 안전 명령 매퍼 (MuJoCo 전용 시뮬레이션).

**이 모듈은 실물 팔로워에 어떤 것도 쓰지 않는다.** 여기서 계산한 ``limited_command_deg``는
오직 MuJoCo 시뮬레이션에만 적용된다 (``live_web_viewer.py``의 ``--command-source
follower-safe`` 모드). 하드웨어 SDK(FeetechMotorsBus 등)를 import하지 않고, write 메서드도
정의하지 않는다.

파이프라인은 요구사항대로 6단계로 명시적으로 분리한다::

    A. raw leader state           (positions_deg 그대로, 이미 degree 단위)
    B. follower coordinate mapping (leader_to_follower_sign - 기본 항등, 확인 안 됨)
    C. absolute follower safe range check (캘리브레이션에서 구한 안전 범위, clamp 아님 - hold)
    D. rate limit                 (max_step = rate_deg_per_sec * elapsed_sec, 시간 기반)
    E. stale/sequence/mode/connection/invalid hold (직전 안전 명령 유지)
    F. final limited command      (A~E를 종합한 최종 값 + hold 여부/사유)

raw tick -> degree 변환은 LeRobot ``MotorNormMode.DEGREES`` 정규화 공식을 그대로 재사용한다
(``~/lerobot/src/lerobot/motors/motors_bus.py`` 의 ``MotorsBus._normalize`` 조사 결과):

    mid = (range_min + range_max) / 2
    max_res = motor_resolution - 1
    degree = (raw_tick - mid) * 360 / max_res

``homing_offset``은 이미 서보 펌웨어(Homing_Offset 레지스터)에 반영되어 있어 raw
Present_Position에 이미 녹아 있다 - 소프트웨어에서 다시 빼면 이중 보정이므로 이 모듈은
``homing_offset``을 계산에 쓰지 않는다 (``configs/follower_safe_mapper.yaml`` 상단 주석 참고).

gripper는 이 프로젝트 전체에서 DEGREES가 아니라 ``RANGE_0_100``(퍼센트) 정규화를 쓴다
(``hardware/state_server/readonly_so101_reader.py``) - 이 매퍼가 다루는 "도" 단위와 맞지
않으므로 gripper는 ``UNVERIFIED_RANGE``로 표시하고 실제 출력 후보에서 제외한다. 이 파일은
그 판단을 추측으로 뒤집지 않는다.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from hardware.state_server.calibration_loader import CalibrationLoadError, load_calibration_file
from simulation.mujoco.remote_state_client import JOINT_NAMES

# gripper는 RANGE_0_100(퍼센트) 정규화라 이 매퍼의 DEGREES 가정과 맞지 않는다 - 확정할 수
# 없는 관절로 취급해 실제 출력 후보에서 제외한다 (요구사항 2번/7번).
UNVERIFIED_JOINTS: frozenset[str] = frozenset({"gripper"})

HOLD_REASONS: tuple[str, ...] = (
    "REMOTE_STALE",
    "SEQUENCE_STALLED",
    "INVALID_LEADER_STATE",
    "MISSING_JOINT",
    "MODE_NOT_READ_ONLY",
    "CONNECTION_LOST",
    "INVALID_VALUE",
    "RANGE_VIOLATION",
    "UNVERIFIED_RANGE",
    "NOT_INITIALIZED",
)

# "연결/프레임 전체" 문제라서 6관절 전부를 hold해야 하는 원인만 여기 포함한다. 나머지
# (RANGE_VIOLATION/UNVERIFIED_RANGE/MISSING_JOINT/INVALID_VALUE/NOT_INITIALIZED)는 항상
# 해당 관절 하나만 hold한다 - 다른 관절에 전혀 전파되지 않는다 (_step_joint가 매 관절마다
# self._calibrations[joint]/self._initialized[joint]만 보고 독립적으로 판단하기 때문에,
# 구조적으로 한 관절의 hold_reason이 다른 관절 계산에 끼어들 방법이 없다).
ARM_WIDE_HOLD_REASONS: frozenset[str] = frozenset(
    {"REMOTE_STALE", "SEQUENCE_STALLED", "MODE_NOT_READ_ONLY", "CONNECTION_LOST"}
)


def summarize_hold(samples: dict[str, "FollowerSafeSample"]) -> dict:
    """요구사항: global_hold/active_joint_count/held_joint_count/held_joints 요약.

    global_hold는 ARM_WIDE_HOLD_REASONS 중 하나가 하나라도 관측된 경우에만 True다 -
    UNVERIFIED_RANGE(예: gripper) 하나만으로는 절대 global_hold가 되지 않는다.
    """
    global_hold = False
    global_hold_reason: str | None = None
    held_joints: dict[str, str] = {}
    for joint, sample in samples.items():
        if sample.hold_reason is not None:
            held_joints[joint] = sample.hold_reason
            if sample.hold_reason in ARM_WIDE_HOLD_REASONS and not global_hold:
                global_hold = True
                global_hold_reason = sample.hold_reason
    return {
        "global_hold": global_hold,
        "global_hold_reason": global_hold_reason,
        "active_joint_count": len(samples) - len(held_joints),
        "held_joint_count": len(held_joints),
        "held_joints": held_joints,
    }


class FollowerSafeMapperError(RuntimeError):
    """캘리브레이션 로딩 실패 등 - preflight에서 그대로 한글 오류로 올린다."""


# ---------------------------------------------------------------------------
# 캘리브레이션
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FollowerJointCalibration:
    joint: str
    verified: bool  # False면 UNVERIFIED_RANGE 대상 (예: gripper)
    range_min_deg: float | None  # verified=False면 None
    range_max_deg: float | None
    raw_min: int | None = None  # 참고/리포트용
    raw_max: int | None = None
    source: str = ""  # "calibration_file" | "fallback_config" | "unverified"


def _ticks_to_degrees(raw_min: int, raw_max: int, motor_resolution: int) -> tuple[float, float]:
    """LeRobot MotorNormMode.DEGREES 공식 그대로 (motors_bus.py._normalize 참고)."""
    mid = (raw_min + raw_max) / 2.0
    max_res = motor_resolution - 1
    lo = (raw_min - mid) * 360.0 / max_res
    hi = (raw_max - mid) * 360.0 / max_res
    return (lo, hi) if lo <= hi else (hi, lo)


def load_follower_calibration(
    *,
    calibration_file_path: str | Path | None,
    fallback_raw_range: dict[str, dict[str, int]] | None,
    motor_resolution: int,
) -> dict[str, FollowerJointCalibration]:
    """팔로워 안전 범위를 구성한다. 우선순위: 파일 -> config fallback -> 오류.

    Raises:
        FollowerSafeMapperError: 파일도 없고 fallback도 없거나, 파일이 있는데 읽기
            실패했고 fallback도 없는 경우.
    """
    raw_ranges: dict[str, tuple[int, int]] | None = None
    source = ""

    if calibration_file_path is not None and Path(calibration_file_path).expanduser().is_file():
        try:
            entries = load_calibration_file(calibration_file_path)
        except CalibrationLoadError as exc:
            if not fallback_raw_range:
                raise FollowerSafeMapperError(
                    f"팔로워 캘리브레이션 파일을 읽는 데 실패했고 fallback_raw_range도 없습니다: {exc}"
                ) from exc
            raw_ranges = None  # fallback으로 진행
        else:
            raw_ranges = {joint: (e.range_min, e.range_max) for joint, e in entries.items()}
            source = "calibration_file"

    if raw_ranges is None:
        if not fallback_raw_range:
            raise FollowerSafeMapperError(
                "팔로워 캘리브레이션 파일이 없고(또는 읽기 실패) fallback_raw_range도 설정되지 않았습니다. "
                "configs/follower_safe_mapper.yaml의 calibration_file_path 또는 fallback_raw_range를 확인하세요."
            )
        missing = [j for j in JOINT_NAMES if j not in fallback_raw_range]
        if missing:
            raise FollowerSafeMapperError(f"fallback_raw_range에 다음 관절이 없습니다: {missing}")
        raw_ranges = {}
        for joint in JOINT_NAMES:
            entry = fallback_raw_range[joint]
            raw_ranges[joint] = (int(entry["min"]), int(entry["max"]))
        source = "fallback_config"

    result: dict[str, FollowerJointCalibration] = {}
    for joint in JOINT_NAMES:
        raw_min, raw_max = raw_ranges[joint]
        if joint in UNVERIFIED_JOINTS:
            result[joint] = FollowerJointCalibration(
                joint=joint, verified=False, range_min_deg=None, range_max_deg=None, raw_min=raw_min, raw_max=raw_max, source="unverified"
            )
            continue
        if raw_min == raw_max:
            raise FollowerSafeMapperError(f"'{joint}' 캘리브레이션 range_min == range_max 입니다 (잘못된 캘리브레이션).")
        lo_deg, hi_deg = _ticks_to_degrees(raw_min, raw_max, motor_resolution)
        result[joint] = FollowerJointCalibration(
            joint=joint, verified=True, range_min_deg=lo_deg, range_max_deg=hi_deg, raw_min=raw_min, raw_max=raw_max, source=source
        )
    return result


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FollowerSafeMapperConfig:
    calibration_file_path: str | None = None
    fallback_raw_range: dict[str, dict[str, int]] | None = None
    motor_resolution: int = 4096
    leader_to_follower_sign: dict[str, float] = field(default_factory=lambda: {name: 1.0 for name in JOINT_NAMES})
    rate_limit_deg_per_sec: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing_rate = [j for j in JOINT_NAMES if j not in self.rate_limit_deg_per_sec]
        if missing_rate:
            raise FollowerSafeMapperError(f"rate_limit_deg_per_sec에 다음 관절이 없습니다: {missing_rate}")
        for joint, rate in self.rate_limit_deg_per_sec.items():
            if rate <= 0:
                raise FollowerSafeMapperError(f"rate_limit_deg_per_sec['{joint}']는 0보다 커야 합니다: {rate}")


def load_config_yaml(path: str | Path) -> FollowerSafeMapperConfig:
    import yaml

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FollowerSafeMapperError(f"follower_safe_mapper 설정 파일을 찾을 수 없습니다: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise FollowerSafeMapperError(f"설정 YAML 최상위는 object여야 합니다: {resolved}")

    # 기본값(항등 sign=+1)을 얻으려고 FollowerSafeMapperConfig()를 그냥 생성하면 안 된다 -
    # __post_init__이 rate_limit_deg_per_sec 6개 관절을 즉시 검증해서 예외를 던진다.
    sign = {name: 1.0 for name in JOINT_NAMES}
    sign.update({k: float(v) for k, v in (raw.get("leader_to_follower_sign") or {}).items()})

    rate = {k: float(v) for k, v in (raw.get("rate_limit_deg_per_sec") or {}).items()}

    return FollowerSafeMapperConfig(
        calibration_file_path=raw.get("calibration_file_path"),
        fallback_raw_range=raw.get("fallback_raw_range"),
        motor_resolution=int(raw.get("motor_resolution", 4096)),
        leader_to_follower_sign=sign,
        rate_limit_deg_per_sec=rate,
    )


# ---------------------------------------------------------------------------
# 프레임당 결과
# ---------------------------------------------------------------------------


@dataclass
class FollowerSafeSample:
    joint: str
    raw_leader_deg: float | None
    mapped_follower_deg: float | None  # B단계 결과 (좌표 변환만, safety 이전)
    limited_command_deg: float | None  # F단계 최종 - MuJoCo에 실제 적용되는 값
    hold: bool
    hold_reason: str | None
    rate_limited: bool
    range_held: bool
    connection_held: bool
    intervention_flags: tuple[str, ...]
    range_min_deg: float | None
    range_max_deg: float | None
    elapsed_sec: float
    max_step_deg: float | None

    def to_dict(self) -> dict:
        return {
            "joint": self.joint,
            "raw_leader_deg": self.raw_leader_deg,
            "mapped_follower_deg": self.mapped_follower_deg,
            "limited_command_deg": self.limited_command_deg,
            "hold": self.hold,
            "hold_reason": self.hold_reason,
            "rate_limited": self.rate_limited,
            "range_held": self.range_held,
            "connection_held": self.connection_held,
            "intervention_flags": list(self.intervention_flags),
            "range_min_deg": self.range_min_deg,
            "range_max_deg": self.range_max_deg,
            "elapsed_sec": self.elapsed_sec,
            "max_step_deg": self.max_step_deg,
        }


@dataclass(frozen=True)
class ConnectionHoldInput:
    """render loop가 이미 계산해 둔 전체(연결 단위) hold 조건 - 여기서 새로 판정하지 않는다."""

    remote_stale: bool = False
    sequence_stalled: bool = False
    mode_not_read_only: bool = False
    connection_lost: bool = False


# ---------------------------------------------------------------------------
# 매퍼 본체
# ---------------------------------------------------------------------------


class FollowerSafeMapper:
    """관절별 ``last_command_deg`` 상태를 들고 매 프레임 A~F 단계를 계산한다."""

    def __init__(self, config: FollowerSafeMapperConfig, calibrations: dict[str, FollowerJointCalibration]) -> None:
        self._config = config
        self._calibrations = calibrations
        self._last_command_deg: dict[str, float | None] = {name: None for name in JOINT_NAMES}
        self._last_step_time: float | None = None
        self._initialized: dict[str, bool] = {name: False for name in JOINT_NAMES}
        self.intervention_count = 0
        self.sample_count = 0

    @property
    def calibrations(self) -> dict[str, FollowerJointCalibration]:
        return self._calibrations

    def last_command_snapshot(self) -> dict[str, float | None]:
        return dict(self._last_command_deg)

    def step(
        self,
        *,
        now: float,
        leader_positions_deg: dict[str, float] | None,
        follower_positions_deg: dict[str, float] | None,
        connection_hold: ConnectionHoldInput,
    ) -> dict[str, FollowerSafeSample]:
        """한 프레임 분의 A~F 단계를 6개 관절 전부에 대해 계산한다."""
        elapsed = 0.0 if self._last_step_time is None else max(0.0, now - self._last_step_time)
        self._last_step_time = now

        arm_wide_hold_reason: str | None = None
        if connection_hold.mode_not_read_only:
            arm_wide_hold_reason = "MODE_NOT_READ_ONLY"
        elif connection_hold.connection_lost:
            arm_wide_hold_reason = "CONNECTION_LOST"
        elif connection_hold.sequence_stalled:
            arm_wide_hold_reason = "SEQUENCE_STALLED"
        elif connection_hold.remote_stale:
            arm_wide_hold_reason = "REMOTE_STALE"

        results: dict[str, FollowerSafeSample] = {}
        for joint in JOINT_NAMES:
            results[joint] = self._step_joint(
                joint,
                elapsed=elapsed,
                leader_positions_deg=leader_positions_deg,
                follower_positions_deg=follower_positions_deg,
                arm_wide_hold_reason=arm_wide_hold_reason,
            )
        self.sample_count += 1
        return results

    def _step_joint(
        self,
        joint: str,
        *,
        elapsed: float,
        leader_positions_deg: dict[str, float] | None,
        follower_positions_deg: dict[str, float] | None,
        arm_wide_hold_reason: str | None,
    ) -> FollowerSafeSample:
        cal = self._calibrations[joint]
        last_command = self._last_command_deg[joint]

        # -- A. raw leader state ------------------------------------------------
        raw_leader_deg = leader_positions_deg.get(joint) if leader_positions_deg else None
        invalid_value = raw_leader_deg is not None and not math.isfinite(raw_leader_deg)
        missing_joint = leader_positions_deg is not None and joint not in leader_positions_deg

        # -- B. follower coordinate mapping (좌표 변환만, 아직 safety 아님) -------
        sign = self._config.leader_to_follower_sign.get(joint, 1.0)
        mapped_follower_deg: float | None = None
        if raw_leader_deg is not None and math.isfinite(raw_leader_deg):
            mapped_follower_deg = raw_leader_deg * sign

        # -- 초기화 (첫 유효 snapshot) --------------------------------------------
        if not self._initialized[joint]:
            follower_current = follower_positions_deg.get(joint) if follower_positions_deg else None
            if follower_current is not None and math.isfinite(follower_current):
                # 요구사항: 가능하면 follower current position을 초기 command 기준으로.
                last_command = follower_current * sign
                self._initialized[joint] = True
            elif mapped_follower_deg is not None:
                # follower state가 아직 없으면 첫 리더 값으로 초기화 (점프할 "이전 값"이
                # 없는 첫 프레임이므로 이 자체는 점프가 아니다).
                last_command = mapped_follower_deg
                self._initialized[joint] = True
            self._last_command_deg[joint] = last_command

        # -- C. absolute follower safe range check (clamp 아님, hold만) ----------
        range_held = False
        if not cal.verified:
            pass  # UNVERIFIED_RANGE에서 별도 처리 (아래)
        elif mapped_follower_deg is not None and cal.range_min_deg is not None and cal.range_max_deg is not None:
            if mapped_follower_deg < cal.range_min_deg or mapped_follower_deg > cal.range_max_deg:
                range_held = True

        # -- D. rate limit (시간 기반, 항상 계산 - 진단/표시용) --------------------
        rate = self._config.rate_limit_deg_per_sec.get(joint, 0.0)
        max_step = rate * elapsed
        rate_limited = False
        rate_limited_candidate: float | None = None
        if mapped_follower_deg is not None and last_command is not None:
            delta = mapped_follower_deg - last_command
            if abs(delta) > max_step:
                rate_limited = True
                rate_limited_candidate = last_command + math.copysign(max_step, delta)
            else:
                rate_limited_candidate = mapped_follower_deg

        # -- E. stale/sequence/mode/connection/invalid hold -----------------------
        hold_reason: str | None = None
        if not cal.verified:
            hold_reason = "UNVERIFIED_RANGE"
        elif not self._initialized[joint]:
            hold_reason = "NOT_INITIALIZED"
        elif missing_joint:
            hold_reason = "MISSING_JOINT"
        elif invalid_value:
            hold_reason = "INVALID_VALUE"
        elif arm_wide_hold_reason is not None:
            hold_reason = arm_wide_hold_reason
        elif range_held:
            hold_reason = "RANGE_VIOLATION"

        connection_held = arm_wide_hold_reason is not None

        # -- F. final limited command ---------------------------------------------
        if hold_reason is not None:
            final = last_command  # 직전 안전값 유지 (clamp 아님)
        else:
            final = rate_limited_candidate
            self._last_command_deg[joint] = final

        intervention_flags: list[str] = []
        if hold_reason is not None:
            intervention_flags.append(hold_reason)
        if rate_limited and hold_reason is None:
            intervention_flags.append("RATE_LIMITED")
        if intervention_flags:
            self.intervention_count += 1

        return FollowerSafeSample(
            joint=joint,
            raw_leader_deg=raw_leader_deg,
            mapped_follower_deg=mapped_follower_deg,
            limited_command_deg=final,
            hold=hold_reason is not None,
            hold_reason=hold_reason,
            rate_limited=rate_limited,
            range_held=range_held,
            connection_held=connection_held,
            intervention_flags=tuple(intervention_flags),
            range_min_deg=cal.range_min_deg,
            range_max_deg=cal.range_max_deg,
            elapsed_sec=elapsed,
            max_step_deg=max_step if rate > 0 else None,
        )


# ---------------------------------------------------------------------------
# 진단 리포트 (reports/remote_mujoco_diagnostic/follower_safe_mapper_<timestamp>.json/.csv)
# ---------------------------------------------------------------------------

CSV_FIELDNAMES: tuple[str, ...] = (
    "timestamp",
    "remote_sequence",
    "joint",
    "raw_leader_deg",
    "follower_current_deg",
    "mapped_target_deg",
    "limited_command_deg",
    "mujoco_actual_deg",
    "rate_limited",
    "range_held",
    "connection_held",
    "hold_reason",
    "elapsed_sec",
    "max_step_deg",
)


def make_report_session_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def resolve_follower_safe_report_paths(reports_dir: Path, session_id: str) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    return (
        reports_dir / f"follower_safe_mapper_{session_id}.json",
        reports_dir / f"follower_safe_mapper_{session_id}.csv",
    )


class FollowerSafeRecorder:
    """매 프레임 A~F 결과를 누적해 요구사항 10번 스키마대로 JSON/CSV를 만든다."""

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._total_samples = 0
        self._rate_limit_interventions = 0
        self._range_holds = 0
        self._stale_holds = 0
        self._sequence_holds = 0
        self._invalid_holds = 0
        self._other_holds = 0
        self._max_raw_delta_by_joint: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        self._max_limited_delta_by_joint: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        self._prev_raw: dict[str, float | None] = {name: None for name in JOINT_NAMES}
        self._prev_limited: dict[str, float | None] = {name: None for name in JOINT_NAMES}

    def record(
        self,
        *,
        now_wall: float,
        remote_sequence: int | None,
        samples: dict[str, FollowerSafeSample],
        follower_current_deg: dict[str, float] | None,
        mujoco_actual_deg: dict[str, float] | None,
    ) -> None:
        self._total_samples += 1
        for joint, s in samples.items():
            if s.rate_limited:
                self._rate_limit_interventions += 1
            if s.hold_reason == "RANGE_VIOLATION":
                self._range_holds += 1
            elif s.hold_reason == "REMOTE_STALE":
                self._stale_holds += 1
            elif s.hold_reason == "SEQUENCE_STALLED":
                self._sequence_holds += 1
            elif s.hold_reason in ("INVALID_VALUE", "MISSING_JOINT"):
                self._invalid_holds += 1
            elif s.hold_reason is not None:
                self._other_holds += 1

            if s.raw_leader_deg is not None:
                prev = self._prev_raw[joint]
                if prev is not None:
                    self._max_raw_delta_by_joint[joint] = max(self._max_raw_delta_by_joint[joint], abs(s.raw_leader_deg - prev))
                self._prev_raw[joint] = s.raw_leader_deg
            if s.limited_command_deg is not None:
                prev_l = self._prev_limited[joint]
                if prev_l is not None:
                    self._max_limited_delta_by_joint[joint] = max(
                        self._max_limited_delta_by_joint[joint], abs(s.limited_command_deg - prev_l)
                    )
                self._prev_limited[joint] = s.limited_command_deg

            self._rows.append(
                {
                    "timestamp": now_wall,
                    "remote_sequence": remote_sequence,
                    "joint": joint,
                    "raw_leader_deg": s.raw_leader_deg,
                    "follower_current_deg": (follower_current_deg or {}).get(joint),
                    "mapped_target_deg": s.mapped_follower_deg,
                    "limited_command_deg": s.limited_command_deg,
                    "mujoco_actual_deg": (mujoco_actual_deg or {}).get(joint),
                    "rate_limited": s.rate_limited,
                    "range_held": s.range_held,
                    "connection_held": s.connection_held,
                    "hold_reason": s.hold_reason,
                    "elapsed_sec": s.elapsed_sec,
                    "max_step_deg": s.max_step_deg,
                }
            )

    def summary(self) -> dict:
        return {
            "total_samples": self._total_samples,
            "rate_limit_interventions": self._rate_limit_interventions,
            "range_holds": self._range_holds,
            "stale_holds": self._stale_holds,
            "sequence_holds": self._sequence_holds,
            "invalid_holds": self._invalid_holds,
            "other_holds": self._other_holds,
            "max_raw_delta_by_joint": dict(self._max_raw_delta_by_joint),
            "max_limited_delta_by_joint": dict(self._max_limited_delta_by_joint),
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "samples": self._rows}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})


def print_follower_safe_debug(samples: dict[str, FollowerSafeSample]) -> None:
    """관절별 raw/mapped/limited/joint_hold(+사유) + global_hold(+사유)를 stdout에 출력한다.

    gripper 하나의 UNVERIFIED_RANGE가 나머지 관절에 어떤 영향도 주지 않는다는 것을 실제
    실행 중에 바로 확인할 수 있게 하기 위한 진단 출력이다 (토큰/원격 응답 원문은 다루지
    않는다 - 여기 있는 값은 이미 mapping을 거친 관절 이름 -> 각도(deg)/bool/문자열뿐).
    """
    summary = summarize_hold(samples)
    print("[Follower-safe 진단]")
    for joint in JOINT_NAMES:
        s = samples.get(joint)
        if s is None:
            continue
        fmt = lambda v: "None" if v is None else f"{v:.2f}"  # noqa: E731
        print(
            f"  {joint:14s} raw_leader={fmt(s.raw_leader_deg)} mapped_target={fmt(s.mapped_follower_deg)} "
            f"limited_command={fmt(s.limited_command_deg)} joint_hold={s.hold} joint_hold_reason={s.hold_reason}"
        )
    print(
        f"  global_hold={summary['global_hold']} global_hold_reason={summary['global_hold_reason']} "
        f"active_joint_count={summary['active_joint_count']} held_joint_count={summary['held_joint_count']} "
        f"held_joints={summary['held_joints']}",
        flush=True,
    )
