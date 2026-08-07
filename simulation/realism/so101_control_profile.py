"""``configs/generated/so101_control_profile_candidate_v1.json`` 로더 + 검증.

이 모듈은 **순수 로딩/검증**만 한다. 파일을 읽어 dataclass로 만들 뿐, 어떤 값도 실물
로봇이나 MuJoCo runtime 전체에 자동 적용하지 않는다 - 그건 이 candidate 파일 자체의
``apply_automatically=false`` 계약이다 (``hardware/diagnostics/control_profile_candidate.py``
참고). 이 profile을 실제로 "쓰는" 곳은 ``so101_realistic_control.py``의
``RealisticControlLayer`` 하나뿐이고, 그 레이어조차 realistic MuJoCo backend를 명시적으로
선택했을 때만 생성된다.

단위 계약 (파일 자체의 ``unit``/``gripper_unit_note`` 필드를 그대로 따른다):
    - arm 5개 관절(shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll): degree
    - gripper: percent_0_100 (도가 아니다)
이 모듈은 이 단위를 바꾸지 않고 그대로 dataclass에 보존한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 이 저장소 여러 모듈(so101_model.SO101_JOINT_NAMES, remote_state_client.JOINT_NAMES,
# readonly_so101_reader.JOINT_ORDER)이 각자 독립적으로 이 6개 관절 이름/순서를 정의해 두는
# 기존 관례를 그대로 따른다 - 이 모듈은 mujoco/requests 등 무거운 의존성을 끌어오지 않기
# 위해 새로 import하지 않고 동일한 튜플을 여기서도 정의한다.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

GRIPPER_JOINT = "gripper"
EXPECTED_GRIPPER_UNIT = "percent_0_100"
EXPECTED_ARM_UNIT = "degree"

STATUS_CANDIDATE_ONLY = "CANDIDATE_ONLY"


class ControlProfileError(RuntimeError):
    """profile JSON 로딩 또는 필수 계약 검증 실패."""


@dataclass(frozen=True)
class JointControlProfile:
    """단일 관절의 candidate 값 (profile에 기록된 단위 그대로 - degree 또는 percent_0_100)."""

    joint: str
    unit: str
    historical_range: tuple[float, float] | None  # (p01, p99) candidate_historical_inner_range
    frame_delta_soft_limit: float | None  # candidate_frame_delta_soft_limit (한 스텝당, profile 원 단위)
    velocity_soft_limit: float | None  # candidate_soft_velocity_limit (원 단위/s)


@dataclass(frozen=True)
class WristRollDeadbandProfile:
    """``wrist_roll_deadband_analysis`` 블록 - v1 simulation approximation 근거값.

    ``no_response_upper_tick``/``no_response_upper_deg``는 "이 폭 이내면 절대 반응하지
    않는다"는 하드웨어 사실이 아니라, 6-run에서 관측된 no-response 후보 구간의 상한값이다
    (원본 JSON의 ``no_response_region_ticks=[0,5]``, ``no_response_region_confidence``).
    """

    no_response_upper_tick: int
    no_response_upper_deg: float
    transition_start_tick: int | None
    tick_to_degree: dict[int, float]


@dataclass(frozen=True)
class TimingProfile:
    """``timing.observed_latency_ms`` 블록 - local command->actual 계측값.

    end-to-end VLA latency가 아니다 (``latency_scope_note`` 참고) - 이 profile을 쓰는
    ``RealisticControlLayer``도 이 사실을 바꾸지 않는다.
    """

    latency_median_ms: float | None
    latency_min_ms: float | None
    latency_max_ms: float | None
    valid_runs: str | None


@dataclass(frozen=True)
class ControlProfile:
    status: str
    source: str
    run_count: int
    apply_automatically: bool
    joints: dict[str, JointControlProfile]
    wrist_roll_deadband: WristRollDeadbandProfile
    timing: TimingProfile
    path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlProfileError(message)


def _parse_joint(joint: str, entry: dict[str, Any]) -> JointControlProfile:
    unit = entry.get("unit")
    expected_unit = EXPECTED_GRIPPER_UNIT if joint == GRIPPER_JOINT else EXPECTED_ARM_UNIT
    _require(
        unit == expected_unit,
        f"'{joint}' unit이 예상과 다릅니다 (기대={expected_unit!r}, 실제={unit!r}). "
        "gripper는 percent_0_100, 나머지 5개 관절은 degree여야 합니다.",
    )

    rng = (entry.get("historical_operating_range") or {}).get("candidate_historical_inner_range")
    historical_range: tuple[float, float] | None = None
    if rng is not None:
        _require(isinstance(rng, list) and len(rng) == 2, f"'{joint}'의 candidate_historical_inner_range 형식이 올바르지 않습니다: {rng!r}")
        historical_range = (float(rng[0]), float(rng[1]))

    frame_delta_soft_limit = (entry.get("frame_delta") or {}).get("candidate_frame_delta_soft_limit")
    velocity_soft_limit = (entry.get("velocity") or {}).get("candidate_soft_velocity_limit")

    return JointControlProfile(
        joint=joint,
        unit=unit,
        historical_range=historical_range,
        frame_delta_soft_limit=(float(frame_delta_soft_limit) if frame_delta_soft_limit is not None else None),
        velocity_soft_limit=(float(velocity_soft_limit) if velocity_soft_limit is not None else None),
    )


def _parse_wrist_roll_deadband(raw: dict[str, Any]) -> WristRollDeadbandProfile:
    ticks = raw.get("no_response_region_ticks")
    _require(
        isinstance(ticks, list) and len(ticks) == 2,
        f"wrist_roll_deadband_analysis.no_response_region_ticks가 없거나 형식이 올바르지 않습니다: {ticks!r}",
    )
    upper_tick = int(ticks[1])

    tick_table_raw = raw.get("tick_to_degree_equivalent") or {}
    _require(
        str(upper_tick) in tick_table_raw,
        f"tick_to_degree_equivalent에 no-response 상한 tick({upper_tick})의 환산값이 없습니다.",
    )
    tick_to_degree = {int(k): float(v) for k, v in tick_table_raw.items()}

    return WristRollDeadbandProfile(
        no_response_upper_tick=upper_tick,
        no_response_upper_deg=tick_to_degree[upper_tick],
        transition_start_tick=(
            int(raw["transition_region_start_ticks"]) if raw.get("transition_region_start_ticks") is not None else None
        ),
        tick_to_degree=tick_to_degree,
    )


def _parse_timing(raw: dict[str, Any]) -> TimingProfile:
    lat = raw.get("observed_latency_ms") or {}
    return TimingProfile(
        latency_median_ms=(float(lat["median"]) if lat.get("median") is not None else None),
        latency_min_ms=(float(lat["min"]) if lat.get("min") is not None else None),
        latency_max_ms=(float(lat["max"]) if lat.get("max") is not None else None),
        valid_runs=lat.get("valid_runs"),
    )


def load_control_profile(path: str | Path) -> ControlProfile:
    """``so101_control_profile_candidate_v1.json``을 읽어 검증된 :class:`ControlProfile`을 만든다.

    검증 항목 (섹션 4 요구사항):
        - ``status == "CANDIDATE_ONLY"``
        - ``apply_automatically == False``
        - ``source``/``run_count`` 존재
        - 6개 관절(``JOINT_NAMES``) 전부에 대한 joint 데이터 존재
        - gripper unit == ``percent_0_100``, 나머지 5개는 ``degree``
        - wrist_roll deadband/latency 블록 존재

    Raises:
        ControlProfileError: 파일이 없거나, JSON 파싱에 실패하거나, 위 계약을 어기는 경우.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ControlProfileError(f"control profile 파일을 찾을 수 없습니다: {resolved}")
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ControlProfileError(f"{resolved} JSON 파싱에 실패했습니다: {exc}") from exc
    if not isinstance(raw, dict):
        raise ControlProfileError(f"{resolved} 최상위가 JSON object가 아닙니다.")

    _require(raw.get("status") == STATUS_CANDIDATE_ONLY, f"status가 {STATUS_CANDIDATE_ONLY!r}가 아닙니다: {raw.get('status')!r}")
    _require(
        raw.get("apply_automatically") is False,
        f"apply_automatically가 False가 아닙니다: {raw.get('apply_automatically')!r}. "
        "이 profile은 명시적으로 opt-in한 realistic 모드에서만 읽혀야 합니다.",
    )
    _require(bool(raw.get("source")), "source 필드가 없습니다.")
    _require(raw.get("run_count") is not None, "run_count 필드가 없습니다.")

    joints_raw = raw.get("joints") or {}
    missing_joints = [j for j in JOINT_NAMES if j not in joints_raw]
    _require(not missing_joints, f"joints에 다음 관절 데이터가 없습니다: {missing_joints}")

    joints = {name: _parse_joint(name, joints_raw[name]) for name in JOINT_NAMES}

    deadband_raw = raw.get("wrist_roll_deadband_analysis")
    _require(isinstance(deadband_raw, dict), "wrist_roll_deadband_analysis 블록이 없습니다.")
    wrist_roll_deadband = _parse_wrist_roll_deadband(deadband_raw)

    timing_raw = raw.get("timing")
    _require(isinstance(timing_raw, dict), "timing 블록이 없습니다.")
    timing = _parse_timing(timing_raw)

    return ControlProfile(
        status=raw["status"],
        source=raw["source"],
        run_count=int(raw["run_count"]),
        apply_automatically=bool(raw["apply_automatically"]),
        joints=joints,
        wrist_roll_deadband=wrist_roll_deadband,
        timing=timing,
        path=resolved,
    )


DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "generated" / "so101_control_profile_candidate_v1.json"
)
