"""상태 서버 내부 도메인 모델과 유효성 검사.

FastAPI 응답 스키마(pydantic)는 여기 두지 않는다 (``app.py`` 참고). 이 모듈은
"읽은 값이 신뢰할 수 있는 정상값인가"를 판정하는 순수 로직만 담당하며, HTTP나
하드웨어 SDK에 의존하지 않는다 (하드웨어 없이 단위 테스트하기 위함).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

MODE_READ_ONLY = "read_only"


class JointReadError(ValueError):
    """읽은 관절 상태가 유효하지 않을 때 발생한다 (누락/잘못된 shape/NaN/Inf 등)."""


def validate_positions_deg(positions: dict[str, float]) -> None:
    """관절 각도 dict가 6개 관절을 정확히 포함하고 유한한 실수인지 검사한다.

    이 검사를 통과하지 못하면 폴러는 해당 샘플을 "정상 상태"로 캐시하지 않는다 -
    서버가 거짓 정상값을 만들지 않기 위한 핵심 안전장치다.
    """

    if not isinstance(positions, dict):
        raise JointReadError(f"positions는 dict여야 합니다. got={type(positions)!r}")

    missing = [joint for joint in JOINT_NAMES if joint not in positions]
    if missing:
        raise JointReadError(f"관절 값이 누락되었습니다: {missing}")

    extra = sorted(set(positions) - set(JOINT_NAMES))
    if extra:
        raise JointReadError(f"알 수 없는 관절 키가 포함되어 shape이 올바르지 않습니다: {extra}")

    for joint in JOINT_NAMES:
        value = positions[joint]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JointReadError(f"'{joint}' 값이 숫자가 아닙니다: {value!r}")
        if math.isnan(value) or math.isinf(value):
            raise JointReadError(f"'{joint}' 값이 NaN/Inf입니다: {value!r}")


@dataclass(frozen=True)
class ArmSample:
    """한 번의 폴링에서 얻은 유효한(검증 통과) 관절 상태."""

    positions_deg: dict[str, float]
    raw_ticks: dict[str, int] | None
    timestamp: float


@dataclass
class ArmPollState:
    """한 팔(리더 또는 팔로워)의 최신 폴링 결과 누적 상태 (가변, lock으로 보호됨)."""

    last_good: ArmSample | None = None
    consecutive_errors: int = 0
    last_error: str | None = None
    connected: bool = False


@dataclass(frozen=True)
class ArmView:
    """API 응답으로 내보낼, 특정 시점의 한 팔 상태 스냅샷."""

    connected: bool
    positions_deg: dict[str, float] | None
    raw_ticks: dict[str, int] | None
    stale: bool
    age_ms: float | None
    read_error_count: int
    last_error: str | None
