"""SO-101 캘리브레이션 JSON 파일(LeRobot 형식)을 읽기 전용으로 로드한다.

LeRobot이 실제로 저장하는 형식(예:
``~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<id>.json``)은
관절 이름을 key로 하고, 각 관절마다 ``id``/``drive_mode``/``homing_offset``/
``range_min``/``range_max``를 담은 JSON object다. 이 모듈은 그 파일을 파싱만 하며,
쓰기 함수는 없다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class CalibrationLoadError(RuntimeError):
    """캘리브레이션 파일을 읽거나 해석할 수 없을 때 발생한다."""


@dataclass(frozen=True)
class MotorCalibrationEntry:
    """캘리브레이션 파일 한 관절 항목의 읽기 전용 사본."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def load_calibration_file(path: str | Path) -> dict[str, MotorCalibrationEntry]:
    """캘리브레이션 JSON 파일을 읽어 관절별 :class:`MotorCalibrationEntry`를 반환한다.

    Raises:
        CalibrationLoadError: 파일이 없거나, JSON 파싱에 실패하거나, 6개 관절 중
            일부가 없거나, 필드 타입이 올바르지 않거나, range_min == range_max인 경우.
    """

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise CalibrationLoadError(f"캘리브레이션 파일을 찾을 수 없습니다: {resolved}")

    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationLoadError(f"캘리브레이션 파일을 읽을 수 없습니다: {resolved} ({exc})") from exc

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CalibrationLoadError(f"캘리브레이션 JSON 파싱 실패: {resolved} ({exc})") from exc

    if not isinstance(raw, dict):
        raise CalibrationLoadError(f"캘리브레이션 파일 최상위는 JSON object여야 합니다: {resolved}")

    missing = [joint for joint in JOINT_NAMES if joint not in raw]
    if missing:
        raise CalibrationLoadError(
            f"캘리브레이션 파일에 다음 관절이 없습니다: {missing} ({resolved})"
        )

    entries: dict[str, MotorCalibrationEntry] = {}
    for joint in JOINT_NAMES:
        entry_raw = raw[joint]
        if not isinstance(entry_raw, dict):
            raise CalibrationLoadError(f"'{joint}' 캘리브레이션 항목이 JSON object가 아닙니다: {resolved}")

        try:
            entry = MotorCalibrationEntry(
                id=int(entry_raw["id"]),
                drive_mode=int(entry_raw.get("drive_mode", 0)),
                homing_offset=int(entry_raw["homing_offset"]),
                range_min=int(entry_raw["range_min"]),
                range_max=int(entry_raw["range_max"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationLoadError(
                f"'{joint}' 캘리브레이션 필드가 올바르지 않습니다: {resolved} ({exc})"
            ) from exc

        if entry.range_min == entry.range_max:
            raise CalibrationLoadError(
                f"'{joint}' range_min과 range_max가 같습니다 (잘못된 캘리브레이션): {resolved}"
            )

        entries[joint] = entry

    return entries


def to_motor_calibration_map(entries: dict[str, MotorCalibrationEntry]):
    """``lerobot.motors.MotorCalibration`` dict로 변환한다 (FeetechMotorsBus 생성자용).

    lerobot import는 함수 내부에서 지연 수행한다. calibration_loader 자체는
    lerobot(및 하드웨어 SDK) 없이도 단위 테스트가 가능해야 하기 때문이다.
    """

    from lerobot.motors import MotorCalibration

    return {
        joint: MotorCalibration(
            id=entry.id,
            drive_mode=entry.drive_mode,
            homing_offset=entry.homing_offset,
            range_min=entry.range_min,
            range_max=entry.range_max,
        )
        for joint, entry in entries.items()
    }


def to_public_dict(entries: dict[str, MotorCalibrationEntry]) -> dict[str, dict[str, int]]:
    """``GET /calibration`` 응답용으로 변환한다.

    캘리브레이션 파일의 절대 경로 등 민감한 시스템 경로는 여기 포함하지 않는다.
    ``id``, ``drive_mode``는 관절 순서를 바꾸지 않는 한 클라이언트에 불필요하므로 제외하고
    실제 진단에 필요한 ``homing_offset``/``range_min``/``range_max``만 노출한다.
    """

    return {
        joint: {
            "homing_offset": entry.homing_offset,
            "range_min": entry.range_min,
            "range_max": entry.range_max,
        }
        for joint, entry in entries.items()
    }
