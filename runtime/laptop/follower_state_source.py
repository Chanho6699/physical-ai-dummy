"""실물 SO-101 follower의 읽기 전용 state source.

``hardware/state_server/readonly_so101_reader.ReadOnlySO101Reader``를 그대로 감싼다 -
이 파일은 새로운 read 경로를 만들지 않고, 그 reader가 이미 감사(audit)된 안전성 근거를
그대로 물려받는다 (``docs/hardware_state_server.md`` 3절 참고: connect/disconnect가
어떤 write도 하지 않는다는 근거).

REAL_FOLLOWER_WRITE_COUNT는 이 모듈 안에서 항상 0이다 - 이 클래스에는애초에 write
메서드가 정의되어 있지 않다 (``ReadOnlySO101Reader`` 공개 API가 이미 write를 노출하지
않으므로, 이 wrapper가 아무리 잘못 호출돼도 실물에 쓸 방법이 구조적으로 없다).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from hardware.state_server.calibration_loader import CalibrationLoadError, load_calibration_file, to_motor_calibration_map
from hardware.state_server.readonly_so101_reader import JOINT_ORDER, ReadOnlyReaderError, ReadOnlySO101Reader

# 이 wrapper는 항상 0이다 - write 메서드 자체가 존재하지 않으므로 실행 중 값이 바뀔 방법이
# 없다 (섹션 17: "boolean 하나에만 의존하지 마세요" - 그래서 write 경로 자체를 구조적으로
# 제거하는 쪽을 택했다. 이 상수는 리포트에 그 사실을 명시적으로 남기기 위한 것뿐이다).
REAL_FOLLOWER_WRITE_COUNT: int = 0


class FollowerStateSourceError(RuntimeError):
    """연결/읽기 실패 (하드웨어 통신 오류)."""


@dataclass(frozen=True)
class FollowerStateSnapshot:
    positions_deg: dict[str, float]  # 몸통 5관절=degree, gripper=percent_0_100 (readonly_so101_reader 관례)
    read_at_monotonic: float
    read_at_wall: float


class ReadOnlyRealFollowerStateSource:
    """실물 팔로워 state를 읽기만 하는 source. write 메서드가 없다."""

    def __init__(self, reader: ReadOnlySO101Reader) -> None:
        self._reader = reader

    @classmethod
    def from_port(
        cls,
        *,
        port: str,
        follower_id: str,
        calibration_path: str | None = None,
    ) -> "ReadOnlyRealFollowerStateSource":
        resolved_path = calibration_path or f"~/.cache/huggingface/lerobot/calibration/robots/so_follower/{follower_id}.json"
        try:
            entries = load_calibration_file(resolved_path)
        except CalibrationLoadError as exc:
            raise FollowerStateSourceError(f"팔로워 캘리브레이션 로딩 실패: {exc}") from exc
        calibration = to_motor_calibration_map(entries)
        reader = ReadOnlySO101Reader(name=follower_id, port=port, calibration=calibration)
        return cls(reader)

    def connect(self) -> None:
        try:
            self._reader.connect()
        except Exception as exc:  # noqa: BLE001 - 하드웨어 SDK 예외를 일관된 타입으로 감싼다
            raise FollowerStateSourceError(f"팔로워 연결 실패: {exc}") from exc

    @property
    def is_connected(self) -> bool:
        return self._reader.is_connected

    def read(self) -> FollowerStateSnapshot:
        t0 = time.monotonic()
        try:
            positions = self._reader.read_positions()
        except Exception as exc:  # noqa: BLE001
            raise FollowerStateSourceError(f"팔로워 state 읽기 실패: {exc}") from exc
        missing = [j for j in JOINT_ORDER if j not in positions]
        if missing:
            raise FollowerStateSourceError(f"팔로워 state에 관절이 누락되었습니다: {missing}")
        return FollowerStateSnapshot(
            positions_deg={j: float(positions[j]) for j in JOINT_ORDER},
            read_at_monotonic=t0,
            read_at_wall=time.time(),
        )

    def disconnect(self) -> None:
        self._reader.disconnect()
