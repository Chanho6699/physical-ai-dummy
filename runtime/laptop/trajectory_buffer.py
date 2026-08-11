"""Phase C-1B: thread-safe ``TimestampedActionChunk`` 저장소.

이 모듈은 "여러 chunk 중 뭘 골라서 어떻게 섞을지"(temporal ensembling)는 전혀 하지 않는다
(이 세션 범위 밖) - 그냥 최근 chunk 몇 개를 시간 순서를 지키며 안전하게 보관하고, 매
호출자(향후 motor control loop)가 일관된 snapshot을 읽어갈 수 있게 하는 게 유일한 역할이다.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from runtime.laptop.trajectory_chunk import TimestampedActionChunk

# 향후 ensemble 가능성을 위한 history retention 기본값 - temporal ensemble의 "몇 개를 실제로
# 섞을지" 값이 아니다(섹션 2 명시) - 그냥 "최근 몇 개까지 버릴지"의 보수적인 기본값이다.
DEFAULT_MAX_CHUNKS = 4


@dataclass(frozen=True)
class PublishResult:
    """``TrajectoryBuffer.publish()``의 결과 - 성공/거부와 사유를 분리한다(예외를 던지지
    않음 - 네트워크에서 온 신뢰할 수 없는 chunk를 다루는 이 저장소의 다른 validate류
    함수들과 동일한 관례)."""

    accepted: bool
    reason: str | None  # 거부됐을 때만 채워짐
    discarded_as_stale_out_of_order: bool  # True: chunk 자체는 유효했지만 더 최신 것보다
    # 오래된 sequence/관측이라 거부됨(진짜 실패가 아님 - worker가 consecutive_failures에
    # 반영하면 안 되는 케이스, 섹션 8 참고)


class TrajectoryBuffer:
    """최근 chunk를 시간 순서를 지키며 보관하는 thread-safe 저장소.

    Lock 범위(섹션 9 "buffer lock과 worker lock deadlock 없음"): 이 클래스 안의
    ``threading.Lock`` 하나만 쓰고, 그 lock을 쥔 채로 다른 어떤 객체(worker, VLA client
    등)도 호출하지 않는다 - 순수하게 자기 내부 state(``deque``, ``_latest_sequence``,
    ``_latest_observation_time``)만 보호한다. 그래서 이 lock이 다른 lock을 기다리게 될
    경로가 아예 존재하지 않는다(cycle 불가능).
    """

    def __init__(self, *, max_chunks: int = DEFAULT_MAX_CHUNKS) -> None:
        if max_chunks < 1:
            raise ValueError(f"max_chunks는 1 이상이어야 합니다: {max_chunks}")
        self._lock = threading.Lock()
        self._max_chunks = max_chunks
        self._chunks: deque[TimestampedActionChunk] = deque(maxlen=max_chunks)
        self._latest_sequence: int | None = None
        self._latest_observation_time: float | None = None

    def publish(self, chunk: TimestampedActionChunk) -> PublishResult:
        """새 chunk를 등록한다. 검증 순서(섹션 2 요구사항 전부 커버):

        1. ``chunk.validate()`` - 자기 자신의 내적 일관성(chunk finite/spacing>0/
           chunk_size 일치) - 이건 이 클래스가 아니라 ``TimestampedActionChunk`` 자신이
           안다(재사용).
        2. observation timestamp 유효성 - finite인지는 1번에 포함, 추가로 여기서는
           "관측 시각이 존재하는가"만 별도로 다시 확인하지 않는다(1번이 이미 finite를
           검사함 - 중복 검사 안 함).
        3. sequence monotonic / out-of-order 정책 - **더 오래된 sequence/관측이 나중에
           도착하면 latest를 덮어쓰지 못한다**(섹션 2 "out-of-order 정책" 요구사항,
           Case C). ``sequence``를 1차 판단 기준으로 쓴다(worker가 요청을 보낸 순서를
           그대로 반영하는 값이므로) - ``sequence``가 같거나 더 작으면 무조건 거부한다.
        """
        ok, reason = chunk.validate()
        if not ok:
            return PublishResult(accepted=False, reason=reason, discarded_as_stale_out_of_order=False)

        with self._lock:
            if self._latest_sequence is not None and chunk.sequence <= self._latest_sequence:
                return PublishResult(
                    accepted=False,
                    reason=(
                        f"out-of-order: sequence={chunk.sequence}가 이미 발행된 latest_sequence="
                        f"{self._latest_sequence} 이하입니다 - latest를 덮어쓰지 않습니다."
                    ),
                    discarded_as_stale_out_of_order=True,
                )
            self._chunks.append(chunk)
            self._latest_sequence = chunk.sequence
            self._latest_observation_time = chunk.observation_time_monotonic

        return PublishResult(accepted=True, reason=None, discarded_as_stale_out_of_order=False)

    def latest(self) -> TimestampedActionChunk | None:
        """가장 최근에 성공적으로 publish된 chunk (없으면 ``None``)."""
        with self._lock:
            return self._chunks[-1] if self._chunks else None

    def snapshot(self) -> tuple[TimestampedActionChunk, ...]:
        """현재 보관 중인 모든 chunk를 시간(발행) 순서대로 담은 불변 tuple - 가장 오래된
        게 [0], 가장 최신이 [-1]. 향후 temporal ensembling이 여러 chunk를 훑어볼 때 쓸
        진입점이다(이 세션에서는 아직 아무도 이걸로 ensemble하지 않는다)."""
        with self._lock:
            return tuple(self._chunks)

    def valid_chunks(self, now_monotonic: float, *, max_chunk_age_ms: float | None = None) -> tuple[TimestampedActionChunk, ...]:
        """``snapshot()`` 중 아직 expire되지 않은 것만 걸러서 반환한다 (``is_expired()``
        기준, 섹션 7 기본 정책 - ``remaining_future_count<=0``이면 제외, 선택적으로
        ``max_chunk_age_ms``도 함께 적용)."""
        return tuple(c for c in self.snapshot() if not c.is_expired(now_monotonic, max_chunk_age_ms=max_chunk_age_ms))

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._latest_sequence = None
            self._latest_observation_time = None

    @property
    def max_chunks(self) -> int:
        return self._max_chunks
