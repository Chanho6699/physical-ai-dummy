"""runtime/laptop/trajectory_buffer.py의 TrajectoryBuffer 검증 - publish validation,
out-of-order 정책(섹션 2 Case C), history retention, expiry 필터링, thread-safety."""

from __future__ import annotations

import threading

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.trajectory_buffer import DEFAULT_MAX_CHUNKS, TrajectoryBuffer
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

SPACING = 1.0 / 30.0
CHUNK_SIZE = 50


def _actions(n: int = CHUNK_SIZE) -> tuple[dict[str, float], ...]:
    return tuple({j: 0.0 for j in JOINT_ORDER} for _ in range(n))


def _chunk(*, sequence: int, observation_time_monotonic: float, **overrides) -> TimestampedActionChunk:
    defaults = dict(
        sequence=sequence,
        session_id="s1",
        observation_time_monotonic=observation_time_monotonic,
        request_started_time_monotonic=observation_time_monotonic + 0.001,
        response_received_time_monotonic=observation_time_monotonic + 0.05,
        server_received_at=None,
        server_responded_at=None,
        inference_latency_ms=49.0,
        chunk_index_spacing_s=SPACING,
        chunk_size=CHUNK_SIZE,
        actions=_actions(),
        model_id="fake",
        backend="fake",
    )
    defaults.update(overrides)
    return TimestampedActionChunk(**defaults)


# ---------------------------------------------------------------------------
# publish() 기본 동작
# ---------------------------------------------------------------------------


def test_publish_accepts_first_chunk() -> None:
    buf = TrajectoryBuffer()
    result = buf.publish(_chunk(sequence=0, observation_time_monotonic=100.0))
    assert result.accepted is True
    assert result.reason is None
    assert buf.latest().sequence == 0


def test_publish_rejects_invalid_chunk_without_touching_state() -> None:
    buf = TrajectoryBuffer()
    bad = _chunk(sequence=0, observation_time_monotonic=100.0, chunk_size=49)  # actions 길이 불일치
    result = buf.publish(bad)
    assert result.accepted is False
    assert result.discarded_as_stale_out_of_order is False
    assert buf.latest() is None


def test_default_max_chunks_constant() -> None:
    assert 3 <= DEFAULT_MAX_CHUNKS <= 4  # 섹션 2: "초기 권장 max_chunks = 3~4"


def test_max_chunks_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        TrajectoryBuffer(max_chunks=0)


# ---------------------------------------------------------------------------
# sequence 증가에 따른 정상 publish + history retention
# ---------------------------------------------------------------------------


def test_publish_sequence_monotonic_updates_latest() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    for seq in range(5):
        result = buf.publish(_chunk(sequence=seq, observation_time_monotonic=100.0 + seq * 0.34))
        assert result.accepted is True
    assert buf.latest().sequence == 4


def test_history_retention_limited_to_max_chunks() -> None:
    """chunk 전체(50-step)는 보존하되, 몇 개의 chunk까지만 history에 남긴다(섹션 2)."""
    buf = TrajectoryBuffer(max_chunks=3)
    for seq in range(6):
        buf.publish(_chunk(sequence=seq, observation_time_monotonic=100.0 + seq * 0.34))
    snap = buf.snapshot()
    assert len(snap) == 3
    assert [c.sequence for c in snap] == [3, 4, 5]  # 가장 오래된 게 [0], 최신이 [-1]
    # 각 chunk 자체의 50-step은 그대로 유지된다.
    assert all(len(c.actions) == CHUNK_SIZE for c in snap)


def test_snapshot_is_chronological_oldest_first() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    for seq in range(3):
        buf.publish(_chunk(sequence=seq, observation_time_monotonic=100.0 + seq * 0.34))
    snap = buf.snapshot()
    assert [c.sequence for c in snap] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Case B (섹션 10): 다음 chunk가 340ms 뒤 도착 -> latest 교체, old chunk는 history에 남음
# ---------------------------------------------------------------------------


def test_case_b_next_chunk_arrival_replaces_latest_keeps_history() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    first = _chunk(sequence=0, observation_time_monotonic=100.0)
    buf.publish(first)
    assert buf.latest().sequence == 0

    second = _chunk(sequence=1, observation_time_monotonic=100.0 + 0.340)
    result = buf.publish(second)
    assert result.accepted is True
    assert buf.latest().sequence == 1  # latest는 새 chunk
    seqs_in_history = [c.sequence for c in buf.snapshot()]
    assert 0 in seqs_in_history  # old chunk도 history에 남아 있음
    assert 1 in seqs_in_history


# ---------------------------------------------------------------------------
# Case C (섹션 2/10): out-of-order 응답 - seq 11 먼저, seq 10 늦게 도착 -> latest 안 덮임
# ---------------------------------------------------------------------------


def test_case_c_out_of_order_response_does_not_overwrite_latest() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    result_11 = buf.publish(_chunk(sequence=11, observation_time_monotonic=200.0))
    assert result_11.accepted is True
    assert buf.latest().sequence == 11

    # seq 10(더 오래된 관측/요청)이 늦게 도착
    result_10 = buf.publish(_chunk(sequence=10, observation_time_monotonic=199.5))
    assert result_10.accepted is False
    assert result_10.discarded_as_stale_out_of_order is True
    assert buf.latest().sequence == 11  # 여전히 11 - 덮이지 않음


def test_out_of_order_rejected_chunk_not_added_to_history() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    buf.publish(_chunk(sequence=11, observation_time_monotonic=200.0))
    buf.publish(_chunk(sequence=10, observation_time_monotonic=199.5))
    seqs = [c.sequence for c in buf.snapshot()]
    assert 10 not in seqs
    assert seqs == [11]


def test_duplicate_sequence_rejected() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    buf.publish(_chunk(sequence=5, observation_time_monotonic=100.0))
    result = buf.publish(_chunk(sequence=5, observation_time_monotonic=100.34))
    assert result.accepted is False
    assert result.discarded_as_stale_out_of_order is True


def test_out_of_order_by_sequence_even_if_observation_time_looks_newer() -> None:
    """sequence가 판단 기준이다 - observation_time만 더 최신이어도 sequence가 낮으면
    거부한다(요구사항 원문: "더 오래된 observation/**sequence** 결과")."""
    buf = TrajectoryBuffer(max_chunks=4)
    buf.publish(_chunk(sequence=10, observation_time_monotonic=100.0))
    # sequence는 더 작지만 observation_time은 더 큰(이상한) 케이스 - 그래도 거부돼야 함.
    result = buf.publish(_chunk(sequence=9, observation_time_monotonic=999.0))
    assert result.accepted is False
    assert buf.latest().sequence == 10


# ---------------------------------------------------------------------------
# valid_chunks() / expiry 필터링 (섹션 7)
# ---------------------------------------------------------------------------


def test_valid_chunks_excludes_expired() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    buf.publish(_chunk(sequence=0, observation_time_monotonic=100.0))
    now_not_expired = 100.0 + 0.5
    now_expired = 100.0 + 100.0  # chunk horizon(≈1.667s)을 한참 넘김

    assert len(buf.valid_chunks(now_not_expired)) == 1
    assert len(buf.valid_chunks(now_expired)) == 0


def test_valid_chunks_with_max_age_guard() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    buf.publish(_chunk(sequence=0, observation_time_monotonic=100.0))
    now = 100.0 + 0.5  # remaining_future_count>0이지만 500ms 지남
    assert len(buf.valid_chunks(now)) == 1  # max_chunk_age_ms 없으면 아직 유효
    assert len(buf.valid_chunks(now, max_chunk_age_ms=100.0)) == 0  # 100ms 한도는 넘음


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


def test_clear_resets_buffer_and_allows_lower_sequence_after() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    buf.publish(_chunk(sequence=10, observation_time_monotonic=100.0))
    buf.clear()
    assert buf.latest() is None
    assert buf.snapshot() == ()
    # clear 이후에는 낮은 sequence도 다시 정상 publish 가능 (out-of-order 상태가 리셋됨).
    result = buf.publish(_chunk(sequence=0, observation_time_monotonic=200.0))
    assert result.accepted is True


# ---------------------------------------------------------------------------
# Thread-safety - 여러 스레드가 동시에 publish해도 latest_sequence가 어긋나지 않아야 함.
# ---------------------------------------------------------------------------


def test_concurrent_publish_is_thread_safe() -> None:
    buf = TrajectoryBuffer(max_chunks=4)
    n = 200

    def _publish(seq: int) -> None:
        buf.publish(_chunk(sequence=seq, observation_time_monotonic=100.0 + seq * 0.001))

    threads = [threading.Thread(target=_publish, args=(seq,)) for seq in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()

    # 순서 보장은 없지만(동시 실행이므로), 최종 latest는 실제로 publish에 성공한 것 중
    # 하나여야 하고, 어떤 예외도 없이 전부 끝나야 한다(race condition으로 인한 crash 없음).
    assert buf.latest() is not None
    assert buf.latest().sequence < n
