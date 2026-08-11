"""runtime/laptop/trajectory_chunk.py의 TimestampedActionChunk 검증 - 특히 시간 계산
헬퍼(current_index/first_future_index/remaining_future_count/age_ms)의 boundary 동작
(섹션 6 "exact boundary 처리 명확히 테스트" 요구사항)."""

from __future__ import annotations

import math

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

SPACING = 1.0 / 30.0  # Candidate B dataset fps=30
CHUNK_SIZE = 50
OBS_TIME = 1000.0  # 임의의 monotonic 기준점


def _actions(n: int = CHUNK_SIZE) -> tuple[dict[str, float], ...]:
    return tuple({j: float(i) for j in JOINT_ORDER} for i in range(n))


def _chunk(**overrides) -> TimestampedActionChunk:
    defaults = dict(
        sequence=0,
        session_id="s1",
        observation_time_monotonic=OBS_TIME,
        request_started_time_monotonic=OBS_TIME + 0.001,
        response_received_time_monotonic=OBS_TIME + 0.05,
        server_received_at=None,
        server_responded_at=None,
        inference_latency_ms=49.0,
        chunk_index_spacing_s=SPACING,
        chunk_size=CHUNK_SIZE,
        actions=_actions(),
        model_id="fake-model",
        backend="fake",
    )
    defaults.update(overrides)
    return TimestampedActionChunk(**defaults)


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_valid_chunk_passes_validate() -> None:
    ok, reason = _chunk().validate()
    assert ok is True
    assert reason is None


def test_chunk_size_mismatch_rejected() -> None:
    ok, reason = _chunk(actions=_actions(49)).validate()
    assert ok is False
    assert "chunk_size" in reason or "actions" in reason


def test_zero_chunk_size_rejected() -> None:
    ok, reason = _chunk(chunk_size=0, actions=()).validate()
    assert ok is False


def test_non_positive_spacing_rejected() -> None:
    ok, reason = _chunk(chunk_index_spacing_s=0.0).validate()
    assert ok is False
    assert "spacing" in reason.lower() or "chunk_index_spacing_s" in reason


def test_negative_spacing_rejected() -> None:
    ok, reason = _chunk(chunk_index_spacing_s=-0.01).validate()
    assert ok is False


def test_nan_action_value_rejected() -> None:
    actions = list(_actions())
    actions[3] = {**actions[3], "wrist_roll": math.nan}
    ok, reason = _chunk(actions=tuple(actions)).validate()
    assert ok is False
    assert "actions[3]" in reason


def test_inf_action_value_rejected() -> None:
    actions = list(_actions())
    actions[0] = {**actions[0], "gripper": math.inf}
    ok, reason = _chunk(actions=tuple(actions)).validate()
    assert ok is False


def test_non_finite_observation_time_rejected() -> None:
    ok, reason = _chunk(observation_time_monotonic=math.nan).validate()
    assert ok is False


# ---------------------------------------------------------------------------
# nominal_target_time / horizon
# ---------------------------------------------------------------------------


def test_nominal_target_time_k0_equals_observation_time() -> None:
    chunk = _chunk()
    assert chunk.nominal_target_time(0) == pytest.approx(OBS_TIME)


def test_nominal_target_time_increases_by_spacing() -> None:
    chunk = _chunk()
    assert chunk.nominal_target_time(1) == pytest.approx(OBS_TIME + SPACING)
    assert chunk.nominal_target_time(10) == pytest.approx(OBS_TIME + 10 * SPACING)


def test_horizon_duration_matches_chunk_size_times_spacing() -> None:
    chunk = _chunk()
    assert chunk.horizon_duration_s == pytest.approx(CHUNK_SIZE * SPACING)
    assert chunk.horizon_duration_s == pytest.approx(50.0 / 30.0)  # ≈1.667s, 섹션 11


def test_horizon_end_time_monotonic() -> None:
    chunk = _chunk()
    assert chunk.horizon_end_time_monotonic == pytest.approx(OBS_TIME + 50 * SPACING)


# ---------------------------------------------------------------------------
# current_index boundary (섹션 6)
# ---------------------------------------------------------------------------


def test_current_index_at_observation_time_is_zero() -> None:
    chunk = _chunk()
    assert chunk.current_index(OBS_TIME) == 0


def test_current_index_before_observation_time_clamped_to_zero() -> None:
    chunk = _chunk()
    assert chunk.current_index(OBS_TIME - 10.0) == 0


def test_current_index_exact_slot_boundary() -> None:
    chunk = _chunk()
    # now == obs_time + k*spacing (정확히) -> current_index == k
    for k in (1, 5, 9, 25, 49):
        now = OBS_TIME + k * SPACING
        assert chunk.current_index(now) == k, f"k={k}"


def test_current_index_just_before_slot_boundary() -> None:
    chunk = _chunk()
    now = OBS_TIME + 10 * SPACING - 1e-9
    assert chunk.current_index(now) == 9


def test_current_index_beyond_chunk_end_not_clamped() -> None:
    """current_index 자체는 chunk_size 이상을 반환할 수 있다 (expiry 판단은 별도 메서드)."""
    chunk = _chunk()
    now = OBS_TIME + 1000 * SPACING
    assert chunk.current_index(now) == 1000


# ---------------------------------------------------------------------------
# first_future_index / remaining_future_count boundary (섹션 6/7)
# ---------------------------------------------------------------------------


def test_first_future_index_before_observation_time_is_zero() -> None:
    chunk = _chunk()
    assert chunk.first_future_index(OBS_TIME - 1.0) == 0


def test_first_future_index_at_observation_time_is_one() -> None:
    """now == observation_time 정확히면 index 0은 "지금"이지 미래가 아니다 -> 1."""
    chunk = _chunk()
    assert chunk.first_future_index(OBS_TIME) == 1


def test_first_future_index_case_a_330ms_latency() -> None:
    """섹션 11/Case A: latency=330ms, spacing=33.33ms -> response 시점에는 index 9까지
    이미 지났고 index 10부터 미래 (0.330/0.033333=9.9 -> floor 9 -> +1=10)."""
    chunk = _chunk()
    now = OBS_TIME + 0.330
    assert chunk.current_index(now) == 9
    assert chunk.first_future_index(now) == 10
    assert chunk.remaining_future_count(now) == 40  # 50-10


def test_first_future_index_exact_slot_boundary() -> None:
    chunk = _chunk()
    now = OBS_TIME + 20 * SPACING  # 정확히 슬롯 20 시각
    assert chunk.first_future_index(now) == 21  # 슬롯 20은 "지금"이지 미래가 아님


def test_remaining_future_count_full_at_observation_time() -> None:
    chunk = _chunk()
    assert chunk.remaining_future_count(OBS_TIME - 1.0) == CHUNK_SIZE  # 전부 미래


def test_first_future_index_none_when_fully_expired() -> None:
    chunk = _chunk()
    now = chunk.horizon_end_time_monotonic + 1.0
    assert chunk.first_future_index(now) is None
    assert chunk.remaining_future_count(now) == 0


def test_first_future_index_last_valid_index_boundary() -> None:
    """마지막 index(49)의 target time 바로 이전까지는 first_future_index==49가 나와야
    하고, 그 시각을 넘는 순간 None(expired)이 돼야 한다."""
    chunk = _chunk()
    just_before_last_slot = OBS_TIME + 48 * SPACING + 1e-9
    assert chunk.first_future_index(just_before_last_slot) == 49
    at_last_slot_exact = OBS_TIME + 49 * SPACING
    assert chunk.first_future_index(at_last_slot_exact) is None  # 49는 "지금"이라 미래 아님
    assert chunk.remaining_future_count(at_last_slot_exact) == 0


# ---------------------------------------------------------------------------
# is_expired / age_ms (섹션 7)
# ---------------------------------------------------------------------------


def test_not_expired_right_after_observation() -> None:
    chunk = _chunk()
    assert chunk.is_expired(OBS_TIME) is False


def test_expired_when_remaining_future_count_zero() -> None:
    chunk = _chunk()
    now = chunk.horizon_end_time_monotonic
    assert chunk.is_expired(now) is True


def test_expired_with_max_chunk_age_even_if_future_remains() -> None:
    """remaining_future_count>0이어도 max_chunk_age_ms를 넘으면 expired여야 한다
    (섹션 7 "추가 freshness guard" - observation age 기준)."""
    chunk = _chunk()
    now = OBS_TIME + 0.5  # 아직 index 15 근방부터 미래 -> remaining>0
    assert chunk.remaining_future_count(now) > 0
    assert chunk.is_expired(now, max_chunk_age_ms=None) is False
    assert chunk.is_expired(now, max_chunk_age_ms=100.0) is True  # 500ms 지났는데 한도 100ms


def test_age_ms_based_on_observation_time_not_response_time() -> None:
    chunk = _chunk(response_received_time_monotonic=OBS_TIME + 0.05)
    now = OBS_TIME + 0.330
    assert chunk.age_ms(now) == pytest.approx(330.0)  # response_received(50ms)가 아니라 obs 기준


def test_response_latency_ms_helper() -> None:
    chunk = _chunk(observation_time_monotonic=OBS_TIME, response_received_time_monotonic=OBS_TIME + 0.353)
    assert chunk.response_latency_ms() == pytest.approx(353.0)


# ---------------------------------------------------------------------------
# actions immutability (tuple, not list)
# ---------------------------------------------------------------------------


def test_actions_is_tuple_not_list() -> None:
    chunk = _chunk()
    assert isinstance(chunk.actions, tuple)
