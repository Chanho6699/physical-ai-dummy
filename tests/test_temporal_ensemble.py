"""runtime/laptop/temporal_ensemble.py의 TemporalEnsembler 검증 (Phase C-2, 섹션 10 Case A-J).

전부 synthetic TimestampedActionChunk + 손으로 계산 가능한 값으로 구성해, 절대시간
alignment/interpolation/weighting 수식을 직접 검증한다. 실제 하드웨어 접근 없음.
"""

from __future__ import annotations

import math

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.temporal_ensemble import EnsembledTarget, TemporalEnsembler
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

SPACING = 1.0 / 30.0
CHUNK_SIZE = 50


def _chunk(*, sequence: int, obs_time: float, value_at_k=None, chunk_size: int = CHUNK_SIZE, spacing: float = SPACING) -> TimestampedActionChunk:
    """``value_at_k(k) -> float``이 주어지면 모든 joint에 그 값을 채운다(기본은 ``k``
    그대로) - 테스트에서 손으로 검증하기 쉬운 간단한 패턴."""
    fn = value_at_k or (lambda k: float(k))
    actions = tuple({j: fn(k) for j in JOINT_ORDER} for k in range(chunk_size))
    return TimestampedActionChunk(
        sequence=sequence, session_id="s1", observation_time_monotonic=obs_time,
        request_started_time_monotonic=obs_time, response_received_time_monotonic=obs_time + 0.05,
        server_received_at=None, server_responded_at=None, inference_latency_ms=50.0,
        chunk_index_spacing_s=spacing, chunk_size=chunk_size, actions=actions,
        model_id="fake", backend="fake",
    )


# ---------------------------------------------------------------------------
# 생성자 검증
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        TemporalEnsembler(max_contributors=0)
    with pytest.raises(ValueError):
        TemporalEnsembler(half_life_s=0.0)
    with pytest.raises(ValueError):
        TemporalEnsembler(half_life_s=-1.0)
    with pytest.raises(ValueError):
        TemporalEnsembler(lookahead_s=-0.1)


def test_default_half_life_within_suggested_range() -> None:
    ensembler = TemporalEnsembler()
    assert 0.3 <= ensembler.half_life_s <= 0.5  # 섹션 4 권장 범위


def test_default_max_contributors_is_three() -> None:
    assert TemporalEnsembler().max_contributors == 3


# ---------------------------------------------------------------------------
# Case A (섹션 10): 두 chunk가 동일 절대시각 T를 서로 다른 index로 커버 - index/time
# alignment가 chunk마다 독립적으로 올바르게 계산되는지 (index를 그대로 평균하지 않는지)
# ---------------------------------------------------------------------------


def test_case_a_same_absolute_time_different_index_per_chunk() -> None:
    chunk_a = _chunk(sequence=0, obs_time=0.0, value_at_k=lambda k: 100.0 + k)  # T=0.50 -> k=15 -> 115.0
    chunk_b = _chunk(sequence=1, obs_time=0.30, value_at_k=lambda k: 200.0 + k)  # T=0.50 -> k=6 -> 206.0
    ensembler = TemporalEnsembler(half_life_s=0.338)

    target = ensembler.compute_target([chunk_a, chunk_b], target_time_monotonic=0.50)

    assert target is not None
    assert target.num_contributors == 2
    # sequence 1(chunk_b, obs=0.30)이 더 최신이므로 contributing_sequences 첫 자리
    assert target.contributing_sequences == (1, 0)
    assert target.contributing_chunk_indices == ((6, 6), (0 + 15, 0 + 15))  # chunk_b: k=6, chunk_a: k=15
    # 두 값(115.0, 206.0)의 가중 평균 - index를 그대로 섞은 엉뚱한 값(예: 두 chunk 모두
    # index 15 취급)이 아니라, 각 chunk 고유의 정확한 index에서 나온 값끼리 섞였는지 확인.
    assert 115.0 < target.action["shoulder_pan"] < 206.0


# ---------------------------------------------------------------------------
# Case B (섹션 10): T가 두 action index 사이 - linear interpolation 정확성
# ---------------------------------------------------------------------------


def test_case_b_linear_interpolation_between_two_indices() -> None:
    chunk = _chunk(sequence=0, obs_time=0.0, value_at_k=lambda k: 10.0 * k)  # k=5 -> 50, k=6 -> 60
    ensembler = TemporalEnsembler()
    target_time = 5.5 * SPACING  # 정확히 index 5와 6 사이 중간

    target = ensembler.compute_target([chunk], target_time_monotonic=target_time)

    assert target is not None
    assert target.contributing_chunk_indices == ((5, 6),)
    for joint in JOINT_ORDER:
        assert target.action[joint] == pytest.approx(55.0)  # 50 + 0.5*(60-50)


def test_interpolation_fraction_quarter_point() -> None:
    chunk = _chunk(sequence=0, obs_time=0.0, value_at_k=lambda k: 10.0 * k)
    ensembler = TemporalEnsembler()
    target_time = 5.25 * SPACING  # index 5~6 사이 25% 지점

    target = ensembler.compute_target([chunk], target_time_monotonic=target_time)
    assert target.action["gripper"] == pytest.approx(52.5)  # 50 + 0.25*10


def test_exact_index_boundary_no_interpolation_needed() -> None:
    chunk = _chunk(sequence=0, obs_time=0.0, value_at_k=lambda k: 10.0 * k)
    ensembler = TemporalEnsembler()
    target_time = 7 * SPACING  # 정확히 슬롯 7

    target = ensembler.compute_target([chunk], target_time_monotonic=target_time)
    assert target.contributing_chunk_indices == ((7, 7),)
    assert target.action["shoulder_lift"] == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# Case C (섹션 10): 3 contributors - newest > middle > oldest weight
# ---------------------------------------------------------------------------


def test_case_c_newest_gt_middle_gt_oldest_weight() -> None:
    oldest = _chunk(sequence=0, obs_time=10.00)
    middle = _chunk(sequence=1, obs_time=10.34)
    newest = _chunk(sequence=2, obs_time=10.68)
    ensembler = TemporalEnsembler(half_life_s=0.338, max_contributors=3)

    target = ensembler.compute_target([oldest, middle, newest], target_time_monotonic=10.80)

    assert target is not None
    assert target.num_contributors == 3
    assert target.contributing_sequences == (2, 1, 0)  # 최신순
    w_newest, w_middle, w_oldest = target.weights
    assert w_newest > w_middle > w_oldest
    assert sum(target.weights) == pytest.approx(1.0)

    # 손으로 계산한 기대값과 비교 (age는 newest=10.68 기준)
    lam = math.log(2.0) / 0.338
    raw = [math.exp(-lam * age) for age in (0.0, 10.68 - 10.34, 10.68 - 10.00)]
    expected = [w / sum(raw) for w in raw]
    assert w_newest == pytest.approx(expected[0])
    assert w_middle == pytest.approx(expected[1])
    assert w_oldest == pytest.approx(expected[2])


def test_case_c_age_fields_relative_to_target_time() -> None:
    oldest = _chunk(sequence=0, obs_time=10.00)
    newest = _chunk(sequence=1, obs_time=10.68)
    ensembler = TemporalEnsembler(half_life_s=0.338)

    target = ensembler.compute_target([oldest, newest], target_time_monotonic=10.80)

    assert target.newest_observation_age_ms == pytest.approx((10.80 - 10.68) * 1000.0)
    assert target.oldest_observation_age_ms == pytest.approx((10.80 - 10.00) * 1000.0)


def test_half_life_parameter_changes_weight_distribution() -> None:
    """half_life_s가 짧을수록(빨리 죽을수록) 오래된 contributor의 weight가 더 작아져야 한다."""
    oldest = _chunk(sequence=0, obs_time=10.00)
    newest = _chunk(sequence=1, obs_time=10.34)

    short_half_life = TemporalEnsembler(half_life_s=0.1)
    long_half_life = TemporalEnsembler(half_life_s=2.0)

    t_short = short_half_life.compute_target([oldest, newest], target_time_monotonic=10.50)
    t_long = long_half_life.compute_target([oldest, newest], target_time_monotonic=10.50)

    w_oldest_short = t_short.weights[t_short.contributing_sequences.index(0)]
    w_oldest_long = t_long.weights[t_long.contributing_sequences.index(0)]
    assert w_oldest_short < w_oldest_long  # 짧은 half-life가 oldest를 더 강하게 죽임


# ---------------------------------------------------------------------------
# Case D (섹션 10): max_contributors=2 -> 가장 최신 2개만 사용
# ---------------------------------------------------------------------------


def test_case_d_max_contributors_limits_to_newest_n() -> None:
    oldest = _chunk(sequence=0, obs_time=10.00)
    middle = _chunk(sequence=1, obs_time=10.34)
    newest = _chunk(sequence=2, obs_time=10.68)
    ensembler = TemporalEnsembler(max_contributors=2, half_life_s=0.338)

    target = ensembler.compute_target([oldest, middle, newest], target_time_monotonic=10.80)

    assert target.num_contributors == 2
    assert target.contributing_sequences == (2, 1)  # oldest(seq 0)는 제외됨
    assert 0 not in target.contributing_sequences


# ---------------------------------------------------------------------------
# Case E (섹션 10): 유효 chunk 1개뿐 -> 그 trajectory sample 그대로 반환
# ---------------------------------------------------------------------------


def test_case_e_single_contributor_returns_its_interpolated_sample_exactly() -> None:
    chunk = _chunk(sequence=7, obs_time=1.0, value_at_k=lambda k: 42.0 + k)
    ensembler = TemporalEnsembler()

    target = ensembler.compute_target([chunk], target_time_monotonic=1.0 + 3 * SPACING)

    assert target.num_contributors == 1
    assert target.weights == (1.0,)
    assert target.contributing_sequences == (7,)
    for joint in JOINT_ORDER:
        assert target.action[joint] == pytest.approx(45.0)  # 42+3


# ---------------------------------------------------------------------------
# Case F (섹션 6/10): 0 contributors -> fail-closed None (extrapolation 없음)
# ---------------------------------------------------------------------------


def test_case_f_no_chunks_returns_none() -> None:
    ensembler = TemporalEnsembler()
    assert ensembler.compute_target([], target_time_monotonic=100.0) is None


def test_case_f_target_time_before_any_chunk_observation_returns_none() -> None:
    chunk = _chunk(sequence=0, obs_time=100.0)
    ensembler = TemporalEnsembler()
    assert ensembler.compute_target([chunk], target_time_monotonic=99.0) is None


def test_case_f_target_time_far_beyond_all_horizons_returns_none() -> None:
    chunk = _chunk(sequence=0, obs_time=100.0, chunk_size=10)  # horizon 짧게
    ensembler = TemporalEnsembler()
    far_future = 100.0 + 1000 * SPACING
    assert ensembler.compute_target([chunk], target_time_monotonic=far_future) is None


# ---------------------------------------------------------------------------
# Case G (섹션 10): expired(범위 밖) chunk 제외, 유효한 것만 기여
# ---------------------------------------------------------------------------


def test_case_g_out_of_range_chunk_excluded_valid_one_still_used() -> None:
    stale = _chunk(sequence=0, obs_time=0.0, chunk_size=10)  # T를 커버 못 함(horizon 짧음)
    fresh = _chunk(sequence=1, obs_time=0.90, chunk_size=CHUNK_SIZE)  # T를 커버함
    ensembler = TemporalEnsembler()
    target_time = 1.0  # stale의 horizon(0.0~10*spacing≈0.333)을 훨씬 넘음

    target = ensembler.compute_target([stale, fresh], target_time_monotonic=target_time)

    assert target is not None
    assert target.num_contributors == 1
    assert target.contributing_sequences == (1,)  # stale(seq 0)은 제외됨


# ---------------------------------------------------------------------------
# Case H (섹션 10): target time이 특정 chunk의 horizon 밖 -> 그 chunk만 contributor에서 제외
# ---------------------------------------------------------------------------


def test_case_h_target_beyond_one_chunk_horizon_but_within_another() -> None:
    short_horizon = _chunk(sequence=0, obs_time=0.0, chunk_size=5)  # last_sample=4*spacing≈0.1333
    long_horizon = _chunk(sequence=1, obs_time=0.0, chunk_size=CHUNK_SIZE)  # last_sample≈1.633
    ensembler = TemporalEnsembler()
    target_time = 0.5  # short_horizon 범위 밖, long_horizon 범위 안

    target = ensembler.compute_target([short_horizon, long_horizon], target_time_monotonic=target_time)

    assert target is not None
    assert target.num_contributors == 1
    assert target.contributing_sequences == (1,)


def test_covers_boundary_exact_last_sample_time_included() -> None:
    chunk = _chunk(sequence=0, obs_time=0.0, chunk_size=10)
    last_sample_time = chunk.nominal_target_time(9)
    ensembler = TemporalEnsembler()
    assert ensembler.compute_target([chunk], target_time_monotonic=last_sample_time) is not None
    just_past = last_sample_time + SPACING  # 그 다음 슬롯(존재하지 않음)
    assert ensembler.compute_target([chunk], target_time_monotonic=just_past) is None


# ---------------------------------------------------------------------------
# Case I (섹션 7/10): NaN/Inf defense-in-depth
# ---------------------------------------------------------------------------


def test_case_i_nan_contributor_excluded_good_one_still_used() -> None:
    def _bad_values(k: int) -> float:
        return math.nan if k == 5 else float(k)

    bad = _chunk(sequence=0, obs_time=0.0, value_at_k=_bad_values)
    good = _chunk(sequence=1, obs_time=0.0, value_at_k=lambda k: 10.0 * k)
    ensembler = TemporalEnsembler()
    target_time = 5 * SPACING  # bad chunk의 index 5(=NaN)를 정확히 가리킴

    target = ensembler.compute_target([bad, good], target_time_monotonic=target_time)

    assert target is not None
    assert target.num_contributors == 1
    assert target.contributing_sequences == (1,)  # bad(seq 0)는 배제됨
    for v in target.action.values():
        assert math.isfinite(v)


def test_case_i_all_contributors_invalid_returns_none() -> None:
    def _all_nan(k: int) -> float:
        return math.nan

    bad1 = _chunk(sequence=0, obs_time=0.0, value_at_k=_all_nan)
    bad2 = _chunk(sequence=1, obs_time=0.01, value_at_k=_all_nan)
    ensembler = TemporalEnsembler()
    assert ensembler.compute_target([bad1, bad2], target_time_monotonic=0.05) is None


def test_case_i_inf_contributor_excluded() -> None:
    def _inf_at_3(k: int) -> float:
        return math.inf if k == 3 else float(k)

    bad = _chunk(sequence=0, obs_time=0.0, value_at_k=_inf_at_3)
    good = _chunk(sequence=1, obs_time=0.0, value_at_k=lambda k: 1.0)
    ensembler = TemporalEnsembler()
    target = ensembler.compute_target([bad, good], target_time_monotonic=3 * SPACING)
    assert target is not None
    assert target.contributing_sequences == (1,)


# ---------------------------------------------------------------------------
# Case J (섹션 8/10): gripper도 동일한 weighted average 적용 (특별 취급 없음)
# ---------------------------------------------------------------------------


def test_case_j_gripper_uses_same_weighted_average_as_arm_joints() -> None:
    def _distinct_per_joint(base_arm: float, base_gripper: float):
        def _fn(k: int) -> dict[str, float]:
            return {j: (base_gripper if j == "gripper" else base_arm) for j in JOINT_ORDER}
        return _fn

    # _chunk 헬퍼는 단일 value_at_k(k)->float를 기대하므로, gripper 전용 chunk를 직접 만든다.
    def _make(sequence, obs_time, arm_val, gripper_val):
        actions = tuple(
            {**{j: arm_val for j in JOINT_ORDER}, "gripper": gripper_val} for _ in range(CHUNK_SIZE)
        )
        return TimestampedActionChunk(
            sequence=sequence, session_id="s1", observation_time_monotonic=obs_time,
            request_started_time_monotonic=obs_time, response_received_time_monotonic=obs_time + 0.05,
            server_received_at=None, server_responded_at=None, inference_latency_ms=50.0,
            chunk_index_spacing_s=SPACING, chunk_size=CHUNK_SIZE, actions=actions,
            model_id="fake", backend="fake",
        )

    chunk_a = _make(0, 0.0, arm_val=10.0, gripper_val=90.0)
    chunk_b = _make(1, 0.0, arm_val=20.0, gripper_val=10.0)
    ensembler = TemporalEnsembler(half_life_s=0.338)

    target = ensembler.compute_target([chunk_a, chunk_b], target_time_monotonic=0.0)

    assert target is not None
    w_a, w_b = None, None
    for seq, w in zip(target.contributing_sequences, target.weights):
        if seq == 0:
            w_a = w
        else:
            w_b = w
    expected_arm = w_a * 10.0 + w_b * 20.0
    expected_gripper = w_a * 90.0 + w_b * 10.0
    assert target.action["shoulder_pan"] == pytest.approx(expected_arm)
    assert target.action["gripper"] == pytest.approx(expected_gripper)
    # 동일한 (w_a, w_b) 가중치가 arm/gripper 양쪽에 똑같이 쓰였는지(별도 취급 없음) 확인.
    assert (target.action["gripper"] - w_a * 90.0 - w_b * 10.0) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# EnsembledTarget 필드 존재/타입 스모크
# ---------------------------------------------------------------------------


def test_ensembled_target_has_all_required_fields() -> None:
    chunk = _chunk(sequence=0, obs_time=0.0)
    ensembler = TemporalEnsembler()
    target = ensembler.compute_target([chunk], target_time_monotonic=0.0)
    assert isinstance(target, EnsembledTarget)
    for field in (
        "target_time_monotonic", "action", "contributing_sequences", "contributing_chunk_indices",
        "weights", "newest_observation_age_ms", "oldest_observation_age_ms", "num_contributors",
    ):
        assert hasattr(target, field)
    assert set(target.action) == set(JOINT_ORDER)


# ---------------------------------------------------------------------------
# lookahead_s / compute_target_for_now
# ---------------------------------------------------------------------------


def test_lookahead_zero_by_default_uses_now_directly() -> None:
    chunk = _chunk(sequence=0, obs_time=0.0, value_at_k=lambda k: 10.0 * k)
    ensembler = TemporalEnsembler(lookahead_s=0.0)
    now = 5 * SPACING
    target_now = ensembler.compute_target_for_now([chunk], now_monotonic=now)
    target_direct = ensembler.compute_target([chunk], target_time_monotonic=now)
    assert target_now.action == target_direct.action
    assert target_now.target_time_monotonic == pytest.approx(now)


def test_lookahead_positive_shifts_target_time_forward() -> None:
    chunk = _chunk(sequence=0, obs_time=0.0, value_at_k=lambda k: 10.0 * k)
    lookahead = 3 * SPACING
    ensembler = TemporalEnsembler(lookahead_s=lookahead)
    now = 2 * SPACING
    target = ensembler.compute_target_for_now([chunk], now_monotonic=now)
    assert target.target_time_monotonic == pytest.approx(now + lookahead)
    assert target.contributing_chunk_indices == ((5, 5),)  # (2+3)=index 5


# Phase continuity: admission/removal are weighted by actual contributor phase.
def _constant_chunk(sequence: int, obs_time: float, value: float, *, chunk_size: int = 50):
    actions = tuple({j: value for j in JOINT_ORDER} for _ in range(chunk_size))
    return TimestampedActionChunk(
        sequence=sequence, session_id="phase", observation_time_monotonic=obs_time,
        request_started_time_monotonic=obs_time,
        response_received_time_monotonic=obs_time + 0.05,
        server_received_at=None, server_responded_at=None, inference_latency_ms=50.0,
        chunk_index_spacing_s=SPACING, chunk_size=chunk_size, actions=actions,
        model_id="fake", backend="fake",
    )


def test_phase_continuity_a_contributor_addition_1_to_2() -> None:
    old = _constant_chunk(1, 0.0, 10.0)
    new = _constant_chunk(2, 0.2, 40.0)
    ens = TemporalEnsembler(phase_continuity=True)
    t = new.response_received_time_monotonic
    before = ens.compute_target([old], t)
    after = ens.compute_target([old, new], t)
    assert after.action == pytest.approx(before.action)


def test_phase_continuity_b_contributor_removal_2_to_1() -> None:
    old = _constant_chunk(1, 0.0, 10.0, chunk_size=10)
    new = _constant_chunk(2, 0.2, 40.0)
    ens = TemporalEnsembler(phase_continuity=True)
    last = old.nominal_target_time(old.chunk_size - 1)
    before = ens.compute_target([old, new], last - 1e-6)
    after = ens.compute_target([new], last + 1e-6)
    assert after.action["shoulder_pan"] == pytest.approx(before.action["shoulder_pan"], abs=1e-3)


def test_phase_continuity_c_three_contributor_roll() -> None:
    c1 = _constant_chunk(1, 0.0, 10.0)
    c2 = _constant_chunk(2, 0.2, 20.0)
    c3 = _constant_chunk(3, 0.4, 30.0)
    c4 = _constant_chunk(4, 0.6, 100.0)
    ens = TemporalEnsembler(max_contributors=3, phase_continuity=True)
    t = c4.response_received_time_monotonic
    before = ens.compute_target([c1, c2, c3], t)
    after = ens.compute_target([c1, c2, c3, c4], t)
    assert after.action == pytest.approx(before.action)


def test_phase_continuity_d_identical_chunks_remain_identical() -> None:
    old = _constant_chunk(1, 0.0, 7.0)
    new = _constant_chunk(2, 0.2, 7.0)
    target = TemporalEnsembler(phase_continuity=True).compute_target(
        [old, new], new.response_received_time_monotonic + 0.1
    )
    assert all(value == pytest.approx(7.0) for value in target.action.values())


def test_phase_continuity_e_small_difference_is_admitted_without_boundary_jump() -> None:
    old = _constant_chunk(1, 0.0, 10.0)
    new = _constant_chunk(2, 0.2, 10.5)
    ens = TemporalEnsembler(phase_continuity=True)
    t = new.response_received_time_monotonic
    at_boundary = ens.compute_target([old, new], t)
    mid = ens.compute_target([old, new], t + 0.1)
    assert at_boundary.action["elbow_flex"] == pytest.approx(10.0)
    assert 10.0 < mid.action["elbow_flex"] < 10.5


def test_phase_continuity_f_large_stochastic_difference_has_no_boundary_jump() -> None:
    old = _constant_chunk(1, 0.0, -50.0)
    new = _constant_chunk(2, 0.2, 80.0)
    ens = TemporalEnsembler(phase_continuity=True)
    t = new.response_received_time_monotonic
    before = ens.compute_target([old], t)
    after = ens.compute_target([old, new], t)
    assert after.action == pytest.approx(before.action)
