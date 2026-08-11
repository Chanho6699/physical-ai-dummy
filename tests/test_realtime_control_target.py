"""runtime/laptop/realtime_control_target.py의 RealTimeControlTargetGenerator 통합 검증
(Phase C-3A + C-3A.1 correction: Intent Validation 삽입 + feedforward/correction guard).

전부 synthetic TimestampedActionChunk + controllable timestamps로 결정적으로 검증한다.
실제 하드웨어 접근 없음 - connect/send_action 전혀 호출하지 않는다.
"""

from __future__ import annotations

import math

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.motion_guard import DEFAULT_JOINT_MOTION_LIMITS, JointMotionLimits
from runtime.laptop.realtime_control_target import (
    STOP_REASON_GUARD_INVALID,
    STOP_REASON_INTENT_PREFIX,
    STOP_REASON_NO_TARGET,
    STOP_REASON_SAFETY_PREFIX,
    STOP_REASON_STALE_TRAJECTORY,
    RealTimeControlTargetGenerator,
)
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
from runtime.laptop.temporal_ensemble import TemporalEnsembler
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

SPACING = 1.0 / 30.0
CHUNK_SIZE = 50
DT_60HZ = 1.0 / 60.0


def _neutral(value: float = 0.0) -> dict[str, float]:
    return {j: value for j in JOINT_ORDER}


def _chunk(*, sequence: int, obs_time: float, target_action: dict[str, float], chunk_size: int = CHUNK_SIZE, spacing: float = SPACING) -> TimestampedActionChunk:
    """모든 index k에 동일한 target_action을 채운 constant chunk - 이 테스트 스위트는
    "T에 따라 chunk 안에서 어떤 값이 나오는지"가 아니라 Intent Validation/Motion Guard/
    Final Safety 통합을 검증하는 게 목적이므로, ensemble의 raw target을 항상
    target_action으로 안정시킨다(시간 보간 자체의 정확성은 tests/test_temporal_ensemble.py가
    이미 검증)."""
    actions = tuple(dict(target_action) for _ in range(chunk_size))
    return TimestampedActionChunk(
        sequence=sequence, session_id="s1", observation_time_monotonic=obs_time,
        request_started_time_monotonic=obs_time, response_received_time_monotonic=obs_time + 0.05,
        server_received_at=None, server_responded_at=None, inference_latency_ms=50.0,
        chunk_index_spacing_s=spacing, chunk_size=chunk_size, actions=actions,
        model_id="fake", backend="fake",
    )


def _generous_safety_gate() -> SafetyGate:
    """Safety가 전혀 걸리지 않는 관대한 설정 - motion guard 자체의 동작만 격리해서
    보고 싶은 테스트용."""
    joint_range = {j: (-1000.0, 1000.0) for j in JOINT_ORDER}
    max_step = {j: 1000.0 for j in JOINT_ORDER}
    return SafetyGate(SafetyGateConfig(joint_range_deg=joint_range, max_step_deg=max_step))


def _tight_range_safety_gate() -> SafetyGate:
    """아주 좁은 mechanical range - raw target 자체가 범위를 크게 벗어나게 함."""
    joint_range = {j: (-1.0, 1.0) for j in JOINT_ORDER}
    max_step = {j: 1000.0 for j in JOINT_ORDER}  # step 자체는 안 걸리게(순수 range 검증)
    return SafetyGate(SafetyGateConfig(joint_range_deg=joint_range, max_step_deg=max_step))


def _tiny_step_safety_gate() -> SafetyGate:
    """아주 작은 max_step - 작은 step조차 GROSS_STEP(REJECT)에 걸리게 함."""
    joint_range = {j: (-1000.0, 1000.0) for j in JOINT_ORDER}
    max_step = {j: 1e-6 for j in JOINT_ORDER}
    return SafetyGate(SafetyGateConfig(joint_range_deg=joint_range, max_step_deg=max_step))


def _generous_motion_limits() -> dict[str, JointMotionLimits]:
    return {j: JointMotionLimits(velocity_limit=1000.0, acceleration_limit=1e7, jerk_limit=1e10) for j in JOINT_ORDER}


def _make_generator(*, safety_gate=None, motion_limits=None, control_hz=60.0, lookahead_s=0.0, half_life_s=0.338) -> RealTimeControlTargetGenerator:
    ensembler = TemporalEnsembler(half_life_s=half_life_s, lookahead_s=lookahead_s)
    return RealTimeControlTargetGenerator(
        ensembler=ensembler, safety_gate=safety_gate or _generous_safety_gate(),
        motion_limits=motion_limits or _generous_motion_limits(), control_hz=control_hz, lookahead_s=lookahead_s,
    )


def _ramp_chunk(*, sequence: int, obs_time: float, rate: float, chunk_size: int = CHUNK_SIZE, spacing: float = SPACING) -> TimestampedActionChunk:
    """실제 VLA chunk처럼 index별로 서로 다른(=시간에 따라 진짜 변화하는) action을
    담은 선형 ramp chunk - index k의 절대시간은 ``obs_time + k*spacing``이고 그 시점의
    목표값은 ``rate * (그 절대시간)``이다. ``_chunk()``(모든 index가 동일 상수)와 달리
    이 헬퍼는 Motion Guard의 feedforward 샘플링(``T``와 ``T+dt`` 두 시점)이 실제로
    의미 있는 기울기를 보게 한다 - 매 tick 새 "상수" chunk를 만들어 보내는 방식은 chunk
    *내부에* 속도 정보가 전혀 없어(항상 flat) feedforward=0으로 관측되는, 비현실적인
    테스트 아티팩트였다(이 파일 히스토리 참고)."""
    actions = tuple(_neutral(rate * (obs_time + k * spacing)) for k in range(chunk_size))
    return TimestampedActionChunk(
        sequence=sequence, session_id="s1", observation_time_monotonic=obs_time,
        request_started_time_monotonic=obs_time, response_received_time_monotonic=obs_time + 0.05,
        server_received_at=None, server_responded_at=None, inference_latency_ms=50.0,
        chunk_index_spacing_s=spacing, chunk_size=chunk_size, actions=actions,
        model_id="fake", backend="fake",
    )


def _real_generator(*, safety_gate=None) -> RealTimeControlTargetGenerator:
    """실제 recalibrated Safety threshold + 실제 demo 기반 motion limits를 쓰는
    generator - Intent/dangerous-wrist/normal-case 회귀 테스트 전용."""
    return RealTimeControlTargetGenerator(
        ensembler=TemporalEnsembler(half_life_s=0.338), safety_gate=safety_gate or SafetyGate(SafetyGateConfig.from_repo_defaults()),
        motion_limits=DEFAULT_JOINT_MOTION_LIMITS, control_hz=60.0,
    )


# ---------------------------------------------------------------------------
# 60Hz tick spacing / period
# ---------------------------------------------------------------------------


def test_control_hz_and_period_default_60hz() -> None:
    gen = _make_generator()
    assert gen.control_hz == pytest.approx(60.0)
    assert gen.period_s == pytest.approx(1.0 / 60.0)


def test_control_hz_configurable() -> None:
    gen = _make_generator(control_hz=50.0)
    assert gen.period_s == pytest.approx(0.02)


def test_control_hz_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _make_generator(control_hz=0.0)


# ---------------------------------------------------------------------------
# smooth constant trajectory - raw==current이면 guard가 아무것도 할 필요 없음
# ---------------------------------------------------------------------------


def test_constant_trajectory_no_motion_needed() -> None:
    gen = _make_generator()
    current = _neutral(5.0)
    chunk = _chunk(sequence=0, obs_time=100.0, target_action=_neutral(5.0))

    result = gen.tick(chunks=[chunk], now_monotonic=100.0, current_follower_state_deg=current)

    assert result.intent_decision == "ACCEPT"
    assert result.target_valid is True
    for j in JOINT_ORDER:
        assert result.guarded_target[j] == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# linear trajectory - 정상 속도 요청은 몇 tick 뒤 raw를 거의 그대로 따라감
# ---------------------------------------------------------------------------


def test_linear_trajectory_within_limits_tracks_closely_after_warmup() -> None:
    """chunk 내부에 진짜 ramp(속도 정보)가 encoding된 realistic 케이스 - feedforward가
    이 기울기를 그대로 샘플링하므로, correction_time_constant 지연 없이 (jerk/accel
    warmup만 지나면) raw를 거의 그대로 따라가야 한다."""
    limits = {j: JointMotionLimits(velocity_limit=50.0, acceleration_limit=5000.0, jerk_limit=1e7) for j in JOINT_ORDER}
    gen = _make_generator(motion_limits=limits)
    current = dict(_neutral(0.0))
    rate = 10.0  # deg/s, well within velocity_limit=50
    chunk = _ramp_chunk(sequence=0, obs_time=0.0, rate=rate)  # 하나의 chunk가 전체 궤적을 encoding
    now = 0.0
    for i in range(20):
        now += DT_60HZ
        result = gen.tick(chunks=[chunk], now_monotonic=now, current_follower_state_deg=current)
        assert result.target_valid is True
        current = result.guarded_target  # 다음 tick엔 "실제로 거기 도달했다"고 가정(오프라인 시뮬레이션)
    # 20 tick(≈0.33s) 후에는 warmup이 끝나고 요청 속도를 거의 그대로 따라가야 한다.
    assert current["shoulder_pan"] == pytest.approx(rate * now, abs=0.5)


# ---------------------------------------------------------------------------
# velocity / acceleration / jerk spike clamp (generator 레벨 - motion_guard 단위테스트와
# 별개로, tick() 전체 파이프라인을 통해서도 동일하게 동작하는지)
# ---------------------------------------------------------------------------


def test_velocity_spike_clamped_through_full_tick() -> None:
    limits = {j: JointMotionLimits(velocity_limit=10.0, acceleration_limit=1e6, jerk_limit=1e9) for j in JOINT_ORDER}
    gen = _make_generator(motion_limits=limits)
    current = _neutral(0.0)
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=_neutral(1000.0))

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    assert result.target_valid is True
    expected = 10.0 * gen.period_s  # 첫 tick, dt=period_s(이력 없음), target_lookahead==target_now(단일 chunk 상수)
    for j in JOINT_ORDER:
        assert result.guarded_target[j] == pytest.approx(expected, rel=1e-6)
        assert result.raw_ensemble_target[j] == pytest.approx(1000.0)  # raw는 그대로 보존


def test_acceleration_spike_clamped_through_full_tick() -> None:
    limits = {j: JointMotionLimits(velocity_limit=1e6, acceleration_limit=100.0, jerk_limit=1e9) for j in JOINT_ORDER}
    gen = _make_generator(motion_limits=limits)
    current = _neutral(0.0)
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=_neutral(1000.0))

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    dt = gen.period_s
    expected = 100.0 * dt * dt
    for j in JOINT_ORDER:
        assert result.guarded_target[j] == pytest.approx(expected, rel=1e-6)


def test_jerk_spike_clamped_through_full_tick() -> None:
    limits = {j: JointMotionLimits(velocity_limit=1e6, acceleration_limit=1e6, jerk_limit=1000.0) for j in JOINT_ORDER}
    gen = _make_generator(motion_limits=limits)
    current = _neutral(0.0)
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=_neutral(1000.0))

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    dt = gen.period_s
    expected = 1000.0 * dt ** 3
    for j in JOINT_ORDER:
        assert result.guarded_target[j] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# NaN fail closed
# ---------------------------------------------------------------------------


def test_nan_current_state_fails_closed() -> None:
    gen = _make_generator()
    current = _neutral(0.0)
    current["wrist_roll"] = math.nan
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=_neutral(1.0))

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    assert result.target_valid is False
    # NaN current_state는 Intent Validation의 SafetyGate.evaluate() 자체가 먼저 거부하거나
    # (validate_joint_dict 방어) Motion Guard가 GUARD_INVALID로 거부한다 - 어느 쪽이든
    # fail-closed(target_valid=False)면 충분하다.
    assert result.stop_reason.startswith(STOP_REASON_INTENT_PREFIX) or result.stop_reason.startswith(STOP_REASON_GUARD_INVALID)
    assert result.guarded_target is None


def test_nan_raw_target_fails_closed() -> None:
    """raw ensemble target 자체가 NaN이면(방어적 상황 - 정상 경로에서는 ensembler가
    걸러내지만, defense-in-depth로 이 레이어도 fail-closed여야 한다) Intent
    Validation 단계에서 이미 막혀야 한다."""
    gen = _make_generator()
    current = _neutral(0.0)
    chunk = _chunk(sequence=0, obs_time=0.0, target_action={**_neutral(0.0), "wrist_roll": math.nan})

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    assert result.target_valid is False
    assert result.guarded_target is None


# ---------------------------------------------------------------------------
# no trajectory / expired trajectory fail closed
# ---------------------------------------------------------------------------


def test_no_chunks_fails_closed_no_target() -> None:
    gen = _make_generator()
    result = gen.tick(chunks=[], now_monotonic=100.0, current_follower_state_deg=_neutral())
    assert result.target_valid is False
    assert result.stop_reason == STOP_REASON_NO_TARGET
    assert result.raw_ensemble_target is None


def test_expired_trajectory_fails_closed_stale() -> None:
    gen = _make_generator()
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=_neutral(1.0), chunk_size=5)  # 짧은 horizon
    now = 100.0  # chunk horizon(0~5*spacing≈0.167s)을 훨씬 넘음
    result = gen.tick(chunks=[chunk], now_monotonic=now, current_follower_state_deg=_neutral())
    assert result.target_valid is False
    assert result.stop_reason == STOP_REASON_STALE_TRAJECTORY


# ---------------------------------------------------------------------------
# Intent Validation이 raw target 단계에서 target eligibility를 막는지 (신규,
# C-3A.1 correction 섹션 1/2)
# ---------------------------------------------------------------------------


def test_mechanical_range_violation_blocks_target_eligibility() -> None:
    """raw target이 mechanical joint_range_deg를 크게 벗어나면(이 SafetyGate 설정
    기준) 지금 파이프라인에서는 **Intent Validation 단계에서 이미** 막힌다 - raw
    target 자체가 신뢰할 수 없는 policy 의도이기 때문이다. 그리고 설령 Intent를
    어떻게든 통과한 값이 있더라도(이 테스트가 다루는 케이스는 아님), Final SafetyGate가
    guarded target에 대해 동일한 mechanical range 체크를 다시 수행하므로(코드 6단계,
    ``realtime_control_target.py`` 참고) mechanical range는 두 레이어 모두에서 근본적으로
    보장된다 - 어느 쪽이 실제로 먼저 걸리는지는 raw/guarded 값에 따라 달라질 뿐, "막힌다"는
    보장 자체는 항상 성립한다."""
    gate = _tight_range_safety_gate()  # range=[-1,1]
    limits = {j: JointMotionLimits(velocity_limit=100.0, acceleration_limit=1e6, jerk_limit=1e9) for j in JOINT_ORDER}
    gen = _make_generator(safety_gate=gate, motion_limits=limits)
    current = _neutral(0.99)  # range 경계 바로 안쪽
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=_neutral(50.0))  # range 밖 raw target

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    assert result.target_valid is False
    assert result.stop_reason.startswith(STOP_REASON_INTENT_PREFIX)  # 이 설정에서는 Intent가 먼저 막음
    assert result.intent_decision in ("WOULD_CLAMP", "REJECT")
    assert result.guarded_target is None  # Motion Guard까지 도달하지 않았음


def test_safety_reject_blocks_target_eligibility_at_intent_stage() -> None:
    gate = _tiny_step_safety_gate()  # max_step=1e-6 -> gross(5x)도 5e-6
    limits = {j: JointMotionLimits(velocity_limit=100.0, acceleration_limit=1e6, jerk_limit=1e9) for j in JOINT_ORDER}
    gen = _make_generator(safety_gate=gate, motion_limits=limits)
    current = _neutral(0.0)
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=_neutral(50.0))

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    assert result.intent_decision == "REJECT"
    assert result.target_valid is False
    assert result.stop_reason == f"{STOP_REASON_INTENT_PREFIX}REJECT"
    assert result.guarded_target is None


def test_final_safety_still_evaluates_guarded_target_when_intent_passes() -> None:
    """Intent가 raw target을 승인한 정상 케이스에서도, Final SafetyGate가 guarded
    target에 대해 실제로(다시) 평가돼 ``safety_decision``이 채워지는지 확인한다 -
    "Intent만 통과하면 Final은 형식적으로 통과된다"가 아니라 파이프라인의 마지막
    단계로 실제로 실행됨을 보장."""
    gen = _make_generator()  # 관대한 safety/motion
    current = _neutral(0.0)
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=_neutral(1.0))

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    assert result.intent_decision == "ACCEPT"
    assert result.safety_decision == "ACCEPT"  # Final 단계가 실제로 실행되어 별도로 ACCEPT를 냄
    assert result.target_valid is True


# ---------------------------------------------------------------------------
# Dangerous wrist regression (C-3A.1 correction 섹션 2/9/10) - 실제 recalibrated
# Safety + 실제 demo 기반 motion limits로, 이전 실물 사례(wrist_flex 53.67 -> 40.49,
# delta≈13.18deg, training distribution 밖 true outlier로 분류됨)를 재현한다.
#
# [정정] 이전 C-3A 보고서는 "Motion Guard가 이 값을 잘게 쪼개 매 tick ACCEPT시키지만
# mechanical hard limit은 우회하지 않는다"고 결론지었다 - 이 결론은 사용자가 지적한
# 대로 **policy intent/outlier Safety semantic을 우회한다는 점에서 부정확했다.** 이제는
# Intent Validation이 Motion Guard보다 먼저 raw target을 검사하므로, 이 위험한 target은
# **Motion Guard에 도달하기 전에** 막힌다 - "잘게 쪼개서 우회"가 구조적으로 불가능해졌다.
# ---------------------------------------------------------------------------


def test_dangerous_wrist_raw_target_blocked_before_motion_guard() -> None:
    """실제 Safety threshold(wrist_flex=4.01)와 실제 demo 기반 motion limits로,
    위험한 wrist_flex raw target(delta=13.18deg > threshold, < gross 5x=20.05deg이므로
    REJECT가 아니라 WOULD_CLAMP)이 Intent Validation에서 즉시 막히고, target_valid=False,
    guarded_target=None(=Motion Guard가 이 tick에 대해 전혀 호출되지 않았음)임을 확인한다."""
    gen = _real_generator()
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703  # 실제 실물 사례의 before 값
    target = dict(current)
    target["wrist_flex"] = 40.4879  # 실제 실물 사례의 raw predict 값 (delta ≈ -13.18)
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=target)

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    assert result.intent_decision == "WOULD_CLAMP"
    assert result.target_valid is False
    assert result.stop_reason == f"{STOP_REASON_INTENT_PREFIX}WOULD_CLAMP"
    # Motion Guard가 전혀 호출되지 않았음 - guarded/smoothed/safety_decision이 전부 비어있다.
    assert result.guarded_target is None
    assert result.smoothed_target is None
    assert result.safety_decision is None
    # raw target 자체는 진단용으로 보존된다(무엇이 왜 막혔는지 로그로 남기기 위함).
    assert result.raw_ensemble_target["wrist_flex"] == pytest.approx(40.4879)


def test_dangerous_wrist_cannot_be_chopped_into_pieces_to_bypass_intent() -> None:
    """섹션 10 명시 요구: "Motion Guard가 위험한 raw target을 잘게 쪼개 Safety를
    우회할 수 없음"을 직접 증명한다. current_follower_state_deg를 매 tick 고정한 채
    (=guarded_target이 없으므로 피드백할 값 자체가 없음 - Intent가 막으면 애초에 다음
    tick으로 이어질 "진행된 위치"가 생기지 않는다) 여러 tick을 반복해도, 매번 동일하게
    INTENT_WOULD_CLAMP로 막히고 단 한 번도 Motion Guard 이하 단계에 도달하지 않는다 -
    즉 "60 tick에 걸쳐 결국 원래 목표에 도달"하던 이전 C-3A 버그가 더 이상 존재하지
    않는다."""
    gen = _real_generator()
    current = _neutral(0.0)
    current["wrist_flex"] = 53.6703
    target = dict(_neutral(0.0))
    target["wrist_flex"] = 40.4879

    now = 0.0
    for i in range(60):  # 1초 분량 - 이전 버그 재현 테스트와 동일한 tick 수
        now += DT_60HZ
        chunk = _chunk(sequence=i, obs_time=now, target_action=target)
        result = gen.tick(chunks=[chunk], now_monotonic=now, current_follower_state_deg=current)
        assert result.target_valid is False, f"tick {i}에서 예상과 다르게 target_valid=True"
        assert result.stop_reason == f"{STOP_REASON_INTENT_PREFIX}WOULD_CLAMP"
        assert result.guarded_target is None
        # current를 갱신하지 않는다 - 실제로도 write가 일어나지 않으므로(target_valid=False)
        # follower는 물리적으로 전혀 움직이지 않았을 것이기 때문에 이게 올바른 오프라인 시뮬레이션이다.

    # 1초 동안 단 한 번도 이 위험한 target 방향으로 "진행"하지 못했다 - current는 시작
    # 위치 그대로다(우리가 애초에 갱신하지 않았으므로 자명하지만, 이게 바로 핵심 안전 속성:
    # write eligibility가 전혀 생기지 않으므로 실제 follower도 절대 움직이지 않는다).
    assert current["wrist_flex"] == pytest.approx(53.6703)


# ---------------------------------------------------------------------------
# 정상 Candidate B target은 Intent Validation을 통과 (C-3A.1 correction 섹션 3) -
# 재캘리브레이션된 threshold(pan 6.00/lift 9.00/elbow 8.50) 기준.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("joint", "delta"),
    [
        ("elbow_flex", 7.08),
        ("shoulder_pan", 5.45),
        ("shoulder_lift", 5.63),
    ],
)
def test_normal_candidate_b_target_passes_intent_validation(joint: str, delta: float) -> None:
    gen = _real_generator()
    current = _neutral(0.0)
    target = dict(current)
    target[joint] = delta
    chunk = _chunk(sequence=0, obs_time=0.0, target_action=target)

    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=current)

    assert result.intent_decision == "ACCEPT"
    # 정상 케이스는 Intent뿐 아니라 전체 파이프라인도 끝까지 통과해야 한다(guard가
    # 정상 속도를 막지 않는다는 것도 함께 확인).
    assert result.target_valid is True
    assert result.safety_decision == "ACCEPT"


# ---------------------------------------------------------------------------
# demo-like trajectory mostly unaffected (실제 demo 기반 motion limits + 정상 속도)
# ---------------------------------------------------------------------------


def test_demo_like_slow_trajectory_barely_clamped() -> None:
    """chunk 내부에 진짜 ramp가 encoding된 realistic 케이스(위 테스트와 동일한 이유)."""
    gate = _generous_safety_gate()
    gen = RealTimeControlTargetGenerator(
        ensembler=TemporalEnsembler(half_life_s=0.338), safety_gate=gate,
        motion_limits=DEFAULT_JOINT_MOTION_LIMITS, control_hz=60.0,
    )
    current = _neutral(0.0)
    rate = 5.0  # deg/s - demo shoulder_lift p50=18.46, 이 값보다 느긋한 속도
    chunk = _ramp_chunk(sequence=0, obs_time=0.0, rate=rate)
    now = 0.0
    for i in range(30):
        now += DT_60HZ
        result = gen.tick(chunks=[chunk], now_monotonic=now, current_follower_state_deg=current)
        assert result.target_valid is True
        # warmup 이후에는 raw와 guarded가 거의 같아야 한다(정상 teleop 속도 안 막힘).
        if i > 10:
            assert abs(result.guarded_target["shoulder_pan"] - result.raw_ensemble_target["shoulder_pan"]) < 0.1
        current = result.guarded_target


# ---------------------------------------------------------------------------
# guard state reset behavior
# ---------------------------------------------------------------------------


def test_reset_guard_state_restarts_like_fresh_first_tick() -> None:
    limits = {j: JointMotionLimits(velocity_limit=10.0, acceleration_limit=1e6, jerk_limit=1e9) for j in JOINT_ORDER}
    gen = _make_generator(motion_limits=limits)
    current = _neutral(0.0)

    # 먼저 몇 tick 돌려서 velocity 이력을 쌓는다.
    now = 0.0
    for i in range(5):
        now += DT_60HZ
        chunk = _chunk(sequence=i, obs_time=now, target_action=_neutral(1000.0))
        result = gen.tick(chunks=[chunk], now_monotonic=now, current_follower_state_deg=current)
        current = result.guarded_target
    assert current["shoulder_pan"] > 0  # 이미 어느 정도 움직인 상태

    gen.reset_guard_state()
    fresh_current = _neutral(0.0)  # 실제로도 리셋된 것처럼(테스트 목적상 위치도 초기화)
    now += DT_60HZ
    chunk = _chunk(sequence=99, obs_time=now, target_action=_neutral(1000.0))
    result = gen.tick(chunks=[chunk], now_monotonic=now, current_follower_state_deg=fresh_current)

    # reset 직후 첫 tick은 dt=period_s로 계산된, "첫 tick과 동일한" 결과가 나와야 한다.
    expected_first_tick = 10.0 * gen.period_s  # velocity_limit_binding과 동일한 계산(accel/jerk 관대함)
    assert result.guarded_target["shoulder_pan"] == pytest.approx(expected_first_tick, rel=1e-6)


# ---------------------------------------------------------------------------
# variable dt handling
# ---------------------------------------------------------------------------


def test_variable_dt_uses_actual_elapsed_time_not_nominal_period() -> None:
    limits = {j: JointMotionLimits(velocity_limit=10.0, acceleration_limit=1e6, jerk_limit=1e9) for j in JOINT_ORDER}
    gen = _make_generator(motion_limits=limits)
    current = _neutral(0.0)

    chunk0 = _chunk(sequence=0, obs_time=0.0, target_action=_neutral(1000.0))
    r0 = gen.tick(chunks=[chunk0], now_monotonic=0.0, current_follower_state_deg=current)
    current = r0.guarded_target

    # 두 번째 tick을 nominal period(1/60≈16.67ms)가 아니라 50ms 뒤에 호출 - 실제 다른 dt.
    big_dt = 0.05
    chunk1 = _chunk(sequence=1, obs_time=big_dt, target_action=_neutral(1000.0))
    r1 = gen.tick(chunks=[chunk1], now_monotonic=big_dt, current_follower_state_deg=current)

    # velocity가 이미 velocity_limit(10)에 도달해 있으므로, 이번 tick 이동량은
    # velocity_limit * big_dt(실제 경과 시간)여야 한다 - nominal period가 아니라.
    expected_delta = 10.0 * big_dt
    actual_delta = r1.guarded_target["shoulder_pan"] - current["shoulder_pan"]
    assert actual_delta == pytest.approx(expected_delta, rel=1e-6)


# ---------------------------------------------------------------------------
# raw target/contributing_sequences/필드 보존 스모크
# ---------------------------------------------------------------------------


def test_result_preserves_contributing_sequences_and_raw_target() -> None:
    gen = _make_generator()
    chunk = _chunk(sequence=42, obs_time=0.0, target_action=_neutral(1.0))
    result = gen.tick(chunks=[chunk], now_monotonic=0.0, current_follower_state_deg=_neutral())
    assert result.contributing_sequences == (42,)
    assert result.raw_ensemble_target is not None
    assert result.smoothed_target == result.guarded_target
    assert result.intent_decision == "ACCEPT"
