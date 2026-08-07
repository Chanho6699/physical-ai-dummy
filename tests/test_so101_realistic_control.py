"""simulation/realism/so101_realistic_control.py 테스트.

전부 hardware-free/시뮬레이터-free 순수 유닛 테스트다 (mujoco도 import하지 않는다).
합성(synthetic) action 값만 사용해 4가지 v1 특성(rate/frame-delta, wrist_roll deadband,
latency, historical range diagnostic)과 baseline identity/결정론을 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.realism.so101_control_profile import load_control_profile
from simulation.realism.so101_realistic_control import (
    CSV_FIELDNAMES,
    RealisticControlConfig,
    RealisticControlError,
    RealisticControlLayer,
    RealisticControlRecorder,
    compute_tracking_error,
    config_all_disabled,
)

REAL_PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "generated" / "so101_control_profile_candidate_v1.json"


@pytest.fixture()
def profile():
    return load_control_profile(REAL_PROFILE_PATH)


def _all_joint_action(value: float = 0.0, **overrides) -> dict[str, float]:
    base = {
        "shoulder_pan": value,
        "shoulder_lift": value,
        "elbow_flex": value,
        "wrist_flex": value,
        "wrist_roll": value,
        "gripper": value,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Baseline identity: 4가지 특성을 모두 끄면 desired == processed (섹션 10 "Baseline")
# ---------------------------------------------------------------------------


def test_all_disabled_config_is_exact_passthrough(profile):
    layer = RealisticControlLayer(profile, config_all_disabled())
    desired = _all_joint_action(shoulder_pan=10.0, wrist_roll=0.2, gripper=45.0)

    result = layer.process(desired, now=0.0)
    assert result.processed_action == desired

    # 큰 점프도 rate limit 없이 즉시 반영되어야 한다 (baseline과 동일해야 하므로).
    jump = _all_joint_action(shoulder_pan=90.0, wrist_roll=50.0, gripper=90.0)
    result2 = layer.process(jump, now=0.1)
    assert result2.processed_action == jump
    for diag in result2.diagnostics.values():
        assert diag.rate_limited is False
        assert diag.deadband_applied is False
        assert diag.latency_applied_ms == 0.0


def test_unknown_joint_raises(profile):
    layer = RealisticControlLayer(profile, config_all_disabled())
    with pytest.raises(RealisticControlError):
        layer.process({"not_a_joint": 1.0}, now=0.0)


# ---------------------------------------------------------------------------
# A. rate / frame-delta 특성
# ---------------------------------------------------------------------------


def test_large_delta_is_soft_limited_to_profile_candidate(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=False, enable_rate_limit=True, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)

    layer.process(_all_joint_action(shoulder_pan=0.0), now=0.0)
    # shoulder_pan candidate_frame_delta_soft_limit ~= 1.4066 deg. 훨씬 큰 점프를 준다.
    result = layer.process(_all_joint_action(shoulder_pan=50.0), now=0.01)

    limit = profile.joints["shoulder_pan"].frame_delta_soft_limit
    diag = result.diagnostics["shoulder_pan"]
    assert diag.rate_limited is True
    assert result.processed_action["shoulder_pan"] == pytest.approx(limit, abs=1e-9)
    assert diag.processed_action < 50.0


def test_small_delta_within_candidate_is_not_rate_limited(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=False, enable_rate_limit=True, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)

    layer.process(_all_joint_action(shoulder_pan=0.0), now=0.0)
    limit = profile.joints["shoulder_pan"].frame_delta_soft_limit
    small_step = limit * 0.5
    result = layer.process(_all_joint_action(shoulder_pan=small_step), now=0.01)

    diag = result.diagnostics["shoulder_pan"]
    assert diag.rate_limited is False
    assert result.processed_action["shoulder_pan"] == pytest.approx(small_step)


def test_rate_limit_is_labeled_realism_not_hard_safety_limit(profile):
    """섹션 6-A: candidate 값은 hard safety limit이 아니다 - config 필드 이름/문서에
    'safety'라는 단어가 쓰이지 않는지 소스 레벨로도 확인한다."""
    import inspect

    from simulation.realism import so101_realistic_control as mod

    source = inspect.getsource(mod)
    assert "safety_limit" not in source.lower()
    assert "hard_limit" not in source.lower()


def test_disabling_rate_limit_allows_full_jump(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=False, enable_rate_limit=False, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)
    layer.process(_all_joint_action(shoulder_pan=0.0), now=0.0)
    result = layer.process(_all_joint_action(shoulder_pan=50.0), now=0.01)
    assert result.processed_action["shoulder_pan"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# B. wrist_roll deadband (REALISM_APPROXIMATION)
# ---------------------------------------------------------------------------


def test_wrist_roll_small_change_within_no_response_region_is_held(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=True, enable_rate_limit=False, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)

    layer.process(_all_joint_action(wrist_roll=0.0), now=0.0, simulated_actual=_all_joint_action(wrist_roll=0.0))
    threshold = profile.wrist_roll_deadband.no_response_upper_deg
    tiny_change = threshold * 0.5  # no-response candidate 폭보다 작은 변화

    result = layer.process(
        _all_joint_action(wrist_roll=tiny_change), now=0.01, simulated_actual=_all_joint_action(wrist_roll=0.0)
    )
    diag = result.diagnostics["wrist_roll"]
    assert diag.deadband_applied is True
    assert result.processed_action["wrist_roll"] == pytest.approx(0.0)  # 직전 명령 유지


def test_wrist_roll_transition_region_change_passes_through(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=True, enable_rate_limit=False, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)

    layer.process(_all_joint_action(wrist_roll=0.0), now=0.0, simulated_actual=_all_joint_action(wrist_roll=0.0))
    # transition_start_tick(6)에 해당하는 각도보다 큰 변화 - no-response 상한(threshold)의 2배.
    threshold = profile.wrist_roll_deadband.no_response_upper_deg
    big_change = threshold * 3.0

    result = layer.process(
        _all_joint_action(wrist_roll=big_change), now=0.01, simulated_actual=_all_joint_action(wrist_roll=0.0)
    )
    diag = result.diagnostics["wrist_roll"]
    assert diag.deadband_applied is False
    assert result.processed_action["wrist_roll"] == pytest.approx(big_change)


def test_deadband_only_applies_to_wrist_roll_not_other_joints(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=True, enable_rate_limit=False, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)

    layer.process(_all_joint_action(shoulder_pan=0.0), now=0.0, simulated_actual=_all_joint_action(shoulder_pan=0.0))
    tiny = profile.wrist_roll_deadband.no_response_upper_deg * 0.5
    result = layer.process(
        _all_joint_action(shoulder_pan=tiny), now=0.01, simulated_actual=_all_joint_action(shoulder_pan=0.0)
    )
    diag = result.diagnostics["shoulder_pan"]
    assert diag.deadband_applied is False
    assert result.processed_action["shoulder_pan"] == pytest.approx(tiny)


def test_deadband_diagnostic_is_labeled_realism_approximation_not_absolute_fact():
    """섹션 6-B: "0~5 tick이면 절대 안 움직인다"는 표현을 코드/주석에 쓰지 않는지 확인."""
    import inspect

    from simulation.realism import so101_realistic_control as mod

    source = inspect.getsource(mod)
    assert "REALISM_APPROXIMATION" in source
    # 코드에 "6 ticks가 항상 반응함을 보장한다"는 식의 절대적 표현이 없는지 확인한다
    # (docstring이 "그렇게 부르지 않는다"는 취지로 이 단어를 인용하는 것 자체는 허용).
    assert "guarantees response" not in source.lower()
    assert "always respond" not in source.lower()


def test_deadband_uses_last_output_when_simulated_actual_not_provided(profile):
    """simulated_actual을 안 주면 직전 processed 값을 기준으로 deadband를 판단해야 한다."""
    config = RealisticControlConfig(enable_latency=False, enable_deadband=True, enable_rate_limit=False, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)

    layer.process(_all_joint_action(wrist_roll=0.0), now=0.0)  # simulated_actual 없음
    threshold = profile.wrist_roll_deadband.no_response_upper_deg
    result = layer.process(_all_joint_action(wrist_roll=threshold * 0.5), now=0.01)
    assert result.diagnostics["wrist_roll"].deadband_applied is True
    assert result.processed_action["wrist_roll"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# C. command latency
# ---------------------------------------------------------------------------


def test_latency_delays_command_until_profile_median_elapsed(profile):
    config = RealisticControlConfig(enable_latency=True, enable_deadband=False, enable_rate_limit=False, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)
    median_s = profile.timing.latency_median_ms / 1000.0

    # 첫 샘플(t=0)은 즉시 채택된다 (연결 시작 시점 - 지연 기준점).
    r0 = layer.process(_all_joint_action(shoulder_pan=0.0), now=0.0)
    assert r0.processed_action["shoulder_pan"] == pytest.approx(0.0)

    # median latency 이전: 새 값(10.0)이 아직 반영되면 안 된다.
    r1 = layer.process(_all_joint_action(shoulder_pan=10.0), now=median_s * 0.3)
    assert r1.processed_action["shoulder_pan"] == pytest.approx(0.0)

    # median latency 이후: 이제는 반영되어야 한다 (동일한 값을 계속 보내 큐가 flush되게 함).
    r2 = layer.process(_all_joint_action(shoulder_pan=10.0), now=median_s * 1.5)
    assert r2.processed_action["shoulder_pan"] == pytest.approx(10.0)


def test_latency_disabled_applies_immediately(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=False, enable_rate_limit=False, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)

    layer.process(_all_joint_action(shoulder_pan=0.0), now=0.0)
    result = layer.process(_all_joint_action(shoulder_pan=10.0), now=0.001)
    assert result.processed_action["shoulder_pan"] == pytest.approx(10.0)
    assert result.diagnostics["shoulder_pan"].latency_applied_ms == 0.0


def test_latency_override_is_deterministic_for_tests(profile):
    config = RealisticControlConfig(
        enable_latency=True, enable_deadband=False, enable_rate_limit=False, enable_historical_range_diagnostic=False, latency_ms_override=50.0
    )
    layer = RealisticControlLayer(profile, config)

    # override=50ms - 첫 명령(10.0)은 t=0.03에 큐에 들어가 t=0.08(=0.03+0.05)에야 ready된다.
    layer.process(_all_joint_action(shoulder_pan=0.0), now=0.0)
    r1 = layer.process(_all_joint_action(shoulder_pan=10.0), now=0.03)
    assert r1.processed_action["shoulder_pan"] == pytest.approx(0.0)
    r2 = layer.process(_all_joint_action(shoulder_pan=10.0), now=0.09)
    assert r2.processed_action["shoulder_pan"] == pytest.approx(10.0)


def test_negative_latency_override_raises():
    with pytest.raises(RealisticControlError):
        RealisticControlConfig(latency_ms_override=-1.0)


# ---------------------------------------------------------------------------
# D. historical operating range - 진단만, clip 없음
# ---------------------------------------------------------------------------


def test_outside_historical_range_flags_but_does_not_clip(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=False, enable_rate_limit=False, enable_historical_range_diagnostic=True)
    layer = RealisticControlLayer(profile, config)

    lo, hi = profile.joints["shoulder_pan"].historical_range
    far_outside = hi + 50.0
    result = layer.process(_all_joint_action(shoulder_pan=far_outside), now=0.0)

    diag = result.diagnostics["shoulder_pan"]
    assert diag.outside_historical_range is True
    # clip되지 않음 - 값 그대로 통과해야 한다.
    assert result.processed_action["shoulder_pan"] == pytest.approx(far_outside)


def test_within_historical_range_is_not_flagged(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=False, enable_rate_limit=False, enable_historical_range_diagnostic=True)
    layer = RealisticControlLayer(profile, config)

    lo, hi = profile.joints["shoulder_pan"].historical_range
    midpoint = (lo + hi) / 2.0
    result = layer.process(_all_joint_action(shoulder_pan=midpoint), now=0.0)
    assert result.diagnostics["shoulder_pan"].outside_historical_range is False


def test_disabling_historical_range_diagnostic_never_flags(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=False, enable_rate_limit=False, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)
    lo, hi = profile.joints["shoulder_pan"].historical_range
    result = layer.process(_all_joint_action(shoulder_pan=hi + 100.0), now=0.0)
    assert result.diagnostics["shoulder_pan"].outside_historical_range is False


# ---------------------------------------------------------------------------
# gripper: percent semantics 유지 (섹션 8)
# ---------------------------------------------------------------------------


def test_gripper_uses_own_frame_delta_and_range_candidates_not_degree_scale(profile):
    """gripper의 candidate 값이 degree 관절과 다른(percent 단위) 값임을 그대로 쓰는지 확인.
    이 레이어는 gripper에 어떤 deg/s 또는 degree delta 스케일도 적용하지 않는다."""
    gripper_profile = profile.joints["gripper"]
    shoulder_pan_profile = profile.joints["shoulder_pan"]
    assert gripper_profile.unit == "percent_0_100"
    # 두 관절의 candidate 값이 서로 다른 소스(percent vs degree)에서 온 것이므로 같을
    # 필요가 없다는 것만 확인 - 레이어가 gripper에 shoulder_pan의 값을 재사용하지 않는다.
    assert gripper_profile.frame_delta_soft_limit != shoulder_pan_profile.frame_delta_soft_limit

    config = RealisticControlConfig(enable_latency=False, enable_deadband=False, enable_rate_limit=True, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)
    layer.process(_all_joint_action(gripper=0.0), now=0.0)
    result = layer.process(_all_joint_action(gripper=100.0), now=0.01)
    limit = gripper_profile.frame_delta_soft_limit
    assert result.processed_action["gripper"] == pytest.approx(limit, abs=1e-9)
    assert result.processed_action["gripper"] <= 100.0  # percent 범위를 넘어서는 재해석 없음


def test_deadband_never_applied_to_gripper(profile):
    config = RealisticControlConfig(enable_latency=False, enable_deadband=True, enable_rate_limit=False, enable_historical_range_diagnostic=False)
    layer = RealisticControlLayer(profile, config)
    layer.process(_all_joint_action(gripper=0.0), now=0.0, simulated_actual=_all_joint_action(gripper=0.0))
    result = layer.process(_all_joint_action(gripper=0.1), now=0.01, simulated_actual=_all_joint_action(gripper=0.0))
    assert result.diagnostics["gripper"].deadband_applied is False


# ---------------------------------------------------------------------------
# tracking error (섹션 7) - noise 주입 없음, 순수 diff
# ---------------------------------------------------------------------------


def test_compute_tracking_error_is_pure_diff_no_injected_noise():
    processed = {"shoulder_pan": 10.0, "wrist_roll": 2.0}
    actual = {"shoulder_pan": 9.5, "wrist_roll": 2.5}
    errors = compute_tracking_error(processed, actual)
    assert errors == {"shoulder_pan": pytest.approx(0.5), "wrist_roll": pytest.approx(-0.5)}


def test_compute_tracking_error_ignores_joints_missing_in_either_dict():
    processed = {"shoulder_pan": 10.0, "gripper": 50.0}
    actual = {"shoulder_pan": 9.0}
    errors = compute_tracking_error(processed, actual)
    assert errors == {"shoulder_pan": pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# 결정론 (같은 입력 시퀀스 -> 같은 출력 시퀀스)
# ---------------------------------------------------------------------------


def test_process_is_deterministic_given_same_inputs(profile):
    config = RealisticControlConfig(latency_ms_override=20.0)
    trajectory = [(_all_joint_action(shoulder_pan=float(i), wrist_roll=float(i) * 0.1), i * 0.01) for i in range(30)]

    def run() -> list[dict[str, float]]:
        layer = RealisticControlLayer(profile, config)
        outputs = []
        for action, t in trajectory:
            outputs.append(layer.process(action, now=t).processed_action)
        return outputs

    assert run() == run()


# ---------------------------------------------------------------------------
# 진단 리포트 기록
# ---------------------------------------------------------------------------


def test_recorder_writes_json_and_csv(tmp_path, profile):
    layer = RealisticControlLayer(profile, config_all_disabled())
    recorder = RealisticControlRecorder()
    for step in range(3):
        result = layer.process(_all_joint_action(shoulder_pan=float(step)), now=step * 0.01)
        recorder.record(step, result.diagnostics)

    json_path = tmp_path / "diag.json"
    csv_path = tmp_path / "diag.csv"
    recorder.write_json(json_path)
    recorder.write_csv(csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 3 * 6  # 6개 관절 x 3 step
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].split(",") == list(CSV_FIELDNAMES)
