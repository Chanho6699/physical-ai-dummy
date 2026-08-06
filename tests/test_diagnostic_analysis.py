"""simulation/mujoco/diagnostic_analysis.py 단위 테스트.

실제 MuJoCo/HTTP 없이, DiagnosticAnalyzer.update()에 합성 시계열을 직접 주입해
각 이상 패턴(지속 차이/포화/부호 불일치/offset/range 초과)이 올바른 시점에만
(edge-triggered) 감지되는지 확인한다.
"""

from __future__ import annotations

from simulation.mujoco.diagnostic_analysis import DiagnosticAnalyzer, DiagnosticConfig, summarize_events


def _codes(events) -> list[str]:
    return [e.code for e in events]


# ---------------------------------------------------------------------------
# persistent_difference
# ---------------------------------------------------------------------------


def test_persistent_difference_fires_once_after_duration_threshold():
    config = DiagnosticConfig(persistent_difference_deg=5.0, persistent_duration_sec=1.0)
    analyzer = DiagnosticAnalyzer(config)

    _, events0 = analyzer.update("wrist_flex", 0.0, leader_deg=10.0, follower_deg=0.0)  # diff=10
    assert "persistent_difference" not in _codes(events0)

    _, events1 = analyzer.update("wrist_flex", 0.5, leader_deg=10.0, follower_deg=0.0)
    assert "persistent_difference" not in _codes(events1)  # 아직 duration 미달

    _, events2 = analyzer.update("wrist_flex", 1.1, leader_deg=10.0, follower_deg=0.0)
    assert "persistent_difference" in _codes(events2)

    _, events3 = analyzer.update("wrist_flex", 1.5, leader_deg=10.0, follower_deg=0.0)
    assert "persistent_difference" not in _codes(events3)  # 같은 run에서는 한 번만


def test_persistent_difference_resets_when_sign_flips():
    config = DiagnosticConfig(persistent_difference_deg=5.0, persistent_duration_sec=1.0)
    analyzer = DiagnosticAnalyzer(config)
    analyzer.update("wrist_flex", 0.0, leader_deg=10.0, follower_deg=0.0)
    analyzer.update("wrist_flex", 1.1, leader_deg=10.0, follower_deg=0.0)  # 이벤트 발생, active=True

    _, events = analyzer.update("wrist_flex", 1.2, leader_deg=0.0, follower_deg=10.0)  # 부호 반전 (diff=-10)
    assert "persistent_difference" not in _codes(events)

    _, events2 = analyzer.update("wrist_flex", 2.3, leader_deg=0.0, follower_deg=10.0)  # 새 방향으로 1.1초 지속
    assert "persistent_difference" in _codes(events2)


def test_persistent_difference_clears_when_back_under_threshold():
    config = DiagnosticConfig(persistent_difference_deg=5.0, persistent_duration_sec=1.0)
    analyzer = DiagnosticAnalyzer(config)
    analyzer.update("wrist_flex", 0.0, leader_deg=10.0, follower_deg=0.0)
    analyzer.update("wrist_flex", 1.1, leader_deg=10.0, follower_deg=0.0)  # 이벤트 1회

    analyzer.update("wrist_flex", 1.2, leader_deg=1.0, follower_deg=0.0)  # diff=1, 임계값 아래로 회복
    analyzer.update("wrist_flex", 2.3, leader_deg=10.0, follower_deg=0.0)  # 새 run 시작 (아직 미발생)
    _, events = analyzer.update("wrist_flex", 3.4, leader_deg=10.0, follower_deg=0.0)  # 1.1초 더 지속
    assert "persistent_difference" in _codes(events)  # 재무장 후 재발생


# ---------------------------------------------------------------------------
# follower_saturation_suspected
# ---------------------------------------------------------------------------


def test_follower_saturation_detected_over_window_not_single_sample():
    config = DiagnosticConfig(
        follower_stationary_delta_deg=0.3, leader_motion_delta_deg=1.0, saturation_duration_sec=1.0
    )
    analyzer = DiagnosticAnalyzer(config)

    # 리더는 꾸준히 증가, 팔로워는 거의 그대로 (포화 패턴)
    samples = [(0.0, 0.0, 0.0), (0.3, 0.3, 0.05), (0.6, 0.6, 0.1), (1.0, 1.0, 0.15)]
    fired = []
    for t, leader, follower in samples:
        _, events = analyzer.update("wrist_flex", t, leader_deg=leader, follower_deg=follower)
        fired.extend(_codes(events))

    assert "follower_saturation_suspected" in fired


def test_follower_saturation_not_reported_before_window_filled():
    config = DiagnosticConfig(
        follower_stationary_delta_deg=0.3, leader_motion_delta_deg=1.0, saturation_duration_sec=1.0
    )
    analyzer = DiagnosticAnalyzer(config)
    _, events = analyzer.update("wrist_flex", 0.0, leader_deg=5.0, follower_deg=0.0)
    assert "follower_saturation_suspected" not in _codes(events)


def test_follower_saturation_not_reported_when_follower_also_moves():
    config = DiagnosticConfig(
        follower_stationary_delta_deg=0.3, leader_motion_delta_deg=1.0, saturation_duration_sec=1.0
    )
    analyzer = DiagnosticAnalyzer(config)
    samples = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)]  # 팔로워가 리더를 잘 따라감
    fired = []
    for t, leader, follower in samples:
        _, events = analyzer.update("wrist_flex", t, leader_deg=leader, follower_deg=follower)
        fired.extend(_codes(events))
    assert "follower_saturation_suspected" not in fired


# ---------------------------------------------------------------------------
# sign_mismatch_suspected
# ---------------------------------------------------------------------------


def test_sign_mismatch_detected_when_directions_repeatedly_oppose():
    config = DiagnosticConfig(sign_mismatch_window_size=5, sign_mismatch_min_count=3, sign_mismatch_min_delta_deg=0.3)
    analyzer = DiagnosticAnalyzer(config)

    leader_values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    follower_values = [0.0, -1.0, 0.0, -1.0, 0.0, -1.0]  # 매 구간 리더와 반대 방향
    fired = []
    for i, (leader, follower) in enumerate(zip(leader_values, follower_values)):
        _, events = analyzer.update("wrist_flex", float(i), leader_deg=leader, follower_deg=follower)
        fired.extend(_codes(events))

    assert "sign_mismatch_suspected" in fired


def test_sign_mismatch_not_reported_when_directions_agree():
    config = DiagnosticConfig(sign_mismatch_window_size=5, sign_mismatch_min_count=3, sign_mismatch_min_delta_deg=0.3)
    analyzer = DiagnosticAnalyzer(config)
    leader_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    follower_values = [0.0, 0.9, 1.8, 2.7, 3.6, 4.5]  # 같은 방향으로 함께 움직임
    fired = []
    for i, (leader, follower) in enumerate(zip(leader_values, follower_values)):
        _, events = analyzer.update("wrist_flex", float(i), leader_deg=leader, follower_deg=follower)
        fired.extend(_codes(events))
    assert "sign_mismatch_suspected" not in fired


def test_sign_mismatch_ignores_small_noise_deltas():
    config = DiagnosticConfig(sign_mismatch_window_size=5, sign_mismatch_min_count=3, sign_mismatch_min_delta_deg=0.5)
    analyzer = DiagnosticAnalyzer(config)
    # 아주 작은 반대방향 흔들림은 노이즈로 취급되어야 한다 (delta < min_delta_deg)
    leader_values = [0.0, 0.05, 0.0, 0.05, 0.0, 0.05]
    follower_values = [0.0, -0.05, 0.0, -0.05, 0.0, -0.05]
    fired = []
    for i, (leader, follower) in enumerate(zip(leader_values, follower_values)):
        _, events = analyzer.update("wrist_flex", float(i), leader_deg=leader, follower_deg=follower)
        fired.extend(_codes(events))
    assert "sign_mismatch_suspected" not in fired


# ---------------------------------------------------------------------------
# offset_suspected
# ---------------------------------------------------------------------------


def test_offset_suspected_when_difference_stable_across_varied_poses():
    config = DiagnosticConfig(
        offset_window_sec=2.0, offset_pose_variation_deg=1.0, offset_stability_deg=0.3, offset_min_samples=4
    )
    analyzer = DiagnosticAnalyzer(config)

    leader_values = [0.0, 0.5, 1.0, 1.5]
    times = [0.0, 0.5, 1.0, 1.5]
    fired = []
    for t, leader in zip(times, leader_values):
        follower = leader - 2.0  # 항상 정확히 2deg 차이 (고정 offset)
        _, events = analyzer.update("wrist_flex", t, leader_deg=leader, follower_deg=follower)
        fired.extend(_codes(events))

    assert "offset_suspected" in fired
    assert fired.count("offset_suspected") == 1  # 세션당 한 번만 보고


def test_offset_suspected_not_reported_when_leader_barely_moves():
    config = DiagnosticConfig(
        offset_window_sec=2.0, offset_pose_variation_deg=1.0, offset_stability_deg=0.3, offset_min_samples=4
    )
    analyzer = DiagnosticAnalyzer(config)
    times = [0.0, 0.5, 1.0, 1.5]
    fired = []
    for t in times:
        _, events = analyzer.update("wrist_flex", t, leader_deg=0.01, follower_deg=-1.99)  # 리더가 거의 정지
        fired.extend(_codes(events))
    assert "offset_suspected" not in fired


def test_offset_suspected_not_reported_when_difference_unstable():
    config = DiagnosticConfig(
        offset_window_sec=2.0, offset_pose_variation_deg=1.0, offset_stability_deg=0.3, offset_min_samples=4
    )
    analyzer = DiagnosticAnalyzer(config)
    leader_values = [0.0, 0.5, 1.0, 1.5]
    follower_values = [0.0, 2.0, 0.0, 2.0]  # 차이가 요동침 (offset이 아님)
    fired = []
    for t, leader, follower in zip([0.0, 0.5, 1.0, 1.5], leader_values, follower_values):
        _, events = analyzer.update("wrist_flex", t, leader_deg=leader, follower_deg=follower)
        fired.extend(_codes(events))
    assert "offset_suspected" not in fired


# ---------------------------------------------------------------------------
# range exceed (leader만 MuJoCo range를 벗어나는 경우 - wrist_flex 사례)
# ---------------------------------------------------------------------------


def test_leader_out_of_mujoco_range_when_only_leader_exceeds():
    analyzer = DiagnosticAnalyzer()
    sample, events = analyzer.update(
        "wrist_flex", 0.0, leader_deg=96.82, follower_deg=90.0, joint_range_deg=(-95.0, 95.0)
    )
    assert sample.leader_out_of_range is True
    assert sample.follower_out_of_range is False
    codes = _codes(events)
    assert "leader_out_of_mujoco_range" in codes
    evt = next(e for e in events if e.code == "leader_out_of_mujoco_range")
    assert evt.details["sub_event"] == "follower_inside_mujoco_range"


def test_range_exceed_not_reported_when_both_out_of_range():
    analyzer = DiagnosticAnalyzer()
    _, events = analyzer.update(
        "wrist_flex", 0.0, leader_deg=100.0, follower_deg=98.0, joint_range_deg=(-95.0, 95.0)
    )
    assert "leader_out_of_mujoco_range" not in _codes(events)


def test_range_exceed_fires_every_sample_while_condition_holds():
    """CSV가 프레임마다 event_code를 남길 수 있도록, 이 이벤트는 edge-trigger가 아니다."""
    analyzer = DiagnosticAnalyzer()
    _, events1 = analyzer.update(
        "wrist_flex", 0.0, leader_deg=96.0, follower_deg=90.0, joint_range_deg=(-95.0, 95.0)
    )
    _, events2 = analyzer.update(
        "wrist_flex", 0.05, leader_deg=96.5, follower_deg=90.5, joint_range_deg=(-95.0, 95.0)
    )
    assert "leader_out_of_mujoco_range" in _codes(events1)
    assert "leader_out_of_mujoco_range" in _codes(events2)


def test_no_range_check_when_joint_range_not_provided():
    analyzer = DiagnosticAnalyzer()
    sample, events = analyzer.update("wrist_flex", 0.0, leader_deg=999.0, follower_deg=0.0, joint_range_deg=None)
    assert sample.leader_out_of_range is False
    assert "leader_out_of_mujoco_range" not in _codes(events)


# ---------------------------------------------------------------------------
# 관절별 독립성 + summarize_events
# ---------------------------------------------------------------------------


def test_joints_are_tracked_independently():
    config = DiagnosticConfig(persistent_difference_deg=5.0, persistent_duration_sec=1.0)
    analyzer = DiagnosticAnalyzer(config)
    analyzer.update("wrist_flex", 0.0, leader_deg=10.0, follower_deg=0.0)
    _, events = analyzer.update("shoulder_pan", 0.0, leader_deg=0.0, follower_deg=0.0)
    assert events == []  # 다른 관절의 히스토리에 영향받지 않음


def test_summarize_events_counts_by_code():
    config = DiagnosticConfig(persistent_difference_deg=5.0, persistent_duration_sec=1.0)
    analyzer = DiagnosticAnalyzer(config)
    all_events = []
    _, e0 = analyzer.update("wrist_flex", 0.0, leader_deg=10.0, follower_deg=0.0)
    all_events.extend(e0)
    _, e1 = analyzer.update("wrist_flex", 1.1, leader_deg=10.0, follower_deg=0.0)
    all_events.extend(e1)
    counts = summarize_events(all_events)
    assert counts.get("persistent_difference") == 1
