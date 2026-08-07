"""simulation/mujoco/safety_event_tracker.py 단위 테스트.

전부 명시적 ``now_wall``을 넘겨 결정적으로(real sleep 없이) 검증한다 - sticky window/
clear_after_samples 병합 로직은 시간 경과가 핵심이라 실제 sleep에 의존하면 느리고
불안정해지기 때문이다.
"""

from __future__ import annotations

import json

import pytest

from simulation.mujoco.safety_checks import SafetyEvent
from simulation.mujoco.safety_event_tracker import (
    CONNECTION_WIDE_JOINT,
    REASON_CODES,
    SafetyEventTracker,
    SafetyEventTrackerConfig,
    SafetyIssue,
    classify_frame_event_reason,
    make_event_session_id,
    pick_most_severe_event,
    resolve_safety_event_paths,
    write_safety_events_csv,
    write_safety_events_json,
)

T0 = 1_700_000_000.0  # 임의의 고정 기준 시각 (time.time() 스케일)


def _issue(joint: str, severity: str | None, reason_code: str | None, **kwargs) -> SafetyIssue:
    return SafetyIssue(joint=joint, severity=severity, reason_code=reason_code, **kwargs)


def _pass(joint: str) -> SafetyIssue:
    return _issue(joint, None, None)


# ---------------------------------------------------------------------------
# 원인 코드 분류 - 기존 SafetyEvent만 보고 판단, 확정 안 되면 UNKNOWN_SAFETY_REASON
# ---------------------------------------------------------------------------


def test_classify_joint_limit_high():
    evt = SafetyEvent("BLOCKED", "joint_limit", "초과", joint="wrist_flex", value=1.7, limit=(-1.6, 1.6))
    assert classify_frame_event_reason(evt) == "JOINT_RANGE_HIGH"


def test_classify_joint_limit_low():
    evt = SafetyEvent("BLOCKED", "joint_limit", "미만", joint="wrist_flex", value=-1.7, limit=(-1.6, 1.6))
    assert classify_frame_event_reason(evt) == "JOINT_RANGE_LOW"


def test_classify_actuator_limit_high():
    evt = SafetyEvent("BLOCKED", "actuator_limit", "초과", joint="gripper", value=2.0, limit=(-0.2, 1.8))
    assert classify_frame_event_reason(evt) == "JOINT_RANGE_HIGH"


def test_classify_max_delta_is_frame_delta_high():
    evt = SafetyEvent("WARN", "max_delta", "변화량 초과", joint="shoulder_pan", value=0.5, limit=(0.0, 0.3))
    assert classify_frame_event_reason(evt) == "FRAME_DELTA_HIGH"


def test_classify_simulation_nan_is_invalid_value():
    evt = SafetyEvent("BLOCKED", "simulation_nan", "NaN 발생")
    assert classify_frame_event_reason(evt) == "INVALID_VALUE"


@pytest.mark.parametrize("code", ["velocity_limit", "self_collision", "table_collision", "contact_spike", "simulation_divergence", "totally_new_code"])
def test_classify_unmappable_codes_fall_back_to_unknown(code):
    """목록에 없는 코드로 새 원인을 추측해 만들지 않는다 - 반드시 UNKNOWN_SAFETY_REASON."""
    evt = SafetyEvent("WARN", code, "메시지")
    assert classify_frame_event_reason(evt) == "UNKNOWN_SAFETY_REASON"


def test_classify_joint_limit_without_value_or_limit_is_unknown():
    """value/limit이 없으면 LOW/HIGH를 확정할 수 없다 - 추측하지 않는다."""
    evt = SafetyEvent("BLOCKED", "joint_limit", "초과", joint="wrist_flex")
    assert classify_frame_event_reason(evt) == "UNKNOWN_SAFETY_REASON"


def test_all_classify_results_are_within_fixed_reason_codes():
    codes = ["joint_limit", "actuator_limit", "max_delta", "simulation_nan", "velocity_limit", "unknown"]
    for code in codes:
        evt = SafetyEvent("WARN", code, "x", value=1.0, limit=(0.0, 2.0))
        assert classify_frame_event_reason(evt) in REASON_CODES


def test_pick_most_severe_prefers_blocked_over_warn():
    warn = SafetyEvent("WARN", "max_delta", "x", joint="wrist_flex")
    blocked = SafetyEvent("BLOCKED", "joint_limit", "x", joint="wrist_flex", value=2.0, limit=(-1.0, 1.0))
    assert pick_most_severe_event([warn, blocked]) is blocked
    assert pick_most_severe_event([blocked, warn]) is blocked


def test_pick_most_severe_empty_is_none():
    assert pick_most_severe_event([]) is None


# ---------------------------------------------------------------------------
# 이벤트 추적/병합
# ---------------------------------------------------------------------------


def test_single_sample_blocked_event_appears_immediately_in_recent_events():
    """한 샘플만 BLOCKED여도 sticky 목록에 바로 남아야 한다 (요구사항)."""
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=3, sticky_display_sec=10.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("wrist_flex", "BLOCKED", "JOINT_RANGE_HIGH", requested_target_deg=96.2, joint_max_deg=95.0)])
    tracker.observe(now_wall=T0 + 0.05, remote_sequence=2, issues=[_pass("wrist_flex")])  # 바로 다음 샘플은 정상

    events = tracker.recent_events(now_wall=T0 + 0.1)
    assert len(events) == 1
    assert events[0]["severity"] == "BLOCKED"
    assert events[0]["reason_code"] == "JOINT_RANGE_HIGH"
    assert events[0]["sample_count"] == 1


def test_consecutive_same_joint_reason_merges_into_one_event():
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=3, sticky_display_sec=10.0))
    for i in range(5):
        tracker.observe(
            now_wall=T0 + i * 0.05,
            remote_sequence=10 + i,
            issues=[_issue("shoulder_lift", "WARN", "NEAR_JOINT_LIMIT", margin_deg=2.4)],
        )
    events = tracker.recent_events(now_wall=T0 + 1.0)
    assert len(events) == 1
    e = events[0]
    assert e["sample_count"] == 5
    assert e["first_remote_sequence"] == 10
    assert e["last_remote_sequence"] == 14


def test_severity_escalates_to_blocked_but_never_downgrades_within_one_event():
    """같은 (관절, 원인 코드) 안에서 severity가 WARN -> BLOCKED -> WARN으로 흔들려도, 한 번이라도
    BLOCKED였다면 그 이벤트 전체는 BLOCKED로 남아야 한다 (현재 safety_checks.py 코드들은 각각
    severity가 고정이라 실제로는 거의 안 나오는 경우지만, 추적기 자체의 병합 규칙을 검증한다)."""
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=3, sticky_display_sec=10.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("gripper", "WARN", "FRAME_DELTA_HIGH")])
    tracker.observe(now_wall=T0 + 0.05, remote_sequence=2, issues=[_issue("gripper", "BLOCKED", "FRAME_DELTA_HIGH")])
    tracker.observe(now_wall=T0 + 0.10, remote_sequence=3, issues=[_issue("gripper", "WARN", "FRAME_DELTA_HIGH")])

    events = tracker.recent_events(now_wall=T0 + 1.0)
    assert len(events) == 1
    assert events[0]["severity"] == "BLOCKED"  # 한 번이라도 BLOCKED였으면 그 이벤트는 BLOCKED로 남음
    assert events[0]["sample_count"] == 3


def test_reason_code_change_on_same_joint_is_a_different_event():
    """같은 관절이라도 원인 코드가 바뀌면(예: NEAR_JOINT_LIMIT -> JOINT_RANGE_HIGH) 별개
    이벤트다 - "동일 원인"이 연속될 때만 병합한다."""
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=3, sticky_display_sec=10.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("gripper", "WARN", "NEAR_JOINT_LIMIT", margin_deg=3.0)])
    tracker.observe(now_wall=T0 + 0.05, remote_sequence=2, issues=[_issue("gripper", "BLOCKED", "JOINT_RANGE_HIGH", requested_target_deg=999.0)])

    events = tracker.recent_events(now_wall=T0 + 1.0)
    reason_codes = {e["reason_code"] for e in events}
    assert reason_codes == {"NEAR_JOINT_LIMIT", "JOINT_RANGE_HIGH"}
    assert len(events) == 2


def test_brief_gap_within_clear_after_samples_does_not_split_event():
    """clear_after_samples=3이면 1~2샘플 정상이 끼어도 같은 이벤트로 유지되어야 한다."""
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=3, sticky_display_sec=10.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("wrist_flex", "WARN", "FRAME_DELTA_HIGH")])
    tracker.observe(now_wall=T0 + 0.05, remote_sequence=2, issues=[_pass("wrist_flex")])  # 1샘플만 빠짐
    tracker.observe(now_wall=T0 + 0.10, remote_sequence=3, issues=[_issue("wrist_flex", "WARN", "FRAME_DELTA_HIGH")])

    events = tracker.recent_events(now_wall=T0 + 1.0)
    assert len(events) == 1
    assert events[0]["sample_count"] == 2  # 정상이었던 샘플은 세지 않음, 문제였던 2개만


def test_clear_after_samples_reached_closes_event():
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=2, sticky_display_sec=10.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("wrist_flex", "WARN", "FRAME_DELTA_HIGH")])
    tracker.observe(now_wall=T0 + 0.05, remote_sequence=2, issues=[_pass("wrist_flex")])
    tracker.observe(now_wall=T0 + 0.10, remote_sequence=3, issues=[_pass("wrist_flex")])  # 2번째 연속 정상 -> 종료

    events = tracker.recent_events(now_wall=T0 + 0.2)
    assert len(events) == 1
    assert events[0]["ended_at"] == pytest.approx(T0, abs=1e-6)  # 마지막으로 "관측된" 시각에 종료


def test_recovery_then_recurrence_creates_new_event():
    """정상 상태가 clear_after_samples 이상 유지된 뒤 다시 발생하면 새 이벤트여야 한다."""
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=2, sticky_display_sec=100.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("wrist_flex", "WARN", "FRAME_DELTA_HIGH")])
    tracker.observe(now_wall=T0 + 0.05, remote_sequence=2, issues=[_pass("wrist_flex")])
    tracker.observe(now_wall=T0 + 0.10, remote_sequence=3, issues=[_pass("wrist_flex")])  # 종료됨
    tracker.observe(now_wall=T0 + 0.15, remote_sequence=4, issues=[_pass("wrist_flex")])
    tracker.observe(now_wall=T0 + 5.0, remote_sequence=5, issues=[_issue("wrist_flex", "WARN", "FRAME_DELTA_HIGH")])  # 재발

    events = tracker.recent_events(now_wall=T0 + 5.1)
    assert len(events) == 2
    event_ids = {e["event_id"] for e in events}
    assert len(event_ids) == 2  # 서로 다른 event_id


def test_different_reason_codes_on_same_joint_tracked_independently():
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=2, sticky_display_sec=100.0))
    tracker.observe(
        now_wall=T0,
        remote_sequence=1,
        issues=[
            _issue("wrist_flex", "WARN", "FRAME_DELTA_HIGH"),
        ],
    )
    events = tracker.recent_events(now_wall=T0 + 0.1)
    assert len(events) == 1
    assert events[0]["reason_code"] == "FRAME_DELTA_HIGH"


def test_stale_event_recorded_with_connection_wide_joint():
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=2, sticky_display_sec=100.0))
    tracker.observe(
        now_wall=T0,
        remote_sequence=1,
        issues=[_issue(CONNECTION_WIDE_JOINT, "WARN", "REMOTE_STALE", remote_age_ms=812.0, stale=True)],
    )
    events = tracker.recent_events(now_wall=T0 + 0.1)
    assert len(events) == 1
    assert events[0]["reason_code"] == "REMOTE_STALE"
    assert events[0]["joint"] == CONNECTION_WIDE_JOINT
    assert events[0]["stale"] is True
    assert events[0]["remote_age_ms"] == pytest.approx(812.0)


def test_current_safety_reflects_only_latest_sample():
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=3, sticky_display_sec=10.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("wrist_flex", "BLOCKED", "JOINT_RANGE_HIGH")])
    assert tracker.current_safety()["level"] == "BLOCKED"

    tracker.observe(now_wall=T0 + 0.05, remote_sequence=2, issues=[_pass("wrist_flex")])
    current = tracker.current_safety()
    assert current["level"] == "PASS"
    assert current["joints"] == {}
    # 하지만 recent_events(sticky)에는 여전히 남아 있어야 한다 (아직 clear_after_samples 안 채움)
    assert len(tracker.recent_events(now_wall=T0 + 0.1)) == 1


def test_event_counts_within_sticky_window():
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=1, sticky_display_sec=5.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("wrist_flex", "BLOCKED", "JOINT_RANGE_HIGH")])
    tracker.observe(now_wall=T0 + 0.05, remote_sequence=2, issues=[_issue("gripper", "WARN", "NEAR_JOINT_LIMIT")])
    tracker.observe(now_wall=T0 + 0.10, remote_sequence=3, issues=[_pass("wrist_flex"), _pass("gripper")])  # 둘 다 종료

    counts = tracker.event_counts(now_wall=T0 + 0.2)
    assert counts == {"WARN": 1, "BLOCKED": 1}


def test_sticky_window_expires_old_events():
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=1, sticky_display_sec=10.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("wrist_flex", "BLOCKED", "JOINT_RANGE_HIGH")])
    tracker.observe(now_wall=T0 + 0.05, remote_sequence=2, issues=[_pass("wrist_flex")])  # 종료 (ended_at ~= T0)

    assert len(tracker.recent_events(now_wall=T0 + 5.0)) == 1  # 아직 10초 안 지남
    assert len(tracker.recent_events(now_wall=T0 + 10.1)) == 0  # 10초 지나서 사라짐


def test_finalize_closes_still_active_events():
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=3, sticky_display_sec=10.0))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("wrist_flex", "BLOCKED", "JOINT_RANGE_HIGH")])
    tracker.finalize(now_wall=T0 + 2.0)

    all_events = tracker.all_events()
    assert len(all_events) == 1
    assert all_events[0].ended_at == pytest.approx(T0 + 2.0)
    # all_events()가 반환하는 기록 자체가 "종료됨"으로 바뀌어야 한다 (active=False).
    assert all_events[0].to_dict()["active"] is False
    # current_safety()는 "마지막으로 관측된 샘플" 스냅샷이므로 finalize 자체로는 안 바뀐다
    # (finalize는 새 관측이 아니라 세션 종료 처리이기 때문) - 여전히 BLOCKED로 남아 있어야 정상.
    assert tracker.current_safety()["level"] == "BLOCKED"


def test_default_config_matches_requested_yaml_defaults():
    cfg = SafetyEventTrackerConfig()
    assert cfg.clear_after_samples == 3
    assert cfg.sticky_display_sec == 10


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        SafetyEventTrackerConfig(clear_after_samples=0)
    with pytest.raises(ValueError):
        SafetyEventTrackerConfig(sticky_display_sec=-1.0)
    with pytest.raises(ValueError):
        SafetyEventTrackerConfig(near_limit_margin_deg=-1.0)


# ---------------------------------------------------------------------------
# JSON/CSV 저장
# ---------------------------------------------------------------------------


def test_write_json_and_csv_reports(tmp_path):
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=1, sticky_display_sec=10.0))
    tracker.observe(
        now_wall=T0,
        remote_sequence=5,
        issues=[
            _issue(
                "wrist_flex",
                "BLOCKED",
                "JOINT_RANGE_HIGH",
                requested_target_deg=96.2,
                applied_target_deg=94.0,
                joint_min_deg=-95.0,
                joint_max_deg=95.0,
                margin_deg=1.0,
                delta_deg=2.2,
                remote_age_ms=48.0,
                stale=False,
            )
        ],
    )
    tracker.finalize(now_wall=T0 + 0.048)

    events = tracker.all_events()
    session_id = make_event_session_id()
    json_path, csv_path = resolve_safety_event_paths(tmp_path, session_id)
    write_safety_events_json(json_path, events)
    write_safety_events_csv(csv_path, events)

    assert json_path.name == f"safety_events_{session_id}.json"
    assert csv_path.name == f"safety_events_{session_id}.csv"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["event_count"] == 1
    assert payload["events"][0]["reason_code"] == "JOINT_RANGE_HIGH"
    assert payload["events"][0]["joint"] == "wrist_flex"
    assert payload["events"][0]["requested_target_deg"] == pytest.approx(96.2)
    assert payload["events"][0]["applied_target_deg"] == pytest.approx(94.0)

    csv_text = csv_path.read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    for field in (
        "event_id",
        "severity",
        "reason_code",
        "joint",
        "started_at",
        "ended_at",
        "duration_ms",
        "sample_count",
        "first_remote_sequence",
        "last_remote_sequence",
        "requested_target_deg",
        "applied_target_deg",
        "joint_min_deg",
        "joint_max_deg",
        "margin_deg",
        "delta_deg",
        "remote_age_ms",
        "stale",
    ):
        assert field in header
    assert "wrist_flex" in csv_text
    assert "JOINT_RANGE_HIGH" in csv_text


def test_report_never_contains_token_like_keys(tmp_path):
    """SafetyEventRecord/리포트 어디에도 인증정보/원격 응답 원문이 담기지 않는다."""
    tracker = SafetyEventTracker(SafetyEventTrackerConfig(clear_after_samples=1))
    tracker.observe(now_wall=T0, remote_sequence=1, issues=[_issue("wrist_flex", "WARN", "FRAME_DELTA_HIGH")])
    tracker.finalize(now_wall=T0 + 0.1)
    events = tracker.all_events()

    json_path, csv_path = resolve_safety_event_paths(tmp_path, make_event_session_id())
    write_safety_events_json(json_path, events)
    write_safety_events_csv(csv_path, events)

    for path in (json_path, csv_path):
        text = path.read_text(encoding="utf-8").lower()
        assert "token" not in text
        assert "authorization" not in text
        assert "bearer" not in text
