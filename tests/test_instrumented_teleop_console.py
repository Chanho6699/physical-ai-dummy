"""hardware/diagnostics/instrumented_teleop_console.py 단위 테스트 (passive monitoring 대시보드).

터미널/하드웨어 없이 순수 문자열 생성 + io.StringIO 갱신 로직만 검증한다.
"""

from __future__ import annotations

import io

from hardware.diagnostics.instrumented_teleop import TeleopCycleSample
from hardware.diagnostics.instrumented_teleop_console import (
    DASHBOARD_TITLE,
    TerminalDashboard,
    render_dashboard_lines,
)


def _sample(**overrides) -> TeleopCycleSample:
    base = dict(
        loop_index=3,
        timestamp_iso="2026-08-07T00:00:00+00:00",
        elapsed_sec=1.5,
        loop_hz=59.4,
        leader_wrist_roll_deg=-1.80,
        leader_wrist_roll_delta_from_start_deg=0.35,
        command_wrist_roll_deg=-1.80,
        follower_goal_raw=2000,
        follower_goal_deg=-1.80,
        follower_present_raw=1998,
        follower_present_deg=-1.98,
        goal_present_error_raw=2,
        goal_present_error_deg=0.176,
        follower_present_delta_from_prev_raw=0,
        follower_present_delta_from_prev_deg=0.0,
        follower_present_delta_from_start_deg=0.176,
        follower_torque_enable=1,
        follower_acceleration=254,
        follower_acceleration_multiplier=1,
        follower_moving=1,
        follower_status=0,
        send_action_executed=True,
        leader_command_all_joints={},
        follower_sent_all_joints={},
        follower_observation_all_joints={},
        register_read_error=None,
        warning_types=(),
    )
    base.update(overrides)
    return TeleopCycleSample(**base)


def test_render_dashboard_lines_includes_title_and_key_values():
    lines = render_dashboard_lines(_sample(), loop_hz=59.4, total_warning_count=0)
    text = "\n".join(lines)
    assert lines[0] == DASHBOARD_TITLE
    assert "-1.80" in text
    assert "raw 2000" in text
    assert "raw 1998" in text
    assert "59.4 Hz" in text
    assert "ON" in text  # torque_enable=1
    assert "Warnings (total)  : 0" in text
    assert "Warnings (cycle)  : -" in text


def test_render_dashboard_lines_shows_warning_types_for_cycle():
    lines = render_dashboard_lines(
        _sample(warning_types=("WARNING_LARGE_COMMAND_DELTA", "WARNING_POSITION_JUMP")),
        loop_hz=59.4,
        total_warning_count=5,
    )
    text = "\n".join(lines)
    assert "WARNING_LARGE_COMMAND_DELTA" in text
    assert "WARNING_POSITION_JUMP" in text
    assert "Warnings (total)  : 5" in text


def test_render_dashboard_lines_never_mentions_blocking():
    # passive 모드: "BLOCKED"/"safety limit" 같은 개입 관련 문구가 대시보드에 없어야 한다.
    lines = render_dashboard_lines(_sample(), loop_hz=59.4, total_warning_count=0)
    text = "\n".join(lines)
    assert "BLOCKED" not in text
    assert "Safety limit" not in text


def test_render_dashboard_lines_torque_off_and_unknown():
    lines_off = render_dashboard_lines(_sample(follower_torque_enable=0), loop_hz=10.0, total_warning_count=0)
    assert any("OFF" in line for line in lines_off)

    lines_unknown = render_dashboard_lines(
        _sample(follower_torque_enable=None, follower_moving=None, follower_status=None),
        loop_hz=10.0,
        total_warning_count=0,
    )
    text = "\n".join(lines_unknown)
    assert "?" in text


def test_terminal_dashboard_non_tty_writes_lines_without_ansi_codes():
    stream = io.StringIO()
    dashboard = TerminalDashboard(stream=stream, use_inplace_redraw=False)
    dashboard.update(["line1", "line2"])
    output = stream.getvalue()
    assert output == "line1\nline2\n"
    assert "\x1b" not in output


def test_terminal_dashboard_tty_redraws_in_place_on_second_update():
    stream = io.StringIO()
    dashboard = TerminalDashboard(stream=stream, use_inplace_redraw=True)
    dashboard.update(["a", "b"])
    first_output = stream.getvalue()
    assert "\x1b[2A" not in first_output

    dashboard.update(["c", "d", "e"])
    second_output = stream.getvalue()[len(first_output) :]
    assert second_output.startswith("\x1b[2A")
    assert "\x1b[2K" in second_output
