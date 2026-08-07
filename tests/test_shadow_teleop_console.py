"""hardware/safety/shadow_teleop_console.py 단위 테스트.

터미널/하드웨어 없이 순수 문자열 생성 + io.StringIO로 갱신 로직만 검증한다.
"""

from __future__ import annotations

import io

from hardware.safety.shadow_teleop_console import (
    DASHBOARD_TITLE,
    TerminalDashboard,
    render_dashboard_lines,
)
from hardware.safety.shadow_teleop_diagnostic import ShadowSample


def _sample(**overrides) -> ShadowSample:
    base = dict(
        sample_index=3,
        timestamp_iso="2026-08-07T00:00:00+00:00",
        elapsed_sec=1.5,
        leader_wrist_roll_raw=2148,
        leader_wrist_roll_deg=-1.92,
        leader_delta_from_start_deg=0.23,
        follower_goal_raw=2021,
        follower_goal_deg=-2.25,
        follower_present_raw=2023,
        follower_present_deg=-2.15,
        follower_present_delta_from_start_deg=0.0,
        goal_present_error_raw=-2,
        goal_present_error_deg=-0.10,
        leader_vs_follower_present_deg=0.23,
        follower_acceleration=0,
        follower_acceleration_multiplier=1,
        follower_torque_enable=1,
        follower_moving=0,
        follower_status=0,
    )
    base.update(overrides)
    return ShadowSample(**base)


def test_render_dashboard_lines_includes_title_and_key_values():
    lines = render_dashboard_lines(_sample(), read_rate_hz=27.4, write_count=0, sample_count=4)
    text = "\n".join(lines)
    assert lines[0] == DASHBOARD_TITLE
    assert "-1.92" in text
    assert "raw 2021" in text
    assert "raw 2023" in text
    assert "27.4 Hz" in text
    assert "Writes            : 0" in text
    assert "ON" in text  # torque_enable=1


def test_render_dashboard_lines_torque_off_and_unknown():
    lines_off = render_dashboard_lines(_sample(follower_torque_enable=0), read_rate_hz=10.0, write_count=0, sample_count=1)
    assert any("OFF" in line for line in lines_off)

    lines_unknown = render_dashboard_lines(
        _sample(follower_torque_enable=None, follower_moving=None, follower_status=None),
        read_rate_hz=10.0,
        write_count=0,
        sample_count=1,
    )
    text = "\n".join(lines_unknown)
    assert "?" in text


def test_render_dashboard_lines_never_shows_nonzero_write_count_by_default():
    # write_count는 항상 호출부가 0을 넘긴다는 계약을 명시적으로 확인한다.
    lines = render_dashboard_lines(_sample(), read_rate_hz=1.0, write_count=0, sample_count=1)
    assert any(line.strip() == "Writes            : 0" for line in lines)


def test_terminal_dashboard_non_tty_writes_lines_without_ansi_codes():
    stream = io.StringIO()
    dashboard = TerminalDashboard(stream=stream, use_inplace_redraw=False)
    dashboard.update(["line1", "line2"])
    output = stream.getvalue()
    assert output == "line1\nline2\n"
    assert "\x1b" not in output


def test_terminal_dashboard_tty_uses_cursor_up_and_clear_on_second_update():
    stream = io.StringIO()
    dashboard = TerminalDashboard(stream=stream, use_inplace_redraw=True)
    dashboard.update(["a", "b"])
    first_output = stream.getvalue()
    assert "\x1b[2A" not in first_output  # 첫 프레임은 이전 줄이 없으니 커서 이동은 없다
    assert "\x1b[2K" in first_output  # 각 줄 clear 코드는 첫 프레임부터 붙는다

    dashboard.update(["c", "d", "e"])
    second_output = stream.getvalue()[len(first_output):]
    assert second_output.startswith("\x1b[2A")  # 이전 프레임 줄 수(2)만큼 커서 위로 이동
    assert "\x1b[2K" in second_output  # 각 줄 clear


def test_terminal_dashboard_defaults_to_stream_isatty():
    class _FakeStream(io.StringIO):
        def isatty(self):
            return False

    stream = _FakeStream()
    dashboard = TerminalDashboard(stream=stream)
    dashboard.update(["x"])
    assert "\x1b" not in stream.getvalue()
