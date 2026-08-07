"""Instrumented Teleop Diagnostic 터미널 대시보드 - 순수 문자열 생성 + 화면 갱신만 담당한다.

``hardware/safety/shadow_teleop_console.py``와 동일한 원칙 - 하드웨어 접근 없음, tty
판정으로 in-place 갱신 여부를 결정한다.

passive 계측 모드로 개정: 더 이상 "이 cycle이 차단됐는지"를 표시하지 않는다(그런 개념 자체가
없어졌다 - command는 절대 차단되지 않는다). 대신 이번 cycle에서 발생한 warning 종류와 지금까지
누적된 총 warning 횟수를 표시한다.
"""

from __future__ import annotations

import sys
from typing import TextIO

DASHBOARD_TITLE = "Instrumented SO-101 Teleop (passive monitoring)"
_SEPARATOR = "-" * 60


def render_dashboard_lines(sample, *, loop_hz: float, total_warning_count: int) -> list[str]:
    """대시보드 줄 목록을 만든다 (하드웨어 접근 없음, command에 영향 없음)."""

    torque = sample.follower_torque_enable
    torque_text = "ON" if torque == 1 else ("OFF" if torque == 0 else "?")

    def _fmt(value, fmt: str = "d") -> str:
        return "?" if value is None else format(value, fmt)

    warning_text = ", ".join(sample.warning_types) if sample.warning_types else "-"

    lines = [
        DASHBOARD_TITLE,
        _SEPARATOR,
        f"Leader wrist      : {sample.leader_wrist_roll_deg:+7.2f}°   "
        f"Δ {sample.leader_wrist_roll_delta_from_start_deg:+.2f}°",
        f"Command wrist     : {sample.command_wrist_roll_deg:+7.2f}°",
        f"Follower Goal     : {_fmt(sample.follower_goal_deg, '+7.2f'):>7}°   raw {_fmt(sample.follower_goal_raw)}",
        f"Follower Present  : {_fmt(sample.follower_present_deg, '+7.2f'):>7}°   raw {_fmt(sample.follower_present_raw)}",
        "",
        f"Goal-Present      : {_fmt(sample.goal_present_error_raw, '+d')} ticks / "
        f"{_fmt(sample.goal_present_error_deg, '+.3f')}°",
        f"Follower Δ        : {_fmt(sample.follower_present_delta_from_start_deg, '+.3f')}°",
        "",
        f"Acceleration      : {_fmt(sample.follower_acceleration)}",
        f"Accel Multiplier  : {_fmt(sample.follower_acceleration_multiplier)}",
        f"Torque            : {torque_text}",
        f"Moving            : {_fmt(sample.follower_moving)}",
        f"Status            : {_fmt(sample.follower_status)}",
        "",
        f"Loop rate         : {loop_hz:5.1f} Hz",
        f"Warnings (cycle)  : {warning_text}",
        f"Warnings (total)  : {total_warning_count}   [기록만 함 - command에는 영향 없음]",
        _SEPARATOR,
    ]
    return lines


class TerminalDashboard:
    """대시보드 줄 목록을 터미널에 제자리 갱신(in-place redraw)으로 출력한다.

    ``hardware/safety/shadow_teleop_console.py``의 ``TerminalDashboard``와 동일한 구현이다 -
    이 두 진단 도구가 완전히 분리된 패키지(``hardware/safety`` vs ``hardware/diagnostics``)로
    유지되어야 한다는 상위 지침에 따라 의도적으로 별도 클래스로 둔다(공유 모듈로 합치면
    두 진단이 암묵적으로 결합된다).
    """

    def __init__(self, *, stream: TextIO = sys.stdout, use_inplace_redraw: bool | None = None) -> None:
        self._stream = stream
        self._use_inplace = use_inplace_redraw if use_inplace_redraw is not None else stream.isatty()
        self._printed_lines = 0

    def update(self, lines: list[str]) -> None:
        if self._use_inplace and self._printed_lines:
            self._stream.write(f"\x1b[{self._printed_lines}A")
        for line in lines:
            if self._use_inplace:
                self._stream.write("\x1b[2K")
            self._stream.write(line + "\n")
        self._printed_lines = len(lines)
        self._stream.flush()
