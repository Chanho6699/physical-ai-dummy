"""Shadow Teleop Diagnostic 터미널 대시보드 - 순수 문자열 생성 + 화면 갱신만 담당한다.

``hardware/state_server/console_status.py``와 동일한 원칙: 이 모듈은 시리얼 read/쓰기와
무관하며 하드웨어에 전혀 접근하지 않는다. ``render_dashboard_lines``는 순수 함수라
``ShadowSample`` 하나만 있으면 실제 하드웨어 없이도 테스트할 수 있다. 화면 갱신
(``TerminalDashboard``)은 tty일 때만 커서를 위로 이동해 제자리에서 다시 그리고, tty가
아니면(파일로 리다이렉트 등) 매 프레임을 그냥 이어 붙여 출력한다.
"""

from __future__ import annotations

import sys
from typing import TextIO

DASHBOARD_TITLE = "Shadow Teleop Diagnostic"
_SEPARATOR = "-" * 64


def render_dashboard_lines(sample, *, read_rate_hz: float, write_count: int, sample_count: int) -> list[str]:
    """섹션 6 예시와 동일한 형식의 대시보드 줄 목록을 만든다 (하드웨어 접근 없음)."""

    torque = sample.follower_torque_enable
    torque_text = "ON" if torque == 1 else ("OFF" if torque == 0 else "?")

    def _fmt(value, fmt: str = "d") -> str:
        return "?" if value is None else format(value, fmt)

    return [
        DASHBOARD_TITLE,
        _SEPARATOR,
        f"Leader wrist      : {sample.leader_wrist_roll_deg:+7.2f} deg   "
        f"Δ {sample.leader_delta_from_start_deg:+.2f} deg",
        f"Follower goal     : {sample.follower_goal_deg:+7.2f} deg   raw {sample.follower_goal_raw}",
        f"Follower present  : {sample.follower_present_deg:+7.2f} deg   raw {sample.follower_present_raw}   "
        f"Δ {sample.follower_present_delta_from_start_deg:+.2f} deg",
        f"Goal-Present      : {sample.goal_present_error_raw:+d} ticks / {sample.goal_present_error_deg:+.2f} deg",
        f"Acceleration      : {_fmt(sample.follower_acceleration)}",
        f"Accel Multiplier  : {_fmt(sample.follower_acceleration_multiplier)}",
        f"Torque            : {torque_text}",
        f"Moving            : {_fmt(sample.follower_moving)}",
        f"Status            : {_fmt(sample.follower_status)}",
        f"Read rate         : {read_rate_hz:5.1f} Hz   Samples: {sample_count}",
        f"Writes            : {write_count}",
        _SEPARATOR,
    ]


class TerminalDashboard:
    """대시보드 줄 목록을 터미널에 제자리 갱신(in-place redraw)으로 출력한다.

    tty가 아니면(파이프/파일로 리다이렉트) in-place 갱신을 쓰지 않고 매번 새로 출력한다 -
    ANSI 커서 이동 코드가 로그 파일을 오염시키지 않게 하기 위함
    (``hardware/state_server/console_status.py``의 ``resolve_use_color``와 동일한 tty 판정
    원칙).
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
