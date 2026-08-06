"""SO-101 읽기 전용 상태 서버의 한글 CLI 출력.

simulation/mujoco/console_status.py와 동일한 원칙을 따른다: 이 모듈은 오직
"출력 문자열 생성/인쇄"만 담당하고, 연결/폴링 판정 로직은 state_service.py에 둔다.

표시 규칙:
  - 기본 상태 출력([상태] ...)은 5초에 한 번 이하로 제한한다 (PeriodicStatusPrinter).
  - 오류/경고는 즉시 표시하며 --quiet의 영향을 받지 않는다.
  - --quiet는 배너/PASS 단계 출력/주기 상태 출력만 억제한다.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

_RESET = "\033[0m"
_COLORS = {
    "PASS": "\033[32m",  # green
    "WARN": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "HEADER": "\033[36m",  # cyan
    "DIM": "\033[2m",
}


@dataclass
class ConsoleOptions:
    quiet: bool = False
    verbose: bool = False
    use_color: bool = True


def resolve_use_color(no_color_flag: bool) -> bool:
    if no_color_flag:
        return False
    return sys.stdout.isatty()


def _c(text: str, tag: str, opts: ConsoleOptions) -> str:
    if not opts.use_color:
        return text
    return f"{_COLORS.get(tag, '')}{text}{_RESET}"


def print_startup_banner(
    opts: ConsoleOptions,
    *,
    leader_id: str,
    follower_id: str,
    rate_hz: float,
) -> None:
    if opts.quiet:
        return
    bar = "=" * 68
    print(bar)
    print(_c("[시작] SO-101 읽기 전용 관절 상태 서버", "HEADER", opts))
    print(bar)
    print("[모드] READ ONLY")
    print(f"[리더암] {leader_id}")
    print(f"[팔로워암] {follower_id}")
    print(f"[읽기 주기] {rate_hz:g} Hz")
    print("[쓰기 API] 비활성화")
    print("-" * 68)


def print_step(opts: ConsoleOptions, level: str, message: str) -> None:
    """PASS/WARN/ERROR 단계 메시지. WARN/ERROR는 --quiet에도 항상 표시한다."""

    if opts.quiet and level == "PASS":
        return
    tag = {"PASS": "[통과]", "WARN": "[경고]", "ERROR": "[오류]"}.get(level, f"[{level}]")
    stream = sys.stderr if level == "ERROR" else sys.stdout
    print(f"{_c(tag, level, opts)} {message}", file=stream)


def print_action(message: str) -> None:
    """[조치] 안내 - 오류 직후 무엇을 하는지 즉시 알린다 (quiet 무시)."""

    print(f"[조치] {message}")


def print_server_ready(opts: ConsoleOptions, *, host: str, port: int) -> None:
    if opts.quiet:
        return
    print(f"[서버] http://{host}:{port}")
    print("=" * 68)


def print_shutdown(message: str) -> None:
    print(f"[종료] {message}")


def print_periodic_status(
    opts: ConsoleOptions,
    *,
    samples: int,
    leader_ok: bool,
    follower_ok: bool,
    read_errors: int,
) -> None:
    if opts.quiet:
        return
    leader_txt = "정상" if leader_ok else "오류"
    follower_txt = "정상" if follower_ok else "오류"
    print(
        f"[상태] samples={samples} | leader={leader_txt} | follower={follower_txt} "
        f"| read_errors={read_errors}"
    )


class PeriodicStatusPrinter:
    """[상태] 요약 줄을 최소 간격(기본 5초)마다 한 번만 출력하도록 제한한다."""

    def __init__(self, opts: ConsoleOptions, interval_s: float = 5.0) -> None:
        self._opts = opts
        self._interval_s = interval_s
        self._last_printed_at: float | None = None

    def maybe_print(
        self,
        *,
        samples: int,
        leader_ok: bool,
        follower_ok: bool,
        read_errors: int,
    ) -> None:
        now = time.monotonic()
        if self._last_printed_at is not None and (now - self._last_printed_at) < self._interval_s:
            return
        self._last_printed_at = now
        print_periodic_status(
            self._opts,
            samples=samples,
            leader_ok=leader_ok,
            follower_ok=follower_ok,
            read_errors=read_errors,
        )
