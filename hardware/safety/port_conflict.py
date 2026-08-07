"""직렬 포트 점유(다른 프로세스가 이미 열고 있는지) 여부를 읽기 전용으로 확인한다.

이 모듈은 포트를 열지 않는다 - ``lsof``/``fuser``/``ps`` 같은 읽기 전용 시스템 명령만
실행해서 "누가 이 포트를 이미 쓰고 있는가"를 조사한다. 어떤 프로세스도 종료(kill)하지
않으며, 그런 기능 자체를 제공하지 않는다 (이 파일에 kill/terminate 계열 함수 없음).

판정 원칙: 점유 여부를 확실히 알 수 없으면(포트가 없거나, lsof/fuser를 둘 다 쓸 수
없거나, 명령이 타임아웃되면) **안전 측으로 busy=True 취급**한다 - "확인이 안 되면
연결하지 않는다"가 이 모듈의 기본 태도다. ``busy_confirmed``는 실제로 다른 프로세스를
확인했을 때만 True다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHECK_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    command: str
    args: str

    def to_dict(self) -> dict:
        return {"pid": self.pid, "command": self.command, "args": self.args}


@dataclass(frozen=True)
class PortConflictReport:
    port: str
    resolved_path: str | None
    port_exists: bool
    checked_with: tuple[str, ...]
    busy: bool  # True: 확인된 점유 OR 판정 불가(보수적으로 busy 취급)
    busy_confirmed: bool  # True일 때만 실제로 다른 프로세스를 식별함
    holder_processes: tuple[ProcessInfo, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "port": self.port,
            "resolved_path": self.resolved_path,
            "port_exists": self.port_exists,
            "checked_with": list(self.checked_with),
            "busy": self.busy,
            "busy_confirmed": self.busy_confirmed,
            "holder_processes": [p.to_dict() for p in self.holder_processes],
            "notes": list(self.notes),
        }


def _run(cmd: list[str], timeout_s: float) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _pids_from_lsof(path: str, timeout_s: float) -> set[int] | None:
    if shutil.which("lsof") is None:
        return None
    result = _run(["lsof", "-t", path], timeout_s)
    if result is None:
        return None
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def _pids_from_fuser(path: str, timeout_s: float) -> set[int] | None:
    if shutil.which("fuser") is None:
        return None
    result = _run(["fuser", path], timeout_s)
    if result is None:
        return None
    pids: set[int] = set()
    combined = (result.stdout or "") + (result.stderr or "")
    for token in combined.split():
        token = token.rstrip("cemkK")  # fuser access-mode suffix 문자 제거 (예: "1234c")
        if token.isdigit():
            pids.add(int(token))
    return pids


def _process_info(pid: int, timeout_s: float) -> ProcessInfo:
    result = _run(["ps", "-o", "pid=,comm=,args=", "-p", str(pid)], timeout_s)
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return ProcessInfo(pid=pid, command="(알 수 없음)", args="(ps 조회 실패 - 프로세스가 이미 종료되었을 수 있음)")
    line = result.stdout.strip()
    parts = line.split(None, 2)
    pid_val = int(parts[0]) if parts and parts[0].isdigit() else pid
    command = parts[1] if len(parts) > 1 else "(알 수 없음)"
    args = parts[2] if len(parts) > 2 else command
    return ProcessInfo(pid=pid_val, command=command, args=args)


def check_port_conflict(port: str, *, timeout_s: float = DEFAULT_CHECK_TIMEOUT_S) -> PortConflictReport:
    """포트 파일 존재/점유 여부를 읽기 전용으로 조사한다.

    어떤 프로세스도 종료하지 않는다 - 이 함수가 하는 일은 ``lsof``/``fuser``/``ps``
    출력을 읽는 것뿐이다.
    """

    notes: list[str] = []
    port_path = Path(port)
    port_exists = port_path.exists()
    resolved_path: str | None = None
    if port_exists:
        try:
            resolved_path = str(port_path.resolve())
        except OSError:
            resolved_path = None

    if not port_exists:
        notes.append("포트 파일이 존재하지 않습니다 (케이블/전원/USB 인식 여부를 확인하세요).")
        return PortConflictReport(
            port=port,
            resolved_path=resolved_path,
            port_exists=False,
            checked_with=(),
            busy=True,  # 존재하지 않는 포트는 연결도 불가하므로 안전 측으로 busy 취급
            busy_confirmed=False,
            holder_processes=(),
            notes=tuple(notes),
        )

    checked_with: list[str] = []
    pid_sets: list[set[int]] = []

    lsof_pids = _pids_from_lsof(port, timeout_s)
    if lsof_pids is not None:
        checked_with.append("lsof")
        pid_sets.append(lsof_pids)
    if resolved_path is not None and resolved_path != port:
        lsof_pids_resolved = _pids_from_lsof(resolved_path, timeout_s)
        if lsof_pids_resolved is not None:
            if "lsof" not in checked_with:
                checked_with.append("lsof")
            pid_sets.append(lsof_pids_resolved)

    if not checked_with:
        fuser_pids = _pids_from_fuser(port, timeout_s)
        if fuser_pids is not None:
            checked_with.append("fuser")
            pid_sets.append(fuser_pids)

    if not checked_with:
        notes.append(
            "lsof/fuser를 모두 사용할 수 없어 점유 여부를 판정할 수 없습니다 - 안전을 위해 busy로 취급합니다."
        )
        return PortConflictReport(
            port=port,
            resolved_path=resolved_path,
            port_exists=True,
            checked_with=(),
            busy=True,
            busy_confirmed=False,
            holder_processes=(),
            notes=tuple(notes),
        )

    all_pids: set[int] = set()
    for pid_set in pid_sets:
        all_pids |= pid_set

    my_pid = os.getpid()
    other_pids = {p for p in all_pids if p != my_pid}

    holder_processes = tuple(sorted((_process_info(pid, timeout_s) for pid in other_pids), key=lambda p: p.pid))
    busy = len(holder_processes) > 0
    if busy:
        pid_list = ", ".join(str(p.pid) for p in holder_processes)
        notes.append(
            f"다른 프로세스({pid_list})가 이 포트를 사용 중입니다 - 연결을 시도하지 않습니다. "
            "필요하면 사용자가 직접 해당 프로세스(예: 기존 read-only state server)를 중지한 뒤 다시 실행하세요."
        )
    else:
        notes.append("다른 프로세스의 포트 점유가 감지되지 않았습니다.")

    return PortConflictReport(
        port=port,
        resolved_path=resolved_path,
        port_exists=True,
        checked_with=tuple(checked_with),
        busy=busy,
        busy_confirmed=True,
        holder_processes=holder_processes,
        notes=tuple(notes),
    )
