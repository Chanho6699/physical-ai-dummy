"""hardware/safety/port_conflict.py 단위 테스트.

실제 lsof/fuser/ps를 실행하되, 이 모듈이 어떤 프로세스도 종료하지 않는다는 것과
(kill 계열 함수/명령이 코드에 없음을 감사) 판정 불가 시 안전 측(busy=True)으로
기운다는 것을 검증한다.
"""

from __future__ import annotations

import inspect

from hardware.safety import port_conflict as pc


def test_nonexistent_port_reports_busy_true_and_not_confirmed(tmp_path):
    missing = tmp_path / "does_not_exist_tty"
    report = pc.check_port_conflict(str(missing))
    assert report.port_exists is False
    assert report.busy is True
    assert report.busy_confirmed is False
    assert report.holder_processes == ()


def test_existing_unused_port_file_is_reported_free(tmp_path):
    fake_port = tmp_path / "fake_serial_port"
    fake_port.write_text("not a real serial device, just a regular file for the test")

    report = pc.check_port_conflict(str(fake_port))

    assert report.port_exists is True
    # lsof/fuser 둘 다 사용 불가능한 극히 드문 환경이 아니라면, 아무도 쓰지 않는 일반
    # 파일은 busy=False로 판정되어야 한다.
    if report.checked_with:
        assert report.busy is False
        assert report.busy_confirmed is True
        assert report.holder_processes == ()


def test_port_held_by_current_process_is_not_reported_as_conflict(tmp_path):
    """자기 자신의 PID는 '다른 프로세스가 점유'로 취급하지 않아야 한다."""
    fake_port = tmp_path / "self_held_port"
    fake_port.write_text("x")

    with fake_port.open("r") as _fh:  # 현재 프로세스가 이 파일을 열어 둔 상태로 검사
        report = pc.check_port_conflict(str(fake_port))

    if report.checked_with:
        # 자기 자신의 open 핸들만 있다면 다른 프로세스 점유로 잡히면 안 된다.
        assert report.busy is False


def test_report_serializes_to_plain_dict(tmp_path):
    fake_port = tmp_path / "fake_serial_port"
    fake_port.write_text("x")
    report = pc.check_port_conflict(str(fake_port))
    d = report.to_dict()
    assert d["port"] == str(fake_port)
    assert isinstance(d["holder_processes"], list)
    assert isinstance(d["notes"], list)


# ---------------------------------------------------------------------------
# 감사: 프로세스 종료 기능이 이 모듈에 존재하지 않는다
# ---------------------------------------------------------------------------


def test_module_has_no_kill_or_terminate_capability():
    # 설명 docstring에는 "kill(종료)"라는 단어가 사람이 읽는 설명으로 등장할 수 있으므로,
    # 실제 프로세스 종료를 수행할 수 있는 호출 패턴만 정밀하게 금지한다.
    source = inspect.getsource(pc)
    for banned in ("os.kill(", ".kill(", "SIGKILL", "SIGTERM", '"kill"', "'kill'", ".terminate("):
        assert banned not in source, f"port_conflict.py에 금지된 패턴 '{banned}'가 포함되어 있습니다."


def test_check_port_conflict_module_public_api_is_read_only():
    public_names = [name for name in dir(pc) if not name.startswith("_")]
    banned_substrings = ("kill", "terminate", "write")
    for name in public_names:
        lowered = name.lower()
        for banned in banned_substrings:
            assert banned not in lowered, f"'{name}'에 금지된 패턴 '{banned}'가 포함되어 있습니다."
