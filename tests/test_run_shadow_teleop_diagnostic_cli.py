"""scripts/run_shadow_teleop_diagnostic.py CLI 단위/통합 테스트.

실물 하드웨어/serial 포트에는 절대 연결하지 않는다 - ``LeaderWristRollReader``/
``FollowerWristRollStateReader``를 가짜 reader factory로 통째로 바꿔치기해서(``run()``의
``leader_reader_factory``/``follower_reader_factory`` 주입 지점 사용) CSV 생성, 안전 종료
경로(정상 종료/read 실패/Ctrl+C/follower 이동 감지), write_count=0 보고, disconnect가 항상
호출되는지를 검증한다.
"""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_shadow_teleop_diagnostic.py"

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")

from hardware.safety.shadow_teleop_diagnostic import FollowerAccelSnapshot, FollowerStateSnapshot


def _load_cli_module():
    module_name = "run_shadow_teleop_diagnostic_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli():
    return _load_cli_module()


CALIBRATION_PAYLOAD = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1686, "range_min": 1070, "range_max": 3135},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1007, "range_min": 793, "range_max": 3238},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 1635, "range_min": 873, "range_max": 3084},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 1716, "range_min": 1052, "range_max": 2977},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 1627, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 1523, "range_min": 2047, "range_max": 3496},
}


@pytest.fixture
def leader_calibration_file(tmp_path) -> Path:
    path = tmp_path / "chanho_leader.json"
    path.write_text(json.dumps(CALIBRATION_PAYLOAD), encoding="utf-8")
    return path


@pytest.fixture
def follower_calibration_file(tmp_path) -> Path:
    path = tmp_path / "chanho_follower.json"
    path.write_text(json.dumps(CALIBRATION_PAYLOAD), encoding="utf-8")
    return path


class _ForbiddenWriteCalled(AssertionError):
    pass


class FakeLeaderReader:
    """LeaderWristRollReader 대체 - write에 해당하는 메서드가 아예 없다."""

    instances: list["FakeLeaderReader"] = []

    def __init__(self, *, port, calibration, num_read_retries=2):
        self.port = port
        self.calibration = calibration
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self._raw_sequence = [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]
        self._index = 0
        FakeLeaderReader.instances.append(self)

    def connect(self):
        self.connect_calls += 1
        self.connected = True

    def read_raw(self):
        raw = self._raw_sequence[min(self._index, len(self._raw_sequence) - 1)]
        self._index += 1
        return raw

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    def write(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeLeaderReader.write() 호출됨")


class FakeFollowerReader:
    """FollowerWristRollStateReader 대체 - write에 해당하는 메서드가 아예 없다."""

    instances: list["FakeFollowerReader"] = []

    def __init__(self, *, port, calibration, num_read_retries=2):
        self.port = port
        self.calibration = calibration
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self._present_sequence = [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023]
        self._index = 0
        FakeFollowerReader.instances.append(self)

    def connect(self):
        self.connect_calls += 1
        self.connected = True

    def read_state(self):
        present = self._present_sequence[min(self._index, len(self._present_sequence) - 1)]
        self._index += 1
        return FollowerStateSnapshot(
            goal_raw=2021, present_raw=present, torque_enable=1, moving=0, status_raw=0, read_errors={}
        )

    def read_accel(self):
        return FollowerAccelSnapshot(acceleration=0, acceleration_multiplier=1, read_errors={})

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    def write(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeFollowerReader.write() 호출됨")


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    FakeLeaderReader.instances.clear()
    FakeFollowerReader.instances.clear()
    yield
    FakeLeaderReader.instances.clear()
    FakeFollowerReader.instances.clear()


def _patch_ports_free(monkeypatch, cli):
    import hardware.safety.port_conflict as pc

    def _fake_check(port, *, timeout_s=pc.DEFAULT_CHECK_TIMEOUT_S):
        return pc.PortConflictReport(
            port=port,
            resolved_path=port,
            port_exists=True,
            checked_with=("lsof",),
            busy=False,
            busy_confirmed=True,
            holder_processes=(),
            notes=("테스트: 점유 없음",),
        )

    monkeypatch.setattr(cli.pc, "check_port_conflict", _fake_check)


def _patch_ports_busy(monkeypatch, cli):
    import hardware.safety.port_conflict as pc

    def _fake_check(port, *, timeout_s=pc.DEFAULT_CHECK_TIMEOUT_S):
        return pc.PortConflictReport(
            port=port,
            resolved_path=port,
            port_exists=True,
            checked_with=("lsof",),
            busy=True,
            busy_confirmed=True,
            holder_processes=(pc.ProcessInfo(pid=99999, command="fake_holder", args="python run.py"),),
            notes=("테스트: 다른 프로세스가 점유 중",),
        )

    monkeypatch.setattr(cli.pc, "check_port_conflict", _fake_check)


def _base_args(cli, *, leader_cal, follower_cal, tmp_path, extra=None):
    argv = [
        "--leader-port",
        "/dev/fake_leader",
        "--follower-port",
        "/dev/fake_follower",
        "--leader-calibration-path",
        str(leader_cal),
        "--follower-calibration-path",
        str(follower_cal),
        "--csv-dir",
        str(tmp_path / "csv_out"),
        "--no-dashboard",
        "--duration-sec",
        "100",
    ]
    if extra:
        argv.extend(extra)
    return cli.build_arg_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# 정상 흐름: follower 이동 감지로 결정론적으로 멈춘 뒤 CSV/리포트/write_count=0 확인
# ---------------------------------------------------------------------------


def test_run_writes_csv_and_disconnects_both_with_zero_writes(cli, monkeypatch, leader_calibration_file, follower_calibration_file, tmp_path, capsys):
    _patch_ports_free(monkeypatch, cli)
    # 2번째 샘플에서 follower present_raw가 크게 튀어 follower_move_abort로 결정론적으로 멈춘다.
    args = _base_args(cli, leader_cal=leader_calibration_file, follower_cal=follower_calibration_file, tmp_path=tmp_path)

    def _leader_factory(*, port, calibration, num_read_retries=2):
        reader = FakeLeaderReader(port=port, calibration=calibration, num_read_retries=num_read_retries)
        return reader

    def _follower_factory(*, port, calibration, num_read_retries=2):
        reader = FakeFollowerReader(port=port, calibration=calibration, num_read_retries=num_read_retries)
        reader._present_sequence = [2023, 2023 + 40]  # 2번째 샘플에서 큰 이동
        return reader

    exit_code = cli.run(args, leader_reader_factory=_leader_factory, follower_reader_factory=_follower_factory)

    assert exit_code == 1  # follower_moved_unexpectedly는 duration_elapsed/keyboard_interrupt가 아니다
    assert len(FakeLeaderReader.instances) == 1
    assert len(FakeFollowerReader.instances) == 1
    assert FakeLeaderReader.instances[0].connect_calls == 1
    assert FakeLeaderReader.instances[0].disconnect_calls == 1
    assert FakeFollowerReader.instances[0].connect_calls == 1
    assert FakeFollowerReader.instances[0].disconnect_calls == 1

    csv_files = list((tmp_path / "csv_out").glob("shadow_wrist_roll_*.csv"))
    assert len(csv_files) == 1
    with csv_files[0].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # 문제가 된 샘플까지 포함해서 2행
    assert set(rows[0].keys()) == set(cli.CSV_FIELDNAMES)

    out = capsys.readouterr().out
    assert "write_count=0" in out
    assert "stopped_reason=follower_moved_unexpectedly" in out


def test_run_stops_safely_on_read_error(cli, monkeypatch, leader_calibration_file, follower_calibration_file, tmp_path, capsys):
    _patch_ports_free(monkeypatch, cli)
    args = _base_args(cli, leader_cal=leader_calibration_file, follower_cal=follower_calibration_file, tmp_path=tmp_path)

    class _RaisingLeaderReader(FakeLeaderReader):
        def read_raw(self):
            if self._index >= 1:
                raise RuntimeError("simulated comm failure")
            return super().read_raw()

    exit_code = cli.run(args, leader_reader_factory=_RaisingLeaderReader, follower_reader_factory=FakeFollowerReader)

    assert exit_code == 1
    assert FakeLeaderReader.instances[0].disconnect_calls == 1  # 실패해도 disconnect는 반드시 호출
    assert FakeFollowerReader.instances[0].disconnect_calls == 1
    out = capsys.readouterr().out
    assert "stopped_reason=read_error" in out
    assert "write_count=0" in out


def test_run_stops_safely_on_keyboard_interrupt(cli, monkeypatch, leader_calibration_file, follower_calibration_file, tmp_path, capsys):
    _patch_ports_free(monkeypatch, cli)
    args = _base_args(cli, leader_cal=leader_calibration_file, follower_cal=follower_calibration_file, tmp_path=tmp_path)

    class _InterruptingLeaderReader(FakeLeaderReader):
        def read_raw(self):
            if self._index >= 1:
                raise KeyboardInterrupt
            return super().read_raw()

    exit_code = cli.run(args, leader_reader_factory=_InterruptingLeaderReader, follower_reader_factory=FakeFollowerReader)

    assert exit_code == 0  # keyboard_interrupt도 정상적인 종료로 취급한다
    assert FakeLeaderReader.instances[0].disconnect_calls == 1
    assert FakeFollowerReader.instances[0].disconnect_calls == 1
    out = capsys.readouterr().out
    assert "stopped_reason=keyboard_interrupt" in out
    assert "write_count=0" in out


# ---------------------------------------------------------------------------
# 포트 점유 / calibration 없음 -> 연결 시도 자체를 하지 않는다
# ---------------------------------------------------------------------------


def test_run_does_not_connect_when_port_busy(cli, monkeypatch, leader_calibration_file, follower_calibration_file, tmp_path, capsys):
    _patch_ports_busy(monkeypatch, cli)
    args = _base_args(cli, leader_cal=leader_calibration_file, follower_cal=follower_calibration_file, tmp_path=tmp_path)

    exit_code = cli.run(args, leader_reader_factory=FakeLeaderReader, follower_reader_factory=FakeFollowerReader)

    assert exit_code == 2
    assert len(FakeLeaderReader.instances) == 0
    assert len(FakeFollowerReader.instances) == 0
    out = capsys.readouterr().out
    assert "write_count=0" in out


def test_run_blocked_when_leader_calibration_missing(cli, monkeypatch, follower_calibration_file, tmp_path):
    _patch_ports_free(monkeypatch, cli)
    missing = tmp_path / "does_not_exist.json"
    args = _base_args(cli, leader_cal=missing, follower_cal=follower_calibration_file, tmp_path=tmp_path)

    exit_code = cli.run(args, leader_reader_factory=FakeLeaderReader, follower_reader_factory=FakeFollowerReader)

    assert exit_code == 2
    assert len(FakeLeaderReader.instances) == 0
    assert len(FakeFollowerReader.instances) == 0


def test_run_blocked_when_follower_calibration_missing(cli, monkeypatch, leader_calibration_file, tmp_path):
    _patch_ports_free(monkeypatch, cli)
    missing = tmp_path / "does_not_exist.json"
    args = _base_args(cli, leader_cal=leader_calibration_file, follower_cal=missing, tmp_path=tmp_path)

    exit_code = cli.run(args, leader_reader_factory=FakeLeaderReader, follower_reader_factory=FakeFollowerReader)

    assert exit_code == 2
    assert len(FakeLeaderReader.instances) == 0
    assert len(FakeFollowerReader.instances) == 0


# ---------------------------------------------------------------------------
# 리더 포트/calibration 경로 해석
# ---------------------------------------------------------------------------


def test_resolve_leader_port_prefers_cli_over_local_config(cli):
    port, source = cli.resolve_leader_port(cli_port="/dev/explicit", project_root=PROJECT_ROOT)
    assert port == "/dev/explicit"
    assert source == "cli"


def test_resolve_leader_port_raises_when_nothing_available(cli, tmp_path):
    with pytest.raises(cli.RefusalError):
        cli.resolve_leader_port(cli_port=None, project_root=tmp_path)


def test_resolve_leader_calibration_path_prefers_cli_path(cli, tmp_path):
    explicit = tmp_path / "explicit.json"
    path, source = cli.resolve_leader_calibration_path(
        cli_calibration_path=str(explicit), cli_calibration_id=None, project_root=tmp_path
    )
    assert path == explicit
    assert source == "cli_path"


def test_resolve_leader_calibration_path_falls_back_to_default_id(cli, tmp_path):
    path, source = cli.resolve_leader_calibration_path(
        cli_calibration_path=None, cli_calibration_id=None, project_root=tmp_path
    )
    assert source == "default_fallback"
    assert cli.DEFAULT_LEADER_ID_FALLBACK in str(path)


# ---------------------------------------------------------------------------
# CLI 기본값 + 소스 감사: write 계열 호출이 이 스크립트 어디에도 없음
# ---------------------------------------------------------------------------


def test_build_arg_parser_defaults():
    module = _load_cli_module()
    args = module.build_arg_parser().parse_args([])
    assert args.duration_sec == module.DEFAULT_DURATION_SEC
    assert args.duration_sec == 20.0


def test_cli_source_contains_no_write_call_patterns():
    module = _load_cli_module()
    source = inspect.getsource(module)
    for forbidden in (".write(", ".sync_write(", "enable_torque(", "disable_torque("):
        assert forbidden not in source, f"금지된 패턴 '{forbidden}'이 run_shadow_teleop_diagnostic.py에 있습니다."


def test_cli_never_imports_armed_writer_or_configure():
    module = _load_cli_module()
    source = inspect.getsource(module)
    assert "SingleJointArmedWriter" not in source
    assert "execute_single_armed_write" not in source
    assert "SOFollower" not in source
    assert "SOLeader" not in source
    assert ".configure(" not in source
