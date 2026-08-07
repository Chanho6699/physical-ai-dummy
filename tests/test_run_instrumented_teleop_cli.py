"""scripts/run_instrumented_teleop.py CLI 단위/통합 테스트.

실물 하드웨어/실제 SO101Leader/SO101Follower에는 절대 연결하지 않는다 - ``run()``의
``leader_factory``/``follower_factory``/``processors_factory`` 주입 지점을 통해 가짜 객체로
완전히 대체한다. ``--dry-run``은 그 팩토리들을 아예 호출하지 않는다는 것까지 확인한다.
"""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_instrumented_teleop.py"

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")


def _load_cli_module():
    module_name = "run_instrumented_teleop_under_test"
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


def _wrist_roll_deg(raw: float) -> float:
    return (raw - 2047.5) * 360.0 / 4095


ALL_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _action_dict(wrist_roll_deg: float) -> dict:
    action = {f"{j}.pos": 0.0 for j in ALL_JOINTS}
    action["wrist_roll.pos"] = wrist_roll_deg
    return action


class _ForbiddenWriteCalled(AssertionError):
    pass


class FakeBus:
    def __init__(self, register_values=None):
        self.read_calls: list[tuple] = []
        self._values = dict(
            register_values
            or {
                "Goal_Position": 2021,
                "Present_Position": 2023,
                "Torque_Enable": 1,
                "Moving": 0,
                "Status": 0,
                "Acceleration": 254,
                "Acceleration_Multiplier ": 1,
                "Operating_Mode": 0,
            }
        )

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        self.read_calls.append((data_name, motor, normalize))
        return self._values[data_name]

    def write(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeBus.write() 호출됨")

    def sync_write(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeBus.sync_write() 호출됨")

    def enable_torque(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeBus.enable_torque() 호출됨")

    def disable_torque(self, *a, **k):
        raise _ForbiddenWriteCalled("FakeBus.disable_torque() 호출됨")


@dataclass
class _FakeMotorCalibration:
    range_min: int
    range_max: int


class FakeLeader:
    instances: list["FakeLeader"] = []

    def __init__(self, wrist_roll_sequence):
        self._sequence = list(wrist_roll_sequence)
        self._index = 0
        self.connect_calls = 0
        self.disconnect_calls = 0
        FakeLeader.instances.append(self)

    def connect(self):
        self.connect_calls += 1

    def get_action(self):
        value = self._sequence[min(self._index, len(self._sequence) - 1)]
        self._index += 1
        if isinstance(value, BaseException):
            raise value
        return _action_dict(value)

    def disconnect(self):
        self.disconnect_calls += 1


class FakeFollower:
    instances: list["FakeFollower"] = []

    def __init__(self, *, present_raw=2023):
        self.bus = FakeBus(register_values={"Goal_Position": 2021, "Present_Position": present_raw, "Torque_Enable": 1, "Moving": 0, "Status": 0, "Acceleration": 254, "Acceleration_Multiplier ": 1, "Operating_Mode": 0})
        self.calibration = {"wrist_roll": _FakeMotorCalibration(range_min=0, range_max=4095)}
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.send_action_calls: list[dict] = []
        FakeFollower.instances.append(self)

    def connect(self):
        self.connect_calls += 1

    def get_observation(self):
        return {f"{j}.pos": 0.0 for j in ALL_JOINTS}

    def send_action(self, action):
        self.send_action_calls.append(dict(action))
        return dict(action)

    def disconnect(self):
        self.disconnect_calls += 1


def _identity(pair):
    return pair[0]


def _fake_processors_factory():
    return _identity, _identity, _identity


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    FakeLeader.instances.clear()
    FakeFollower.instances.clear()
    yield
    FakeLeader.instances.clear()
    FakeFollower.instances.clear()


def _patch_ports_free(monkeypatch, cli):
    import hardware.safety.port_conflict as pc

    def _fake_check(port, *, timeout_s=pc.DEFAULT_CHECK_TIMEOUT_S):
        return pc.PortConflictReport(
            port=port, resolved_path=port, port_exists=True, checked_with=("lsof",), busy=False,
            busy_confirmed=True, holder_processes=(), notes=("테스트: 점유 없음",),
        )

    monkeypatch.setattr(cli.pc, "check_port_conflict", _fake_check)


def _patch_ports_busy(monkeypatch, cli):
    import hardware.safety.port_conflict as pc

    def _fake_check(port, *, timeout_s=pc.DEFAULT_CHECK_TIMEOUT_S):
        return pc.PortConflictReport(
            port=port, resolved_path=port, port_exists=True, checked_with=("lsof",), busy=True,
            busy_confirmed=True, holder_processes=(pc.ProcessInfo(pid=1, command="x", args="x"),), notes=("점유",),
        )

    monkeypatch.setattr(cli.pc, "check_port_conflict", _fake_check)


def _base_args(cli, *, leader_cal, follower_cal, tmp_path, extra=None):
    argv = [
        "--leader-port", "/dev/fake_leader",
        "--follower-port", "/dev/fake_follower",
        "--leader-id", "chanho_leader",
        "--follower-id", "chanho_follower",
        "--csv-dir", str(tmp_path / "csv_out"),
        "--no-dashboard",
        "--duration-sec", "100",
    ]
    if extra:
        argv.extend(extra)
    args = cli.build_arg_parser().parse_args(argv)
    # leader calibration path 해석은 id 기반 템플릿이라 테스트 tmp 경로를 직접 넣을 수 없으니
    # resolve_leader_calibration_path를 monkeypatch 없이 우회하기 위해 템플릿을 임시로 바꾼다.
    return args


# ---------------------------------------------------------------------------
# 정상 흐름
# ---------------------------------------------------------------------------


def test_run_normal_completion_writes_csv_json_and_disconnects(cli, monkeypatch, tmp_path):
    _patch_ports_free(monkeypatch, cli)
    monkeypatch.setattr(cli, "LEADER_CALIBRATION_PATH_TEMPLATE", str(tmp_path / "{id}.json"))
    (tmp_path / "chanho_leader.json").write_text(json.dumps(CALIBRATION_PAYLOAD), encoding="utf-8")
    follower_cal = tmp_path / "chanho_follower.json"
    follower_cal.write_text(json.dumps(CALIBRATION_PAYLOAD), encoding="utf-8")
    monkeypatch.setattr(cli.calres, "FOLLOWER_CALIBRATION_PATH_TEMPLATE", str(tmp_path / "{id}.json"))

    args = _base_args(cli, leader_cal=None, follower_cal=None, tmp_path=tmp_path, extra=["--duration-sec", "0.3"])

    leader_start = _wrist_roll_deg(2023)
    exit_code = cli.run(
        args,
        leader_factory=lambda: FakeLeader([leader_start, leader_start + 0.05, leader_start + 0.1]),
        follower_factory=lambda: FakeFollower(present_raw=2023),
        processors_factory=_fake_processors_factory,
    )

    assert exit_code == 0
    assert len(FakeLeader.instances) == 1
    assert len(FakeFollower.instances) == 1
    assert FakeLeader.instances[0].connect_calls == 1
    assert FakeLeader.instances[0].disconnect_calls == 1
    assert FakeFollower.instances[0].connect_calls == 1
    assert FakeFollower.instances[0].disconnect_calls == 1

    csv_files = list((tmp_path / "csv_out").glob("instrumented_wrist_roll_*.csv"))
    json_files = list((tmp_path / "csv_out").glob("instrumented_wrist_roll_*_report.json"))
    assert len(csv_files) == 1
    assert len(json_files) == 1

    with csv_files[0].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1

    report = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert report["direct_register_write_count"] == 0
    assert report["stopped_reason"] == "DURATION_ELAPSED"


def test_run_logs_warning_but_never_blocks_command_on_large_delta(cli, monkeypatch, tmp_path, capsys):
    """passive 모드 핵심 계약: command-delta 문턱값을 넘어도 send_action은 계속 호출되고,
    프로그램은 duration_sec까지 정상적으로 계속 진행되며(exit_code=0), warning만 기록된다."""
    _patch_ports_free(monkeypatch, cli)
    monkeypatch.setattr(cli, "LEADER_CALIBRATION_PATH_TEMPLATE", str(tmp_path / "{id}.json"))
    (tmp_path / "chanho_leader.json").write_text(json.dumps(CALIBRATION_PAYLOAD), encoding="utf-8")
    (tmp_path / "chanho_follower.json").write_text(json.dumps(CALIBRATION_PAYLOAD), encoding="utf-8")
    monkeypatch.setattr(cli.calres, "FOLLOWER_CALIBRATION_PATH_TEMPLATE", str(tmp_path / "{id}.json"))

    args = _base_args(cli, leader_cal=None, follower_cal=None, tmp_path=tmp_path, extra=["--duration-sec", "0.3"])

    leader_start = _wrist_roll_deg(2023)
    # 두 번째 cycle부터 계속 한계(2°)를 넘는 command를 보낸다 - 예전이면 여기서 멈췄을 것이다.
    exit_code = cli.run(
        args,
        leader_factory=lambda: FakeLeader([leader_start, leader_start + 5.0, leader_start + 6.0, leader_start + 7.0]),
        follower_factory=lambda: FakeFollower(present_raw=2023),
        processors_factory=_fake_processors_factory,
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "stopped_reason=DURATION_ELAPSED" in out
    assert "direct_register_write_count=0" in out
    # 한계를 넘은 command들도 전부 send_action에 그대로 전달됐어야 한다.
    sent = FakeFollower.instances[0].send_action_calls
    assert len(sent) >= 3
    assert sent[1]["wrist_roll.pos"] == pytest.approx(leader_start + 5.0)
    assert sent[2]["wrist_roll.pos"] == pytest.approx(leader_start + 6.0)

    json_files = list((tmp_path / "csv_out").glob("instrumented_wrist_roll_*_report.json"))
    report = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert report["warning_thresholds"]["command_delta_max_deg"] == pytest.approx(2.0)
    assert any(w["event_type"] == "WARNING_LARGE_COMMAND_DELTA" for w in report["warnings"])
    assert report["analysis"]["warning_counts"].get("WARNING_LARGE_COMMAND_DELTA", 0) >= 1


# ---------------------------------------------------------------------------
# 포트 점유/calibration 없음 -> factory 호출 자체를 하지 않는다
# ---------------------------------------------------------------------------


def test_run_does_not_connect_when_port_busy(cli, monkeypatch, tmp_path):
    _patch_ports_busy(monkeypatch, cli)
    args = _base_args(cli, leader_cal=None, follower_cal=None, tmp_path=tmp_path)

    exit_code = cli.run(
        args,
        leader_factory=lambda: FakeLeader([0.0]),
        follower_factory=lambda: FakeFollower(),
        processors_factory=_fake_processors_factory,
    )

    assert exit_code == 2
    assert len(FakeLeader.instances) == 0
    assert len(FakeFollower.instances) == 0


def test_run_blocked_when_calibration_file_missing(cli, monkeypatch, tmp_path):
    _patch_ports_free(monkeypatch, cli)
    monkeypatch.setattr(cli, "LEADER_CALIBRATION_PATH_TEMPLATE", str(tmp_path / "does_not_exist_{id}.json"))
    monkeypatch.setattr(cli.calres, "FOLLOWER_CALIBRATION_PATH_TEMPLATE", str(tmp_path / "does_not_exist_{id}.json"))
    args = _base_args(cli, leader_cal=None, follower_cal=None, tmp_path=tmp_path)

    exit_code = cli.run(
        args,
        leader_factory=lambda: FakeLeader([0.0]),
        follower_factory=lambda: FakeFollower(),
        processors_factory=_fake_processors_factory,
    )

    assert exit_code == 2
    assert len(FakeLeader.instances) == 0
    assert len(FakeFollower.instances) == 0


# ---------------------------------------------------------------------------
# dry-run: factory를 아예 호출하지 않는다
# ---------------------------------------------------------------------------


def test_dry_run_never_calls_factories_and_prints_exact_command(cli, monkeypatch, tmp_path, capsys):
    _patch_ports_free(monkeypatch, cli)
    monkeypatch.setattr(cli, "LEADER_CALIBRATION_PATH_TEMPLATE", str(tmp_path / "{id}.json"))
    (tmp_path / "chanho_leader.json").write_text(json.dumps(CALIBRATION_PAYLOAD), encoding="utf-8")
    (tmp_path / "chanho_follower.json").write_text(json.dumps(CALIBRATION_PAYLOAD), encoding="utf-8")
    monkeypatch.setattr(cli.calres, "FOLLOWER_CALIBRATION_PATH_TEMPLATE", str(tmp_path / "{id}.json"))

    args = _base_args(cli, leader_cal=None, follower_cal=None, tmp_path=tmp_path, extra=["--dry-run", "--duration-sec", "15", "--fps", "60"])

    exit_code = cli.run(
        args,
        leader_factory=lambda: (_ for _ in ()).throw(AssertionError("dry-run이 leader_factory를 호출했습니다")),
        follower_factory=lambda: (_ for _ in ()).throw(AssertionError("dry-run이 follower_factory를 호출했습니다")),
        processors_factory=lambda: (_ for _ in ()).throw(AssertionError("dry-run이 processors_factory를 호출했습니다")),
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "write_count=0" in out
    assert "run_instrumented_teleop.py" in out
    assert "--duration-sec 15" in out
    assert "--fps 60" in out
    assert len(FakeLeader.instances) == 0
    assert len(FakeFollower.instances) == 0


def test_dry_run_reports_busy_port(cli, monkeypatch, tmp_path, capsys):
    _patch_ports_busy(monkeypatch, cli)
    args = _base_args(cli, leader_cal=None, follower_cal=None, tmp_path=tmp_path, extra=["--dry-run"])
    exit_code = cli.run(args)
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "busy=True" in out


# ---------------------------------------------------------------------------
# CLI 기본값
# ---------------------------------------------------------------------------


def test_build_arg_parser_defaults():
    module = _load_cli_module()
    args = module.build_arg_parser().parse_args([])
    assert args.duration_sec == module.DEFAULT_DURATION_SEC == 15.0
    assert args.fps == module.DEFAULT_FPS == 60
    assert args.command_delta_max_deg == pytest.approx(2.0)
    assert args.position_jump_max_deg == pytest.approx(3.0)
    assert args.deadband_lookahead_ms == pytest.approx(module.DEFAULT_DEADBAND_LOOKAHEAD_MS)
    assert args.motion_response_noise_threshold_ticks == module.DEFAULT_MOTION_RESPONSE_NOISE_THRESHOLD_TICKS


def test_resolve_leader_port_prefers_cli(cli):
    port, source = cli.resolve_leader_port(cli_port="/dev/explicit", project_root=PROJECT_ROOT)
    assert port == "/dev/explicit"
    assert source == "cli"


def test_resolve_leader_port_raises_without_config(cli, tmp_path):
    with pytest.raises(cli.RefusalError):
        cli.resolve_leader_port(cli_port=None, project_root=tmp_path)


def test_resolve_leader_id_falls_back_to_default(cli, tmp_path):
    leader_id, source = cli.resolve_leader_id(cli_id=None, project_root=tmp_path)
    assert leader_id == cli.DEFAULT_LEADER_ID_FALLBACK
    assert source == "default_fallback"


# ---------------------------------------------------------------------------
# 소스 감사: 직접 write 호출/armed writer 사용 흔적 없음
# ---------------------------------------------------------------------------


def _code_only_source(module) -> str:
    source = inspect.getsource(module)
    first = source.index('"""')
    second = source.index('"""', first + 3)
    return source[second + 3 :]


def test_cli_source_contains_no_direct_write_call_patterns():
    module = _load_cli_module()
    source = _code_only_source(module)
    for forbidden in (".write(", ".sync_write(", "enable_torque(", "disable_torque("):
        assert forbidden not in source, f"금지된 패턴 '{forbidden}'이 run_instrumented_teleop.py 코드에 있습니다."


def test_cli_never_uses_single_joint_writers():
    module = _load_cli_module()
    source = _code_only_source(module)
    for forbidden in (
        "single_joint_parameter_writer",
        "single_joint_writer",
        "SingleJointArmedWriter",
        "SingleJointParameterWriter",
        "execute_single_armed_write",
        "execute_single_parameter_write",
    ):
        assert forbidden not in source


def test_cli_does_not_execute_real_run_at_import_time():
    """모듈을 import(실행)해도 실제 connect가 발생하지 않아야 한다 (main()은 __main__ guard 안)."""
    source = inspect.getsource(_load_cli_module())
    assert 'if __name__ == "__main__"' in source
