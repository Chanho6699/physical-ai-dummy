"""scripts/run_single_joint_hardware_test.py CLI 단위/통합 테스트.

실물 하드웨어에 연결하지 않는다 - ``hardware.safety.single_joint_hardware_inspector.
SingleJointInspector``를 가짜 클래스로 바꿔치기해서 inspect-only/dry-run 흐름을
검증한다.

**armed 모드는 이 파일에서 절대 실행하지 않는다** (``_run_armed``를 호출하는 테스트가
없다) - 두 번의 명시적 확인 게이트가 소스에 존재하는지만 정적으로 확인한다. 이는
작업 지시(Claude Code가 구현/테스트 중 armed 모드를 호출해서는 안 됨)를 그대로
따른 것이다.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_single_joint_hardware_test.py"


def _load_cli_module():
    module_name = "run_single_joint_hardware_test_under_test"
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
def calibration_file(tmp_path) -> Path:
    path = tmp_path / "chanho_follower.json"
    path.write_text(json.dumps(CALIBRATION_PAYLOAD), encoding="utf-8")
    return path


class _ForbiddenWriteCalled(AssertionError):
    pass


class FakeInspector:
    """SingleJointInspector 대체 - write에 해당하는 메서드가 아예 없고, 호출 기록만 남긴다."""

    instances: list["FakeInspector"] = []

    def __init__(self, *, port, calibration):
        self.port = port
        self.calibration = calibration
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        FakeInspector.instances.append(self)

    def connect(self):
        self.connect_calls += 1
        self.connected = True

    def read_raw(self):
        return 2048  # raw_to_degrees(2048, 0, 4095) ~= 0.04°

    def read_degrees(self):
        return 0.0439

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    def write(self, *args, **kwargs):  # 방어적: 실수로라도 호출되면 즉시 실패
        raise _ForbiddenWriteCalled("FakeInspector.write() 호출됨")


@pytest.fixture(autouse=True)
def _reset_fake_inspector_instances():
    FakeInspector.instances.clear()
    yield
    FakeInspector.instances.clear()


def _patch_fake_inspector(monkeypatch):
    import hardware.safety.single_joint_hardware_inspector as inspector_module

    monkeypatch.setattr(inspector_module, "SingleJointInspector", FakeInspector)


def _patch_port_free(monkeypatch, cli):
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


def _patch_port_busy(monkeypatch, cli):
    import hardware.safety.port_conflict as pc

    def _fake_check(port, *, timeout_s=pc.DEFAULT_CHECK_TIMEOUT_S):
        return pc.PortConflictReport(
            port=port,
            resolved_path=port,
            port_exists=True,
            checked_with=("lsof",),
            busy=True,
            busy_confirmed=True,
            holder_processes=(pc.ProcessInfo(pid=99999, command="fake_state_server", args="python run.py"),),
            notes=("테스트: 다른 프로세스가 점유 중",),
        )

    monkeypatch.setattr(cli.pc, "check_port_conflict", _fake_check)


# ---------------------------------------------------------------------------
# inspect-only: write 0회, 정상 흐름
# ---------------------------------------------------------------------------


def test_inspect_only_connects_and_reads_with_zero_writes(cli, monkeypatch, calibration_file):
    _patch_fake_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        ["--mode", "inspect-only", "--port", "/dev/fake_port", "--calibration-path", str(calibration_file)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_inspect_only(args, config)

    assert report["connected"] is True
    assert report["motor_id"] == 5
    assert report["write_count"] == 0
    assert len(FakeInspector.instances) == 1
    assert FakeInspector.instances[0].connect_calls == 1
    assert FakeInspector.instances[0].disconnect_calls == 1


def test_inspect_only_reports_full_turn_and_teleop_status(cli, monkeypatch, calibration_file):
    _patch_fake_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        ["--mode", "inspect-only", "--port", "/dev/fake_port", "--calibration-path", str(calibration_file)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_inspect_only(args, config)

    assert report["is_full_turn"] is True
    assert report["unit"] == "degree"
    # 실제 저장소의 configs/generated/teleop_safe_ranges.json에서 wrist_roll은
    # MARGIN_COLLAPSED이다 - 이 CLI는 그 파일을 있는 그대로 반영해야 한다.
    assert report["teleop_status"] == "MARGIN_COLLAPSED"
    assert report["historical_range_applied"] is False


# ---------------------------------------------------------------------------
# dry-run: write 0회, positive/negative, 반대 방향 자동 전환 없음
# ---------------------------------------------------------------------------


def test_dry_run_positive_direction_zero_writes_and_ten_steps(cli, monkeypatch, calibration_file):
    _patch_fake_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "dry-run",
            "--direction",
            "positive",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_dry_run(args, config)

    assert report["write_count"] == 0
    assert report["direction"] == "positive"
    assert len(report["planned_targets"]) == 10
    assert report["final_verdict"] == "PASS"
    assert FakeInspector.instances[0].connect_calls == 1


def test_dry_run_requires_direction(cli, monkeypatch, calibration_file):
    _patch_fake_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        ["--mode", "dry-run", "--port", "/dev/fake_port", "--calibration-path", str(calibration_file)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    with pytest.raises(cli.RefusalError):
        cli._run_dry_run(args, config)


def test_dry_run_never_auto_switches_direction(cli, monkeypatch, calibration_file):
    """FakeInspector가 이음매 근처 값을 반환하도록 해서 positive가 BLOCKED되어도
    negative로 자동 전환되지 않는지 확인한다."""
    import hardware.safety.single_joint_hardware_inspector as inspector_module

    class NearSeamInspector(FakeInspector):
        def read_degrees(self):
            return 170.0  # inner band(±165) 밖

        def read_raw(self):
            return 4000

    monkeypatch.setattr(inspector_module, "SingleJointInspector", NearSeamInspector)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "dry-run",
            "--direction",
            "positive",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_dry_run(args, config)

    assert report["direction"] == "positive"
    assert report["final_verdict"] == "BLOCKED"


def test_dry_run_rejects_step_size_over_point_one(cli, monkeypatch, calibration_file):
    _patch_fake_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "dry-run",
            "--direction",
            "positive",
            "--step-size-deg",
            "0.2",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    with pytest.raises(cli.RefusalError):
        cli._run_dry_run(args, config)


def test_dry_run_rejects_max_delta_over_one_degree(cli, monkeypatch, calibration_file):
    _patch_fake_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "dry-run",
            "--direction",
            "positive",
            "--max-delta-deg",
            "1.5",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    with pytest.raises(cli.RefusalError):
        cli._run_dry_run(args, config)


# ---------------------------------------------------------------------------
# 포트 점유 -> BLOCKED, 연결 시도 안 함
# ---------------------------------------------------------------------------


def test_inspect_only_does_not_connect_when_port_busy(cli, monkeypatch, calibration_file):
    _patch_fake_inspector(monkeypatch)
    _patch_port_busy(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        ["--mode", "inspect-only", "--port", "/dev/fake_port", "--calibration-path", str(calibration_file)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_inspect_only(args, config)

    assert report["connected"] is False
    assert report["connect_skipped_reason"] == "port_busy_or_unknown"
    assert len(FakeInspector.instances) == 0  # 아예 생성/연결을 시도하지 않았다
    assert report["port_conflict"]["busy"] is True
    assert report["port_conflict"]["holder_processes"][0]["pid"] == 99999


def test_dry_run_is_blocked_when_port_busy(cli, monkeypatch, calibration_file):
    _patch_fake_inspector(monkeypatch)
    _patch_port_busy(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "dry-run",
            "--direction",
            "positive",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_dry_run(args, config)

    assert report["final_verdict"] == "BLOCKED"
    assert len(FakeInspector.instances) == 0


# ---------------------------------------------------------------------------
# calibration 파일 없음 -> BLOCKED
# ---------------------------------------------------------------------------


def test_inspect_only_blocked_when_calibration_file_missing(cli, monkeypatch, tmp_path):
    _patch_fake_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    missing_path = tmp_path / "does_not_exist.json"
    args = cli.build_arg_parser().parse_args(
        ["--mode", "inspect-only", "--port", "/dev/fake_port", "--calibration-path", str(missing_path)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_inspect_only(args, config)

    assert report["calibration_loaded"] is False
    assert report["connected"] is False
    assert len(FakeInspector.instances) == 0


def test_dry_run_blocked_when_calibration_file_missing(cli, monkeypatch, tmp_path):
    _patch_fake_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    missing_path = tmp_path / "does_not_exist.json"
    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "dry-run",
            "--direction",
            "positive",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(missing_path),
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_dry_run(args, config)

    assert report["final_verdict"] == "BLOCKED"


# ---------------------------------------------------------------------------
# register-diagnostic: write 0회, 판정 라벨, 포트 점유/calibration 없음 시 BLOCKED
# ---------------------------------------------------------------------------


class FakeRegisterDiagnosticInspector:
    """RegisterDiagnosticInspector 대체 - write에 해당하는 메서드가 아예 없다."""

    instances: list["FakeRegisterDiagnosticInspector"] = []

    def __init__(self, *, port, calibration):
        self.port = port
        self.calibration = calibration
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        FakeRegisterDiagnosticInspector.instances.append(self)

    def connect(self):
        self.connect_calls += 1
        self.connected = True

    def read_snapshot(self):
        from hardware.safety.single_joint_register_diagnostic import RegisterSnapshot

        return RegisterSnapshot(
            torque_enable=1,
            goal_position_raw=2024,
            present_position_raw=2023,
            moving=0,
            present_load=0,
            present_current=0,
            present_velocity=0,
            present_voltage=74,
            present_temperature=30,
            status_raw=0,
            read_errors={},
        )

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False


@pytest.fixture(autouse=True)
def _reset_fake_register_diagnostic_instances():
    FakeRegisterDiagnosticInspector.instances.clear()
    yield
    FakeRegisterDiagnosticInspector.instances.clear()


def _patch_fake_register_diagnostic_inspector(monkeypatch):
    import hardware.safety.single_joint_register_diagnostic as diag_module

    monkeypatch.setattr(diag_module, "RegisterDiagnosticInspector", FakeRegisterDiagnosticInspector)


def test_register_diagnostic_connects_and_reads_with_zero_writes(cli, monkeypatch, calibration_file):
    _patch_fake_register_diagnostic_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "register-diagnostic",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
            "--expected-start-raw",
            "2023",
            "--expected-goal-raw",
            "2024",
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_register_diagnostic(args, config)

    assert report["write_count"] == 0
    assert report["connected"] is True
    assert report["registers"]["Torque_Enable"] == 1
    assert report["registers"]["Goal_Position"] == 2024
    assert report["registers"]["Present_Position"] == 2023
    assert report["registers"]["Moving"] == 0
    assert report["registers"]["Moving_Status"] == "NOT_AVAILABLE_IN_INSTALLED_TABLE"
    assert report["goal_latched"] is True
    assert report["goal_present_delta"] == 1
    assert report["diagnostic_verdict"] == "COMMAND_LATCHED_BUT_NO_MOTION"
    assert len(FakeRegisterDiagnosticInspector.instances) == 1
    assert FakeRegisterDiagnosticInspector.instances[0].connect_calls == 1
    assert FakeRegisterDiagnosticInspector.instances[0].disconnect_calls == 1


def test_register_diagnostic_does_not_connect_when_port_busy(cli, monkeypatch, calibration_file):
    _patch_fake_register_diagnostic_inspector(monkeypatch)
    _patch_port_busy(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        ["--mode", "register-diagnostic", "--port", "/dev/fake_port", "--calibration-path", str(calibration_file)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_register_diagnostic(args, config)

    assert report["connected"] is False
    assert report["write_count"] == 0
    assert len(FakeRegisterDiagnosticInspector.instances) == 0
    assert report["diagnostic_verdict"] == "UNKNOWN"


def test_register_diagnostic_blocked_when_calibration_file_missing(cli, monkeypatch, tmp_path):
    _patch_fake_register_diagnostic_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    missing_path = tmp_path / "does_not_exist.json"
    args = cli.build_arg_parser().parse_args(
        ["--mode", "register-diagnostic", "--port", "/dev/fake_port", "--calibration-path", str(missing_path)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_register_diagnostic(args, config)

    assert report["calibration_loaded"] is False
    assert report["connected"] is False
    assert report["write_count"] == 0
    assert len(FakeRegisterDiagnosticInspector.instances) == 0


def test_register_diagnostic_json_report_has_no_secrets_or_home_paths(cli, monkeypatch, calibration_file):
    _patch_fake_register_diagnostic_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "register-diagnostic",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
            "--expected-start-raw",
            "2023",
            "--expected-goal-raw",
            "2024",
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_register_diagnostic(args, config)

    serialized = json.dumps(report, default=str)
    assert str(calibration_file) not in serialized
    for forbidden in ("token", "Token", "TOKEN", "password", "secret", "Authorization", "Bearer"):
        assert forbidden not in serialized


def test_register_diagnostic_mode_never_imports_writer_execute_function(cli):
    import inspect

    source = inspect.getsource(cli._run_register_diagnostic)
    assert "execute_single_armed_write" not in source
    assert "write_goal_position_once" not in source
    assert "Goal_Position\"," not in source  # write() 형태 호출 흔적이 없어야 함


# ---------------------------------------------------------------------------
# servo-parameter-diagnostic: write 0회, 판정/후보 계산, 포트 점유/calibration 없음 시 BLOCKED
# ---------------------------------------------------------------------------


class FakeServoParameterDiagnosticInspector:
    """ServoParameterDiagnosticInspector 대체 - write에 해당하는 메서드가 아예 없다."""

    instances: list["FakeServoParameterDiagnosticInspector"] = []

    def __init__(self, *, port, calibration):
        self.port = port
        self.calibration = calibration
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        FakeServoParameterDiagnosticInspector.instances.append(self)

    def connect(self):
        self.connect_calls += 1
        self.connected = True

    def read_snapshot(self):
        from hardware.safety.single_joint_servo_parameter_diagnostic import ServoParameterSnapshot

        return ServoParameterSnapshot(
            torque_enable=1,
            goal_position_raw=2021,
            present_position_raw=2023,
            moving=0,
            status_raw=0,
            cw_dead_zone=0,
            ccw_dead_zone=0,
            minimum_startup_force=0,
            operating_mode=0,
            acceleration=0,
            maximum_acceleration=0,
            goal_velocity=0,
            maximum_velocity_limit=0,
            moving_velocity_threshold=0,
            torque_limit=1000,
            max_torque_limit=1000,
            p_coefficient=32,
            i_coefficient=0,
            d_coefficient=0,
            lock=1,
            angular_resolution=1,
            read_errors={},
            unavailable_registers=(),
        )

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False


@pytest.fixture(autouse=True)
def _reset_fake_servo_parameter_diagnostic_instances():
    FakeServoParameterDiagnosticInspector.instances.clear()
    yield
    FakeServoParameterDiagnosticInspector.instances.clear()


def _patch_fake_servo_parameter_diagnostic_inspector(monkeypatch):
    import hardware.safety.single_joint_servo_parameter_diagnostic as diag_module

    monkeypatch.setattr(diag_module, "ServoParameterDiagnosticInspector", FakeServoParameterDiagnosticInspector)


def test_servo_parameter_diagnostic_connects_and_reads_with_zero_writes(cli, monkeypatch, calibration_file):
    _patch_fake_servo_parameter_diagnostic_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "servo-parameter-diagnostic",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
            "--expected-start-raw",
            "2023",
            "--expected-goal-raw",
            "2021",
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_servo_parameter_diagnostic(args, config)

    assert report["write_count"] == 0
    assert report["connected"] is True
    assert report["registers"]["state"]["Goal_Position"] == 2021
    assert report["goal_latched"] is True
    assert "VELOCITY_OR_ACCELERATION_RESTRICTION" in report["verdicts"]
    assert len(report["next_step_candidates"]) == 2
    for candidate in report["next_step_candidates"]:
        assert abs(candidate["requested_delta_deg"]) < 0.5
    assert len(FakeServoParameterDiagnosticInspector.instances) == 1
    assert FakeServoParameterDiagnosticInspector.instances[0].connect_calls == 1
    assert FakeServoParameterDiagnosticInspector.instances[0].disconnect_calls == 1


def test_servo_parameter_diagnostic_does_not_connect_when_port_busy(cli, monkeypatch, calibration_file):
    _patch_fake_servo_parameter_diagnostic_inspector(monkeypatch)
    _patch_port_busy(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        ["--mode", "servo-parameter-diagnostic", "--port", "/dev/fake_port", "--calibration-path", str(calibration_file)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_servo_parameter_diagnostic(args, config)

    assert report["connected"] is False
    assert report["write_count"] == 0
    assert len(FakeServoParameterDiagnosticInspector.instances) == 0


def test_servo_parameter_diagnostic_blocked_when_calibration_file_missing(cli, monkeypatch, tmp_path):
    _patch_fake_servo_parameter_diagnostic_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    missing_path = tmp_path / "does_not_exist.json"
    args = cli.build_arg_parser().parse_args(
        ["--mode", "servo-parameter-diagnostic", "--port", "/dev/fake_port", "--calibration-path", str(missing_path)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_servo_parameter_diagnostic(args, config)

    assert report["calibration_loaded"] is False
    assert report["connected"] is False
    assert report["write_count"] == 0
    assert len(FakeServoParameterDiagnosticInspector.instances) == 0


def test_servo_parameter_diagnostic_json_report_has_no_secrets_or_home_paths(cli, monkeypatch, calibration_file):
    _patch_fake_servo_parameter_diagnostic_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "servo-parameter-diagnostic",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
            "--expected-start-raw",
            "2023",
            "--expected-goal-raw",
            "2021",
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_servo_parameter_diagnostic(args, config)

    serialized = json.dumps(report, default=str)
    assert str(calibration_file) not in serialized
    for forbidden in ("token", "Token", "TOKEN", "password", "secret", "Authorization", "Bearer"):
        assert forbidden not in serialized


def test_servo_parameter_diagnostic_mode_never_writes_goal_position(cli):
    import inspect

    source = inspect.getsource(cli._run_servo_parameter_diagnostic)
    assert "execute_single_armed_write" not in source
    assert "write_goal_position_once" not in source
    assert ".write(" not in source
    assert "sync_write" not in source


# ---------------------------------------------------------------------------
# all-joint-parameter-diagnostic: write 0회, 6개 관절 비교, 판정 로직
# ---------------------------------------------------------------------------


ALL_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class FakeAllJointParameterDiagnosticInspector:
    """AllJointParameterDiagnosticInspector 대체 - write에 해당하는 메서드가 아예 없다."""

    instances: list["FakeAllJointParameterDiagnosticInspector"] = []

    def __init__(self, *, port, calibration):
        self.port = port
        self.calibration = calibration
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        FakeAllJointParameterDiagnosticInspector.instances.append(self)

    def connect(self):
        self.connect_calls += 1
        self.connected = True

    def read_all_snapshots(self):
        from hardware.safety.all_joint_parameter_diagnostic import JointRegisterSnapshot

        snapshots = {}
        for joint in ALL_JOINT_NAMES:
            accel = 0 if joint == "wrist_roll" else 254
            snapshots[joint] = JointRegisterSnapshot(
                joint=joint,
                motor_id=self.calibration[joint].id,
                torque_enable=1,
                operating_mode=0,
                goal_position_raw=2024 if joint == "wrist_roll" else 2000,
                present_position_raw=2023 if joint == "wrist_roll" else 2000,
                moving=0,
                status_raw=0,
                acceleration=accel,
                maximum_acceleration=254,
                cw_dead_zone=0,
                ccw_dead_zone=0,
                minimum_startup_force=0,
                torque_limit=1000,
                read_errors={},
                unavailable_registers=(),
            )
        return snapshots

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False


@pytest.fixture(autouse=True)
def _reset_fake_all_joint_parameter_diagnostic_instances():
    FakeAllJointParameterDiagnosticInspector.instances.clear()
    yield
    FakeAllJointParameterDiagnosticInspector.instances.clear()


def _patch_fake_all_joint_parameter_diagnostic_inspector(monkeypatch):
    import hardware.safety.all_joint_parameter_diagnostic as diag_module

    monkeypatch.setattr(diag_module, "AllJointParameterDiagnosticInspector", FakeAllJointParameterDiagnosticInspector)


def test_all_joint_parameter_diagnostic_connects_and_reads_with_zero_writes(cli, monkeypatch, calibration_file):
    _patch_fake_all_joint_parameter_diagnostic_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        ["--mode", "all-joint-parameter-diagnostic", "--port", "/dev/fake_port", "--calibration-path", str(calibration_file)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_all_joint_parameter_diagnostic(args, config)

    assert report["write_count"] == 0
    assert report["connected"] is True
    assert set(report["joints"]) == set(ALL_JOINT_NAMES)
    assert report["motor_ids"] == {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}
    assert report["acceleration_verdict"] == "WRIST_ROLL_ONLY_ZERO"
    assert "wrist_roll" in report["stuck_joints"]
    assert len(FakeAllJointParameterDiagnosticInspector.instances) == 1
    assert FakeAllJointParameterDiagnosticInspector.instances[0].connect_calls == 1
    assert FakeAllJointParameterDiagnosticInspector.instances[0].disconnect_calls == 1


def test_all_joint_parameter_diagnostic_does_not_connect_when_port_busy(cli, monkeypatch, calibration_file):
    _patch_fake_all_joint_parameter_diagnostic_inspector(monkeypatch)
    _patch_port_busy(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        ["--mode", "all-joint-parameter-diagnostic", "--port", "/dev/fake_port", "--calibration-path", str(calibration_file)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_all_joint_parameter_diagnostic(args, config)

    assert report["connected"] is False
    assert report["write_count"] == 0
    assert len(FakeAllJointParameterDiagnosticInspector.instances) == 0


def test_all_joint_parameter_diagnostic_blocked_when_calibration_file_missing(cli, monkeypatch, tmp_path):
    _patch_fake_all_joint_parameter_diagnostic_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    missing_path = tmp_path / "does_not_exist.json"
    args = cli.build_arg_parser().parse_args(
        ["--mode", "all-joint-parameter-diagnostic", "--port", "/dev/fake_port", "--calibration-path", str(missing_path)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_all_joint_parameter_diagnostic(args, config)

    assert report["calibration_loaded"] is False
    assert report["connected"] is False
    assert report["write_count"] == 0
    assert len(FakeAllJointParameterDiagnosticInspector.instances) == 0


def test_all_joint_parameter_diagnostic_json_report_has_no_secrets_or_home_paths(cli, monkeypatch, calibration_file):
    _patch_fake_all_joint_parameter_diagnostic_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        ["--mode", "all-joint-parameter-diagnostic", "--port", "/dev/fake_port", "--calibration-path", str(calibration_file)]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_all_joint_parameter_diagnostic(args, config)

    serialized = json.dumps(report, default=str)
    assert str(calibration_file) not in serialized
    for forbidden in ("token", "Token", "TOKEN", "password", "secret", "Authorization", "Bearer"):
        assert forbidden not in serialized


def test_all_joint_parameter_diagnostic_mode_never_writes(cli):
    import inspect

    source = inspect.getsource(cli._run_all_joint_parameter_diagnostic)
    assert "execute_single_armed_write" not in source
    assert "write_goal_position_once" not in source
    assert ".write(" not in source
    assert "sync_write" not in source
    # docstring 설명(prose)이 아니라 실제 인스턴스화/호출 흔적이 없는지 확인한다.
    assert "SOFollower(" not in source
    assert ".configure()" not in source


# ---------------------------------------------------------------------------
# JSON 리포트: 토큰/비밀값/불필요한 개인 경로 없음
# ---------------------------------------------------------------------------


def test_report_json_has_no_secrets_or_home_paths(cli, monkeypatch, calibration_file):
    _patch_fake_inspector(monkeypatch)
    _patch_port_free(monkeypatch, cli)

    args = cli.build_arg_parser().parse_args(
        [
            "--mode",
            "dry-run",
            "--direction",
            "positive",
            "--port",
            "/dev/fake_port",
            "--calibration-path",
            str(calibration_file),
        ]
    )
    config = cli._resolve_config(args, allow_default_fallback=True)
    report = cli._run_dry_run(args, config)

    serialized = json.dumps(report, default=str)
    assert str(calibration_file) not in serialized  # 실제 calibration 절대경로는 리포트에 없어야 함
    for forbidden in ("token", "Token", "TOKEN", "password", "secret", "Authorization", "Bearer"):
        assert forbidden not in serialized


# ---------------------------------------------------------------------------
# armed 게이트: 실행하지 않고 소스만 정적으로 확인
# ---------------------------------------------------------------------------


def test_armed_mode_is_a_recognized_choice_but_requires_two_confirmations(cli):
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--mode", "armed"])
    assert args.mode == "armed"

    source = inspect.getsource(cli._run_armed)
    # 두 확인 플래그가 모두 체크되는지 (소스 레벨) 확인한다 - 실행은 하지 않는다.
    assert "i_have_read_the_safety_plan" in source
    assert "confirm_single_write" in source
    assert "missing_flags" in source
    assert "expected_start_raw" in source and "expected_start_deg" in source
    # armed 첫 실행은 정확히 0.1°/0.1°만 허용해야 한다.
    assert "REQUIRED_ARMED_TOTAL_DELTA_DEG" in source
    assert "REQUIRED_ARMED_STEP_SIZE_DEG" in source
    # write 직전 재검사(포트 점유/calibration)를 다시 수행해야 한다.
    assert "check_port_conflict" in source
    assert "load_calibration_file" in source
    # override 옵션은 armed에서 금지되어야 한다.
    assert "start_deg_override" in source and "start_raw_override" in source
    # 실제 write 경로는 writer 모듈에만 위임한다.
    assert "execute_single_armed_write" in source
    assert "SingleJointArmedWriter" in source


def test_armed_mode_requires_direction_and_rejects_non_default_delta_values(cli):
    """소스에 "자동 방향 선택 없음"과 "0.1이 아니면 거부" 로직이 있는지 정적으로 확인한다."""
    source = inspect.getsource(cli._run_armed)
    assert "args.direction is None" in source
    assert "abs(args.max_delta_deg" in source
    assert "abs(args.step_size_deg" in source


def test_armed_mode_disallows_calibration_default_fallback(cli):
    source = inspect.getsource(cli.main)
    assert "allow_default_fallback=False" in source


def test_main_never_calls_run_armed_outside_of_explicit_armed_mode_branch():
    """main()의 armed 분기 외에는 _run_armed가 호출되지 않는다는 것을 소스로 확인한다
    (이 테스트 자체도 _run_armed를 실행하지 않는다)."""
    source = inspect.getsource(_load_cli_module().main)
    occurrences = source.count("_run_armed(")
    assert occurrences == 1


# ---------------------------------------------------------------------------
# acceleration-write 게이트: 실행하지 않고 소스만 정적으로 확인
# ---------------------------------------------------------------------------


def test_acceleration_write_mode_is_a_recognized_choice_but_requires_two_confirmations(cli):
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--mode", "acceleration-write"])
    assert args.mode == "acceleration-write"

    source = inspect.getsource(cli._run_acceleration_write)
    assert "i_understand_this_changes_servo_state" in source
    assert "confirm_acceleration_write" in source
    assert "missing_flags" in source
    assert "expected_current_acceleration" in source
    # write 직전 재검사(포트 점유/calibration)를 다시 수행해야 한다.
    assert "check_port_conflict" in source
    assert "load_calibration_file" in source
    # 실제 write 경로는 writer 모듈에만 위임한다.
    assert "execute_single_parameter_write" in source
    assert "SingleJointParameterWriter" in source


def test_acceleration_write_mode_never_touches_goal_position_or_armed_writer(cli):
    """요구사항 9번: 같은 실행에서 Goal_Position write/armed 이동을 절대 하지 않는다."""
    source = inspect.getsource(cli._run_acceleration_write)
    assert "execute_single_armed_write" not in source
    assert "write_goal_position_once" not in source
    assert "SingleJointArmedWriter" not in source


def test_acceleration_write_mode_disallows_calibration_default_fallback(cli):
    source = inspect.getsource(cli.main)
    # armed와 acceleration-write 둘 다 allow_default_fallback=False 분기를 타야 한다.
    assert source.count("allow_default_fallback=False") >= 2


def test_main_never_calls_run_acceleration_write_outside_of_explicit_branch():
    """main()의 acceleration-write 분기 외에는 _run_acceleration_write가 호출되지 않는다는
    것을 소스로 확인한다 (이 테스트 자체도 _run_acceleration_write를 실행하지 않는다)."""
    source = inspect.getsource(_load_cli_module().main)
    occurrences = source.count("_run_acceleration_write(")
    assert occurrences == 1
