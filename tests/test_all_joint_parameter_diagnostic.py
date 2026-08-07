"""hardware/safety/all_joint_parameter_diagnostic.py 단위 테스트.

실물 ``/dev/tty*``/``/dev/serial*``에는 절대 접근하지 않는다. 두 층으로 나눠 검증한다:

1. ``classify_acceleration_state``/``find_stuck_joints`` (순수 함수, lerobot 불필요).
2. ``AllJointParameterDiagnosticInspector`` + 가짜 ``FeetechMotorsBus`` - 6개 관절
   외 접근 없음, write 계열 메서드가 호출되는 즉시 ``AssertionError``, disconnect 중
   write 없음.
"""

from __future__ import annotations

import pytest

from hardware.safety.all_joint_parameter_diagnostic import (
    JOINT_NAMES,
    PARAMETER_REGISTERS,
    STATE_REGISTERS,
    AllJointVerdict,
    JointRegisterSnapshot,
    classify_acceleration_state,
    find_stuck_joints,
)
from hardware.state_server.calibration_loader import MotorCalibrationEntry

# 실제 chanho_follower.json 값 그대로.
CALIBRATION = {
    "shoulder_pan": MotorCalibrationEntry(id=1, drive_mode=0, homing_offset=-1686, range_min=1070, range_max=3135),
    "shoulder_lift": MotorCalibrationEntry(id=2, drive_mode=0, homing_offset=-1007, range_min=793, range_max=3238),
    "elbow_flex": MotorCalibrationEntry(id=3, drive_mode=0, homing_offset=1635, range_min=873, range_max=3084),
    "wrist_flex": MotorCalibrationEntry(id=4, drive_mode=0, homing_offset=1716, range_min=1052, range_max=2977),
    "wrist_roll": MotorCalibrationEntry(id=5, drive_mode=0, homing_offset=1627, range_min=0, range_max=4095),
    "gripper": MotorCalibrationEntry(id=6, drive_mode=0, homing_offset=1523, range_min=2047, range_max=3496),
}


def _snapshot(joint: str, **overrides) -> JointRegisterSnapshot:
    base = dict(
        joint=joint,
        motor_id=CALIBRATION[joint].id,
        torque_enable=1,
        operating_mode=0,
        goal_position_raw=2000,
        present_position_raw=2000,
        moving=0,
        status_raw=0,
        acceleration=254,
        maximum_acceleration=254,
        cw_dead_zone=0,
        ccw_dead_zone=0,
        minimum_startup_force=0,
        torque_limit=1000,
        read_errors={},
        unavailable_registers=(),
    )
    base.update(overrides)
    return JointRegisterSnapshot(**base)


def _all_joints(**per_joint_overrides) -> dict[str, JointRegisterSnapshot]:
    return {joint: _snapshot(joint, **per_joint_overrides.get(joint, {})) for joint in JOINT_NAMES}


# ---------------------------------------------------------------------------
# 매핑 확인
# ---------------------------------------------------------------------------


def test_joint_names_match_calibration_order():
    assert JOINT_NAMES == ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def test_motor_id_mapping_matches_real_calibration():
    for joint, entry in CALIBRATION.items():
        assert entry.id == {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}[
            joint
        ]


# ---------------------------------------------------------------------------
# classify_acceleration_state
# ---------------------------------------------------------------------------


def test_all_acceleration_zero_when_every_joint_zero():
    snapshots = _all_joints(**{j: {"acceleration": 0} for j in JOINT_NAMES})
    verdict, reasons = classify_acceleration_state(snapshots)
    assert verdict == AllJointVerdict.ALL_ACCELERATION_ZERO
    assert reasons


def test_wrist_roll_only_zero():
    overrides = {j: {"acceleration": 254} for j in JOINT_NAMES}
    overrides["wrist_roll"] = {"acceleration": 0}
    snapshots = _all_joints(**overrides)
    verdict, reasons = classify_acceleration_state(snapshots)
    assert verdict == AllJointVerdict.WRIST_ROLL_ONLY_ZERO
    assert reasons


def test_mixed_acceleration_state():
    overrides = {j: {"acceleration": 254} for j in JOINT_NAMES}
    overrides["wrist_roll"] = {"acceleration": 0}
    overrides["gripper"] = {"acceleration": 0}
    snapshots = _all_joints(**overrides)
    verdict, _ = classify_acceleration_state(snapshots)
    assert verdict == AllJointVerdict.MIXED_ACCELERATION_STATE


def test_all_acceleration_configured():
    snapshots = _all_joints()  # 기본값 acceleration=254 전부
    verdict, reasons = classify_acceleration_state(snapshots)
    assert verdict == AllJointVerdict.ALL_ACCELERATION_CONFIGURED
    assert reasons


def test_read_incomplete_when_one_joint_missing_required_register():
    snapshots = _all_joints()
    snapshots["gripper"] = _snapshot("gripper", acceleration=None, read_errors={"Acceleration": "comm timeout"})
    verdict, reasons = classify_acceleration_state(snapshots)
    assert verdict == AllJointVerdict.READ_INCOMPLETE
    assert any("gripper" in r for r in reasons)


def test_read_incomplete_takes_priority_over_zero_pattern():
    overrides = {j: {"acceleration": 0} for j in JOINT_NAMES}
    snapshots = _all_joints(**overrides)
    snapshots["shoulder_pan"] = _snapshot("shoulder_pan", torque_enable=None)
    verdict, _ = classify_acceleration_state(snapshots)
    assert verdict == AllJointVerdict.READ_INCOMPLETE


# ---------------------------------------------------------------------------
# find_stuck_joints
# ---------------------------------------------------------------------------


def test_find_stuck_joints_detects_wrist_roll_pattern():
    overrides = {"wrist_roll": {"torque_enable": 1, "goal_position_raw": 2024, "present_position_raw": 2023, "moving": 0}}
    snapshots = _all_joints(**overrides)
    stuck = find_stuck_joints(snapshots)
    assert stuck == ("wrist_roll",)


def test_find_stuck_joints_empty_when_goal_equals_present():
    snapshots = _all_joints()  # 기본값: goal==present
    assert find_stuck_joints(snapshots) == ()


def test_find_stuck_joints_ignores_torque_disabled_joints():
    overrides = {"gripper": {"torque_enable": 0, "goal_position_raw": 100, "present_position_raw": 50, "moving": 0}}
    snapshots = _all_joints(**overrides)
    assert find_stuck_joints(snapshots) == ()


def test_find_stuck_joints_ignores_moving_joints():
    overrides = {"elbow_flex": {"goal_position_raw": 100, "present_position_raw": 50, "moving": 1}}
    snapshots = _all_joints(**overrides)
    assert find_stuck_joints(snapshots) == ()


def test_snapshot_to_dict_includes_goal_present_delta():
    snap = _snapshot("wrist_roll", goal_position_raw=2024, present_position_raw=2023)
    d = snap.to_dict()
    assert d["goal_present_delta"] == 1


def test_required_registers_match_installed_control_table_names():
    assert STATE_REGISTERS == ("Torque_Enable", "Operating_Mode", "Goal_Position", "Present_Position", "Moving", "Status")
    assert set(PARAMETER_REGISTERS) == {
        "Acceleration",
        "Maximum_Acceleration",
        "CW_Dead_Zone",
        "CCW_Dead_Zone",
        "Minimum_Startup_Force",
        "Torque_Limit",
    }


# ---------------------------------------------------------------------------
# AllJointParameterDiagnosticInspector + 가짜 FeetechMotorsBus
# ---------------------------------------------------------------------------

pytest.importorskip("lerobot", reason="lerobot이 설치된 환경(~/lerobot venv)에서만 실행")

from hardware.safety.all_joint_parameter_diagnostic import AllJointParameterDiagnosticInspector  # noqa: E402

_FULL_FAKE_CTRL_TABLE = {
    "sts3215": {
        "Torque_Enable": (40, 1),
        "Operating_Mode": (33, 1),
        "Goal_Position": (42, 2),
        "Present_Position": (56, 2),
        "Moving": (66, 1),
        "Status": (65, 1),
        "Acceleration": (41, 1),
        "Maximum_Acceleration": (85, 1),
        "CW_Dead_Zone": (26, 1),
        "CCW_Dead_Zone": (27, 1),
        "Minimum_Startup_Force": (24, 2),
        "Torque_Limit": (48, 2),
    }
}


class _ForbiddenWriteCalled(AssertionError):
    pass


class FakeAllJointBus:
    """FeetechMotorsBus를 대체하는 가짜 버스 - 6개 관절 read만 허용한다."""

    def __init__(self, *, per_joint_values: dict | None = None, ctrl_table: dict | None = None) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls: list[bool] = []
        self.read_calls: list[tuple[str, str]] = []
        self.port = "/dev/null"
        self.model_ctrl_table = ctrl_table if ctrl_table is not None else _FULL_FAKE_CTRL_TABLE
        default_values = {
            "Torque_Enable": 1,
            "Operating_Mode": 0,
            "Goal_Position": 2000,
            "Present_Position": 2000,
            "Moving": 0,
            "Status": 0,
            "Acceleration": 254,
            "Maximum_Acceleration": 254,
            "CW_Dead_Zone": 0,
            "CCW_Dead_Zone": 0,
            "Minimum_Startup_Force": 0,
            "Torque_Limit": 1000,
        }
        self._per_joint_values = per_joint_values or {joint: dict(default_values) for joint in JOINT_NAMES}

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, handshake: bool = True) -> None:
        self.connect_calls += 1
        self.connected = True

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        assert motor in JOINT_NAMES, f"알 수 없는 관절 read 시도: motor={motor}"
        assert data_name in self.model_ctrl_table["sts3215"], f"control table에 없는 레지스터 read 시도: {data_name}"
        self.read_calls.append((data_name, motor))
        values = self._per_joint_values.get(motor, {})
        if data_name not in values:
            raise ConnectionError(f"register not available in fake: {motor}.{data_name}")
        return values[data_name]

    def write(self, *args, **kwargs):
        raise _ForbiddenWriteCalled(f"write() 호출됨: args={args} kwargs={kwargs}")

    def sync_write(self, *args, **kwargs):
        raise _ForbiddenWriteCalled(f"sync_write() 호출됨: args={args} kwargs={kwargs}")

    def enable_torque(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("enable_torque() 호출됨")

    def disable_torque(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("disable_torque() 호출됨")

    def write_calibration(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("write_calibration() 호출됨")

    def set_half_turn_homings(self, *args, **kwargs):
        raise _ForbiddenWriteCalled("set_half_turn_homings() 호출됨")

    def disconnect(self, disable_torque: bool = True) -> None:
        self.disconnect_calls.append(disable_torque)
        self.connected = False


def _build_inspector_with_fake_bus(**bus_kwargs) -> tuple[AllJointParameterDiagnosticInspector, FakeAllJointBus]:
    inspector = AllJointParameterDiagnosticInspector(port="/dev/null", calibration=CALIBRATION)
    fake_bus = FakeAllJointBus(**bus_kwargs)
    inspector._bus = fake_bus  # noqa: SLF001 - 의도적인 테스트용 내부 교체
    return inspector, fake_bus


def test_inspector_rejects_incomplete_calibration():
    incomplete = dict(CALIBRATION)
    del incomplete["gripper"]
    with pytest.raises(ValueError, match="gripper"):
        AllJointParameterDiagnosticInspector(port="/dev/null", calibration=incomplete)


def test_inspector_bus_registers_all_six_motors():
    inspector = AllJointParameterDiagnosticInspector(port="/dev/null", calibration=CALIBRATION)
    assert set(inspector._bus.motors) == set(JOINT_NAMES)  # noqa: SLF001


def test_read_all_snapshots_returns_six_joints():
    inspector, _ = _build_inspector_with_fake_bus()
    inspector.connect()
    snapshots = inspector.read_all_snapshots()
    assert set(snapshots) == set(JOINT_NAMES)
    for joint in JOINT_NAMES:
        assert snapshots[joint].motor_id == CALIBRATION[joint].id


def test_read_all_snapshots_only_reads_known_joints():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_all_snapshots()
    for _data_name, motor in fake_bus.read_calls:
        assert motor in JOINT_NAMES


def test_one_joint_read_failure_produces_none_for_that_register_only():
    values = {joint: {
        "Torque_Enable": 1, "Operating_Mode": 0, "Goal_Position": 2000, "Present_Position": 2000,
        "Moving": 0, "Status": 0, "Acceleration": 254, "Maximum_Acceleration": 254,
        "CW_Dead_Zone": 0, "CCW_Dead_Zone": 0, "Minimum_Startup_Force": 0, "Torque_Limit": 1000,
    } for joint in JOINT_NAMES}
    del values["gripper"]["Acceleration"]  # gripper의 Acceleration read만 실패하도록 유도

    inspector, _ = _build_inspector_with_fake_bus(per_joint_values=values)
    inspector.connect()
    snapshots = inspector.read_all_snapshots()

    assert snapshots["gripper"].acceleration is None
    assert "Acceleration" in snapshots["gripper"].read_errors
    assert snapshots["wrist_roll"].acceleration == 254  # 다른 관절은 영향받지 않는다


def test_read_all_snapshots_skips_registers_not_in_control_table():
    limited_table = {
        "sts3215": {
            "Torque_Enable": (40, 1),
            "Operating_Mode": (33, 1),
            "Goal_Position": (42, 2),
            "Present_Position": (56, 2),
            "Moving": (66, 1),
            "Status": (65, 1),
            "Acceleration": (41, 1),
            "Maximum_Acceleration": (85, 1),
        }
    }
    inspector, fake_bus = _build_inspector_with_fake_bus(ctrl_table=limited_table)
    inspector.connect()
    snapshots = inspector.read_all_snapshots()

    read_names = {name for name, _motor in fake_bus.read_calls}
    assert "CW_Dead_Zone" not in read_names
    assert snapshots["wrist_roll"].cw_dead_zone is None


def test_disconnect_never_writes():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_all_snapshots()
    inspector.disconnect()
    assert fake_bus.disconnect_calls == [False]


def test_full_diagnostic_flow_triggers_zero_writes_and_zero_sync_writes():
    inspector, fake_bus = _build_inspector_with_fake_bus()
    inspector.connect()
    inspector.read_all_snapshots()
    inspector.disconnect()
    # 예외 없이 여기 도달하면 write/sync_write/torque 계열이 한 번도 호출되지 않았다는 뜻이다.
    assert fake_bus.connect_calls == 1


def test_inspector_exposes_only_expected_read_only_interface():
    class_level_attrs = {name for name in dir(AllJointParameterDiagnosticInspector) if not name.startswith("_")}
    assert class_level_attrs == {"is_connected", "connect", "read_all_snapshots", "disconnect"}
