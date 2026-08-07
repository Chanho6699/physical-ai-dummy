"""simulation/mujoco/follower_safe_mapper.py 단위 테스트.

전부 명시적 ``now``를 넘겨 결정적으로(real sleep 없이) 검증한다.
"""

from __future__ import annotations

import json
import math

import pytest

from simulation.mujoco.follower_safe_mapper import (
    ARM_WIDE_HOLD_REASONS,
    ConnectionHoldInput,
    FollowerSafeMapper,
    FollowerSafeMapperConfig,
    FollowerSafeMapperError,
    load_config_yaml,
    load_follower_calibration,
    print_follower_safe_debug,
    summarize_hold,
)
from simulation.mujoco.remote_state_client import JOINT_NAMES

FALLBACK_RAW_RANGE = {
    "shoulder_pan": {"min": 1070, "max": 3135},
    "shoulder_lift": {"min": 793, "max": 3238},
    "elbow_flex": {"min": 873, "max": 3084},
    "wrist_flex": {"min": 1052, "max": 2977},
    "wrist_roll": {"min": 0, "max": 4095},
    "gripper": {"min": 2047, "max": 3496},
}

RATE = {name: 20.0 for name in JOINT_NAMES}
T0 = 1_700_000_000.0


def _positions(**overrides) -> dict[str, float]:
    base = {name: 0.0 for name in JOINT_NAMES}
    base.update(overrides)
    return base


def _default_calibrations():
    return load_follower_calibration(calibration_file_path=None, fallback_raw_range=FALLBACK_RAW_RANGE, motor_resolution=4096)


def _mapper(**rate_overrides) -> FollowerSafeMapper:
    rate = dict(RATE)
    rate.update(rate_overrides)
    config = FollowerSafeMapperConfig(fallback_raw_range=FALLBACK_RAW_RANGE, rate_limit_deg_per_sec=rate)
    return FollowerSafeMapper(config, _default_calibrations())


# ---------------------------------------------------------------------------
# 캘리브레이션 로딩
# ---------------------------------------------------------------------------


def test_fallback_calibration_computes_degrees_matching_lerobot_formula():
    cals = _default_calibrations()
    # shoulder_pan: mid=(1070+3135)/2=2102.5, max_res=4095 -> range = ±(3135-1070)/2*360/4095
    expected_half = (3135 - 1070) / 2 * 360 / 4095
    cal = cals["shoulder_pan"]
    assert cal.verified is True
    assert cal.range_min_deg == pytest.approx(-expected_half, abs=1e-6)
    assert cal.range_max_deg == pytest.approx(expected_half, abs=1e-6)
    assert cal.source == "fallback_config"


def test_gripper_is_unverified_and_excluded():
    cals = _default_calibrations()
    cal = cals["gripper"]
    assert cal.verified is False
    assert cal.range_min_deg is None
    assert cal.range_max_deg is None


def test_wrist_roll_full_range_is_plus_minus_180():
    cals = _default_calibrations()
    cal = cals["wrist_roll"]
    assert cal.range_min_deg == pytest.approx(-180.0, abs=0.01)
    assert cal.range_max_deg == pytest.approx(180.0, abs=0.01)


def test_calibration_file_takes_priority_over_fallback(tmp_path):
    cal_file = tmp_path / "chanho_follower.json"
    entries = {
        joint: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1000 + i, "range_max": 3000 + i}
        for i, joint in enumerate(JOINT_NAMES)
    }
    cal_file.write_text(json.dumps(entries), encoding="utf-8")

    cals = load_follower_calibration(calibration_file_path=cal_file, fallback_raw_range=FALLBACK_RAW_RANGE, motor_resolution=4096)
    # 파일의 range_min=1000, range_max=3000(shoulder_pan)을 썼는지 확인 (fallback의 1070/3135가 아님).
    expected_half = (3000 - 1000) / 2 * 360 / 4095
    assert cals["shoulder_pan"].range_max_deg == pytest.approx(expected_half, abs=1e-6)


def test_missing_file_and_no_fallback_raises():
    with pytest.raises(FollowerSafeMapperError):
        load_follower_calibration(calibration_file_path="/nonexistent/path.json", fallback_raw_range=None, motor_resolution=4096)


def test_rate_limit_config_requires_all_joints():
    with pytest.raises(FollowerSafeMapperError):
        FollowerSafeMapperConfig(fallback_raw_range=FALLBACK_RAW_RANGE, rate_limit_deg_per_sec={"wrist_flex": 10.0})


def test_load_config_yaml_reads_real_file():
    config = load_config_yaml("configs/follower_safe_mapper.yaml")
    assert config.motor_resolution == 4096
    assert set(config.rate_limit_deg_per_sec.keys()) == set(JOINT_NAMES)
    assert all(v == 1.0 for v in config.leader_to_follower_sign.values())


# ---------------------------------------------------------------------------
# 초기화 (첫 snapshot, follower current 우선)
# ---------------------------------------------------------------------------


def test_first_snapshot_initializes_without_jump():
    mapper = _mapper()
    results = mapper.step(
        now=T0,
        leader_positions_deg=_positions(wrist_flex=50.0),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(),
    )
    # 첫 프레임(elapsed=0)이므로 리더가 50도여도 즉시 50도로 점프하지 않는다 -
    # follower_current가 없으면 리더의 첫 값 자체를 초기 command로 삼는다("점프할 이전 값"이
    # 없는 프레임 0 이므로 이 자체는 점프가 아님).
    wf = results["wrist_flex"]
    assert wf.limited_command_deg == pytest.approx(50.0)
    assert wf.hold is False


def test_follower_current_used_as_initial_command_when_available():
    mapper = _mapper()
    results = mapper.step(
        now=T0,
        leader_positions_deg=_positions(wrist_flex=50.0),
        follower_positions_deg=_positions(wrist_flex=5.0),
        connection_hold=ConnectionHoldInput(),
    )
    wf = results["wrist_flex"]
    # follower_current(5.0)을 초기 command로 써야 한다 - 리더(50.0)로 즉시 점프하지 않음.
    assert wf.limited_command_deg == pytest.approx(5.0)
    assert wf.mapped_follower_deg == pytest.approx(50.0)  # B단계(좌표 변환) 자체는 그대로 기록됨


# ---------------------------------------------------------------------------
# rate limit
# ---------------------------------------------------------------------------


def test_fast_leader_jump_is_rate_limited():
    mapper = _mapper(wrist_flex=20.0)
    mapper.step(now=T0, leader_positions_deg=_positions(), follower_positions_deg=_positions(), connection_hold=ConnectionHoldInput())

    results = mapper.step(
        now=T0 + 0.1,  # 0.1초 경과, rate=20deg/s -> max_step=2deg
        leader_positions_deg=_positions(wrist_flex=80.0),  # 순간 점프 요청
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(),
    )
    wf = results["wrist_flex"]
    assert wf.rate_limited is True
    # T0가 1.7e9 규모라 float64 정밀도상 1e-6 오차는 나올 수 있다 - 느슨하게 비교한다.
    assert wf.max_step_deg == pytest.approx(2.0, abs=1e-3)
    assert wf.limited_command_deg == pytest.approx(2.0, abs=1e-3)  # 0 -> 2 (80이 아님)
    assert wf.limited_command_deg < 80.0
    assert "RATE_LIMITED" in wf.intervention_flags


def test_slow_leader_change_passes_through_unlimited():
    mapper = _mapper(wrist_flex=20.0)
    mapper.step(now=T0, leader_positions_deg=_positions(), follower_positions_deg=_positions(), connection_hold=ConnectionHoldInput())

    results = mapper.step(
        now=T0 + 0.5,  # max_step = 20*0.5 = 10deg
        leader_positions_deg=_positions(wrist_flex=3.0),  # 3deg 변화 - 한도 안
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(),
    )
    wf = results["wrist_flex"]
    assert wf.rate_limited is False
    assert wf.limited_command_deg == pytest.approx(3.0, abs=1e-6)
    assert "RATE_LIMITED" not in wf.intervention_flags


def test_max_step_scales_with_elapsed_time():
    mapper = _mapper(wrist_flex=15.0)
    mapper.step(now=T0, leader_positions_deg=_positions(), follower_positions_deg=_positions(), connection_hold=ConnectionHoldInput())

    r1 = mapper.step(now=T0 + 0.2, leader_positions_deg=_positions(wrist_flex=100.0), follower_positions_deg=None, connection_hold=ConnectionHoldInput())
    assert r1["wrist_flex"].max_step_deg == pytest.approx(15.0 * 0.2, abs=1e-6)

    r2 = mapper.step(now=T0 + 0.2 + 1.0, leader_positions_deg=_positions(wrist_flex=100.0), follower_positions_deg=None, connection_hold=ConnectionHoldInput())
    assert r2["wrist_flex"].max_step_deg == pytest.approx(15.0 * 1.0, abs=1e-6)


def test_gradual_rate_limited_convergence_reaches_target_eventually():
    mapper = _mapper(wrist_flex=20.0)
    mapper.step(now=T0, leader_positions_deg=_positions(), follower_positions_deg=_positions(), connection_hold=ConnectionHoldInput())

    now = T0
    last = None
    for _ in range(50):
        now += 0.1
        results = mapper.step(now=now, leader_positions_deg=_positions(wrist_flex=80.0), follower_positions_deg=None, connection_hold=ConnectionHoldInput())
        last = results["wrist_flex"].limited_command_deg
    assert last == pytest.approx(80.0, abs=0.1)  # 충분한 시간 후 결국 목표에 도달


# ---------------------------------------------------------------------------
# hold: stale / sequence stalled / connection / mode
# ---------------------------------------------------------------------------


def test_stale_holds_command():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(wrist_flex=10.0), follower_positions_deg=_positions(wrist_flex=10.0), connection_hold=ConnectionHoldInput())

    results = mapper.step(
        now=T0 + 0.5,
        leader_positions_deg=_positions(wrist_flex=90.0),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(remote_stale=True),
    )
    wf = results["wrist_flex"]
    assert wf.hold is True
    assert wf.hold_reason == "REMOTE_STALE"
    assert wf.limited_command_deg == pytest.approx(10.0)  # 직전 값 유지, 90으로 안 감


def test_sequence_stalled_holds_command():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(wrist_flex=10.0), follower_positions_deg=_positions(wrist_flex=10.0), connection_hold=ConnectionHoldInput())

    results = mapper.step(
        now=T0 + 0.5,
        leader_positions_deg=_positions(wrist_flex=90.0),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(sequence_stalled=True),
    )
    wf = results["wrist_flex"]
    assert wf.hold is True
    assert wf.hold_reason == "SEQUENCE_STALLED"
    assert wf.limited_command_deg == pytest.approx(10.0)


def test_mode_not_read_only_holds_command():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(wrist_flex=10.0), follower_positions_deg=_positions(wrist_flex=10.0), connection_hold=ConnectionHoldInput())
    results = mapper.step(
        now=T0 + 0.5,
        leader_positions_deg=_positions(wrist_flex=90.0),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(mode_not_read_only=True),
    )
    assert results["wrist_flex"].hold_reason == "MODE_NOT_READ_ONLY"


def test_connection_lost_holds_command():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(wrist_flex=10.0), follower_positions_deg=_positions(wrist_flex=10.0), connection_hold=ConnectionHoldInput())
    results = mapper.step(
        now=T0 + 0.5,
        leader_positions_deg=_positions(wrist_flex=90.0),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(connection_lost=True),
    )
    assert results["wrist_flex"].hold_reason == "CONNECTION_LOST"
    assert results["wrist_flex"].connection_held is True


def test_resume_after_hold_converges_smoothly_without_jump():
    # wrist_flex의 실제 캘리브레이션 안전 range는 약 ±84.6도(1052~2977 tick) - 90도는 그
    # 범위를 넘어가 RANGE_VIOLATION이 되므로, 이 테스트는 range 안(50도)의 목표를 쓴다.
    mapper = _mapper(wrist_flex=20.0)
    mapper.step(now=T0, leader_positions_deg=_positions(wrist_flex=10.0), follower_positions_deg=_positions(wrist_flex=10.0), connection_hold=ConnectionHoldInput())

    # 1초간 stale (리더는 계속 50도를 요청하지만 hold)
    for i in range(5):
        mapper.step(
            now=T0 + 0.2 * (i + 1),
            leader_positions_deg=_positions(wrist_flex=50.0),
            follower_positions_deg=None,
            connection_hold=ConnectionHoldInput(remote_stale=True),
        )
    held_value = mapper.last_command_snapshot()["wrist_flex"]
    assert held_value == pytest.approx(10.0)  # 계속 hold라 안 움직임

    # 복구 - 다음 프레임에 50으로 즉시 점프하면 안 되고, rate limit만큼만 움직여야 한다.
    results = mapper.step(
        now=T0 + 1.0 + 0.1,  # 마지막 hold 프레임(T0+1.0) 대비 0.1초 경과 -> max_step=2deg
        leader_positions_deg=_positions(wrist_flex=50.0),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(),  # hold 해제
    )
    wf = results["wrist_flex"]
    assert wf.hold is False
    assert wf.limited_command_deg == pytest.approx(12.0, abs=1e-3)  # 10 -> 12 (50 아님)
    assert wf.limited_command_deg < 50.0


# ---------------------------------------------------------------------------
# range 정책 (clamp 아니라 hold)
# ---------------------------------------------------------------------------


def test_range_violation_holds_instead_of_clamping():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(wrist_flex=0.0), follower_positions_deg=_positions(wrist_flex=0.0), connection_hold=ConnectionHoldInput())

    cal = mapper.calibrations["wrist_flex"]
    out_of_range = cal.range_max_deg + 20.0  # 명백히 범위 초과

    results = mapper.step(
        now=T0 + 1.0,
        leader_positions_deg=_positions(wrist_flex=out_of_range),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(),
    )
    wf = results["wrist_flex"]
    assert wf.hold is True
    assert wf.hold_reason == "RANGE_VIOLATION"
    assert wf.range_held is True
    # clamp라면 range_max_deg 근처 값이 나와야 하지만, hold이므로 직전 안전값(0.0)이어야 한다.
    assert wf.limited_command_deg == pytest.approx(0.0)
    assert wf.limited_command_deg != pytest.approx(cal.range_max_deg, abs=1.0)


def test_unverified_joint_always_held_and_excluded():
    mapper = _mapper()
    results = mapper.step(
        now=T0,
        leader_positions_deg=_positions(gripper=50.0),
        follower_positions_deg=_positions(gripper=10.0),
        connection_hold=ConnectionHoldInput(),
    )
    g = results["gripper"]
    assert g.hold is True
    assert g.hold_reason == "UNVERIFIED_RANGE"
    assert g.range_min_deg is None and g.range_max_deg is None


# ---------------------------------------------------------------------------
# NaN/Inf
# ---------------------------------------------------------------------------


def test_nan_leader_value_is_held_not_crashed():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(wrist_flex=0.0), follower_positions_deg=_positions(wrist_flex=0.0), connection_hold=ConnectionHoldInput())
    results = mapper.step(
        now=T0 + 0.5,
        leader_positions_deg=_positions(wrist_flex=float("nan")),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(),
    )
    wf = results["wrist_flex"]
    assert wf.hold is True
    assert wf.hold_reason == "INVALID_VALUE"
    assert wf.limited_command_deg == pytest.approx(0.0)


def test_inf_leader_value_is_held_not_crashed():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(wrist_flex=0.0), follower_positions_deg=_positions(wrist_flex=0.0), connection_hold=ConnectionHoldInput())
    results = mapper.step(
        now=T0 + 0.5,
        leader_positions_deg=_positions(wrist_flex=float("inf")),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(),
    )
    wf = results["wrist_flex"]
    assert wf.hold is True
    assert wf.hold_reason == "INVALID_VALUE"


def test_missing_joint_in_leader_state_is_held():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(wrist_flex=0.0), follower_positions_deg=_positions(wrist_flex=0.0), connection_hold=ConnectionHoldInput())
    incomplete = _positions()
    del incomplete["wrist_flex"]
    results = mapper.step(now=T0 + 0.5, leader_positions_deg=incomplete, follower_positions_deg=None, connection_hold=ConnectionHoldInput())
    wf = results["wrist_flex"]
    assert wf.hold is True
    assert wf.hold_reason == "MISSING_JOINT"


# ---------------------------------------------------------------------------
# intervention count
# ---------------------------------------------------------------------------


def test_intervention_count_increments_on_hold_and_rate_limit():
    mapper = _mapper(wrist_flex=5.0)
    mapper.step(now=T0, leader_positions_deg=_positions(), follower_positions_deg=_positions(), connection_hold=ConnectionHoldInput())
    # gripper는 항상 UNVERIFIED_RANGE라 첫 프레임부터 이미 1건 잡힌다 (관절별로 카운트).
    baseline = mapper.intervention_count
    assert baseline >= 1

    mapper.step(now=T0 + 0.1, leader_positions_deg=_positions(wrist_flex=50.0), follower_positions_deg=None, connection_hold=ConnectionHoldInput())
    assert mapper.intervention_count > baseline  # wrist_flex의 rate limit 개입이 추가로 잡힘


# ---------------------------------------------------------------------------
# summarize_hold / global vs joint hold 분리 (실물 재현 이슈 조사)
#
# 보고된 증상: "gripper가 UNVERIFIED_RANGE로 hold되면 팔 5개 관절도 안 움직인다."
# 코드 감사 + 재현 결과: _step_joint()는 관절마다 완전히 독립적으로 계산되고(공유 mutable
# 상태 없음, self._calibrations[joint]/self._initialized[joint]만 참조), live_web_viewer.py의
# data.ctrl 적용 루프도 매핑 엔트리(관절)별로 개별 판단한다 - 구조적으로 한 관절의 hold가
# 다른 관절에 전파될 방법이 없다. 아래 테스트들은 그 사실을 직접 검증한다.
# ---------------------------------------------------------------------------


def test_arm_wide_hold_reasons_are_exactly_the_four_connection_reasons():
    assert ARM_WIDE_HOLD_REASONS == {"REMOTE_STALE", "SEQUENCE_STALLED", "MODE_NOT_READ_ONLY", "CONNECTION_LOST"}
    # UNVERIFIED_RANGE/RANGE_VIOLATION/MISSING_JOINT/INVALID_VALUE/NOT_INITIALIZED는 절대
    # 포함되면 안 된다 - 이게 포함되면 "관절 하나의 문제가 전역 hold가 된다"는 버그가 생긴다.
    assert "UNVERIFIED_RANGE" not in ARM_WIDE_HOLD_REASONS
    assert "RANGE_VIOLATION" not in ARM_WIDE_HOLD_REASONS
    assert "MISSING_JOINT" not in ARM_WIDE_HOLD_REASONS
    assert "INVALID_VALUE" not in ARM_WIDE_HOLD_REASONS
    assert "NOT_INITIALIZED" not in ARM_WIDE_HOLD_REASONS


def test_summarize_hold_gripper_only_reports_5_active_1_held():
    mapper = _mapper()
    results = mapper.step(
        now=T0,
        leader_positions_deg=_positions(gripper=45.0),
        follower_positions_deg=_positions(),
        connection_hold=ConnectionHoldInput(),
    )
    summary = summarize_hold(results)
    assert summary["global_hold"] is False
    assert summary["global_hold_reason"] is None
    assert summary["active_joint_count"] == 5
    assert summary["held_joint_count"] == 1
    assert summary["held_joints"] == {"gripper": "UNVERIFIED_RANGE"}


def test_summarize_hold_range_violation_does_not_trigger_global_hold():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(), follower_positions_deg=_positions(), connection_hold=ConnectionHoldInput())
    cal = mapper.calibrations["wrist_flex"]
    results = mapper.step(
        now=T0 + 1.0,
        leader_positions_deg=_positions(wrist_flex=cal.range_max_deg + 20.0),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(),
    )
    summary = summarize_hold(results)
    assert summary["global_hold"] is False  # 관절 하나의 RANGE_VIOLATION은 전역이 아니다
    assert summary["held_joints"].get("wrist_flex") == "RANGE_VIOLATION"
    assert summary["held_joint_count"] == 2  # wrist_flex + 항상 held인 gripper


def test_summarize_hold_remote_stale_is_global():
    mapper = _mapper()
    mapper.step(now=T0, leader_positions_deg=_positions(), follower_positions_deg=_positions(), connection_hold=ConnectionHoldInput())
    results = mapper.step(
        now=T0 + 0.5,
        leader_positions_deg=_positions(wrist_flex=10.0),
        follower_positions_deg=None,
        connection_hold=ConnectionHoldInput(remote_stale=True),
    )
    summary = summarize_hold(results)
    assert summary["global_hold"] is True
    assert summary["global_hold_reason"] == "REMOTE_STALE"
    assert summary["held_joint_count"] == 6  # 6관절 전부 hold (gripper 포함)
    assert summary["active_joint_count"] == 0


def test_print_follower_safe_debug_smoke(capsys):
    mapper = _mapper()
    results = mapper.step(
        now=T0, leader_positions_deg=_positions(gripper=45.0), follower_positions_deg=_positions(), connection_hold=ConnectionHoldInput()
    )
    print_follower_safe_debug(results)
    out = capsys.readouterr().out
    assert "[Follower-safe 진단]" in out
    assert "gripper" in out and "UNVERIFIED_RANGE" in out
    assert "global_hold=False" in out
    assert "active_joint_count=5" in out
