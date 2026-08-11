"""runtime/laptop/safety_gate.py 테스트 - ACCEPT/WOULD_CLAMP/REJECT 3분류 검증."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.action_adapter import AdaptedAction, adapt_vla_action
from runtime.laptop.safety_gate import (
    RANGE_SOURCE_FALLBACK_CONFIG,
    RANGE_SOURCE_GRIPPER_CONVENTION,
    SafetyGate,
    SafetyGateConfig,
    SafetyGateConfigError,
    load_excessive_step_config,
)

RANGE = (-90.0, 90.0)
GRIPPER_RANGE = (0.0, 100.0)
MAX_STEP = 5.0


def _config() -> SafetyGateConfig:
    joint_range = {j: (GRIPPER_RANGE if j == "gripper" else RANGE) for j in JOINT_ORDER}
    max_step = {j: MAX_STEP for j in JOINT_ORDER}
    return SafetyGateConfig(joint_range_deg=joint_range, max_step_deg=max_step)


def _current_state(value: float = 0.0) -> dict[str, float]:
    return {j: value for j in JOINT_ORDER}


def _action(**overrides) -> AdaptedAction:
    command = _current_state(0.0)
    command.update(overrides)
    return AdaptedAction(valid=True, command_deg=command)


def _gate() -> SafetyGate:
    return SafetyGate(_config())


def test_accept_when_within_range_and_small_step() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(shoulder_pan=2.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "ACCEPT"
    assert decision.would_clamp is False
    assert decision.safe_action["shoulder_pan"] == 2.0


def test_would_clamp_on_mild_mechanical_range_violation() -> None:
    # 현재/명령 모두 range 경계 근처 - step 자체는 작게 유지해 rate limit이 아니라
    # mechanical range 위반만 단독으로 관측되게 한다. shoulder_pan만 90도에서 시작하고
    # 나머지 관절은 command/current 둘 다 기본값(0.0)으로 맞춰 다른 관절에서 우발적인
    # excessive-step REJECT가 섞이지 않게 한다.
    current = _current_state(0.0)
    current["shoulder_pan"] = 90.0
    decision = _gate().evaluate(adapted_action=_action(shoulder_pan=92.0), current_state_deg=current, observation_valid=True)
    assert decision.decision == "WOULD_CLAMP"
    assert decision.safe_action["shoulder_pan"] == 90.0
    assert any("MECHANICAL_LIMIT_CLAMPED" in r for r in decision.reasons)


def test_reject_on_gross_mechanical_range_violation() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(shoulder_pan=10_000.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "REJECT"
    assert decision.safe_action is None
    assert any("MECHANICAL_LIMIT_GROSS_VIOLATION" in r for r in decision.reasons)


def test_would_clamp_on_excessive_step() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(wrist_flex=8.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "WOULD_CLAMP"
    assert decision.safe_action["wrist_flex"] == MAX_STEP
    assert any("EXCESSIVE_STEP_CLAMPED" in r for r in decision.reasons)


def test_reject_on_gross_excessive_step() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(wrist_flex=89.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "REJECT"
    assert any("EXCESSIVE_STEP_GROSS" in r for r in decision.reasons)


def test_reject_on_invalid_action_schema() -> None:
    bad_action = AdaptedAction(valid=False, command_deg={}, invalid_reason="차원이 틀렸습니다")
    decision = _gate().evaluate(adapted_action=bad_action, current_state_deg=_current_state(0.0), observation_valid=True)
    assert decision.decision == "REJECT"
    assert decision.safe_action is None
    assert any("ACTION_SCHEMA_INVALID" in r for r in decision.reasons)


def test_reject_on_stale_state() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(),
        current_state_deg=_current_state(0.0),
        observation_valid=True,
        state_stale=True,
        state_stale_reason="너무 오래된 값",
    )
    assert decision.decision == "REJECT"
    assert any("STATE_STALE" in r for r in decision.reasons)


def test_reject_on_invalid_observation() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(),
        current_state_deg=_current_state(0.0),
        observation_valid=False,
        observation_reasons=("wrist 카메라가 없습니다",),
    )
    assert decision.decision == "REJECT"
    assert any("OBSERVATION_INVALID" in r for r in decision.reasons)


def test_reject_on_missing_current_state() -> None:
    decision = _gate().evaluate(adapted_action=_action(), current_state_deg=None, observation_valid=True)
    assert decision.decision == "REJECT"
    assert any("STATE_MISSING" in r for r in decision.reasons)


def test_gripper_semantic_mismatch_mild_clamps_to_percent_range() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(gripper=105.0), current_state_deg=_current_state(0.0) | {"gripper": 100.0},
        observation_valid=True,
    )
    assert decision.decision == "WOULD_CLAMP"
    assert decision.safe_action["gripper"] == 100.0


def test_gripper_semantic_mismatch_gross_is_rejected() -> None:
    decision = _gate().evaluate(
        adapted_action=_action(gripper=99999.0), current_state_deg=_current_state(0.0), observation_valid=True
    )
    assert decision.decision == "REJECT"


# ---------------------------------------------------------------------------
# 하위 호환 - 기존처럼 joint_range_deg/max_step_deg 두 필드만 직접 넘겨도 동작해야 한다.
# ---------------------------------------------------------------------------


def test_direct_construction_still_works_with_default_source_fields() -> None:
    cfg = _config()
    assert cfg.joint_range_source == {}
    assert cfg.uses_calibration_fallback is False
    assert cfg.calibration_file_path is None
    summary = cfg.source_summary()
    assert summary["uses_calibration_fallback"] is False


# ---------------------------------------------------------------------------
# from_repo_defaults() - source tracking (2026-08 Grid35 감사 요구사항)
# ---------------------------------------------------------------------------


def test_from_repo_defaults_reports_calibration_source_per_joint() -> None:
    """gripper는 항상 percent 관례 라벨이어야 하고, 나머지 5관절은 "전부 calibration_file"
    아니면 "전부 fallback_config"로 일관돼야 한다.

    ``configs/follower_safe_mapper.yaml``의 ``calibration_file_path``가 실제로 존재하는
    노트북(실제 chanho_follower 캘리브레이션 파일이 있는 환경, 2026-08 조사로 확인)에서는
    calibration_file이어야 하고, 그 파일이 없는 환경(예: 캘리브레이션 캐시가 없는 CI
    sandbox)에서는 fallback_config여야 한다 - "이 sandbox엔 파일이 없다"고 한쪽만
    하드코딩하지 않고 실제 파일 존재 여부로 분기한다(과거 버전은 fallback만 하드코딩해서
    실제 캘리브레이션 파일이 있는 노트북에서 거짓으로 실패했다)."""
    cfg = SafetyGateConfig.from_repo_defaults()
    assert set(cfg.joint_range_source) == set(JOINT_ORDER)
    assert cfg.joint_range_source["gripper"] == RANGE_SOURCE_GRIPPER_CONVENTION

    file_exists = cfg.calibration_file_path is not None and Path(cfg.calibration_file_path).is_file()
    expected_source = "calibration_file" if file_exists else RANGE_SOURCE_FALLBACK_CONFIG
    for joint in JOINT_ORDER:
        if joint == "gripper":
            continue
        assert cfg.joint_range_source[joint] == expected_source
    assert cfg.uses_calibration_fallback is (not file_exists)


# ---------------------------------------------------------------------------
# 실제 chanho_follower 캘리브레이션 사용 검증 (2026-08 노트북 하드웨어 조사) - mechanical
# hard-limit이 fallback 숫자가 아니라 ~/.cache/huggingface/lerobot/calibration/robots/
# so_follower/chanho_follower.json을 실제로 읽어서 나온 값인지 결정적으로 고정한다.
# ---------------------------------------------------------------------------

REAL_FOLLOWER_CALIBRATION_PATH = Path(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/chanho_follower.json"
).expanduser()


@pytest.mark.skipif(
    not REAL_FOLLOWER_CALIBRATION_PATH.is_file(),
    reason="이 노트북의 실제 chanho_follower 캘리브레이션 파일이 없는 환경(예: CI sandbox)에서는 스킵",
)
def test_uses_real_chanho_follower_calibration_when_present() -> None:
    """실제 캘리브레이션 파일이 있으면 fallback이 아니라 그 파일의 raw tick(예: elbow_flex
    range_min=873/range_max=3084)에서 계산한 degree 값이 mechanical hard-limit이 돼야 한다."""
    cfg = SafetyGateConfig.from_repo_defaults()
    assert cfg.uses_calibration_fallback is False
    assert cfg.calibration_file_path == str(REAL_FOLLOWER_CALIBRATION_PATH)
    for joint in JOINT_ORDER:
        if joint == "gripper":
            continue
        assert cfg.joint_range_source[joint] == "calibration_file"

    # chanho_follower.json raw range (elbow_flex: 873~3084) -> degree 변환 (LeRobot
    # MotorNormMode.DEGREES 공식, follower_safe_mapper.py._ticks_to_degrees와 동일).
    lo, hi = cfg.joint_range_deg["elbow_flex"]
    expected_hi = (3084 - 873) / 2 * 360 / 4095
    assert hi == pytest.approx(expected_hi, abs=1e-6)
    assert lo == pytest.approx(-expected_hi, abs=1e-6)


def test_falls_back_when_calibration_file_missing(tmp_path: Path) -> None:
    """calibration_file_path가 존재하지 않는 파일을 가리키면 fallback_raw_range를 써야
    한다 - 실제 노트북에 캘리브레이션 파일이 있는지 여부와 무관하게, 임시 mapper config로
    "파일 없음" 경로 자체를 결정적으로 검증한다."""
    mapper_yaml = tmp_path / "follower_safe_mapper.yaml"
    mapper_yaml.write_text(
        """
calibration_file_path: "/nonexistent/definitely_missing_chanho_follower.json"
motor_resolution: 4096
fallback_raw_range:
  shoulder_pan: {min: 1070, max: 3135}
  shoulder_lift: {min: 793, max: 3238}
  elbow_flex: {min: 873, max: 3084}
  wrist_flex: {min: 1052, max: 2977}
  wrist_roll: {min: 0, max: 4095}
  gripper: {min: 2047, max: 3496}
leader_to_follower_sign:
  shoulder_pan: 1.0
  shoulder_lift: 1.0
  elbow_flex: 1.0
  wrist_flex: 1.0
  wrist_roll: 1.0
  gripper: 1.0
rate_limit_deg_per_sec:
  shoulder_pan: 20
  shoulder_lift: 15
  elbow_flex: 20
  wrist_flex: 15
  wrist_roll: 25
  gripper: 30
""",
        encoding="utf-8",
    )
    cfg = SafetyGateConfig.from_repo_defaults(follower_safe_mapper_config_path=mapper_yaml)
    assert cfg.uses_calibration_fallback is True
    for joint in JOINT_ORDER:
        if joint == "gripper":
            continue
        assert cfg.joint_range_source[joint] == RANGE_SOURCE_FALLBACK_CONFIG


def test_follower_safe_mapper_config_never_points_at_leader_calibration() -> None:
    """leader 캘리브레이션은 teleop mapping/command envelope 검증용일 뿐 follower
    mechanical hard-limit으로 복사/재사용되면 안 된다 (요구사항 7번) -
    ``configs/follower_safe_mapper.yaml``의 ``calibration_file_path``가 실수로
    ``so_leader`` 캘리브레이션 파일을 가리키는 회귀를 막는다."""
    cfg = SafetyGateConfig.from_repo_defaults()
    assert cfg.calibration_file_path is not None
    assert "so_leader" not in cfg.calibration_file_path
    assert "leader" not in Path(cfg.calibration_file_path).name


def test_real_elbow_flex_present_position_overshoot_is_clamped_not_rejected() -> None:
    """실물 로그(reports/real_follower_staged_safety_test_v1/reports/stage1_1786433719.json)에서
    관측된 실제 사례: elbow_flex 현재 위치가 97.626deg로, chanho_follower.json 기준
    calibrated max(~97.19deg)를 0.4deg 정도 넘겼다. 이는 calibration/fallback 값이 잘못됐다는
    신호가 아니라 캘리브레이션 종료점 근처의 정상적인 encoder/backlash 수준 변동이다 -
    GROSS_RANGE_MULTIPLIER(3x) 경계에는 한참 못 미쳐 REJECT가 아니라 MECHANICAL_LIMIT_CLAMPED로
    안전하게 흡수돼야 한다."""
    cfg = SafetyGateConfig.from_repo_defaults()
    lo, hi = cfg.joint_range_deg["elbow_flex"]
    gate = SafetyGate(cfg)

    current = _current_state(0.0)
    current["elbow_flex"] = hi  # calibrated max 지점에서 시작
    real_overshoot_deg = 97.62637362637362  # 실물 로그 관측값

    decision = gate.evaluate(
        adapted_action=_action(elbow_flex=real_overshoot_deg), current_state_deg=current, observation_valid=True
    )
    assert decision.decision == "WOULD_CLAMP"
    assert any("MECHANICAL_LIMIT_CLAMPED" in r for r in decision.reasons)
    assert decision.safe_action["elbow_flex"] == pytest.approx(hi, abs=1e-6)


def test_from_repo_defaults_no_longer_needs_mujoco_dataset_replay_config() -> None:
    """excessive-step 임계값이 이제 configs/safety_gate.yaml에서 오고, mujoco_so101.yaml
    (다른 실험 dataset에 종속된 진단 도구 설정)과 더 이상 공유되지 않는다."""
    cfg = SafetyGateConfig.from_repo_defaults()
    assert cfg.excessive_step_config_path is not None
    assert Path(cfg.excessive_step_config_path).name == "safety_gate.yaml"
    assert set(cfg.max_step_deg) == set(JOINT_ORDER)


def test_source_summary_matches_documented_shape() -> None:
    cfg = SafetyGateConfig.from_repo_defaults()
    summary = cfg.source_summary()
    assert set(summary) == {
        "joint_range_source",
        "uses_calibration_fallback",
        "calibration_file_path",
        "excessive_step_config_path",
    }


# ---------------------------------------------------------------------------
# load_excessive_step_config
# ---------------------------------------------------------------------------


def test_load_excessive_step_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SafetyGateConfigError):
        load_excessive_step_config(tmp_path / "does_not_exist.yaml")


def test_load_excessive_step_config_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "safety_gate.yaml"
    path.write_text("other_key: 1\n")
    with pytest.raises(SafetyGateConfigError):
        load_excessive_step_config(path)


def test_load_excessive_step_config_missing_joint(tmp_path: Path) -> None:
    path = tmp_path / "safety_gate.yaml"
    incomplete = {j: 1.0 for j in JOINT_ORDER if j != "gripper"}
    path.write_text(json.dumps({"excessive_step_deg": incomplete}))  # YAML superset of JSON
    with pytest.raises(SafetyGateConfigError):
        load_excessive_step_config(path)


def test_load_excessive_step_config_valid(tmp_path: Path) -> None:
    path = tmp_path / "safety_gate.yaml"
    values = {j: float(i) for i, j in enumerate(JOINT_ORDER)}
    path.write_text(json.dumps({"excessive_step_deg": values}))
    loaded = load_excessive_step_config(path)
    assert loaded == values


# ---------------------------------------------------------------------------
# 실제 grid35 checkpoint 010000 REJECT 원인 재현 - 회귀 방지용 (요구사항 4번 분석의 핵심
# 발견을 코드로 고정한다: ground-truth demonstration은 gross 위반이 없어야 한다).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# gripper tiny-negative normalization x Safety Gate 통합 회귀 (2026-08 실물 Stage 3 조사
# 근거 - runtime/laptop/action_adapter.py 모듈 docstring 참고). Action Adapter의 tiny
# negative normalization이 Safety Gate 자체의 판정(mechanical limit/excessive step/gross
# reject)에 영향을 주지 않는지 확인한다 - Safety Gate 코드는 이 요구사항에서 수정되지
# 않았다.
# ---------------------------------------------------------------------------


def test_tiny_negative_gripper_normalized_by_adapter_then_accepted_by_safety_gate() -> None:
    """실물 Stage 3 재현: gripper raw action=-0.025는 adapter에서 0.0으로 정규화되고,
    그 결과 Safety Gate에는 gripper=0.0으로 도달해 (다른 관절이 전부 정상이면) ACCEPT된다 -
    더 이상 gripper 하나 때문에 6관절 전체 write가 막히지 않는다."""
    raw_action = _current_state(0.0)
    raw_action["gripper"] = -0.025
    adapted = adapt_vla_action(raw_action)
    assert adapted.valid is True
    assert adapted.command_deg["gripper"] == 0.0

    decision = _gate().evaluate(adapted_action=adapted, current_state_deg=_current_state(0.0), observation_valid=True)
    assert decision.decision == "ACCEPT"
    assert decision.safe_action["gripper"] == 0.0


def test_moderate_negative_gripper_not_normalized_and_still_would_clamp() -> None:
    """gripper=-0.3(Candidate B negative prediction의 median 근방)은 adapter를 그대로
    통과해 Safety Gate에 -0.3으로 도달해야 하고, Safety Gate는 지금까지와 동일하게
    MECHANICAL_LIMIT_CLAMPED로 WOULD_CLAMP를 반환해야 한다 - tolerance로 삼켜지면 안 된다."""
    raw_action = _current_state(0.0)
    raw_action["gripper"] = -0.3
    adapted = adapt_vla_action(raw_action)
    assert adapted.valid is True
    assert adapted.command_deg["gripper"] == -0.3  # 정규화되지 않고 그대로 유지되어야 함

    decision = _gate().evaluate(adapted_action=adapted, current_state_deg=_current_state(0.0), observation_valid=True)
    assert decision.decision == "WOULD_CLAMP"
    assert any("MECHANICAL_LIMIT_CLAMPED" in r for r in decision.reasons)
    assert decision.safe_action["gripper"] == 0.0


def test_gripper_normalization_does_not_change_arm_excessive_step_behavior() -> None:
    """gripper tiny-negative normalization이 같은 action 벡터 안의 arm excessive-step
    판정에 어떤 영향도 주면 안 된다 - wrist_flex의 EXCESSIVE_STEP_CLAMPED 동작은
    test_would_clamp_on_excessive_step()과 동일하게 유지되어야 한다."""
    raw_action = _current_state(0.0)
    raw_action["wrist_flex"] = 8.0  # test_would_clamp_on_excessive_step()과 동일한 값
    raw_action["gripper"] = -0.025  # 같은 step에 tiny-negative gripper가 섞여 있어도

    adapted = adapt_vla_action(raw_action)
    assert adapted.command_deg["gripper"] == 0.0
    assert adapted.command_deg["wrist_flex"] == 8.0  # arm 관절은 adapter에서 전혀 변경되지 않음

    decision = _gate().evaluate(adapted_action=adapted, current_state_deg=_current_state(0.0), observation_valid=True)
    assert decision.decision == "WOULD_CLAMP"
    assert decision.safe_action["wrist_flex"] == MAX_STEP
    assert any("EXCESSIVE_STEP_CLAMPED" in r for r in decision.reasons)
    # gripper 자체는 정규화된 0.0이 range/step 양쪽 다 문제없어 clamp 사유가 없어야 한다.
    assert decision.per_joint["gripper"].clamped is False
    assert decision.per_joint["gripper"].safe_value == 0.0


# ---------------------------------------------------------------------------
# 2026-08 재캘리브레이션 (Candidate B + 실물 Stage 1/3 근거 - configs/safety_gate.yaml
# 상단 주석 표 참고). shoulder_pan/shoulder_lift/elbow_flex 3개 joint의 excessive-step
# threshold만 갱신됐고, mechanical range/gross reject 구조/wrist_flex 등 나머지 joint는
# 변경되지 않았다 - 이 절의 테스트는 그 두 가지(갱신됨/안 됨)를 모두 회귀로 고정한다.
# ---------------------------------------------------------------------------


def test_recalibrated_thresholds_loaded_from_repo_config() -> None:
    cfg = SafetyGateConfig.from_repo_defaults()
    assert cfg.max_step_deg["shoulder_pan"] == 6.00
    assert cfg.max_step_deg["shoulder_lift"] == 9.00
    assert cfg.max_step_deg["elbow_flex"] == 8.50
    # 변경 근거가 없었던 3개 joint는 그대로 유지되어야 한다.
    assert cfg.max_step_deg["wrist_flex"] == 4.01
    assert cfg.max_step_deg["wrist_roll"] == 1.15
    assert cfg.max_step_deg["gripper"] == 9.17


def _isolated_step(joint: str, before: float, predicted: float) -> dict:
    """다른 5개 joint는 delta 0으로 고정한 채, 지정한 joint 하나만 실물 사례의 before(현재
    state) 값으로 채운다 - 재캘리브레이션 acceptance test용 실물 Stage 1/3 사례 재현 헬퍼.
    action 쪽은 호출부가 ``_action(**{joint: predicted})``로 직접 만든다."""
    state = _current_state(0.0)
    state[joint] = before
    return state


def test_staged_case_A_elbow_flex_7_08deg_now_accepts() -> None:
    """실물 Stage 3: elbow 97.63->90.54 (delta~7.08deg)는 구 threshold(5.73)에 막혔으나,
    combined69 학습 분포 안에서 정상적으로 관측되는 크기(max 9.19deg 이내)다 - 재캘리브레이션
    후에는 ACCEPT되어야 한다."""
    state = _isolated_step("elbow_flex", 97.6264, 90.5416)
    cfg = SafetyGateConfig.from_repo_defaults()
    decision = SafetyGate(cfg).evaluate(
        adapted_action=_action(elbow_flex=90.5416), current_state_deg=state, observation_valid=True
    )
    assert decision.decision == "ACCEPT"


def test_staged_case_B_shoulder_pan_5_45deg_now_accepts() -> None:
    """실물 Stage 1: shoulder_pan +2.68->-2.77 (delta~5.45deg)는 구 threshold(4.58)에
    막혔으나 학습 분포 p99(5.71deg) 이내다 - 재캘리브레이션 후에는 ACCEPT되어야 한다."""
    state = _isolated_step("shoulder_pan", 2.6813, -2.7667)
    cfg = SafetyGateConfig.from_repo_defaults()
    decision = SafetyGate(cfg).evaluate(
        adapted_action=_action(shoulder_pan=-2.7667), current_state_deg=state, observation_valid=True
    )
    assert decision.decision == "ACCEPT"


def test_staged_case_C_shoulder_lift_5_63deg_now_accepts() -> None:
    """실물 Stage 1: shoulder_lift -96.57->-90.94 (delta~5.63deg)는 구 threshold(5.16)에
    막혔으나 학습 분포 p95(6.86deg) 이내다 - 재캘리브레이션 후에는 ACCEPT되어야 한다."""
    state = _isolated_step("shoulder_lift", -96.57, -90.94)
    cfg = SafetyGateConfig.from_repo_defaults()
    decision = SafetyGate(cfg).evaluate(
        adapted_action=_action(shoulder_lift=-90.94), current_state_deg=state, observation_valid=True
    )
    assert decision.decision == "ACCEPT"


def test_staged_case_D_wrist_flex_13_18deg_still_would_clamp() -> None:
    """실물 Stage 1: wrist_flex 53.67->40.49 (delta~13.18deg)는 combined69 학습 데이터
    전체에서 한 번도 관측된 적 없는 magnitude(학습 max 12.53deg 초과)인 true outlier다 -
    재캘리브레이션 후에도 계속 WOULD_CLAMP돼야 한다(wrist_flex threshold는 변경하지 않음)."""
    state = _isolated_step("wrist_flex", 53.6703, 40.4879)
    cfg = SafetyGateConfig.from_repo_defaults()
    decision = SafetyGate(cfg).evaluate(
        adapted_action=_action(wrist_flex=40.4879), current_state_deg=state, observation_valid=True
    )
    assert decision.decision == "WOULD_CLAMP"
    assert any("EXCESSIVE_STEP_CLAMPED" in r for r in decision.reasons)


def test_recalibration_does_not_loosen_gross_reject_for_a_genuinely_gross_elbow_jump() -> None:
    """excessive-step threshold를 올려도 GROSS_STEP_MULTIPLIER(5x, safety_gate.py 상수,
    이번에 변경 안 함)는 그대로이므로, threshold의 5배를 넘는 진짜 gross 위반은 여전히
    REJECT여야 한다 - 재캘리브레이션이 gross reject 구조 자체를 흔들지 않았음을 고정한다."""
    cfg = SafetyGateConfig.from_repo_defaults()
    gate = SafetyGate(cfg)
    new_elbow_threshold = cfg.max_step_deg["elbow_flex"]
    assert new_elbow_threshold == 8.50
    gross_delta = new_elbow_threshold * 5 + 1.0  # 새 GROSS 경계(42.5deg)를 살짝 넘김
    current = _current_state(0.0)
    action = _action(elbow_flex=gross_delta)
    decision = gate.evaluate(adapted_action=action, current_state_deg=current, observation_valid=True)
    assert decision.decision == "REJECT"
    assert any("EXCESSIVE_STEP_GROSS" in r for r in decision.reasons)


def test_repo_default_gross_step_threshold_does_not_flag_typical_demo_sized_step() -> None:
    """excessive-step CLAMP 임계값은 타이트해도 되지만(진단 목적), GROSS(REJECT)는 정상적인
    시연 동작 크기(수 deg~십수 deg)에서는 절대 걸리면 안 된다 - 안 그러면 REJECT가
    "모델이 이상하다"가 아니라 "threshold가 잘못됐다"는 신호가 되어버린다."""
    cfg = SafetyGateConfig.from_repo_defaults()
    gate = SafetyGate(cfg)
    # grid35 checkpoint 010000 held-out 평가에서 실측한 gt_demo_state_delta.max(pooled) 근방.
    typical_large_demo_delta_deg = 16.0
    current = _current_state(0.0)
    action = _action(elbow_flex=typical_large_demo_delta_deg)
    decision = gate.evaluate(adapted_action=action, current_state_deg=current, observation_valid=True)
    assert decision.decision != "REJECT"
