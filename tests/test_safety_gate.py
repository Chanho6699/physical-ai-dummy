"""runtime/laptop/safety_gate.py 테스트 - ACCEPT/WOULD_CLAMP/REJECT 3분류 검증."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.action_adapter import AdaptedAction
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
    """실제 캘리브레이션 파일이 없는 환경에서는 gripper를 제외한 5관절이 전부
    fallback_config여야 하고, gripper는 percent 관례 라벨이어야 한다."""
    cfg = SafetyGateConfig.from_repo_defaults()
    assert set(cfg.joint_range_source) == set(JOINT_ORDER)
    assert cfg.joint_range_source["gripper"] == RANGE_SOURCE_GRIPPER_CONVENTION
    for joint in JOINT_ORDER:
        if joint == "gripper":
            continue
        assert cfg.joint_range_source[joint] in (RANGE_SOURCE_FALLBACK_CONFIG, "calibration_file")
    # 이 저장소 sandbox에는 실제 캘리브레이션 파일이 없으므로(조사 결과) fallback이어야 한다.
    assert cfg.uses_calibration_fallback is True


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
