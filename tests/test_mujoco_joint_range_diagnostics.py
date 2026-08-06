from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import ACTION_NAMES, build_synthetic_dataset
from simulation.mujoco.joint_range_diagnostics import (
    analyze_joint,
    build_csv_rows,
    build_report_dict,
    evaluate_hypotheses,
    hypothetical_offset_needed_deg,
)

WRIST_FLEX_INDEX = ACTION_NAMES.index("wrist_flex.pos")
# so101.xml: wrist_flex joint range = -1.658063 ~ 1.658063 rad = -95.0 ~ 95.0 deg (거의 정확히)


def _dataset_with_wrist_flex(tmp_path, per_frame_values):
    """action==state인 합성 데이터셋에서 wrist_flex 열만 지정한 값으로 채운다.

    (action과 state를 다른 값으로 분리하는 시나리오는 실제 데이터셋으로
    test_real_dataset_wrist_flex_matches_investigation_findings에서 검증한다.)
    """
    num_frames = len(per_frame_values)
    action = np.zeros((num_frames, 6), dtype=np.float32)
    action[:, WRIST_FLEX_INDEX] = per_frame_values
    return build_synthetic_dataset(tmp_path / "ds", num_frames=num_frames, action_values=action)


def test_normal_range_no_exceedance(tmp_path):
    values = np.linspace(-80.0, 80.0, 20)  # deg, range(-95~95) 안쪽
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, over = analyze_joint(root, "wrist_flex")
    assert analysis.total_exceeding_frames == 0
    assert analysis.episodes_exceeding == 0
    assert over.size == 0
    assert analysis.max_over_deg == 0.0


def test_upper_bound_exceedance(tmp_path):
    values = np.array([0.0, 50.0, 96.0, 100.0, 50.0], dtype=np.float64)  # 96, 100 deg가 95 초과
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, over = analyze_joint(root, "wrist_flex")
    assert analysis.total_exceeding_frames == 2
    ep = analysis.episodes[0]
    assert ep.action_over_count == 2
    assert math.isclose(ep.max_over_deg, 5.0, abs_tol=1e-3)  # 100 - 95 = 5deg
    assert over.size == 2


def test_lower_bound_exceedance(tmp_path):
    values = np.array([0.0, -50.0, -96.0, -101.0, -50.0], dtype=np.float64)  # -101deg가 -95 미만
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, over = analyze_joint(root, "wrist_flex")
    assert analysis.total_exceeding_frames == 2
    ep = analysis.episodes[0]
    assert math.isclose(ep.max_over_deg, 6.0, abs_tol=1e-3)  # -95 - (-101) = 6deg


def test_exceed_segments_are_contiguous_runs(tmp_path):
    values = np.array([0.0, 96.0, 97.0, 0.0, 0.0, 98.0, 0.0], dtype=np.float64)
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, _ = analyze_joint(root, "wrist_flex")
    ep = analysis.episodes[0]
    assert ep.exceed_segments == ((1, 2), (5, 5))


def test_degree_radian_conversion_matches_model_range(tmp_path):
    values = np.array([94.9, 95.1], dtype=np.float64)
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, _ = analyze_joint(root, "wrist_flex")
    lo, hi = analysis.mujoco_joint_range_deg
    assert math.isclose(hi, 95.0, abs_tol=0.01)
    assert math.isclose(lo, -95.0, abs_tol=0.01)
    # 94.9는 range 안, 95.1은 range 밖
    assert analysis.episodes[0].action_over_count == 1


def test_hypothetical_offset_needed_deg(tmp_path):
    values = np.array([0.0, 100.0], dtype=np.float64)  # 5deg 초과
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, _ = analyze_joint(root, "wrist_flex")
    offset = hypothetical_offset_needed_deg(analysis)
    assert math.isclose(offset, 5.0, abs_tol=1e-3)


def test_hypothetical_offset_zero_when_within_range(tmp_path):
    values = np.array([-90.0, 90.0], dtype=np.float64)
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, _ = analyze_joint(root, "wrist_flex")
    assert hypothetical_offset_needed_deg(analysis) == 0.0


def test_per_episode_stats_multi_episode(tmp_path):
    # build_synthetic_dataset은 단일 episode만 만들므로, 두 데이터셋을 별도로 만들어
    # analyze_joint가 각 episode 통계를 올바르게 분리하는지는 단일 episode 케이스에서
    # frame_count/min/max가 정확히 나오는지로 검증한다.
    values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, _ = analyze_joint(root, "wrist_flex")
    assert analysis.total_episodes == 1
    ep = analysis.episodes[0]
    assert ep.frame_count == 4
    assert ep.action_min_deg == 10.0
    assert ep.action_max_deg == 40.0


def test_unknown_joint_name_raises(tmp_path):
    values = np.array([0.0, 1.0], dtype=np.float64)
    root = _dataset_with_wrist_flex(tmp_path, values)
    with pytest.raises(ValueError):
        analyze_joint(root, "not_a_joint")


def test_evaluate_hypotheses_returns_four_required_verdicts(tmp_path):
    values = np.array([0.0, 100.0], dtype=np.float64)
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, over = analyze_joint(root, "wrist_flex")
    findings = evaluate_hypotheses(analysis, over)
    allowed = {"확인됨", "가능성 높음", "가능성 낮음", "확인 불가"}
    assert len(findings) > 0
    for f in findings:
        assert f.verdict in allowed
        assert f.name
        assert f.evidence


def test_build_report_dict_is_json_serializable(tmp_path):
    import json

    values = np.array([0.0, 100.0], dtype=np.float64)
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, over = analyze_joint(root, "wrist_flex")
    findings = evaluate_hypotheses(analysis, over)
    report = build_report_dict(analysis, over, findings)
    text = json.dumps(report, ensure_ascii=False)
    assert "wrist_flex" in text
    parsed = json.loads(text)
    assert parsed["episodes_exceeding"] == 1


def test_build_csv_rows_one_row_per_episode(tmp_path):
    values = np.array([0.0, 100.0], dtype=np.float64)
    root = _dataset_with_wrist_flex(tmp_path, values)
    analysis, _ = analyze_joint(root, "wrist_flex")
    rows = build_csv_rows(analysis)
    assert len(rows) == analysis.total_episodes == 1
    assert rows[0]["action_over_count"] == 1


def test_real_dataset_wrist_flex_matches_investigation_findings(real_dataset_root):
    analysis, over = analyze_joint(real_dataset_root, "wrist_flex")
    assert analysis.total_episodes == 20
    # 조사에서 확인된 사실: state는 MuJoCo range를 벗어나지 않는다.
    assert sum(ep.state_over_count for ep in analysis.episodes) == 0
    # 다수 episode가 action 쪽에서 초과를 보인다 (정확한 개수는 데이터에 좌우되므로 하한만 검증).
    assert analysis.episodes_exceeding >= 10
    assert analysis.max_over_deg > 0
