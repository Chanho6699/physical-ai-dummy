"""scripts/evaluate_smolvla_midpoint.py 테스트.

순수 계산(percentile/frame plan/aggregate/markdown) 테스트는 GPU/lerobot 없이도 돌아간다
(``SafetyGate``가 없는 환경에서는 ``try_build_safety_gate``가 ``(None, reason)``으로
degrade하는 것까지 포함). 실제 checkpoint 로딩 + 추론은 ``@pytest.mark.slow``로 표시된
smoke 테스트에서만 하고, 데이터/체크포인트가 없으면 skip한다(``pytest.ini``의 ``slow`` 마커
관례 - ``tests/test_mujoco_action_replay.py`` 등과 동일).

느린 smoke 테스트를 실제로 돌리려면(체크포인트 로딩 + GPU 추론 필요):

    source ~/lerobot/.venv/bin/activate
    pytest tests/test_evaluate_smolvla_midpoint.py -m slow
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.common.vla_contract import JOINT_ORDER  # noqa: E402
import scripts.evaluate_smolvla_midpoint as m  # noqa: E402

REAL_EVAL_DATASET = PROJECT_ROOT / "data" / "so101_cube_xy_midpoint_test10_v1"
REAL_TRAIN_DATASET = PROJECT_ROOT / "data" / "so101_cube_xy_grid35_v1"
REAL_CHECKPOINT_002500 = (
    PROJECT_ROOT
    / "outputs"
    / "grid35"
    / "smolvla_grid35_fresh_v1"
    / "checkpoints"
    / "002500"
    / "pretrained_model"
)


# ---------------------------------------------------------------------------
# percentile / error_stats
# ---------------------------------------------------------------------------


def test_percentile_median_and_p95() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert m.percentile(values, 50) == 3.0
    assert m.percentile(values, 0) == 1.0
    assert m.percentile(values, 100) == 5.0


def test_percentile_single_value() -> None:
    assert m.percentile([7.0], 50) == 7.0


def test_percentile_empty_raises() -> None:
    with pytest.raises(ValueError):
        m.percentile([], 50)


def test_error_stats_empty_returns_none_fields() -> None:
    stats = m.error_stats([])
    assert stats == {"median": None, "p95": None, "max": None, "n": 0}


def test_error_stats_basic() -> None:
    stats = m.error_stats([1.0, 2.0, 3.0])
    assert stats["median"] == 2.0
    assert stats["max"] == 3.0
    assert stats["n"] == 3


def test_mean_or_none() -> None:
    assert m.mean_or_none([]) is None
    assert m.mean_or_none([1.0, 3.0]) == 2.0


# ---------------------------------------------------------------------------
# frame selection - 요구사항 9번 "골고루 포함, 앞부분만 자르지 않음"
# ---------------------------------------------------------------------------


def test_select_frame_local_indices_full_when_unset() -> None:
    assert m.select_frame_local_indices(10, None) == list(range(10))


def test_select_frame_local_indices_full_when_cap_exceeds_length() -> None:
    assert m.select_frame_local_indices(5, 100) == list(range(5))


def test_select_frame_local_indices_spans_whole_episode_not_just_prefix() -> None:
    idxs = m.select_frame_local_indices(100, 5)
    assert idxs[0] == 0
    assert idxs[-1] == 99  # 마지막 프레임까지 포함 - 앞부분만 자르지 않는다
    assert len(idxs) == 5
    assert idxs == sorted(idxs)


def test_select_frame_local_indices_single_frame_cap() -> None:
    assert m.select_frame_local_indices(50, 1) == [0]


def test_select_frame_local_indices_zero_length() -> None:
    assert m.select_frame_local_indices(0, 10) == []


def test_build_frame_plans_covers_all_episodes_in_index_order() -> None:
    episodes_meta = [
        {"episode_index": 2, "length": 10, "dataset_from_index": 20},
        {"episode_index": 0, "length": 10, "dataset_from_index": 0},
        {"episode_index": 1, "length": 10, "dataset_from_index": 10},
    ]
    plans = m.build_frame_plans(episodes_meta)
    assert [p.episode_index for p in plans] == [0, 1, 2]
    assert plans[0].dataset_indices == tuple(range(0, 10))
    assert plans[1].dataset_indices == tuple(range(10, 20))
    assert plans[2].dataset_indices == tuple(range(20, 30))


def test_build_frame_plans_with_cap_still_includes_every_episode() -> None:
    episodes_meta = [{"episode_index": i, "length": 300, "dataset_from_index": i * 300} for i in range(10)]
    plans = m.build_frame_plans(episodes_meta, max_frames_per_episode=3)
    assert len(plans) == 10
    for p in plans:
        assert len(p.dataset_indices) == 3


# ---------------------------------------------------------------------------
# infer_step_from_path / round_floats
# ---------------------------------------------------------------------------


def test_infer_step_from_path_pretrained_model_dir() -> None:
    p = Path("outputs/grid35/smolvla_grid35_fresh_v1/checkpoints/007500/pretrained_model")
    assert m.infer_step_from_path(p) == 7500


def test_infer_step_from_path_bare_numeric_dir() -> None:
    assert m.infer_step_from_path(Path("checkpoints/2500")) == 2500


def test_infer_step_from_path_non_numeric_returns_none() -> None:
    assert m.infer_step_from_path(Path("checkpoints/last/pretrained_model")) is None


def test_round_floats_rounds_nested_structures() -> None:
    data = {"a": 1.123456789, "b": [1.987654321, {"c": 2.0000001}]}
    rounded = m.round_floats(data, ndigits=3)
    assert rounded["a"] == 1.123
    assert rounded["b"][0] == 1.988
    assert rounded["b"][1]["c"] == 2.0


# ---------------------------------------------------------------------------
# tensor_to_joint_dict
# ---------------------------------------------------------------------------


def test_tensor_to_joint_dict_matches_joint_order() -> None:
    import torch

    t = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    result = m.tensor_to_joint_dict(t)
    assert result == dict(zip(JOINT_ORDER, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))


def test_tensor_to_joint_dict_accepts_batch_dim() -> None:
    import torch

    t = torch.zeros(1, 6)
    result = m.tensor_to_joint_dict(t)
    assert set(result.keys()) == set(JOINT_ORDER)


def test_tensor_to_joint_dict_wrong_dim_raises() -> None:
    import torch

    with pytest.raises(ValueError):
        m.tensor_to_joint_dict(torch.zeros(5))


# ---------------------------------------------------------------------------
# SafetyTally
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeJointReport:
    clamped: bool
    rejected: bool


def test_safety_tally_counts_decisions_and_joints() -> None:
    tally = m.SafetyTally()
    per_joint_accept = {j: _FakeJointReport(clamped=False, rejected=False) for j in JOINT_ORDER}
    per_joint_clamp = dict(per_joint_accept)
    per_joint_clamp["gripper"] = _FakeJointReport(clamped=True, rejected=False)

    tally.record("ACCEPT", per_joint_accept)
    tally.record("WOULD_CLAMP", per_joint_clamp)
    tally.record("WOULD_CLAMP", per_joint_clamp)
    tally.record("REJECT", per_joint_accept)

    result = tally.to_dict()
    assert result["WOULD_PASS"] == 1
    assert result["WOULD_CLAMP"] == 2
    assert result["WOULD_REJECT"] == 1
    assert result["joint_clamp_count"]["gripper"] == 2
    assert result["joint_clamp_count"]["shoulder_pan"] == 0


def test_safety_tally_rejects_unknown_decision() -> None:
    tally = m.SafetyTally()
    with pytest.raises(ValueError):
        tally.record("MAYBE", {})


# ---------------------------------------------------------------------------
# aggregate_report
# ---------------------------------------------------------------------------


def _fake_records() -> list[dict]:
    base = dict.fromkeys(JOINT_ORDER, 0.0)
    return [
        {
            "episode_index": 0,
            "frame_index": 0,
            "dataset_row": 0,
            "pred_action": {**base, "shoulder_pan": 2.0},
            "gt_action": {**base, "shoulder_pan": 0.0},
            "state": {**base, "shoulder_pan": 1.0},
        },
        {
            "episode_index": 1,
            "frame_index": 0,
            "dataset_row": 300,
            "pred_action": {**base, "shoulder_pan": 4.0},
            "gt_action": {**base, "shoulder_pan": 0.0},
            "state": {**base, "shoulder_pan": 1.0},
        },
    ]


def test_aggregate_report_computes_mae_and_delta() -> None:
    gt_demo_delta = m.error_stats([0.5, 0.5])
    report = m.aggregate_report(
        checkpoint_dir=Path("outputs/grid35/x/checkpoints/002500/pretrained_model"),
        records=_fake_records(),
        safety_tally=None,
        seed=42,
        elapsed_sec=1.23,
        warnings=[],
        gt_demo_delta=gt_demo_delta,
    )
    assert report["checkpoint_step"] == 2500
    assert report["num_frames_evaluated"] == 2
    assert report["num_episodes_evaluated"] == 2
    # shoulder_pan errors: |2-0|=2, |4-0|=4 -> mean over all 12 joint-errors (2 frames * 6 joints,
    # 나머지 5개 joint는 pred==gt==0이므로 오차 0)
    expected_overall = (2.0 + 4.0) / (2 * len(JOINT_ORDER))
    assert report["metrics"]["action_mae_overall"] == pytest.approx(expected_overall)
    assert report["metrics"]["action_mae_per_joint"]["shoulder_pan"] == pytest.approx(3.0)
    assert report["metrics"]["action_mae_per_joint"]["wrist_roll"] == pytest.approx(0.0)
    assert report["safety"] is None
    assert report["metrics"]["gt_demo_state_delta"] == gt_demo_delta


# ---------------------------------------------------------------------------
# check_train_dataset_provenance / validate_checkpoint_camera_mapping
# ---------------------------------------------------------------------------


def test_check_train_dataset_provenance_matches(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "pretrained_model"
    ckpt_dir.mkdir()
    (ckpt_dir / "train_config.json").write_text(
        json.dumps({"dataset": {"root": "/somewhere/data/so101_cube_xy_grid35_v1"}}), encoding="utf-8"
    )
    warning = m.check_train_dataset_provenance(ckpt_dir, Path("/other/machine/data/so101_cube_xy_grid35_v1"))
    assert warning is None  # basename만 비교하므로 절대경로가 달라도 통과해야 함


def test_check_train_dataset_provenance_mismatch(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "pretrained_model"
    ckpt_dir.mkdir()
    (ckpt_dir / "train_config.json").write_text(
        json.dumps({"dataset": {"root": "/somewhere/data/so101_cube_train_v6"}}), encoding="utf-8"
    )
    warning = m.check_train_dataset_provenance(ckpt_dir, Path("/x/data/so101_cube_xy_grid35_v1"))
    assert warning is not None
    assert "so101_cube_train_v6" in warning


def test_check_train_dataset_provenance_missing_file(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "pretrained_model"
    ckpt_dir.mkdir()
    warning = m.check_train_dataset_provenance(ckpt_dir, Path("/x/data/so101_cube_xy_grid35_v1"))
    assert warning is not None
    assert "train_config.json" in warning


def test_validate_checkpoint_camera_mapping_correct(tmp_path: Path) -> None:
    ckpt_dir = tmp_path
    preprocessor = {
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {
                    "rename_map": {
                        "observation.images.workspace": "observation.images.camera1",
                        "observation.images.wrist": "observation.images.camera2",
                    }
                },
            }
        ]
    }
    (ckpt_dir / "policy_preprocessor.json").write_text(json.dumps(preprocessor), encoding="utf-8")
    warnings = m.validate_checkpoint_camera_mapping(ckpt_dir)
    assert warnings == []


def test_validate_checkpoint_camera_mapping_wrong_mapping(tmp_path: Path) -> None:
    ckpt_dir = tmp_path
    preprocessor = {
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {"observation.images.workspace": "observation.images.camera3"}},
            }
        ]
    }
    (ckpt_dir / "policy_preprocessor.json").write_text(json.dumps(preprocessor), encoding="utf-8")
    warnings = m.validate_checkpoint_camera_mapping(ckpt_dir)
    assert any("카메라 rename 불일치" in w for w in warnings)


def test_validate_checkpoint_camera_mapping_missing_file(tmp_path: Path) -> None:
    warnings = m.validate_checkpoint_camera_mapping(tmp_path)
    assert len(warnings) == 1
    assert "policy_preprocessor.json" in warnings[0]


def test_validate_checkpoint_camera_mapping_no_rename_step(tmp_path: Path) -> None:
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps({"steps": []}), encoding="utf-8")
    warnings = m.validate_checkpoint_camera_mapping(tmp_path)
    assert len(warnings) == 1
    assert "rename_observations_processor" in warnings[0]


# ---------------------------------------------------------------------------
# try_build_safety_gate - 어떤 환경에서도 예외를 던지지 않아야 한다
# ---------------------------------------------------------------------------


def test_try_build_safety_gate_never_raises() -> None:
    gate, reason = m.try_build_safety_gate()
    if gate is None:
        assert isinstance(reason, str) and reason
    else:
        assert reason is None


# ---------------------------------------------------------------------------
# build_summary / render_summary_markdown
# ---------------------------------------------------------------------------


def _fake_report(step: int, mae: float) -> dict:
    gt_demo_delta = {"median": 1.0, "p95": 2.0, "max": 3.0, "n": 10}
    per_joint = dict.fromkeys(JOINT_ORDER, mae)
    return {
        "checkpoint": f"outputs/grid35/x/checkpoints/{step:06d}/pretrained_model",
        "checkpoint_step": step,
        "num_frames_evaluated": 100,
        "num_episodes_evaluated": 10,
        "warnings": [],
        "metrics": {
            "action_mae_overall": mae,
            "action_mae_per_joint": per_joint,
            "pred_state_delta": {"median": 0.5, "p95": 1.5, "max": 4.0, "n": 100},
            "gt_demo_state_delta": gt_demo_delta,
        },
        "safety": {
            "WOULD_PASS": 90,
            "WOULD_CLAMP": 8,
            "WOULD_REJECT": 2,
            "joint_clamp_count": dict.fromkeys(JOINT_ORDER, 1),
            "joint_reject_count": dict.fromkeys(JOINT_ORDER, 0),
        },
    }


def test_build_summary_sorts_by_checkpoint_step() -> None:
    reports = [_fake_report(10000, 5.0), _fake_report(2500, 9.0), _fake_report(5000, 7.0)]
    summary = m.build_summary(
        reports,
        eval_dataset="ds",
        train_dataset="train_ds",
        task="task",
        seed=42,
        safety_gate_available=True,
        safety_gate_unavailable_reason=None,
    )
    assert [row["checkpoint_step"] for row in summary["rows"]] == [2500, 5000, 10000]
    assert summary["rows"][0]["action_mae"] == 9.0


def test_render_summary_markdown_contains_required_columns() -> None:
    reports = [_fake_report(2500, 5.0), _fake_report(5000, 3.0)]
    summary = m.build_summary(
        reports,
        eval_dataset="ds",
        train_dataset="train_ds",
        task="pick",
        seed=42,
        safety_gate_available=True,
        safety_gate_unavailable_reason=None,
    )
    md = m.render_summary_markdown(summary)
    # 요구사항: checkpoint | action MAE | delta median | delta p95 | delta max | WOULD_CLAMP | WOULD_REJECT
    for col in ["checkpoint", "action MAE", "delta median", "delta p95", "delta max", "WOULD_CLAMP", "WOULD_REJECT"]:
        assert col in md
    for joint in JOINT_ORDER:
        assert joint in md
    assert "2500" in md
    assert "5000" in md


def test_render_summary_markdown_without_safety_gate_skips_clamp_table() -> None:
    reports = [_fake_report(2500, 5.0)]
    reports[0]["safety"] = None
    summary = m.build_summary(
        reports,
        eval_dataset="ds",
        train_dataset="train_ds",
        task="pick",
        seed=42,
        safety_gate_available=False,
        safety_gate_unavailable_reason="mujoco 없음",
    )
    md = m.render_summary_markdown(summary)
    assert "mujoco 없음" in md
    assert "Joint별 WOULD_CLAMP count" not in md


# ---------------------------------------------------------------------------
# slow smoke test - 실제 checkpoint + held-out dataset (GPU/lerobot 필요)
# ---------------------------------------------------------------------------


def _skip_if_no_real_assets() -> None:
    if not REAL_EVAL_DATASET.is_dir():
        pytest.skip(f"실제 held-out 데이터셋을 찾을 수 없습니다: {REAL_EVAL_DATASET}")
    if not REAL_CHECKPOINT_002500.is_dir():
        pytest.skip(f"실제 checkpoint를 찾을 수 없습니다: {REAL_CHECKPOINT_002500}")
    if m.SafetyGate is None:
        pytest.skip("safety_gate를 임포트할 수 없습니다 (mujoco 등 의존성 부재) - lerobot venv에서 실행하세요.")


@pytest.mark.slow
def test_smoke_run_checkpoint_on_real_held_out_data() -> None:
    _skip_if_no_real_assets()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=m.DEFAULT_EVAL_REPO_ID, root=str(REAL_EVAL_DATASET))
    episodes_meta = [dataset.meta.episodes[i] for i in range(dataset.num_episodes)]
    # 매우 작은 max_frames_per_episode로 전체 episode를 다 건드리되 실행 시간은 짧게 유지한다
    # (요구사항 9번 커버리지는 build_frame_plans 단위 테스트에서 이미 별도로 검증했다).
    frame_plans = m.build_frame_plans(episodes_meta, max_frames_per_episode=2)
    assert len(frame_plans) == dataset.num_episodes  # 10개 episode 전부 포함됐는지

    gt_demo_delta = m.compute_gt_demo_delta_stats(dataset, frame_plans)
    safety_gate, reason = m.try_build_safety_gate()
    assert safety_gate is not None, reason

    report = m.run_checkpoint(
        REAL_CHECKPOINT_002500,
        dataset,
        frame_plans,
        task=m.DEFAULT_TASK,
        seed=42,
        device=None,
        policy_type="smolvla",
        safety_gate=safety_gate,
        gt_demo_delta=gt_demo_delta,
    )

    expected_frames = sum(len(p.dataset_indices) for p in frame_plans)
    assert report["num_frames_evaluated"] == expected_frames
    assert report["num_episodes_evaluated"] == dataset.num_episodes
    assert report["metrics"]["action_mae_overall"] >= 0.0
    assert set(report["metrics"]["action_mae_per_joint"].keys()) == set(JOINT_ORDER)

    safety = report["safety"]
    assert safety is not None
    assert safety["WOULD_PASS"] + safety["WOULD_CLAMP"] + safety["WOULD_REJECT"] == expected_frames


@pytest.mark.slow
def test_smoke_main_cli_writes_outputs(tmp_path: Path) -> None:
    _skip_if_no_real_assets()
    output_dir = tmp_path / "grid35_midpoint_eval"
    exit_code = m.main(
        [
            "--checkpoints",
            str(REAL_CHECKPOINT_002500),
            "--eval-dataset",
            str(REAL_EVAL_DATASET),
            "--train-dataset",
            str(REAL_TRAIN_DATASET),
            "--max-frames-per-episode",
            "2",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    assert (output_dir / "checkpoint_002500.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.md").is_file()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["rows"]) == 1
    assert summary["rows"][0]["checkpoint_step"] == 2500


def test_main_rejects_eval_dataset_equal_to_train_dataset(tmp_path: Path) -> None:
    same = tmp_path / "same_dataset"
    same.mkdir()
    exit_code = m.main(
        [
            "--eval-dataset",
            str(same),
            "--train-dataset",
            str(same),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 2
