"""data_collection/teleop_safe_range_analysis.py 단위 테스트.

실제 로봇/시리얼 포트에 접근하지 않으며, 실제 data/ 데이터셋에도 의존하지 않는다.
전부 tmp_path 아래에 만든 합성(synthetic) LeRobot v3 형식 fixture만 사용한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_collection.teleop_safe_range_analysis import (
    STATUS_INSUFFICIENT_SAMPLES,
    STATUS_MARGIN_COLLAPSED,
    STATUS_TRUSTED,
    STATUS_UNKNOWN_UNIT,
    AnalysisPolicy,
    TeleopAnalysisError,
    _compute_joint_result,
    analyze,
    compute_joint_delta_diagnostics,
    deduplicate_by_content,
    discover_dataset_roots,
    evaluate_dataset_trust,
    find_calibration_files,
)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
POS_NAMES = [f"{name}.pos" for name in JOINT_NAMES]


def _write_dataset(
    root: Path,
    *,
    state_values: np.ndarray,
    action_values: np.ndarray | None = None,
    episode_index: np.ndarray | None = None,
    state_names: list[str] | None = None,
    robot_type: str = "so_follower",
    codebase_version: str = "v3.0",
    include_state_feature: bool = True,
    task: str = "synthetic",
) -> Path:
    """analyze()가 요구하는 최소 LeRobot v3 데이터셋을 만든다 (읽기 전용 분석 대상).

    ``state_values``와 ``action_values``를 서로 다르게 줄 수 있어, action이 아니라
    observation.state만 쓰이는지(leader vs follower 구분)를 테스트할 수 있다.
    """
    names = state_names or POS_NAMES
    num_frames = state_values.shape[0]
    dim = state_values.shape[1]
    action = action_values if action_values is not None else state_values
    episodes = episode_index if episode_index is not None else np.zeros(num_frames, dtype=np.int64)

    meta_dir = root / "meta"
    (meta_dir / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    features = {"action": {"dtype": "float32", "names": names, "shape": [dim]}}
    if include_state_feature:
        features["observation.state"] = {"dtype": "float32", "names": names, "shape": [dim]}

    unique_episodes = sorted(set(int(e) for e in episodes))
    info = {
        "codebase_version": codebase_version,
        "fps": 30,
        "features": features,
        "total_episodes": len(unique_episodes),
        "total_frames": num_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "robot_type": robot_type,
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    data_df = pd.DataFrame(
        {
            "action": list(action.astype(np.float32)),
            "observation.state": list(state_values.astype(np.float32)),
            "timestamp": np.arange(num_frames, dtype=np.float64) / 30.0,
            "frame_index": np.arange(num_frames, dtype=np.int64),
            "episode_index": episodes,
            "index": np.arange(num_frames, dtype=np.int64),
            "task_index": np.zeros(num_frames, dtype=np.int64),
        }
    )
    if not include_state_feature:
        data_df = data_df.drop(columns=["observation.state"])
    data_df.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")

    lengths = [int(np.sum(episodes == ep)) for ep in unique_episodes]
    episodes_df = pd.DataFrame(
        {
            "episode_index": unique_episodes,
            "tasks": [[task]] * len(unique_episodes),
            "length": lengths,
            "data/chunk_index": [0] * len(unique_episodes),
            "data/file_index": [0] * len(unique_episodes),
        }
    )
    episodes_df.to_parquet(meta_dir / "episodes" / "chunk-000" / "file-000.parquet")
    return root


def _uniform_state(num_frames: int, low: float, high: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # deterministic-ish spread across all 6 joints, same distribution per joint
    # for arms/gripper alike unless overridden by caller.
    base = np.linspace(low, high, num_frames)
    rng.shuffle(base)
    return np.tile(base[:, None], (1, len(JOINT_NAMES)))


DEFAULT_POLICY = AnalysisPolicy(
    lower_percentile=1.0,
    upper_percentile=99.0,
    margin_degree=2.0,
    margin_percent=2.0,
    minimum_samples=50,
)


# ---------------------------------------------------------------------------
# percentile 계산
# ---------------------------------------------------------------------------


def test_percentile_calculation_matches_numpy():
    samples = np.linspace(0.0, 100.0, 1000)
    result = _compute_joint_result(
        "shoulder_pan", samples, nan_count=0, episode_indices=set(), source_files=set(), policy=DEFAULT_POLICY
    )
    assert result.p01 == pytest.approx(np.percentile(samples, 1.0))
    assert result.p05 == pytest.approx(np.percentile(samples, 5.0))
    assert result.p50 == pytest.approx(np.percentile(samples, 50.0))
    assert result.p95 == pytest.approx(np.percentile(samples, 95.0))
    assert result.p99 == pytest.approx(np.percentile(samples, 99.0))


# ---------------------------------------------------------------------------
# margin 적용
# ---------------------------------------------------------------------------


def test_margin_applied_inside_percentile_band():
    samples = np.linspace(0.0, 100.0, 1000)
    policy = AnalysisPolicy(lower_percentile=1.0, upper_percentile=99.0, margin_degree=3.0, margin_percent=3.0, minimum_samples=10)
    result = _compute_joint_result(
        "elbow_flex", samples, nan_count=0, episode_indices=set(), source_files=set(), policy=policy
    )
    assert result.status == STATUS_TRUSTED
    assert result.historical_safe_min == pytest.approx(result.policy_percentile_low + 3.0)
    assert result.historical_safe_max == pytest.approx(result.policy_percentile_high - 3.0)


def test_margin_collapse_when_band_too_narrow():
    # 1.0~1.2 사이의 아주 좁은 범위 - margin 2.0을 적용하면 역전된다.
    samples = np.linspace(1.0, 1.2, 500)
    policy = AnalysisPolicy(lower_percentile=1.0, upper_percentile=99.0, margin_degree=2.0, margin_percent=2.0, minimum_samples=10)
    result = _compute_joint_result(
        "wrist_roll", samples, nan_count=0, episode_indices=set(), source_files=set(), policy=policy
    )
    assert result.status == STATUS_MARGIN_COLLAPSED
    assert result.historical_safe_min is None
    assert result.historical_safe_max is None


def test_gripper_uses_percent_margin_not_degree_margin():
    samples = np.linspace(0.0, 100.0, 1000)
    policy = AnalysisPolicy(lower_percentile=1.0, upper_percentile=99.0, margin_degree=10.0, margin_percent=1.0, minimum_samples=10)
    result = _compute_joint_result(
        "gripper", samples, nan_count=0, episode_indices=set(), source_files=set(), policy=policy
    )
    assert result.unit == "percent_0_100"
    assert result.margin_applied == pytest.approx(1.0)
    assert result.historical_safe_min == pytest.approx(result.policy_percentile_low + 1.0)


# ---------------------------------------------------------------------------
# NaN 제외
# ---------------------------------------------------------------------------


def test_nan_values_excluded_from_stats_and_counted():
    samples = np.concatenate([np.linspace(0.0, 10.0, 100), np.full(5, np.nan)])
    result = _compute_joint_result(
        "wrist_flex", samples, nan_count=5, episode_indices=set(), source_files=set(), policy=DEFAULT_POLICY
    )
    assert result.sample_count == 100
    assert result.nan_count == 5
    assert np.isfinite(result.min) and np.isfinite(result.max)


# ---------------------------------------------------------------------------
# 샘플 부족 거부
# ---------------------------------------------------------------------------


def test_insufficient_samples_rejected():
    samples = np.linspace(0.0, 10.0, 5)
    policy = AnalysisPolicy(minimum_samples=50)
    result = _compute_joint_result(
        "shoulder_lift", samples, nan_count=0, episode_indices=set(), source_files=set(), policy=policy
    )
    assert result.status == STATUS_INSUFFICIENT_SAMPLES
    assert result.historical_safe_min is None
    assert result.historical_safe_max is None


# ---------------------------------------------------------------------------
# unknown unit 거부
# ---------------------------------------------------------------------------


def test_unknown_unit_rejected_for_unmapped_joint_name():
    samples = np.linspace(0.0, 10.0, 200)
    result = _compute_joint_result(
        "wrist_twist_unknown", samples, nan_count=0, episode_indices=set(), source_files=set(), policy=DEFAULT_POLICY
    )
    assert result.status == STATUS_UNKNOWN_UNIT
    assert result.unit is None
    assert result.historical_safe_min is None


def test_dataset_with_unrecognized_joint_name_is_not_trusted(tmp_path):
    names = [f"{n}.pos" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "wrist_twist")]
    state = _uniform_state(60, -10.0, 10.0)
    root = _write_dataset(tmp_path / "ds", state_values=state, state_names=names)
    trust = evaluate_dataset_trust(root, project_root=tmp_path)
    assert trust.trusted is False
    assert "관절" in trust.reason


# ---------------------------------------------------------------------------
# leader-only(=observation.state 없음) 거부 / follower 데이터 승인
# ---------------------------------------------------------------------------


def test_leader_only_dataset_without_observation_state_is_rejected(tmp_path):
    state = _uniform_state(60, -10.0, 10.0)
    root = _write_dataset(tmp_path / "leader_only", state_values=state, include_state_feature=False)
    trust = evaluate_dataset_trust(root, project_root=tmp_path)
    assert trust.trusted is False
    assert "observation.state" in trust.reason


def test_untrusted_robot_type_is_rejected(tmp_path):
    state = _uniform_state(60, -10.0, 10.0)
    root = _write_dataset(tmp_path / "sim_ds", state_values=state, robot_type="mujoco_sim_follower")
    trust = evaluate_dataset_trust(root, project_root=tmp_path)
    assert trust.trusted is False
    assert "robot_type" in trust.reason


def test_follower_data_is_accepted_and_action_is_ignored(tmp_path):
    num_frames = 300
    # follower state가 좁은 범위, leader-derived action은 완전히 다른(훨씬 넓은) 범위 -
    # action이 실제로 쓰였다면 결과 범위가 크게 달라질 것이다.
    state = _uniform_state(num_frames, -10.0, 10.0, seed=1)
    action = _uniform_state(num_frames, 500.0, 900.0, seed=2)
    root = _write_dataset(tmp_path / "ds", state_values=state, action_values=action)

    policy = AnalysisPolicy(minimum_samples=50)
    result, resolved = analyze(
        [root], project_root=tmp_path, default_search_root=tmp_path, policy=policy
    )
    assert resolved == [root.resolve()]
    assert result.trusted is True
    joint = result.joints["shoulder_pan"]
    assert joint.status == STATUS_TRUSTED
    # action 값(500~900)의 영향이 전혀 없어야 한다.
    assert joint.max < 20.0
    assert joint.min > -20.0


def test_action_only_used_when_untraceable_is_never_a_fallback(tmp_path):
    """observation.state가 없는 데이터셋은, action이 있어도 follower 범위 계산에 쓰이지 않는다."""
    num_frames = 60
    action = _uniform_state(num_frames, 0.0, 10.0)
    root = _write_dataset(tmp_path / "ds", state_values=action, action_values=action, include_state_feature=False)
    with pytest.raises(TeleopAnalysisError):
        analyze([root], project_root=tmp_path, default_search_root=tmp_path, policy=DEFAULT_POLICY)


# ---------------------------------------------------------------------------
# 출력 JSON schema
# ---------------------------------------------------------------------------


def test_output_json_schema(tmp_path):
    num_frames = 300
    state = _uniform_state(num_frames, -10.0, 10.0)
    root = _write_dataset(tmp_path / "ds", state_values=state)

    policy = AnalysisPolicy(minimum_samples=50)
    result, _ = analyze([root], project_root=tmp_path, default_search_root=tmp_path, policy=policy)
    payload = result.to_dict()

    for key in (
        "schema_version",
        "generated_at",
        "source_type",
        "trusted",
        "unit",
        "source_summary",
        "analysis_policy",
        "joints",
        "provenance",
    ):
        assert key in payload

    assert payload["source_type"] == "real_follower_teleop"
    for name in JOINT_NAMES:
        joint_payload = payload["joints"][name]
        for key in (
            "status",
            "unit",
            "sample_count",
            "min",
            "max",
            "mean",
            "std",
            "p01",
            "p05",
            "p50",
            "p95",
            "p99",
            "historical_safe_min",
            "historical_safe_max",
        ):
            assert key in joint_payload

    # 결과가 실제로 json.dumps 가능한지(= 순수 파이썬/JSON 타입만 담고 있는지) 확인.
    json.dumps(payload)


# ---------------------------------------------------------------------------
# 원본 파일 미수정
# ---------------------------------------------------------------------------


def test_source_files_are_not_modified(tmp_path):
    num_frames = 200
    state = _uniform_state(num_frames, -10.0, 10.0)
    root = _write_dataset(tmp_path / "ds", state_values=state)

    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    info_file = root / "meta" / "info.json"
    before_data = data_file.read_bytes()
    before_info = info_file.read_bytes()
    before_data_mtime = data_file.stat().st_mtime_ns
    before_info_mtime = info_file.stat().st_mtime_ns

    policy = AnalysisPolicy(minimum_samples=50)
    analyze([root], project_root=tmp_path, default_search_root=tmp_path, policy=policy)

    assert data_file.read_bytes() == before_data
    assert info_file.read_bytes() == before_info
    assert data_file.stat().st_mtime_ns == before_data_mtime
    assert info_file.stat().st_mtime_ns == before_info_mtime


def test_analyze_never_writes_inside_dataset_root_even_when_read_only(tmp_path):
    """데이터셋 디렉터리를 읽기 전용으로 만들어도 analyze()가 성공해야 한다(=쓰기 시도가 없다)."""
    num_frames = 200
    state = _uniform_state(num_frames, -10.0, 10.0)
    root = _write_dataset(tmp_path / "ds", state_values=state)

    import os
    import stat

    def _make_tree_read_only(path: Path) -> None:
        for p in path.rglob("*"):
            os.chmod(p, stat.S_IREAD | stat.S_IEXEC if p.is_dir() else stat.S_IREAD)
        os.chmod(path, stat.S_IREAD | stat.S_IEXEC)

    def _restore_writable(path: Path) -> None:
        for p in path.rglob("*"):
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)

    _make_tree_read_only(root)
    try:
        policy = AnalysisPolicy(minimum_samples=50)
        result, _ = analyze([root], project_root=tmp_path, default_search_root=tmp_path, policy=policy)
        assert result.trusted is True
    finally:
        _restore_writable(root)


# ---------------------------------------------------------------------------
# 여러 데이터셋 통합 / episode·file 집계
# ---------------------------------------------------------------------------


def test_multiple_datasets_pool_samples_and_count_episodes(tmp_path):
    state_a = _uniform_state(150, -5.0, 5.0, seed=10)
    ep_a = np.array([0] * 75 + [1] * 75, dtype=np.int64)
    root_a = _write_dataset(tmp_path / "a", state_values=state_a, episode_index=ep_a)

    state_b = _uniform_state(150, -5.0, 5.0, seed=11)
    root_b = _write_dataset(tmp_path / "b", state_values=state_b)

    policy = AnalysisPolicy(minimum_samples=50)
    result, _ = analyze(
        [root_a, root_b], project_root=tmp_path, default_search_root=tmp_path, policy=policy
    )
    assert result.total_sample_count == 300
    assert result.total_episode_count == 3  # a: episode 0,1 / b: episode 0
    for joint in result.joints.values():
        assert joint.source_file_count == 2


# ---------------------------------------------------------------------------
# 여러 dataset root(자동 탐색) 통합
# ---------------------------------------------------------------------------


def test_discover_dataset_roots_across_multiple_search_roots(tmp_path):
    state = _uniform_state(60, -5.0, 5.0)
    search_a = tmp_path / "search_a"
    search_b = tmp_path / "search_b"
    root_1 = _write_dataset(search_a / "ds1", state_values=state, task="a")
    root_2 = _write_dataset(search_b / "ds2", state_values=state, task="b")

    found = discover_dataset_roots([search_a, search_b])
    assert set(found) == {root_1.resolve(), root_2.resolve()}


def test_discover_dataset_roots_ignores_non_dataset_directory(tmp_path):
    # calibration 전용 디렉터리처럼 meta/info.json이 없는 곳은 후보에 아예 잡히지 않는다.
    calib_dir = tmp_path / "calibration_only"
    calib_dir.mkdir()
    (calib_dir / "chanho_follower.json").write_text("{}", encoding="utf-8")
    assert discover_dataset_roots([calib_dir]) == []


def test_analyze_auto_discovers_across_multiple_search_roots(tmp_path):
    state_a = _uniform_state(150, -5.0, 5.0, seed=20)
    state_b = _uniform_state(150, -5.0, 5.0, seed=21)
    search_a = tmp_path / "search_a"
    search_b = tmp_path / "search_b"
    _write_dataset(search_a / "ds1", state_values=state_a)
    _write_dataset(search_b / "ds2", state_values=state_b)

    policy = AnalysisPolicy(minimum_samples=50)
    result, resolved = analyze(
        None, project_root=tmp_path, default_search_root=[search_a, search_b], policy=policy
    )
    assert len(resolved) == 2
    assert len(result.used_datasets) == 2
    assert result.total_sample_count == 300


# ---------------------------------------------------------------------------
# 동일 데이터셋 복사본 중복 제거
# ---------------------------------------------------------------------------


def test_identical_copy_of_dataset_is_deduplicated_not_summed(tmp_path):
    state = _uniform_state(200, -8.0, 8.0, seed=30)

    root_original = _write_dataset(tmp_path / "original", state_values=state)
    # 완전히 동일한 내용으로 다른 경로에 다시 씀 (예: HF cache 사본 시나리오를 흉내).
    root_copy = _write_dataset(tmp_path / "mirror_copy", state_values=state)

    policy = AnalysisPolicy(minimum_samples=50)
    result, _ = analyze(
        [root_original, root_copy], project_root=tmp_path, default_search_root=tmp_path, policy=policy
    )

    # 200개 프레임짜리 데이터셋 2개(내용 동일)를 줬지만, 실제로는 1개 recording으로 취급되어야 한다.
    assert result.total_sample_count == 200
    assert len(result.used_datasets) == 1
    excluded_reasons = " ".join(d.reason for d in result.excluded_datasets)
    assert "중복" in excluded_reasons


def test_deduplicate_by_content_keeps_lexicographically_first(tmp_path):
    state = _uniform_state(60, -1.0, 1.0)
    root_z = _write_dataset(tmp_path / "z_dataset", state_values=state)
    root_a = _write_dataset(tmp_path / "a_dataset", state_values=state)

    trust_z = evaluate_dataset_trust(root_z, tmp_path)
    trust_a = evaluate_dataset_trust(root_a, tmp_path)
    assert trust_z.content_hash == trust_a.content_hash  # 동일 content

    kept, duplicates = deduplicate_by_content([trust_z, trust_a])
    assert [d.relative_path for d in kept] == ["a_dataset"]
    assert [d.relative_path for d in duplicates] == ["z_dataset"]


def test_deduplicate_by_content_does_not_merge_different_recordings(tmp_path):
    state_1 = _uniform_state(60, -1.0, 1.0, seed=1)
    state_2 = _uniform_state(60, -1.0, 1.0, seed=2)
    root_1 = _write_dataset(tmp_path / "ds1", state_values=state_1)
    root_2 = _write_dataset(tmp_path / "ds2", state_values=state_2)

    trust_1 = evaluate_dataset_trust(root_1, tmp_path)
    trust_2 = evaluate_dataset_trust(root_2, tmp_path)
    assert trust_1.content_hash != trust_2.content_hash

    kept, duplicates = deduplicate_by_content([trust_1, trust_2])
    assert len(kept) == 2
    assert len(duplicates) == 0


# ---------------------------------------------------------------------------
# calibration JSON은 episode 통계에 절대 섞이지 않는다
# ---------------------------------------------------------------------------


def test_calibration_json_is_inventoried_separately_never_merged_into_stats(tmp_path):
    calib_dir = tmp_path / "calibration" / "robots" / "so_follower"
    calib_dir.mkdir(parents=True)
    calibration_payload = {
        "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1686, "range_min": 1070, "range_max": 3135},
        "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 1627, "range_min": 0, "range_max": 4095},
    }
    calib_path = calib_dir / "chanho_follower.json"
    calib_path.write_text(json.dumps(calibration_payload), encoding="utf-8")

    found = find_calibration_files([tmp_path / "calibration"])
    assert len(found) == 1
    assert found[0]["joints"] == ["shoulder_pan", "wrist_roll"]
    assert found[0]["raw_ranges"]["wrist_roll"] == {"range_min": 0, "range_max": 4095}

    # calibration-only 디렉터리는 dataset 후보로 절대 발견되지 않는다 (meta/info.json이 없음).
    assert discover_dataset_roots([tmp_path / "calibration"]) == []

    # analyze()에 calibration_search_roots로 넘겨도 joint 통계(sample_count 등)에는 전혀
    # 영향이 없다 - provenance에만 별도로 나타난다.
    state = _uniform_state(200, -5.0, 5.0)
    dataset_root = _write_dataset(tmp_path / "ds", state_values=state)
    policy = AnalysisPolicy(minimum_samples=50)
    result, _ = analyze(
        [dataset_root],
        project_root=tmp_path,
        default_search_root=tmp_path,
        policy=policy,
        calibration_search_roots=[tmp_path / "calibration"],
    )
    payload = result.to_dict()
    assert payload["provenance"]["calibration_files"], "calibration 인벤토리가 비어 있으면 안 됨"
    assert payload["source_summary"]["sample_count"] == 200  # calibration 값이 섞여 늘어나지 않음
    for joint in result.joints.values():
        assert joint.sample_count in (0, 200)  # gripper 아닌 관절은 200이어야 하고, calibration으로 부풀지 않음


def test_calibration_shaped_json_at_meta_info_json_path_is_still_rejected(tmp_path):
    """calibration JSON을 잘못해서 meta/info.json 자리에 놓아도, features/codebase_version이 없어 거부된다."""
    root = tmp_path / "weird"
    (root / "meta").mkdir(parents=True)
    calibration_payload = {
        "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1686, "range_min": 1070, "range_max": 3135},
    }
    (root / "meta" / "info.json").write_text(json.dumps(calibration_payload), encoding="utf-8")
    trust = evaluate_dataset_trust(root, tmp_path)
    assert trust.trusted is False


# ---------------------------------------------------------------------------
# wrist_roll 범위 확장 검증 (실제 데이터의 MARGIN_COLLAPSED가 파이프라인 버그가 아님을 대조 검증)
# ---------------------------------------------------------------------------


def test_wrist_roll_widens_when_data_actually_covers_a_wide_range(tmp_path):
    """실제 데이터는 wrist_roll이 좁아 MARGIN_COLLAPSED였다 - 여기서는 관절이 실제로 넓게
    쓰였다면 같은 파이프라인이 정상적으로 넓은 안전 범위를 만든다는 것을 대조 검증한다."""
    num_frames = 500
    rng = np.random.default_rng(42)
    state = np.zeros((num_frames, len(JOINT_NAMES)), dtype=np.float64)
    # 다른 관절은 좁게(현실적인 노이즈만), wrist_roll만 넓게(-90~90도) 움직였다고 가정.
    for col in range(len(JOINT_NAMES)):
        state[:, col] = rng.uniform(-1.0, 1.0, size=num_frames)
    wrist_roll_idx = JOINT_NAMES.index("wrist_roll")
    state[:, wrist_roll_idx] = np.linspace(-90.0, 90.0, num_frames)

    root = _write_dataset(tmp_path / "wide_wrist_roll", state_values=state)
    policy = AnalysisPolicy(minimum_samples=50, margin_degree=2.0, margin_percent=2.0)
    result, _ = analyze([root], project_root=tmp_path, default_search_root=tmp_path, policy=policy)

    wrist_roll = result.joints["wrist_roll"]
    assert wrist_roll.status == STATUS_TRUSTED
    assert wrist_roll.historical_safe_min is not None
    assert wrist_roll.historical_safe_max is not None
    assert (wrist_roll.historical_safe_max - wrist_roll.historical_safe_min) > 150.0


def test_joint_delta_diagnostics_detects_barely_moved_joint(tmp_path):
    """실제 wrist_roll처럼 '거의 안 움직인' 관절의 delta 진단이 그 사실을 정확히 드러내는지 확인."""
    num_frames = 300
    state = np.zeros((num_frames, len(JOINT_NAMES)), dtype=np.float64)
    wrist_roll_idx = JOINT_NAMES.index("wrist_roll")
    # 실제 데이터와 비슷하게: 거의 -2.6도 근방에서 아주 작은 지터만 있음.
    rng = np.random.default_rng(7)
    state[:, wrist_roll_idx] = -2.6 + rng.uniform(-0.05, 0.05, size=num_frames)

    root = _write_dataset(tmp_path / "barely_moved", state_values=state)
    trust = evaluate_dataset_trust(root, tmp_path)
    diag = compute_joint_delta_diagnostics(trust, "wrist_roll")

    assert diag["frame_count"] == num_frames
    assert diag["max_abs_consecutive_delta"] < 0.2
    assert diag["frac_abs_delta_ge_1_0"] == 0.0
    assert (diag["max"] - diag["min"]) < 0.2


def test_joint_delta_diagnostics_detects_actively_moved_joint(tmp_path):
    # step size = 180 / (num_frames - 1); 90개 프레임이면 step ≈ 2.02도로 1도 임계값을 확실히 넘는다.
    num_frames = 90
    state = np.zeros((num_frames, len(JOINT_NAMES)), dtype=np.float64)
    wrist_roll_idx = JOINT_NAMES.index("wrist_roll")
    state[:, wrist_roll_idx] = np.linspace(-90.0, 90.0, num_frames)

    root = _write_dataset(tmp_path / "actively_moved", state_values=state)
    trust = evaluate_dataset_trust(root, tmp_path)
    diag = compute_joint_delta_diagnostics(trust, "wrist_roll")

    assert diag["frac_abs_delta_ge_1_0"] == pytest.approx(1.0)
    assert diag["unique_value_count"] == num_frames


def test_joint_delta_diagnostics_never_diffs_across_episode_boundary(tmp_path):
    """에피소드 경계를 넘어서는 (진짜 움직임이 아닌) 점프를 delta로 잘못 세지 않는지 확인."""
    num_frames = 100
    state = np.zeros((num_frames, len(JOINT_NAMES)), dtype=np.float64)
    wrist_roll_idx = JOINT_NAMES.index("wrist_roll")
    # 에피소드 0: 전부 0.0. 에피소드 1: 전부 50.0 (에피소드 사이에서만 큰 점프).
    state[:50, wrist_roll_idx] = 0.0
    state[50:, wrist_roll_idx] = 50.0
    episode_index = np.array([0] * 50 + [1] * 50, dtype=np.int64)

    root = _write_dataset(tmp_path / "episode_boundary", state_values=state, episode_index=episode_index)
    trust = evaluate_dataset_trust(root, tmp_path)
    diag = compute_joint_delta_diagnostics(trust, "wrist_roll")

    # 에피소드 내부에서는 전혀 움직이지 않았으므로 delta는 전부 0이어야 한다.
    assert diag["max_abs_consecutive_delta"] == 0.0


# ---------------------------------------------------------------------------
# calibration 파일도 원본 미수정 (find_calibration_files는 읽기 전용)
# ---------------------------------------------------------------------------


def test_find_calibration_files_does_not_modify_source(tmp_path):
    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    calib_path = calib_dir / "chanho_follower.json"
    payload = {
        "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 1627, "range_min": 0, "range_max": 4095},
    }
    calib_path.write_text(json.dumps(payload), encoding="utf-8")
    before_bytes = calib_path.read_bytes()
    before_mtime = calib_path.stat().st_mtime_ns

    find_calibration_files([calib_dir])
    find_calibration_files([calib_dir])  # 두 번 호출해도 동일하게 읽기 전용

    assert calib_path.read_bytes() == before_bytes
    assert calib_path.stat().st_mtime_ns == before_mtime
