"""scripts/generate_control_profile_candidate.py CLI 테스트 - 전부 fake JSON/YAML, offline.

lerobot/serial 접근이 전혀 없으므로 lerobot 설치 여부와 무관하게 항상 실행 가능해야 한다
(이 스크립트/모듈이 lerobot을 import하지 않는다는 것 자체도 아래에서 검증한다).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_control_profile_candidate.py"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _load_cli_module():
    module_name = "generate_control_profile_candidate_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli():
    return _load_cli_module()


def _fake_joint_aggregate() -> dict:
    return {
        "aggregate": {
            "range": {"min": -10.0, "max": 10.0, "p01": -9.0, "p99": 9.0, "run_min_max_spread": 20.0},
            "frame_delta": {"mean": 0.1, "p50": 0.0, "p90": 0.5, "p95": 0.8, "p99": 1.2, "max": 1.5, "min": 0.0, "n": 100},
            "velocity": {"mean": 5.0, "p50": 0.0, "p90": 20.0, "p95": 30.0, "p99": 40.0, "max": 50.0, "min": 0.0, "n": 100},
            "tracking_error": {"mean": 1.0, "p50": 0.5, "p90": 2.0, "p95": 3.0, "p99": 4.0, "max": 5.0, "min": 0.0, "n": 100},
        }
    }


def _write_fake_aggregate(path: Path, *, run_count: int = 6) -> None:
    per_joint_stability = {
        j: {
            "tracking_error_mean_across_runs": {"mean": 1.0, "std": 0.1, "cv": 0.1, "n": run_count},
            "mean_velocity_across_runs": {"mean": 5.0, "std": 0.5, "cv": 0.1, "n": run_count},
        }
        for j in JOINT_NAMES
    }
    aggregate = {
        "generated_at": "2026-08-07T09:45:32+00:00",
        "run_count": run_count,
        "joint_aggregates": {j: _fake_joint_aggregate() for j in JOINT_NAMES},
        "run_to_run_stability": {
            "actual_loop_hz": {"mean": 59.13, "std": 0.07, "cv": 0.0012, "n": run_count},
            "latency_ms": {"mean": 88.76, "std": 13.97, "cv": 0.157, "n": 4},
            "per_joint": per_joint_stability,
        },
        "latency_aggregate": {
            "verdict": "AVAILABLE",
            "n_runs_with_valid_lag": 4,
            "n_runs_total": run_count,
            "lag_ms_median": 92.9,
            "lag_ms_mean": 88.76,
            "lag_ms_min": 67.69,
            "lag_ms_max": 101.57,
            "lag_ms_std": 13.97,
        },
        "deadband_aggregate": {
            "verdict": "DEADBAND_AGGREGATE_AVAILABLE",
            "buckets": [
                {
                    "abs_goal_present_error_ticks": t,
                    "sample_count": 100,
                    "response_count": 0,
                    "no_response_count": 100,
                    "opposite_motion_count": 0,
                    "response_fraction": 0.0,
                    "region_candidate": "NO_RESPONSE_REGION",
                    "runs_with_any_response": 0,
                }
                for t in range(6)
            ]
            + [
                {
                    "abs_goal_present_error_ticks": "6+",
                    "sample_count": 100,
                    "response_count": 70,
                    "no_response_count": 30,
                    "opposite_motion_count": 0,
                    "response_fraction": 0.706,
                    "region_candidate": "TRANSITION_REGION",
                    "runs_with_any_response": 4,
                }
            ],
        },
    }
    path.write_text(json.dumps(aggregate), encoding="utf-8")


FAKE_MAPPER_YAML = """\
rate_limit_deg_per_sec:
  shoulder_pan: 20
  shoulder_lift: 15
  elbow_flex: 20
  wrist_flex: 15
  wrist_roll: 25
  gripper: 30
"""


def test_cli_runs_end_to_end_and_writes_outputs(cli, tmp_path):
    agg_path = tmp_path / "aggregate_6runs_20260101_000000.json"
    _write_fake_aggregate(agg_path)
    mapper_path = tmp_path / "follower_safe_mapper.yaml"
    mapper_path.write_text(FAKE_MAPPER_YAML, encoding="utf-8")

    out_json = tmp_path / "generated" / "so101_control_profile_candidate_v1.json"
    out_doc = tmp_path / "docs" / "so101_control_profile_candidate_v1.md"

    args = cli.build_arg_parser().parse_args(
        [
            "--aggregate-json",
            str(agg_path),
            "--output",
            str(out_json),
            "--doc-output",
            str(out_doc),
            "--follower-safe-mapper",
            str(mapper_path),
        ]
    )
    exit_code = cli.run(args)
    assert exit_code == 0
    assert out_json.is_file()
    assert out_doc.is_file()

    profile = json.loads(out_json.read_text(encoding="utf-8"))
    assert profile["status"] == "CANDIDATE_ONLY"
    assert profile["apply_automatically"] is False
    assert profile["run_count"] == 6
    assert profile["source"] == "instrumented_teleop_6runs"

    doc_text = out_doc.read_text(encoding="utf-8")
    assert "CANDIDATE_ONLY" in doc_text
    assert "NOT_ESTABLISHED" in doc_text
    assert "CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP" in doc_text
    assert "percent_0_100" in doc_text


def test_cli_auto_discovers_latest_aggregate_when_not_specified(cli, tmp_path):
    older = tmp_path / "aggregate_6runs_20260101_000000.json"
    newer = tmp_path / "aggregate_6runs_20260102_000000.json"
    _write_fake_aggregate(older)
    _write_fake_aggregate(newer)

    args = cli.build_arg_parser().parse_args(
        [
            "--runs-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.json"),
            "--doc-output",
            str(tmp_path / "out.md"),
            "--follower-safe-mapper",
            str(tmp_path / "does_not_exist.yaml"),
        ]
    )
    exit_code = cli.run(args)
    assert exit_code == 0


def test_cli_refuses_missing_aggregate_json(cli, tmp_path):
    args = cli.build_arg_parser().parse_args(["--aggregate-json", str(tmp_path / "missing.json")])
    exit_code = cli.run(args)
    assert exit_code == 2


def test_cli_refuses_when_runs_dir_has_no_aggregate(cli, tmp_path):
    args = cli.build_arg_parser().parse_args(["--runs-dir", str(tmp_path)])
    exit_code = cli.run(args)
    assert exit_code == 2


def test_cli_still_writes_output_without_follower_safe_mapper(cli, tmp_path):
    agg_path = tmp_path / "aggregate_6runs_20260101_000000.json"
    _write_fake_aggregate(agg_path)
    out_json = tmp_path / "out.json"
    out_doc = tmp_path / "out.md"

    args = cli.build_arg_parser().parse_args(
        [
            "--aggregate-json",
            str(agg_path),
            "--output",
            str(out_json),
            "--doc-output",
            str(out_doc),
            "--follower-safe-mapper",
            str(tmp_path / "does_not_exist.yaml"),
        ]
    )
    exit_code = cli.run(args)
    assert exit_code == 0
    assert out_json.is_file()
    doc_text = out_doc.read_text(encoding="utf-8")
    assert "비교를 생략" in doc_text


def test_cli_does_not_modify_original_input_files(cli, tmp_path):
    agg_path = tmp_path / "aggregate_6runs_20260101_000000.json"
    _write_fake_aggregate(agg_path)
    mapper_path = tmp_path / "follower_safe_mapper.yaml"
    mapper_path.write_text(FAKE_MAPPER_YAML, encoding="utf-8")

    agg_before = agg_path.read_bytes()
    mapper_before = mapper_path.read_bytes()

    args = cli.build_arg_parser().parse_args(
        [
            "--aggregate-json",
            str(agg_path),
            "--output",
            str(tmp_path / "out.json"),
            "--doc-output",
            str(tmp_path / "out.md"),
            "--follower-safe-mapper",
            str(mapper_path),
        ]
    )
    cli.run(args)

    assert agg_path.read_bytes() == agg_before
    assert mapper_path.read_bytes() == mapper_before


def test_cli_never_imports_lerobot_or_hardware_safety():
    import inspect

    module = _load_cli_module()
    source = inspect.getsource(module)
    for forbidden in ("import lerobot", "from lerobot", "FeetechMotorsBus", "hardware.safety.single_joint"):
        assert forbidden not in source


def test_real_repo_follower_safe_mapper_is_not_modified_by_a_real_run(cli, tmp_path):
    """실제 저장소의 configs/follower_safe_mapper.yaml을 read-only 비교 대상으로 넘겨도
    바뀌지 않는지 확인한다 (섹션 9/12 요구사항 - 이 CLI는 그 파일을 절대 쓰지 않는다).
    """
    real_mapper_path = PROJECT_ROOT / "configs" / "follower_safe_mapper.yaml"
    before = real_mapper_path.read_bytes()

    agg_path = tmp_path / "aggregate_6runs_20260101_000000.json"
    _write_fake_aggregate(agg_path)

    args = cli.build_arg_parser().parse_args(
        [
            "--aggregate-json",
            str(agg_path),
            "--output",
            str(tmp_path / "out.json"),
            "--doc-output",
            str(tmp_path / "out.md"),
            "--follower-safe-mapper",
            str(real_mapper_path),
        ]
    )
    cli.run(args)

    assert real_mapper_path.read_bytes() == before


# ---------------------------------------------------------------------------
# 섹션 12: generated candidate가 runtime에서 자동 로드/사용되지 않는지 저장소 전체 검증
# ---------------------------------------------------------------------------

GENERATED_CANDIDATE_BASENAME = "so101_control_profile_candidate_v1"
RUNTIME_SOURCE_DIRS = ("hardware", "data_collection", "simulation", "runtime", "safety", "policies", "evaluation")
ALLOWED_REFERRERS = {
    Path("scripts") / "generate_control_profile_candidate.py",
    # 2026-08-07 추가: "Realistic SO-101 Control Layer" 작업 - realistic MuJoCo control-mode를
    # 명시적으로(opt-in) 선택했을 때만 이 candidate를 읽는 로더. 여전히 apply_automatically=false
    # 검증을 강제하고(simulation/realism/so101_control_profile.py의 load_control_profile 참고),
    # runtime 전체에 자동 적용되지 않는다 - control_mode="realistic"을 명시적으로 고른 경로에서만
    # 참조된다 (simulation/mujoco/live_web_viewer.py의 _setup_realistic_control 참고).
    Path("simulation") / "realism" / "so101_control_profile.py",
}


def test_generated_candidate_path_not_referenced_by_runtime_source():
    """``configs/generated/so101_control_profile_candidate_v1.json``이라는 파일명/경로를
    실제 runtime 코드(위 디렉터리들) 어디에서도 참조하지 않아야 한다 - 참조가 있다면 그건
    이 candidate가 어딘가에서 자동 로드된다는 뜻이므로 실패해야 한다. 이 스크립트 자신과
    테스트 파일들은 당연히 이 이름을 알고 있으므로 검사 대상에서 제외한다.
    """
    hits: list[str] = []
    for dirname in RUNTIME_SOURCE_DIRS:
        directory = PROJECT_ROOT / dirname
        if not directory.is_dir():
            continue
        for py_file in directory.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            relative = py_file.relative_to(PROJECT_ROOT)
            if relative in ALLOWED_REFERRERS:
                continue
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if GENERATED_CANDIDATE_BASENAME in text:
                hits.append(str(relative))

    assert hits == [], f"runtime 코드가 candidate 파일을 참조합니다 (자동 적용 금지 위반 가능성): {hits}"


def test_generated_candidate_json_itself_declares_no_auto_apply():
    out_path = PROJECT_ROOT / "configs" / "generated" / GENERATED_CANDIDATE_BASENAME.__add__(".json")
    if not out_path.is_file():
        pytest.skip("아직 candidate JSON이 생성되지 않았습니다 (이 CLI를 먼저 실행해야 함).")
    profile = json.loads(out_path.read_text(encoding="utf-8"))
    assert profile["apply_automatically"] is False
    assert profile["status"] == "CANDIDATE_ONLY"
