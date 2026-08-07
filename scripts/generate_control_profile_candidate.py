#!/usr/bin/env python3
"""6-run instrumented teleop aggregate -> candidate control profile v1 생성 CLI - 완전 offline.

기존에 ``scripts/analyze_instrumented_teleop_runs.py``로 만들어 둔
``reports/instrumented_teleop/aggregate_*runs_*.json``을 읽어서
``hardware/diagnostics/control_profile_candidate.py``의 순수 변환 함수로
"VLA/follower 제어용 candidate control profile v1"을 만든다.

**이 스크립트는 하드웨어에 전혀 접근하지 않는다.** ``lerobot``도, 이 저장소의 serial 접근
클래스도 import하지 않는다. 읽는 파일은 이미 디스크에 있는 aggregate JSON과(비교표용으로)
``configs/follower_safe_mapper.yaml`` 뿐이고 **둘 다 읽기만 한다 - 절대 수정하지 않는다.**

쓰는 파일은 다음 둘뿐이다:
  - ``configs/generated/so101_control_profile_candidate_v1.json`` (CANDIDATE_ONLY, apply_automatically=false)
  - ``docs/so101_control_profile_candidate_v1.md`` (사람이 읽는 검토용 문서)

이 스크립트를 실행해도 실제 로봇 제어 코드/런타임 경로는 전혀 바뀌지 않는다 - 이 두 파일을
어디에서도 자동으로 불러오는 코드가 없다(이 스크립트 자신과 테스트만 참조한다).

실행 예시:
    python scripts/generate_control_profile_candidate.py
    python scripts/generate_control_profile_candidate.py --aggregate-json reports/instrumented_teleop/aggregate_6runs_20260807_094532.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.diagnostics.control_profile_candidate import (
    build_control_profile_candidate,
    compare_with_existing_rate_limits,
    render_comparison_markdown,
)

DEFAULT_RUNS_DIR = PROJECT_ROOT / "reports" / "instrumented_teleop"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "configs" / "generated" / "so101_control_profile_candidate_v1.json"
DEFAULT_DOC_OUTPUT = PROJECT_ROOT / "docs" / "so101_control_profile_candidate_v1.md"
DEFAULT_FOLLOWER_SAFE_MAPPER = PROJECT_ROOT / "configs" / "follower_safe_mapper.yaml"

JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="6-run instrumented teleop aggregate JSON -> candidate control profile v1 (offline, CANDIDATE_ONLY)."
    )
    parser.add_argument(
        "--aggregate-json",
        default=None,
        help="입력 aggregate JSON 경로 (생략하면 --runs-dir에서 가장 최신 aggregate_*runs_*.json을 고른다).",
    )
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help=f"--aggregate-json 생략 시 탐색할 디렉터리 (기본 {DEFAULT_RUNS_DIR})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSON), help=f"candidate profile JSON 출력 경로 (기본 {DEFAULT_OUTPUT_JSON})")
    parser.add_argument("--doc-output", default=str(DEFAULT_DOC_OUTPUT), help=f"candidate profile markdown 문서 출력 경로 (기본 {DEFAULT_DOC_OUTPUT})")
    parser.add_argument(
        "--follower-safe-mapper",
        default=str(DEFAULT_FOLLOWER_SAFE_MAPPER),
        help=f"read-only 비교 대상 기존 mapper 설정 (기본 {DEFAULT_FOLLOWER_SAFE_MAPPER}) - 이 파일은 수정되지 않는다.",
    )
    return parser


def _find_latest_aggregate(runs_dir: Path) -> Path | None:
    candidates = sorted(runs_dir.glob("aggregate_*runs_*.json"))
    return candidates[-1] if candidates else None


def _load_existing_rate_limits(follower_safe_mapper_path: Path) -> dict[str, float] | None:
    """``configs/follower_safe_mapper.yaml``의 ``rate_limit_deg_per_sec``만 읽기 전용으로 읽는다.

    PyYAML이 없거나 파일이 없거나 파싱에 실패해도 이 스크립트 전체를 실패시키지 않는다 -
    비교표 없이 candidate profile만 만들고 진행한다(핵심 산출물이 아니라 부가 비교이므로).
    """
    if not follower_safe_mapper_path.is_file():
        return None
    try:
        import yaml  # 지역 import - 이 의존성이 없어도 candidate profile 생성 자체는 되게 한다.
    except ImportError:
        return None
    try:
        data = yaml.safe_load(follower_safe_mapper_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    rate_limits = data.get("rate_limit_deg_per_sec")
    return rate_limits if isinstance(rate_limits, dict) else None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown_doc(profile: dict[str, Any], comparison: dict[str, Any] | None, *, source_aggregate_path: str) -> str:
    lines: list[str] = []
    lines.append("# SO-101 Control Profile Candidate v1")
    lines.append("")
    lines.append(f"- status: **{profile['status']}**")
    lines.append(f"- source: {profile['source']}")
    lines.append(f"- run_count: {profile['run_count']}")
    lines.append(f"- apply_automatically: **{profile['apply_automatically']}**")
    lines.append(f"- source_aggregate_path: `{source_aggregate_path}`")
    lines.append(f"- generated_at: {profile['generated_at']}")
    lines.append("")
    lines.append(
        "> 이 문서는 실측 6-run instrumented teleop 통계를 VLA/follower 제어용 후보 기준표로 "
        "정리한 것입니다. **어떤 값도 실제 robot runtime에 적용되지 않았습니다.**"
    )
    lines.append("")

    lines.append("## 현재 확보한 것")
    lines.append("")
    lines.append("- joint별 historical operating range (observed min/max, p01/p99)")
    lines.append("- joint별 frame delta 분포 (p50/p95/p99/max) + candidate soft limit")
    lines.append("- joint별 velocity 분포 (p50/p95/p99/max) + candidate soft limit")
    lines.append("- joint별 tracking error 분포 (MAE/p95/p99/max) + warning/severe candidate")
    lines.append("- wrist_roll deadband candidate (0~5 tick NO_RESPONSE, 6+ tick TRANSITION, HIGH_RESPONSE는 미확립)")
    lines.append("- local instrumented teleop 계측 latency (leader command -> follower actual, end-to-end 아님)")
    lines.append("")

    lines.append("## 향후 VLA에서 어디에 사용할지 (계획만 - 이번 작업에서 코드에 적용하지 않음)")
    lines.append("")
    lines.append("```text")
    lines.append("VLA action")
    lines.append("   |")
    lines.append("   v")
    lines.append("Action Adapter")
    lines.append("   |")
    lines.append("   v")
    lines.append("candidate deadband / rate profile   <- 이 문서/JSON (CANDIDATE_ONLY)")
    lines.append("   |")
    lines.append("   v")
    lines.append("Follower command")
    lines.append("   |")
    lines.append("   v")
    lines.append("Execution Monitor")
    lines.append("   |")
    lines.append("   v")
    lines.append("tracking error / latency monitoring")
    lines.append("```")
    lines.append("")

    lines.append("## Joint별 candidate 값 (pooled 6-run aggregate)")
    lines.append("")
    lines.append(
        "| joint | unit | confidence | historical inner range (p01~p99) | frame_delta p95/p99 | "
        "velocity p95/p99 | tracking_error p95/p99 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for joint in JOINT_ORDER:
        j = profile["joints"].get(joint, {})
        rng = j.get("historical_operating_range", {}) or {}
        inner = rng.get("candidate_historical_inner_range")
        inner_str = f"[{_fmt(inner[0])}, {_fmt(inner[1])}]" if inner else "N/A"
        fd = j.get("frame_delta", {}) or {}
        vel = j.get("velocity", {}) or {}
        vel_prof = vel.get("observed_velocity_profile", {}) or {}
        te = j.get("tracking_error", {}) or {}
        lines.append(
            f"| {joint} | {j.get('unit', 'N/A')} | {j.get('confidence', 'N/A')} | {inner_str} | "
            f"{_fmt(fd.get('p95'))} / {_fmt(fd.get('p99'))} | "
            f"{_fmt(vel_prof.get('p95'))} / {_fmt(vel_prof.get('p99'))} | "
            f"{_fmt(te.get('p95'))} / {_fmt(te.get('p99'))} |"
        )
    lines.append("")
    lines.append(profile.get("gripper_unit_note", ""))
    lines.append("")

    wr = profile.get("wrist_roll_deadband_analysis", {}) or {}
    lines.append("## wrist_roll deadband candidate")
    lines.append("")
    lines.append(f"- no_response_region_ticks: {wr.get('no_response_region_ticks')} (confidence: {wr.get('no_response_region_confidence')})")
    lines.append(f"- transition_region_start_ticks: {wr.get('transition_region_start_ticks')} (confidence: {wr.get('transition_region_confidence')})")
    lines.append(
        f"- transition_region_aggregate_response_fraction: {_fmt(wr.get('transition_region_aggregate_response_fraction'), 4)} "
        f"(runs_with_response: {wr.get('transition_region_runs_with_response')}/6)"
    )
    lines.append(f"- **high_response_region: {wr.get('high_response_region')}**")
    lines.append(f"  - {wr.get('high_response_region_rationale')}")
    lines.append("")
    deg_table = wr.get("tick_to_degree_equivalent", {}) or {}
    lines.append("| tick | degree equivalent |")
    lines.append("|---|---|")
    for tick in sorted(deg_table, key=lambda k: int(k)):
        lines.append(f"| {tick} | {deg_table[tick]:.4f}° |")
    lines.append("")
    lines.append(f"> {wr.get('note', '')}")
    lines.append("")

    timing = profile.get("timing", {}) or {}
    lat = timing.get("observed_latency_ms", {}) or {}
    lines.append("## Control timing profile")
    lines.append("")
    lines.append(f"- nominal_control_hz: {_fmt(timing.get('nominal_control_hz'), 2)} (confidence: {timing.get('control_hz_confidence')})")
    lines.append(
        f"- observed_latency_ms: median={_fmt(lat.get('median'), 2)} min={_fmt(lat.get('min'), 2)} "
        f"max={_fmt(lat.get('max'), 2)} std={_fmt(lat.get('std'), 2)} valid_runs={lat.get('valid_runs')} "
        f"(confidence: {timing.get('latency_confidence')})"
    )
    lines.append(f"- **latency_scope: {timing.get('latency_scope')}**")
    lines.append(f"  - {timing.get('latency_scope_note')}")
    lines.append("")

    lines.append("## Confidence legend")
    lines.append("")
    for level, desc in (profile.get("confidence_legend") or {}).items():
        lines.append(f"- **{level}**: {desc}")
    lines.append("")

    lines.append("## 기존 follower-safe mapper (configs/follower_safe_mapper.yaml, read-only 비교)")
    lines.append("")
    if comparison is None:
        lines.append("(follower_safe_mapper.yaml을 읽지 못했거나 rate_limit_deg_per_sec가 없어 비교를 생략했습니다.)")
    else:
        lines.append(render_comparison_markdown(comparison))
        lines.append("")
        lines.append(
            "이 표는 `configs/follower_safe_mapper.yaml`을 **읽기만** 했습니다 - 수정하지 않았습니다. "
            "`CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP`은 기존에 미검증 상태로 넣어둔 rate limit이 "
            "실제 6-run teleop에서 관측된 velocity p95보다 낮다(더 보수적이다)는 뜻일 뿐, "
            "지금 기존 설정이 틀렸다거나 바꿔야 한다는 의미는 아닙니다."
        )
    lines.append("")

    lines.append("## 적용 제한")
    lines.append("")
    for restriction in profile.get("usage_restrictions", []):
        lines.append(f"- {restriction}")
    lines.append("")
    lines.append("---")
    lines.append("direct_register_write_count=0, hardware_execution_count=0, git_commit_count=0")

    return "\n".join(lines)


def run(args: argparse.Namespace, *, stdout=None) -> int:
    stdout = stdout if stdout is not None else sys.stdout

    def _p(msg: str) -> None:
        print(msg, file=stdout)

    if args.aggregate_json:
        aggregate_path = Path(args.aggregate_json).expanduser()
    else:
        runs_dir = Path(args.runs_dir).expanduser()
        found = _find_latest_aggregate(runs_dir)
        if found is None:
            _p(f"거부: {runs_dir}에서 aggregate_*runs_*.json을 찾지 못했습니다. --aggregate-json으로 직접 지정하세요.")
            return 2
        aggregate_path = found

    if not aggregate_path.is_file():
        _p(f"거부: aggregate JSON을 찾을 수 없습니다: {aggregate_path}")
        return 2

    _p(f"=== 입력 aggregate (읽기 전용): {aggregate_path} ===")
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _p(f"거부: aggregate JSON 파싱 실패: {exc}")
        return 2

    profile = build_control_profile_candidate(aggregate, source_aggregate_path=str(aggregate_path))

    follower_safe_mapper_path = Path(args.follower_safe_mapper).expanduser()
    existing_rate_limits = _load_existing_rate_limits(follower_safe_mapper_path)
    comparison = None
    if existing_rate_limits is not None:
        comparison = compare_with_existing_rate_limits(profile, existing_rate_limits)
        _p(f"=== 기존 rate limit 비교 대상 (읽기 전용): {follower_safe_mapper_path} ===")
    else:
        _p(f"(참고: {follower_safe_mapper_path}에서 rate_limit_deg_per_sec를 읽지 못해 비교표를 생략합니다.)")

    output_json_path = Path(args.output).expanduser()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _p(f"\nJSON 저장: {output_json_path}")

    doc_path = Path(args.doc_output).expanduser()
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        render_markdown_doc(profile, comparison, source_aggregate_path=str(aggregate_path)),
        encoding="utf-8",
    )
    _p(f"Markdown 저장: {doc_path}")

    _p(f"\nstatus={profile['status']} apply_automatically={profile['apply_automatically']}")
    _p("direct_register_write_count=0")
    _p("hardware_execution_count=0")
    _p(f"(원본 {aggregate_path.name}과 {follower_safe_mapper_path.name}은 읽기만 했고 수정하지 않았습니다.)")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
