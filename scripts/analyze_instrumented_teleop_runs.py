#!/usr/bin/env python3
"""Instrumented Teleop 다중-run 통합 분석 CLI - 완전 offline.

기존에 ``scripts/run_instrumented_teleop.py``로 실제 실행해 만들어 둔
``reports/instrumented_teleop/instrumented_wrist_roll_*.csv``(+ 대응 ``*_report.json``)를
읽어서, 최신 N개(기본 6개) run을 골라 joint별 range/frame-delta/velocity/tracking-error,
latency, wrist_roll causal deadband를 통합 집계한다.

**이 스크립트는 하드웨어에 전혀 접근하지 않는다** - ``lerobot``도, 이 저장소의 serial 접근
클래스도 import하지 않는다(``hardware/diagnostics/instrumented_teleop_aggregate.py`` 모듈
docstring 참고). 원본 CSV/JSON은 읽기만 하고 절대 수정하지 않는다.

실행 예시:
    python scripts/analyze_instrumented_teleop_runs.py
    python scripts/analyze_instrumented_teleop_runs.py --runs-dir reports/instrumented_teleop --count 6
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.diagnostics.instrumented_teleop_aggregate import (
    build_aggregate_report,
    load_run_bundle,
    render_markdown_report,
    select_latest_runs,
)

DEFAULT_RUNS_DIR = PROJECT_ROOT / "reports" / "instrumented_teleop"
DEFAULT_COUNT = 6


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instrumented Teleop 다중-run 통합(offline) 분석기.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help=f"run CSV/JSON이 있는 디렉터리 (기본 {DEFAULT_RUNS_DIR})")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"분석에 쓸 최신 run 개수 (기본 {DEFAULT_COUNT})")
    parser.add_argument("--output-dir", default=None, help="결과 JSON/MD 저장 디렉터리 (기본: --runs-dir와 동일)")
    parser.add_argument("--deadband-lookahead-ms", type=float, default=None, help="causal deadband lookahead(ms) - 생략하면 core 기본값 사용")
    parser.add_argument("--motion-response-noise-threshold-ticks", type=int, default=None, help="causal deadband 노이즈 문턱값(tick) - 생략하면 core 기본값 사용")
    return parser


def run(args: argparse.Namespace, *, stdout=None) -> int:
    stdout = stdout if stdout is not None else sys.stdout

    def _p(msg: str) -> None:
        print(msg, file=stdout)

    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.is_dir():
        _p(f"거부: run 디렉터리를 찾을 수 없습니다: {runs_dir}")
        return 2

    selected = select_latest_runs(runs_dir, count=args.count)
    if len(selected) < args.count:
        _p(f"경고: 요청한 {args.count}개보다 적은 {len(selected)}개의 유효한 run만 찾았습니다.")
    if not selected:
        _p("거부: 유효한 run을 찾지 못했습니다 (CSV/JSON 매칭 실패 또는 전부 빈 파일).")
        return 2

    _p(f"=== 선택된 run ({len(selected)}개, {runs_dir}) ===")
    for ts, csv_path, json_path in selected:
        _p(f"  {ts}: {csv_path.name}  (json={'있음' if json_path else '없음'})")

    kwargs = {}
    if args.deadband_lookahead_ms is not None:
        kwargs["lookahead_ms"] = args.deadband_lookahead_ms
    if args.motion_response_noise_threshold_ticks is not None:
        kwargs["noise_threshold_ticks"] = args.motion_response_noise_threshold_ticks

    _p("\n=== run 로딩 중 (CSV/JSON 읽기만 - 원본 수정 없음) ===")
    bundles = []
    for ts, csv_path, json_path in selected:
        bundle = load_run_bundle(ts, csv_path, json_path)
        bundles.append(bundle)
        _p(
            f"  {ts}: samples={len(bundle.samples)} malformed_rows={bundle.malformed_row_count} "
            f"quality={bundle.quality['verdict']}"
        )
        for reason in bundle.quality.get("reasons", []):
            _p(f"    - {reason}")

    _p("\n=== 통합 분석 중 (기존 causal deadband 로직 재사용) ===")
    report = build_aggregate_report(bundles, **kwargs)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path_out = output_dir / f"aggregate_{len(bundles)}runs_{timestamp}.json"
    md_path_out = output_dir / f"aggregate_{len(bundles)}runs_{timestamp}.md"

    report_with_meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs_dir": str(runs_dir),
        **report,
    }
    json_path_out.write_text(json.dumps(report_with_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md_path_out.write_text(render_markdown_report(report), encoding="utf-8")

    _p(f"\nJSON 저장: {json_path_out}")
    _p(f"Markdown 저장: {md_path_out}")
    _p(f"\ndirect_register_write_count={report['direct_register_write_count']}")
    _p(f"hardware_execution_count={report['hardware_execution_count']}")
    _p("(원본 CSV/JSON은 읽기만 했고 수정하지 않았습니다.)")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
