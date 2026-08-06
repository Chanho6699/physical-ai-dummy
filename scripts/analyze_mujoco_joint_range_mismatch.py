#!/usr/bin/env python3
"""데이터셋 action/state가 MuJoCo 관절 range와 얼마나, 왜 어긋나는지 조사하는 CLI.

이 스크립트는 순수 조사/분석 도구다. MJCF나 configs/mujoco_so101.yaml을 수정하지 않고,
dataset_action_replay.py의 재생 동작에도 전혀 영향을 주지 않는다 (읽기 전용 분석).
자세한 배경은 docs/wrist_flex_range_mismatch_investigation.md 참고.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.mujoco import console_status as cs
from simulation.mujoco.dataset_loader import DatasetLoadError
from simulation.mujoco.joint_range_diagnostics import (
    analyze_joint,
    build_csv_rows,
    build_report_dict,
    evaluate_hypotheses,
    hypothetical_offset_needed_deg,
)
from simulation.mujoco.so101_model import SO101ModelError

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "joint_range_analysis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="데이터셋 action/state와 MuJoCo 관절 range의 불일치 원인을 조사합니다.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot 데이터셋 루트 경로")
    parser.add_argument("--joint", type=str, required=True, help="조사할 관절 이름 (예: wrist_flex)")
    parser.add_argument("--scene-path", type=Path, default=None, help="MuJoCo 씬 XML 경로 override (기본: simulation/mujoco/assets/scene.xml)")
    parser.add_argument("--json-report-path", type=Path, default=None, help="JSON 리포트 저장 경로")
    parser.add_argument("--csv-report-path", type=Path, default=None, help="CSV 리포트 저장 경로")
    parser.add_argument("--quiet", action="store_true", help="최종 결과만 출력")
    parser.add_argument("--verbose", action="store_true", help="episode별 상세 통계까지 출력")
    parser.add_argument("--no-color", action="store_true", help="ANSI 색상 출력을 끔")
    return parser


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def main() -> int:
    args = build_parser().parse_args()
    opts = cs.ConsoleOptions(quiet=args.quiet, verbose=args.verbose, use_color=cs.resolve_use_color(args.no_color))

    cs.print_joint_range_header(opts, joint_name=args.joint, dataset_root=str(args.dataset_root))

    try:
        analysis, over_amounts_deg = analyze_joint(args.dataset_root, args.joint, scene_path=args.scene_path)
    except DatasetLoadError as exc:
        cs.print_error(str(exc))
        return 2
    except SO101ModelError as exc:
        cs.print_error(str(exc))
        return 2
    except ValueError as exc:
        cs.print_error(str(exc))
        return 2

    cs.print_joint_range_summary(opts, analysis, over_amounts_deg)

    findings = evaluate_hypotheses(analysis, over_amounts_deg)
    cs.print_hypotheses(opts, findings)

    offset = hypothetical_offset_needed_deg(analysis)
    if not opts.quiet:
        print("-" * 68)
        print(
            f"[참고] action 전체를 range 안에 넣으려면 균일 offset ≈ {offset:.4f} deg가 필요함 "
            "(가상 계산일 뿐, 실제 매핑에는 적용하지 않았습니다)."
        )

    report = build_report_dict(analysis, over_amounts_deg, findings)

    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(args.dataset_root).name
    json_path = args.json_report_path or (DEFAULT_REPORT_DIR / f"{dataset_name}_{args.joint}.json")
    csv_path = args.csv_report_path or (DEFAULT_REPORT_DIR / f"{dataset_name}_{args.joint}.csv")
    json_path = _unique_path(json_path)
    csv_path = _unique_path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = build_csv_rows(analysis)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    cs.print_joint_range_footer(opts, json_path=str(json_path), csv_path=str(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
