#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.validator import (
    DatasetValidationError,
    print_validation_report,
    validate_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a LeRobot v3 dataset.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument(
        "--expected-camera",
        action="append",
        dest="expected_cameras",
        help="Expected role name. Repeat for multiple cameras.",
    )
    parser.add_argument("--frame-tolerance", type=int, default=2)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path. Defaults to reports/dataset_validation/<name>.json.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return failure exit code when warnings exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expected_cameras = tuple(args.expected_cameras or ["workspace", "wrist"])

    try:
        report = validate_dataset(
            args.dataset_root,
            expected_episodes=args.expected_episodes,
            expected_cameras=expected_cameras,
            frame_tolerance=args.frame_tolerance,
        )
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected validation failure: {exc}", file=sys.stderr)
        return 1

    print_validation_report(report)

    report_path = args.report
    if report_path is None:
        dataset_name = Path(args.dataset_root).expanduser().resolve().name
        report_path = (
            PROJECT_ROOT
            / "reports"
            / "dataset_validation"
            / f"{dataset_name}.json"
        )
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[REPORT] {report_path}")

    if report.failures:
        return 1
    if args.strict and report.warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
