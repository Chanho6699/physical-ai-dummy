#!/usr/bin/env python3
"""실제 SO-101 follower 텔레옵 기록에서 관절별 안전 범위 요약 JSON을 만드는 CLI.

읽기 전용 분석 도구다: 원본 LeRobot 데이터셋(parquet/JSON)을 읽기만 하고 절대
수정하지 않으며, 실물 로봇에 어떤 write(torque/goal position/calibration)도
보내지 않는다. 실제 로직은 data_collection/teleop_safe_range_analysis.py에 있다.

예시
----
자동 탐색 (기본 검색 경로: <repo>/data) + dry-run으로 먼저 확인::

    python scripts/analyze_teleop_safe_ranges.py --dry-run

특정 데이터셋만 지정, 실제로 결과 파일 생성::

    python scripts/analyze_teleop_safe_ranges.py \\
        --input data/so101_cube_train_v1 --input data/so101_cube_xy_train_v1 \\
        --output configs/generated/teleop_safe_ranges.json

percentile/margin/minimum-samples 조정::

    python scripts/analyze_teleop_safe_ranges.py \\
        --lower-percentile 2 --upper-percentile 98 \\
        --margin-degree 3 --margin-percent 3 --minimum-samples 500

<repo>/data 외에 다른 위치(예: ~/.cache/huggingface/lerobot)도 읽기 전용으로 자동 탐색에
포함하려면 --search-root를 반복 지정(이 저장소의 기본 검색 경로는 사라지므로 함께 다시
지정해야 함)::

    python scripts/analyze_teleop_safe_ranges.py --dry-run \\
        --search-root data \\
        --search-root ~/.cache/huggingface/lerobot

캘리브레이션 JSON 인벤토리(episode 통계에는 절대 섞이지 않고 보고서에만 별도로 표시)::

    python scripts/analyze_teleop_safe_ranges.py --dry-run \\
        --calibration-search-root ~/.cache/huggingface/lerobot/calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.teleop_safe_range_analysis import (
    AnalysisPolicy,
    TeleopAnalysisError,
    analyze,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "generated" / "teleop_safe_ranges.json"
DEFAULT_SEARCH_ROOT = PROJECT_ROOT / "data"
# LeRobot's default dataset cache root when --dataset.root is not given
# (HF_LEROBOT_HOME, see ~/lerobot/src/lerobot/utils/constants.py). Not scanned
# by default (this repo's recordings always pass an explicit --dataset.root),
# but offered as a ready-made --search-root value for auto-discovery.
DEFAULT_HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"
DEFAULT_CALIBRATION_SEARCH_ROOT = DEFAULT_HF_LEROBOT_HOME / "calibration"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "실제 SO-101 follower 텔레옵 기록(observation.state)만 사용해 관절별 "
            "역사적 안전 범위 JSON을 생성합니다. 읽기 전용이며 로봇에 아무것도 쓰지 않습니다."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        type=Path,
        help=(
            "분석할 LeRobot v3 데이터셋 루트 경로. 여러 번 지정 가능. "
            "생략하면 --search-root 아래를 자동 탐색합니다."
        ),
    )
    parser.add_argument(
        "--search-root",
        action="append",
        dest="search_roots",
        type=Path,
        help=(
            "--input을 생략했을 때 자동 탐색할 상위 경로. 여러 번 지정 가능 "
            f"(기본, 아무것도 지정하지 않으면: {DEFAULT_SEARCH_ROOT} 1곳만). "
            "지정한 경로 아래는 읽기 전용으로만 탐색합니다."
        ),
    )
    parser.add_argument(
        "--calibration-search-root",
        action="append",
        dest="calibration_search_roots",
        type=Path,
        default=[DEFAULT_CALIBRATION_SEARCH_ROOT],
        help=(
            "calibration JSON(homing_offset/range_min/range_max 등)을 읽기 전용으로 인벤토리만 "
            f"할 경로. 여러 번 지정 가능 (기본: {DEFAULT_CALIBRATION_SEARCH_ROOT}). "
            "episode 통계 계산에는 절대 사용되지 않습니다."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"결과 JSON 경로 (기본: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--lower-percentile", type=float, default=1.0, help="안전 범위 하한 기준 percentile (기본 1).")
    parser.add_argument("--upper-percentile", type=float, default=99.0, help="안전 범위 상한 기준 percentile (기본 99).")
    parser.add_argument(
        "--margin-degree",
        type=float,
        default=2.0,
        help="degree 단위 관절(arm 5개)에 percentile 구간 안쪽으로 적용할 margin (기본 2.0).",
    )
    parser.add_argument(
        "--margin-percent",
        type=float,
        default=2.0,
        help="0-100 range 단위 관절(gripper)에 percentile 구간 안쪽으로 적용할 margin (기본 2.0).",
    )
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=200,
        help="관절별 안전 범위를 생성하기 위한 최소 유효 샘플 수 (기본 200).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="분석과 결과 계산만 수행하고 출력 파일을 쓰지 않습니다 (stdout에만 JSON을 출력).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="출력 JSON indent (기본 2).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy = AnalysisPolicy(
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        margin_degree=args.margin_degree,
        margin_percent=args.margin_percent,
        minimum_samples=args.minimum_samples,
    )

    search_roots = args.search_roots or [DEFAULT_SEARCH_ROOT]

    try:
        result, resolved_roots = analyze(
            args.inputs,
            project_root=PROJECT_ROOT,
            default_search_root=search_roots,
            policy=policy,
            calibration_search_roots=args.calibration_search_roots,
        )
    except TeleopAnalysisError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure surface
        print(f"[ERROR] 예상치 못한 실패: {exc}", file=sys.stderr)
        return 1

    print("[선택된 데이터셋 경로]")
    for root in resolved_roots:
        print(f"  - {root}")

    print("\n[신뢰됨]")
    for dataset in result.used_datasets:
        print(f"  - {dataset.relative_path}: {dataset.reason}")

    if result.excluded_datasets:
        print("\n[제외됨]")
        for dataset in result.excluded_datasets:
            print(f"  - {dataset.relative_path}: {dataset.reason}")

    calibration_files = result.provenance.get("calibration_files") or []
    if calibration_files:
        print("\n[calibration JSON 인벤토리 (episode 통계에 미포함)]")
        for entry in calibration_files:
            print(f"  - {entry['path']}: joints={entry['joints']}")

    print("\n[관절별 상태]")
    for joint, joint_result in result.joints.items():
        print(
            f"  - {joint}: status={joint_result.status} unit={joint_result.unit} "
            f"samples={joint_result.sample_count} "
            f"safe_range=[{joint_result.historical_safe_min}, {joint_result.historical_safe_max}]"
        )

    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=args.indent)

    if args.dry_run:
        print("\n[DRY-RUN] 파일을 쓰지 않았습니다. 계산된 JSON:")
        print(text)
        return 0

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n[완료] 결과 저장: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
