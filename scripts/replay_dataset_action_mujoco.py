#!/usr/bin/env python3
"""LeRobot 데이터셋 action을 SO-101 MuJoCo 모델에서 재생하는 CLI.

실물 하드웨어(USB serial, ROS2)나 SmolVLA 추론은 호출하지 않는다.
자세한 설명은 docs/mujoco_action_replay.md 참고.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.mujoco.dataset_action_replay import ReplayArgs, run_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SO-101 MuJoCo에서 LeRobot 데이터셋 action을 안전하게 재생/검증합니다.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot 데이터셋 루트 경로")
    parser.add_argument("--episode-index", type=int, required=True, help="재생할 episode index")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", help="MuJoCo GUI viewer로 재생")
    mode_group.add_argument("--headless", action="store_true", help="GUI 없이 재생 (기본값)")

    parser.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율 (예: 0.25, 0.5, 1.0, 2.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="최대 처리 프레임 수")
    parser.add_argument("--start-frame", type=int, default=0, help="재생 시작 프레임 index")
    parser.add_argument("--report-path", type=Path, default=None, help="JSON 리포트 저장 경로")
    parser.add_argument("--config", type=Path, default=None, help="safety 설정 YAML 경로 (기본: configs/mujoco_so101.yaml)")

    parser.add_argument("--dry-run", action="store_true", help="actuator에 실제로 적용하지 않고 검사만 수행")
    parser.add_argument("--continue-on-warning", action="store_true", help="WARN 발생 시에도 재생을 계속 진행")

    parser.add_argument("--quiet", action="store_true", help="최종 결과와 오류만 출력")
    parser.add_argument("--verbose", action="store_true", help="joint/actuator mapping, 프레임별 상세 진단 출력")
    parser.add_argument("--no-color", action="store_true", help="ANSI 색상 출력을 끔")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.quiet and args.verbose:
        print("[오류] --quiet와 --verbose는 동시에 사용할 수 없습니다.", file=sys.stderr)
        return 2

    mode = "gui" if args.gui else "headless"

    replay_args = ReplayArgs(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        speed=args.speed,
        mode=mode,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        report_path=args.report_path,
        config_path=args.config,
        quiet=args.quiet,
        verbose=args.verbose,
        no_color=args.no_color,
        dry_run=args.dry_run,
        continue_on_warning=args.continue_on_warning,
    )

    outcome = run_replay(replay_args)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
