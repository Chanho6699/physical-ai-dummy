#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.recorder import RecordingError, RecordingRequest, record_episodes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SO-101 + dual-camera LeRobot episodes recorder wrapper."
    )
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=PROJECT_ROOT / "configs/hardware.local.json",
        help="Local robot and camera JSON config.",
    )
    parser.add_argument("--dataset-name", required=True, help="New dataset folder name.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Parent directory that stores LeRobot datasets.",
    )
    parser.add_argument(
        "--task",
        default="Pick up the cube and place it in the target area.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--reset-seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--display-data", action="store_true")
    parser.add_argument("--streaming-encoding", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the final Enter confirmation.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request = RecordingRequest(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        task=args.task,
        num_episodes=args.episodes,
        episode_time_s=args.episode_seconds,
        reset_time_s=args.reset_seconds,
        fps=args.fps,
        resume=args.resume,
        display_data=args.display_data,
        streaming_encoding=args.streaming_encoding,
    )

    try:
        record_episodes(
            args.hardware_config,
            request,
            dry_run=args.dry_run,
            assume_yes=args.yes,
        )
    except RecordingError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Unexpected failure: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
