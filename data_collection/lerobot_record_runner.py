#!/usr/bin/env python3
"""Runs ``lerobot-record`` with a pre-recording leader/follower sync-warmup stage.

Why this file exists (not a wrapper-side ``time.sleep``)
----------------------------------------------------------
``scripts/record_episodes.py`` never talks to the robot/teleop directly - it
shells out to the ``lerobot-record`` console script, whose actual recording
lifecycle lives in ``lerobot.scripts.lerobot_record`` (installed in the
separate LeRobot venv, e.g. ``~/lerobot``). Inside that module, ``record()``
does, in order:

    dataset = LeRobotDataset.create(...) / .resume(...)
    teleop.connect(); robot.connect()
    while recorded_episodes < num_episodes:
        record_loop(..., dataset=dataset, control_time_s=episode_time_s)   # <- writes frames
        record_loop(..., dataset=None,    control_time_s=reset_time_s)     # <- reset, no frames
        dataset.save_episode()

The follower snaps to the leader's live pose the moment actions actually
start flowing to it - i.e. the *first* iteration of a ``record_loop`` call
with an active teleop. For the very first episode of a process, that first
iteration is also the first frame written to the dataset, so the snap gets
recorded. (Empirically, per ``reports/grid35_episode_start_analysis``, this
shows up at frame 0 of *every* episode, not just the first - most likely
because the operator is still settling the leader arm back into the start
pose as the reset window ends.)

This script closes that gap without touching the wrapper's subprocess
boundary or faking a sleep anywhere: it imports the real
``lerobot.scripts.lerobot_record`` module (so every other lifecycle stage -
dataset create/resume, connect order, keyboard listener, reset loop,
save_episode/finalize, push_to_hub, --resume - runs completely unmodified)
and monkey-patches its module-level ``record_loop`` symbol. ``record()``
looks up ``record_loop`` by name in the module's globals at call time (it is
defined in the same module, not imported), so replacing
``lerobot.scripts.lerobot_record.record_loop`` before calling ``record()``
is enough for the patched version to run in its place - no copy of
``record()``'s ~140 lines of dataset/robot/teleop setup needed.

The patched ``record_loop`` recognizes a *real* recording call by
``dataset is not None`` (exactly how ``record()`` calls it - the reset-phase
calls omit ``dataset`` entirely, defaulting to ``None``). Before such a call,
it drives the genuine teleoperation control loop - real
``robot.get_observation()`` / ``teleop.get_action()`` / ``robot.send_action()``
each control tick, at the dataset's own fps - for ``--sync-warmup-seconds``
by calling the *original* ``record_loop`` with ``dataset=None`` in 1-second
countdown chunks. Because ``dataset`` stays ``None`` for every warmup chunk,
``record_loop`` never calls ``dataset.add_frame()`` during warmup (see the
``if dataset is not None:`` guards in ``lerobot_record.record_loop``), so no
parquet row or video frame is ever produced for the warmup window, and it
never touches ``episode_time_s``/``reset_time_s`` or the dataset's
frame_index/timestamp counters. Only once warmup completes does this script
call the *real* ``record_loop`` (with the caller's original ``dataset`` and
``control_time_s=episode_time_s``) - that call is what the terminal's
"녹화를 시작합니다" line is printed immediately before.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator
from typing import Any

DEFAULT_SYNC_WARMUP_S = 3.0

_SYNC_WARMUP_FLAG = "--sync-warmup-seconds"


def parse_runner_args(argv: list[str]) -> tuple[float, list[str]]:
    """Split ``--sync-warmup-seconds`` out of ``argv``.

    Everything else is passed through untouched so it reaches draccus'
    ``lerobot-record`` argument parser exactly as before (it would reject an
    unrecognized flag).
    """
    sync_warmup_s = DEFAULT_SYNC_WARMUP_S
    remaining: list[str] = []
    i = 0
    n = len(argv)
    while i < n:
        arg = argv[i]
        if arg == _SYNC_WARMUP_FLAG:
            if i + 1 >= n:
                raise ValueError(f"{_SYNC_WARMUP_FLAG} requires a value.")
            sync_warmup_s = float(argv[i + 1])
            i += 2
            continue
        if arg.startswith(_SYNC_WARMUP_FLAG + "="):
            sync_warmup_s = float(arg.split("=", 1)[1])
            i += 1
            continue
        remaining.append(arg)
        i += 1

    if sync_warmup_s < 0:
        raise ValueError(f"{_SYNC_WARMUP_FLAG} must be >= 0, got {sync_warmup_s}.")
    return sync_warmup_s, remaining


def iter_warmup_chunks(total_seconds: float) -> Iterator[tuple[int, float]]:
    """Yield ``(countdown_label, chunk_duration_s)`` covering ``total_seconds``.

    Each chunk is at most 1s so the caller can print a 1-second countdown
    between chunks. A fractional ``total_seconds`` (e.g. 3.5) is handled
    naturally: the final chunk is simply shorter than 1s, so the sum of all
    chunk durations always equals ``total_seconds`` exactly.
    """
    remaining = max(0.0, float(total_seconds))
    while remaining > 1e-9:
        label = max(1, math.ceil(remaining - 1e-9))
        chunk = min(1.0, remaining)
        yield label, chunk
        remaining -= chunk


def _run_sync_warmup(
    original_record_loop: Callable[..., Any],
    sync_warmup_s: float,
    recording_kwargs: dict[str, Any],
    print_fn: Callable[[str], None],
) -> None:
    """Keep teleoperation live (follower tracking leader) for ``sync_warmup_s`` before recording."""
    warmup_kwargs = {
        "robot": recording_kwargs["robot"],
        "events": recording_kwargs["events"],
        "fps": recording_kwargs["fps"],
        "teleop_action_processor": recording_kwargs["teleop_action_processor"],
        "robot_action_processor": recording_kwargs["robot_action_processor"],
        "robot_observation_processor": recording_kwargs["robot_observation_processor"],
        "teleop": recording_kwargs.get("teleop"),
        "dataset": None,  # <- guarantees no parquet/video frame is written during warmup
        "single_task": None,
        "display_data": recording_kwargs.get("display_data", False),
        "display_mode": recording_kwargs.get("display_mode", "rerun"),
        "display_compressed_images": recording_kwargs.get("display_compressed_images", False),
    }

    print_fn("")
    print_fn("=" * 68)
    print_fn("[준비] 리더암과 팔로워암을 동기화합니다.")
    print_fn("[주의] 팔로워암이 움직일 수 있습니다. 리더암을 시작 자세로 유지하세요.")

    chunks = list(iter_warmup_chunks(sync_warmup_s))
    if not chunks:
        print_fn("[동기화] 대기 시간이 0으로 설정되어 안정화 없이 바로 진행합니다.")
    for label, chunk_s in chunks:
        print_fn(f"[동기화] 안정화까지 {label}...")
        original_record_loop(**warmup_kwargs, control_time_s=chunk_s)

    print_fn("[완료] 동기화 안정화가 끝났습니다.")
    print_fn("[녹화 준비] 시작 자세를 유지하세요.")
    print_fn("[녹화] 시작 자세를 잠깐 유지한 뒤(약 0.3~0.5초) 동작을 시작하세요.")
    print_fn("🔴 녹화를 시작합니다!")
    # Deliberately no sleep here: the caller invokes the real record_loop
    # (the actual dataset-writing call) on the very next line.


def make_patched_record_loop(
    original_record_loop: Callable[..., Any],
    sync_warmup_s: float,
    print_fn: Callable[[str], None] = print,
) -> Callable[..., Any]:
    """Wrap ``record_loop`` so every real recording call is preceded by a sync warmup.

    A call is "real recording" exactly when it is passed a non-``None``
    ``dataset`` kwarg - that's how ``lerobot.scripts.lerobot_record.record()``
    itself distinguishes an episode-recording call from a reset-phase call
    (the latter omits ``dataset``, defaulting to ``None``). Reset-phase calls
    pass straight through untouched, so ``--reset-seconds`` behavior and
    timing are unaffected.
    """

    def patched_record_loop(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("dataset") is not None:
            _run_sync_warmup(original_record_loop, sync_warmup_s, kwargs, print_fn)
        return original_record_loop(*args, **kwargs)

    return patched_record_loop


def main() -> int:
    sync_warmup_s, remaining_argv = parse_runner_args(sys.argv[1:])

    import lerobot.scripts.lerobot_record as lerobot_record_module

    original_record_loop = lerobot_record_module.record_loop
    lerobot_record_module.record_loop = make_patched_record_loop(
        original_record_loop, sync_warmup_s
    )

    sys.argv = [sys.argv[0], *remaining_argv]
    lerobot_record_module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
