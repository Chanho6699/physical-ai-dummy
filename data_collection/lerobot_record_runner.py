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
import threading
import time
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.episode_labels import (
    EpisodeLabelError, append_episode_label, validate_label_alignment,
)

DEFAULT_SYNC_WARMUP_S = 3.0

_SYNC_WARMUP_FLAG = "--sync-warmup-seconds"


def apply_episode_key(name: str, events: dict[str, Any], print_fn: Callable[[str], None] = print) -> None:
    """Apply a key according to the explicit RECORDING/REVIEW lifecycle state."""
    key = name.lower()
    state = events.get("lifecycle_state")
    if state == "recording":
        if key in ("enter", "space"):
            events["lifecycle_state"] = "recording_finished"
            events["exit_early"] = True
            print_fn("[FINISHED] recording stopped; episode is not saved yet.")
        return
    if state != "review" or key not in ("c", "v", "r", "q"):
        return
    events["review_choice"] = key
    review_event = events.get("review_event")
    if review_event is not None:
        review_event.set()


def get_writer_episode_buffer(dataset: Any) -> dict[str, Any]:
    """Return the real LeRobot write buffer owned by DatasetWriter."""
    writer = getattr(dataset, "writer", None)
    buffer = getattr(writer, "episode_buffer", None)
    if not isinstance(buffer, dict):
        raise EpisodeLabelError("LeRobot dataset.writer.episode_buffer is unavailable")
    return buffer


def validate_episode_buffer(
    buffer: dict[str, Any], features: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Validate the installed LeRobot v3 DatasetWriter buffer representation."""
    required_features = (
        "observation.state",
        "action",
        "observation.images.workspace",
        "observation.images.wrist",
    )
    if features is not None:
        missing_schema = [key for key in required_features if key not in features]
        if missing_schema:
            return False, f"required dataset features missing: {missing_schema}"

    required_buffer = (*required_features, "timestamp", "frame_index", "task")
    missing = [key for key in required_buffer if key not in buffer]
    if missing:
        return False, f"writer buffer missing keys: {missing}"
    try:
        frame_count = int(buffer["size"])
    except (KeyError, TypeError, ValueError):
        return False, "writer buffer has invalid size"
    if frame_count <= 0:
        return False, "episode is empty"

    lengths = {key: len(buffer[key]) for key in required_buffer}
    mismatched = {key: length for key, length in lengths.items() if length != frame_count}
    if mismatched:
        return False, f"unaligned lengths: size={frame_count}, fields={mismatched}"

    timestamps = buffer["timestamp"]
    if any(float(b) <= float(a) for a, b in zip(timestamps, timestamps[1:])):
        return False, "timestamps are not strictly increasing"
    if any(not str(task).strip() for task in buffer["task"]):
        return False, "empty task string"
    return True, "ok"


def make_episode_keyboard_initializer(create_key_listener: Callable[..., Any], print_fn=print):
    """Return a non-blocking LeRobot-compatible keyboard initializer."""
    def initialize():
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "episode_outcome": None,
            "lifecycle_state": "prepare",
            "review_choice": None,
            "review_event": threading.Event(),
        }
        listener = create_key_listener(
            lambda name: apply_episode_key(name, events, print_fn),
            controls_help="Recording: Enter/Space=finish; Review: C=clean, V=recovery, R=retry, Q=quit",
        )
        return listener, events

    return initialize


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
    monotonic_fn: Callable[[], float] = time.perf_counter,
    review_wait_fn: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[..., Any]:
    """Add sync warmup and an explicit post-recording classification review."""

    def patched_record_loop(*args: Any, **kwargs: Any) -> Any:
        recording = kwargs.get("dataset") is not None
        if not recording:
            events = kwargs.get("events")
            if events is not None and events.get("lifecycle_state") == "handled":
                events["lifecycle_state"] = "reset"
                result = original_record_loop(*args, **kwargs)
                events["lifecycle_state"] = "prepare"
                return result
            return original_record_loop(*args, **kwargs)

        dataset = kwargs["dataset"]
        events = kwargs["events"]
        events["lifecycle_state"] = "prepare"
        events["lifecycle_state"] = "sync"
        _run_sync_warmup(original_record_loop, sync_warmup_s, kwargs, print_fn)
        events["episode_outcome"] = None
        events["review_choice"] = None
        events["lifecycle_state"] = "recording"
        review_event = events.get("review_event")
        if review_event is not None:
            review_event.clear()
        print_fn(f"EPISODE {getattr(dataset, 'num_episodes', '?')}")
        print_fn("RECORDING")
        print_fn("[ENTER/SPACE] finish recording")

        recording_started = monotonic_fn()
        result = original_record_loop(*args, **kwargs)
        duration = max(0.0, monotonic_fn() - recording_started)
        try:
            buffer = get_writer_episode_buffer(dataset)
        except EpisodeLabelError:
            buffer = {}
        frames = int(buffer.get("size", 0))

        if events.get("lifecycle_state") != "recording_finished":
            events["episode_outcome"] = "timeout"
            events["rerecord_episode"] = True
            dataset.clear_episode_buffer(delete_images=True)
            setattr(dataset, "_manual_episode_already_cleared", True)
            events["lifecycle_state"] = "handled"
            print_fn("[TIMEOUT] emergency limit reached: DISCARD/RETRY (not success).")
        else:
            events["lifecycle_state"] = "review"
            print_fn("=" * 68)
            print_fn(f"EPISODE {getattr(dataset, 'num_episodes', '?')} FINISHED")
            print_fn(f"duration: {duration:.2f} s")
            print_fn(f"frames: {frames}")
            print_fn(f"effective fps: {frames / duration if duration else 0.0:.2f}")
            print_fn("How should this episode be handled?")
            print_fn("[C] CLEAN SUCCESS | [V] RECOVERY SUCCESS | [R] DISCARD / RETRY | [Q] DISCARD / QUIT")
            if review_wait_fn is not None:
                review_wait_fn(events)
            else:
                events["review_event"].wait()
            choice = events.get("review_choice")
            if choice in ("c", "v"):
                outcome = "clean" if choice == "c" else "recovery"
                valid, reason = validate_episode_buffer(buffer, getattr(dataset, "features", None))
                if not valid:
                    events["episode_outcome"] = "integrity_error"
                    events["rerecord_episode"] = True
                    dataset.clear_episode_buffer(delete_images=True)
                    setattr(dataset, "_manual_episode_already_cleared", True)
                    print_fn(f"[INTEGRITY ERROR] DISCARD/RETRY: {reason}")
                else:
                    before = int(dataset.num_episodes)
                    setattr(dataset, "_pending_episode_type", outcome)
                    dataset.save_episode()
                    if int(dataset.num_episodes) != before + 1:
                        raise EpisodeLabelError("save_episode did not advance the dataset episode index")
                    setattr(dataset, "_manual_episode_already_saved", True)
                    events["episode_outcome"] = outcome
                    print_fn(
                        f"[SAVED] episode={before} type={outcome} "
                        f"frames={frames} duration={duration:.2f}s"
                    )
            else:
                quit_session = choice == "q"
                events["episode_outcome"] = "quit" if quit_session else "discard"
                events["rerecord_episode"] = True
                events["stop_recording"] = quit_session
                dataset.clear_episode_buffer(delete_images=True)
                setattr(dataset, "_manual_episode_already_cleared", True)
                print_fn(
                    "[QUIT] episode discarded."
                    if quit_session
                    else "[DISCARD] retrying the same episode index."
                )
            events["lifecycle_state"] = "handled"

        print_fn(
            f"[EPISODE END] duration={duration:.2f}s frames={frames} "
            f"effective_fps={frames / duration if duration else 0.0:.2f} "
            f"outcome={events['episode_outcome']} dataset_index="
            f"{getattr(dataset, 'num_episodes', '?')}"
        )
        return result

    return patched_record_loop



def make_labeled_save_episode(original_save_episode: Callable[..., Any]) -> Callable[..., Any]:
    """Label only after the official LeRobot save succeeds."""
    def labeled_save_episode(dataset: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(dataset, "_manual_episode_already_saved", False):
            delattr(dataset, "_manual_episode_already_saved")
            return None
        episode_type = getattr(dataset, "_pending_episode_type", None)
        if episode_type not in ("clean", "recovery"):
            raise EpisodeLabelError("save_episode called without an explicit clean/recovery label")
        episode_index = int(dataset.num_episodes)
        validate_label_alignment(dataset.root, episode_index)
        result = original_save_episode(dataset, *args, **kwargs)
        try:
            append_episode_label(
                dataset.root,
                episode_index=episode_index,
                episode_type=episode_type,
                dataset_episode_count=int(dataset.num_episodes),
            )
        except Exception as exc:
            raise EpisodeLabelError(
                f"episode {episode_index} was saved but its sidecar label failed; "
                "dataset/manifest mismatch requires manual inspection"
            ) from exc
        delattr(dataset, "_pending_episode_type")
        return result

    return labeled_save_episode


def make_idempotent_clear_episode_buffer(original_clear: Callable[..., Any]) -> Callable[..., Any]:
    """Suppress only the upstream clear following an already-completed discard."""
    def clear_episode_buffer(dataset: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(dataset, "_manual_episode_already_cleared", False):
            delattr(dataset, "_manual_episode_already_cleared")
            return None
        return original_clear(dataset, *args, **kwargs)

    return clear_episode_buffer


def main() -> int:
    sync_warmup_s, remaining_argv = parse_runner_args(sys.argv[1:])

    import lerobot.scripts.lerobot_record as lerobot_record_module
    from lerobot.utils.keyboard_input import create_key_listener
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    original_record_loop = lerobot_record_module.record_loop
    LeRobotDataset.save_episode = make_labeled_save_episode(LeRobotDataset.save_episode)
    LeRobotDataset.clear_episode_buffer = make_idempotent_clear_episode_buffer(
        LeRobotDataset.clear_episode_buffer
    )
    lerobot_record_module.record_loop = make_patched_record_loop(
        original_record_loop, sync_warmup_s
    )
    lerobot_record_module.init_keyboard_listener = make_episode_keyboard_initializer(
        create_key_listener
    )

    sys.argv = [sys.argv[0], *remaining_argv]
    lerobot_record_module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
