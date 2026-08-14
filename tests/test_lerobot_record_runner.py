"""data_collection/lerobot_record_runner.py 단위 테스트.

실제 로봇/시리얼 포트/카메라에 절대 접근하지 않는다. ``lerobot`` 패키지도 이 테스트
프로세스에는 설치되어 있지 않을 수 있으므로 (별도 LeRobot venv에서만 실행되는 스크립트),
``main()`` 안의 ``import lerobot...`` 줄은 절대 실행하지 않고 순수 함수
(``parse_runner_args`` / ``iter_warmup_chunks`` / ``make_patched_record_loop``)만
가짜(fake) ``record_loop`` 콜백으로 검증한다.

핵심 검증 대상 (요청 사항 8):
  * warmup 동안 dataset이 항상 None으로 호출되는지 (parquet/video frame 미기록)
  * warmup 총 시간이 --sync-warmup-seconds 값과 정확히 일치하는지 (1초 단위 countdown 유지)
  * warmup이 reset 구간(dataset=None으로 "진짜" 호출되는 경우)에는 끼어들지 않는지
  * 실제 recording 호출(dataset != None)의 kwargs(episode_time_s 등)가 그대로 전달되는지
    -> episode_time_s/frame 수가 warmup 때문에 바뀌지 않음을 보장
  * "녹화를 시작합니다" 문구가 실제 recording 호출 직전, 가장 마지막으로 출력되는지
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.lerobot_record_runner import (
    DEFAULT_SYNC_WARMUP_S,
    apply_episode_key,
    make_episode_keyboard_initializer,
    make_idempotent_clear_episode_buffer,
    make_labeled_save_episode,
    validate_episode_buffer,
    get_writer_episode_buffer,
    iter_warmup_chunks,
    make_patched_record_loop,
    parse_runner_args,
)


# ---------------------------------------------------------------------------
# parse_runner_args
# ---------------------------------------------------------------------------


def test_parse_runner_args_extracts_space_separated_flag():
    sync_s, remaining = parse_runner_args(
        ["--sync-warmup-seconds", "2.5", "--robot.type=so101_follower"]
    )
    assert sync_s == 2.5
    assert remaining == ["--robot.type=so101_follower"]


def test_parse_runner_args_extracts_equals_form():
    sync_s, remaining = parse_runner_args(["--sync-warmup-seconds=1.5", "--dataset.fps=30"])
    assert sync_s == 1.5
    assert remaining == ["--dataset.fps=30"]


def test_parse_runner_args_defaults_when_flag_absent():
    sync_s, remaining = parse_runner_args(["--robot.type=so101_follower", "--dataset.fps=30"])
    assert sync_s == DEFAULT_SYNC_WARMUP_S
    assert remaining == ["--robot.type=so101_follower", "--dataset.fps=30"]


def test_parse_runner_args_never_leaks_flag_into_remaining():
    _, remaining = parse_runner_args(["--sync-warmup-seconds=3", "--a=1", "--b=2"])
    assert all("sync-warmup-seconds" not in arg for arg in remaining)


def test_parse_runner_args_rejects_negative_value():
    with pytest.raises(ValueError):
        parse_runner_args(["--sync-warmup-seconds=-1"])


def test_parse_runner_args_rejects_missing_value():
    with pytest.raises(ValueError):
        parse_runner_args(["--sync-warmup-seconds"])


# ---------------------------------------------------------------------------
# iter_warmup_chunks
# ---------------------------------------------------------------------------


def test_warmup_chunks_sum_to_integer_total():
    chunks = list(iter_warmup_chunks(3.0))
    assert [label for label, _ in chunks] == [3, 2, 1]
    assert sum(chunk for _, chunk in chunks) == pytest.approx(3.0)


def test_warmup_chunks_handle_fractional_total_naturally():
    chunks = list(iter_warmup_chunks(3.5))
    assert sum(chunk for _, chunk in chunks) == pytest.approx(3.5)
    # every chunk is at most 1s so a 1-second countdown line can be printed between them
    assert all(chunk <= 1.0 + 1e-9 for _, chunk in chunks)
    assert [label for label, _ in chunks] == [4, 3, 2, 1]


def test_warmup_chunks_zero_total_yields_nothing():
    assert list(iter_warmup_chunks(0.0)) == []


def test_warmup_chunks_sub_second_total():
    chunks = list(iter_warmup_chunks(0.4))
    assert chunks == [(1, pytest.approx(0.4))]


# ---------------------------------------------------------------------------
# make_patched_record_loop
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Hardware-free stand-in for the LeRobot dataset facade/writer lifecycle."""

    def __init__(self, frames: int = 1):
        self.num_episodes = 0
        self.features = {
            "observation.state": {}, "action": {},
            "observation.images.workspace": {}, "observation.images.wrist": {},
        }
        self.writer = type("Writer", (), {"episode_buffer": _valid_episode_buffer(frames)})()
        self.save_calls = 0
        self.clear_calls = 0

    def save_episode(self):
        self.save_calls += 1
        self.num_episodes += 1
        self.writer.episode_buffer = _valid_episode_buffer(0)

    def clear_episode_buffer(self, delete_images=True):
        assert delete_images is True
        self.clear_calls += 1
        self.writer.episode_buffer = _valid_episode_buffer(0)


def _make_fake_original_record_loop(calls: list[dict]):
    def fake_record_loop(*, dataset=None, **kwargs):
        # Mirrors the real record_loop's signature: `dataset` defaults to None
        # when the caller omits it (the reset-phase call pattern).
        calls.append({"dataset": dataset, **kwargs})
        return None

    return fake_record_loop


def _base_recording_kwargs(**overrides) -> dict:
    kwargs = dict(
        robot="ROBOT",
        events={"exit_early": False, "rerecord_episode": False, "stop_recording": False},
        fps=30,
        teleop_action_processor="TELEOP_PROC",
        robot_action_processor="ROBOT_ACTION_PROC",
        robot_observation_processor="ROBOT_OBS_PROC",
        teleop="TELEOP",
        single_task="Pick up the cube.",
        display_data=False,
        display_mode="rerun",
        display_compressed_images=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_recording_call_is_preceded_by_warmup_chunks_with_dataset_none():
    calls: list[dict] = []
    original = _make_fake_original_record_loop(calls)
    patched = make_patched_record_loop(original, sync_warmup_s=3.0, print_fn=lambda _: None)

    dataset = _FakeDataset()
    patched(**_base_recording_kwargs(dataset=dataset, control_time_s=10.0))

    # 3 warmup chunks (dataset=None) + 1 real recording call (dataset=dataset)
    assert len(calls) == 4
    warmup_calls, real_call = calls[:3], calls[3]

    for call in warmup_calls:
        assert call["dataset"] is None
    assert sum(call["control_time_s"] for call in warmup_calls) == pytest.approx(3.0)

    # the real call's kwargs (dataset, control_time_s=episode_time_s, ...) reach
    # the original record_loop completely unmodified
    assert real_call["dataset"] is dataset
    assert real_call["control_time_s"] == 10.0
    assert real_call["single_task"] == "Pick up the cube."
    assert real_call["robot"] == "ROBOT"
    assert real_call["teleop"] == "TELEOP"


def test_reset_call_dataset_none_passes_through_without_warmup():
    calls: list[dict] = []
    original = _make_fake_original_record_loop(calls)
    patched = make_patched_record_loop(original, sync_warmup_s=3.0, print_fn=lambda _: None)

    # Reset-phase calls never pass `dataset` at all (defaults to None) - exactly
    # how lerobot.scripts.lerobot_record.record() calls record_loop for reset.
    patched(**_base_recording_kwargs(control_time_s=20.0))

    assert len(calls) == 1
    assert calls[0]["dataset"] is None
    assert calls[0]["control_time_s"] == 20.0


def test_warmup_runs_before_every_recording_call_not_just_the_first():
    """Empirically (reports/grid35_episode_start_analysis) the sync jump shows up at
    frame 0 of *every* episode, not only the first - so every real recording call
    must get its own warmup, not just the first one in the process."""
    calls: list[dict] = []
    original = _make_fake_original_record_loop(calls)
    patched = make_patched_record_loop(original, sync_warmup_s=1.0, print_fn=lambda _: None)

    ds1, ds2 = _FakeDataset(), _FakeDataset()
    patched(**_base_recording_kwargs(dataset=ds1, control_time_s=10.0))  # episode 1
    patched(**_base_recording_kwargs(control_time_s=20.0))  # reset
    patched(**_base_recording_kwargs(dataset=ds2, control_time_s=10.0))  # episode 2

    # 1 warmup chunk + real call, reset call, 1 warmup chunk + real call
    dataset_sequence = [call["dataset"] for call in calls]
    assert dataset_sequence == [None, ds1, None, None, ds2]


def test_zero_sync_warmup_skips_chunks_but_still_calls_real_loop():
    calls: list[dict] = []
    original = _make_fake_original_record_loop(calls)
    patched = make_patched_record_loop(original, sync_warmup_s=0.0, print_fn=lambda _: None)

    dataset = _FakeDataset()
    patched(**_base_recording_kwargs(dataset=dataset, control_time_s=10.0))

    assert len(calls) == 1
    assert calls[0]["dataset"] is dataset


def test_recording_message_is_last_line_printed_before_the_real_call():
    printed: list[str] = []
    call_order: list[str] = []

    def fake_record_loop(**kwargs):
        call_order.append("real_dataset_call" if kwargs["dataset"] is not None else "warmup_call")
        return None

    patched = make_patched_record_loop(fake_record_loop, sync_warmup_s=2.0, print_fn=printed.append)
    patched(**_base_recording_kwargs(dataset=_FakeDataset(), control_time_s=10.0))

    assert "🔴 녹화를 시작합니다!" in printed
    assert printed.index("🔴 녹화를 시작합니다!") < printed.index("RECORDING")
    # the printed "start" line is immediately followed by the real (non-warmup) call
    assert call_order[-1] == "real_dataset_call"
    assert call_order.count("warmup_call") == 2


def test_korean_guidance_lines_appear_in_order():
    printed: list[str] = []
    patched = make_patched_record_loop(
        _make_fake_original_record_loop([]), sync_warmup_s=3.0, print_fn=printed.append
    )
    patched(**_base_recording_kwargs(dataset=_FakeDataset(), control_time_s=10.0))

    expected_in_order = [
        "[준비] 리더암과 팔로워암을 동기화합니다.",
        "[주의] 팔로워암이 움직일 수 있습니다. 리더암을 시작 자세로 유지하세요.",
        "[동기화] 안정화까지 3...",
        "[동기화] 안정화까지 2...",
        "[동기화] 안정화까지 1...",
        "[완료] 동기화 안정화가 끝났습니다.",
        "[녹화 준비] 시작 자세를 유지하세요.",
        "🔴 녹화를 시작합니다!",
    ]
    indices = [printed.index(line) for line in expected_in_order]
    assert indices == sorted(indices)


def test_missing_required_kwarg_fails_loudly_instead_of_silently_skipping_warmup():
    """If lerobot's record_loop signature ever drops one of these kwargs, warmup
    should error out clearly rather than silently recording without sync."""
    patched = make_patched_record_loop(
        _make_fake_original_record_loop([]), sync_warmup_s=1.0, print_fn=lambda _: None
    )
    incomplete_kwargs = _base_recording_kwargs(dataset=_FakeDataset(), control_time_s=10.0)
    del incomplete_kwargs["robot"]

    with pytest.raises(KeyError):
        patched(**incomplete_kwargs)


# ---------------------------------------------------------------------------
# Explicit episode completion lifecycle
# ---------------------------------------------------------------------------


def _valid_episode_buffer(frames: int, fps: int = 30) -> dict:
    return {
        "size": frames,
        "episode_index": 0,
        "observation.state": [[0.0] * 6 for _ in range(frames)],
        "action": [[0.0] * 6 for _ in range(frames)],
        "observation.images.workspace": [object() for _ in range(frames)],
        "observation.images.wrist": [object() for _ in range(frames)],
        "timestamp": [i / fps for i in range(frames)],
        "frame_index": list(range(frames)),
        "task": ["Pick up blue cube."] * frames,
    }


def _events() -> dict:
    return {
        "exit_early": False, "rerecord_episode": False,
        "stop_recording": False, "episode_outcome": None,
        "lifecycle_state": "prepare", "review_choice": None,
    }


def _run_review(choice: str, *, frames: int = 150, order: list[str] | None = None):
    dataset = _FakeDataset(frames)
    events = _events()

    def original(**kwargs):
        apply_episode_key("enter", kwargs["events"], lambda _: None)
        if order is not None:
            order.append("record_stopped")

    def review(current_events):
        assert dataset.writer.episode_buffer["size"] == frames
        assert dataset.save_calls == 0
        assert dataset.clear_calls == 0
        if order is not None:
            order.append("review")
        apply_episode_key(choice, current_events, lambda _: None)

    clock_values = iter((10.0, 15.0))
    patched = make_patched_record_loop(
        original, 0, print_fn=lambda _: None,
        monotonic_fn=lambda: next(clock_values), review_wait_fn=review,
    )
    patched(**_base_recording_kwargs(dataset=dataset, events=events, control_time_s=60.0))
    return dataset, events


@pytest.mark.parametrize("key", ["enter", "space"])
def test_recording_finish_key_enters_review_without_classifying_or_saving(key):
    events = _events()
    events["lifecycle_state"] = "recording"
    apply_episode_key(key, events, lambda _: None)
    assert events["lifecycle_state"] == "recording_finished"
    assert events["episode_outcome"] is None
    assert events["exit_early"] is True


@pytest.mark.parametrize("key", ["c", "v", "r", "q"])
def test_recording_state_ignores_classification_keys(key):
    events = _events()
    events["lifecycle_state"] = "recording"
    apply_episode_key(key, events, lambda _: None)
    assert events["exit_early"] is False
    assert events["review_choice"] is None


def test_review_enter_does_not_implicitly_select_clean():
    events = _events()
    events["lifecycle_state"] = "review"
    apply_episode_key("enter", events, lambda _: None)
    assert events["review_choice"] is None


@pytest.mark.parametrize(("key", "outcome"), [("c", "clean"), ("v", "recovery")])
def test_review_success_saves_exact_buffer_and_sets_manual_label(key, outcome):
    dataset, events = _run_review(key)
    assert dataset.save_calls == 1
    assert dataset.num_episodes == 1
    assert dataset.clear_calls == 0
    assert events["episode_outcome"] == outcome
    assert dataset._pending_episode_type == outcome


def test_review_discard_clears_without_advancing_index_or_label():
    dataset, events = _run_review("r")
    assert dataset.save_calls == 0
    assert dataset.clear_calls == 1
    assert dataset.num_episodes == 0
    assert not hasattr(dataset, "_pending_episode_type")
    assert events["rerecord_episode"] is True
    assert events["stop_recording"] is False


def test_review_quit_discards_and_stops_without_label():
    dataset, events = _run_review("q")
    assert dataset.save_calls == 0
    assert dataset.clear_calls == 1
    assert dataset.num_episodes == 0
    assert not hasattr(dataset, "_pending_episode_type")
    assert events["stop_recording"] is True


def test_classification_and_save_complete_before_caller_can_reset():
    order = []
    dataset, _ = _run_review("c", order=order)
    order.append("reset")
    assert order == ["record_stopped", "review", "reset"]
    assert dataset.num_episodes == 1


@pytest.mark.parametrize("seconds", [3, 8, 15])
def test_variable_duration_episode_buffers_are_valid(seconds):
    valid, reason = validate_episode_buffer(_valid_episode_buffer(seconds * 30))
    assert valid is True
    assert reason == "ok"


def test_emergency_timeout_discards_without_review_or_success():
    events = _events()
    dataset = _FakeDataset(150)
    reviewed = []
    patched = make_patched_record_loop(
        lambda **_: None, sync_warmup_s=0, print_fn=lambda _: None,
        review_wait_fn=lambda _: reviewed.append(True),
    )
    patched(**_base_recording_kwargs(dataset=dataset, events=events, control_time_s=5.0))
    assert events["episode_outcome"] == "timeout"
    assert events["rerecord_episode"] is True
    assert dataset.clear_calls == 1
    assert dataset.save_calls == 0
    assert reviewed == []


def test_keyboard_initializer_only_registers_nonblocking_callback():
    captured = {}
    sentinel = object()

    def create_listener(dispatch, **kwargs):
        captured["dispatch"] = dispatch
        captured["help"] = kwargs["controls_help"]
        return sentinel

    listener, events = make_episode_keyboard_initializer(create_listener, lambda _: None)()
    assert listener is sentinel
    assert events["exit_early"] is False
    events["lifecycle_state"] = "recording"
    captured["dispatch"]("space")
    assert events["lifecycle_state"] == "recording_finished"


def test_integrity_rejects_unaligned_or_nonmonotonic_samples():
    buffer = _valid_episode_buffer(10)
    buffer["action"].pop()
    assert validate_episode_buffer(buffer)[0] is False
    buffer = _valid_episode_buffer(10)
    buffer["timestamp"][5] = buffer["timestamp"][4]
    assert validate_episode_buffer(buffer)[0] is False


def test_real_lerobot_facade_uses_writer_episode_buffer():
    buffer = _valid_episode_buffer(150)
    dataset = type("Dataset", (), {
        "writer": type("Writer", (), {"episode_buffer": buffer})(),
    })()
    assert get_writer_episode_buffer(dataset) is buffer
    assert validate_episode_buffer(get_writer_episode_buffer(dataset))[0] is True


def test_empty_real_writer_buffer_fails():
    assert validate_episode_buffer(_valid_episode_buffer(0))[0] is False


def test_missing_single_camera_fails():
    buffer = _valid_episode_buffer(30)
    del buffer["observation.images.wrist"]
    valid, reason = validate_episode_buffer(buffer)
    assert valid is False
    assert "wrist" in reason


def test_actual_writer_buffer_allows_streaming_video_none_placeholders():
    buffer = _valid_episode_buffer(30)
    buffer["observation.images.workspace"] = [None] * 30
    buffer["observation.images.wrist"] = [None] * 30
    assert validate_episode_buffer(buffer)[0] is True


def test_episode_report_uses_writer_size_and_wall_duration():
    printed = []
    dataset = _FakeDataset(150)
    events = _events()

    def original(**kwargs):
        apply_episode_key("enter", kwargs["events"], lambda _: None)

    def review(current_events):
        apply_episode_key("c", current_events, lambda _: None)

    clock_values = iter((10.0, 15.0))
    patched = make_patched_record_loop(
        original, 0, print_fn=printed.append,
        monotonic_fn=lambda: next(clock_values), review_wait_fn=review,
    )
    patched(**_base_recording_kwargs(dataset=dataset, events=events, control_time_s=60.0))
    summary = next(line for line in printed if line.startswith("[EPISODE END]"))
    assert "duration=5.00s" in summary
    assert "frames=150" in summary
    assert events["episode_outcome"] == "clean"



def test_upstream_post_review_save_is_suppressed_exactly_once():
    calls = []

    def original(dataset):
        calls.append("save")

    wrapped = make_labeled_save_episode(original)
    dataset = type("Dataset", (), {})()
    dataset._manual_episode_already_saved = True
    wrapped(dataset)
    assert calls == []
    assert not hasattr(dataset, "_manual_episode_already_saved")


def test_upstream_post_review_clear_is_suppressed_exactly_once():
    calls = []

    def original(dataset, *args, **kwargs):
        calls.append((args, kwargs))

    wrapped = make_idempotent_clear_episode_buffer(original)
    dataset = type("Dataset", (), {})()
    dataset._manual_episode_already_cleared = True
    wrapped(dataset)
    assert calls == []
    assert not hasattr(dataset, "_manual_episode_already_cleared")
    wrapped(dataset, delete_images=True)
    assert calls == [((), {"delete_images": True})]


def test_reset_call_transitions_only_after_episode_is_handled():
    observed = []
    events = _events()
    events["lifecycle_state"] = "handled"

    def original(**kwargs):
        observed.append(kwargs["events"]["lifecycle_state"])

    patched = make_patched_record_loop(original, 0, print_fn=lambda _: None)
    patched(**_base_recording_kwargs(events=events, control_time_s=2.0))
    assert observed == ["reset"]
    assert events["lifecycle_state"] == "prepare"
