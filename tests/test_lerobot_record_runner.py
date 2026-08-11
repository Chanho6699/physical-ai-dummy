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
    """Stand-in for LeRobotDataset - identity is all that matters here."""


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

    assert printed[-1] == "🔴 녹화를 시작합니다!"
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
