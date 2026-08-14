"""data_collection/recorder.py 의 sync-warmup 관련 변경 단위/드라이런 테스트.

실제 로봇을 움직이지 않는다: 이 파일의 모든 테스트는 dry_run=True로만
``record_episodes``를 호출하거나(= 장치 검사와 명령 생성까지만 수행하고
subprocess를 실행하지 않음), 순수 함수(``build_record_command`` 등)만 검증한다.
"/dev/..." 장치 경로도 전부 tmp_path 아래 가짜 파일로 대체한다.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.config import CameraConfig, DeviceConfig, HardwareConfig
from data_collection.recorder import (
    RecordingError,
    RecordingRequest,
    _resolve_lerobot_python,
    build_lerobot_record_args,
    build_record_command,
    record_episodes,
)

REAL_LEROBOT_RECORD = Path("/home/sunglee/lerobot/lerobot/bin/lerobot-record")


def _hardware(**overrides) -> HardwareConfig:
    base = dict(
        robot=DeviceConfig(type="so101_follower", port="/fake/follower", id="test_follower"),
        teleop=DeviceConfig(type="so101_leader", port="/fake/leader", id="test_leader"),
        cameras={
            "workspace": CameraConfig(type="opencv", index_or_path="/fake/cam0", width=640, height=480, fps=30),
            "wrist": CameraConfig(type="opencv", index_or_path="/fake/cam1", width=640, height=480, fps=30),
        },
    )
    base.update(overrides)
    return HardwareConfig(**base)


def _request(**overrides) -> RecordingRequest:
    base = dict(
        dataset_name="unit_test_dataset",
        data_dir=Path("/tmp/does-not-matter"),
        task="Pick up the cube and place it in the target area.",
        num_episodes=1,
        max_episode_duration_s=10.0,
        reset_time_s=5.0,
    )
    base.update(overrides)
    return RecordingRequest(**base)


# ---------------------------------------------------------------------------
# RecordingRequest.sync_warmup_s
# ---------------------------------------------------------------------------


def test_sync_warmup_defaults_to_3_seconds():
    assert _request().sync_warmup_s == 3.0


def test_sync_warmup_negative_rejected():
    with pytest.raises(RecordingError):
        _request(sync_warmup_s=-0.5).validate()


def test_sync_warmup_zero_allowed():
    _request(sync_warmup_s=0.0).validate()  # must not raise


def test_existing_validations_untouched():
    with pytest.raises(RecordingError):
        _request(max_episode_duration_s=0).validate()
    with pytest.raises(RecordingError):
        _request(num_episodes=0).validate()
    with pytest.raises(RecordingError):
        _request(dataset_name="has space").validate()


# ---------------------------------------------------------------------------
# build_lerobot_record_args - the real lerobot-record flags must be untouched
# ---------------------------------------------------------------------------


def test_lerobot_record_args_have_no_sync_warmup_flag_and_keep_existing_flags():
    hardware = _hardware()
    request = _request(max_episode_duration_s=10.0, reset_time_s=5.0, resume=True)
    args = build_lerobot_record_args(hardware, request)

    assert not any("sync-warmup" in arg or "sync_warmup" in arg for arg in args)
    assert "--dataset.episode_time_s=10.0" in args
    assert "--dataset.reset_time_s=5.0" in args
    assert "--resume=true" in args
    assert any(arg.startswith("--robot.type=so101_follower") for arg in args)
    assert any(arg.startswith("--teleop.type=so101_leader") for arg in args)


# ---------------------------------------------------------------------------
# build_record_command - full subprocess argv routes through the runner script
# ---------------------------------------------------------------------------


def test_build_record_command_routes_through_runner_with_warmup_flag(tmp_path):
    runner = tmp_path / "lerobot_record_runner.py"
    runner.write_text("# fake runner\n", encoding="utf-8")

    hardware = _hardware()
    request = _request(sync_warmup_s=2.5)
    command = build_record_command(
        hardware, request, python_executable="/fake/venv/bin/python", runner_script=runner
    )

    assert command[0] == "/fake/venv/bin/python"
    assert command[1] == str(runner)
    assert command[2] == "--sync-warmup-seconds=2.5"
    # everything else is exactly the lerobot-record flags, unchanged and in order
    assert command[3:] == build_lerobot_record_args(hardware, request)


def test_build_record_command_formats_default_warmup_without_trailing_zeros(tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text("", encoding="utf-8")
    command = build_record_command(
        _hardware(), _request(), python_executable="/fake/python", runner_script=runner
    )
    assert command[2] == "--sync-warmup-seconds=3"


# ---------------------------------------------------------------------------
# _resolve_lerobot_python
# ---------------------------------------------------------------------------


def test_resolve_lerobot_python_reads_shebang(tmp_path):
    script = tmp_path / "lerobot-record"
    script.write_text("#!/fake/venv/bin/python\nimport sys\n", encoding="utf-8")
    assert _resolve_lerobot_python(str(script)) == "/fake/venv/bin/python"


def test_resolve_lerobot_python_rejects_non_shebang_file(tmp_path):
    script = tmp_path / "lerobot-record"
    script.write_text("not a shebang\n", encoding="utf-8")
    with pytest.raises(RecordingError):
        _resolve_lerobot_python(str(script))


def test_resolve_lerobot_python_matches_real_installed_launcher():
    if os.name == "nt":
        pytest.skip("POSIX LeRobot launcher path is validated in the WSL runtime.")
    if not REAL_LEROBOT_RECORD.is_file():
        pytest.skip(f"실제 lerobot-record 런처를 찾을 수 없습니다: {REAL_LEROBOT_RECORD}")
    interpreter = _resolve_lerobot_python(str(REAL_LEROBOT_RECORD))
    assert interpreter.endswith("/python")
    assert Path(interpreter).is_file()


# ---------------------------------------------------------------------------
# record_episodes(..., dry_run=True) end to end - no subprocess is ever run
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_hardware_config(tmp_path) -> Path:
    follower = tmp_path / "follower_port"
    leader = tmp_path / "leader_port"
    cam0 = tmp_path / "cam0"
    cam1 = tmp_path / "cam1"
    for p in (follower, leader, cam0, cam1):
        p.touch()

    config_path = tmp_path / "hardware.json"
    config_path.write_text(
        f"""
        {{
          "robot": {{"type": "so101_follower", "port": "{follower}", "id": "test_follower"}},
          "teleop": {{"type": "so101_leader", "port": "{leader}", "id": "test_leader"}},
          "cameras": {{
            "workspace": {{"type": "opencv", "index_or_path": "{cam0}", "width": 640, "height": 480, "fps": 30}},
            "wrist": {{"type": "opencv", "index_or_path": "{cam1}", "width": 640, "height": 480, "fps": 30}}
          }}
        }}
        """,
        encoding="utf-8",
    )
    return config_path


def test_record_episodes_dry_run_needs_lerobot_record_on_path(fake_hardware_config, tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX PATH dry-run is validated in the WSL runtime.")
    if shutil.which("lerobot-record") is None and not REAL_LEROBOT_RECORD.is_file():
        pytest.skip("lerobot-record가 이 머신에 없습니다 (LeRobot venv 필요).")

    lerobot_bin_dir = REAL_LEROBOT_RECORD.parent
    monkeypatch.setenv("PATH", f"{lerobot_bin_dir}:{shutil.os.environ.get('PATH', '')}")

    request = _request(
        dataset_name="unit_test_dry_run_dataset",
        data_dir=tmp_path / "data",
        num_episodes=1,
        max_episode_duration_s=10.0,
        reset_time_s=5.0,
        sync_warmup_s=3.0,
    )

    dataset_root = record_episodes(fake_hardware_config, request, dry_run=True, assume_yes=True)

    # dry-run must not create/require the dataset folder, and must never touch real hardware
    assert dataset_root == request.dataset_root


def test_record_episodes_dry_run_missing_lerobot_record_raises_clear_error(
    fake_hardware_config, tmp_path, monkeypatch
):
    monkeypatch.setenv("PATH", "/nonexistent-empty-bin-dir")
    request = _request(dataset_name="ds", data_dir=tmp_path / "data")
    with pytest.raises(RecordingError, match="lerobot-record"):
        record_episodes(fake_hardware_config, request, dry_run=True, assume_yes=True)
