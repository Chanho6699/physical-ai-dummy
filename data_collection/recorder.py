from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import HardwareConfig, load_hardware_config

# The runner script that lerobot-record's own Python interpreter executes
# instead of the plain `lerobot-record` console script, so a sync-warmup
# stage can run inside the real LeRobot recording lifecycle. See its module
# docstring for why this indirection is necessary.
_RUNNER_SCRIPT = Path(__file__).resolve().parent / "lerobot_record_runner.py"


class RecordingError(RuntimeError):
    """Raised when an episode recording cannot be started or completed."""


@dataclass(frozen=True)
class RecordingRequest:
    dataset_name: str
    data_dir: Path
    task: str
    num_episodes: int = 5
    episode_time_s: float = 30.0
    reset_time_s: float = 20.0
    fps: int = 30
    resume: bool = False
    display_data: bool = False
    streaming_encoding: bool = False
    # Seconds of active leader->follower teleoperation, held right after the
    # reset phase (and after the initial connect, for episode 1) *before* the
    # real dataset recording starts. Lets the follower settle onto the
    # leader's current pose so that catch-up motion isn't captured as the
    # episode's first recorded frames. Does not shrink episode_time_s /
    # reset_time_s - it is strictly additional time. See
    # data_collection/lerobot_record_runner.py for where this is enforced.
    sync_warmup_s: float = 3.0

    @property
    def dataset_root(self) -> Path:
        return self.data_dir.expanduser().resolve() / self.dataset_name

    @property
    def repo_id(self) -> str:
        return f"local/{self.dataset_name}"

    def validate(self) -> None:
        if not self.dataset_name or any(ch.isspace() for ch in self.dataset_name):
            raise RecordingError("dataset_name must be non-empty and contain no spaces.")
        if "/" in self.dataset_name or "\\" in self.dataset_name:
            raise RecordingError("dataset_name must be a single folder name, not a path.")
        if self.num_episodes <= 0:
            raise RecordingError("num_episodes must be greater than 0.")
        if self.episode_time_s <= 0:
            raise RecordingError("episode_time_s must be greater than 0.")
        if self.reset_time_s < 0:
            raise RecordingError("reset_time_s must be 0 or greater.")
        if self.fps <= 0:
            raise RecordingError("fps must be greater than 0.")
        if not self.task.strip():
            raise RecordingError("task must not be empty.")
        if self.sync_warmup_s < 0:
            raise RecordingError("sync_warmup_s must be 0 or greater.")


def _bool_cli(value: bool) -> str:
    return "true" if value else "false"


def _check_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RecordingError(
            f"'{command}' was not found. Activate the LeRobot environment first:\n"
            "  source ~/lerobot/lerobot/bin/activate"
        )


def _resolve_lerobot_python(lerobot_record_path: str) -> str:
    """Return the interpreter ``lerobot-record`` itself runs under.

    ``lerobot-record`` is a pip-generated console-script whose first line is
    a shebang pointing straight at the LeRobot venv's Python (e.g.
    ``#!/home/user/lerobot/lerobot/bin/python``). Reusing that exact
    interpreter - instead of assuming ``python3`` on PATH resolves to the
    same venv - is what lets ``lerobot_record_runner.py`` ``import
    lerobot.scripts.lerobot_record`` regardless of shell configuration,
    while still requiring the same "activate the LeRobot environment first"
    precondition as before (``_check_command`` above).
    """
    try:
        with open(lerobot_record_path, encoding="utf-8") as handle:
            first_line = handle.readline()
    except OSError as exc:
        raise RecordingError(
            f"Could not read the 'lerobot-record' launcher to find its Python interpreter: {exc}"
        ) from exc

    if not first_line.startswith("#!"):
        raise RecordingError(
            "'lerobot-record' does not look like a Python console-script "
            f"(no shebang line): {lerobot_record_path}"
        )

    interpreter = first_line[2:].strip()
    if not interpreter:
        raise RecordingError(f"Empty shebang in 'lerobot-record' launcher: {lerobot_record_path}")
    return interpreter


def _check_device_path(path: str, label: str) -> None:
    expanded = Path(path).expanduser()
    if not expanded.exists():
        raise RecordingError(f"{label} device does not exist: {expanded}")


def check_hardware(hardware: HardwareConfig) -> None:
    _check_device_path(hardware.robot.port, "Follower")
    _check_device_path(hardware.teleop.port, "Leader")

    for role, camera in hardware.cameras.items():
        if isinstance(camera.index_or_path, str):
            _check_device_path(camera.index_or_path, f"Camera '{role}'")

    fps_values = {camera.fps for camera in hardware.cameras.values()}
    if len(fps_values) > 1:
        raise RecordingError(
            "All cameras should use the same FPS for this collection pipeline: "
            f"{sorted(fps_values)}"
        )


def build_lerobot_record_args(
    hardware: HardwareConfig,
    request: RecordingRequest,
) -> list[str]:
    """CLI flags for the real ``lerobot-record`` entry point (unchanged behavior)."""
    request.validate()

    cameras = {
        role: camera.to_lerobot_dict()
        for role, camera in hardware.cameras.items()
    }

    return [
        f"--robot.type={hardware.robot.type}",
        f"--robot.port={hardware.robot.port}",
        f"--robot.id={hardware.robot.id}",
        f"--robot.cameras={json.dumps(cameras, ensure_ascii=False)}",
        f"--teleop.type={hardware.teleop.type}",
        f"--teleop.port={hardware.teleop.port}",
        f"--teleop.id={hardware.teleop.id}",
        f"--dataset.repo_id={request.repo_id}",
        f"--dataset.root={request.dataset_root}",
        f"--dataset.single_task={request.task}",
        f"--dataset.fps={request.fps}",
        f"--dataset.episode_time_s={request.episode_time_s}",
        f"--dataset.reset_time_s={request.reset_time_s}",
        f"--dataset.num_episodes={request.num_episodes}",
        "--dataset.video=true",
        "--dataset.push_to_hub=false",
        "--dataset.no_stamp=true",
        f"--dataset.streaming_encoding={_bool_cli(request.streaming_encoding)}",
        f"--display_data={_bool_cli(request.display_data)}",
        "--play_sounds=false",
        f"--resume={_bool_cli(request.resume)}",
    ]


def build_record_command(
    hardware: HardwareConfig,
    request: RecordingRequest,
    *,
    python_executable: str,
    runner_script: Path = _RUNNER_SCRIPT,
) -> list[str]:
    """Full subprocess argv: the LeRobot venv's Python running the sync-warmup
    runner script, which in turn parses ``--sync-warmup-seconds`` and forwards
    every other flag to the real ``lerobot-record`` (``lerobot.scripts.lerobot_record``).
    """
    return [
        python_executable,
        str(runner_script),
        f"--sync-warmup-seconds={request.sync_warmup_s:g}",
        *build_lerobot_record_args(hardware, request),
    ]


def _print_korean_guide(request: RecordingRequest, hardware: HardwareConfig) -> None:
    camera_roles = ", ".join(hardware.cameras)
    print("\n" + "=" * 68)
    print("[준비] SO-101 실물 에피소드 녹화")
    print(f"[데이터셋] {request.dataset_name}")
    print(f"[작업] {request.task}")
    print(f"[에피소드] {request.num_episodes}개")
    print(f"[녹화 시간] 각 {request.episode_time_s:g}초")
    print(f"[초기화 시간] 각 {request.reset_time_s:g}초")
    print(f"[동기화 대기] 매 녹화 시작 전 {request.sync_warmup_s:g}초 (리더/팔로워 안정화, 녹화 시간에 미포함)")
    print(f"[카메라] {camera_roles}")
    print("-" * 68)
    print("[동기화] 매 녹화 직전 리더암과 팔로워암이 동기화될 때까지 잠시 대기합니다.")
    print("[녹화 중] 작업을 자연스럽게 완료하고 마지막 자세에서 멈추세요.")
    print("[초기화 중] 녹화 종료 안내 뒤에만 물체와 팔을 원위치로 옮기세요.")
    print("[종료] 마지막 에피소드 후 프롬프트가 돌아올 때까지 기다리세요.")
    print("[비상] 충돌·과부하 위험일 때만 Ctrl+C를 사용하세요.")
    print("=" * 68 + "\n")


def record_episodes(
    hardware_config_path: str | Path,
    request: RecordingRequest,
    *,
    dry_run: bool = False,
    assume_yes: bool = False,
) -> Path:
    """Validate devices, then delegate synchronized capture to ``lerobot-record``."""

    request.validate()
    _check_command("lerobot-record")
    hardware = load_hardware_config(hardware_config_path)
    check_hardware(hardware)

    if not _RUNNER_SCRIPT.is_file():
        raise RecordingError(f"Sync-warmup runner script is missing: {_RUNNER_SCRIPT}")

    dataset_root = request.dataset_root
    if dataset_root.exists() and not request.resume:
        raise RecordingError(
            f"Dataset already exists: {dataset_root}\n"
            "Use a new dataset name, or pass --resume only when intentionally appending."
        )
    if request.resume and not dataset_root.exists():
        raise RecordingError(f"Cannot resume because the dataset does not exist: {dataset_root}")

    request.data_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    lerobot_record_path = shutil.which("lerobot-record")
    assert lerobot_record_path is not None  # guaranteed by _check_command above
    python_executable = _resolve_lerobot_python(lerobot_record_path)
    command = build_record_command(hardware, request, python_executable=python_executable)

    _print_korean_guide(request, hardware)
    print("[실행 명령]")
    print(shlex.join(command))
    print()

    if dry_run:
        print("[DRY-RUN] 장치 검사와 명령 생성만 완료했습니다.")
        return dataset_root

    if not assume_yes:
        answer = input("장비와 작업 공간이 안전하면 Enter를 누르세요. 취소는 q: ").strip().lower()
        if answer in {"q", "quit", "n", "no"}:
            raise RecordingError("Recording was cancelled by the user.")

    env = os.environ.copy()
    try:
        completed = subprocess.run(command, env=env, check=False)
    except KeyboardInterrupt as exc:
        raise RecordingError("Recording was interrupted by the user.") from exc

    if completed.returncode != 0:
        raise RecordingError(
            f"lerobot-record exited with code {completed.returncode}. "
            "Check the terminal output above."
        )

    if not dataset_root.exists():
        raise RecordingError(
            "lerobot-record exited successfully, but the dataset folder was not created: "
            f"{dataset_root}"
        )

    print(f"\n[완료] 데이터셋 저장 완료: {dataset_root}")
    return dataset_root
