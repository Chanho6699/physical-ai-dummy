"""simulation/mujoco/offscreen_recorder.py 단위 테스트.

GUI(GLFW) 창을 전혀 만들지 않는 단일 스레드 오프스크린 경로만 검증한다 - CI/헤드리스
환경에서도 항상 돌아가야 한다 (mujoco.Renderer는 DISPLAY 없이도 EGL/OSMesa/software GLX
중 하나로 동작한다).
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from simulation.mujoco.offscreen_recorder import OffscreenRecorder, OffscreenRecorderError
from simulation.mujoco.so101_model import DEFAULT_SCENE_PATH, load_model, make_data


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return load_model(DEFAULT_SCENE_PATH)


def test_requires_at_least_one_output_target(model):
    with pytest.raises(OffscreenRecorderError):
        OffscreenRecorder(model, save_frames_dir=None, video_path=None)


def test_capture_saves_png_and_records_manifest(model, tmp_path):
    data = make_data(model)
    mujoco.mj_forward(model, data)
    out_dir = tmp_path / "frames"

    with OffscreenRecorder(model, width=64, height=48, save_frames_dir=out_dir) as recorder:
        for i in range(3):
            frame = recorder.capture(data, remote_sequence=i, remote_timestamp=float(i))
            assert frame.shape == (48, 64, 3)
        summary = recorder.close()

    assert summary.frame_count == 3
    saved = sorted(out_dir.glob("frame_*.png"))
    assert len(saved) == 3

    manifest = json.loads((out_dir / "frames_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 3
    assert [entry["remote_sequence"] for entry in manifest] == [0, 1, 2]


def test_capture_produces_non_blank_frame(model):
    """물리 시뮬레이션/scene 자체가 정상이라면 렌더링된 프레임은 단색이 아니어야 한다."""
    data = make_data(model)
    mujoco.mj_forward(model, data)
    with OffscreenRecorder(model, width=64, height=48, save_frames_dir=None, video_path=_dummy_path()) as recorder:
        frame = recorder.capture(data)
    assert float(np.std(frame)) > 1.0  # 완전한 단색(블랭크)이 아님


def _dummy_path():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mktemp(suffix=".mp4"))


def test_capture_after_close_raises(model, tmp_path):
    data = make_data(model)
    mujoco.mj_forward(model, data)
    recorder = OffscreenRecorder(model, save_frames_dir=tmp_path / "frames")
    recorder.capture(data)
    recorder.close()
    with pytest.raises(OffscreenRecorderError):
        recorder.capture(data)


def test_double_close_is_safe(model, tmp_path):
    data = make_data(model)
    mujoco.mj_forward(model, data)
    recorder = OffscreenRecorder(model, save_frames_dir=tmp_path / "frames")
    recorder.capture(data)
    recorder.close()
    recorder.close()  # 두 번째 호출도 예외를 던지면 안 된다


def test_video_output_requires_cv2_or_reports_clear_error(model, tmp_path, monkeypatch):
    """cv2가 없는 환경을 흉내내 PNG 없이 MP4만 요청했을 때 한글 오류로 명확히 실패하는지 확인한다."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2 for test")
        return real_import(name, *args, **kwargs)

    data = make_data(model)
    mujoco.mj_forward(model, data)
    with OffscreenRecorder(model, video_path=tmp_path / "out.mp4") as recorder:
        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(OffscreenRecorderError, match="opencv-python"):
            recorder.capture(data)
