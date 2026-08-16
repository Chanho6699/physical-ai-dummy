from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np

from data_collection.config import CameraConfig
from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY
from runtime.laptop.camera_source import RealCameraObservationSource


class FakeCapture:
    instances = []

    def __init__(self, device):
        self.device = device
        self.counter = 0
        self.released = False
        self.settings = {}
        self.__class__.instances.append(self)

    def set(self, prop, value):
        self.settings[prop] = value
        return True

    def isOpened(self):
        return not self.released

    def read(self):
        time.sleep(0.002)
        if self.released:
            return False, None
        self.counter += 1
        value = self.counter % 255
        return True, np.full((4, 6, 3), value, dtype=np.uint8)

    def release(self):
        self.released = True


def _fake_cv2():
    return SimpleNamespace(
        VideoCapture=FakeCapture,
        VideoWriter_fourcc=lambda *args: 1234,
        cvtColor=lambda image, code: image[..., ::-1].copy(),
        COLOR_BGR2RGB=10,
        CAP_PROP_FOURCC=1,
        CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3,
        CAP_PROP_FPS=4,
        CAP_PROP_BUFFERSIZE=5,
    )


def test_latest_frame_threads_drain_continuously_without_queue(monkeypatch) -> None:
    FakeCapture.instances.clear()
    fake_cv2 = _fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    config = {
        "workspace": CameraConfig("opencv", "/dev/video2", 6, 4, 30, "MJPG"),
        "wrist": CameraConfig("opencv", "/dev/video0", 6, 4, 30, "MJPG"),
    }
    camera = RealCameraObservationSource(config, startup_timeout_s=1.0)
    camera.open()
    try:
        first = camera.capture_all()
        first_values = {key: int(frame.image_rgb[0, 0, 0]) for key, frame in first.items()}
        first_times = {key: frame.captured_at_monotonic for key, frame in first.items()}
        # No inference-side capture call occurs here. Background readers must
        # continue draining and replace the one-slot latest frame.
        time.sleep(0.04)
        second = camera.capture_all()
        second_values = {key: int(frame.image_rgb[0, 0, 0]) for key, frame in second.items()}
        for key in (CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY):
            assert second_values[key] > first_values[key]
            assert second[key].captured_at_monotonic > first_times[key]
        assert len(camera._latest) == 2
        assert all(cap.settings[fake_cv2.CAP_PROP_BUFFERSIZE] == 1 for cap in FakeCapture.instances)
    finally:
        camera.close()

    assert camera._captures == {}
    assert camera._threads == {}
