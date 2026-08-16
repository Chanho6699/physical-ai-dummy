"""Latest-frame-only capture for the real workspace and wrist cameras.

OpenCV/V4L2 may buffer frames internally. Reading only when a ~3 Hz inference
starts can therefore consume an old frame from a 30 FPS queue. Each camera here
is continuously drained by one background thread. Only one immutable RGB frame
per camera is retained; there is no application frame queue.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from data_collection.config import CameraConfig, HardwareConfig, load_hardware_config
from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY

_CAMERA_KEY_MAP: dict[str, str] = {
    "workspace": CAMERA_WORKSPACE_KEY,
    "wrist": CAMERA_WRIST_KEY,
}


class CameraSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraFrame:
    image_rgb: "object"
    captured_at_wall: float
    width: int
    height: int
    captured_at_monotonic: float | None = None


class RealCameraObservationSource:
    """Continuously drain both cameras and expose only their latest frames."""

    def __init__(
        self,
        cameras: dict[str, CameraConfig],
        *,
        monotonic_fn=time.monotonic,
        wall_fn=time.time,
        startup_timeout_s: float = 3.0,
    ) -> None:
        missing = [key for key in _CAMERA_KEY_MAP if key not in cameras]
        if missing:
            raise CameraSourceError(
                f"hardware config is missing cameras: {missing} (required={list(_CAMERA_KEY_MAP)})"
            )
        self._configs = {wire: cameras[local] for local, wire in _CAMERA_KEY_MAP.items()}
        self._monotonic = monotonic_fn
        self._wall = wall_fn
        self._startup_timeout_s = startup_timeout_s
        self._captures: dict[str, object] = {}
        self._latest: dict[str, CameraFrame] = {}
        self._errors: dict[str, str] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._condition = threading.Condition()

    @classmethod
    def from_hardware_config_path(cls, path: str) -> "RealCameraObservationSource":
        config: HardwareConfig = load_hardware_config(path)
        return cls(config.cameras)

    def open(self) -> None:
        import cv2

        if self._captures:
            raise CameraSourceError("camera source is already open")
        self._stop_event.clear()
        with self._condition:
            self._latest.clear()
            self._errors.clear()

        try:
            for wire_key, cam in self._configs.items():
                cap = cv2.VideoCapture(cam.index_or_path)
                if cam.fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
                cap.set(cv2.CAP_PROP_FPS, cam.fps)
                # Best-effort backend hint. Some V4L2/OpenCV combinations ignore
                # this property, so correctness comes from continuous draining.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    cap.release()
                    raise CameraSourceError(f"cannot open camera: {wire_key} ({cam.index_or_path})")
                self._captures[wire_key] = cap

            for wire_key, cap in self._captures.items():
                thread = threading.Thread(
                    target=self._capture_loop, args=(wire_key, cap),
                    name=f"LatestFrameCapture[{wire_key}]", daemon=True,
                )
                self._threads[wire_key] = thread
                thread.start()

            deadline = self._monotonic() + self._startup_timeout_s
            with self._condition:
                while len(self._latest) < len(self._captures) and not self._errors:
                    remaining = deadline - self._monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=min(remaining, 0.1))
                if self._errors:
                    raise CameraSourceError(f"camera capture startup failed: {self._errors}")
                missing_latest = [key for key in self._captures if key not in self._latest]
                if missing_latest:
                    raise CameraSourceError(
                        f"no camera frame within {self._startup_timeout_s:.1f}s: {missing_latest}"
                    )
        except Exception:
            self.close()
            raise

    def _capture_loop(self, wire_key: str, cap) -> None:
        import cv2

        while not self._stop_event.is_set():
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                if self._stop_event.is_set():
                    return
                with self._condition:
                    self._errors[wire_key] = "cap.read() failed"
                    self._condition.notify_all()
                return
            captured_monotonic = self._monotonic()
            captured_wall = self._wall()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            frame = CameraFrame(
                image_rgb=frame_rgb,
                captured_at_wall=captured_wall,
                captured_at_monotonic=captured_monotonic,
                width=width,
                height=height,
            )
            with self._condition:
                self._latest[wire_key] = frame
                self._condition.notify_all()

    def capture_all(self) -> dict[str, CameraFrame]:
        if not self._captures:
            raise CameraSourceError("open() must be called before capture_all()")
        with self._condition:
            if self._errors:
                raise CameraSourceError(f"camera capture failed: {self._errors}")
            missing = [key for key in self._captures if key not in self._latest]
            if missing:
                raise CameraSourceError(f"latest camera frame is unavailable: {missing}")
            # Snapshot two latest slots atomically. No read(), dequeue, or queue
            # traversal occurs on the inference thread.
            return {key: self._latest[key] for key in self._captures}

    def close(self) -> None:
        self._stop_event.set()
        captures = list(self._captures.values())
        # release() also unblocks a backend read that did not return promptly.
        for cap in captures:
            try:
                cap.release()
            except Exception:
                pass
        for thread in self._threads.values():
            thread.join(timeout=1.0)
        self._threads.clear()
        self._captures.clear()
        with self._condition:
            self._latest.clear()
            self._errors.clear()

    def __enter__(self) -> "RealCameraObservationSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
