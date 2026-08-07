"""실물 카메라(fixed workspace / wrist) 캡처.

카메라 device path/해상도는 여기서 새로 하드코딩하지 않고 기존
``data_collection/config.py``의 ``HardwareConfig``/``CameraConfig``(``configs/hardware.local.json``)를
그대로 재사용한다 (섹션 5 요구사항). ``configs/hardware.local.json``의 카메라 key
(``workspace``/``wrist``)는 데이터셋 feature 이름(``observation.images.workspace``/
``observation.images.wrist``)과 접두사만 다르므로, ``runtime.common.vla_contract``의
``CAMERA_WORKSPACE_KEY``/``CAMERA_WRIST_KEY``로 그대로 매핑한다.

MuJoCo virtual camera는 이 모듈에서 전혀 다루지 않는다 - Shadow Mode의 VLA 입력은 항상
실제 카메라 프레임이다 (섹션 5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from data_collection.config import CameraConfig, HardwareConfig, load_hardware_config
from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY, CAMERA_WRIST_KEY

# hardware.local.json의 카메라 key -> 데이터셋 feature(wire) key.
_CAMERA_KEY_MAP: dict[str, str] = {
    "workspace": CAMERA_WORKSPACE_KEY,
    "wrist": CAMERA_WRIST_KEY,
}


class CameraSourceError(RuntimeError):
    """카메라 오픈/캡처 실패."""


@dataclass(frozen=True)
class CameraFrame:
    image_rgb: "object"  # HWC uint8 RGB numpy array
    captured_at_wall: float
    width: int
    height: int


class RealCameraObservationSource:
    """``configs/hardware.local.json``의 ``cameras`` 설정으로 OpenCV 캡처를 연다.

    이 클래스는 프레임을 읽기만 한다 - 실물 로봇/서보에 대한 참조가 전혀 없다.
    """

    def __init__(self, cameras: dict[str, CameraConfig]) -> None:
        missing = [k for k in _CAMERA_KEY_MAP if k not in cameras]
        if missing:
            raise CameraSourceError(
                f"hardware config에 다음 카메라가 없습니다: {missing} (필요: {list(_CAMERA_KEY_MAP)})"
            )
        self._configs = {wire_key: cameras[local_key] for local_key, wire_key in _CAMERA_KEY_MAP.items()}
        self._captures: dict[str, object] = {}

    @classmethod
    def from_hardware_config_path(cls, path: str) -> "RealCameraObservationSource":
        config: HardwareConfig = load_hardware_config(path)
        return cls(config.cameras)

    def open(self) -> None:
        import cv2

        for wire_key, cam in self._configs.items():
            cap = cv2.VideoCapture(cam.index_or_path)
            if cam.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
            cap.set(cv2.CAP_PROP_FPS, cam.fps)
            if not cap.isOpened():
                self.close()
                raise CameraSourceError(f"카메라를 열 수 없습니다: {wire_key} ({cam.index_or_path})")
            self._captures[wire_key] = cap

    def capture_all(self) -> dict[str, CameraFrame]:
        import cv2

        if not self._captures:
            raise CameraSourceError("open()을 먼저 호출해야 합니다.")
        frames: dict[str, CameraFrame] = {}
        for wire_key, cap in self._captures.items():
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise CameraSourceError(f"카메라 프레임을 읽을 수 없습니다: {wire_key}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            frames[wire_key] = CameraFrame(image_rgb=frame_rgb, captured_at_wall=time.time(), width=w, height=h)
        return frames

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures = {}

    def __enter__(self) -> "RealCameraObservationSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
