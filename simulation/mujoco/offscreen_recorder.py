"""GUI 창 없이 MuJoCo 시뮬레이션 프레임을 PNG/MP4로 저장하는 오프스크린 렌더러.

WSLg GUI 경로(mujoco.viewer)가 실패하거나 확인이 안 되는 상황에서도, 물리
시뮬레이션·scene 자체가 정상인지 분리해서 검증할 수 있게 하는 대체 경로다
(`remote_diagnostic.py`의 `mode="offscreen"`에서 사용). GUI(GLFW) 창과 달리
백그라운드 렌더 스레드를 만들지 않는 단일 스레드 경로이며, 네트워크/리더암/
팔로워암 제어와는 무관하다 - 여기서 만드는 것은 순수하게 이미 계산된
``mujoco.MjData``의 스냅샷을 이미지로 바꾸는 것뿐이다.

실패는 항상 :class:`OffscreenRecorderError`로 감싸서 올린다 - 호출자가 한글
오류 메시지로 사용자에게 명확히 알릴 수 있게 하기 위함이다 (조용히 삼키지 않는다).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


class OffscreenRecorderError(RuntimeError):
    """오프스크린 렌더러 생성, 프레임 렌더링, PNG/MP4 저장 중 발생한 오류."""


@dataclass(frozen=True)
class OffscreenRecorderSummary:
    frame_count: int
    save_frames_dir: str | None
    manifest_path: str | None
    video_path: str | None


class OffscreenRecorder:
    """mujoco.Renderer를 감싸서 프레임을 PNG/MP4로 순차 저장한다.

    ``save_frames_dir``/``video_path`` 중 최소 하나는 지정해야 한다 - 둘 다
    없으면 아무 것도 저장하지 않는 호출이므로 생성 시점에 바로 오류를 낸다.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        width: int = 640,
        height: int = 480,
        save_frames_dir: Path | None = None,
        video_path: Path | None = None,
        video_fps: float = 20.0,
    ) -> None:
        if save_frames_dir is None and video_path is None:
            raise OffscreenRecorderError(
                "PNG 저장 디렉터리(--save-frames)나 MP4 출력 경로(--video-output) 중 최소 하나는 지정해야 합니다."
            )
        if video_fps <= 0:
            raise OffscreenRecorderError(f"video_fps는 0보다 커야 합니다: {video_fps}")

        try:
            self._renderer = mujoco.Renderer(model, height=height, width=width)
        except Exception as exc:  # mujoco/OpenGL 쪽 예외 타입이 backend마다 달라 광범위하게 잡는다
            raise OffscreenRecorderError(
                f"오프스크린 렌더러(mujoco.Renderer) 생성에 실패했습니다 (width={width}, height={height}): {exc}"
            ) from exc

        self._width = width
        self._height = height
        self._save_frames_dir = save_frames_dir
        if self._save_frames_dir is not None:
            self._save_frames_dir.mkdir(parents=True, exist_ok=True)
        self._video_path = video_path
        self._video_fps = video_fps
        self._video_writer = None
        self._cv2 = None

        self._manifest: list[dict] = []
        self._frame_index = 0
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_index

    def capture(
        self,
        data: mujoco.MjData,
        *,
        remote_sequence: int | None = None,
        remote_timestamp: float | None = None,
    ) -> np.ndarray:
        """현재 ``data`` 상태를 한 프레임 렌더링하고 설정된 대상(PNG/MP4)에 저장한다."""
        if self._closed:
            raise OffscreenRecorderError("이미 close()된 OffscreenRecorder에는 프레임을 추가할 수 없습니다.")

        try:
            self._renderer.update_scene(data)
            frame = self._renderer.render()
        except Exception as exc:
            raise OffscreenRecorderError(f"프레임 렌더링 실패 (frame_index={self._frame_index}): {exc}") from exc

        if self._save_frames_dir is not None:
            self._save_png(frame)
        if self._video_path is not None:
            self._write_video_frame(frame)

        self._manifest.append(
            {
                "frame_index": self._frame_index,
                "local_timestamp": time.time(),
                "remote_sequence": remote_sequence,
                "remote_timestamp": remote_timestamp,
            }
        )
        self._frame_index += 1
        return frame

    def _save_png(self, frame: np.ndarray) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise OffscreenRecorderError(
                "Pillow(PIL)가 설치되어 있지 않아 PNG로 저장할 수 없습니다. `pip install pillow`가 필요합니다."
            ) from exc
        assert self._save_frames_dir is not None
        path = self._save_frames_dir / f"frame_{self._frame_index:06d}.png"
        try:
            Image.fromarray(frame).save(path)
        except Exception as exc:
            raise OffscreenRecorderError(f"PNG 저장 실패 ({path}): {exc}") from exc

    def _write_video_frame(self, frame: np.ndarray) -> None:
        if self._video_writer is None:
            try:
                import cv2
            except ImportError as exc:
                raise OffscreenRecorderError(
                    "opencv-python(cv2)이 설치되어 있지 않아 MP4로 저장할 수 없습니다. `pip install opencv-python`이 필요합니다."
                ) from exc
            assert self._video_path is not None
            self._video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self._video_path), fourcc, self._video_fps, (self._width, self._height))
            if not writer.isOpened():
                raise OffscreenRecorderError(
                    f"MP4 writer를 열 수 없습니다: {self._video_path} (codec/ffmpeg 지원 여부를 확인하세요)"
                )
            self._cv2 = cv2
            self._video_writer = writer

        assert self._cv2 is not None
        bgr = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        self._video_writer.write(bgr)

    def close(self) -> OffscreenRecorderSummary:
        """렌더러/video writer를 정리하고 frame manifest(JSON)를 저장한다. 여러 번 호출해도 안전하다."""
        if self._closed:
            return OffscreenRecorderSummary(
                frame_count=self._frame_index,
                save_frames_dir=str(self._save_frames_dir) if self._save_frames_dir else None,
                manifest_path=None,
                video_path=str(self._video_path) if self._video_path else None,
            )
        self._closed = True

        if self._video_writer is not None:
            self._video_writer.release()
        self._renderer.close()

        manifest_path: str | None = None
        if self._save_frames_dir is not None:
            manifest_file = self._save_frames_dir / "frames_manifest.json"
            manifest_file.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path = str(manifest_file)

        return OffscreenRecorderSummary(
            frame_count=self._frame_index,
            save_frames_dir=str(self._save_frames_dir) if self._save_frames_dir else None,
            manifest_path=manifest_path,
            video_path=str(self._video_path) if self._video_path else None,
        )

    def __enter__(self) -> "OffscreenRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
