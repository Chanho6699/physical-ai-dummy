from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a hardware configuration is invalid."""


@dataclass(frozen=True)
class DeviceConfig:
    type: str
    port: str
    id: str


@dataclass(frozen=True)
class CameraConfig:
    type: str
    index_or_path: str | int
    width: int
    height: int
    fps: int
    fourcc: str | None = None

    def to_lerobot_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": self.type,
            "index_or_path": self.index_or_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
        if self.fourcc:
            value["fourcc"] = self.fourcc
        return value


@dataclass(frozen=True)
class HardwareConfig:
    robot: DeviceConfig
    teleop: DeviceConfig
    cameras: dict[str, CameraConfig]


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a JSON object.")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty string.")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"'{name}' must be a positive integer.")
    return value


def _load_device(raw: Any, name: str) -> DeviceConfig:
    obj = _require_mapping(raw, name)
    return DeviceConfig(
        type=_require_string(obj.get("type"), f"{name}.type"),
        port=_require_string(obj.get("port"), f"{name}.port"),
        id=_require_string(obj.get("id"), f"{name}.id"),
    )


def _load_camera(raw: Any, name: str) -> CameraConfig:
    obj = _require_mapping(raw, f"cameras.{name}")
    index_or_path = obj.get("index_or_path")
    if not isinstance(index_or_path, (str, int)) or isinstance(index_or_path, bool):
        raise ConfigError(
            f"'cameras.{name}.index_or_path' must be a string path or integer index."
        )

    fourcc = obj.get("fourcc")
    if fourcc is not None:
        fourcc = _require_string(fourcc, f"cameras.{name}.fourcc")
        if len(fourcc) != 4:
            raise ConfigError(f"'cameras.{name}.fourcc' must contain exactly 4 characters.")

    return CameraConfig(
        type=_require_string(obj.get("type"), f"cameras.{name}.type"),
        index_or_path=index_or_path,
        width=_require_positive_int(obj.get("width"), f"cameras.{name}.width"),
        height=_require_positive_int(obj.get("height"), f"cameras.{name}.height"),
        fps=_require_positive_int(obj.get("fps"), f"cameras.{name}.fps"),
        fourcc=fourcc,
    )


def load_hardware_config(path: str | Path) -> HardwareConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Hardware config does not exist: {config_path}")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc

    root = _require_mapping(raw, "root")
    cameras_raw = _require_mapping(root.get("cameras"), "cameras")
    if not cameras_raw:
        raise ConfigError("At least one camera must be configured.")

    cameras = {
        name: _load_camera(camera_raw, name)
        for name, camera_raw in cameras_raw.items()
    }

    return HardwareConfig(
        robot=_load_device(root.get("robot"), "robot"),
        teleop=_load_device(root.get("teleop"), "teleop"),
        cameras=cameras,
    )
