from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


Level = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class CheckResult:
    level: Level
    name: str
    message: str


@dataclass
class ValidationReport:
    dataset_root: str
    checks: list[CheckResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.level == "FAIL"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [check for check in self.checks if check.level == "WARN"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def add(self, level: Level, name: str, message: str) -> None:
        self.checks.append(CheckResult(level=level, name=name, message=message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "passed": self.passed,
            "failure_count": len(self.failures),
            "warning_count": len(self.warnings),
            "summary": self.summary,
            "checks": [asdict(check) for check in self.checks],
        }


class DatasetValidationError(RuntimeError):
    """Raised when the validator cannot inspect the dataset at all."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Invalid JSON file: {path}: {exc}") from exc


def _read_parquet_files(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise DatasetValidationError("No parquet files were provided.")
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _safe_vector_length(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(np.asarray(value).size)
    except Exception:
        return None


def _series_has_non_finite_vectors(series: pd.Series) -> bool:
    for value in series:
        if value is None:
            return True
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return True
        if array.size == 0 or not np.isfinite(array).all():
            return True
    return False


def _ffprobe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise DatasetValidationError(
            f"ffprobe failed for {path}: {completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise DatasetValidationError(f"No video stream found in {path}")
    stream = streams[0]
    fmt = payload.get("format") or {}

    rate_text = stream.get("avg_frame_rate", "0/1")
    numerator, denominator = rate_text.split("/", maxsplit=1)
    fps = float(numerator) / float(denominator) if float(denominator) else 0.0

    frame_text = stream.get("nb_frames")
    frames = int(frame_text) if frame_text not in {None, "N/A"} else None
    duration_text = fmt.get("duration")
    duration = float(duration_text) if duration_text is not None else None

    return {
        "path": str(path),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": frames,
        "duration": duration,
        "codec": stream.get("codec_name"),
    }


def _feature_shape(info: dict[str, Any], feature_name: str) -> list[int] | None:
    feature = (info.get("features") or {}).get(feature_name) or {}
    shape = feature.get("shape")
    return shape if isinstance(shape, list) else None


def validate_dataset(
    dataset_root: str | Path,
    *,
    expected_episodes: int | None = None,
    expected_cameras: tuple[str, ...] = ("workspace", "wrist"),
    frame_tolerance: int = 2,
) -> ValidationReport:
    root = Path(dataset_root).expanduser().resolve()
    report = ValidationReport(dataset_root=str(root))

    if not root.is_dir():
        raise DatasetValidationError(f"Dataset directory does not exist: {root}")

    info_path = root / "meta/info.json"
    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    episode_files = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    task_path = root / "meta/tasks.parquet"

    required = {
        "meta/info.json": info_path,
        "data parquet": data_files[0] if data_files else None,
        "episode parquet": episode_files[0] if episode_files else None,
        "meta/tasks.parquet": task_path,
    }
    missing = [name for name, path in required.items() if path is None or not Path(path).exists()]
    if missing:
        for name in missing:
            report.add("FAIL", "required_files", f"Missing: {name}")
        return report
    report.add("PASS", "required_files", "Required metadata and parquet files exist.")

    info = _load_json(info_path)
    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    fps = int(info.get("fps", -1))
    report.summary.update(
        total_episodes=total_episodes,
        total_frames=total_frames,
        fps=fps,
    )

    if total_episodes <= 0:
        report.add("FAIL", "total_episodes", f"Invalid total_episodes: {total_episodes}")
    elif expected_episodes is not None and total_episodes != expected_episodes:
        report.add(
            "FAIL",
            "total_episodes",
            f"Expected {expected_episodes}, found {total_episodes}.",
        )
    else:
        report.add("PASS", "total_episodes", f"Episodes: {total_episodes}")

    if total_frames <= 0:
        report.add("FAIL", "total_frames", f"Invalid total_frames: {total_frames}")
    else:
        report.add("PASS", "total_frames", f"Frames: {total_frames}")

    if fps <= 0:
        report.add("FAIL", "fps", f"Invalid FPS: {fps}")
    else:
        report.add("PASS", "fps", f"FPS: {fps}")

    try:
        data = _read_parquet_files(data_files)
        episodes = _read_parquet_files(episode_files)
    except Exception as exc:
        report.add("FAIL", "parquet_read", str(exc))
        return report

    if len(data) != total_frames:
        report.add(
            "FAIL",
            "row_count",
            f"Main parquet rows={len(data)}, info.total_frames={total_frames}.",
        )
    else:
        report.add("PASS", "row_count", f"Main parquet rows match: {len(data)}")

    if len(episodes) != total_episodes:
        report.add(
            "FAIL",
            "episode_metadata_count",
            f"Episode metadata rows={len(episodes)}, expected={total_episodes}.",
        )
    else:
        report.add("PASS", "episode_metadata_count", f"Episode metadata rows: {len(episodes)}")

    required_columns = {
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        report.add("FAIL", "columns", f"Missing columns: {missing_columns}")
    else:
        report.add("PASS", "columns", "All required state/action/index columns exist.")

    for column in ("action", "observation.state"):
        if column not in data.columns:
            continue
        declared_shape = _feature_shape(info, column)
        expected_dim = declared_shape[0] if declared_shape else None
        observed_dims = sorted(
            {
                dim
                for dim in (_safe_vector_length(value) for value in data[column])
                if dim is not None
            }
        )
        if expected_dim is None:
            report.add("WARN", f"{column}_shape", "Feature shape is missing from info.json.")
        elif observed_dims != [expected_dim]:
            report.add(
                "FAIL",
                f"{column}_shape",
                f"Declared dimension={expected_dim}, observed dimensions={observed_dims}.",
            )
        else:
            report.add("PASS", f"{column}_shape", f"Dimension: {expected_dim}")

        if _series_has_non_finite_vectors(data[column]):
            report.add("FAIL", f"{column}_finite", "Null, empty, or non-finite vectors found.")
        else:
            report.add("PASS", f"{column}_finite", "All vectors are finite and non-empty.")

    scalar_columns = [
        column
        for column in ("timestamp", "frame_index", "episode_index", "index", "task_index")
        if column in data.columns
    ]
    null_counts = {column: int(data[column].isna().sum()) for column in scalar_columns}
    if any(null_counts.values()):
        report.add("FAIL", "scalar_nulls", f"Null counts: {null_counts}")
    else:
        report.add("PASS", "scalar_nulls", "No nulls in timestamp/index columns.")

    if "index" in data.columns:
        actual = data["index"].astype(int).to_numpy()
        expected = np.arange(len(data), dtype=np.int64)
        if np.array_equal(actual, expected):
            report.add("PASS", "global_index", "Global index is continuous.")
        else:
            report.add("FAIL", "global_index", "Global index is not continuous from 0.")

    episode_lengths: dict[int, int] = {}
    if "episode_index" in data.columns:
        for episode_index, group in data.groupby("episode_index", sort=True):
            episode_number = int(episode_index)
            episode_lengths[episode_number] = len(group)

            if "frame_index" in group.columns:
                frame_values = group["frame_index"].astype(int).to_numpy()
                expected_values = np.arange(len(group), dtype=np.int64)
                if not np.array_equal(frame_values, expected_values):
                    report.add(
                        "FAIL",
                        "frame_index",
                        f"Episode {episode_number}: frame_index is not continuous from 0.",
                    )

            if "timestamp" in group.columns:
                timestamps = group["timestamp"].astype(float).to_numpy()
                if len(timestamps) > 1 and np.any(np.diff(timestamps) < -1e-6):
                    report.add(
                        "FAIL",
                        "timestamp_monotonic",
                        f"Episode {episode_number}: timestamp decreases.",
                    )

        if not any(check.name == "frame_index" and check.level == "FAIL" for check in report.checks):
            report.add("PASS", "frame_index", "frame_index is continuous in every episode.")
        if not any(
            check.name == "timestamp_monotonic" and check.level == "FAIL"
            for check in report.checks
        ):
            report.add("PASS", "timestamp_monotonic", "Timestamps are monotonic in every episode.")

    if {"episode_index", "length"}.issubset(episodes.columns):
        mismatches: list[str] = []
        for row in episodes[["episode_index", "length"]].itertuples(index=False):
            actual_length = episode_lengths.get(int(row.episode_index))
            if actual_length != int(row.length):
                mismatches.append(
                    f"episode {int(row.episode_index)} metadata={int(row.length)} data={actual_length}"
                )
        if mismatches:
            report.add("FAIL", "episode_lengths", "; ".join(mismatches))
        else:
            lengths = [int(value) for value in episodes["length"].tolist()]
            report.summary["episode_lengths"] = lengths
            report.add("PASS", "episode_lengths", f"Episode lengths match metadata: {lengths}")

    features = info.get("features") or {}
    camera_features = sorted(
        name for name in features if name.startswith("observation.images.")
    )
    camera_names = [name.removeprefix("observation.images.") for name in camera_features]
    report.summary["cameras"] = camera_names

    for expected_camera in expected_cameras:
        feature_name = f"observation.images.{expected_camera}"
        if feature_name not in camera_features:
            report.add("FAIL", "camera_feature", f"Missing camera feature: {feature_name}")

    if camera_features:
        report.add("PASS", "camera_features", f"Camera features: {camera_names}")
    else:
        report.add("FAIL", "camera_features", "No camera features found in info.json.")

    if shutil.which("ffprobe") is None:
        report.add("WARN", "ffprobe", "ffprobe is unavailable; video metadata was not checked.")
        return report

    video_summary: dict[str, Any] = {}
    for feature_name in camera_features:
        videos = sorted((root / "videos" / feature_name).glob("chunk-*/file-*.mp4"))
        if not videos:
            report.add("FAIL", "video_files", f"No MP4 files for {feature_name}.")
            continue

        try:
            probes = [_ffprobe_video(path) for path in videos]
        except Exception as exc:
            report.add("FAIL", "video_probe", f"{feature_name}: {exc}")
            continue

        total_video_frames = sum(
            int(probe["frames"]) for probe in probes if probe["frames"] is not None
        )
        known_frame_counts = all(probe["frames"] is not None for probe in probes)
        total_duration = sum(
            float(probe["duration"]) for probe in probes if probe["duration"] is not None
        )
        widths = sorted({probe["width"] for probe in probes})
        heights = sorted({probe["height"] for probe in probes})
        fps_values = sorted({round(float(probe["fps"]), 6) for probe in probes})

        video_summary[feature_name] = {
            "files": probes,
            "total_frames": total_video_frames if known_frame_counts else None,
            "total_duration": total_duration,
        }

        declared = features.get(feature_name) or {}
        declared_info = declared.get("info") or {}
        declared_width = declared_info.get("video.width")
        declared_height = declared_info.get("video.height")
        declared_fps = declared_info.get("video.fps")

        if declared_width is not None and widths != [int(declared_width)]:
            report.add(
                "FAIL",
                "video_resolution",
                f"{feature_name}: declared width={declared_width}, observed={widths}.",
            )
        elif declared_height is not None and heights != [int(declared_height)]:
            report.add(
                "FAIL",
                "video_resolution",
                f"{feature_name}: declared height={declared_height}, observed={heights}.",
            )
        else:
            report.add(
                "PASS",
                "video_resolution",
                f"{feature_name}: {widths[0]}x{heights[0]}",
            )

        if declared_fps is not None and any(
            not math.isclose(value, float(declared_fps), rel_tol=0.0, abs_tol=0.01)
            for value in fps_values
        ):
            report.add(
                "FAIL",
                "video_fps",
                f"{feature_name}: declared FPS={declared_fps}, observed={fps_values}.",
            )
        else:
            report.add("PASS", "video_fps", f"{feature_name}: FPS={fps_values}")

        if known_frame_counts:
            delta = abs(total_video_frames - total_frames)
            if delta > frame_tolerance:
                report.add(
                    "FAIL",
                    "video_frame_count",
                    f"{feature_name}: video frames={total_video_frames}, "
                    f"dataset frames={total_frames}, delta={delta}.",
                )
            else:
                report.add(
                    "PASS",
                    "video_frame_count",
                    f"{feature_name}: video frames={total_video_frames}, delta={delta}.",
                )
        else:
            report.add(
                "WARN",
                "video_frame_count",
                f"{feature_name}: ffprobe did not expose nb_frames for every file.",
            )

    report.summary["videos"] = video_summary
    return report


def print_validation_report(report: ValidationReport) -> None:
    icons = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
    print("\n===== LeRobot 데이터셋 자동 검증 =====")
    print(f"경로: {report.dataset_root}\n")
    for check in report.checks:
        print(f"{icons[check.level]} {check.name}: {check.message}")

    print("\n===== 결과 =====")
    if report.failures:
        print(f"[FAIL] 실패 {len(report.failures)}개, 경고 {len(report.warnings)}개")
    else:
        print(f"[PASS] 구조 검증 통과, 경고 {len(report.warnings)}개")
    print("※ 자동 검증은 파일·동기화·형식 검사입니다. 실제 grasp 성공은 영상으로 별도 검수하세요.")
