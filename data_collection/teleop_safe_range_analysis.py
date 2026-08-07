"""Read-only analysis of real SO-101 *follower* teleop joint position ranges.

This module never sends anything to a robot. It only reads existing LeRobot
v3 dataset files from disk (``meta/info.json``, ``meta/episodes/**``,
``data/**``) and computes summary statistics that later safety tooling can
use to build a "historical safe range" per joint. It never writes to the
source dataset files, never touches motor torque/goal position/calibration,
and never derives a follower range from leader-only data.

Why only ``observation.state`` and never ``action``
-----------------------------------------------------
Traced from ``~/lerobot/src/lerobot/scripts/lerobot_record.py`` (the
``record_loop`` used by ``lerobot-record``, which is what
``data_collection/recorder.py`` shells out to)::

    obs = robot.get_observation()                       # -> observation.state
    act = teleop.get_action()                            # leader arm state
    act_processed_teleop = teleop_action_processor((act, obs))
    action_values = act_processed_teleop                 # -> action

``action`` is built from the *leader* arm's commanded pose (``teleop.get_action()``),
not a follower position readback. Only ``observation.state`` comes from
``SOFollower.get_observation()`` -> ``self.bus.sync_read("Present_Position", ...)``
(``~/lerobot/src/lerobot/robots/so_follower/so_follower.py``), i.e. the true
follower encoder position at record time. This module therefore only ever
pools ``observation.state`` samples for range analysis, matching the task
requirement to never derive follower safety ranges from leader-only data.

Units
-----
``SOFollower.__init__`` (``so_follower.py``) builds the motor bus with::

    norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
    shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll -> norm_mode_body
    gripper -> always MotorNormMode.RANGE_0_100  (not configurable)

``SOFollowerConfig.use_degrees`` defaults to ``True``
(``~/lerobot/src/lerobot/robots/so_follower/config_so_follower.py``). This
repo's only recording path (``scripts/record_episodes.py`` ->
``data_collection/recorder.py`` -> ``lerobot-record``) never passes a
``--robot.use_degrees`` override, and ``configs/hardware.local.json`` does
not declare that key either, so the default applies: the five arm joints
are stored in **degrees**, and ``gripper`` is always stored as a **0-100
range** (unconditionally, regardless of ``use_degrees``). This is verified
from source code, not guessed.

If a future dataset's ``meta/info.json`` cannot be mapped to this known
unit scheme (unrecognized joint names, unsupported codebase version, or an
unrecognized ``robot_type``), that dataset is excluded from the analysis
rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"

# Joint names this module knows how to interpret, and their confirmed units.
# See module docstring "Units" for the code citations backing this table.
DEGREE_JOINTS: frozenset[str] = frozenset(
    {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}
)
PERCENT_JOINTS: frozenset[str] = frozenset({"gripper"})
KNOWN_JOINTS: frozenset[str] = DEGREE_JOINTS | PERCENT_JOINTS

# robot_type values (LeRobot's Robot.name) that this module trusts as "real
# SO-101/SO-100 follower hardware". Anything else (including simulated or
# unrecognized robots) is excluded rather than assumed to be real hardware.
TRUSTED_ROBOT_TYPES: frozenset[str] = frozenset({"so_follower", "so101_follower", "so100_follower"})

# Only LeRobot v3 datasets have been inspected for this module's assumptions
# (parquet layout, observation.state semantics). Anything else is rejected.
SUPPORTED_CODEBASE_PREFIX = "v3"

STATUS_TRUSTED = "TRUSTED"
STATUS_INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
STATUS_UNKNOWN_UNIT = "UNKNOWN_UNIT"
STATUS_UNTRUSTED_SOURCE = "UNTRUSTED_SOURCE"
STATUS_MARGIN_COLLAPSED = "MARGIN_COLLAPSED"


class TeleopAnalysisError(RuntimeError):
    """Raised when the requested input path(s) cannot be analyzed at all."""


@dataclass(frozen=True)
class AnalysisPolicy:
    """All knobs that influence the computed safety ranges.

    Every value here is echoed back into the output JSON's
    ``analysis_policy`` block so nothing is a hidden default.
    """

    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    margin_degree: float = 2.0
    margin_percent: float = 2.0
    minimum_samples: int = 200

    def margin_for(self, joint: str) -> float | None:
        if joint in DEGREE_JOINTS:
            return self.margin_degree
        if joint in PERCENT_JOINTS:
            return self.margin_percent
        return None


@dataclass
class DatasetTrust:
    """Result of inspecting a single candidate dataset directory."""

    root: Path
    relative_path: str
    trusted: bool
    reason: str
    joint_names: tuple[str, ...] | None = None
    codebase_version: str | None = None
    robot_type: str | None = None
    declared_episode_count: int | None = None
    declared_frame_count: int | None = None
    data_file_count: int = 0
    info_json_path: Path | None = None
    # sha256 over the *content* of every data/chunk-*/file-*.parquet file (name + bytes),
    # independent of where the directory lives on disk. Two dataset roots with the same
    # content_hash are the same recording (copy/mirror), never summed together.
    content_hash: str | None = None


@dataclass
class JointResult:
    joint: str
    status: str
    unit: str | None
    sample_count: int
    nan_count: int
    episode_count: int
    source_file_count: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    p01: float | None = None
    p05: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    policy_percentile_low: float | None = None
    policy_percentile_high: float | None = None
    margin_applied: float | None = None
    historical_safe_min: float | None = None
    historical_safe_max: float | None = None
    status_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_detail": self.status_detail,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "nan_or_invalid_count": self.nan_count,
            "episode_count": self.episode_count,
            "source_file_count": self.source_file_count,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "p01": self.p01,
            "p05": self.p05,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "policy_percentile_low": self.policy_percentile_low,
            "policy_percentile_high": self.policy_percentile_high,
            "margin_applied": self.margin_applied,
            "historical_safe_min": self.historical_safe_min,
            "historical_safe_max": self.historical_safe_max,
        }


@dataclass
class AnalysisResult:
    generated_at: str
    trusted: bool
    used_datasets: list[DatasetTrust]
    excluded_datasets: list[DatasetTrust]
    joints: dict[str, JointResult]
    policy: AnalysisPolicy
    total_episode_count: int
    total_sample_count: int
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": self.generated_at,
            "generator": {
                "script": "scripts/analyze_teleop_safe_ranges.py",
                "module": "data_collection.teleop_safe_range_analysis",
                "version": SCRIPT_VERSION,
            },
            "source_type": "real_follower_teleop",
            "trusted": self.trusted,
            "unit_by_joint_group": {
                "arm_joints": "degree",
                "gripper": "percent_0_100",
            },
            "unit": "mixed",
            "source_summary": {
                "dataset_paths": [d.relative_path for d in self.used_datasets],
                "dataset_count": len(self.used_datasets),
                "episode_count": self.total_episode_count,
                "sample_count": self.total_sample_count,
                "source_file_count": sum(d.data_file_count for d in self.used_datasets),
            },
            "excluded_sources": [
                {"path": d.relative_path, "reason": d.reason} for d in self.excluded_datasets
            ],
            "analysis_policy": {
                "lower_percentile": self.policy.lower_percentile,
                "upper_percentile": self.policy.upper_percentile,
                "margin_degree": self.policy.margin_degree,
                "margin_percent": self.policy.margin_percent,
                "minimum_samples": self.policy.minimum_samples,
            },
            "joints": {name: result.to_dict() for name, result in self.joints.items()},
            "provenance": self.provenance,
            "notes": [
                "action은 leader가 보낸 명령(teleop.get_action() 기반)이라 follower readback이 "
                "아니므로 이 분석에 전혀 사용하지 않았다 (observation.state만 사용).",
                "arm 5개 관절 단위(degree)는 SOFollowerConfig.use_degrees 기본값(True)과 이 저장소의 "
                "유일한 녹화 경로(record_episodes.py -> lerobot-record)가 그 값을 오버라이드하지 않는다는 "
                "코드 사실로 확정했다.",
                "gripper 단위(0-100)는 MotorNormMode.RANGE_0_100이 코드에서 무조건 적용되어 "
                "use_degrees 설정과 무관하게 항상 확정된다.",
                "data/ 와 configs/hardware.local.json은 .gitignore 대상이라 과거 녹화 시점의 "
                "하드웨어 설정 이력을 git으로 재검증할 수는 없다 - 현재 파일 내용과 코드 기본값에 "
                "근거한 결론이다.",
            ],
        }


def _relative_path(root: Path, project_root: Path) -> str:
    try:
        return str(root.resolve().relative_to(project_root.resolve()))
    except ValueError:
        # Outside the project root (e.g. an explicit --input elsewhere on disk).
        # Never leak the absolute path; fall back to just the directory name.
        return root.resolve().name


def discover_dataset_roots(search_root: Path | list[Path]) -> list[Path]:
    """Read-only discovery of LeRobot-shaped dataset directories.

    ``search_root`` can be a single directory or a list of directories (e.g.
    the repo's ``data/`` plus ``~/.cache/huggingface/lerobot`` to check for
    datasets recorded straight to LeRobot's default cache location). A
    directory one level below any search root is a *candidate* if it has
    ``meta/info.json`` directly under it (``<root>/<name>/meta/info.json``),
    matching how ``lerobot-record`` lays out ``--dataset.root``. Trust is
    decided later by :func:`evaluate_dataset_trust`; discovery alone does not
    imply anything is usable, and this never opens/writes anything besides
    the glob itself (a read-only filesystem walk).
    """
    search_roots = [search_root] if isinstance(search_root, Path) else list(search_root)
    found: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        found.update(info_path.parent.parent for info_path in root.glob("*/meta/info.json"))
        # Also match "search_root itself is one dataset root" (info.json directly under it).
        direct_info = root / "meta" / "info.json"
        if direct_info.is_file():
            found.add(root)
    return sorted(found)


_CALIBRATION_JOINT_KEYS = frozenset({"id", "drive_mode", "homing_offset", "range_min", "range_max"})


def find_calibration_files(search_roots: list[Path]) -> list[dict[str, Any]]:
    """Read-only inventory of SO-101/SO-100 calibration JSON files under ``search_roots``.

    A JSON file is classified as "calibration" if it parses to a dict whose
    values are themselves dicts containing calibration keys
    (``id``/``drive_mode``/``homing_offset``/``range_min``/``range_max``) -
    the exact shape LeRobot writes to
    ``~/.cache/huggingface/lerobot/calibration/{robots,teleoperators}/**/*.json``.
    These are reported separately and are **never** merged into episode/joint
    statistics (they describe raw-tick motor range-of-motion at calibration
    time, not how the joint was actually used during teleop).
    """
    results: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if not isinstance(payload, dict) or not payload:
                continue
            joint_entries = [v for v in payload.values() if isinstance(v, dict)]
            if len(joint_entries) != len(payload):
                continue
            if not all(_CALIBRATION_JOINT_KEYS.issubset(entry.keys()) for entry in joint_entries):
                continue
            seen_paths.add(resolved)
            results.append(
                {
                    "path": str(path),
                    "joints": sorted(payload.keys()),
                    "raw_ranges": {
                        joint: {"range_min": entry["range_min"], "range_max": entry["range_max"]}
                        for joint, entry in payload.items()
                    },
                }
            )
    return results


def evaluate_dataset_trust(root: Path, project_root: Path) -> DatasetTrust:
    """Read-only inspection of a single dataset directory.

    Never raises for a "bad" dataset - it always returns a :class:`DatasetTrust`
    with ``trusted=False`` and a human-readable ``reason`` instead, so callers
    can report every exclusion explicitly.
    """
    relative_path = _relative_path(root, project_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return DatasetTrust(root, relative_path, False, "meta/info.json이 없습니다.")

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return DatasetTrust(root, relative_path, False, f"meta/info.json 파싱 실패: {exc}")

    codebase_version = str(info.get("codebase_version", ""))
    if not codebase_version.startswith(SUPPORTED_CODEBASE_PREFIX):
        return DatasetTrust(
            root,
            relative_path,
            False,
            f"지원하지 않는 codebase_version: {codebase_version!r} (v3.x만 검증됨)",
            codebase_version=codebase_version or None,
            info_json_path=info_path,
        )

    robot_type = info.get("robot_type")
    if robot_type not in TRUSTED_ROBOT_TYPES:
        return DatasetTrust(
            root,
            relative_path,
            False,
            f"알 수 없거나 신뢰할 수 없는 robot_type: {robot_type!r} "
            f"(허용: {sorted(TRUSTED_ROBOT_TYPES)}) - 시뮬레이션이거나 미확인 출처일 수 있습니다.",
            codebase_version=codebase_version,
            robot_type=str(robot_type) if robot_type is not None else None,
            info_json_path=info_path,
        )

    features = info.get("features") or {}
    state_feature = features.get("observation.state")
    if not state_feature:
        return DatasetTrust(
            root,
            relative_path,
            False,
            "meta/info.json에 observation.state feature가 없습니다 (follower readback 확인 불가).",
            codebase_version=codebase_version,
            robot_type=robot_type,
            info_json_path=info_path,
        )

    raw_names = state_feature.get("names") or []
    shape = state_feature.get("shape") or []
    declared_dim = int(shape[0]) if shape else len(raw_names)
    if not raw_names or declared_dim <= 0 or len(raw_names) != declared_dim:
        return DatasetTrust(
            root,
            relative_path,
            False,
            f"observation.state 이름/차원이 일관되지 않습니다 (names={raw_names}, shape={shape}).",
            codebase_version=codebase_version,
            robot_type=robot_type,
            info_json_path=info_path,
        )

    joint_names: list[str] = []
    for raw_name in raw_names:
        canonical = raw_name.removesuffix(".pos") if raw_name.endswith(".pos") else raw_name
        if canonical not in KNOWN_JOINTS:
            return DatasetTrust(
                root,
                relative_path,
                False,
                f"관절 순서/이름을 확정할 수 없습니다: {raw_name!r} (알려진 관절: {sorted(KNOWN_JOINTS)}).",
                codebase_version=codebase_version,
                robot_type=robot_type,
                info_json_path=info_path,
            )
        joint_names.append(canonical)

    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        return DatasetTrust(
            root,
            relative_path,
            False,
            "data/chunk-*/file-*.parquet 파일이 없습니다.",
            codebase_version=codebase_version,
            robot_type=robot_type,
            joint_names=tuple(joint_names),
            info_json_path=info_path,
        )

    declared_episodes = info.get("total_episodes")
    declared_frames = info.get("total_frames")

    return DatasetTrust(
        root=root,
        relative_path=relative_path,
        trusted=True,
        reason="observation.state(follower readback) / degree+percent 단위 / 관절 순서 확인됨.",
        joint_names=tuple(joint_names),
        codebase_version=codebase_version,
        robot_type=robot_type,
        declared_episode_count=int(declared_episodes) if declared_episodes is not None else None,
        declared_frame_count=int(declared_frames) if declared_frames is not None else None,
        data_file_count=len(data_files),
        info_json_path=info_path,
        content_hash=_content_hash(data_files),
    )


def _content_hash(data_files: list[Path]) -> str:
    """sha256 over (filename, file bytes) for every data parquet file, sorted by name.

    Used to detect "same recording, different path" (e.g. a copy of a dataset
    directory) so it is never counted twice. This deliberately hashes file
    *content*, not path/mtime, so a byte-identical copy anywhere on disk
    collapses to the same hash regardless of where it was found.
    """
    hasher = hashlib.sha256()
    for path in sorted(data_files, key=lambda p: p.name):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _load_state_matrix(dataset: DatasetTrust) -> tuple[np.ndarray, np.ndarray, int]:
    """Read-only load of pooled ``observation.state`` rows for a trusted dataset.

    Returns ``(matrix, episode_index_array, frame_count)`` where ``matrix`` has
    shape ``(N, len(dataset.joint_names))``. Only ``pd.read_parquet`` is used;
    nothing is written back.
    """
    assert dataset.joint_names is not None
    data_files = sorted(dataset.root.glob("data/chunk-*/file-*.parquet"))
    frames = [pd.read_parquet(path, columns=["observation.state", "episode_index"]) for path in data_files]
    combined = pd.concat(frames, ignore_index=True)
    matrix = np.stack(combined["observation.state"].to_numpy()).astype(np.float64)
    episode_index = combined["episode_index"].to_numpy()
    return matrix, episode_index, len(combined)


def _percentile_or_none(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def compute_joint_delta_diagnostics(dataset: DatasetTrust, joint: str) -> dict[str, Any]:
    """Read-only per-dataset diagnostics for one joint: how much did it actually move?

    Complements :func:`_compute_joint_result` (which only looks at the pooled
    value distribution) with *motion* diagnostics computed from consecutive
    frame-to-frame deltas *within each episode* (episode boundaries are never
    diffed across, since frame N of episode 1 and frame 0 of episode 2 are not
    a real transition). Used to investigate joints (like ``wrist_roll``) whose
    percentile band came out surprisingly narrow: is that because the joint
    was genuinely held still, or because of some other data issue?
    """
    if dataset.joint_names is None or joint not in dataset.joint_names:
        raise TeleopAnalysisError(f"'{joint}'는 {dataset.relative_path}의 joint_names에 없습니다.")
    column_index = dataset.joint_names.index(joint)

    data_files = sorted(dataset.root.glob("data/chunk-*/file-*.parquet"))
    frames = [
        pd.read_parquet(path, columns=["observation.state", "episode_index", "frame_index"])
        for path in data_files
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    values = np.stack(combined["observation.state"].to_numpy()).astype(np.float64)[:, column_index]
    finite = values[np.isfinite(values)]

    deltas: list[np.ndarray] = []
    for _, group in combined.groupby("episode_index", sort=True):
        idx = group.index.to_numpy()
        episode_values = values[idx]
        episode_values = episode_values[np.isfinite(episode_values)]
        if episode_values.size > 1:
            deltas.append(np.abs(np.diff(episode_values)))
    all_deltas = np.concatenate(deltas) if deltas else np.array([])

    return {
        "dataset": dataset.relative_path,
        "joint": joint,
        "episode_count": int(combined["episode_index"].nunique()),
        "frame_count": int(len(combined)),
        "sample_count": int(finite.size),
        "nan_count": int(values.size - finite.size),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "p01": _percentile_or_none(finite, 1.0),
        "p05": _percentile_or_none(finite, 5.0),
        "p50": _percentile_or_none(finite, 50.0),
        "p95": _percentile_or_none(finite, 95.0),
        "p99": _percentile_or_none(finite, 99.0),
        "unique_value_count": int(np.unique(finite).size) if finite.size else 0,
        "max_abs_consecutive_delta": float(np.max(all_deltas)) if all_deltas.size else None,
        "frac_abs_delta_ge_0_1": float(np.mean(all_deltas >= 0.1)) if all_deltas.size else None,
        "frac_abs_delta_ge_0_5": float(np.mean(all_deltas >= 0.5)) if all_deltas.size else None,
        "frac_abs_delta_ge_1_0": float(np.mean(all_deltas >= 1.0)) if all_deltas.size else None,
        "delta_sample_count": int(all_deltas.size),
    }


def _compute_joint_result(
    joint: str,
    samples: np.ndarray,
    nan_count: int,
    episode_indices: set[tuple[str, int]],
    source_files: set[str],
    policy: AnalysisPolicy,
) -> JointResult:
    unit = "degree" if joint in DEGREE_JOINTS else ("percent_0_100" if joint in PERCENT_JOINTS else None)
    finite = samples[np.isfinite(samples)]
    sample_count = int(finite.size)
    episode_count = len(episode_indices)
    source_file_count = len(source_files)

    if sample_count == 0:
        return JointResult(
            joint=joint,
            status=STATUS_UNTRUSTED_SOURCE,
            unit=unit,
            sample_count=0,
            nan_count=nan_count,
            episode_count=episode_count,
            source_file_count=source_file_count,
            status_detail="신뢰 가능한 follower 샘플이 전혀 없습니다.",
        )

    if unit is None:
        return JointResult(
            joint=joint,
            status=STATUS_UNKNOWN_UNIT,
            unit=None,
            sample_count=sample_count,
            nan_count=nan_count,
            episode_count=episode_count,
            source_file_count=source_file_count,
            status_detail="단위를 코드/metadata로 확정할 수 없습니다.",
        )

    stats = dict(
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        p01=_percentile_or_none(finite, 1.0),
        p05=_percentile_or_none(finite, 5.0),
        p50=_percentile_or_none(finite, 50.0),
        p95=_percentile_or_none(finite, 95.0),
        p99=_percentile_or_none(finite, 99.0),
    )
    policy_low = _percentile_or_none(finite, policy.lower_percentile)
    policy_high = _percentile_or_none(finite, policy.upper_percentile)
    margin = policy.margin_for(joint)

    if sample_count < policy.minimum_samples:
        return JointResult(
            joint=joint,
            status=STATUS_INSUFFICIENT_SAMPLES,
            unit=unit,
            sample_count=sample_count,
            nan_count=nan_count,
            episode_count=episode_count,
            source_file_count=source_file_count,
            policy_percentile_low=policy_low,
            policy_percentile_high=policy_high,
            margin_applied=margin,
            status_detail=(
                f"샘플 수 {sample_count} < minimum_samples {policy.minimum_samples}."
            ),
            **stats,
        )

    assert policy_low is not None and policy_high is not None and margin is not None
    safe_min = policy_low + margin
    safe_max = policy_high - margin
    if safe_min >= safe_max:
        return JointResult(
            joint=joint,
            status=STATUS_MARGIN_COLLAPSED,
            unit=unit,
            sample_count=sample_count,
            nan_count=nan_count,
            episode_count=episode_count,
            source_file_count=source_file_count,
            policy_percentile_low=policy_low,
            policy_percentile_high=policy_high,
            margin_applied=margin,
            status_detail=(
                f"margin({margin}) 적용 후 안전 범위가 역전/소멸했습니다: "
                f"[{policy_low}+{margin}={safe_min}, {policy_high}-{margin}={safe_max}]."
            ),
            **stats,
        )

    return JointResult(
        joint=joint,
        status=STATUS_TRUSTED,
        unit=unit,
        sample_count=sample_count,
        nan_count=nan_count,
        episode_count=episode_count,
        source_file_count=source_file_count,
        policy_percentile_low=policy_low,
        policy_percentile_high=policy_high,
        margin_applied=margin,
        historical_safe_min=safe_min,
        historical_safe_max=safe_max,
        status_detail=None,
        **stats,
    )


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dataset_provenance(dataset: DatasetTrust, data_files: list[Path]) -> dict[str, Any]:
    info_bytes = dataset.info_json_path.read_bytes() if dataset.info_json_path else b""
    file_manifest = []
    for path in data_files:
        stat = path.stat()
        file_manifest.append(
            {
                "file": path.name,
                "size_bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return {
        "name": dataset.root.name,
        "relative_path": dataset.relative_path,
        "codebase_version": dataset.codebase_version,
        "robot_type": dataset.robot_type,
        "declared_episode_count": dataset.declared_episode_count,
        "declared_frame_count": dataset.declared_frame_count,
        "data_file_count": dataset.data_file_count,
        "info_json_sha256": _hash_bytes(info_bytes),
        "content_hash": dataset.content_hash,
        "data_files": file_manifest,
    }


def _label_path(path: Path) -> str:
    """Human-readable label for a filesystem path that never leaks the raw absolute path.

    Paths under the user's home directory are shown as ``~/...``; everything else
    falls back to just the final path component. Used only for report/JSON display.
    """
    home = Path.home()
    resolved = path.resolve()
    try:
        return "~/" + str(resolved.relative_to(home))
    except ValueError:
        return resolved.name


def deduplicate_by_content(trusted: list[DatasetTrust]) -> tuple[list[DatasetTrust], list[DatasetTrust]]:
    """Split trusted datasets into (kept, duplicates) by ``content_hash``.

    When two or more trusted dataset roots share the same ``content_hash`` (the
    exact same recording, found via more than one path - e.g. a mirrored/copied
    directory), only the lexicographically-first ``relative_path`` is kept; the
    rest are returned as duplicates so callers can report them without summing
    their samples/episodes twice.
    """
    by_hash: dict[str, list[DatasetTrust]] = {}
    for dataset in trusted:
        by_hash.setdefault(dataset.content_hash or f"__no_hash__:{dataset.relative_path}", []).append(dataset)

    kept: list[DatasetTrust] = []
    duplicates: list[DatasetTrust] = []
    for group in by_hash.values():
        group_sorted = sorted(group, key=lambda d: d.relative_path)
        kept.append(group_sorted[0])
        for extra in group_sorted[1:]:
            duplicates.append(
                DatasetTrust(
                    root=extra.root,
                    relative_path=extra.relative_path,
                    trusted=False,
                    reason=(
                        f"중복 데이터셋 - {group_sorted[0].relative_path}와 content_hash가 동일한 "
                        "동일 recording의 사본입니다 (샘플을 두 번 합산하지 않음)."
                    ),
                    joint_names=extra.joint_names,
                    codebase_version=extra.codebase_version,
                    robot_type=extra.robot_type,
                    declared_episode_count=extra.declared_episode_count,
                    declared_frame_count=extra.declared_frame_count,
                    data_file_count=extra.data_file_count,
                    info_json_path=extra.info_json_path,
                    content_hash=extra.content_hash,
                )
            )
    return sorted(kept, key=lambda d: d.relative_path), duplicates


def analyze(
    input_paths: list[Path] | None,
    *,
    project_root: Path,
    default_search_root: Path | list[Path],
    policy: AnalysisPolicy,
    calibration_search_roots: list[Path] | None = None,
) -> tuple[AnalysisResult, list[Path]]:
    """Run the full read-only analysis.

    If ``input_paths`` is empty/None, datasets are auto-discovered under
    ``default_search_root`` (a single directory or a list of directories).
    Returns ``(result, resolved_dataset_roots)`` - the caller should always
    print ``resolved_dataset_roots`` so the actual selected paths are visible
    even in auto-discovery mode.

    Datasets that are byte-for-byte the same recording as one already selected
    (see :func:`deduplicate_by_content`) are excluded and reported as
    duplicates rather than pooled a second time. ``calibration_search_roots``,
    if given, is scanned read-only for calibration JSON files
    (:func:`find_calibration_files`) purely for the report's ``provenance`` -
    they are never mixed into joint statistics.
    """
    if input_paths:
        candidate_roots = [Path(p).expanduser().resolve() for p in input_paths]
    else:
        candidate_roots = discover_dataset_roots(default_search_root)

    if not candidate_roots:
        raise TeleopAnalysisError(f"분석할 데이터셋을 찾지 못했습니다 (탐색 경로: {default_search_root}).")

    trust_results = [evaluate_dataset_trust(root, project_root) for root in candidate_roots]
    trusted_candidates = [d for d in trust_results if d.trusted]
    excluded = [d for d in trust_results if not d.trusted]

    used, duplicates = deduplicate_by_content(trusted_candidates)
    excluded = excluded + duplicates

    if not used:
        raise TeleopAnalysisError(
            "신뢰 가능한 follower 데이터셋이 하나도 없습니다. 제외 사유:\n"
            + "\n".join(f"  - {d.relative_path}: {d.reason}" for d in excluded)
        )

    per_joint_samples: dict[str, list[np.ndarray]] = {joint: [] for joint in KNOWN_JOINTS}
    per_joint_episode_ids: dict[str, set[tuple[str, int]]] = {joint: set() for joint in KNOWN_JOINTS}
    per_joint_source_files: dict[str, set[str]] = {joint: set() for joint in KNOWN_JOINTS}
    per_joint_nan_counts: dict[str, int] = {joint: 0 for joint in KNOWN_JOINTS}

    provenance_datasets = []
    total_episode_ids: set[tuple[str, int]] = set()
    total_sample_rows = 0

    for dataset in used:
        assert dataset.joint_names is not None
        matrix, episode_index, frame_count = _load_state_matrix(dataset)
        total_sample_rows += frame_count
        data_files = sorted(dataset.root.glob("data/chunk-*/file-*.parquet"))
        provenance_datasets.append(_dataset_provenance(dataset, data_files))

        for column_index, joint in enumerate(dataset.joint_names):
            column = matrix[:, column_index]
            nan_mask = ~np.isfinite(column)
            per_joint_nan_counts[joint] += int(np.count_nonzero(nan_mask))
            per_joint_samples[joint].append(column)
            per_joint_source_files[joint].add(dataset.relative_path)
            for ep in np.unique(episode_index):
                key = (dataset.relative_path, int(ep))
                per_joint_episode_ids[joint].add(key)
                total_episode_ids.add(key)

    joints: dict[str, JointResult] = {}
    for joint in KNOWN_JOINTS:
        pooled = (
            np.concatenate(per_joint_samples[joint]) if per_joint_samples[joint] else np.array([])
        )
        joints[joint] = _compute_joint_result(
            joint,
            pooled,
            per_joint_nan_counts[joint],
            per_joint_episode_ids[joint],
            per_joint_source_files[joint],
            policy,
        )

    manifest_json = json.dumps(provenance_datasets, sort_keys=True, ensure_ascii=False).encode("utf-8")

    calibration_files: list[dict[str, Any]] = []
    if calibration_search_roots:
        for entry in find_calibration_files(calibration_search_roots):
            calibration_files.append({**entry, "path": _label_path(Path(entry["path"]))})

    provenance = {
        "hash_method": (
            "sha256 over each dataset's meta/info.json bytes, plus a manifest listing each "
            "data parquet file's name/size/mtime (not its content) - full parquet payloads are "
            "not hashed for efficiency. content_hash (sha256 over each data parquet file's name+bytes) "
            "is used separately to deduplicate identical recordings found via more than one path."
        ),
        "manifest_sha256": _hash_bytes(manifest_json),
        "datasets": provenance_datasets,
        "duplicate_datasets_excluded": [
            {"path": d.relative_path, "reason": d.reason} for d in duplicates
        ],
        "calibration_files": calibration_files,
        "calibration_note": (
            "위 calibration_files는 raw tick range-of-motion(관절을 골고루 움직여 얻은 캘리브레이션 "
            "범위)이다. 텔레옵 percentile 통계에는 전혀 섞이지 않았다 - 별도 카테고리로만 보고된다."
        ),
        "reproducibility_note": (
            "data/ 와 configs/hardware.local.json은 .gitignore 대상이라 과거 녹화 시점 하드웨어 "
            "설정의 git 이력이 없습니다. 단위 확정은 현재 configs/hardware.local.json 내용과 "
            "SOFollowerConfig 기본값(use_degrees=True)에 근거합니다."
        ),
    }

    ordered_joints = {
        name: joints[name]
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    }

    result = AnalysisResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        trusted=True,
        used_datasets=used,
        excluded_datasets=excluded,
        joints=ordered_joints,
        policy=policy,
        total_episode_count=len(total_episode_ids),
        total_sample_count=total_sample_rows,
        provenance=provenance,
    )
    return result, candidate_roots
