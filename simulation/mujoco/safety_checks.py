"""Safety 검사: 실행 전 정적 검사 + 재생 중 동적 검사.

검사 결과는 PASS / WARN / BLOCKED 세 단계로 분류한다.
BLOCKED가 발생하면 기본적으로 즉시 재생을 중단한다 (--continue-on-warning으로도 우회 불가).
threshold는 코드에 흩어두지 않고 configs/mujoco_so101.yaml에서 읽는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import yaml

from simulation.mujoco.action_mapping import JointMapping
from simulation.mujoco.dataset_loader import DatasetInfo, EpisodeData
from simulation.mujoco.so101_model import get_geom_id, get_joint_limits

Level = Literal["PASS", "WARN", "BLOCKED"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "mujoco_so101.yaml"


class SafetyConfigError(RuntimeError):
    """configs/mujoco_so101.yaml 로딩/검증 실패."""


@dataclass(frozen=True)
class SafetyEvent:
    level: Level
    code: str
    message: str
    frame: int | None = None
    joint: str | None = None
    value: float | None = None
    limit: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "frame": self.frame,
            "joint": self.joint,
            "value": self.value,
            "limit": list(self.limit) if self.limit is not None else None,
        }


@dataclass(frozen=True)
class SafetyConfig:
    scene_path: Path
    table_geom_name: str
    default_dataset_root: str
    default_episode_index: int
    default_speed: float
    joint_limit_tolerance_rad: float
    stop_on_joint_limit: bool
    stop_on_nan: bool
    stop_on_collision: bool
    max_joint_delta_per_frame: dict[str, float]
    max_velocity: dict[str, float]
    warn_contact_count: int
    blocked_contact_count: int
    progress_interval_seconds: float
    progress_interval_frames: int


def load_safety_config(config_path: str | Path | None = None, project_root: Path | None = None) -> SafetyConfig:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise SafetyConfigError(f"safety 설정 파일을 찾을 수 없습니다: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SafetyConfigError(f"{path} 파싱 실패: {exc}") from exc

    root = project_root or Path(__file__).resolve().parents[2]
    try:
        model_cfg = raw["model"]
        dataset_cfg = raw["dataset"]
        playback_cfg = raw["playback"]
        safety_cfg = raw["safety"]
        console_cfg = raw["console"]
        return SafetyConfig(
            scene_path=(root / model_cfg["scene_path"]).resolve(),
            table_geom_name=model_cfg["table_geom_name"],
            default_dataset_root=dataset_cfg["default_dataset_root"],
            default_episode_index=int(dataset_cfg["default_episode_index"]),
            default_speed=float(playback_cfg["default_speed"]),
            joint_limit_tolerance_rad=float(safety_cfg["joint_limit_tolerance_rad"]),
            stop_on_joint_limit=bool(safety_cfg["stop_on_joint_limit"]),
            stop_on_nan=bool(safety_cfg["stop_on_nan"]),
            stop_on_collision=bool(safety_cfg["stop_on_collision"]),
            max_joint_delta_per_frame={k: float(v) for k, v in safety_cfg["max_joint_delta_per_frame"].items()},
            max_velocity={k: float(v) for k, v in safety_cfg["max_velocity"].items()},
            warn_contact_count=int(safety_cfg["contact"]["warn_contact_count"]),
            blocked_contact_count=int(safety_cfg["contact"]["blocked_contact_count"]),
            progress_interval_seconds=float(console_cfg["progress_interval_seconds"]),
            progress_interval_frames=int(console_cfg["progress_interval_frames"]),
        )
    except KeyError as exc:
        raise SafetyConfigError(f"{path}에 필수 설정 키가 없습니다: {exc}") from exc


# ---------------------------------------------------------------------------
# 실행 전 정적 검사
# ---------------------------------------------------------------------------


def run_static_checks(
    episode: EpisodeData,
    dataset_info: DatasetInfo,
    mapping: tuple[JointMapping, ...],
    model: mujoco.MjModel,
) -> list[SafetyEvent]:
    events: list[SafetyEvent] = []

    # action shape 검사
    expected_dim = dataset_info.action_dim
    if episode.action.ndim != 2 or episode.action.shape[1] != expected_dim:
        events.append(
            SafetyEvent(
                "BLOCKED",
                "action_shape",
                f"action shape이 예상과 다릅니다: {episode.action.shape}, 예상 차원={expected_dim}",
            )
        )
        return events  # 이후 검사는 shape을 전제하므로 여기서 중단

    if episode.length == 0 or episode.action.shape[0] == 0:
        events.append(SafetyEvent("BLOCKED", "empty_action", "action이 비어 있습니다 (프레임 0개)."))
        return events

    # NaN / Inf
    if not np.isfinite(episode.action).all():
        bad_frames = np.where((~np.isfinite(episode.action)).any(axis=1))[0]
        first = int(bad_frames[0]) if len(bad_frames) else None
        events.append(
            SafetyEvent(
                "BLOCKED",
                "action_non_finite",
                f"action에 NaN 또는 Inf가 있습니다. 첫 발생 프레임={first}, 총 {len(bad_frames)}개 프레임.",
                frame=first,
            )
        )

    # 관절 이름 누락/중복은 build_default_mapping()에서 이미 예외로 처리되므로,
    # 여기서는 매핑이 dataset_info.action_names를 빠짐없이 1:1로 덮는지만 재확인한다.
    mapped_names = [entry.dataset_feature_name for entry in mapping]
    if sorted(mapped_names) != sorted(dataset_info.action_names):
        events.append(
            SafetyEvent(
                "BLOCKED",
                "joint_mapping_incomplete",
                f"매핑된 이름({mapped_names})이 데이터셋 action 이름({list(dataset_info.action_names)})과 다릅니다.",
            )
        )
    if len({entry.mujoco_joint_name for entry in mapping}) != len(mapping):
        events.append(SafetyEvent("BLOCKED", "joint_mapping_duplicate", "동일한 MuJoCo joint에 매핑된 항목이 2개 이상입니다."))

    # actuator mapping 존재 여부 (모델에 실제로 있는지)
    actuator_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)}
    for entry in mapping:
        if entry.mujoco_actuator_name not in actuator_names:
            events.append(
                SafetyEvent(
                    "BLOCKED",
                    "actuator_mapping_missing",
                    f"'{entry.dataset_name}' -> actuator '{entry.mujoco_actuator_name}'가 모델에 없습니다.",
                    joint=entry.dataset_name,
                )
            )

    # timestamp 역전
    if len(episode.timestamp) > 1:
        diffs = np.diff(episode.timestamp)
        reversed_at = np.where(diffs < -1e-9)[0]
        if len(reversed_at):
            events.append(
                SafetyEvent(
                    "BLOCKED",
                    "timestamp_reversed",
                    f"timestamp가 역전되는 지점이 {len(reversed_at)}개 있습니다. 첫 지점 프레임={int(reversed_at[0])}.",
                    frame=int(reversed_at[0]),
                )
            )

    # frame_index 불연속 (0부터 연속 증가해야 함)
    expected_frame_index = np.arange(len(episode.frame_index), dtype=np.int64)
    if not np.array_equal(episode.frame_index, expected_frame_index):
        mismatch = np.where(episode.frame_index != expected_frame_index)[0]
        first = int(mismatch[0]) if len(mismatch) else None
        events.append(
            SafetyEvent(
                "BLOCKED",
                "frame_index_discontinuous",
                f"frame_index가 0부터 연속이 아닙니다. 첫 불일치 위치={first}.",
                frame=first,
            )
        )

    # action 범위 사전 스캔 (WARN 수준 사전 경고 - 정확한 프레임 판정은 동적 검사에서 수행)
    if np.isfinite(episode.action).all():
        limits = get_joint_limits(model, tuple(entry.mujoco_joint_name for entry in mapping))
        over_count = 0
        for entry in mapping:
            lo, hi = limits[entry.mujoco_joint_name].joint_range
            values = episode.action[:, entry.dataset_index] * entry.scale * entry.sign + entry.offset
            over = np.sum((values < lo) | (values > hi))
            over_count += int(over)
        if over_count > 0:
            events.append(
                SafetyEvent(
                    "WARN",
                    "action_range_prescan",
                    f"사전 스캔 결과 관절 range를 벗어나는 값-프레임 조합이 총 {over_count}개 있습니다. "
                    "재생 중 해당 프레임에서 BLOCKED로 중단될 수 있습니다.",
                )
            )

    if not any(evt.level == "BLOCKED" for evt in events):
        events.append(SafetyEvent("PASS", "static_checks", "실행 전 정적 검사를 통과했습니다."))
    return events


# ---------------------------------------------------------------------------
# 재생 중 동적 검사
# ---------------------------------------------------------------------------


@dataclass
class DynamicCheckState:
    """프레임 간 상태(이전 target 등)를 들고 있는 누적 상태."""

    previous_target_rad: dict[str, float] | None = None
    joint_limit_violations: list[SafetyEvent] = field(default_factory=list)
    actuator_limit_violations: list[SafetyEvent] = field(default_factory=list)
    delta_violations: list[SafetyEvent] = field(default_factory=list)
    velocity_violations: list[SafetyEvent] = field(default_factory=list)
    collisions: list[SafetyEvent] = field(default_factory=list)
    nan_events: list[SafetyEvent] = field(default_factory=list)


def check_frame_targets(
    frame: int,
    target_rad: dict[str, float],
    mapping: tuple[JointMapping, ...],
    model: mujoco.MjModel,
    config: SafetyConfig,
    state: DynamicCheckState,
) -> list[SafetyEvent]:
    """물리 스텝 이전: 이번 프레임 target이 관절/actuator range, 프레임간 변화량을 지키는지 확인."""
    events: list[SafetyEvent] = []
    limits = get_joint_limits(model, tuple(entry.mujoco_joint_name for entry in mapping))
    tol = config.joint_limit_tolerance_rad

    for entry in mapping:
        joint_name = entry.mujoco_joint_name
        value = target_rad[entry.mujoco_actuator_name]
        lim = limits[joint_name]

        lo, hi = lim.joint_range
        if value < lo - tol or value > hi + tol:
            evt = SafetyEvent(
                "BLOCKED",
                "joint_limit",
                f"{joint_name}가 허용 범위를 초과했습니다.",
                frame=frame,
                joint=joint_name,
                value=value,
                limit=(lo, hi),
            )
            events.append(evt)
            state.joint_limit_violations.append(evt)

        if lim.actuator_ctrlrange is not None:
            clo, chi = lim.actuator_ctrlrange
            if value < clo - tol or value > chi + tol:
                evt = SafetyEvent(
                    "BLOCKED",
                    "actuator_limit",
                    f"{entry.mujoco_actuator_name} actuator control range를 초과했습니다.",
                    frame=frame,
                    joint=joint_name,
                    value=value,
                    limit=(clo, chi),
                )
                events.append(evt)
                state.actuator_limit_violations.append(evt)

        if state.previous_target_rad is not None:
            prev = state.previous_target_rad[entry.mujoco_actuator_name]
            delta = abs(value - prev)
            max_delta = config.max_joint_delta_per_frame.get(entry.dataset_name)
            if max_delta is not None and delta > max_delta:
                evt = SafetyEvent(
                    "WARN",
                    "max_delta",
                    f"{joint_name}의 프레임간 변화량이 임계값을 초과했습니다.",
                    frame=frame,
                    joint=joint_name,
                    value=delta,
                    limit=(0.0, max_delta),
                )
                events.append(evt)
                state.delta_violations.append(evt)

    state.previous_target_rad = dict(target_rad)
    return events


def check_simulation_state(
    frame: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapping: tuple[JointMapping, ...],
    config: SafetyConfig,
    state: DynamicCheckState,
) -> list[SafetyEvent]:
    """물리 스텝 이후: qpos/qvel NaN, 속도, 접촉을 확인한다."""
    events: list[SafetyEvent] = []

    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        evt = SafetyEvent(
            "BLOCKED", "simulation_nan", "MuJoCo qpos 또는 qvel에 NaN/Inf가 발생했습니다.", frame=frame
        )
        events.append(evt)
        state.nan_events.append(evt)
        return events  # 발산 상태에서 이후 검사는 의미가 없음

    for entry in mapping:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
        dof_id = model.jnt_dofadr[joint_id]
        velocity = float(data.qvel[dof_id])
        max_vel = config.max_velocity.get(entry.dataset_name)
        if max_vel is not None and abs(velocity) > max_vel:
            evt = SafetyEvent(
                "WARN",
                "velocity_limit",
                f"{entry.mujoco_joint_name}의 속도가 임계값을 초과했습니다.",
                frame=frame,
                joint=entry.mujoco_joint_name,
                value=velocity,
                limit=(-max_vel, max_vel),
            )
            events.append(evt)
            state.velocity_violations.append(evt)

    # 발산 감시 (임계값을 크게 벗어난 값 - NaN은 아니지만 수치적으로 폭주하는 경우)
    if np.abs(data.qvel).max() > 1e4 or np.abs(data.qpos).max() > 1e4:
        evt = SafetyEvent(
            "BLOCKED", "simulation_divergence", "시뮬레이션 상태값이 비정상적으로 발산했습니다.", frame=frame
        )
        events.append(evt)
        state.nan_events.append(evt)

    # 접촉(contact) 분류: table_surface geom과의 접촉 vs 그 외(로봇 자기 자신 간 접촉으로 간주).
    table_geom_id = get_geom_id(model, config.table_geom_name)
    n_contacts = data.ncon
    table_contacts = 0
    self_contacts = 0
    for i in range(n_contacts):
        contact = data.contact[i]
        if table_geom_id is not None and (contact.geom1 == table_geom_id or contact.geom2 == table_geom_id):
            table_contacts += 1
        else:
            self_contacts += 1

    if table_contacts > 0:
        evt = SafetyEvent(
            "WARN", "table_collision", f"로봇이 table 표면과 접촉했습니다 ({table_contacts}개 contact).", frame=frame
        )
        events.append(evt)
        state.collisions.append(evt)
    if self_contacts > 0:
        evt = SafetyEvent(
            "WARN", "self_collision", f"로봇 자기 자신 간 접촉이 감지되었습니다 ({self_contacts}개 contact).", frame=frame
        )
        events.append(evt)
        state.collisions.append(evt)

    if n_contacts >= config.blocked_contact_count and config.stop_on_collision:
        evt = SafetyEvent(
            "BLOCKED",
            "contact_spike",
            f"동시 contact 개수({n_contacts})가 임계값({config.blocked_contact_count})을 초과했습니다.",
            frame=frame,
            value=float(n_contacts),
            limit=(0.0, float(config.blocked_contact_count)),
        )
        events.append(evt)
        state.collisions.append(evt)
    elif n_contacts >= config.warn_contact_count:
        evt = SafetyEvent(
            "WARN",
            "contact_spike",
            f"동시 contact 개수({n_contacts})가 주의 임계값({config.warn_contact_count})을 초과했습니다.",
            frame=frame,
            value=float(n_contacts),
            limit=(0.0, float(config.warn_contact_count)),
        )
        events.append(evt)
        state.collisions.append(evt)

    return events


def filter_active_blocking(events: list[SafetyEvent], config: SafetyConfig) -> list[SafetyEvent]:
    """stop_on_* 플래그가 꺼져 있으면 해당 카테고리의 BLOCKED를 WARN으로 낮춘다."""
    result = []
    for evt in events:
        level = evt.level
        if level == "BLOCKED":
            if evt.code in ("joint_limit", "actuator_limit") and not config.stop_on_joint_limit:
                level = "WARN"
            elif evt.code in ("simulation_nan", "simulation_divergence") and not config.stop_on_nan:
                level = "WARN"
            elif evt.code == "contact_spike" and not config.stop_on_collision:
                level = "WARN"
        if level != evt.level:
            result.append(SafetyEvent(level, evt.code, evt.message, evt.frame, evt.joint, evt.value, evt.limit))
        else:
            result.append(evt)
    return result
