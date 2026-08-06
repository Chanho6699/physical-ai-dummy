"""데이터셋 action을 SO-101 MuJoCo 모델에서 재생하는 orchestrator.

이 모듈은 실물 하드웨어를 전혀 호출하지 않는다 (USB serial 없음, ROS2 없음, SmolVLA 추론 없음).
로딩 -> 매핑 -> 정적 검사 -> (dry-run이 아니면) 물리 재생 + 동적 검사 -> 리포트 저장 순서로 진행한다.

WARN/BLOCKED 처리 정책:
  - BLOCKED가 발생하면 --continue-on-warning 여부와 무관하게 즉시 재생을 중단한다.
  - WARN이 발생하면 기본적으로도 재생을 멈추고 사용자가 검토하도록 한다
    (안전 검증 도구이므로 기본값을 보수적으로 둔다).
    --continue-on-warning을 주면 WARN은 무시하고 계속 진행한다 (BLOCKED는 여전히 항상 중단).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from simulation.mujoco import console_status as cs
from simulation.mujoco.action_mapping import (
    ActionMappingError,
    JointMapping,
    build_default_mapping,
    map_action_row,
    validate_mapping_against_model,
)
from simulation.mujoco.dataset_loader import (
    DatasetLoadError,
    EpisodeData,
    InvalidEpisodeIndexError,
    load_dataset_info,
    load_episode,
)
from simulation.mujoco.safety_checks import (
    DynamicCheckState,
    SafetyConfig,
    SafetyConfigError,
    SafetyEvent,
    check_frame_targets,
    check_simulation_state,
    filter_active_blocking,
    load_safety_config,
    run_static_checks,
)
from simulation.mujoco.so101_model import SO101ModelError, load_model, make_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ReplayArgs:
    dataset_root: Path
    episode_index: int
    speed: float = 1.0
    mode: str = "headless"  # "headless" | "gui"
    max_frames: int | None = None
    start_frame: int = 0
    report_path: Path | None = None
    config_path: Path | None = None
    quiet: bool = False
    verbose: bool = False
    no_color: bool = False
    dry_run: bool = False
    continue_on_warning: bool = False


@dataclass
class ReplayOutcome:
    final_result: str  # PASS | WARN | BLOCKED
    report_path: Path | None
    report: dict
    exit_code: int


class ReplayAbortedError(RuntimeError):
    """정적 검사 단계에서 BLOCKED가 발생해 재생을 시작할 수 없는 경우."""


def _unique_report_path(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def _resolve_report_path(explicit_path: Path | None, episode_index: int) -> Path:
    """리포트 저장 경로를 결정한다. 지정하지 않으면 기본 경로를 쓰고, 이미 파일이
    있으면 (기본/명시 경로 모두) timestamp suffix를 붙여 덮어쓰지 않는다."""
    path = explicit_path or (PROJECT_ROOT / "reports" / "mujoco_replay" / f"episode_{episode_index:03d}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return _unique_report_path(path)


def _events_to_dicts(events: list[SafetyEvent]) -> list[dict]:
    return [evt.to_dict() for evt in events]


def _build_report(
    *,
    args: ReplayArgs,
    episode: EpisodeData,
    mapping: tuple[JointMapping, ...],
    static_events: list[SafetyEvent],
    dynamic_state: DynamicCheckState,
    processed_frames: int,
    blocked_event: SafetyEvent | None,
    warning_stop_event: SafetyEvent | None,
) -> dict:
    action_min = {entry.dataset_name: float(np.min(episode.action[:, entry.dataset_index])) for entry in mapping}
    action_max = {entry.dataset_name: float(np.max(episode.action[:, entry.dataset_index])) for entry in mapping}

    max_frame_delta_rad: dict[str, float] = {}
    if episode.action.shape[0] > 1:
        for entry in mapping:
            rad_values = episode.action[:, entry.dataset_index] * entry.scale * entry.sign + entry.offset
            max_frame_delta_rad[entry.dataset_name] = float(np.max(np.abs(np.diff(rad_values))))

    all_events = static_events + list(dynamic_state.joint_limit_violations) + list(dynamic_state.actuator_limit_violations)
    warnings_events = (
        [e for e in static_events if e.level == "WARN"]
        + list(dynamic_state.delta_violations)
    )

    final_result = "PASS"
    if blocked_event is not None:
        final_result = "BLOCKED"
    elif warnings_events or dynamic_state.velocity_violations or dynamic_state.collisions or warning_stop_event is not None:
        final_result = "WARN"

    return {
        "dataset_root": str(args.dataset_root),
        "episode_index": episode.episode_index,
        "task": episode.task,
        "frame_count": episode.length,
        "processed_frames": processed_frames,
        "fps": episode.fps,
        "playback_speed": args.speed,
        "mode": args.mode,
        "dry_run": args.dry_run,
        "joint_names": [entry.dataset_name for entry in mapping],
        "action_shape": list(episode.action.shape),
        "action_min": action_min,
        "action_max": action_max,
        "action_unit": "deg (dataset 원본 단위, mujoco 반영 시 rad로 변환)",
        "max_frame_delta": max_frame_delta_rad,
        "max_frame_delta_unit": "rad",
        "joint_limit_violations": _events_to_dicts(dynamic_state.joint_limit_violations),
        "actuator_limit_violations": _events_to_dicts(dynamic_state.actuator_limit_violations),
        "velocity_violations": _events_to_dicts(dynamic_state.velocity_violations),
        "collisions": _events_to_dicts(dynamic_state.collisions),
        "simulation_nan_count": len(dynamic_state.nan_events),
        "warnings": _events_to_dicts(warnings_events),
        "blocked_reason": blocked_event.message if blocked_event is not None else None,
        "warning_stop_reason": warning_stop_event.message if warning_stop_event is not None else None,
        "final_result": final_result,
    }


def run_replay(args: ReplayArgs) -> ReplayOutcome:
    opts = cs.ConsoleOptions(quiet=args.quiet, verbose=args.verbose, use_color=cs.resolve_use_color(args.no_color))

    try:
        config = load_safety_config(args.config_path)
    except SafetyConfigError as exc:
        cs.print_error(str(exc))
        return ReplayOutcome("BLOCKED", None, {}, 2)

    try:
        dataset_info = load_dataset_info(args.dataset_root)
        episode = load_episode(args.dataset_root, args.episode_index, dataset_info)
    except DatasetLoadError as exc:
        cs.print_error(str(exc))
        return ReplayOutcome("BLOCKED", None, {}, 2)
    except InvalidEpisodeIndexError as exc:
        cs.print_error(f"존재하지 않는 에피소드입니다.\n[요청] {exc.requested}\n[허용 범위] 0 ~ {exc.total_episodes - 1}")
        return ReplayOutcome("BLOCKED", None, {}, 2)

    if args.start_frame > 0 or args.max_frames is not None:
        end = episode.length if args.max_frames is None else min(episode.length, args.start_frame + args.max_frames)
        episode = EpisodeData(
            episode_index=episode.episode_index,
            length=end - args.start_frame,
            task=episode.task,
            fps=episode.fps,
            joint_names=episode.joint_names,
            action=episode.action[args.start_frame:end],
            state=episode.state[args.start_frame:end],
            timestamp=episode.timestamp[args.start_frame:end],
            frame_index=episode.frame_index[args.start_frame:end],
        )

    mode_label = "GUI" if args.mode == "gui" else "Headless"
    if args.dry_run:
        mode_label += " (dry-run)"
    cs.print_header(
        opts,
        dataset_root=str(args.dataset_root),
        episode_index=episode.episode_index,
        total_frames=episode.length,
        fps=episode.fps,
        speed=args.speed,
        mode=mode_label,
    )

    cs.print_section(opts, "데이터셋 파일 확인 중")
    cs.print_check(opts, "PASS", "필수 metadata 파일 확인", "meta/info.json, episodes, data parquet 확인 완료")

    cs.print_section(opts, "action feature와 관절 이름 확인 중")
    try:
        mapping = build_default_mapping(list(dataset_info.action_names))
    except ActionMappingError as exc:
        cs.print_check(opts, "BLOCKED", "action mapping", str(exc))
        return ReplayOutcome("BLOCKED", None, {}, 1)
    cs.print_check(opts, "PASS", f"{dataset_info.action_dim}차원 action 확인", f"{list(dataset_info.action_names)}")

    cs.print_section(opts, "MuJoCo 모델 로딩 중")
    try:
        model = load_model(config.scene_path)
        validate_mapping_against_model(mapping, model)
    except (SO101ModelError, ActionMappingError) as exc:
        cs.print_check(opts, "BLOCKED", "MuJoCo 모델", str(exc))
        return ReplayOutcome("BLOCKED", None, {}, 1)
    cs.print_check(opts, "PASS", "SO-101 모델 로딩 완료", f"{config.scene_path.name} (joint={model.njnt}, actuator={model.nu})")

    cs.print_mapping_table(opts, mapping)

    cs.print_section(opts, "Safety 사전 검사 중")
    static_events = run_static_checks(episode, dataset_info, mapping, model)
    for evt in static_events:
        cs.print_check(opts, evt.level, evt.code, evt.message)

    blocked_static = [e for e in static_events if e.level == "BLOCKED"]
    if blocked_static:
        for evt in blocked_static:
            cs.print_blocked(opts, evt)
        dynamic_state = DynamicCheckState()
        report = _build_report(
            args=args,
            episode=episode,
            mapping=mapping,
            static_events=static_events,
            dynamic_state=dynamic_state,
            processed_frames=0,
            blocked_event=blocked_static[0],
            warning_stop_event=None,
        )
        report_path = _resolve_report_path(args.report_path, episode.episode_index)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        cs.print_final_summary(
            opts,
            processed_frames=0,
            total_frames=episode.length,
            joint_limit_violations=0,
            actuator_limit_violations=0,
            max_delta_violations=0,
            collisions=0,
            nan_count=0,
            final_result="BLOCKED",
            report_path=str(report_path),
        )
        return ReplayOutcome("BLOCKED", report_path, report, 1)

    if args.dry_run:
        warn_count = len([e for e in static_events if e.level == "WARN"])
        cs.print_dry_run_summary(opts, would_process_frames=episode.length, precheck_warnings=warn_count)
        dynamic_state = DynamicCheckState()
        report = _build_report(
            args=args,
            episode=episode,
            mapping=mapping,
            static_events=static_events,
            dynamic_state=dynamic_state,
            processed_frames=0,
            blocked_event=None,
            warning_stop_event=None,
        )
        report_path = _resolve_report_path(args.report_path, episode.episode_index)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        cs.print_final_summary(
            opts,
            processed_frames=0,
            total_frames=episode.length,
            joint_limit_violations=0,
            actuator_limit_violations=0,
            max_delta_violations=0,
            collisions=0,
            nan_count=0,
            final_result=report["final_result"],
            report_path=str(report_path),
        )
        return ReplayOutcome(report["final_result"], report_path, report, 0)

    if not opts.quiet:
        print(f"[재생 시작] 에피소드 {episode.episode_index}을(를) 재생합니다.")
        print("=" * 68)

    data = make_data(model)
    mujoco.mj_resetData(model, data)

    # 실물 로봇은 이 에피소드를 기록하기 시작한 시점에 이미 observation.state[0] 위치에 있었다.
    # MuJoCo 기본 keyframe(전부 0)에서 시작하면, 첫 프레임에 0 -> action[0]으로 순간 이동하려는
    # 인위적인 "튐"이 생겨 관절 속도가 실제로는 없던 급변으로 오检되므로, qpos를 state[0]으로
    # 먼저 맞춘 뒤 재생을 시작한다.
    initial_targets = map_action_row(episode.state[0], mapping)
    for entry in mapping:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.mujoco_joint_name)
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = initial_targets[entry.mujoco_actuator_name]
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, entry.mujoco_actuator_name)] = (
            initial_targets[entry.mujoco_actuator_name]
        )
    mujoco.mj_forward(model, data)

    steps_per_frame = max(1, round((1.0 / episode.fps) / model.opt.timestep))

    dynamic_state = DynamicCheckState()
    dynamic_state.previous_target_rad = dict(initial_targets)
    blocked_event: SafetyEvent | None = None
    warning_stop_event: SafetyEvent | None = None
    processed_frames = 0

    viewer_cm = None
    viewer = None
    if args.mode == "gui":
        try:
            import mujoco.viewer as mj_viewer

            viewer_cm = mj_viewer.launch_passive(model, data)
            viewer = viewer_cm.__enter__()
        except Exception as exc:
            cs.print_error(f"GUI viewer를 시작할 수 없습니다: {exc}. headless로 계속 진행하지 않고 중단합니다.")
            return ReplayOutcome("BLOCKED", None, {}, 1)

    last_progress_time = time.monotonic()
    frame_dt = (1.0 / episode.fps) / max(args.speed, 1e-6)

    try:
        for local_frame in range(episode.length):
            frame_start = time.monotonic()
            target_rad = map_action_row(episode.action[local_frame], mapping)

            frame_events = check_frame_targets(local_frame, target_rad, mapping, model, config, dynamic_state)
            frame_events = filter_active_blocking(frame_events, config)
            cs.print_verbose_frame(opts, local_frame, frame_events)

            frame_blocked = next((e for e in frame_events if e.level == "BLOCKED"), None)
            if frame_blocked is not None:
                blocked_event = frame_blocked
                cs.print_blocked(opts, frame_blocked)
                break

            for actuator_name, value in target_rad.items():
                actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                data.ctrl[actuator_id] = value

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            sim_events = check_simulation_state(local_frame, model, data, mapping, config, dynamic_state)
            sim_events = filter_active_blocking(sim_events, config)
            cs.print_verbose_frame(opts, local_frame, sim_events)

            sim_blocked = next((e for e in sim_events if e.level == "BLOCKED"), None)
            processed_frames = local_frame + 1
            if sim_blocked is not None:
                blocked_event = sim_blocked
                cs.print_blocked(opts, sim_blocked)
                break

            all_frame_warns = [e for e in frame_events + sim_events if e.level == "WARN"]
            if all_frame_warns and not args.continue_on_warning and warning_stop_event is None:
                warning_stop_event = all_frame_warns[0]
                for evt in all_frame_warns:
                    cs.print_check(opts, "WARN", evt.code, f"(frame {local_frame}) {evt.message}")
                if not opts.quiet:
                    print("[중지] WARN이 발생해 재생을 멈췄습니다. 계속하려면 --continue-on-warning 옵션을 사용하세요.")
                break

            if viewer is not None:
                viewer.sync()

            now = time.monotonic()
            elapsed = now - last_progress_time
            if (
                elapsed >= config.progress_interval_seconds
                or local_frame % max(config.progress_interval_frames, 1) == 0
                or local_frame == episode.length - 1
            ):
                current_level = "BLOCKED" if blocked_event else ("WARN" if all_frame_warns else "PASS")
                cs.print_progress(opts, local_frame + 1, episode.length, current_level)
                last_progress_time = now

            if args.mode == "gui":
                sleep_for = frame_dt - (time.monotonic() - frame_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)

    report = _build_report(
        args=args,
        episode=episode,
        mapping=mapping,
        static_events=static_events,
        dynamic_state=dynamic_state,
        processed_frames=processed_frames,
        blocked_event=blocked_event,
        warning_stop_event=warning_stop_event,
    )
    report_path = _resolve_report_path(args.report_path, episode.episode_index)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    cs.print_final_summary(
        opts,
        processed_frames=processed_frames,
        total_frames=episode.length,
        joint_limit_violations=len(dynamic_state.joint_limit_violations),
        actuator_limit_violations=len(dynamic_state.actuator_limit_violations),
        max_delta_violations=len(dynamic_state.delta_violations),
        collisions=len(dynamic_state.collisions),
        nan_count=len(dynamic_state.nan_events),
        final_result=report["final_result"],
        report_path=str(report_path),
    )

    exit_code = 1 if report["final_result"] == "BLOCKED" else 0
    return ReplayOutcome(report["final_result"], report_path, report, exit_code)
