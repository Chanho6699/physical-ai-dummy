#!/usr/bin/env python3
"""baseline vs realistic MuJoCo control-mode 비교 벤치마크 (hardware-free).

리더암/팔로워암/노트북 상태 서버가 전혀 필요 없다 - 순수 합성(synthetic) action
trajectory를 만들어 기존 MuJoCo executor(``simulation/mujoco/so101_model.py`` +
``mujoco.mj_step``)에 두 가지 경로로 흘려보내고 비교한다::

    A. baseline:   synthetic action -> (기존) action_mapping -> mj_step
    B. realistic:  synthetic action -> RealisticControlLayer -> (기존) action_mapping -> mj_step

"baseline"은 ``RealisticControlLayer``를 아예 만들지 않는 것이 아니라, 4가지 특성을 모두
끈(``config_all_disabled()``) 같은 레이어를 통과시킨다 - ``tests/test_so101_realistic_control.py``
의 ``test_all_disabled_config_is_exact_passthrough``가 이미 이 조합이 완전한 identity
passthrough임을 증명해 두었으므로, 두 pass를 정확히 같은 코드 경로로 측정할 수 있어
비교가 더 공정해진다 (baseline이 다른 코드를 타서 생기는 우연한 차이를 배제).

이 스크립트는 실물 SO-101을 전혀 건드리지 않고, LeRobot 소스도 import하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.mujoco.action_mapping import build_default_mapping, map_positions_dict, validate_mapping_against_model
from simulation.mujoco.so101_model import SO101_JOINT_NAMES, get_actuator_id, get_joint_id, load_model, make_data
from simulation.realism.so101_control_profile import DEFAULT_PROFILE_PATH, ControlProfileError, load_control_profile
from simulation.realism.so101_realistic_control import (
    RealisticControlConfig,
    RealisticControlLayer,
    RealisticControlRecorder,
    config_all_disabled,
)

BAR = "=" * 72
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "realistic_control_benchmark"

WRIST_ROLL_DEADBAND_PHASE_STEPS = 60  # 이 구간 동안은 no-response candidate 폭 이내로만 흔든다
SHOULDER_PAN_JUMP_STEP = 5  # 이 스텝에서 shoulder_pan이 순간 점프한다 (delay/rate 관찰용)
SHOULDER_PAN_JUMP_TARGET_DEG = 60.0
GRIPPER_JUMP_STEP = 10
GRIPPER_JUMP_TARGET_PERCENT = 80.0


# ---------------------------------------------------------------------------
# 합성 action trajectory (결정론적 - RNG 없음)
# ---------------------------------------------------------------------------


def build_synthetic_trajectory(num_steps: int, *, wrist_roll_dither_deg: float) -> list[dict[str, float]]:
    """리더암/실물 데이터 없이 만드는 결정론적 합성 action 시퀀스.

    의도적으로 두 구간을 분리한다:
      - shoulder_pan/gripper: 특정 스텝에 큰 순간 점프 (rate/frame-delta, latency 관찰용).
      - wrist_roll: 처음 ``WRIST_ROLL_DEADBAND_PHASE_STEPS`` 스텝은 ``wrist_roll_dither_deg``
        진폭으로만 흔들다가(기본값은 profile의 no-response candidate 폭보다 작게 호출자가
        설정), 그 이후 큰 점프 1회 (transition region 이상 - deadband 통과 여부 관찰용).
    """
    trajectory: list[dict[str, float]] = []
    for step in range(num_steps):
        shoulder_pan = 0.0 if step < SHOULDER_PAN_JUMP_STEP else SHOULDER_PAN_JUMP_TARGET_DEG
        gripper = 10.0 if step < GRIPPER_JUMP_STEP else GRIPPER_JUMP_TARGET_PERCENT
        if step < WRIST_ROLL_DEADBAND_PHASE_STEPS:
            wrist_roll = wrist_roll_dither_deg if step % 2 == 0 else 0.0
        else:
            wrist_roll = 15.0
        trajectory.append(
            {
                "shoulder_pan": shoulder_pan,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": wrist_roll,
                "gripper": gripper,
            }
        )
    return trajectory


# ---------------------------------------------------------------------------
# 실행 (기존 MuJoCo executor 재사용 - mj_step을 이 스크립트가 새로 구현하지 않음)
# ---------------------------------------------------------------------------


@dataclass
class PassResult:
    label: str
    rows: list[dict]  # step별 {"processed": {...}, "actual": {...}, "velocity_deg_s": {...}}


def run_pass(
    *,
    label: str,
    trajectory: list[dict[str, float]],
    layer: RealisticControlLayer,
    model: mujoco.MjModel,
    mapping,
    dt_control: float,
    steps_per_frame: int,
    recorder: RealisticControlRecorder | None,
) -> PassResult:
    data = make_data(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    qpos_adr = {name: model.jnt_qposadr[get_joint_id(model, name)] for name in SO101_JOINT_NAMES}
    dof_adr = {name: model.jnt_dofadr[get_joint_id(model, name)] for name in SO101_JOINT_NAMES}
    actuator_ids = {entry.mujoco_actuator_name: get_actuator_id(model, entry.mujoco_actuator_name) for entry in mapping}

    rows: list[dict] = []
    for step, desired in enumerate(trajectory):
        now = step * dt_control
        simulated_actual = {name: math.degrees(float(data.qpos[adr])) for name, adr in qpos_adr.items()}

        result = layer.process(desired, now=now, simulated_actual=simulated_actual)
        if recorder is not None:
            recorder.record(step, result.diagnostics)

        processed_rad = map_positions_dict(result.processed_action, mapping)
        for actuator_name, value in processed_rad.items():
            data.ctrl[actuator_ids[actuator_name]] = value
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        actual_after = {name: math.degrees(float(data.qpos[adr])) for name, adr in qpos_adr.items()}
        velocity_after = {name: math.degrees(float(data.qvel[adr])) for name, adr in dof_adr.items()}
        rows.append({"step": step, "processed": dict(result.processed_action), "actual": actual_after, "velocity_deg_s": velocity_after})

    return PassResult(label=label, rows=rows)


# ---------------------------------------------------------------------------
# 비교 지표
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)), "max": float(np.max(arr))}


def summarize_joint(rows: list[dict], joint: str) -> dict:
    processed = [r["processed"][joint] for r in rows]
    actual = [r["actual"][joint] for r in rows]
    velocity = [abs(r["velocity_deg_s"][joint]) for r in rows]
    frame_delta = [abs(processed[i] - processed[i - 1]) for i in range(1, len(processed))]
    tracking_error = [abs(processed[i] - actual[i]) for i in range(len(processed))]
    return {
        "frame_delta": _percentiles(frame_delta) if frame_delta else None,
        "velocity_deg_s": _percentiles(velocity) if velocity else None,
        "tracking_error_deg_or_pct": _percentiles(tracking_error) if tracking_error else None,
    }


def command_to_motion_delay_steps(rows: list[dict], joint: str, jump_step: int, target: float, *, tolerance_fraction: float = 0.9) -> int | None:
    """``jump_step``에서 target으로 순간 점프한 뒤, MuJoCo actual이 target의
    ``tolerance_fraction``에 처음 도달하는 데 걸린 step 수. 도달 못하면 None."""
    start_value = rows[jump_step - 1]["actual"][joint] if jump_step > 0 else rows[0]["actual"][joint]
    threshold = start_value + tolerance_fraction * (target - start_value)
    for row in rows[jump_step:]:
        value = row["actual"][joint]
        reached = value >= threshold if target > start_value else value <= threshold
        if reached:
            return row["step"] - jump_step
    return None


def wrist_roll_motion_fraction(rows: list[dict], phase_end_step: int, *, epsilon_deg: float = 0.05) -> float:
    """deadband 구간(``phase_end_step`` 이전) 동안 실제로 step-to-step 위치가
    ``epsilon_deg``보다 크게 움직인 비율. baseline은 거의 매 스텝 움직이고, realistic
    (deadband 적용)은 훨씬 낮은 비율이 나올 것으로 기대한다 - 확정값이 아니라 관찰 지표."""
    moved = 0
    total = 0
    for i in range(1, min(phase_end_step, len(rows))):
        total += 1
        if abs(rows[i]["actual"]["wrist_roll"] - rows[i - 1]["actual"]["wrist_roll"]) > epsilon_deg:
            moved += 1
    return (moved / total) if total else 0.0


def build_comparison(baseline: PassResult, realistic: PassResult) -> dict:
    joints_out = {}
    for joint in SO101_JOINT_NAMES:
        joints_out[joint] = {
            "baseline": summarize_joint(baseline.rows, joint),
            "realistic": summarize_joint(realistic.rows, joint),
        }

    return {
        "joints": joints_out,
        "shoulder_pan_step_jump": {
            "jump_step": SHOULDER_PAN_JUMP_STEP,
            "target_deg": SHOULDER_PAN_JUMP_TARGET_DEG,
            "baseline_delay_steps": command_to_motion_delay_steps(baseline.rows, "shoulder_pan", SHOULDER_PAN_JUMP_STEP, SHOULDER_PAN_JUMP_TARGET_DEG),
            "realistic_delay_steps": command_to_motion_delay_steps(realistic.rows, "shoulder_pan", SHOULDER_PAN_JUMP_STEP, SHOULDER_PAN_JUMP_TARGET_DEG),
            "note": "step 수 - 실제 ms는 control_hz로 나눠서 해석 (delay_steps / control_hz * 1000).",
        },
        "wrist_roll_deadband_phase": {
            "phase_end_step": WRIST_ROLL_DEADBAND_PHASE_STEPS,
            "baseline_motion_fraction": wrist_roll_motion_fraction(baseline.rows, WRIST_ROLL_DEADBAND_PHASE_STEPS),
            "realistic_motion_fraction": wrist_roll_motion_fraction(realistic.rows, WRIST_ROLL_DEADBAND_PHASE_STEPS),
            "note": "REALISM_APPROXIMATION - realistic 쪽 비율이 baseline보다 낮게 나오는지만 관찰 (하드웨어 사실 확정 아님).",
        },
    }


def render_comparison_table(comparison: dict, control_hz: float) -> str:
    lines = [BAR, "[비교] joint별 frame_delta / velocity / tracking_error (p99)", BAR]
    lines.append(f"{'joint':14s} {'frame_delta(base)':>18s} {'frame_delta(real)':>18s} {'velocity(base)':>16s} {'velocity(real)':>16s}")
    for joint, entry in comparison["joints"].items():
        b, r = entry["baseline"], entry["realistic"]
        b_fd = b["frame_delta"]["p99"] if b["frame_delta"] else float("nan")
        r_fd = r["frame_delta"]["p99"] if r["frame_delta"] else float("nan")
        b_v = b["velocity_deg_s"]["p99"] if b["velocity_deg_s"] else float("nan")
        r_v = r["velocity_deg_s"]["p99"] if r["velocity_deg_s"] else float("nan")
        lines.append(f"{joint:14s} {b_fd:18.3f} {r_fd:18.3f} {b_v:16.3f} {r_v:16.3f}")

    jump = comparison["shoulder_pan_step_jump"]
    lines.append("")
    lines.append(f"[shoulder_pan 순간 점프] jump_step={jump['jump_step']} target={jump['target_deg']}deg")

    def _fmt_delay(steps):
        if steps is None:
            return "도달 못함"
        return f"{steps} step (~{steps / control_hz * 1000:.1f} ms)"

    lines.append(f"  baseline  command->motion delay: {_fmt_delay(jump['baseline_delay_steps'])}")
    lines.append(f"  realistic command->motion delay: {_fmt_delay(jump['realistic_delay_steps'])}")

    deadband = comparison["wrist_roll_deadband_phase"]
    lines.append("")
    lines.append(f"[wrist_roll deadband 구간] phase_end_step={deadband['phase_end_step']}")
    lines.append(f"  baseline  실제 움직인 step 비율:  {deadband['baseline_motion_fraction']:.1%}")
    lines.append(f"  realistic 실제 움직인 step 비율:  {deadband['realistic_motion_fraction']:.1%}")
    lines.append(BAR)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="hardware-free 합성 trajectory로 baseline vs realistic MuJoCo control-mode를 비교합니다.",
    )
    parser.add_argument("--num-steps", type=int, default=200, help="합성 trajectory 길이 (control step 수, 기본 200)")
    parser.add_argument("--control-hz", type=float, default=30.0, help="control step 주기 Hz (기본 30 - 렌더 fps와 무관, 벤치마크 전용)")
    parser.add_argument("--profile", type=Path, default=None, help="Control Profile Candidate JSON 경로 (기본: configs/generated/so101_control_profile_candidate_v1.json)")
    parser.add_argument("--scene", type=Path, default=None, help="MuJoCo scene XML 경로 (기본: simulation/mujoco/assets/scene.xml)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="비교 결과 JSON/CSV 저장 디렉터리")
    parser.add_argument("--disable-latency", action="store_true")
    parser.add_argument("--disable-deadband", action="store_true")
    parser.add_argument("--disable-rate-limit", action="store_true")
    parser.add_argument("--disable-historical-range-diagnostic", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        profile = load_control_profile(args.profile or DEFAULT_PROFILE_PATH)
    except ControlProfileError as exc:
        print(f"[오류] control profile 로딩 실패: {exc}")
        return 1

    model = load_model(args.scene)
    mapping = build_default_mapping([f"{name}.pos" for name in SO101_JOINT_NAMES])
    validate_mapping_against_model(mapping, model)

    dt_control = 1.0 / args.control_hz
    steps_per_frame = max(1, round(dt_control / model.opt.timestep))

    trajectory = build_synthetic_trajectory(args.num_steps, wrist_roll_dither_deg=profile.wrist_roll_deadband.no_response_upper_deg * 0.5)

    baseline_layer = RealisticControlLayer(profile, config_all_disabled())
    realistic_config = RealisticControlConfig(
        enable_latency=not args.disable_latency,
        enable_deadband=not args.disable_deadband,
        enable_rate_limit=not args.disable_rate_limit,
        enable_historical_range_diagnostic=not args.disable_historical_range_diagnostic,
    )
    realistic_layer = RealisticControlLayer(profile, realistic_config)
    realistic_recorder = RealisticControlRecorder()

    print(BAR)
    print("[준비] baseline vs realistic MuJoCo control-mode 비교 (hardware-free 합성 trajectory)")
    print(BAR)
    print(f"[profile] {profile.path} (source={profile.source}, run_count={profile.run_count}, status={profile.status})")
    print(f"[trajectory] num_steps={args.num_steps} control_hz={args.control_hz}")

    baseline_result = run_pass(
        label="baseline", trajectory=trajectory, layer=baseline_layer, model=model, mapping=mapping,
        dt_control=dt_control, steps_per_frame=steps_per_frame, recorder=None,
    )
    realistic_result = run_pass(
        label="realistic", trajectory=trajectory, layer=realistic_layer, model=model, mapping=mapping,
        dt_control=dt_control, steps_per_frame=steps_per_frame, recorder=realistic_recorder,
    )

    comparison = build_comparison(baseline_result, realistic_result)
    print(render_comparison_table(comparison, args.control_hz))

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"comparison_{session_id}.json"
    comparison_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "profile_path": str(profile.path),
                "profile_source": profile.source,
                "profile_run_count": profile.run_count,
                "num_steps": args.num_steps,
                "control_hz": args.control_hz,
                "realistic_config": {
                    "enable_latency": realistic_config.enable_latency,
                    "enable_deadband": realistic_config.enable_deadband,
                    "enable_rate_limit": realistic_config.enable_rate_limit,
                    "enable_historical_range_diagnostic": realistic_config.enable_historical_range_diagnostic,
                },
                "comparison": comparison,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    realistic_recorder.write_json(output_dir / f"realistic_diagnostics_{session_id}.json")
    realistic_recorder.write_csv(output_dir / f"realistic_diagnostics_{session_id}.csv")

    print(f"[저장] 비교 결과: {comparison_path}")
    print(f"[저장] realistic 진단(JSON/CSV): {output_dir / f'realistic_diagnostics_{session_id}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
