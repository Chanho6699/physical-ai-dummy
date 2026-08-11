#!/usr/bin/env python3
"""Headless full-rollout benchmark driver - Primary(real-observation replay)/Secondary(synthetic
closed-loop) tracks, candidate A vs B.

``reports/mujoco_full_rollout_candidate_comparison_v1``의 headless 산출물(CSV/JSON)을 만든다.
Visual/interactive 실행은 ``scripts/run_mujoco_full_rollout_visual.py``가 별도로 담당한다
(요구사항: headless benchmark와 visual mode 분리).

실물 팔로워에는 어떤 write도 하지 않는다 - candidate당 checkpoint를 한 번만 로딩해 순차 실행하고
(동시 로딩 금지, GPU 메모리 재사용), 매 rollout이 끝나면 ``real_follower_write_count == 0``을
검증한다.

Candidate 체크포인트 경로는 CLI로 바꿀 수 없다 - 사용자가 명시적으로 지정한 두 후보만 다루는
비교 benchmark이므로 실수로 다른 checkpoint를 넣는 사고를 막기 위해 상수로 고정한다
(``docs/mujoco_scene_to_so101_semantics.md``, plan 문서 amendment 참고).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _json_default(o):
    """numpy scalar(np.float32/64, np.bool_ 등)를 JSON 직렬화 가능한 파이썬 기본형으로 변환한다.
    ``StepRecord.ee_pos``/``cube_pos``가 ``tuple(data.site_xpos[...])``에서 나온 numpy scalar를
    그대로 담고 있어서 필요하다."""
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig
from simulation.mujoco.pick_drop_eval import ReferenceZones
from simulation.mujoco.primary_replay_rollout import PrimaryReplayResult, run_primary_replay
from simulation.mujoco.rollout_env import DEFAULT_MAX_STEPS, SyntheticRolloutResult, run_synthetic_closed_loop
from simulation.mujoco.smolvla_chunk_runner import SmolVLAChunkRunner
from simulation.mujoco.so101_model import load_model

CANDIDATES = {
    "A": {
        "label": "V2+V3 reweight2:1 @10000 (accuracy-oriented)",
        "checkpoint": PROJECT_ROOT / "outputs" / "reweight_ablation" / "combined65_reweight_new2_old1_v1"
        / "checkpoints" / "010000" / "pretrained_model",
    },
    "B": {
        "label": "V3+V4 uniform @10000 (safety-oriented)",
        "checkpoint": PROJECT_ROOT / "outputs" / "pick_drop_v3_v4_combined69" / "smolvla_pick_drop_v3_v4_combined69_uniform_fresh"
        / "checkpoints" / "010000" / "pretrained_model",
    },
}

SCENE_PATH = PROJECT_ROOT / "simulation" / "mujoco" / "assets" / "scene_pick_drop.xml"
SCENES_CONFIG_PATH = PROJECT_ROOT / "configs" / "mujoco_rollout_scenes_v1.json"

# Primary track's real held-out episode source per scene (scene N <-> episode N-1 of this
# dataset - see scripts/generate_mujoco_pick_drop_scene.py / docs/mujoco_scene_to_so101_semantics.md
# for why this specific dataset was chosen: confirmed excluded from both candidates' training data).
PRIMARY_DATASET_ROOT = PROJECT_ROOT / "data" / "so101_cube_xy_midpoint_test10_v2_clean"

DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "mujoco_full_rollout_candidate_comparison_v1"
MIN_FREE_DISK_GB = 5.0


def disk_preflight(out_dir: Path) -> dict:
    total, used, free = shutil.disk_usage(out_dir.parent if out_dir.exists() else PROJECT_ROOT)
    free_gb = free / (1024**3)
    info = {"free_gb": round(free_gb, 1), "total_gb": round(total / (1024**3), 1), "ok": free_gb >= MIN_FREE_DISK_GB}
    print(f"[디스크 preflight] 여유 공간 {free_gb:.1f}GB (임계값 {MIN_FREE_DISK_GB}GB) - "
          f"{'충분함, 삭제 없이 진행' if info['ok'] else '부족 - 실행 중단'}")
    if not info["ok"]:
        raise SystemExit(f"디스크 여유 공간 부족: {free_gb:.1f}GB < {MIN_FREE_DISK_GB}GB")
    return info


def load_scenes(n: int) -> list[dict]:
    config = json.loads(SCENES_CONFIG_PATH.read_text(encoding="utf-8"))
    scenes = config["scenes"][:n]
    return scenes, config


def result_to_row(*, candidate: str, track: str, scene_id: str, seed: int, wall_time_s: float,
                   result, extra: dict) -> dict:
    ev = result.eval_result
    k, p, q, f = ev.kinematic, ev.physics, ev.trajectory_quality, ev.failure
    row = {
        "candidate": candidate, "track": track, "scene_id": scene_id, "seed": seed,
        "wall_time_s": round(wall_time_s, 2), "step_count": len(result.step_records),
        "ended_reason": result.ended_reason, "ended_by_safety_reject": result.ended_by_safety_reject,
        "real_follower_write_count": result.real_follower_write_count,
        "kinematic_pick_drop_success": k.kinematic_pick_drop_success,
        "physics_pick_drop_success": p.physics_pick_drop_success,
        "approach_success": k.approach_success, "approach_min_dist_m": round(k.approach_min_dist_m, 4),
        "grasp_pose_reached": k.grasp_pose_reached,
        "gripper_close_detected": k.gripper_close_detected,
        "gripper_close_timing_ok": k.gripper_close_timing_ok,
        "lift_success": k.lift_success, "lift_max_height_m": round(k.lift_max_height_m, 4),
        "carry_direction_ok": k.carry_direction_ok, "carry_bin_dist_trend": round(k.carry_bin_dist_trend, 4),
        "bin_vicinity_reached": k.bin_vicinity_reached, "ee_bin_min_dist_m": round(k.ee_bin_min_dist_m, 4),
        "gripper_release_detected": k.gripper_release_detected, "release_timing_ok": k.release_timing_ok,
        "grasp_contact_detected": p.grasp_contact_detected, "cube_secured": p.cube_secured,
        "physics_lifted": p.lifted, "physics_carried": p.carried, "physics_released": p.released,
        "physics_dropped_early": p.dropped_early, "physics_final_in_bin": p.final_in_bin,
        "safety_accept_count": q.safety_accept_count, "safety_would_clamp_count": q.safety_would_clamp_count,
        "safety_reject_count": q.safety_reject_count, "clamp_free": q.clamp_free,
        "mean_abs_jerk_deg": q.mean_abs_jerk_deg, "max_single_step_delta_deg": q.max_single_step_delta_deg,
        "failure_reason": f.reason, "failure_track": f.track, "failure_cause": f.likely_cause,
    }
    row.update(extra)
    return row


def run_candidate(
    *, candidate: str, tracks: list[str], scenes: list[dict], seeds: list[int], out_dir: Path,
    scenes_config: dict, max_steps: int, chunk_size_cap: int | None, save_trajectories: bool,
) -> list[dict]:
    ckpt = CANDIDATES[candidate]["checkpoint"]
    print(f"\n=== candidate {candidate} ({CANDIDATES[candidate]['label']}) ===")
    print(f"checkpoint: {ckpt}")
    t_load = time.time()
    runner = SmolVLAChunkRunner(str(ckpt))
    print(f"로딩 완료: {time.time() - t_load:.1f}s, device={runner.device}")

    model = load_model(SCENE_PATH)
    safety_gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    chunk_size = int(runner.policy.config.chunk_size)

    rows: list[dict] = []
    traj_dir = out_dir / "trajectories"
    if save_trajectories:
        traj_dir.mkdir(parents=True, exist_ok=True)

    try:
        for track in tracks:
            for scene in scenes:
                zones = ReferenceZones(bin_center_xy=tuple(scene["bin_center_xy"]), bin_inner_half=scenes_config["bin_inner_half"])
                for seed in seeds:
                    t0 = time.time()
                    if track == "primary":
                        episode_index = int(scene["scene_id"].replace("mujoco_rollout_test", "")) - 1
                        result: PrimaryReplayResult = run_primary_replay(
                            chunk_runner=runner, model=model, safety_gate=safety_gate, scene_id=scene["scene_id"],
                            dataset_root=PRIMARY_DATASET_ROOT, episode_index=episode_index, zones=zones,
                            cube_xy=tuple(scene["cube_xy"]), cube_z_init=scene["cube_z_init"], seed=seed,
                            chunk_size=chunk_size, max_chunks=chunk_size_cap,
                        )
                        extra = {"episode_index": episode_index, "dataset_root": str(PRIMARY_DATASET_ROOT)}
                    else:
                        result: SyntheticRolloutResult = run_synthetic_closed_loop(
                            chunk_runner=runner, model=model, safety_gate=safety_gate, scene_id=scene["scene_id"],
                            initial_pose_deg=scene["initial_pose_deg"], cube_xy=tuple(scene["cube_xy"]),
                            cube_z_init=scene["cube_z_init"], zones=zones, seed=seed, max_steps=max_steps,
                        )
                        extra = {}
                    wall = time.time() - t0
                    assert result.real_follower_write_count == 0, "REAL_FOLLOWER_WRITE invariant violated!"
                    row = result_to_row(candidate=candidate, track=track, scene_id=scene["scene_id"], seed=seed,
                                         wall_time_s=wall, result=result, extra=extra)
                    rows.append(row)
                    print(f"  [{track}] {scene['scene_id']} seed={seed}: "
                          f"kin={row['kinematic_pick_drop_success']} phys={row['physics_pick_drop_success']} "
                          f"failure={row['failure_reason']}({row['failure_cause']}) "
                          f"steps={row['step_count']} {wall:.1f}s")

                    if save_trajectories:
                        traj_path = traj_dir / f"{candidate}_{track}_{scene['scene_id']}_seed{seed}.json"
                        traj_path.write_text(json.dumps({
                            "candidate": candidate, "track": track, "scene_id": scene["scene_id"], "seed": seed,
                            "checkpoint": str(ckpt),
                            "step_records": [asdict(r) for r in result.step_records],
                            "raw_command_log": result.raw_command_log,
                            "safe_command_log": result.safe_command_log,
                            "result": result.to_dict(),
                        }, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    finally:
        runner.close()

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_per_scene_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["candidate"], r["track"], r["scene_id"])].append(r)
    out = []
    for (candidate, track, scene_id), rs in sorted(groups.items()):
        n = len(rs)
        out.append({
            "candidate": candidate, "track": track, "scene_id": scene_id, "n_seeds": n,
            "kinematic_success_rate": round(sum(r["kinematic_pick_drop_success"] for r in rs) / n, 3),
            "physics_success_rate": round(sum(r["physics_pick_drop_success"] for r in rs) / n, 3),
            "safety_reject_rate": round(sum(r["ended_by_safety_reject"] for r in rs) / n, 3),
            "clamp_free_rate": round(sum(r["clamp_free"] for r in rs) / n, 3),
            "mean_wall_time_s": round(sum(r["wall_time_s"] for r in rs) / n, 2),
        })
    return out


def build_failure_reasons(rows: list[dict]) -> list[dict]:
    counts: dict[tuple, Counter] = defaultdict(Counter)
    for r in rows:
        counts[(r["candidate"], r["track"])][r["failure_reason"]] += 1
    out = []
    for (candidate, track), counter in sorted(counts.items()):
        total = sum(counter.values())
        for reason, n in sorted(counter.items(), key=lambda kv: -kv[1]):
            out.append({"candidate": candidate, "track": track, "failure_reason": reason, "count": n,
                         "share": round(n / total, 3)})
    return out


def build_safety_metrics(rows: list[dict]) -> list[dict]:
    return [
        {k: r[k] for k in (
            "candidate", "track", "scene_id", "seed", "safety_accept_count", "safety_would_clamp_count",
            "safety_reject_count", "clamp_free", "max_single_step_delta_deg", "ended_by_safety_reject",
            "real_follower_write_count",
        )}
        for r in rows
    ]


def build_candidate_comparison(rows: list[dict]) -> list[dict]:
    """Primary track이 주 비교, secondary는 참고용 - 별도 track 값으로 명확히 구분해 둔다."""
    out = []
    for track in ("primary", "secondary"):
        for candidate in ("A", "B"):
            rs = [r for r in rows if r["candidate"] == candidate and r["track"] == track]
            if not rs:
                continue
            n = len(rs)
            out.append({
                "track": track, "candidate": candidate, "n_rollouts": n,
                "kinematic_pick_drop_success_rate": round(sum(r["kinematic_pick_drop_success"] for r in rs) / n, 3),
                "physics_pick_drop_success_rate": round(sum(r["physics_pick_drop_success"] for r in rs) / n, 3),
                "approach_success_rate": round(sum(r["approach_success"] for r in rs) / n, 3),
                "grasp_pose_reached_rate": round(sum(r["grasp_pose_reached"] for r in rs) / n, 3),
                "lift_success_rate": round(sum(r["lift_success"] for r in rs) / n, 3),
                "bin_vicinity_reached_rate": round(sum(r["bin_vicinity_reached"] for r in rs) / n, 3),
                "safety_reject_rate": round(sum(r["ended_by_safety_reject"] for r in rs) / n, 3),
                "clamp_free_rate": round(sum(r["clamp_free"] for r in rs) / n, 3),
                "mean_wall_time_s": round(sum(r["wall_time_s"] for r in rs) / n, 2),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default="A,B")
    ap.add_argument("--tracks", default="primary,secondary")
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seed list")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="secondary track cap")
    ap.add_argument("--max-chunks", type=int, default=None, help="primary track cap (None = full episode)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-trajectories", action="store_true")
    args = ap.parse_args()

    candidates = args.candidates.split(",")
    tracks = args.tracks.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    disk_preflight(args.out_dir)

    scenes, scenes_config = load_scenes(args.scenes)
    print(f"scenes: {[s['scene_id'] for s in scenes]}")
    print(f"seeds: {seeds}")
    print(f"tracks: {tracks}")

    t_start = time.time()
    all_rows: list[dict] = []
    for candidate in candidates:
        all_rows.extend(run_candidate(
            candidate=candidate, tracks=tracks, scenes=scenes, seeds=seeds, out_dir=args.out_dir,
            scenes_config=scenes_config, max_steps=args.max_steps, chunk_size_cap=args.max_chunks,
            save_trajectories=not args.no_trajectories,
        ))
    total_wall = time.time() - t_start

    write_csv(args.out_dir / "rollout_results.csv", all_rows)
    write_csv(args.out_dir / "per_scene_summary.csv", build_per_scene_summary(all_rows))
    write_csv(args.out_dir / "failure_reasons.csv", build_failure_reasons(all_rows))
    write_csv(args.out_dir / "safety_metrics.csv", build_safety_metrics(all_rows))
    write_csv(args.out_dir / "candidate_comparison.csv", build_candidate_comparison(all_rows))

    summary = {
        "candidates": {k: v["label"] for k, v in CANDIDATES.items() if k in candidates},
        "tracks": tracks, "n_scenes": len(scenes), "seeds": seeds,
        "n_rollouts": len(all_rows), "total_wall_time_s": round(total_wall, 1),
        "real_follower_write_count_total": sum(r["real_follower_write_count"] for r in all_rows),
        "candidate_comparison": build_candidate_comparison(all_rows),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(f"\n=== 완료: {len(all_rows)} rollouts, {total_wall:.1f}s, real_follower_write_count_total="
          f"{summary['real_follower_write_count_total']} ===")
    print(f"결과: {args.out_dir}")


if __name__ == "__main__":
    main()
