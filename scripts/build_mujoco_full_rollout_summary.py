#!/usr/bin/env python3
"""``rollout_results.csv`` 등 headless benchmark 산출물을 읽어 사람이 읽는
``summary.md``(A vs B 비교 서사 + 요청 §10의 9개 질문 답변)를 만든다.

``run_mujoco_full_rollout_benchmark.py``가 이미 만드는 ``summary.json``/``candidate_comparison.csv``는
바꾸지 않는다 - 이 스크립트는 그 CSV들을 다시 읽어 더 풍부한(연속값 거리 지표 등) 분석과 서사를
추가로 만드는 별도 단계다. 여러 번 다시 실행해도 안전하다(READ-ONLY, 입력 CSV를 바꾸지 않음).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "reports" / "mujoco_full_rollout_candidate_comparison_v1"

CANDIDATE_LABEL = {
    "A": "Candidate A (V2+V3 reweight2:1 @10000, accuracy-oriented)",
    "B": "Candidate B (V3+V4 uniform @10000, safety-oriented)",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1")


def _float(v: str, default: float | None = None) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(_bool(r[key]) for r in rows) / len(rows)


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [_float(r[key]) for r in rows]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def group(rows: list[dict], *, candidate: str, track: str) -> list[dict]:
    return [r for r in rows if r["candidate"] == candidate and r["track"] == track]


def render_group_stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "kinematic_success_rate": _rate(rows, "kinematic_pick_drop_success"),
        "physics_success_rate": _rate(rows, "physics_pick_drop_success"),
        "approach_success_rate": _rate(rows, "approach_success"),
        "grasp_pose_reached_rate": _rate(rows, "grasp_pose_reached"),
        "gripper_close_rate": _rate(rows, "gripper_close_detected"),
        "lift_success_rate": _rate(rows, "lift_success"),
        "carry_direction_ok_rate": _rate(rows, "carry_direction_ok"),
        "bin_vicinity_reached_rate": _rate(rows, "bin_vicinity_reached"),
        "release_rate": _rate(rows, "gripper_release_detected"),
        "safety_reject_rate": _rate(rows, "ended_by_safety_reject"),
        "clamp_free_rate": _rate(rows, "clamp_free"),
        "mean_approach_min_dist_m": _mean(rows, "approach_min_dist_m"),
        "mean_ee_bin_min_dist_m": _mean(rows, "ee_bin_min_dist_m"),
        "mean_jerk_deg": _mean(rows, "mean_abs_jerk_deg"),
        "mean_max_step_delta_deg": _mean(rows, "max_single_step_delta_deg"),
        "mean_wall_time_s": _mean(rows, "wall_time_s"),
    }


def seed_sensitivity(rows: list[dict]) -> dict:
    """같은 scene, 다른 seed 사이에서 kinematic_pick_drop_success가 얼마나 갈리는지 -
    scene별 seed 간 표준편차(0/1 값의)로 대략적인 stochastic sensitivity를 잰다."""
    by_scene: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_scene[r["scene_id"]].append(_bool(r["kinematic_pick_drop_success"]))
    stds = [statistics.pstdev([1.0 if v else 0.0 for v in vs]) for vs in by_scene.values() if len(vs) > 1]
    return {"mean_within_scene_seed_std": statistics.fmean(stds) if stds else None, "n_scenes": len(by_scene)}


def failure_distribution(rows: list[dict]) -> Counter:
    return Counter(r["failure_reason"] for r in rows)


def format_pct(x: float | None) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x * 100:.0f}%"


def format_m(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}m"


def format_deg(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}deg"


def build_markdown(out_dir: Path) -> str:
    rows = _read_csv(out_dir / "rollout_results.csv")
    safety_rows = _read_csv(out_dir / "safety_metrics.csv")
    n_write = sum(int(r.get("real_follower_write_count", 0)) for r in safety_rows) if safety_rows else 0

    lines: list[str] = []
    lines.append("# MuJoCo full-rollout candidate comparison - summary")
    lines.append("")
    lines.append(f"- total rollouts: {len(rows)}")
    lines.append(f"- real_follower_write_count (전체 합): {n_write} (항상 0이어야 함)")
    lines.append("")
    lines.append(
        "**중요**: MuJoCo 환경은 실제 SO-101 환경과 동일하지 않다. 아래 physics 성공률(contact 기반)을 "
        "SmolVLA policy quality와 동일시하지 말 것 - 이 benchmark의 1차 목적은 policy가 만든 action "
        "trajectory가 운동학적/의미론적으로 올바른지 검증하는 것이다 (Track A/kinematic이 주 지표, "
        "Track B/physics는 부차 지표)."
    )
    lines.append("")

    lines.append("## Track별/candidate별 핵심 지표")
    lines.append("")
    lines.append(
        "| track | candidate | n | kinematic 성공 | physics 성공 | approach | grasp pose | lift | "
        "bin 근접 | safety reject | clamp-free | mean approach dist | mean EE-bin dist | mean jerk |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for track in ("primary", "secondary"):
        for candidate in ("A", "B"):
            g = group(rows, candidate=candidate, track=track)
            s = render_group_stats(g)
            if not s:
                continue
            lines.append(
                f"| {track} | {candidate} | {s['n']} | {format_pct(s['kinematic_success_rate'])} | "
                f"{format_pct(s['physics_success_rate'])} | {format_pct(s['approach_success_rate'])} | "
                f"{format_pct(s['grasp_pose_reached_rate'])} | {format_pct(s['lift_success_rate'])} | "
                f"{format_pct(s['bin_vicinity_reached_rate'])} | {format_pct(s['safety_reject_rate'])} | "
                f"{format_pct(s['clamp_free_rate'])} | {format_m(s['mean_approach_min_dist_m'])} | "
                f"{format_m(s['mean_ee_bin_min_dist_m'])} | {format_deg(s['mean_jerk_deg'])} |"
            )
    lines.append("")

    lines.append("## Primary track (real observation replay) - 주 비교")
    lines.append("")
    for candidate in ("A", "B"):
        g = group(rows, candidate=candidate, track="primary")
        if not g:
            continue
        s = render_group_stats(g)
        sens = seed_sensitivity(g)
        fd = failure_distribution(g)
        lines.append(f"### {CANDIDATE_LABEL[candidate]}")
        lines.append("")
        lines.append(f"- n={s['n']}, kinematic 성공률={format_pct(s['kinematic_success_rate'])}, "
                      f"safety reject율={format_pct(s['safety_reject_rate'])}, "
                      f"clamp-free율={format_pct(s['clamp_free_rate'])}")
        lines.append(f"- mean approach min-dist={format_m(s['mean_approach_min_dist_m'])} "
                      f"(참고 zone까지 - 실제 그 episode의 진짜 물체 위치가 아님, 아래 한계 참고)")
        lines.append(f"- mean trajectory jerk proxy={format_deg(s['mean_jerk_deg'])}, "
                      f"mean max single-step delta={format_deg(s['mean_max_step_delta_deg'])}")
        lines.append(f"- seed sensitivity(within-scene std of kinematic success)="
                      f"{sens['mean_within_scene_seed_std']:.3f}" if sens['mean_within_scene_seed_std'] is not None else "- seed sensitivity: n/a")
        lines.append(f"- failure reasons: {dict(fd.most_common())}")
        lines.append("")

    lines.append("## Secondary / exploratory track (synthetic closed-loop) - 참고용")
    lines.append("")
    lines.append(
        "**주의**: MuJoCo가 렌더링한 이미지는 SmolVLA가 학습한 실제 카메라 영상과 다르다 "
        "(visual domain gap, `docs/mujoco_scene_to_so101_semantics.md` §4). 아래 수치는 "
        "\"MuJoCo 렌더링이라는 낯선 입력에서 policy가 어떻게 반응하는가\"이지 실물 성능 예측이 아니다."
    )
    lines.append("")
    for candidate in ("A", "B"):
        g = group(rows, candidate=candidate, track="secondary")
        if not g:
            continue
        s = render_group_stats(g)
        fd = failure_distribution(g)
        lines.append(f"### {CANDIDATE_LABEL[candidate]}")
        lines.append("")
        lines.append(f"- n={s['n']}, kinematic 성공률={format_pct(s['kinematic_success_rate'])}, "
                      f"physics 성공률={format_pct(s['physics_success_rate'])}, "
                      f"safety reject율={format_pct(s['safety_reject_rate'])}")
        lines.append(f"- failure reasons: {dict(fd.most_common())}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()

    md = build_markdown(args.out_dir)
    path = args.out_dir / "summary_detailed.md"
    path.write_text(md, encoding="utf-8")
    print(f"작성됨: {path}")
    print(md[:2000])


if __name__ == "__main__":
    main()
