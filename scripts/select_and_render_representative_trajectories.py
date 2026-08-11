#!/usr/bin/env python3
"""``rollout_results.csv``에서 candidate/track별 "best success case"와 "representative failure
case"를 골라 ``videos/``에 MP4로 렌더링한다 (요청 §9: representative rollout 저장).

- best success: kinematic_pick_drop_success가 있으면 그중 approach_min_dist_m이 가장 작은 것
  (=가장 깔끔하게 성공한 것). 성공이 하나도 없으면 "가장 근접까지 간" rollout을 success 대신
  대표로 고르고 그 사실을 파일명/manifest에 명시한다.
- representative failure: 가장 흔한 failure_reason을 가진 rollout 중 하나(중앙값 approach_min_dist).

``run_mujoco_full_rollout_visual.py``와 같은 ``render_mujoco_rollout_video.py``를 그대로 재사용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "reports" / "mujoco_full_rollout_candidate_comparison_v1"


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1")


def _traj_path(traj_dir: Path, row: dict) -> Path:
    return traj_dir / f"{row['candidate']}_{row['track']}_{row['scene_id']}_seed{row['seed']}.json"


def pick_best_success(rows: list[dict]) -> dict | None:
    successes = [r for r in rows if _bool(r["kinematic_pick_drop_success"])]
    if not successes:
        return None
    return min(successes, key=lambda r: float(r["approach_min_dist_m"]))


def pick_closest_near_miss(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: float(r["approach_min_dist_m"]))


def pick_representative_failure(rows: list[dict]) -> dict | None:
    failures = [r for r in rows if not _bool(r["kinematic_pick_drop_success"])]
    if not failures:
        return None
    reason_counts = Counter(r["failure_reason"] for r in failures)
    top_reason, _ = reason_counts.most_common(1)[0]
    candidates = sorted(
        [r for r in failures if r["failure_reason"] == top_reason],
        key=lambda r: float(r["approach_min_dist_m"]),
    )
    return candidates[len(candidates) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()

    results_path = args.out_dir / "rollout_results.csv"
    if not results_path.is_file():
        raise SystemExit(f"{results_path}가 없습니다 - benchmark를 먼저 실행하세요.")
    rows = _read_csv(results_path)
    traj_dir = args.out_dir / "trajectories"
    video_dir = args.out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["candidate"], r["track"])].append(r)

    for (candidate, track), rs in sorted(groups.items()):
        best = pick_best_success(rs)
        kind = "best_success"
        if best is None:
            best = pick_closest_near_miss(rs)
            kind = "closest_near_miss (no full success in this group)"
        fail = pick_representative_failure(rs)

        # "success" 라벨은 실제 kinematic_pick_drop_success==True인 경우에만 쓴다 - 그런 rollout이
        # 하나도 없으면(이번 smoke 결과처럼) 파일명 자체를 "near_miss"로 남겨 "성공했다"고 오해하지
        # 않게 한다 (kind는 이미 "closest_near_miss (no full success in this group)"로 구분돼 있다).
        best_label = "success" if kind == "best_success" else "near_miss"
        for row, label in ((best, best_label), (fail, "failure")):
            if row is None:
                continue
            traj = _traj_path(traj_dir, row)
            if not traj.is_file():
                print(f"[건너뜀] trajectory 없음: {traj}")
                continue
            out_video = video_dir / f"{candidate}_{track}_{label}_{row['scene_id']}_seed{row['seed']}.mp4"
            print(f"[렌더링] {label} ({kind if label != 'failure' else row['failure_reason']}): {traj.name} -> {out_video.name}")
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "render_mujoco_rollout_video.py"),
                 str(traj), "--out", str(out_video)],
                check=True,
            )
            manifest.append({
                "candidate": candidate, "track": track, "label": label, "kind": kind if label != "failure" else row["failure_reason"],
                "scene_id": row["scene_id"], "seed": row["seed"], "trajectory": str(traj), "video": str(out_video),
            })

    (video_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n완료: {len(manifest)}개 영상, manifest: {video_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
