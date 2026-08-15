#!/usr/bin/env python3
"""Design read-only canonicalized training indices for V1 static pre-roll.

The source LeRobot dataset is never modified.  This script emits candidate index
manifests under reports/ which can later be passed to ``torch.utils.data.Subset``.
Every sample at or after the empirically detected arm-motion onset is retained.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "so101_blue_cube_place_return_v1"
DEFAULT_ANALYSIS = ROOT / "reports" / "v1_initial_target_semantics_analysis" / "analysis.json"
DEFAULT_OUTPUT = ROOT / "reports" / "v1_canonicalized_training_view"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CHUNK_SIZE = 50


def load_episodes(dataset_root: Path) -> list[dict[str, np.ndarray | int]]:
    episodes = []
    for episode in range(41):
        path = dataset_root / "data" / "chunk-000" / f"file-{episode:03d}.parquet"
        frame = pd.read_parquet(path).sort_values("frame_index")
        episodes.append(
            {
                "state": np.stack(frame["observation.state"]).astype(np.float64),
                "action": np.stack(frame["action"]).astype(np.float64),
                "index": frame["index"].to_numpy(np.int64),
                "length": len(frame),
            }
        )
    return episodes


def sustained_onset(action: np.ndarray, thresholds: np.ndarray) -> int:
    normalized = np.sqrt(np.sum(((action[:, :5] - action[0, :5]) / thresholds[:5]) ** 2, axis=1))
    above = normalized > 1.0
    hits = np.flatnonzero(above[:-2] & above[1:-1] & above[2:])
    if not len(hits):
        raise RuntimeError("episode has no sustained arm-motion onset")
    return int(hits[0])


def anchors(onset: int, candidate: str) -> list[int]:
    last = onset - 1
    if candidate == "A":
        values = [0]
    elif candidate == "B":
        values = [0, last]
    elif candidate == "C":
        values = [0, last // 2, last]
    else:
        raise ValueError(candidate)
    return sorted(set(values))


def decode_prefix(dataset_root: Path, camera: str, episode: int, length: int) -> np.ndarray:
    path = dataset_root / "videos" / f"observation.images.{camera}" / "chunk-000" / f"file-{episode:03d}.mp4"
    container = av.open(str(path))
    values = []
    for index, frame in enumerate(container.decode(video=0)):
        if index >= length:
            break
        image = frame.to_ndarray(format="rgb24")
        values.append(cv2.resize(image, (32, 24), interpolation=cv2.INTER_AREA).reshape(-1) / 255.0)
    container.close()
    return np.stack(values)


def target_chunk(action: np.ndarray, frame: int) -> np.ndarray:
    return np.stack([action[min(frame + k, len(action) - 1)] for k in range(CHUNK_SIZE)])


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p5": None, "median": None, "p95": None}
    return {key: float(value) for key, value in zip(("p5", "median", "p95"), np.percentile(values, [5, 50, 95]))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = json.loads(args.source_analysis.read_text())
    thresholds = np.array([source["noise"][joint]["threshold"] for joint in JOINTS], dtype=np.float64)
    conflict_cuts = source["conflicts"]["cuts"]
    episodes = load_episodes(args.dataset_root)
    onsets = [sustained_onset(ep["action"], thresholds) for ep in episodes]

    workspace = [decode_prefix(args.dataset_root, "workspace", ep, onset) for ep, onset in enumerate(onsets)]
    wrist = [decode_prefix(args.dataset_root, "wrist", ep, onset) for ep, onset in enumerate(onsets)]
    workspace_scale = np.maximum(np.std(np.stack([value[0] for value in workspace]), axis=0), 0.03)
    wrist_scale = np.maximum(np.std(np.stack([value[0] for value in wrist]), axis=0), 0.03)

    total_original = sum(int(ep["length"]) for ep in episodes)
    results = {}
    manifests = {}
    for candidate in ("A", "B", "C"):
        selected_relative: dict[str, list[int]] = {}
        selected_absolute = []
        static_indices: dict[int, list[int]] = {}
        close_pairs = []
        chunk_progression = []
        pan_progression = []
        gripper_progression = []

        for episode, (data, onset) in enumerate(zip(episodes, onsets)):
            static = anchors(onset, candidate)
            keep = static + list(range(onset, int(data["length"])))
            static_indices[episode] = static
            selected_relative[str(episode)] = keep
            selected_absolute.extend(int(data["index"][frame]) for frame in keep)

            chunks = {frame: target_chunk(data["action"], frame) for frame in static}
            for left, right in combinations(static, 2):
                state_distance = float(
                    np.sqrt(np.mean(((data["state"][left] - data["state"][right]) / thresholds) ** 2))
                )
                workspace_distance = float(
                    np.sqrt(np.mean(((workspace[episode][left] - workspace[episode][right]) / workspace_scale) ** 2))
                )
                wrist_distance = float(
                    np.sqrt(np.mean(((wrist[episode][left] - wrist[episode][right]) / wrist_scale) ** 2))
                )
                if (
                    state_distance <= conflict_cuts["s_p10"]
                    and workspace_distance <= conflict_cuts["w_p10"]
                    and wrist_distance <= conflict_cuts["v_p10"]
                ):
                    mae = float(np.mean(np.abs(chunks[left] - chunks[right])))
                    close_pairs.append(
                        {
                            "episode": episode,
                            "frames": [left, right],
                            "state_distance": state_distance,
                            "workspace_distance": workspace_distance,
                            "wrist_distance": wrist_distance,
                            "chunk_mae": mae,
                            "high_conflict": mae >= conflict_cuts["chunk_p90"],
                        }
                    )

            for frame in keep:
                chunk = target_chunk(data["action"], frame)
                delta = chunk[40] - chunk[0]
                chunk_progression.append(float(abs(delta[1]) + abs(delta[2])))
                pan_progression.append(float(abs(delta[0])))
                gripper_progression.append(float(delta[5]))

        high = [pair for pair in close_pairs if pair["high_conflict"]]
        static_count = sum(len(value) for value in static_indices.values())
        selected_count = len(selected_absolute)
        results[candidate] = {
            "anchors": {"A": "initial", "B": "initial+pre_onset", "C": "initial+midpoint+pre_onset"}[candidate],
            "total_training_samples": selected_count,
            "removed_static_samples": total_original - selected_count,
            "retained_static_samples": static_count,
            "retained_motion_samples": selected_count - static_count,
            "static_fraction": static_count / selected_count,
            "motion_fraction": (selected_count - static_count) / selected_count,
            "near_identical_conflict_pairs": len(close_pairs),
            "high_conflict_pairs": len(high),
            "high_conflict_fraction": len(high) / len(close_pairs) if close_pairs else 0.0,
            "high_conflict_pairs_per_training_sample": len(high) / selected_count,
            "action_chunk_distribution": {
                "lift_elbow_l1_step0_to_40": quantiles(chunk_progression),
                "abs_pan_step0_to_40": quantiles(pan_progression),
                "gripper_step0_to_40": quantiles(gripper_progression),
            },
            "close_pair_details": close_pairs,
        }
        manifests[candidate] = {
            "candidate": candidate,
            "dataset_root": str(args.dataset_root.resolve()),
            "source_dataset_modified": False,
            "onset_detector": {
                "thresholds": dict(zip(JOINTS, thresholds.tolist())),
                "sustain_frames": 3,
                "episode_onsets": onsets,
            },
            "selected_relative_indices_by_episode": selected_relative,
            "selected_absolute_indices": selected_absolute,
        }

    baseline = source["within_episode_static_conflicts"]
    report = {
        "schema": "v1-canonicalized-training-view-design-v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "source_dataset_modified": False,
        "baseline": baseline,
        "candidates": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2))
    for candidate, manifest in manifests.items():
        (args.output_dir / f"candidate_{candidate.lower()}_indices.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"baseline": baseline, "candidates": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
