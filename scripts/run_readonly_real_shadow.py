#!/usr/bin/env python3
"""Read-only real-observation capture and deterministic VLA temporal replay.

This module intentionally has no follower writer, robot, MotionGuard, intent, or
Final Safety imports.  Hardware mode only opens the existing RGB camera source
and the audited ``ReadOnlyRealFollowerStateSource`` (Present_Position reads).
Saved observations can then be replayed against different Desktop checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.state_server.readonly_so101_reader import JOINT_ORDER
from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY as WORKSPACE_KEY
from runtime.common.vla_contract import CAMERA_WRIST_KEY as WRIST_KEY
from runtime.laptop.camera_source import RealCameraObservationSource
from runtime.laptop.follower_state_source import REAL_FOLLOWER_WRITE_COUNT, ReadOnlyRealFollowerStateSource
from runtime.laptop.temporal_ensemble import TemporalEnsembler
from runtime.laptop.trajectory_buffer import TrajectoryBuffer
from runtime.laptop.trajectory_chunk import TimestampedActionChunk
from runtime.laptop.vla_client import VLAClientConfig, VLAHttpClient

TASK = 'Pick up the blue cube, place it inside the blue rectangle labeled "BLUE", then return to the starting pose.'
DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113538-if00"
DEFAULT_CALIBRATION = "~/.cache/huggingface/lerobot/calibration/robots/so_follower/chanho_follower.json"
WRITER_CREATED = False
WRITER_CALL_COUNT = 0
PRODUCTION_PHASE_CONTINUITY = True
PRODUCTION_PHASE_FADE_CADENCE_SCALE = 0.5


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _health(server_url: str, timeout_s: float) -> dict:
    url = server_url.rstrip("/") + "/health"
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or data.get("status") != "ok" or not data.get("model_loaded"):
        raise RuntimeError(f"Desktop /health is not ready: {data!r}")
    return data


def capture(args: argparse.Namespace) -> int:
    """Capture one reusable real observation sequence. No network or writer."""
    if not 20.0 <= args.duration_s <= 30.0:
        raise ValueError("--duration-s must be within 20..30 seconds")
    if args.observation_hz <= 0:
        raise ValueError("--observation-hz must be positive")

    root = Path(args.output_dir).resolve()
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    camera = RealCameraObservationSource.from_hardware_config_path(args.hardware_config)
    follower = ReadOnlyRealFollowerStateSource.from_port(
        port=args.follower_port,
        follower_id=args.follower_id,
        calibration_path=str(Path(args.calibration).expanduser()),
    )
    rows: list[dict] = []
    period = 1.0 / args.observation_hz
    started_wall = time.time()
    started_mono = time.monotonic()
    try:
        camera.open()
        follower.connect()
        for _ in range(args.camera_warmup_frames):
            camera.capture_all()
        sequence = 0
        deadline = started_mono
        while True:
            now = time.monotonic()
            if now - started_mono >= args.duration_s:
                break
            if now < deadline:
                time.sleep(deadline - now)
            capture_mono = time.monotonic()
            frames = camera.capture_all()
            snapshot = follower.read()  # exactly one Present_Position read per observation
            workspace = np.asarray(frames[WORKSPACE_KEY].image_rgb)
            wrist = np.asarray(frames[WRIST_KEY].image_rgb)
            if workspace.shape != (480, 640, 3) or wrist.shape != (480, 640, 3):
                raise RuntimeError(f"unexpected image shapes: workspace={workspace.shape}, wrist={wrist.shape}")
            wp = images / f"{sequence:05d}_workspace.png"
            rp = images / f"{sequence:05d}_wrist.png"
            Image.fromarray(workspace).save(wp)
            Image.fromarray(wrist).save(rp)
            rows.append({
                "sequence": sequence,
                "timestamp": time.time(),
                "capture_offset_s": capture_mono - started_mono,
                "frame_timestamps": {
                    "workspace": frames[WORKSPACE_KEY].captured_at_wall,
                    "wrist": frames[WRIST_KEY].captured_at_wall,
                },
                "frame_hashes_rgb_sha256": {"workspace": _hash(workspace), "wrist": _hash(wrist)},
                "image_paths": {"workspace": str(wp), "wrist": str(rp)},
                "state": {joint: float(snapshot.positions_deg[joint]) for joint in JOINT_ORDER},
            })
            sequence += 1
            deadline = started_mono + sequence * period
    finally:
        follower.disconnect()  # audited disconnect(disable_torque=False)
        camera.close()

    _jsonl(root / "observations.jsonl", rows)
    manifest = {
        "schema": "readonly-real-shadow-observations-v1",
        "task": args.task,
        "started_at": started_wall,
        "duration_s": time.monotonic() - started_mono,
        "observation_hz": args.observation_hz,
        "observation_count": len(rows),
        "joint_names": list(JOINT_ORDER),
        "follower_port": args.follower_port,
        "follower_id": args.follower_id,
        "calibration": str(Path(args.calibration).expanduser()),
        "writer_created": WRITER_CREATED,
        "write_count": WRITER_CALL_COUNT + REAL_FOLLOWER_WRITE_COUNT,
        "observations_jsonl": str(root / "observations.jsonl"),
    }
    _json(root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def _load_observations(root: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / "observations.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError("observation sequence is empty")
    return manifest, rows


def _sample_newest(chunk: TimestampedActionChunk, target_time: float) -> dict[str, float] | None:
    if not TemporalEnsembler._covers(chunk, target_time):
        return None
    action, _, _ = TemporalEnsembler._sample_action_at(chunk, target_time)
    return action


def _delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {j: float(a[j] - b[j]) for j in JOINT_ORDER}


def _norm(delta: dict[str, float]) -> float:
    return math.sqrt(sum(delta[j] ** 2 for j in JOINT_ORDER))


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _variance(values: list[float]) -> float | None:
    return statistics.pvariance(values) if len(values) >= 2 else None


def _analyze(chunks: list[dict], ticks: list[dict], observations: list[dict]) -> dict:
    first_jumps, step10_jumps = [], []
    for prev, cur in zip(chunks, chunks[1:]):
        first_jumps.append(_norm(_delta(cur["raw_chunk"][0], prev["raw_chunk"][0])))
        step10_jumps.append(_norm(_delta(cur["raw_chunk"][10], prev["raw_chunk"][10])))

    raw_handoff, ensemble_handoff = [], []
    for prev, cur in zip(ticks, ticks[1:]):
        if cur["handoff_tick"] and prev.get("raw_interpolated_target") and cur.get("raw_interpolated_target"):
            raw_handoff.append(_norm(_delta(cur["raw_interpolated_target"], prev["raw_interpolated_target"])))
        if cur["handoff_tick"] and prev.get("temporal_ensemble_interpolated_target") and cur.get("temporal_ensemble_interpolated_target"):
            ensemble_handoff.append(_norm(_delta(cur["temporal_ensemble_interpolated_target"], prev["temporal_ensemble_interpolated_target"])))

    targets = [t for t in ticks if t.get("temporal_ensemble_interpolated_target")]
    duration = max((ticks[-1]["tick_offset_s"] - ticks[0]["tick_offset_s"]), 1e-9) if len(ticks) > 1 else 1.0
    reversals = {}
    vel_var, acc_var = {}, {}
    for joint in JOINT_ORDER:
        values = [t["temporal_ensemble_interpolated_target"][joint] for t in targets]
        velocity = [(b - a) * 60.0 for a, b in zip(values, values[1:])]
        acceleration = [(b - a) * 60.0 for a, b in zip(velocity, velocity[1:])]
        signs = [1 if v > 1e-6 else -1 if v < -1e-6 else 0 for v in velocity]
        nonzero = [s for s in signs if s]
        reversals[joint] = sum(a != b for a, b in zip(nonzero, nonzero[1:])) / duration
        vel_var[joint] = _variance(velocity)
        acc_var[joint] = _variance(acceleration)

    obs_delta, action_delta = [], []
    for prev, cur, pchunk, cchunk in zip(observations, observations[1:], chunks, chunks[1:]):
        obs_delta.append(_norm(_delta(cur["state"], prev["state"])))
        action_delta.append(_norm(_delta(cchunk["raw_chunk"][0], pchunk["raw_chunk"][0])))
    corr = float(np.corrcoef(obs_delta, action_delta)[0, 1]) if len(obs_delta) > 1 and np.std(obs_delta) > 0 and np.std(action_delta) > 0 else None
    raw_mean, ens_mean = _mean(raw_handoff), _mean(ensemble_handoff)
    return {
        "consecutive_chunk_first_action_jump_l2": {"mean": _mean(first_jumps), "max": max(first_jumps, default=None)},
        "consecutive_chunk_step10_jump_l2": {"mean": _mean(step10_jumps), "max": max(step10_jumps, default=None)},
        "handoff_target_jump_l2": {"raw_mean": raw_mean, "ensemble_mean": ens_mean,
            "reduction_fraction": (1.0 - ens_mean / raw_mean) if raw_mean and ens_mean is not None else None},
        "reversals_per_s": reversals,
        "target_velocity_variance": vel_var,
        "target_acceleration_variance": acc_var,
        "observation_state_delta_vs_first_action_delta_pearson": corr,
    }


def replay(args: argparse.Namespace) -> int:
    observation_root = Path(args.observation_dir).resolve()
    manifest, observations = _load_observations(observation_root)
    health = _health(args.server_url, args.timeout_s)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    session_id = f"readonly-shadow-{args.model_label}-{uuid.uuid4().hex[:10]}"
    replay_start = time.monotonic()
    observation_epoch = float(observations[0]["timestamp"])
    chunks: list[dict] = []
    with VLAHttpClient(VLAClientConfig(server_url=args.server_url, timeout_s=args.timeout_s)) as client:
        for observation in observations:
            images = {
                WORKSPACE_KEY: np.asarray(Image.open(observation["image_paths"]["workspace"]).convert("RGB")),
                WRIST_KEY: np.asarray(Image.open(observation["image_paths"]["wrist"]).convert("RGB")),
            }
            request_start = time.monotonic()
            result = client.predict_chunk(session_id=session_id, task=manifest["task"],
                sequence=int(observation["sequence"]), state=observation["state"], images=images)
            response_time = time.monotonic()
            if not result.ok or result.chunk is None or result.chunk_index_spacing_s is None:
                raise RuntimeError(f"predict_chunk failed at sequence {observation['sequence']}: {result.error_kind}: {result.error_message}")
            observation_virtual_s = float(observation["timestamp"]) - observation_epoch
            latency_s = 0.0 if args.publication_mode == "zero-compute" else float(result.inference_latency_ms) / 1000.0
            chunks.append({
                "sequence": result.sequence, "observation_offset_s": observation_virtual_s,
                "request_offset_s": observation_virtual_s, "response_offset_s": observation_virtual_s + latency_s,
                "compute_request_offset_s": request_start - replay_start,
                "compute_response_offset_s": response_time - replay_start,
                "raw_chunk": result.chunk, "chunk_size": result.chunk_size,
                "chunk_index_spacing_s": result.chunk_index_spacing_s, "model_id": result.model_id,
                "backend": result.backend, "inference_latency_ms": result.inference_latency_ms,
                "request_latency_ms": result.request_latency_ms,
            })

    buffer = TrajectoryBuffer(max_chunks=4)
    ensembler = TemporalEnsembler(half_life_s=args.ensemble_half_life_s, max_contributors=3, phase_continuity=PRODUCTION_PHASE_CONTINUITY, phase_fade_cadence_scale=PRODUCTION_PHASE_FADE_CADENCE_SCALE)
    tick_rows: list[dict] = []
    publish_index = 0
    previous_contributors: tuple[int, ...] = ()
    end_s = float(observations[-1]["timestamp"]) - observation_epoch
    for tick_index in range(int(math.ceil(end_s * args.control_hz)) + 1):
        tick_s = tick_index / args.control_hz
        while publish_index < len(chunks) and chunks[publish_index]["response_offset_s"] <= tick_s:
            item = chunks[publish_index]
            chunk = TimestampedActionChunk(
                sequence=item["sequence"], session_id=session_id,
                observation_time_monotonic=item["observation_offset_s"],
                request_started_time_monotonic=item["request_offset_s"],
                response_received_time_monotonic=item["response_offset_s"],
                server_received_at=None, server_responded_at=None,
                inference_latency_ms=item["inference_latency_ms"],
                chunk_index_spacing_s=item["chunk_index_spacing_s"], chunk_size=item["chunk_size"],
                actions=tuple(item["raw_chunk"]), model_id=item["model_id"], backend=item["backend"],
            )
            accepted = buffer.publish(chunk)
            if not accepted.accepted:
                raise RuntimeError(accepted.reason)
            publish_index += 1
        valid = buffer.valid_chunks(tick_s)
        target = ensembler.compute_target(valid, tick_s)
        lookahead = ensembler.compute_target(valid, tick_s + 1.0 / args.control_hz)
        newest = valid[-1] if valid else None
        contributors = target.contributing_sequences if target else ()
        handoff = bool(previous_contributors and contributors != previous_contributors)
        tick_rows.append({
            "tick_index": tick_index, "timestamp": time.time(), "tick_offset_s": tick_s,
            "sequence_id": newest.sequence if newest else None, "contributor_ids": list(contributors),
            "raw_interpolated_target": _sample_newest(newest, tick_s) if newest else None,
            "raw_ensemble_target": target.action if target else None,
            "target_lookahead": lookahead.action if lookahead else None,
            "temporal_ensemble_interpolated_target": target.action if target else None,
            "handoff_tick": handoff,
        })
        if contributors:
            previous_contributors = contributors

    _jsonl(output / "chunks.jsonl", chunks)
    _jsonl(output / "shadow_targets_60hz.jsonl", tick_rows)
    analysis = _analyze(chunks, tick_rows, observations)
    _json(output / "analysis.json", analysis)
    report = {
        "schema": "readonly-real-shadow-replay-v1", "model_label": args.model_label,
        "requested_deterministic_seed": args.seed, "health": health,
        "server_url": args.server_url, "session_id": session_id,
        "source_observation_manifest": str(observation_root / "manifest.json"),
        "source_observation_sha256": hashlib.sha256((observation_root / "observations.jsonl").read_bytes()).hexdigest(),
        "chunk_count": len(chunks), "target_tick_count": len(tick_rows),
        "writer_created": WRITER_CREATED, "write_count": WRITER_CALL_COUNT + REAL_FOLLOWER_WRITE_COUNT,
        "pipeline": ["raw_chunk", "TrajectoryBuffer", "TemporalEnsembler/interpolation", "60Hz target stream"],
        "outputs": {"chunks": str(output / "chunks.jsonl"), "targets": str(output / "shadow_targets_60hz.jsonl"), "analysis": str(output / "analysis.json")},
    }
    _json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--output-dir", default=str(ROOT / "reports/readonly_real_shadow/observations/current"))
    c.add_argument("--duration-s", type=float, default=25.0)
    c.add_argument("--observation-hz", type=float, default=3.0)
    c.add_argument("--camera-warmup-frames", type=int, default=8)
    c.add_argument("--hardware-config", default=str(ROOT / "configs/hardware.local.json"))
    c.add_argument("--follower-port", default=DEFAULT_PORT)
    c.add_argument("--follower-id", default="chanho_follower")
    c.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    c.add_argument("--task", default=TASK)
    c.set_defaults(func=capture)
    r = sub.add_parser("replay")
    r.add_argument("--observation-dir", default=str(ROOT / "reports/readonly_real_shadow/observations/current"))
    r.add_argument("--output-dir", required=True)
    r.add_argument("--server-url", default="http://100.75.147.72:9200")
    r.add_argument("--timeout-s", type=float, default=60.0)
    r.add_argument("--model-label", required=True)
    r.add_argument("--seed", type=int, default=20260815)
    r.add_argument("--control-hz", type=float, default=60.0)
    r.add_argument("--ensemble-half-life-s", type=float, default=0.338)
    r.add_argument("--realtime-pacing", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--publication-mode", choices=("latency-replay", "zero-compute"), default="latency-replay")
    r.set_defaults(func=replay)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
