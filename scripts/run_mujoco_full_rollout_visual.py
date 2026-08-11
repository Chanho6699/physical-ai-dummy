#!/usr/bin/env python3
"""Visual/interactive rollout viewer - headless benchmark(``run_mujoco_full_rollout_benchmark.py``)와
분리된 사람 관찰용 실행기.

두 모드:
    --mode web    (기본, 권장) offscreen ``mujoco.Renderer`` + 표준 라이브러리 ``http.server``로
                  MJPEG 스트리밍. ``simulation/mujoco/live_web_viewer.py``와 같은 검증된 패턴
                  (GLFW 창을 띄우지 않음 - 이 WSLg 환경에서 네이티브 뷰어 종료 시 문서화된
                  segfault 위험(``docs/mujoco_action_replay.md`` §11.8)이 없다).
    --mode native  ``mujoco.viewer.launch_passive`` 네이티브 GLFW 창. 위 segfault 위험이 있다
                  (데이터는 창을 닫기 전에 이미 저장되어 있으므로 크래시해도 결과 손실은 없다).

두 실행 방식:
    (기본) 실제 candidate checkpoint로 라이브 추론하며 관찰.
    --replay PATH  이전에 저장된 trajectory JSON(``run_mujoco_full_rollout_benchmark.py``가
                  ``trajectories/*.json``에 남긴 것과 동일 스키마)을 다시 재생한다 - 추론 없이
                  저장된 qpos_deg/cube_quat를 그대로 적용하는 kinematic scrub이라 physics를
                  다시 돌리지 않고도 원래 rollout과 정확히 같은 장면을 보여준다.

콘솔에 scene id, seed, 현재 stage, safety reject 여부를 계속 표시한다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402

from runtime.common.vla_contract import JOINT_ORDER  # noqa: E402
from runtime.laptop.safety_gate import SafetyGate, SafetyGateConfig  # noqa: E402
from scripts.run_mujoco_full_rollout_benchmark import CANDIDATES, PRIMARY_DATASET_ROOT, _json_default  # noqa: E402
from simulation.mujoco.pick_drop_eval import ReferenceZones, StepRecord, current_stage_label  # noqa: E402
from simulation.mujoco.primary_replay_rollout import run_primary_replay  # noqa: E402
from simulation.mujoco.rollout_env import DEFAULT_MAX_STEPS, run_synthetic_closed_loop  # noqa: E402
from simulation.mujoco.smolvla_chunk_runner import SmolVLAChunkRunner  # noqa: E402
from simulation.mujoco.so101_model import load_model  # noqa: E402

SCENE_PATH = PROJECT_ROOT / "simulation" / "mujoco" / "assets" / "scene_pick_drop.xml"
SCENES_CONFIG_PATH = PROJECT_ROOT / "configs" / "mujoco_rollout_scenes_v1.json"
DISPLAY_CAMERA = "workspace_cam"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "mujoco_full_rollout_candidate_comparison_v1" / "trajectories" / "manual"


# ---------------------------------------------------------------------------
# 공유 상태 (rollout 스레드가 쓰고, HTTP 스레드가 읽는다)
# ---------------------------------------------------------------------------


class _FrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None

    def publish(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg_bytes

    def get(self) -> bytes | None:
        with self._lock:
            return self._jpeg


class _StatusBoard:
    def __init__(self, *, candidate: str, track: str, scene_id: str, seed: int) -> None:
        self._lock = threading.Lock()
        self._base = {"candidate": candidate, "track": track, "scene_id": scene_id, "seed": seed}
        self._live = {"step": 0, "stage": "start", "safety_decision": "-", "done": False}

    def update(self, **kwargs) -> None:
        with self._lock:
            self._live.update(kwargs)

    def to_dict(self) -> dict:
        with self._lock:
            return {**self._base, **self._live}


def _encode_jpeg(frame, *, quality: int = 80) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>MuJoCo rollout viewer</title>
<style>body{{background:#111;color:#eee;font-family:sans-serif;padding:16px}}
img{{border:1px solid #444}} pre{{background:#1b1b1b;padding:8px}}</style></head>
<body>
<h2>MuJoCo full-rollout visual viewer</h2>
<img id="frame" src="/stream.mjpg" width="640" height="480">
<pre id="status">loading...</pre>
<script>
setInterval(() => fetch('/status').then(r => r.json()).then(d => {{
  document.getElementById('status').textContent = JSON.stringify(d, null, 2);
}}), 300);
</script>
</body></html>
"""


def _make_handler(frames: _FrameBuffer, status: _StatusBoard):
    boundary = "so101frame"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003 - 조용히 (기본 stderr 로그 억제)
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path == "/index.html":
                body = _INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/status":
                body = json.dumps(status.to_dict()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.end_headers()
                try:
                    while True:
                        jpeg = frames.get()
                        if jpeg is not None:
                            self.wfile.write(f"--{boundary}\r\n".encode())
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        time.sleep(1.0 / 15.0)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


# ---------------------------------------------------------------------------
# console/stage 표시 공통 로직
# ---------------------------------------------------------------------------


class LiveConsoleAndDisplay:
    """``on_step(rec, data)`` 콜백 - 콘솔에 진행상황을 찍고, web/native 표시를 갱신하고,
    사람이 볼 수 있게 페이싱한다 (계산 자체는 빠르므로 인위적 pacing 없으면 순식간에 끝남)."""

    def __init__(
        self, *, candidate: str, track: str, scene_id: str, seed: int, mode: str, pace_hz: float,
        renderer: "mujoco.Renderer | None" = None, frames: _FrameBuffer | None = None,
        status: _StatusBoard | None = None, native_viewer=None, print_every: int = 1,
    ) -> None:
        self.candidate, self.track, self.scene_id, self.seed = candidate, track, scene_id, seed
        self.mode, self.pace_hz = mode, pace_hz
        self.renderer, self.frames, self.status = renderer, frames, status
        self.native_viewer = native_viewer
        self.print_every = print_every
        self._records: list[StepRecord] = []
        self._last_time = time.monotonic()

    def __call__(self, rec: StepRecord, data) -> None:
        self._records.append(rec)
        stage = current_stage_label(self._records)

        if self.mode == "web" and self.renderer is not None and self.frames is not None:
            self.renderer.update_scene(data, camera=DISPLAY_CAMERA)
            frame = self.renderer.render()
            self.frames.publish(_encode_jpeg(frame))
        elif self.mode == "native" and self.native_viewer is not None:
            self.native_viewer.sync()

        if self.status is not None:
            self.status.update(step=rec.step, stage=stage, safety_decision=rec.safety_decision, done=False)

        if rec.step % self.print_every == 0:
            print(f"[{self.candidate}/{self.track}] scene={self.scene_id} seed={self.seed} "
                  f"step={rec.step} stage={stage} safety={rec.safety_decision}"
                  + ("  <<< SAFETY REJECT" if rec.safety_decision == "REJECT" else ""))

        elapsed = time.monotonic() - self._last_time
        target = 1.0 / self.pace_hz
        if elapsed < target:
            time.sleep(target - elapsed)
        self._last_time = time.monotonic()


def run_live(args) -> None:
    scenes_config = json.loads(SCENES_CONFIG_PATH.read_text(encoding="utf-8"))
    scene = next(s for s in scenes_config["scenes"] if s["scene_id"] == args.scene)
    zones = ReferenceZones(bin_center_xy=tuple(scene["bin_center_xy"]), bin_inner_half=scenes_config["bin_inner_half"])

    model = load_model(SCENE_PATH)
    safety_gate = SafetyGate(SafetyGateConfig.from_repo_defaults())
    ckpt = CANDIDATES[args.candidate]["checkpoint"]
    print(f"체크포인트 로딩 중: {ckpt}")
    runner = SmolVLAChunkRunner(str(ckpt))
    print(f"로딩 완료. device={runner.device}")

    renderer = None
    frames = None
    status = None
    server = None
    server_thread = None
    native_viewer = None

    if args.mode == "web":
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = _FrameBuffer()
        status = _StatusBoard(candidate=args.candidate, track=args.track, scene_id=args.scene, seed=args.seed)
        handler_cls = _make_handler(frames, status)
        server = ThreadingHTTPServer((args.host, args.port), handler_cls)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"\n=== 웹 뷰어: http://{args.host}:{args.port}/ (브라우저로 여세요) ===\n")

    on_step = LiveConsoleAndDisplay(
        candidate=args.candidate, track=args.track, scene_id=args.scene, seed=args.seed,
        mode=args.mode, pace_hz=args.pace_hz, renderer=renderer, frames=frames, status=status,
        print_every=args.print_every,
    )

    # native 모드는 첫 step에서 실제 mujoco.MjData를 받아야 뷰어를 열 수 있다 (rollout 함수가
    # 내부에서 MjData를 만들기 때문) - on_step 콜백 안에서 지연 생성한다.
    if args.mode == "native":
        _native_holder = {"viewer": None}
        base_call = on_step.__call__

        def _native_on_step(rec, data):
            if _native_holder["viewer"] is None:
                from mujoco import viewer as _mj_viewer

                _native_holder["viewer"] = _mj_viewer.launch_passive(model, data)
                on_step.native_viewer = _native_holder["viewer"]
                print("[네이티브 뷰어] 창을 열었습니다 - 창을 닫거나 Ctrl+C로 종료하세요.")
            base_call(rec, data)

        on_step_fn = _native_on_step
    else:
        on_step_fn = on_step

    t0 = time.time()
    try:
        if args.track == "primary":
            episode_index = int(args.scene.replace("mujoco_rollout_test", "")) - 1
            result = run_primary_replay(
                chunk_runner=runner, model=model, safety_gate=safety_gate, scene_id=args.scene,
                dataset_root=PRIMARY_DATASET_ROOT, episode_index=episode_index, zones=zones,
                cube_xy=tuple(scene["cube_xy"]), cube_z_init=scene["cube_z_init"], seed=args.seed,
                chunk_size=int(runner.policy.config.chunk_size), on_step=on_step_fn,
            )
        else:
            result = run_synthetic_closed_loop(
                chunk_runner=runner, model=model, safety_gate=safety_gate, scene_id=args.scene,
                initial_pose_deg=scene["initial_pose_deg"], cube_xy=tuple(scene["cube_xy"]),
                cube_z_init=scene["cube_z_init"], zones=zones, seed=args.seed, max_steps=args.max_steps,
                on_step=on_step_fn,
            )
    finally:
        runner.close()
        if renderer is not None:
            renderer.close()

    wall = time.time() - t0
    if status is not None:
        status.update(done=True)
    print(f"\n=== 완료: {result.ended_reason}, steps={len(result.step_records)}, {wall:.1f}s ===")
    print(f"kinematic_pick_drop_success={result.eval_result.kinematic.kinematic_pick_drop_success} "
          f"physics_pick_drop_success={result.eval_result.physics.physics_pick_drop_success} "
          f"failure={result.eval_result.failure.reason}({result.eval_result.failure.likely_cause})")
    print(f"real_follower_write_count={result.real_follower_write_count}")

    out_path = args.save or (DEFAULT_OUT_DIR / f"{args.candidate}_{args.track}_{args.scene}_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "candidate": args.candidate, "track": args.track, "scene_id": args.scene, "seed": args.seed,
        "checkpoint": str(ckpt),
        "step_records": [asdict(r) for r in result.step_records],
        "raw_command_log": result.raw_command_log, "safe_command_log": result.safe_command_log,
        "result": result.to_dict(),
    }, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"trajectory 저장: {out_path}")
    print(f"이 trajectory를 다시 보려면: python {Path(__file__).name} --replay {out_path} --mode {args.mode}")

    if args.mode == "web" and not args.no_hold:
        print("\n웹 뷰어를 계속 켜 둡니다 (마지막 프레임 표시). Ctrl+C로 종료하세요.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    if server is not None:
        server.shutdown()


def run_replay(args) -> None:
    data_json = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    records_raw = data_json["step_records"]
    candidate, track = data_json["candidate"], data_json["track"]
    scene_id, seed = data_json["scene_id"], data_json["seed"]

    if not records_raw or records_raw[0].get("qpos_deg") is None:
        raise SystemExit(
            "이 trajectory JSON에는 qpos_deg가 없어 replay할 수 없습니다 (오래된 형식이거나 "
            "REJECT로 즉시 끝난 rollout). run_mujoco_full_rollout_benchmark.py를 다시 실행해 "
            "새 형식으로 저장하세요."
        )

    model = load_model(SCENE_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    jnt_id = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINT_ORDER}
    qpos_adr = {j: model.jnt_qposadr[jnt_id[j]] for j in JOINT_ORDER}
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qpos_adr = model.jnt_qposadr[cube_jnt_id]

    import math

    renderer = None
    frames = None
    status = None
    server = None
    native_viewer = None

    if args.mode == "web":
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = _FrameBuffer()
        status = _StatusBoard(candidate=candidate, track=track, scene_id=scene_id, seed=seed)
        handler_cls = _make_handler(frames, status)
        server = ThreadingHTTPServer((args.host, args.port), handler_cls)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"\n=== 웹 뷰어(replay): http://{args.host}:{args.port}/ ===\n")

    print(f"replay: {args.replay} ({candidate}/{track} {scene_id} seed={seed}, {len(records_raw)} steps)")
    prefix_records: list[StepRecord] = []
    for raw in records_raw:
        for j in JOINT_ORDER:
            data.qpos[qpos_adr[j]] = math.radians(raw["qpos_deg"][j])
        if raw.get("cube_quat"):
            data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = raw["cube_pos"]
            data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = raw["cube_quat"]
        mujoco.mj_forward(model, data)

        prefix_records.append(StepRecord(**{**raw, "ee_pos": tuple(raw["ee_pos"]), "cube_pos": tuple(raw["cube_pos"])}))
        stage = current_stage_label(prefix_records)

        if args.mode == "web":
            renderer.update_scene(data, camera=DISPLAY_CAMERA)
            frames.publish(_encode_jpeg(renderer.render()))
            status.update(step=raw["step"], stage=stage, safety_decision=raw["safety_decision"], done=False)
        elif args.mode == "native":
            if native_viewer is None:
                from mujoco import viewer as _mj_viewer

                native_viewer = _mj_viewer.launch_passive(model, data)
                print("[네이티브 뷰어] 창을 열었습니다.")
            native_viewer.sync()

        if raw["step"] % args.print_every == 0:
            print(f"[replay] step={raw['step']} stage={stage} safety={raw['safety_decision']}"
                  + ("  <<< SAFETY REJECT" if raw["safety_decision"] == "REJECT" else ""))
        time.sleep(1.0 / args.pace_hz)

    print("\n=== replay 완료 ===")
    if status is not None:
        status.update(done=True)
    if args.mode == "web" and not args.no_hold:
        print("웹 뷰어를 계속 켜 둡니다. Ctrl+C로 종료하세요.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    if server is not None:
        server.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", choices=["A", "B"])
    ap.add_argument("--track", choices=["primary", "secondary"])
    ap.add_argument("--scene", default="mujoco_rollout_test01")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["web", "native"], default="web")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--pace-hz", type=float, default=15.0, help="사람이 보기 좋은 속도로 인위적으로 페이싱")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--no-hold", action="store_true", help="rollout 끝나면 바로 종료 (기본은 웹서버 유지)")
    ap.add_argument("--replay", type=Path, default=None, help="저장된 trajectory JSON을 재생 (라이브 추론 없음)")
    args = ap.parse_args()

    if args.replay is not None:
        run_replay(args)
        return

    if args.candidate is None or args.track is None:
        raise SystemExit("--replay를 안 쓸 때는 --candidate와 --track이 필수입니다.")
    run_live(args)


if __name__ == "__main__":
    main()
