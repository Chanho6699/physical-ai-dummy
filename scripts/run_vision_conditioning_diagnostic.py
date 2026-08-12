#!/usr/bin/env python3
"""Motor-free scene A/B camera-to-VLA diagnostic; no writer or control loop."""

from __future__ import annotations
import argparse, base64, hashlib, json, sys, time, uuid
from pathlib import Path
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_collection.config import load_hardware_config
from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY as WK, CAMERA_WRIST_KEY as RK, JOINT_ORDER
from runtime.laptop.camera_source import RealCameraObservationSource
from runtime.laptop.vla_client import PREDICT_CHUNK_PATH, _encode_image_base64

VARIANTS = ("normal", "swap", "workspace_duplicate", "wrist_duplicate")

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()

def device_identity(path_value):
    path = Path(str(path_value))
    resolved = str(path.resolve(strict=False))
    name_path = Path("/sys/class/video4linux") / Path(resolved).name / "name"
    try:
        kernel_name = name_path.read_text().strip()
    except OSError:
        kernel_name = None
    return {"configured_path":str(path_value),"resolved_path":resolved,"kernel_name":kernel_name}

def encode(arr):
    text = _encode_image_base64(arr)
    return text, base64.b64decode(text, validate=True)

def state_arg(value):
    path = Path(value)
    raw = json.loads(path.read_text()) if path.is_file() else json.loads(value)
    missing = [j for j in JOINT_ORDER if j not in raw]
    if missing: raise ValueError(f"fixed state missing {missing}")
    return {j: float(raw[j]) for j in JOINT_ORDER}

def variant_images(w, r, variant):
    return {
        "normal": {WK:w, RK:r}, "swap": {WK:r, RK:w},
        "workspace_duplicate": {WK:w, RK:w}, "wrist_duplicate": {WK:r, RK:r},
    }[variant]

def infer(args, session, request_id, sequence, state, images):
    encoded, client = {}, {}
    for key, arr in images.items():
        text, jpeg = encode(arr); encoded[key] = text
        client[key] = {"jpeg_sha256":sha(jpeg), "raw_rgb_sha256":sha(arr.tobytes()), "shape":list(arr.shape)}
    headers = {"Authorization":f"Bearer {args.vla_api_token}"} if args.vla_api_token else {}
    body = {"session_id":session, "request_id":request_id, "sequence":sequence,
            "timestamp":time.time(), "task":args.task,
            "observation":{"state":state, "images":encoded}}
    response = requests.post(args.vla_server_url.rstrip("/")+PREDICT_CHUNK_PATH,
                             json=body, headers=headers, timeout=args.timeout_s)
    response.raise_for_status(); payload = response.json()
    server = payload.get("input_diagnostic")
    if not isinstance(server, dict): raise RuntimeError("server input_diagnostic missing")
    matches = {k:client[k]["jpeg_sha256"] == server["cameras"][k]["jpeg_sha256"] for k in (WK,RK)}
    if not all(matches.values()): raise RuntimeError(f"JPEG hash mismatch {matches}")
    chunk = payload.get("chunk")
    if not isinstance(chunk,list) or len(chunk)!=50: raise RuntimeError("server did not return 50-step chunk")
    return {"request_id":request_id, "sequence":sequence, "state":state, "task":args.task,
            "client_cameras":client, "server_input_diagnostic":server,
            "client_server_jpeg_hash_match":matches, "raw_action_chunk":chunk,
            "model_id":payload.get("model_id"), "backend":payload.get("backend")}

def capture(args):
    from runtime.laptop.follower_state_source import ReadOnlyRealFollowerStateSource

    if not 3 <= args.captures <= 5: raise ValueError("--captures must be 3..5")
    fixed = state_arg(args.fixed_state); hw = load_hardware_config(args.hardware_config)
    root = Path(args.output_dir)/args.scene_label; images_dir=root/"images"; images_dir.mkdir(parents=True,exist_ok=True)
    session=f"vision-{args.scene_label}-{uuid.uuid4().hex[:10]}"
    report={"mode":"MOTOR_WRITE_DISABLED","scene_label":args.scene_label,"session_id":session,
            "camera_devices":{k:device_identity(hw.cameras[k].index_or_path) for k in ("workspace","wrist")},
            "task":args.task,"fixed_state":fixed,"captures":[],"requests":[]}
    camera=RealCameraObservationSource.from_hardware_config_path(args.hardware_config)
    state_source = None
    state_connected = False
    try:
        camera.open()
        state_source=ReadOnlyRealFollowerStateSource.from_port(
            port=args.follower_port, follower_id=args.follower_id,
            calibration_path=args.follower_calibration_path,
        )
        state_source.connect(); state_connected = True; sequence=0
        for ci in range(args.captures):
            stamp=time.monotonic(); frames=camera.capture_all(); actual=state_source.read().positions_deg
            w=np.asarray(frames[WK].image_rgb); r=np.asarray(frames[RK].image_rgb)
            cap={"capture_sequence":ci,"capture_monotonic":stamp,
                 "frame_wall_times":{"workspace":frames[WK].captured_at_wall,"wrist":frames[RK].captured_at_wall},
                 "actual_read_only_state":actual,"images":{}}
            for label,arr in (("workspace",w),("wrist",r)):
                path=images_dir/f"capture_{ci:02d}_{label}.png";Image.fromarray(arr).save(path)
                _,jpeg=encode(arr);cap["images"][label]={"png_path":str(path.resolve()),
                    "raw_rgb_sha256":sha(arr.tobytes()),"jpeg_sha256":sha(jpeg),"shape":list(arr.shape)}
            report["captures"].append(cap)
            for condition,state in (("actual",actual),("fixed",fixed)):
                for name in VARIANTS:
                    rid=f"{args.scene_label}-c{ci:02d}-{condition}-{name}-{uuid.uuid4().hex[:8]}"
                    rec=infer(args,session,rid,sequence,state,variant_images(w,r,name))
                    rec.update({"capture_sequence":ci,"state_condition":condition,"variant":name})
                    report["requests"].append(rec);sequence+=1
            if ci+1<args.captures: time.sleep(args.interval_s)
    finally:
        if state_source is not None and state_connected:
            state_source.disconnect()
        camera.close()
    path=root/"scene_report.json";path.write_text(json.dumps(report,indent=2,ensure_ascii=False));print(path);return 0

def array(rec): return np.array([[step[j] for j in JOINT_ORDER] for step in rec["raw_action_chunk"]],float)
def index(report): return {(r["capture_sequence"],r["state_condition"],r["variant"]):r for r in report["requests"]}
def direction(x):
    d=float(x[-1]-x[0]);return "flat" if abs(d)<1e-6 else ("increasing" if d>0 else "decreasing")

def compare(args):
    a=json.loads(Path(args.scene_a_report).read_text());b=json.loads(Path(args.scene_b_report).read_text())
    ia,ib=index(a),index(b); captures=sorted(set(k[0] for k in ia if k[1:]==("fixed","normal")) & set(k[0] for k in ib if k[1:]==("fixed","normal")))
    rows=[]
    for c in captures:
        x,y=array(ia[(c,"fixed","normal")]),array(ib[(c,"fixed","normal")]);d=np.abs(x-y)
        rows.append({"capture_sequence":c,"overall_mae":float(d.mean()),"first_15_overall_mae":float(d[:15].mean()),
          "joint_mae":{j:float(d[:,i].mean()) for i,j in enumerate(JOINT_ORDER)},
          "first_15_joint_mae":{j:float(d[:15,i].mean()) for i,j in enumerate(JOINT_ORDER)},
          "trajectory_direction":{j:{"scene_a":direction(x[:,i]),"scene_b":direction(y[:,i])} for i,j in enumerate(JOINT_ORDER)}})
    ablations={}
    for scene,idx in (("scene_a",ia),("scene_b",ib)):
        ablations[scene]={}
        for v in VARIANTS[1:]:
            d=np.stack([np.abs(array(idx[(c,"fixed","normal")])-array(idx[(c,"fixed",v)])) for c in captures])
            ablations[scene][v]={"overall_mae_vs_normal":float(d.mean()),"joint_mae_vs_normal":{j:float(d[:,:,i].mean()) for i,j in enumerate(JOINT_ORDER)}}
    hashes={name:{cam:[c["images"][cam]["raw_rgb_sha256"] for c in rep["captures"]] for cam in ("workspace","wrist")} for name,rep in (("scene_a",a),("scene_b",b))}
    out={"same_fixed_state":a["fixed_state"]==b["fixed_state"],"image_hashes":hashes,
      "capture_freeze_candidate":{s:{cam:len(set(vals))==1 for cam,vals in cams.items()} for s,cams in hashes.items()},
      "client_server_all_hashes_match":all(all(r["client_server_jpeg_hash_match"].values()) for rep in (a,b) for r in rep["requests"]),
      "same_state_scene_comparisons":rows,"ablations":ablations}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2,ensure_ascii=False));print(path);return 0

def parse(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("capture");c.add_argument("--scene-label",required=True);c.add_argument("--captures",type=int,default=3);c.add_argument("--interval-s",type=float,default=.25)
    c.add_argument("--hardware-config",default=str(ROOT/"configs/hardware.local.json"));c.add_argument("--follower-port",required=True);c.add_argument("--follower-id",default="chanho_follower");c.add_argument("--follower-calibration-path")
    c.add_argument("--fixed-state",required=True);c.add_argument("--task",required=True);c.add_argument("--vla-server-url",required=True);c.add_argument("--vla-api-token");c.add_argument("--timeout-s",type=float,default=60);c.add_argument("--output-dir",default=str(ROOT/"reports/vision_conditioning"));c.set_defaults(func=capture)
    q=sub.add_parser("compare");q.add_argument("--scene-a-report",required=True);q.add_argument("--scene-b-report",required=True);q.add_argument("--output",required=True);q.set_defaults(func=compare)
    return p.parse_args(argv)
def main(argv=None): args=parse(argv);return args.func(args)
if __name__=="__main__":raise SystemExit(main())
