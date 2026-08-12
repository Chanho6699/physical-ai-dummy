from __future__ import annotations
import base64, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
import io
from runtime.common.vla_contract import CAMERA_WORKSPACE_KEY as WK, CAMERA_WRIST_KEY as RK, JOINT_ORDER
from scripts.run_vision_conditioning_diagnostic import compare

def jpeg(value):
    arr=np.full((12,16,3),value,dtype=np.uint8);buf=io.BytesIO();Image.fromarray(arr).save(buf,format="JPEG",quality=90)
    raw=buf.getvalue();return base64.b64encode(raw).decode(),hashlib.sha256(raw).hexdigest()

def test_server_echoes_and_persists_input_diagnostic(tmp_path):
    pytest = __import__("pytest")
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from runtime.desktop.vla_server import FakePolicyRunner, create_app
    TestClient = fastapi_testclient.TestClient
    w,wh=jpeg(10);r,rh=jpeg(200);state={j:float(i) for i,j in enumerate(JOINT_ORDER)}
    body={"session_id":"s","request_id":"scene-a-0","task":"pick","sequence":3,"timestamp":1.0,
          "observation":{"state":state,"images":{WK:w,RK:r}}}
    response=TestClient(create_app(policy_runner=FakePolicyRunner(),input_diagnostic_dir=tmp_path)).post("/predict_chunk",json=body)
    assert response.status_code==200
    data=response.json();diag=data["input_diagnostic"]
    assert diag["request_id"]=="scene-a-0"
    assert diag["cameras"][WK]["jpeg_sha256"]==wh
    assert diag["cameras"][RK]["jpeg_sha256"]==rh
    assert diag["cameras"][WK]["decoded_rgb_shape"]==[12,16,3]
    assert len(diag["raw_action_chunk"])==50
    line=json.loads((tmp_path/"server_input_diagnostics.jsonl").read_text())
    assert line["request_id"]=="scene-a-0"
    assert len(line["raw_action_chunk"])==50

def request(capture,variant,offset):
    chunk=[{j:float(i+offset+step) for i,j in enumerate(JOINT_ORDER)} for step in range(50)]
    return {"capture_sequence":capture,"state_condition":"fixed","variant":variant,
            "raw_action_chunk":chunk,"client_server_jpeg_hash_match":{WK:True,RK:True}}

def scene(label,offset):
    req=[]
    for c in range(3):
        for v,extra in (("normal",0),("swap",1),("workspace_duplicate",2),("wrist_duplicate",3)):
            req.append(request(c,v,offset+extra))
    return {"scene_label":label,"fixed_state":{j:0. for j in JOINT_ORDER},
            "captures":[{"images":{"workspace":{"raw_rgb_sha256":f"{label}-w-{c}"},"wrist":{"raw_rgb_sha256":f"{label}-r-{c}"}}} for c in range(3)],
            "requests":req}

def test_compare_reports_same_state_mae_and_ablations(tmp_path):
    a,b=scene("A",0),scene("B",5);pa=tmp_path/"a.json";pb=tmp_path/"b.json";out=tmp_path/"out.json"
    pa.write_text(json.dumps(a));pb.write_text(json.dumps(b))
    args=type("Args",(),{"scene_a_report":str(pa),"scene_b_report":str(pb),"output":str(out)})()
    assert compare(args)==0
    result=json.loads(out.read_text())
    assert result["same_fixed_state"] is True
    assert result["client_server_all_hashes_match"] is True
    assert result["same_state_scene_comparisons"][0]["overall_mae"]==5.0
    assert result["same_state_scene_comparisons"][0]["first_15_joint_mae"]["shoulder_lift"]==5.0
    assert result["ablations"]["scene_a"]["swap"]["overall_mae_vs_normal"]==1.0
    assert not any(result["capture_freeze_candidate"]["scene_a"].values())

def test_diagnostic_script_has_no_control_or_motor_write_construction():
    source=(Path(__file__).resolve().parents[1]/"scripts/run_vision_conditioning_diagnostic.py").read_text()
    forbidden=("follower_action_writer","realtime_control_loop","SOFollower(","send_action(","write(")
    assert all(token not in source for token in forbidden)
