#!/usr/bin/env python3
"""Staged SO-101 joint tracking characterization. Dry-run is the default."""

from __future__ import annotations
import argparse,json,signal,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime.common.vla_contract import JOINT_ORDER
from runtime.laptop.safety_gate import SafetyGateConfig
from hardware.safety.joint_tracking_characterization import ROTARY_STEPS,GRIPPER_STEPS,analyze_stage,plan_stage,summarize_joint

CONFIRM="I AM PHYSICALLY PRESENT AND AUTHORIZE ONE JOINT AT A TIME"
TELEMETRY=("Present_Load","Present_Current","Present_Velocity","Moving","Present_Voltage","Present_Temperature","Status")

def sequence_for(joint):
    steps=GRIPPER_STEPS if joint=="gripper" else ROTARY_STEPS
    return tuple(d for step in steps for d in (step,-step))

def selected_joints(args):
    return (args.joint,) if args.joint else JOINT_ORDER

def build_dry_report(args):
    config=SafetyGateConfig.from_repo_defaults(calibration_file_path=args.calibration_path)
    state=json.loads(Path(args.initial_state).read_text()) if Path(args.initial_state).is_file() else json.loads(args.initial_state)
    report={"mode":"DRY_RUN_NO_HARDWARE","created_at":datetime.now().isoformat(),"mechanical_ranges":config.joint_range_deg,"joints":{}}
    for joint in selected_joints(args):
        cumulative=0.;plans=[]
        for delta in sequence_for(joint):
            plan=plan_stage(joint=joint,start=float(state[joint])+cumulative,requested_delta=delta,
              mechanical_range=config.joint_range_deg[joint],margin=args.boundary_margin,max_cumulative=args.max_cumulative,cumulative=cumulative)
            plans.append(plan.__dict__)
            if plan.allowed:cumulative+=delta
        report["joints"][joint]={"plans":plans,"summary":{"minimum_effective_step":None,"tracking_quality":"UNMEASURED"}}
    return report

def telemetry(bus,joint):
    out={}
    for register in TELEMETRY:
        try:out[register]=bus.read(register,joint,normalize=False,num_retry=0)
        except Exception as exc:out[register]={"error":f"{type(exc).__name__}: {exc}"}
    return out

def execute(args):
    if not args.execute or args.confirm!=CONFIRM: raise RuntimeError("armed execution requires --execute and exact --confirm phrase")
    from lerobot.robots.so_follower import SO101FollowerConfig,SOFollower
    from hardware.safety.staged_follower_writer import StagedFollowerArmedWriter
    config=SafetyGateConfig.from_repo_defaults(calibration_file_path=args.calibration_path)
    follower=SOFollower(SO101FollowerConfig(port=args.follower_port,id=args.follower_id,cameras={},disable_torque_on_disconnect=False))
    # One write per planned stage; no retry, no automatic return-to-origin.
    max_writes=len(selected_joints(args))*6
    writer=StagedFollowerArmedWriter(follower=follower,max_write_count=max_writes)
    stop={"value":False}
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__("value",True))
    report={"mode":"ARMED_SINGLE_JOINT","created_at":datetime.now().isoformat(),"joints":{},"telemetry_note":"Feetech raw registers; Present_Current has no trusted physical-unit conversion."}
    try:
      writer.connect()
      for joint in selected_joints(args):
       stages=[];origin=writer.read_state_deg()[joint];cumulative=0.
       for delta in sequence_for(joint):
        if stop["value"]:break
        state=writer.read_state_deg()
        plan=plan_stage(joint=joint,start=state[joint],requested_delta=delta,mechanical_range=config.joint_range_deg[joint],margin=args.boundary_margin,max_cumulative=args.max_cumulative,cumulative=state[joint]-origin)
        row={"requested_delta":delta,"plan":plan.__dict__,"write_executed":False}
        if not plan.allowed:stages.append(row);continue
        command=dict(state);command[joint]=plan.target
        result=writer.write_action_once(command);row["write"]=result.to_dict();row["write_executed"]=result.executed
        if not result.executed:stages.append(row);stop["value"]=True;break
        sent=result.sent_action_deg or {};final=float(sent.get(joint,plan.target));samples=[];t0=time.monotonic()
        while time.monotonic()-t0<=args.timeout_s and not stop["value"]:
          now=time.monotonic();cur=writer.read_state_deg()
          sample={"elapsed_s":now-t0,"position":cur[joint],"all_joint_state":cur}
          bus=getattr(follower,"bus",None)
          if bus is not None:sample["telemetry_raw"]=telemetry(bus,joint)
          samples.append(sample)
          direction=1 if final-state[joint]>0 else -1
          if direction*(cur[joint]-state[joint]) < -args.opposite_threshold:
            row["abort_reason"]="unexpected opposite movement";stop["value"]=True;break
          time.sleep(1/args.sample_hz)
        row["metrics"]=analyze_stage(start=state[joint],commanded=plan.target,final_written=final,samples=samples,response_threshold=args.response_threshold)
        stages.append(row)
        if stop["value"]:break
       report["joints"][joint]={"stages":stages,"summary":summarize_joint(stages)}
       if stop["value"]:break
    finally:
      writer.disconnect()
    return report

def render_summary(report):
 rows=["| joint | minimum effective step | response latency | 0.5 response | 1.0 response | 2.0 response | deadband | sign/mapping | tracking quality |",
       "|---|---:|---:|---:|---:|---:|---|---|---|"]
 for joint in report["joints"]:
  summary=report["joints"][joint].get("summary",{})
  steps=summary.get("steps",{})
  def value(key):
   raw=steps.get(key)
   return "N/A" if raw is None else f"{raw:.4f}"
  minimum=summary.get("minimum_effective_step")
  latency=summary.get("response_latency_median_s")
  rows.append(f"| {joint} | {'N/A' if minimum is None else minimum} | {'N/A' if latency is None else f'{latency:.4f}s'} | {value('0.5')} | {value('1.0')} | {value('2.0')} | {summary.get('deadband','UNMEASURED')} | {summary.get('sign_mapping_ok','UNMEASURED')} | {summary.get('tracking_quality','UNMEASURED')} |")
 return "\n".join(rows)+"\n"

def parse(argv=None):
 p=argparse.ArgumentParser(description="SO-101 staged joint tracking characterization (dry-run default)")
 p.add_argument("--initial-state",required=True,help="dry-run JSON/path")
 p.add_argument("--joint",choices=JOINT_ORDER)
 p.add_argument("--calibration-path",required=True);p.add_argument("--output-root",default=str(ROOT/"reports/joint_tracking_characterization"))
 p.add_argument("--boundary-margin",type=float,default=.25);p.add_argument("--max-cumulative",type=float,default=6.)
 p.add_argument("--sample-hz",type=float,default=50.);p.add_argument("--timeout-s",type=float,default=1.5)
 p.add_argument("--response-threshold",type=float,default=.08);p.add_argument("--opposite-threshold",type=float,default=.15)
 p.add_argument("--execute",action="store_true");p.add_argument("--confirm");p.add_argument("--follower-port");p.add_argument("--follower-id",default="chanho_follower")
 return p.parse_args(argv)
def main(argv=None):
 args=parse(argv)
 if args.execute and not args.follower_port:raise SystemExit("--execute requires --follower-port")
 report=execute(args) if args.execute else build_dry_report(args)
 out=Path(args.output_root)/datetime.now().strftime("%Y%m%d_%H%M%S");out.mkdir(parents=True,exist_ok=False)
 (out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
 (out/"summary.json").write_text(json.dumps({j:v.get("summary") for j,v in report["joints"].items()},indent=2,ensure_ascii=False))
 (out/"summary.md").write_text(render_summary(report))
 print(out);return 0
if __name__=="__main__":raise SystemExit(main())
