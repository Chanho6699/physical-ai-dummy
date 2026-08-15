#!/usr/bin/env python3
"""Convert saved read-only shadow chunks to a wall-clock-free virtual replay."""

from __future__ import annotations
import argparse, hashlib, json, math, statistics, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from hardware.state_server.readonly_so101_reader import JOINT_ORDER
from runtime.laptop.temporal_ensemble import TemporalEnsembler
from runtime.laptop.trajectory_buffer import TrajectoryBuffer
from runtime.laptop.trajectory_chunk import TimestampedActionChunk

def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def write_json(path,x): Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n")
def write_jsonl(path,x): Path(path).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in x))
def summary(x):
    return {"mean":statistics.fmean(x),"p50":float(np.percentile(x,50)),"p95":float(np.percentile(x,95)),"max":max(x)} if x else {"mean":None,"p50":None,"p95":None,"max":None}
def delta(a,b): return {j:float(a[j]-b[j]) for j in JOINT_ORDER}
def norm(d): return math.sqrt(sum(d[j]**2 for j in JOINT_ORDER))
def sample(chunk,t):
    if not TemporalEnsembler._covers(chunk,t): return None
    return TemporalEnsembler._sample_action_at(chunk,t)[0]

def virtual_chunks(observations,chunks,mode):
    if len(observations)!=len(chunks): raise ValueError("observation/chunk count mismatch")
    epoch=float(observations[0]["timestamp"]); out=[]
    for i,(o,c) in enumerate(zip(observations,chunks)):
        if int(o["sequence"])!=i or int(c["sequence"])!=i: raise ValueError(f"sequence mismatch at {i}")
        if len(c["raw_chunk"])!=int(c["chunk_size"]): raise ValueError(f"chunk size mismatch at {i}")
        ot=float(o["timestamp"])-epoch
        latency=0.0 if mode=="zero-compute" else float(c["inference_latency_ms"])/1000
        q=dict(c); q.update(observation_offset_s=ot,request_offset_s=ot,response_offset_s=ot+latency,virtual_publication_offset_s=ot+latency,publication_mode=mode)
        out.append(q)
    return out,float(observations[-1]["timestamp"])-epoch

def simulate(chunks,end_s,hz=60.0,half_life=0.338,phase_continuity=False,fade_scale=1.0):
    buf=TrajectoryBuffer(max_chunks=4); ens=TemporalEnsembler(half_life_s=half_life,max_contributors=3,phase_continuity=phase_continuity,phase_fade_cadence_scale=fade_scale)
    ticks=[]; handoffs=[]; pi=0; ever=False; stale=0; overlap=0; counts={}; dt=1/hz
    for ti in range(int(math.floor(end_s*hz))+1):
        t=ti*dt; before_valid=buf.valid_chunks(t); before=ens.compute_target(before_valid,t)
        old=before_valid[-1] if before_valid else None; raw_old=sample(old,t) if old else None; published=[]
        while pi<len(chunks) and chunks[pi]["response_offset_s"]<=t+1e-12:
            c=chunks[pi]
            obj=TimestampedActionChunk(sequence=c["sequence"],session_id="virtual",observation_time_monotonic=c["observation_offset_s"],request_started_time_monotonic=c["request_offset_s"],response_received_time_monotonic=c["response_offset_s"],server_received_at=None,server_responded_at=None,inference_latency_ms=c["inference_latency_ms"],chunk_index_spacing_s=c["chunk_index_spacing_s"],chunk_size=c["chunk_size"],actions=tuple(c["raw_chunk"]),model_id=c.get("model_id"),backend=c.get("backend"))
            r=buf.publish(obj)
            if not r.accepted: raise ValueError(r.reason)
            published.append(obj.sequence); pi+=1; ever=True
        valid=buf.valid_chunks(t); target=ens.compute_target(valid,t); look=ens.compute_target(valid,t+dt)
        new=valid[-1] if valid else None; raw_new=sample(new,t) if new else None
        ids=target.contributing_sequences if target else (); counts[len(ids)]=counts.get(len(ids),0)+1
        if len(ids)>=2: overlap+=1
        if ever and target is None: stale+=1
        handoff=bool(published and before and target and raw_old and raw_new)
        if handoff:
            rd=delta(raw_new,raw_old); ed=delta(target.action,before.action)
            handoffs.append({"tick_index":ti,"virtual_time_s":t,"published_sequences":published,"contributors_before":list(before.contributing_sequences),"contributors_after":list(ids),"previous_target":before.action,"new_raw_target":raw_new,"ensemble_interpolated_target":target.action,"raw_handoff_jump_l2":norm(rd),"ensemble_handoff_jump_l2":norm(ed),"raw_joint_jump":rd,"ensemble_joint_jump":ed})
        ticks.append({"tick_index":ti,"virtual_time_s":t,"published_sequences":published,"contributor_ids":list(ids),"contributor_count":len(ids),"raw_interpolated_target":raw_new,"raw_ensemble_target":target.action if target else None,"target_lookahead":look.action if look else None,"temporal_ensemble_interpolated_target":target.action if target else None,"handoff_tick":handoff,"stale":bool(ever and target is None)})
    usable=sum(x["temporal_ensemble_interpolated_target"] is not None for x in ticks)
    diag={"total_ticks":len(ticks),"usable_ticks":usable,"usable_target_fraction":usable/len(ticks),"no_target_fraction":1-usable/len(ticks),"stale_fraction":stale/len(ticks),"contributor_count_distribution":{str(k):v/len(ticks) for k,v in sorted(counts.items())},"handoff_count":len(handoffs),"contributor_overlap_duration_s":overlap/hz,"contributor_overlap_fraction":overlap/len(ticks)}
    return ticks,handoffs,diag

def analyze(ticks,handoffs,hz):
    raw=[h["raw_handoff_jump_l2"] for h in handoffs]; sm=[h["ensemble_handoff_jump_l2"] for h in handoffs]
    rmean=statistics.fmean(raw) if raw else None; smean=statistics.fmean(sm) if sm else None
    joints={}
    for j in JOINT_ORDER:
        r=[abs(h["raw_joint_jump"][j]) for h in handoffs]; e=[abs(h["ensemble_joint_jump"][j]) for h in handoffs]
        joints[j]={"raw":summary(r),"ensemble":summary(e),"mean_reduction_fraction":1-statistics.fmean(e)/statistics.fmean(r) if r and statistics.fmean(r) else None}
    stream={}
    for j in JOINT_ORDER:
        samples=[(x["virtual_time_s"],x["temporal_ensemble_interpolated_target"][j]) for x in ticks if x["temporal_ensemble_interpolated_target"]]
        vel=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(samples,samples[1:]) if b[0]-a[0]<=1.5/hz]
        acc=[(b-a)*hz for a,b in zip(vel,vel[1:])]; jerk=[(b-a)*hz for a,b in zip(acc,acc[1:])]
        signs=[1 if x>1e-4 else -1 if x< -1e-4 else 0 for x in vel]; nz=[x for x in signs if x]; dur=samples[-1][0]-samples[0][0] if len(samples)>1 else 0
        stream[j]={"reversal_per_s":sum(a!=b for a,b in zip(nz,nz[1:]))/dur if dur else None,"velocity_variance":statistics.pvariance(vel) if len(vel)>1 else None,"acceleration_rms":math.sqrt(statistics.fmean(x*x for x in acc)) if acc else None,"acceleration_variance":statistics.pvariance(acc) if len(acc)>1 else None,"jerk_rms":math.sqrt(statistics.fmean(x*x for x in jerk)) if jerk else None,"jerk_variance":statistics.pvariance(jerk) if len(jerk)>1 else None}
    by={}
    for n in sorted(set(x["contributor_count"] for x in ticks)):
        vals=[x["temporal_ensemble_interpolated_target"] for x in ticks if x["contributor_count"]==n and x["temporal_ensemble_interpolated_target"]]
        by[str(n)]={"ticks":len(vals),"target_step_l2":summary([norm(delta(b,a)) for a,b in zip(vals,vals[1:])])}
    return {"handoff_jump_l2":{"raw":summary(raw),"ensemble_interpolated":summary(sm),"mean_reduction_fraction":1-smean/rmean if rmean else None},"handoff_joint":joints,"target_stream":stream,"stability_by_contributor_count":by}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--observation-dir",required=True); p.add_argument("--source-replay-dir",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--publication-mode",choices=("latency-replay","zero-compute"),default="latency-replay"); p.add_argument("--control-hz",type=float,default=60); p.add_argument("--ensemble-half-life-s",type=float,default=.338); p.add_argument("--phase-continuity",action="store_true"); p.add_argument("--fade-cadence-scale",type=float,default=1.0); a=p.parse_args(argv)
    od=Path(a.observation_dir); sd=Path(a.source_replay_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    obs=read_jsonl(od/"observations.jsonl"); chunks=read_jsonl(sd/"chunks.jsonl"); timed,end=virtual_chunks(obs,chunks,a.publication_mode); ticks,handoffs,diag=simulate(timed,end,a.control_hz,a.ensemble_half_life_s,a.phase_continuity,a.fade_cadence_scale)
    write_jsonl(out/"virtual_chunks.jsonl",timed); write_jsonl(out/"virtual_targets_60hz.jsonl",ticks); write_jsonl(out/"handoffs.jsonl",handoffs); write_json(out/"analysis.json",analyze(ticks,handoffs,a.control_hz))
    report={"schema":"readonly-real-shadow-virtual-replay-v1","publication_mode":a.publication_mode,"phase_continuity":a.phase_continuity,"fade_cadence_scale":a.fade_cadence_scale,"observation_timeline":"recorded timestamp relative to first observation","compute_wall_clock_used_for_virtual_time":False,"source_replay_dir":str(sd.resolve()),"source_observation_sha256":hashlib.sha256((od/"observations.jsonl").read_bytes()).hexdigest(),"raw_chunk_mapping_verified":True,"control_hz":a.control_hz,"writer_created":False,"write_count":0,**diag}; write_json(out/"report.json",report); print(json.dumps(report))
if __name__=="__main__": main()
