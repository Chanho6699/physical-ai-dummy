"""Pure planning and analysis for staged one-joint actuator tracking tests."""

from __future__ import annotations
import math
from dataclasses import dataclass
from runtime.common.vla_contract import JOINT_ORDER

ROTARY_STEPS = (0.5, 1.0, 2.0)
GRIPPER_STEPS = (0.5, 1.0, 2.0)

@dataclass(frozen=True)
class StagePlan:
    joint: str
    requested_delta: float
    start: float
    target: float
    mechanical_range: tuple[float,float]
    allowed: bool
    reason: str | None = None

def plan_stage(*,joint,start,requested_delta,mechanical_range,margin=0.25,max_cumulative=6.0,cumulative=0.0):
    if joint not in JOINT_ORDER or not all(math.isfinite(x) for x in (start,requested_delta,*mechanical_range)):
        return StagePlan(joint,requested_delta,start,start,mechanical_range,False,"invalid/non-finite input")
    lo,hi=mechanical_range;target=start+requested_delta
    if abs(cumulative+requested_delta)>max_cumulative:
        return StagePlan(joint,requested_delta,start,target,mechanical_range,False,"max cumulative displacement exceeded")
    if not lo+margin <= target <= hi-margin:
        return StagePlan(joint,requested_delta,start,target,mechanical_range,False,"mechanical boundary/margin")
    # Near an endpoint, only motion toward the range center is permitted.
    if start <= lo+margin and requested_delta<0:
        return StagePlan(joint,requested_delta,start,target,mechanical_range,False,"outward motion near lower endpoint")
    if start >= hi-margin and requested_delta>0:
        return StagePlan(joint,requested_delta,start,target,mechanical_range,False,"outward motion near upper endpoint")
    return StagePlan(joint,requested_delta,start,target,mechanical_range,True)

def analyze_stage(*,start,commanded,final_written,samples,response_threshold=0.08,steady_window=5):
    if not samples: raise ValueError("encoder samples are empty")
    ts=[float(x["elapsed_s"]) for x in samples];pos=[float(x["position"]) for x in samples]
    direction=1.0 if final_written>start else -1.0
    projected=[direction*(x-start) for x in pos]
    first=next((ts[i] for i,x in enumerate(projected) if x>=response_threshold),None)
    opposite=any(x<=-response_threshold for x in projected)
    net=pos[-1]-start
    peak=max(projected)
    requested=abs(final_written-start)
    overshoot=max(0.0,peak-requested)
    n=min(steady_window,len(pos))
    steady=sum(pos[-n:])/n
    return {
      "starting_encoder":start,"commanded_target":commanded,"final_written_target":final_written,
      "first_response_latency_s":first,"net_movement":net,
      "tracking_error":final_written-pos[-1],"overshoot":overshoot,
      "steady_state_error":final_written-steady,
      "command_encoder_direction_match":direction*net>0,
      "unexpected_opposite_movement":opposite,
      "no_response_deadband":first is None,
      "encoder_time_series":samples,
    }

def summarize_joint(stages):
    valid=[s for s in stages if s.get("write_executed")]
    responsive=[s for s in valid if not s["metrics"]["no_response_deadband"]]
    minimum=min((abs(s["requested_delta"]) for s in responsive),default=None)
    lat=[s["metrics"]["first_response_latency_s"] for s in responsive]
    return {
      "minimum_effective_step":minimum,
      "response_latency_median_s":None if not lat else sorted(lat)[len(lat)//2],
      "deadband":minimum is None or any(s["metrics"]["no_response_deadband"] for s in valid),
      "sign_mapping_ok":bool(valid) and all(s["metrics"]["command_encoder_direction_match"] for s in responsive) and not any(s["metrics"]["unexpected_opposite_movement"] for s in valid),
      "tracking_quality":"UNMEASURED" if not valid else ("NO_RESPONSE" if not responsive else "MEASURED"),
      "steps":{str(abs(s["requested_delta"])):s["metrics"]["net_movement"] for s in valid},
    }
