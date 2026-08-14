"""Passive all-joint servo lead telemetry for normal leader-to-follower teleoperation.

This module performs reads only. It must be called after the unmodified
SOFollower.send_action() path; it neither filters nor changes teleop commands.
"""
from __future__ import annotations
import csv, json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from runtime.common.vla_contract import JOINT_ORDER

REGISTERS=("Goal_Position","Present_Position","Present_Velocity","Present_Load","Present_Current","Moving")

@dataclass(frozen=True)
class ServoLeadSample:
    timestamp_s: float
    sequence: int
    joint: str
    leader_commanded_position: float | None
    follower_goal_position: float | None
    follower_present_position: float | None
    goal_present_lead: float | None
    present_velocity_raw: int | float | None
    present_load_raw: int | float | None
    present_current_raw: int | float | None
    moving: int | bool | None
    read_errors: dict[str,str]

class AllJointServoLeadLogger:
    """Read all six follower motors after each normal teleop send_action."""

    def __init__(self,*,bus,csv_path:Path):
        self.bus=bus;self.csv_path=Path(csv_path);self.csv_path.parent.mkdir(parents=True,exist_ok=True)
        self._file=self.csv_path.open("w",newline="",encoding="utf-8")
        self._writer=csv.DictWriter(self._file,fieldnames=list(ServoLeadSample.__dataclass_fields__))
        self._writer.writeheader();self.samples=[]

    def _read(self,name,*,normalize):
        try:return self.bus.sync_read(name,list(JOINT_ORDER),normalize=normalize,num_retry=0),None
        except Exception as exc:return {},f"{type(exc).__name__}: {exc}"

    def capture(self,*,timestamp_s,sequence,leader_action):
        values={};errors={}
        for name in REGISTERS:
            result,error=self._read(name,normalize=name in ("Goal_Position","Present_Position"))
            values[name]=result
            if error:errors[name]=error
        rows=[]
        for joint in JOINT_ORDER:
            goal=values["Goal_Position"].get(joint);present=values["Present_Position"].get(joint)
            lead=(float(goal)-float(present)) if goal is not None and present is not None else None
            row=ServoLeadSample(
                timestamp_s=float(timestamp_s),sequence=int(sequence),joint=joint,
                leader_commanded_position=leader_action.get(f"{joint}.pos"),
                follower_goal_position=goal,follower_present_position=present,goal_present_lead=lead,
                present_velocity_raw=values["Present_Velocity"].get(joint),
                present_load_raw=values["Present_Load"].get(joint),
                present_current_raw=values["Present_Current"].get(joint),
                moving=values["Moving"].get(joint),read_errors=dict(errors),
            )
            self.samples.append(row);rows.append(row);self._writer.writerow(asdict(row))
        self._file.flush();return rows

    def close(self):self._file.close()

def _pct(values,p):
    if not values:return None
    x=sorted(values);q=(len(x)-1)*p/100;lo=int(q);hi=min(lo+1,len(x)-1);return x[lo]+(x[hi]-x[lo])*(q-lo)

def analyze_servo_lead(samples,*,motion_threshold_deg=.08,endpoint_min_deg=90.0):
    result={"joints":{}}
    for joint in JOINT_ORDER:
        rows=[s for s in samples if s.joint==joint and s.goal_present_lead is not None]
        leads=[abs(s.goal_present_lead) for s in rows]
        onset=[]
        for before,after in zip(rows,rows[1:]):
            if before.follower_present_position is None or after.follower_present_position is None:continue
            movement=after.follower_present_position-before.follower_present_position
            if abs(movement)>=motion_threshold_deg:
                onset.append({"timestamp_s":after.timestamp_s,"lead_before":before.goal_present_lead,
                  "lead_after":after.goal_present_lead,"movement":movement,
                  "endpoint":abs(before.follower_present_position)>=endpoint_min_deg})
        endpoint=[abs(s.goal_present_lead) for s in rows if s.follower_present_position is not None and abs(s.follower_present_position)>=endpoint_min_deg]
        mid=[abs(s.goal_present_lead) for s in rows if s.follower_present_position is not None and abs(s.follower_present_position)<endpoint_min_deg]
        def stats(x):return {"n":len(x),"median":median(x) if x else None,"p90":_pct(x,90),"p95":_pct(x,95),"p99":_pct(x,99),"max":max(x) if x else None}
        endpoint_onset=[abs(x["lead_before"]) for x in onset if x["endpoint"] and x["lead_before"] is not None]
        result["joints"][joint]={"lead_abs":stats(leads),"endpoint_lead_abs":stats(endpoint),
          "midrange_lead_abs":stats(mid),"movement_onsets":onset,
          "minimum_observed_endpoint_onset_lead":min(endpoint_onset) if endpoint_onset else None}
    times=sorted({s.timestamp_s for s in samples})
    periods=[b-a for a,b in zip(times,times[1:]) if b>a]
    result["teleop_update_rate_hz"]=None if not periods else 1/median(periods)
    return result

def write_analysis(path,samples):
    report=analyze_servo_lead(samples);Path(path).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8");return report


def run_official_teleop_with_post_send_hook(
    *, teleop, robot, teleop_loop_fn, loop_kwargs, logger, clock,
):
    """Run the official lifecycle and loop, observing only after send_action.

    Hook failures are diagnostic-only. The exact action and return value of the
    original robot.send_action are preserved.
    """
    errors=[]
    sequence=0
    original_send=robot.send_action
    teleop_connected=False;robot_connected=False
    try:
        teleop.connect();teleop_connected=True
        robot.connect();robot_connected=True
        def observed_send(action):
            nonlocal sequence
            sent=original_send(action)
            try:
                logger.capture(timestamp_s=clock(),sequence=sequence,leader_action=dict(action))
            except Exception as exc:
                errors.append({"sequence":sequence,"error":f"{type(exc).__name__}: {exc}"})
            sequence+=1
            return sent
        robot.send_action=observed_send
        teleop_loop_fn(teleop=teleop,robot=robot,**loop_kwargs)
    finally:
        robot.send_action=original_send
        # Match lerobot_teleoperate.py exactly: teleoperator disconnects first.
        if teleop_connected:teleop.disconnect()
        if robot_connected:robot.disconnect()
        try:logger.close()
        except Exception as exc:errors.append({"sequence":sequence,"error":f"logger close: {exc}"})
    return errors
