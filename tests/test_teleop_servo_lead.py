from __future__ import annotations
import inspect
import pytest
from hardware.diagnostics.teleop_servo_lead import (
 AllJointServoLeadLogger,
 ServoLeadSample,
 analyze_servo_lead,
 run_official_teleop_with_post_send_hook,
)
from runtime.common.vla_contract import JOINT_ORDER

class Bus:
 def __init__(self):
  self.calls=[]
 def sync_read(self,name,motors,normalize,num_retry):
  self.calls.append((name,tuple(motors),normalize,num_retry))
  if name=="Goal_Position":return {j:95.0 for j in motors}
  if name=="Present_Position":return {j:(96.0 if j=="elbow_flex" else 94.5) for j in motors}
  return {j:0 for j in motors}

def test_logger_reads_six_joints_without_writes(tmp_path):
 bus=Bus();logger=AllJointServoLeadLogger(bus=bus,csv_path=tmp_path/"lead.csv")
 rows=logger.capture(timestamp_s=.1,sequence=1,leader_action={f"{j}.pos":1.0 for j in JOINT_ORDER});logger.close()
 assert len(rows)==6 and rows[2].goal_present_lead==-1.0
 assert {x[0] for x in bus.calls}=={"Goal_Position","Present_Position","Present_Velocity","Present_Load","Present_Current","Moving"}
 source=inspect.getsource(AllJointServoLeadLogger)
 assert "self.bus.write(" not in source and "self.bus.sync_write(" not in source and "self.bus.send_action(" not in source

def test_analysis_reports_endpoint_onset_and_percentiles():
 def row(t,p,g=94.0):
  return ServoLeadSample(t,int(t*10),"elbow_flex",g,g,p,g-p,0,0,0,0,{})
 samples=[row(0,96.0),row(.1,96.0,93.0),row(.2,95.8,93.0)]
 report=analyze_servo_lead(samples)
 elbow=report["joints"]["elbow_flex"]
 assert elbow["minimum_observed_endpoint_onset_lead"]==pytest.approx(3.0)
 assert elbow["lead_abs"]["max"]==pytest.approx(3.0)
 assert report["teleop_update_rate_hz"]==pytest.approx(10.0)


class FakeTeleop:
 def __init__(self,events):self.events=events
 def connect(self):self.events.append("teleop.connect")
 def disconnect(self):self.events.append("teleop.disconnect")


class FakeRobot:
 def __init__(self,events):
  self.events=events;self.commands=[];self.configure_count=0
 def connect(self):
  self.events.append("robot.connect");self.configure_count+=1
 def disconnect(self):self.events.append("robot.disconnect")
 def send_action(self,action):
  self.events.append(("send",dict(action)));self.commands.append(dict(action));return action


class FakeLogger:
 def __init__(self,events,fail=False):
  self.events=events;self.fail=fail
 def capture(self,**kwargs):
  self.events.append(("read",kwargs["sequence"]))
  if self.fail:raise OSError("diagnostic disk failure")
 def close(self):self.events.append("logger.close")


def _run_hook(*,instrumented,logger_fails=False):
 events=[];teleop=FakeTeleop(events);robot=FakeRobot(events)
 commands=[{"elbow_flex.pos":96.0},{"elbow_flex.pos":93.0}]
 def official_loop(*,teleop,robot,**kwargs):
  events.append("loop")
  for command in commands:robot.send_action(command)
 logger=FakeLogger(events,fail=logger_fails)
 if instrumented:
  errors=run_official_teleop_with_post_send_hook(
   teleop=teleop,robot=robot,teleop_loop_fn=official_loop,loop_kwargs={},
   logger=logger,clock=lambda:1.0,
  )
 else:
  teleop.connect();robot.connect();official_loop(teleop=teleop,robot=robot)
  teleop.disconnect();robot.disconnect();logger.close();errors=[]
 return events,robot,errors


def test_official_lifecycle_and_command_path_are_unchanged():
 plain_events,plain_robot,_=_run_hook(instrumented=False)
 measured_events,measured_robot,_=_run_hook(instrumented=True)
 assert plain_robot.commands==measured_robot.commands
 assert measured_robot.configure_count==1
 assert [e for e in measured_events if isinstance(e,tuple) and e[0]=="send"]==[
  ("send",{"elbow_flex.pos":96.0}),("send",{"elbow_flex.pos":93.0})
 ]
 assert measured_events[:3]==["teleop.connect","robot.connect","loop"]
 assert measured_events[-3:]==["teleop.disconnect","robot.disconnect","logger.close"]
 for index,event in enumerate(measured_events):
  if isinstance(event,tuple) and event[0]=="read":
   assert measured_events[index-1][0]=="send"
 assert not any("write" in str(event).lower() for event in measured_events)


def test_logging_failure_does_not_block_motor_commands():
 events,robot,errors=_run_hook(instrumented=True,logger_fails=True)
 assert len(robot.commands)==2
 assert len(errors)==2
 assert [e for e in events if isinstance(e,tuple) and e[0]=="send"]==[
  ("send",{"elbow_flex.pos":96.0}),("send",{"elbow_flex.pos":93.0})
 ]
