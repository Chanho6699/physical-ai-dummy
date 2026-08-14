from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from hardware.safety.joint_tracking_characterization import analyze_stage,plan_stage,summarize_joint
from scripts.run_joint_tracking_characterization import build_dry_report,parse

def test_plan_blocks_boundary_and_cumulative():
 assert not plan_stage(joint="elbow_flex",start=96.9,requested_delta=0.5,mechanical_range=(-97.18,97.18)).allowed
 assert plan_stage(joint="elbow_flex",start=96.9,requested_delta=-0.5,mechanical_range=(-97.18,97.18)).allowed
 assert not plan_stage(joint="shoulder_pan",start=0,requested_delta=2,mechanical_range=(-90,90),cumulative=5,max_cumulative=6).allowed

def test_metrics_response_overshoot_and_direction():
 samples=[{"elapsed_s":0.0,"position":10.0},{"elapsed_s":.02,"position":10.02},{"elapsed_s":.04,"position":10.2},{"elapsed_s":.1,"position":11.1},{"elapsed_s":.2,"position":11.0}]
 m=analyze_stage(start=10,commanded=11,final_written=11,samples=samples,response_threshold=.08)
 assert m["first_response_latency_s"]==pytest.approx(.04)
 assert m["net_movement"]==pytest.approx(1)
 assert m["overshoot"]==pytest.approx(.1)
 assert m["command_encoder_direction_match"] is True

def test_metrics_detects_no_response_and_opposite():
 samples=[{"elapsed_s":0,"position":10},{"elapsed_s":.1,"position":9.8}]
 m=analyze_stage(start=10,commanded=11,final_written=11,samples=samples)
 assert m["no_response_deadband"] is True
 assert m["unexpected_opposite_movement"] is True

def test_summary_minimum_step():
 stages=[{"requested_delta":.5,"write_executed":True,"metrics":{"no_response_deadband":True,"unexpected_opposite_movement":False,"command_encoder_direction_match":False,"net_movement":0}},
 {"requested_delta":1.0,"write_executed":True,"metrics":{"no_response_deadband":False,"unexpected_opposite_movement":False,"command_encoder_direction_match":True,"first_response_latency_s":.1,"net_movement":.8}}]
 assert summarize_joint(stages)["minimum_effective_step"]==1.0

def test_cli_has_no_vla_or_control_loop_and_dry_run_default():
 s=(Path(__file__).parents[1]/"scripts/run_joint_tracking_characterization.py").read_text()
 assert "VLA" not in s and "realtime_control_loop" not in s
 assert "execute(args) if args.execute else build_dry_report(args)" in s
 assert "disable_torque_on_disconnect=False" in s

def test_elbow_joint_filter_creates_only_elbow_stages(monkeypatch):
 class Config:
  joint_range_deg={joint:(-180.0,180.0) for joint in (
   "shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")}
 monkeypatch.setattr(
  "scripts.run_joint_tracking_characterization.SafetyGateConfig.from_repo_defaults",
  lambda **_:Config(),
 )
 args=SimpleNamespace(
  calibration_path="unused",initial_state=json.dumps({joint:0.0 for joint in Config.joint_range_deg}),
  joint="elbow_flex",boundary_margin=.25,max_cumulative=6.0,
 )
 report=build_dry_report(args)
 assert tuple(report["joints"])==("elbow_flex",)
 assert [plan["requested_delta"] for plan in report["joints"]["elbow_flex"]["plans"]]==[
  .5,-.5,1.0,-1.0,2.0,-2.0,
 ]


def test_invalid_joint_is_argparse_error():
 with pytest.raises(SystemExit) as exc:
  parse(["--initial-state","{}","--calibration-path","unused","--joint","not_a_joint"])
 assert exc.value.code==2