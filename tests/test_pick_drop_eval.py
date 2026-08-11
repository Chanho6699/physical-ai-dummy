from __future__ import annotations

from simulation.mujoco.pick_drop_eval import (
    ReferenceZones,
    StepRecord,
    classify_failure,
    evaluate_kinematic,
    evaluate_physics,
    evaluate_trajectory,
    evaluate_trajectory_quality,
)

BIN_XY = (0.30, -0.30)
ZONES = ReferenceZones(bin_center_xy=BIN_XY, bin_inner_half=0.05)
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _cmd(step: int) -> dict[str, float]:
    return {j: 0.0 for j in JOINTS}


def _rec(step, ee, cube, gripper, contact=False, decision="ACCEPT") -> StepRecord:
    return StepRecord(
        step=step, sim_time=step / 30.0, ee_pos=ee, cube_pos=cube,
        gripper_cmd_percent=gripper, gripper_raw_percent=gripper,
        cube_jaw_contact=contact, safety_decision=decision,
    )


def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def build_full_success_trajectory() -> list[StepRecord]:
    """A hand-built trajectory that should pass every kinematic AND physics stage."""
    cube = (0.22, 0.02, 0.012)
    start_ee = (0.05, 0.05, 0.10)
    grasp_ee = (cube[0], cube[1], cube[2] + 0.005)
    records: list[StepRecord] = []
    step = 0

    # approach: open gripper, move toward cube over 30 steps
    for i in range(31):
        ee = _lerp(start_ee, grasp_ee, i / 30)
        records.append(_rec(step, ee, cube, gripper=80.0))
        step += 1

    # close gripper at cube (grasp), hold contact
    for i in range(10):
        records.append(_rec(step, grasp_ee, cube, gripper=10.0, contact=True))
        step += 1

    # lift: EE (and cube, following it) rise together
    lift_ee_top = (grasp_ee[0], grasp_ee[1], grasp_ee[2] + 0.08)
    for i in range(1, 21):
        ee = _lerp(grasp_ee, lift_ee_top, i / 20)
        cube_pos = (ee[0], ee[1], ee[2] - 0.005)
        records.append(_rec(step, ee, cube_pos, gripper=10.0, contact=True))
        step += 1

    # carry toward bin
    bin_ee = (BIN_XY[0], BIN_XY[1], lift_ee_top[2])
    for i in range(1, 41):
        ee = _lerp(lift_ee_top, bin_ee, i / 40)
        cube_pos = (ee[0], ee[1], ee[2] - 0.005)
        records.append(_rec(step, ee, cube_pos, gripper=10.0, contact=True))
        step += 1

    # release over bin: open gripper, cube drops into bin and settles
    for i in range(15):
        ee = bin_ee
        if i == 0:
            cube_pos = (bin_ee[0], bin_ee[1], bin_ee[2] - 0.005)
            contact = True
        else:
            cube_pos = (BIN_XY[0], BIN_XY[1], 0.012)
            contact = False
        records.append(_rec(step, ee, cube_pos, gripper=90.0, contact=contact))
        step += 1

    return records


def test_full_success_trajectory_passes_both_tracks():
    records = build_full_success_trajectory()
    result = evaluate_trajectory(
        records, ZONES,
        per_step_command_deg=[_cmd(r.step) for r in records],
        ended_by_safety_reject=False, ended_by_timeout=False,
    )
    assert result.kinematic.approach_success
    assert result.kinematic.grasp_pose_reached
    assert result.kinematic.gripper_close_detected
    assert result.kinematic.lift_success
    assert result.kinematic.carry_direction_ok
    assert result.kinematic.bin_vicinity_reached
    assert result.kinematic.gripper_release_detected
    assert result.kinematic.kinematic_pick_drop_success

    assert result.physics.grasp_contact_detected
    assert result.physics.cube_secured
    assert result.physics.lifted
    assert result.physics.carried
    assert result.physics.released
    assert result.physics.final_in_bin
    assert result.physics.physics_pick_drop_success

    assert result.failure.reason == "none"


def test_failed_approach_never_gets_close_to_cube():
    cube = (0.22, 0.02, 0.012)
    records = [_rec(i, (0.0 + i * 0.001, 0.30, 0.15), cube, gripper=80.0) for i in range(50)]
    kinematic = evaluate_kinematic(records, ZONES)
    assert not kinematic.approach_success
    assert not kinematic.kinematic_pick_drop_success

    failure = classify_failure(
        kinematic, evaluate_physics(records, ZONES),
        ended_by_safety_reject=False, ended_by_timeout=True,
    )
    assert failure.reason == "failed_approach"
    assert failure.track == "kinematic"


def test_safety_reject_classified_before_anything_else():
    records = build_full_success_trajectory()[:5]
    kinematic = evaluate_kinematic(records, ZONES)
    physics = evaluate_physics(records, ZONES)
    failure = classify_failure(kinematic, physics, ended_by_safety_reject=True, ended_by_timeout=False)
    assert failure.reason == "safety_reject"
    assert failure.likely_cause == "safety_reject"


def test_kinematic_success_but_no_physics_contact_is_sim_artifact_not_policy_failure():
    """EE trajectory does everything right (approach/grasp-pose/close/lift/carry/bin/release)
    but the cube geom never actually reports contact with the jaws - i.e. MuJoCo grasp physics
    failed even though the policy trajectory itself was fine. This must be classified as a
    simulation/physics artifact, not a policy trajectory failure."""
    cube = (0.22, 0.02, 0.012)  # cube never moves - "grasp" never physically holds it
    start_ee = (0.05, 0.05, 0.10)
    grasp_ee = (cube[0], cube[1], cube[2] + 0.005)
    records: list[StepRecord] = []
    step = 0
    for i in range(31):
        ee = _lerp(start_ee, grasp_ee, i / 30)
        records.append(_rec(step, ee, cube, gripper=80.0))
        step += 1
    for i in range(10):
        records.append(_rec(step, grasp_ee, cube, gripper=10.0, contact=False))  # no contact!
        step += 1
    lift_ee_top = (grasp_ee[0], grasp_ee[1], grasp_ee[2] + 0.08)
    for i in range(1, 21):
        ee = _lerp(grasp_ee, lift_ee_top, i / 20)
        records.append(_rec(step, ee, cube, gripper=10.0, contact=False))  # cube stays on table
        step += 1
    bin_ee = (BIN_XY[0], BIN_XY[1], lift_ee_top[2])
    for i in range(1, 41):
        ee = _lerp(lift_ee_top, bin_ee, i / 40)
        records.append(_rec(step, ee, cube, gripper=10.0, contact=False))
        step += 1
    for i in range(15):
        records.append(_rec(step, bin_ee, cube, gripper=90.0, contact=False))
        step += 1

    kinematic = evaluate_kinematic(records, ZONES)
    physics = evaluate_physics(records, ZONES)
    assert kinematic.kinematic_pick_drop_success
    assert not physics.physics_pick_drop_success
    assert not physics.grasp_contact_detected

    failure = classify_failure(kinematic, physics, ended_by_safety_reject=False, ended_by_timeout=False)
    assert failure.track == "physics"
    assert failure.likely_cause == "sim_physics_artifact"


def test_trajectory_quality_clamp_free_and_jerk():
    smooth = [{"shoulder_pan": float(i), "shoulder_lift": 0.0, "elbow_flex": 0.0,
               "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 50.0} for i in range(10)]
    decisions = ["ACCEPT"] * 10
    q = evaluate_trajectory_quality(smooth, decisions)
    assert q.clamp_free
    assert q.mean_abs_jerk_deg == 0.0  # constant velocity -> zero jerk
    assert q.max_single_step_delta_deg == 1.0

    decisions_with_clamp = ["ACCEPT"] * 9 + ["WOULD_CLAMP"]
    q2 = evaluate_trajectory_quality(smooth, decisions_with_clamp)
    assert not q2.clamp_free


def test_empty_trajectory_does_not_crash():
    kinematic = evaluate_kinematic([], ZONES)
    physics = evaluate_physics([], ZONES)
    assert not kinematic.kinematic_pick_drop_success
    assert not physics.physics_pick_drop_success
