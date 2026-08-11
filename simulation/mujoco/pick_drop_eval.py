"""Pick&drop rollout 평가 - kinematic(Track A, 주 지표) / physics(Track B, 부차 지표) 분리.

``docs/mujoco_scene_to_so101_semantics.md`` §4에 적은 대로, MuJoCo의 grasp contact 물리는
신뢰도가 낮고(임계값 미검증, 일부 gripper collision geom 이름 없음) synthetic camera를 쓰는
Secondary track에는 visual domain gap도 있다. 그래서 이 모듈은 두 트랙을 항상 분리해서 판정한다:

    Track A (kinematic/semantic, 주 지표): ``gripperframe`` site와 cube body의 위치만으로
    판정한다 - MuJoCo contact가 전혀 감지되지 않아도(즉 물리적으로 grasp가 실패해도) 정책의
    궤적 자체가 "말이 되는지"(접근했는가/그립 자세에 도달했는가/타이밍이 맞는가/드는 방향인가/
    bin 쪽으로 옮기는가/놓는 타이밍이 맞는가)는 채점할 수 있다.

    Track B (physics, 부차 지표): cube geom과 gripper jaw geom(``fixed_jaw_*``/``moving_jaw_*``,
    ``simulation/mujoco/assets/scene_pick_drop.xml``에 이미 이름이 있는 것만 - 이름 없는 mesh
    collision geom은 쓰지 않는다) 사이의 실제 MuJoCo contact로 판정한다.

이 모듈은 순수 함수/dataclass만 담고 있다 - MuJoCo나 정책을 직접 호출하지 않으므로
``mujoco``/``torch`` 없이도 import와 단위 테스트가 가능하다 (호출자가 이미 계산해 둔
``StepRecord`` 목록만 받는다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Vec3 = tuple[float, float, float]

# -- 트랙 A(kinematic) 임계값 - 전부 이 모듈이 진단용으로 새로 도입한 값이며 Safety Gate
# 임계값(configs/safety_gate.yaml)과는 무관하다. 실측 grasp 통계가 없으므로(문서 §3) 보수적인
# 근사치다 - 튜닝이 필요하면 이 상수들만 바꾸면 된다.
APPROACH_DIST_M = 0.06
GRASP_POSE_DIST_M = 0.028
LIFT_HEIGHT_M = 0.02
BIN_VICINITY_DIST_M = 0.07
GRIPPER_OPEN_PERCENT = 40.0  # 이 값 이상이면 "열림"으로 본다 (0=완전히 닫힘, 100=완전히 열림)
GRIPPER_CLOSED_PERCENT = 20.0  # 이 값 이하이면 "닫힘"으로 본다
CLOSE_TIMING_WINDOW_STEPS = 45  # min-distance 시점 기준 앞뒤 몇 step 이내를 "제때"로 볼지
LIFT_WINDOW_STEPS = 60  # gripper close 이후 몇 step 안에 lift가 일어나야 하는지

# -- 트랙 B(physics) 임계값
SECURED_MIN_CONSECUTIVE_CONTACT_STEPS = 5
DROPPED_EARLY_MARGIN_M = 0.01  # bin 근처가 아닌데 cube z가 이만큼 떨어지면 "일찍 떨어뜨림"


@dataclass(frozen=True)
class StepRecord:
    """rollout 1 physics-step에 대한, 평가에 필요한 최소 스냅샷.

    rollout 실행기(primary_replay_rollout.py/rollout_env.py)가 매 step마다 채워서 리스트로
    누적하고, 끝나면 ``evaluate_trajectory``에 그대로 넘긴다.
    """

    step: int
    sim_time: float
    ee_pos: Vec3  # gripperframe site, world frame (m)
    cube_pos: Vec3  # cube body, world frame (m)
    gripper_cmd_percent: float  # 이번 step에 실제로 mj_step에 적용된(safety-filtered) gripper 값
    gripper_raw_percent: float  # safety gate 이전, policy가 낸 원본 gripper 값
    cube_jaw_contact: bool  # 이번 step에 cube geom - jaw geom(named) contact가 있었는가
    safety_decision: str  # "ACCEPT" | "WOULD_CLAMP" | "REJECT"
    chunk_boundary: bool = False  # 이번 step에서 새 policy chunk를 새로 샘플링했는가 (진단용)
    # visual replay 전용(평가 로직은 쓰지 않음) - 이번 step 직후 로봇 6관절 qpos(deg/percent)와
    # cube 자유관절 orientation(wxyz). 저장해 두면 visual replay 도구가 물리를 다시 돌리지 않고
    # 이 값을 그대로 data.qpos에 적용하는 kinematic scrub만으로 정확히 같은 장면을 재생할 수 있다.
    qpos_deg: dict[str, float] | None = None
    cube_quat: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ReferenceZones:
    bin_center_xy: tuple[float, float]
    bin_inner_half: float
    table_z: float = 0.0


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _xy_dist(a: Vec3, b_xy: tuple[float, float]) -> float:
    return math.hypot(a[0] - b_xy[0], a[1] - b_xy[1])


# ---------------------------------------------------------------------------
# Track A - kinematic/semantic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KinematicResult:
    approach_success: bool
    approach_min_dist_m: float
    approach_min_dist_step: int | None

    grasp_pose_reached: bool
    grasp_pose_min_dist_m: float
    grasp_pose_step: int | None

    gripper_close_detected: bool
    gripper_close_step: int | None
    gripper_close_timing_ok: bool  # close 시점이 grasp_pose_step 근처(윈도우 내)인지
    gripper_close_timing_delta_steps: int | None  # close_step - grasp_pose_step (음수=너무 일찍)

    lift_success: bool
    lift_step: int | None
    lift_max_height_m: float  # cube 근처에서 관측된 EE 최대 상승량 (grasp_pose 이후 기준)

    carry_direction_ok: bool  # lift 이후 EE가 bin 쪽으로 순이동했는가
    carry_bin_dist_trend: float  # (lift 시점 EE-bin distance) - (최종 EE-bin distance); 양수=가까워짐

    bin_vicinity_reached: bool
    bin_vicinity_step: int | None
    ee_bin_min_dist_m: float

    gripper_release_detected: bool
    gripper_release_step: int | None
    release_timing_ok: bool  # release가 bin vicinity 근방에서 일어났는지

    kinematic_pick_drop_success: bool

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_kinematic(records: list[StepRecord], zones: ReferenceZones) -> KinematicResult:
    if not records:
        return KinematicResult(
            approach_success=False, approach_min_dist_m=float("inf"), approach_min_dist_step=None,
            grasp_pose_reached=False, grasp_pose_min_dist_m=float("inf"), grasp_pose_step=None,
            gripper_close_detected=False, gripper_close_step=None, gripper_close_timing_ok=False,
            gripper_close_timing_delta_steps=None,
            lift_success=False, lift_step=None, lift_max_height_m=0.0,
            carry_direction_ok=False, carry_bin_dist_trend=0.0,
            bin_vicinity_reached=False, bin_vicinity_step=None, ee_bin_min_dist_m=float("inf"),
            gripper_release_detected=False, gripper_release_step=None, release_timing_ok=False,
            kinematic_pick_drop_success=False,
        )

    # -- approach / grasp pose: EE-cube distance ------------------------------------------
    ee_cube_dists = [(_dist(r.ee_pos, r.cube_pos), r) for r in records]
    approach_min_dist, approach_rec = min(ee_cube_dists, key=lambda t: t[0])
    approach_success = approach_min_dist <= APPROACH_DIST_M
    grasp_pose_reached = approach_min_dist <= GRASP_POSE_DIST_M
    grasp_pose_step = approach_rec.step if grasp_pose_reached else None

    # -- gripper close timing (첫 번째로 open->closed 전이) -------------------------------
    gripper_close_step: int | None = None
    was_open = records[0].gripper_cmd_percent >= GRIPPER_OPEN_PERCENT
    for r in records[1:]:
        is_closed = r.gripper_cmd_percent <= GRIPPER_CLOSED_PERCENT
        if was_open and is_closed:
            gripper_close_step = r.step
            break
        if r.gripper_cmd_percent >= GRIPPER_OPEN_PERCENT:
            was_open = True
    gripper_close_detected = gripper_close_step is not None

    gripper_close_timing_ok = False
    gripper_close_timing_delta_steps: int | None = None
    if gripper_close_detected and grasp_pose_step is not None:
        gripper_close_timing_delta_steps = gripper_close_step - grasp_pose_step
        gripper_close_timing_ok = abs(gripper_close_timing_delta_steps) <= CLOSE_TIMING_WINDOW_STEPS

    # -- lift: EE z 상승, gripper close 이후 LIFT_WINDOW_STEPS 이내 --------------------------
    lift_success = False
    lift_step: int | None = None
    lift_max_height_m = 0.0
    if gripper_close_detected:
        base_rec = next((r for r in records if r.step == gripper_close_step), records[0])
        base_z = base_rec.ee_pos[2]
        for r in records:
            if r.step < gripper_close_step or r.step > gripper_close_step + LIFT_WINDOW_STEPS:
                continue
            height = r.ee_pos[2] - base_z
            lift_max_height_m = max(lift_max_height_m, height)
            if height >= LIFT_HEIGHT_M and not lift_success:
                lift_success = True
                lift_step = r.step

    # -- carry direction: lift 시점 이후 EE-bin distance가 줄어드는가 ------------------------
    carry_direction_ok = False
    carry_bin_dist_trend = 0.0
    if lift_success:
        post_lift = [r for r in records if r.step >= lift_step]
        if post_lift:
            start_dist = _xy_dist(post_lift[0].ee_pos, zones.bin_center_xy)
            end_dist = _xy_dist(post_lift[-1].ee_pos, zones.bin_center_xy)
            carry_bin_dist_trend = start_dist - end_dist
            carry_direction_ok = carry_bin_dist_trend > 0.0

    # -- bin vicinity ------------------------------------------------------------------------
    ee_bin_dists = [(_xy_dist(r.ee_pos, zones.bin_center_xy), r) for r in records]
    ee_bin_min_dist, bin_rec = min(ee_bin_dists, key=lambda t: t[0])
    bin_vicinity_reached = ee_bin_min_dist <= BIN_VICINITY_DIST_M
    bin_vicinity_step = bin_rec.step if bin_vicinity_reached else None

    # -- release timing: close 이후 첫 closed->open 전이, bin 근방에서 일어났는지 -------------
    gripper_release_step: int | None = None
    if gripper_close_detected:
        was_closed = True
        for r in records:
            if r.step <= gripper_close_step:
                continue
            is_open = r.gripper_cmd_percent >= GRIPPER_OPEN_PERCENT
            if was_closed and is_open:
                gripper_release_step = r.step
                break
            was_closed = r.gripper_cmd_percent <= GRIPPER_CLOSED_PERCENT
    gripper_release_detected = gripper_release_step is not None

    release_timing_ok = False
    if gripper_release_detected:
        release_rec = next(r for r in records if r.step == gripper_release_step)
        release_timing_ok = _xy_dist(release_rec.ee_pos, zones.bin_center_xy) <= BIN_VICINITY_DIST_M * 1.5

    kinematic_pick_drop_success = (
        approach_success
        and grasp_pose_reached
        and gripper_close_detected
        and lift_success
        and carry_direction_ok
        and bin_vicinity_reached
        and gripper_release_detected
        and release_timing_ok
    )

    return KinematicResult(
        approach_success=approach_success, approach_min_dist_m=approach_min_dist,
        approach_min_dist_step=approach_rec.step,
        grasp_pose_reached=grasp_pose_reached, grasp_pose_min_dist_m=approach_min_dist,
        grasp_pose_step=grasp_pose_step,
        gripper_close_detected=gripper_close_detected, gripper_close_step=gripper_close_step,
        gripper_close_timing_ok=gripper_close_timing_ok,
        gripper_close_timing_delta_steps=gripper_close_timing_delta_steps,
        lift_success=lift_success, lift_step=lift_step, lift_max_height_m=lift_max_height_m,
        carry_direction_ok=carry_direction_ok, carry_bin_dist_trend=carry_bin_dist_trend,
        bin_vicinity_reached=bin_vicinity_reached, bin_vicinity_step=bin_vicinity_step,
        ee_bin_min_dist_m=ee_bin_min_dist,
        gripper_release_detected=gripper_release_detected, gripper_release_step=gripper_release_step,
        release_timing_ok=release_timing_ok,
        kinematic_pick_drop_success=kinematic_pick_drop_success,
    )


# ---------------------------------------------------------------------------
# Track B - physics (contact-based)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsResult:
    grasp_contact_detected: bool
    grasp_contact_step: int | None
    cube_secured: bool  # 연속 접촉이 SECURED_MIN_CONSECUTIVE_CONTACT_STEPS 이상 유지됐는가
    lifted: bool
    lifted_step: int | None
    carried: bool
    released: bool  # secured 이후 접촉이 끊긴 시점이 있는가
    released_step: int | None
    dropped_early: bool  # bin 근처가 아닌데 접촉이 끊기고 cube가 테이블에 남았는가
    final_cube_xy: tuple[float, float]
    final_in_bin: bool
    physics_pick_drop_success: bool

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_physics(records: list[StepRecord], zones: ReferenceZones) -> PhysicsResult:
    if not records:
        return PhysicsResult(
            grasp_contact_detected=False, grasp_contact_step=None, cube_secured=False,
            lifted=False, lifted_step=None, carried=False, released=False, released_step=None,
            dropped_early=False, final_cube_xy=(0.0, 0.0), final_in_bin=False,
            physics_pick_drop_success=False,
        )

    grasp_contact_step: int | None = None
    consecutive = 0
    cube_secured = False
    secured_since_step: int | None = None
    for r in records:
        if r.cube_jaw_contact:
            consecutive += 1
            if grasp_contact_step is None:
                grasp_contact_step = r.step
            if consecutive >= SECURED_MIN_CONSECUTIVE_CONTACT_STEPS and not cube_secured:
                cube_secured = True
                secured_since_step = r.step
        else:
            consecutive = 0
    grasp_contact_detected = grasp_contact_step is not None

    lifted = False
    lifted_step: int | None = None
    if cube_secured:
        base_z = next(r.cube_pos[2] for r in records if r.step == secured_since_step)
        for r in records:
            if r.step < secured_since_step:
                continue
            if r.cube_jaw_contact and (r.cube_pos[2] - base_z) >= LIFT_HEIGHT_M:
                lifted = True
                lifted_step = r.step
                break

    carried = False
    if lifted:
        post = [r for r in records if r.step >= lifted_step]
        if len(post) >= 2:
            start_dist = _xy_dist(post[0].cube_pos, zones.bin_center_xy)
            end_dist = _xy_dist(post[-1].cube_pos, zones.bin_center_xy)
            carried = end_dist < start_dist

    released_step: int | None = None
    if cube_secured:
        was_contact = True
        for r in records:
            if secured_since_step is None or r.step <= secured_since_step:
                continue
            if was_contact and not r.cube_jaw_contact:
                released_step = r.step
                break
            was_contact = r.cube_jaw_contact
    released = released_step is not None

    dropped_early = False
    if released:
        release_rec = next(r for r in records if r.step == released_step)
        near_bin = _xy_dist(release_rec.cube_pos, zones.bin_center_xy) <= zones.bin_inner_half * 2.0
        still_elevated_before = release_rec.cube_pos[2] > zones.table_z + LIFT_HEIGHT_M * 0.5
        dropped_early = (not near_bin) and lifted and still_elevated_before is False

    final = records[-1]
    final_cube_xy = (final.cube_pos[0], final.cube_pos[1])
    final_in_bin = (
        abs(final_cube_xy[0] - zones.bin_center_xy[0]) <= zones.bin_inner_half
        and abs(final_cube_xy[1] - zones.bin_center_xy[1]) <= zones.bin_inner_half
    )

    physics_pick_drop_success = lifted and carried and released and final_in_bin

    return PhysicsResult(
        grasp_contact_detected=grasp_contact_detected, grasp_contact_step=grasp_contact_step,
        cube_secured=cube_secured, lifted=lifted, lifted_step=lifted_step, carried=carried,
        released=released, released_step=released_step, dropped_early=dropped_early,
        final_cube_xy=final_cube_xy, final_in_bin=final_in_bin,
        physics_pick_drop_success=physics_pick_drop_success,
    )


# ---------------------------------------------------------------------------
# Trajectory quality (safety-adjacent, track-independent)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryQuality:
    length_steps: int
    mean_abs_jerk_deg: float | None  # gripper 제외 5관절 명령의 2차 차분 절대값 평균 (proxy)
    max_single_step_delta_deg: float | None
    safety_accept_count: int
    safety_would_clamp_count: int
    safety_reject_count: int
    clamp_free: bool  # 전체 rollout에서 WOULD_CLAMP/REJECT가 한 번도 없었는가

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_trajectory_quality(
    per_step_command_deg: list[dict[str, float]], safety_decisions: list[str]
) -> TrajectoryQuality:
    """``per_step_command_deg``: 매 step ``mj_step``에 실제 적용된(safety-filtered) 관절 명령
    (6개 키, JOINT_ORDER)의 리스트. jerk proxy는 gripper를 제외한 5개 몸통 관절만 본다 -
    gripper는 이산적인 open/close 명령이 정상적으로 큰 변화를 만들어 jerk 지표를 왜곡하기
    때문이다."""
    accept = safety_decisions.count("ACCEPT")
    clamp = safety_decisions.count("WOULD_CLAMP")
    reject = safety_decisions.count("REJECT")

    if len(per_step_command_deg) < 3:
        return TrajectoryQuality(
            length_steps=len(per_step_command_deg), mean_abs_jerk_deg=None,
            max_single_step_delta_deg=None, safety_accept_count=accept,
            safety_would_clamp_count=clamp, safety_reject_count=reject, clamp_free=(clamp == 0 and reject == 0),
        )

    body_joints = [j for j in per_step_command_deg[0].keys() if j != "gripper"]
    deltas: list[float] = []
    jerks: list[float] = []
    for i in range(1, len(per_step_command_deg)):
        for j in body_joints:
            deltas.append(abs(per_step_command_deg[i][j] - per_step_command_deg[i - 1][j]))
    for i in range(2, len(per_step_command_deg)):
        for j in body_joints:
            d1 = per_step_command_deg[i][j] - per_step_command_deg[i - 1][j]
            d0 = per_step_command_deg[i - 1][j] - per_step_command_deg[i - 2][j]
            jerks.append(abs(d1 - d0))

    return TrajectoryQuality(
        length_steps=len(per_step_command_deg),
        mean_abs_jerk_deg=(sum(jerks) / len(jerks)) if jerks else None,
        max_single_step_delta_deg=max(deltas) if deltas else None,
        safety_accept_count=accept, safety_would_clamp_count=clamp, safety_reject_count=reject,
        clamp_free=(clamp == 0 and reject == 0),
    )


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

FAILURE_SAFETY_REJECT = "safety_reject"
FAILURE_FAILED_APPROACH = "failed_approach"
FAILURE_MISSED_GRASP = "missed_grasp"
FAILURE_FAILED_LIFT = "failed_lift"
FAILURE_DROPPED_EARLY = "dropped_early"
FAILURE_WRONG_DIRECTION = "wrong_direction"
FAILURE_FAILED_BIN_REACH = "failed_bin_reach"
FAILURE_FAILED_RELEASE = "failed_release"
FAILURE_TIMEOUT = "timeout"
FAILURE_SIM_CONTACT_ARTIFACT = "simulation_contact_artifact"
FAILURE_OTHER = "other"
FAILURE_NONE = "none"

TRACK_KINEMATIC = "kinematic"
TRACK_PHYSICS = "physics"

CAUSE_POLICY_TRAJECTORY = "policy_trajectory"
CAUSE_SIM_PHYSICS_ARTIFACT = "sim_physics_artifact"
CAUSE_SAFETY_REJECT = "safety_reject"
CAUSE_TIMEOUT = "timeout"
CAUSE_OTHER = "other"
CAUSE_NONE = "none"


@dataclass(frozen=True)
class FailureClassification:
    reason: str
    track: str  # "kinematic" | "physics" | "none"
    likely_cause: str

    def to_dict(self) -> dict:
        return {"reason": self.reason, "track": self.track, "likely_cause": self.likely_cause}


def classify_failure(
    kinematic: KinematicResult,
    physics: PhysicsResult,
    *,
    ended_by_safety_reject: bool,
    ended_by_timeout: bool,
) -> FailureClassification:
    """정책 궤적 실패(policy_trajectory)와 MuJoCo contact/physics artifact(sim_physics_artifact)를
    분리한다 - kinematic이 성공인데 physics만 실패하면 항상 후자로 분류한다 (요구사항)."""

    if ended_by_safety_reject:
        return FailureClassification(FAILURE_SAFETY_REJECT, TRACK_KINEMATIC, CAUSE_SAFETY_REJECT)

    if kinematic.kinematic_pick_drop_success:
        if physics.physics_pick_drop_success:
            return FailureClassification(FAILURE_NONE, TRACK_KINEMATIC, CAUSE_NONE)
        # kinematic 궤적은 올바른데 physics만 실패 -> 정책 잘못이 아니라 시뮬레이션 물리 한계.
        return FailureClassification(
            FAILURE_DROPPED_EARLY if physics.grasp_contact_detected and not physics.final_in_bin
            else FAILURE_MISSED_GRASP,
            TRACK_PHYSICS, CAUSE_SIM_PHYSICS_ARTIFACT,
        )

    if not kinematic.approach_success:
        cause = CAUSE_TIMEOUT if ended_by_timeout else CAUSE_POLICY_TRAJECTORY
        return FailureClassification(FAILURE_FAILED_APPROACH, TRACK_KINEMATIC, cause)

    if not kinematic.grasp_pose_reached or not kinematic.gripper_close_detected:
        return FailureClassification(FAILURE_MISSED_GRASP, TRACK_KINEMATIC, CAUSE_POLICY_TRAJECTORY)

    if not kinematic.lift_success:
        # 그립 자세엔 도달했고 gripper도 닫았는데 못 들었다 - kinematic 관점에서는 EE가 안
        # 올라간 것이므로 policy 궤적 문제. (물리적으로 못 쥐어서 못 든 경우는 physics가 이미
        # kinematic_pick_drop_success 갈래에서 걸러진다 - kinematic은 EE 궤적만 보므로 cube를
        # 실제로 쥐었는지와 무관하게 lift_success를 독립적으로 판정한다.)
        return FailureClassification(FAILURE_FAILED_LIFT, TRACK_KINEMATIC, CAUSE_POLICY_TRAJECTORY)

    if not kinematic.carry_direction_ok:
        return FailureClassification(FAILURE_WRONG_DIRECTION, TRACK_KINEMATIC, CAUSE_POLICY_TRAJECTORY)

    if not kinematic.bin_vicinity_reached:
        return FailureClassification(FAILURE_FAILED_BIN_REACH, TRACK_KINEMATIC, CAUSE_POLICY_TRAJECTORY)

    if not kinematic.gripper_release_detected or not kinematic.release_timing_ok:
        return FailureClassification(FAILURE_FAILED_RELEASE, TRACK_KINEMATIC, CAUSE_POLICY_TRAJECTORY)

    if ended_by_timeout:
        return FailureClassification(FAILURE_TIMEOUT, TRACK_KINEMATIC, CAUSE_TIMEOUT)

    return FailureClassification(FAILURE_OTHER, TRACK_KINEMATIC, CAUSE_OTHER)


# ---------------------------------------------------------------------------
# Top-level combined result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PickDropEvalResult:
    kinematic: KinematicResult
    physics: PhysicsResult
    trajectory_quality: TrajectoryQuality
    failure: FailureClassification

    def to_dict(self) -> dict:
        return {
            "kinematic": self.kinematic.to_dict(),
            "physics": self.physics.to_dict(),
            "trajectory_quality": self.trajectory_quality.to_dict(),
            "failure": self.failure.to_dict(),
        }


def evaluate_trajectory(
    records: list[StepRecord],
    zones: ReferenceZones,
    *,
    per_step_command_deg: list[dict[str, float]],
    ended_by_safety_reject: bool,
    ended_by_timeout: bool,
) -> PickDropEvalResult:
    kinematic = evaluate_kinematic(records, zones)
    physics = evaluate_physics(records, zones)
    quality = evaluate_trajectory_quality(per_step_command_deg, [r.safety_decision for r in records])
    failure = classify_failure(
        kinematic, physics, ended_by_safety_reject=ended_by_safety_reject, ended_by_timeout=ended_by_timeout
    )
    return PickDropEvalResult(kinematic=kinematic, physics=physics, trajectory_quality=quality, failure=failure)


def current_stage_label(records: list[StepRecord]) -> str:
    """Visual mode console 표시용 - 지금까지의 prefix만 보고 대략적인 단계 라벨을 반환한다.

    O(n) 재계산이라 매 step 부르면 전체적으로 O(n^2)이지만, rollout이 최대 수백 step이라
    무시할 만하다 (headless benchmark 경로에서는 쓰지 않는다 - 그쪽은 끝에 한 번만
    ``evaluate_trajectory``를 부른다)."""
    if not records:
        return "start"
    k = evaluate_kinematic(records, ReferenceZones(bin_center_xy=(records[-1].cube_pos[0], records[-1].cube_pos[1]), bin_inner_half=0.05))
    if k.gripper_release_detected:
        return "release"
    if k.bin_vicinity_reached:
        return "reach_bin"
    if k.lift_success:
        return "carry"
    if k.gripper_close_detected:
        return "lift" if k.lift_success else "grasp"
    if k.grasp_pose_reached:
        return "pre_grasp"
    if k.approach_success:
        return "approach"
    return "start"
