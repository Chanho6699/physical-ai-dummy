# MuJoCo full-rollout candidate comparison v1

**Candidate A**: V2+V3 reweight2:1 @10000 (accuracy-oriented) —
`outputs/reweight_ablation/combined65_reweight_new2_old1_v1/checkpoints/010000/pretrained_model`
**Candidate B**: V3+V4 uniform @10000 (safety-oriented) —
`outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/010000/pretrained_model`

Scale run so far: **smoke test, 10 scenes × 3 seeds × 2 candidates × 2 tracks = 120 rollouts**
(the request's own instruction: run the 10×3 smoke test first, expand only if compute allows).
Total wall time 6091s (~101.5min). **0 crashes. `real_follower_write_count` summed over all 120
rollouts = 0.** Disk preflight: 807.5GB free before the run, this report's entire output is 72MB —
no cleanup was needed or performed. See "Should this expand to 5 seeds?" at the end for why this
wasn't scaled further.

Every number below comes from `rollout_results.csv` (120 rows) unless noted. Full per-rollout
detail, per-joint clamp breakdown, and this same analysis in data form are in `summary_detailed.md`,
`candidate_comparison.csv`, `per_scene_summary.csv`, `failure_reasons.csv`, `safety_metrics.csv`.

## 0. How to read this report (read before the numbers)

This benchmark has two tracks with **very different evidentiary weight**, per explicit instruction:

- **Primary (real-observation replay, main comparison)**: SmolVLA is driven by **real recorded
  camera images + real recorded state** from 10 held-out episodes of
  `data/so101_cube_xy_midpoint_test10_v2_clean` (confirmed excluded from both candidates' training
  data — see `docs/mujoco_scene_to_so101_semantics.md` §amendment). The resulting action chunks are
  physically executed in MuJoCo. There is **no visual domain gap** in this track — the only
  synthetic element is the reference cube/bin position used to score direction/approach (see §1).
- **Secondary/exploratory (synthetic closed-loop)**: MuJoCo-rendered `workspace_cam`/`wrist_cam`
  images are fed back into SmolVLA every step, true closed loop. **Do not equate this with real
  policy quality** — the rendered images look nothing like the real camera feed SmolVLA trained on.

**Every `kinematic_pick_drop_success` and `physics_pick_drop_success` number below is 0% for both
candidates on both tracks.** This is a real, consistent measurement, not a bug (see §7) — but given
why (§1, §7), it must not be read as "neither model can do anything." The rest of this report uses
continuous/graded metrics (distances, per-step safety rates, per-joint clamp rates, partial-stage
rates) to actually compare A vs B, which is where the real signal is.

## 1. A known limitation that shapes every result below

The 10 cube/bin positions are **not** the real object position for any of these episodes — no real
cube/bin coordinate exists anywhere in this repository for any past experiment (verified by
full-repo grep before any code was written). They are a synthetic, FK-reachability-verified
reference grid (`docs/mujoco_scene_to_so101_semantics.md`), reused identically for both candidates.
Because Primary track's action chunks are generated from the *real* demonstration's images (which
show the *real*, unrecorded cube position — not our reference cube), the EE trajectory is not
actually aiming at our reference point. This is almost certainly why `bin_vicinity_reached`,
`carry_direction_ok`, and `release_timing_ok` are **0% for every group, including both candidates**
(§ per-stage table below) — it reflects the benchmark's reference-zone design, not a model failure.
It affects both candidates identically, so **relative** comparisons (A vs B) below remain valid;
**absolute** "did it complete the task" numbers do not.

## 2. Headline comparison table

| track | candidate | n | approach | grasp pose | lift (kinematic) | bin vicinity | safety reject/rollout | per-step ACCEPT | per-step WOULD_CLAMP | mean approach dist | mean jerk proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | **A** | 30 | 33% | 13% | 63% | 0% | 13% | **35.6%** | **64.4%** | 0.101m | 1.29deg |
| primary | **B** | 30 | 37% | 20% | 73% | 0% | 3% | **55.2%** | **44.8%** | 0.099m | 1.43deg |
| secondary | **A** | 30 | 27% | 10% | 7% | 0% | 0% | **29.2%** | **70.8%** | 0.099m | 0.88deg |
| secondary | **B** | 30 | 27% | 10% | 10% | 0% | 0% | **84.3%** | **15.7%** | 0.092m | 0.95deg |

("per-step ACCEPT/WOULD_CLAMP" = fraction of individual physics steps, pooled across all 30
rollouts per row — ~10,000-12,000 steps per row, not the ~30 per-rollout binary "clamp-free" figure,
which is ~0% for every group simply because a single WOULD_CLAMP anywhere in a 350-400-step rollout
makes the whole rollout "not clamp-free" — not a useful statistic at this rollout length, so this
report uses the per-step rate instead, same convention the historical offline-eval reports used.)

## 3. Per-joint clamp rate (recomputed from saved raw vs safety-filtered trajectories)

| track | candidate | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---:|---:|---:|---:|---:|---:|
| primary | A | 11.9% | **40.7%** | **25.4%** | 9.3% | 0.0% | 9.9% |
| primary | B | 8.4% | **28.4%** | **13.3%** | 1.3% | 0.2% | 5.6% |
| secondary | A | 1.9% | 12.8% | 14.4% | **51.9%** | 0.0% | 1.9% |
| secondary | B | 0.5% | 10.0% | 7.0% | **1.0%** | 0.0% | 0.3% |

Candidate A clamps more than B on shoulder_lift and elbow_flex in **both** tracks — the same two
joints, same direction, as the historical first-action diagnostic (A: shoulder_lift 25%/elbow_flex
60% clamp vs B: 10%/30%, `reports/combined65_reweight_new2_old1_v1/summary.md`). The magnitudes
differ (different measurement: single first-action vs full physical trajectory) but the **direction
replicates independently** — see §8 Q6.

A striking secondary-track-only finding: candidate A's **wrist_flex** clamps 51.9% of steps under
synthetic closed-loop vs only 1.0% for B — far higher than any other joint/track combination for
either candidate. This coincides with the pre-existing, independently-documented wrist_flex
range/calibration mismatch (`docs/wrist_flex_range_mismatch_investigation.md`). This report does not
claim they are the same phenomenon (that would need dedicated investigation), but flags the overlap.

## 4. Kinematic vs physics: is MuJoCo grasp physics distorting the picture?

| track | candidate | n | grasp pose reached (kin) | gripper closed (kin) | lift (kin, EE rose) | **grasp contact (phys)** | **cube secured (phys)** | **cube actually lifted (phys)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| primary | A | 30 | 13% | 63% | 63% | 7% | **0%** | **0%** |
| primary | B | 30 | 20% | 73% | 73% | 3% | **0%** | **0%** |
| secondary | A | 30 | 10% | 10% | 7% | 0% | **0%** | **0%** |
| secondary | B | 30 | 10% | 10% | 10% | 0% | **0%** | **0%** |

The arm closes its gripper and raises the EE (kinematically) in a majority of primary-track
rollouts (63-73%), but the cube is never once actually *secured* by contact physics (0% in every
group, primary or secondary, either candidate). Given `cube_geom` never even briefly registers
sustained contact even when the gripper closes right where the reference cube sits, this is strong
evidence that **MuJoCo's un-tuned grasp-contact model** (already flagged as a limitation before any
object existed in this scene: `docs/mujoco_action_replay.md` §11.4-5 — untuned contact thresholds,
unnamed collision geoms, no prior grasp-contact statistics) is the dominant reason `physics_*`
metrics read near-zero — **not** that the policy never attempts a grasp. This directly motivated
keeping kinematic (Track A) as the primary metric and physics (Track B) as secondary, per
instruction.

## 5. Representative trajectories (videos + replayable JSON)

`videos/manifest.json` lists all of them; every video was rendered from the *exact* saved joint
trajectory (`qpos_deg` per step, no re-simulation) via `scripts/render_mujoco_rollout_video.py`.
Because no rollout achieved `kinematic_pick_drop_success`, "closest near miss" (smallest
EE-to-reference-cube distance) is shown in place of "success" — files are named accordingly
(`*_near_miss_*.mp4`), never mislabeled as `*_success_*` unless a real success exists.

To watch (or re-watch) any of them interactively in the MuJoCo viewer instead of as an MP4:

```bash
source ~/lerobot/.venv/bin/activate
python scripts/run_mujoco_full_rollout_visual.py \
  --replay reports/mujoco_full_rollout_candidate_comparison_v1/trajectories/<candidate>_<track>_<scene_id>_seed<seed>.json \
  --mode web --port 8090
# then open http://<this machine>:8090/ in a browser
```

## 6. Exact visual-mode commands (live, not replay)

```bash
source ~/lerobot/.venv/bin/activate

# Web viewer (recommended in this environment - see §9 for why)
python scripts/run_mujoco_full_rollout_visual.py \
  --candidate A --track primary --scene mujoco_rollout_test01 --seed 0 --mode web --port 8090
# open http://<machine>:8090/ - shows scene id/seed/stage/safety live, console prints the same

# Native MuJoCo GLFW window (works, but has a documented ~30-50% segfault-on-exit risk in this
# WSLg environment - data is already saved before that point, see docs/mujoco_action_replay.md §11.8)
python scripts/run_mujoco_full_rollout_visual.py \
  --candidate B --track secondary --scene mujoco_rollout_test05 --seed 1 --mode native
```

Headless benchmark (this report was produced by):

```bash
python scripts/run_mujoco_full_rollout_benchmark.py \
  --candidates A,B --tracks primary,secondary --scenes 10 --seeds 0,1,2 --max-steps 400 \
  --out-dir reports/mujoco_full_rollout_candidate_comparison_v1
```

## 7. Failure reason distribution

| track | candidate | failed_approach | missed_grasp | wrong_direction | safety_reject |
|---|---|---:|---:|---:|---:|
| primary | A | 18 | 4 | 4 | 4 |
| primary | B | 18 | 5 | 6 | 1 |
| secondary | A | 22 | 8 | 0 | 0 |
| secondary | B | 22 | 8 | 0 | 0 |

`dropped_early`/`sim_physics_artifact`-classified failures never appear: `classify_failure()`
(`simulation/mujoco/pick_drop_eval.py`) only reaches that branch once full kinematic success is
already achieved (per its own logic — kinematic success has to happen first before physics can be
blamed for what happens *after*), and no rollout got that far (§1). This is a real limitation of the
failure-taxonomy as it stood up against this particular benchmark's outcome distribution, not a bug
— §4's kinematic-vs-physics table is the correct way to see the same "policy vs sim artifact"
distinction in this dataset instead.

`safety_reject` count differs 4:1 (A) vs 1 (B) in primary track — consistent with §2/§3.

## 8. Answers to the 9 required questions

**1. 어느 모델이 full rollout success가 높은가?**
Neither — 0% for both candidates, both tracks, by the strict binary criteria (§1 explains why this
number isn't informative at this benchmark's reference-zone design). No winner can be declared on
this specific metric from this data.

**2. 어느 모델이 더 안전한가?**
**Candidate B, clearly and consistently.** Per-step ACCEPT rate: primary 55.2%(B) vs 35.6%(A);
secondary 84.3%(B) vs 29.2%(A) — an especially large gap in true closed-loop. Per-rollout safety
REJECT: primary 3%(B) vs 13%(A). Per-joint: B clamps less on shoulder_lift/elbow_flex in both
tracks (§3). This independently reproduces, via a completely different methodology (physical MuJoCo
execution vs offline dataset MAE), the historical finding that B is the safety-oriented checkpoint.

**3. 어느 모델이 grasp/lift까지 더 잘 가는가?**
By kinematic indicators, B is marginally ahead in primary track: grasp_pose_reached 20%(B) vs
13%(A), gripper closed/lift 73%(B) vs 63%(A). In secondary track the two are statistically tied
(10% vs 10% grasp pose, 10% vs 7% lift). Physics-verified grasp/lift is 0% for both everywhere
(§4) — not informative for ranking.

**4. 어느 모델이 bin까지 더 잘 운반하는가?**
Cannot be answered from this data — `bin_vicinity_reached`/`carry_direction_ok` are 0% for every
group (§1). This benchmark's reference bin position, not either policy, is the limiting factor here.
A follow-up with a bin position matched more closely to real demonstrated targets (not
reconstructable from this repo's data, per the original investigation) would be needed.

**5. offline MAE와 실제 rollout success의 관계는?**
**Offline MAE and this benchmark's safety/trajectory-quality outcomes point in opposite
directions.** Candidate A has the *better* offline historical MAE (3.53 vs B's 3.78) but is
**clearly less safe** in actual physical execution here (§2). Better offline action-prediction
accuracy did not translate into safer or more clamp-free physical trajectories — if anything the
opposite, in this benchmark. Neither candidate's offline MAE predicted its (near-zero, but
approximately-tied) kinematic task-stage progress either.

**6. first-action clamp-free와 full trajectory safety의 관계는?**
**They agree directionally, and this is the clearest positive finding in this report.** Historical
first-action clamp-free rate: A=40%, B=65% (single reference observation, 20 seeds). This
benchmark's full-trajectory per-step ACCEPT rate: A=29-36%, B=55-84% across both tracks — same
ranking, same direction, replicated independently across ~10,000+ physical steps per group rather
than 20 single-step samples. The first-action diagnostic generalizes.

**7. MuJoCo physics가 결과를 왜곡하는 증거가 있는가?**
**Yes, clearly, on two separate axes:**
- *Contact/grasp physics*: §4 — kinematic "closed gripper + lifted EE" happens in up to 73% of
  primary rollouts, but physics-verified secured contact is 0% everywhere. This is the untuned
  contact model (already flagged before any object existed in this scene), not policy failure.
- *Chunk-resync mechanic (primary track only)*: primary track's mean max-single-step-delta
  (18.6-21.4deg) is far larger than secondary's (6.1-6.1deg) for both candidates — structurally
  expected, since primary resyncs the simulated arm to the *real* recorded position at every chunk
  boundary (a deliberate, documented design choice, §amendment in the plan), which can itself
  produce a large one-off jump unrelated to the smoothness of the policy's own within-chunk
  predictions. Some of primary track's WOULD_CLAMP events are attributable to this resync
  mechanic rather than raw policy volatility — secondary track's per-step clamp rate is the cleaner
  read of "how volatile is this policy's own moment-to-moment output."
- *Visual domain gap (secondary track only, §0)*: MuJoCo-rendered images look nothing like the real
  training camera feed — secondary track results should be read as "how the policy responds to an
  unfamiliar image distribution," not real-world prediction.

**8. 제한적 실물 single-step test로 넘길 모델은 무엇인가?**
**Candidate B.** It is safer by every safety metric measured here (per-step ACCEPT/REJECT, per-joint
clamp rates, in both tracks), replicates the historical safety-oriented characterization
independently, and is statistically tied or marginally ahead of A on the (limited, near-zero)
kinematic grasp/lift indicators available. Nothing in this benchmark favors A on any axis except
offline MAE, which (per Q5) did not translate into better physical-execution outcomes here.

**9. 아직 실물 테스트를 막아야 할 명확한 이유가 있는가?**
Not from safety data — B's REJECT rate is low (3% of rollouts, primary track) and physics never
shows a dangerous excursion (no divergence/NaN in 120 rollouts). But this benchmark's fundamental
inability to demonstrate `bin_vicinity_reached`/`carry_direction_ok` for **either** candidate (§1)
means it **cannot positively confirm** the back half of the task (carry + release) works at all —
that absence of evidence is a reason for caution (start any real test at pick/approach only, not a
full autonomous pick-and-drop), not a reason to block a limited single-step test outright.

## 9. Should this expand to 5 seeds?

The per-step safety comparison (§2/§3) already pools ~10,000-24,000 individual physics-step
observations per candidate/track — the gap between candidates (e.g. 84.3% vs 29.2% ACCEPT in
secondary track) is far too large to be seed noise. Expanding seeds would tighten confidence on the
*per-rollout* (n=30) stage-progress numbers (§2 columns other than the per-step safety ones), but is
unlikely to change any of the 9 answers above. Given the compute already spent (~101.5min for this
smoke run; secondary track alone costs ~97-100s/rollout due to un-avoidable 2-camera rendering cost
in this GPU-render-constrained environment — see `docs/mujoco_scene_to_so101_semantics.md`), this
report did not scale to 5 seeds. To do so:

```bash
python scripts/run_mujoco_full_rollout_benchmark.py \
  --candidates A,B --tracks primary,secondary --scenes 10 --seeds 0,1,2,3,4 --max-steps 400 \
  --out-dir reports/mujoco_full_rollout_candidate_comparison_v1_5seed
```
