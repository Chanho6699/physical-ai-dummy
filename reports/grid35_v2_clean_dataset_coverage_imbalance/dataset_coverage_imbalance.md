# Grid35 V2-clean: training-target distribution & start-segment coverage audit

Dataset: `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_grid35_v2_clean` - 35 episodes, 11505 frames total, 30fps.

## 1. Frame-count coverage

| segment | n_frames | % of dataset |
|---|---:|---:|
| frames[0:15] | 525 | 4.6% |
| frames[0:30] | 1050 | 9.1% |
| static (data-driven, movement detector) | 875 | 7.6% |
| moving (data-driven) | 10630 | 92.4% |
| full episode | 11505 | 100.0% |

Median/mean static-segment length: 25.0 / 25.0 frames (~0.83s / ~0.83s).

## 2. Per-joint action(t)-state(t) distribution by segment (shoulder_lift / elbow_flex)

| segment | joint | n | mean\|d\| | median\|d\| | p95\|d\| | max\|d\| |
|---|---|---:|---:|---:|---:|---:|
| all_frames | shoulder_lift | 11505 | 2.858 | 2.374 | 8.264 | 13.187 |
| all_frames | elbow_flex | 11505 | 3.710 | 2.857 | 7.956 | 10.769 |
| frames_0_15 | shoulder_lift | 525 | 0.388 | 0.352 | 0.615 | 4.220 |
| frames_0_15 | elbow_flex | 525 | 1.933 | 1.978 | 2.418 | 4.264 |
| frames_0_30 | shoulder_lift | 1050 | 1.198 | 0.440 | 4.484 | 6.418 |
| frames_0_30 | elbow_flex | 1050 | 2.771 | 2.154 | 5.719 | 6.989 |
| static_segment_data_driven | shoulder_lift | 875 | 0.614 | 0.440 | 2.462 | 4.659 |
| static_segment_data_driven | elbow_flex | 875 | 2.245 | 2.066 | 4.000 | 6.022 |
| moving_segment_data_driven | shoulder_lift | 10630 | 3.043 | 2.637 | 8.440 | 13.187 |
| moving_segment_data_driven | elbow_flex | 10630 | 3.831 | 3.033 | 8.044 | 10.769 |

## 3. Safety Gate WOULD_CLAMP/REJECT threshold coverage (read-only)

WOULD_CLAMP: shoulder_lift=5.16deg, elbow_flex=5.73deg. REJECT (x5): shoulder_lift=25.80deg, elbow_flex=28.65deg.

| segment | joint | % frames > WOULD_CLAMP | % frames > REJECT |
|---|---|---:|---:|
| all_frames | shoulder_lift | 14.8% | 0.0% |
| all_frames | elbow_flex | 24.6% | 0.0% |
| static_segment_data_driven | shoulder_lift | 0.0% | 0.0% |
| static_segment_data_driven | elbow_flex | 0.1% | 0.0% |
| moving_segment_data_driven | shoulder_lift | 16.0% | 0.0% |
| moving_segment_data_driven | elbow_flex | 26.6% | 0.0% |

## 4. Whole-target-chunk mechanism check (static observations only)

chunk_size=50, n_static_frames_total=875

| joint | chunk[0]\|delta\| mean/median/p95 | full-chunk mean\|delta\| mean/median/p95 | full-chunk max\|delta\| mean/median/p95 | % static frames where chunk-mean > WOULD_CLAMP |
|---|---|---|---|---:|
| shoulder_lift | 0.614/0.440/2.462 | 20.503/19.981/34.056 | 48.593/48.088/67.807 | 99.7% |
| elbow_flex | 2.245/2.066/4.000 | 20.975/20.258/34.881 | 52.171/52.527/73.477 | 99.7% |

Interpretation: chunk[0] for a static observation is small (matches the previously-verified training-target alignment - offset is exactly 0, see `reports/smolvla_training_target_alignment/`), but the *rest* of that same observation's 50-step target chunk (chunk[1..49], covering up to 1.67s forward) is frequently large - the model is trained to jointly predict all 50 steps from one static-looking observation, so its output distribution for that observation is shaped by the *whole* chunk's target statistics, not just position 0.

## 5. Start-pose diversity across 35 episodes

| joint | start(frame0) std | start(frame0) range | mid-reach(60%) std | mid-reach(60%) range |
|---|---:|---:|---:|---:|
| shoulder_lift | 1.19 | 5.54 | 21.72 | 80.18 |
| elbow_flex | 0.32 | 1.76 | 18.43 | 76.92 |
| wrist_roll | 2.18 | 5.36 | 2.16 | 5.36 |
| gripper | 0.68 | 2.90 | 1.93 | 8.35 |

Pairwise L2 distance (degrees, over all 6 joints) among the 35 episodes' states:

| basis | min | median | mean | max |
|---|---:|---:|---:|---:|
| start frame 0 | 0.51 | 3.85 | 4.13 | 11.75 |
| start mean frames[0:5) | 0.51 | 3.85 | 4.13 | 11.75 |
| mid-reach (60% of episode) | 8.90 | 41.06 | 43.30 | 102.41 |

> mid_reach_frame is each episode's own 60%-of-length frame (where Grid35's 35 distinct XY cube placements should have already differentiated the arm's reach pose) - compared against the frame-0/frame[0:5) start pose to quantify how much smaller start-pose diversity is relative to genuine task-driven diversity later in the same episodes.

## 6. Large shoulder_lift/elbow_flex action frequency

| segment | joint | >= 1.0deg | >= 2.0deg | >= 3.0deg | >= 5.0deg | >= 7.0deg | >= 10.0deg |
|---|---|---:|---:|---:|---:|---:|---:|
| all_frames | shoulder_lift | 68.4% | 55.8% | 41.3% | 15.9% | 8.0% | 1.6% |
| all_frames | elbow_flex | 87.5% | 74.1% | 47.5% | 30.4% | 13.2% | 0.2% |
| moving_segment_data_driven | shoulder_lift | 73.1% | 59.8% | 44.4% | 17.2% | 8.7% | 1.7% |
| moving_segment_data_driven | elbow_flex | 86.7% | 75.6% | 50.4% | 32.8% | 14.2% | 0.2% |

## 7. Loss-contribution proxy (first-order approximation - see caveat)

(observation, chunk-position) pair counts: static-segment=43750, moving-segment=531500 (7.6% static by count, under LeRobot's uniform per-pair loss weighting).

| joint | sum sq-err static (deg^2) | sum sq-err moving (deg^2) | % of total sq-err from static | % from moving |
|---|---:|---:|---:|---:|
| shoulder_lift | 195448423 | 598471157 | 24.6% | 75.4% |
| elbow_flex | 223049750 | 603191130 | 27.0% | 73.0% |

> First-order proxy only: assumes a predict-the-global-mean-action baseline and LeRobot's actual uniform (sample, chunk-position, action-dim) MSE weighting (confirmed in lerobot/policies/smolvla/modeling_smolvla.py VLAFlowMatching.forward -> F.mse_loss(..., reduction='none') averaged uniformly, pad-masked only). Real flow-matching training does not literally regress to a static per-joint mean, but this proxy is the standard way to reason about which population's characteristic magnitude dominates an uniformly-weighted squared-error objective when the model cannot fully disambiguate similar-looking inputs.

## 8. Answers

### start/low-motion samples가 전체에서 너무 적은가?

YES, structurally small relative to the moving segment: the data-driven static segment covers only 7.6% of all 11505 frames (875 frames; median 25 frames/episode, ~0.83s), vs. 92.4% moving. Fixed windows: frames[0:15]=4.6% of dataset, frames[0:30]=9.1%. Under LeRobot's uniform per-(sample, chunk-position) loss weighting (no oversampling of any segment by default), this segment gets roughly proportional representation in the objective - which is *not* boosted to compensate for it being the behaviourally most safety-sensitive part of the trajectory (the part right before Safety Gate sees the first delivered action).

### shoulder_lift/elbow_flex의 큰 action이 dataset에서 지배적인가?

YES for the moving segment, which is most of the dataset: mean|action-state| in the moving segment is shoulder_lift=3.04deg / elbow_flex=3.83deg (p95 8.44/8.04deg) vs. the static segment's 0.614/2.245deg. See large-action frequency table: 17.2% of moving-segment frames already carry a shoulder_lift action-state delta at/above the Safety Gate WOULD_CLAMP threshold on their own. Threshold coverage table (section 3) gives the exact per-segment WOULD_CLAMP/REJECT exceedance rates.

### 35 episodes 규모에서 static->movement transition을 충분히 학습하기 어려워 보이는가?

YES, plausibly: there are only 35 distinct static->movement transition examples in the entire dataset (one per episode) - and start_pose_diversity (section 5) shows those 35 transitions begin from near-identical proprioceptive states (frame-0 pairwise L2 median 3.85deg, max 11.75deg, vs. mid-reach-frame pairwise L2 median 41.06deg, max 102.41deg - i.e. Grid35's actual 35-way task diversity only shows up well after the hold, not at the start). The chunk-mechanism check (section 4) shows that even though a static observation's own chunk[0] target is small, 99.7% of static-segment observations are paired with a *whole* 50-step target chunk whose mean|delta| already exceeds the shoulder_lift WOULD_CLAMP threshold (since the chunk runs 1.67s forward, well past the ~0.7-1.0s typical movement onset). With only 35 near-identical start contexts each carrying that kind of large-mean target chunk, and no contrasting examples where a similar-looking static start is paired with a genuinely small full-chunk target, there is little data-side signal to teach the model that 'looks static' should imply 'predict small for a while' rather than 'predict this population's typical chunk shape'.

### 데이터 추가 수집이 필요하다면 어떤 샘플을 우선 추가해야 하는가?

Priority: (1) more start-segment *diversity*, not just more of the same 35 near-identical start poses - e.g. episodes where the static-hold duration varies (some near-zero hold, some longer), and/or where the visual scene early in the episode already makes the eventual grid-cell target legible (so the model has something other than a duplicate proprioceptive start state to condition on); (2) episodes/segments that break the correlation this dataset currently has baked in between 'observation looks static' and 'this chunk's mean target is large' - i.e. genuinely-static holds of varying length immediately followed by small-magnitude motion, and slow/gradual movement onsets (not just the current fast-ramp profile) so a range of chunk-mean magnitudes gets paired with static-looking inputs; (3) more Grid35 grid-cell coverage in general (35 episodes = 35 grid cells with 1 demo each - no repetition to average out per-episode teleop-operator variance).

### 재학습 시 start-segment oversampling / weighted loss / 더 많은 demonstrations 중 무엇이 우선인가?

More demonstrations (diversity-focused, see above) over naive start-segment oversampling. Rationale: static-segment (observation, chunk-position) pairs are already 7.6% of the uniformly-weighted training signal by *count* (loss-contribution proxy, section 6) - oversampling would inflate that further without adding a single new start pose or a single counter-example that decorrelates 'static-looking' from 'large upcoming chunk' - given only 35 distinct start contexts exist, oversampling risks overfitting to those exact 35 (state, chunk) pairs rather than teaching the general rule. A cheaper, no-new-data lever worth trying in parallel: a chunk-position-aware loss weighting (e.g. upweight near-term chunk indices relative to far-future ones, or clip/downweight target magnitude beyond the typical movement-onset horizon for still-static observations) - LeRobot's default loss has no such weighting today (see module docstring point 7), so this is a real, currently-unused lever, but it only reduces symptoms of the imbalance already present in the 35 episodes, it does not add the missing static-hold/short-hold/legible-early-visual diversity that (1) above targets. Given the static segment's own targets are small (mean_abs shoulder_lift=0.614deg) and the dataset is small (35 episodes total), the fastest, most robust fix is still more/varied demonstrations; loss weighting and oversampling are reasonable stop-gaps while more data is collected, not substitutes for it.
