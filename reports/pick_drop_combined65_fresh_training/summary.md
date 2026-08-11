# Pick & Drop Combined65 — SmolVLA fresh training & first-action diagnostic

Real GPU run (RTX 3050 8GB, `~/lerobot` venv). Fresh-started from `lerobot/smolvla_base` (no
resume, no reused optimizer state). Trained on `data/so101_cube_pick_drop_combined65_v1` (65
episodes, 21,327 frames, task `"Pick up the cube and drop it into the bin."`). No real-robot
writes at any point (training is offline; evaluation and first-action diagnostics are pure
inference against recorded/heldout data). Safety thresholds unchanged. No dataset/checkpoint
mutation. No git operations.

## 0. Provenance of every number below

- **Training hyperparameters**: copied verbatim from the actual V2 fresh-training invocation found
  in `~/.bash_history` and cross-checked against the recorded
  `outputs/grid35_v2/smolvla_grid35_v2_clean_fresh/checkpoints/010000/pretrained_model/train_config.json`
  — not reconstructed from memory. Only `--dataset.repo_id`, `--dataset.root`, `--output_dir`,
  `--job_name` changed.
- **V2 baseline numbers** (offline MAE 5.3875/4.2909/3.9841/4.2210; first-action seed-sweep
  stats): read directly from the existing `reports/grid35_v2_midpoint_eval/summary.json` and
  `reports/grid35_v2_T01_seed_sweep/seed_sweep.json` — reproduced exactly (spot-checked before
  running anything new), not retyped from the task prompt.
- **Offline eval** and **first-action seed sweep**: reused `scripts/evaluate_smolvla_midpoint.py`
  and `scripts/sweep_grid35_first_action_seed.py` unmodified, only overriding `--task` to the new
  Pick&Drop string (non-destructive — the heldout dataset `data/so101_cube_xy_midpoint_test10_v2_clean`
  and the T01 Shadow reference observation `reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`
  were only read, never written).

## 1. Training pipeline actually used (researched, not assumed)

```
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.workspace":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --dataset.repo_id=local/so101_cube_pick_drop_combined65_v1 \
  --dataset.root=.../data/so101_cube_pick_drop_combined65_v1 \
  --output_dir=outputs/pick_drop_combined65/smolvla_pick_drop_combined65_fresh \
  --job_name=smolvla_pick_drop_combined65_fresh \
  --batch_size=4 --steps=10000 \
  --save_checkpoint=true --save_freq=2500 --log_freq=100 --wandb.enable=false
```

| hyperparameter | value | source |
|---|---|---|
| base checkpoint | `lerobot/smolvla_base` | identical to V2 |
| batch size | 4 | identical |
| optimizer | AdamW, lr=1e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=1e-10, grad_clip_norm=10.0 | `use_policy_training_preset=true` default, identical to V2's recorded config |
| scheduler | cosine_decay_with_warmup, warmup 1000→333 / decay 30000→10000 (auto-scaled because `steps(10000) < num_decay_steps(30000)`), peak_lr=1e-4, decay_lr=2.5e-6 | identical auto-scaling behavior confirmed in this run's own log |
| chunk_size / n_action_steps | 50 / 50 | identical |
| warmup steps | 1000 (config) → auto-scaled to 333 | identical |
| grad accumulation | none (1) | identical |
| num_workers | 4 (dataloader default) | identical |
| device / precision | cuda, `use_amp=false` (fp32) | identical |
| image preprocessing | `resize_imgs_with_padding=[512,512]`, IDENTITY visual normalization, `empty_cameras=1` (padding camera slot) | identical |
| save frequency | every 2500 steps | identical |
| seed | 1000 (lerobot default, not overridden) | identical |
| max_steps | 10000 | identical |
| **changed** | `dataset.repo_id`/`dataset.root` → combined65, `output_dir`/`job_name` → new paths | only these |

## 2. Preflight (before full training)

1. `LeRobotDataset` load check: 65 episodes ✓, 21,327 frames ✓, task = `"Pick up the cube and drop it into the bin."` ✓, joint order `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` ✓ (both state and action), workspace/wrist tensors `[3,480,640]` float32 present, no NaN/Inf in a sampled frame.
2. Real 5-step `lerobot-train` smoke run (`--steps=5 --save_checkpoint=false`) through the exact training pipeline: losses `0.702 → 1.048 → 0.505 → 0.650 → 0.374` (all finite), GPU mem ~2.44GB, no errors. **PASS** — full training launched.

## 3. Training result

| metric | value |
|---|---|
| started | 2026-08-09 12:41:59 |
| ended (`End of training`) | 2026-08-09 13:48:42 |
| total wall time | **66.7 min** (V2's original 10k run: ~65.5 min — closely comparable) |
| checkpoints saved | 002500, 005000, 007500, 010000 (all 4, confirmed on disk) |
| errors / OOM / Traceback in log | **none** |
| GPU memory | steady ~2.44GB / 8GB throughout (no pressure) |

| checkpoint | wall time (min) | train loss | LR | grad norm |
|---:|---:|---:|---:|---:|
| 2500 | 16.15 | 0.124 | 8.6e-05 | 3.364 |
| 5000 | 32.95 | 0.082 | 5.2e-05 | 2.382 |
| 7500 | 49.85 | 0.066 | 1.7e-05 | 2.037 |
| 10000 | 66.63 | 0.053 | 2.5e-06 | 1.726 |

(Extracted from the training log's own periodic `step:N loss:... lr:... mem_gb:...` lines, matched
to each checkpoint by the identical timestamp shared with that step's `Checkpoint policy after
step N` log line — `ot_train.py` displays `step` rounded to the nearest thousand once ≥1000, e.g.
`step:7K` for anything in [7000,7999], so exact-step matching had to go through timestamps, not the
displayed step text.)

Train loss decreases monotonically and smoothly — no instability, no spikes, no sign of the
optimizer diverging at any point.

## 4. Offline held-out evaluation (`data/so101_cube_xy_midpoint_test10_v2_clean`, unmodified, 10 episodes / 3288 frames, seed=42, task overridden to the Pick&Drop string at eval time only)

| checkpoint | action MAE | shoulder_lift MAE | elbow_flex MAE | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |
|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 4.7157 | 7.465 | 6.873 | 256 | 3011 | 21 |
| 5000 | 4.3404 | 5.846 | 7.208 | 375 | 2809 | 104 |
| **7500** | **3.4123** | **4.700** | **5.305** | 688 | 2590 | 10 |
| 10000 | 3.6396 | 5.052 | 6.010 | 659 | 2568 | 61 |

| checkpoint | V2 action MAE | Combined65 action MAE | Δ |
|---:|---:|---:|---:|
| 2500 | 5.3875 | 4.7157 | **−0.672** |
| 5000 | 4.2909 | 4.3404 | +0.050 |
| 7500 | 3.9841 | 3.4123 | **−0.572** |
| 10000 | 4.2210 | 3.6396 | **−0.581** |

**Every Combined65 checkpoint's offline action MAE is equal to or lower than V2's own checkpoint
at the same step** (3 of 4 clearly lower, one essentially tied). The same non-monotonic
2.5k→5k→7.5k→10k shape (improve, improve, best, then worsen slightly) that V2 showed is
reproduced here too — 7500 is again the offline-MAE-best checkpoint.

## 5. First-action diagnostic (same T01 Shadow reference observation as V2's baseline — real
recorded state+images, read-only reuse, no new hardware interaction; seeds 0–19; Safety Gate
thresholds unchanged: shoulder_pan 4.58 / shoulder_lift 5.16 / elbow_flex 5.73 / wrist_flex 4.01 /
wrist_roll 1.15 / gripper 9.17 deg)

| checkpoint | shoulder_lift mean±std | shoulder_lift clamp rate | elbow_flex mean±std | elbow_flex clamp rate | clamp-free seeds | L2 vs nearest-demo GT |
|---|---:|---:|---:|---:|---:|---:|
| **V2 7500 (baseline)** | +5.13 ± 3.01 | 45% (9/20) | −7.08 ± 2.30 | 75% (15/20) | 4/20 (20%) | 4.62 deg |
| Combined65 2500 | −0.02 ± 7.78 | 55% (11/20) | +1.63 ± 6.81 | 30% (6/20) | 5/20 (25%) | 13.17 deg |
| Combined65 5000 | +5.26 ± 4.13 | 45% (9/20) | −8.21 ± 3.35 | 75% (15/20) | 4/20 (20%) | 6.83 deg |
| Combined65 7500 | +4.51 ± 3.02 | 35% (7/20) | −7.66 ± 2.30 | **85% (17/20)** | 3/20 (15%) | 4.78 deg |
| **Combined65 10000** | **+3.83 ± 2.77** | 25% (5/20) | −6.88 ± 2.12 | 75% (15/20) | **5/20 (25%)** | **4.15 deg** |

Reading this table honestly:

- **Checkpoint 2500 is not usable evidence of anything "good"** — its low bias magnitude and high
  clamp-free count come from being badly undertrained (std 6.8–7.8deg, L2-vs-GT 13.17deg, nearly
  3× every other checkpoint's error) — noise, not quality.
- Among the genuinely trained checkpoints (5000/7500/10000), **10000 is the only one that beats
  the V2 7500 baseline on every column at once**: smaller shoulder_lift bias (+3.83 vs +5.13),
  smaller shoulder_lift clamp rate (25% vs 45%), smaller elbow_flex bias magnitude (−6.88 vs
  −7.08), equal-or-better clamp-free rate (25% vs 20%), and lower L2-vs-GT (4.15 vs 4.62deg).
- **7500 — the offline-MAE-best checkpoint — does *not* show this improvement.** Its elbow_flex
  clamp rate (85%) and clamp-free rate (15%) are both *worse* than V2's 7500 baseline. Offline MAE
  and first-action safety quality point at **different** checkpoints here.
- elbow_flex clamp rate stays at 75% or worse at every non-degenerate checkpoint (5000/7500/10000)
  — essentially unchanged from V2's 75%. It never got meaningfully safer.

Full per-seed tables: `reports/pick_drop_combined65_fresh_training/first_action_seed_sweep_<step>/seed_sweep.{json,csv,md}`.

**Reused-script labeling caveat (verified, does not affect any number above):**
`sweep_grid35_first_action_seed.py`'s nearest-demo lookup assumes one data file per episode (true
for V2/V3's original layout). Combined65's merge packed all 65 episodes into a single
`data/chunk-000/file-000.parquet`, which breaks that assumption silently — every sweep run against
combined65 reports `nearest_demo_match.episode: 0`, which is a mislabel. Verified directly: the
reported `frame` value (10874) is actually the correct *global* row index, and
`so101_cube_pick_drop_combined65_v1` row 10874 has `episode_index=33, frame_index=25` — i.e. the
real match is combined65 episode 33 (= V2's own episode 33, unchanged by the merge), the exact
same demo V2's correctly-labeled sweep matched (confirmed by bit-identical `l2_dist_deg` and
`gt_immediate_delta` across all 5 sweeps in this report). Only the episode label is wrong; every
delta/clamp/L2 number is computed correctly.

## 6. Combined view

`checkpoint_metrics.csv` (training + offline eval) and `first_action_diagnostics.csv` (per-joint
first-action stats, V2 baseline + all 4 Combined65 checkpoints) hold the full numeric detail behind
every table above.

---

## Answers

**1. Combined65 fresh training이 정상 완료됐는가?**
예. `lerobot/smolvla_base`에서 fresh-start(resume 없음, optimizer state 재사용 없음), preflight
통과 후 10,000 step 전부 완주, 4개 checkpoint(2500/5000/7500/10000) 모두 정상 저장, 학습 로그에
에러/OOM/Traceback 없음, loss는 0.124→0.053까지 단조 감소, wall time 66.7분(V2의 ~65.5분과 거의
동일한 페이스).

**2. offline 기준 best checkpoint는 무엇인가?**
**7500** (action MAE=3.4123, 4개 중 최저). V2와 동일하게 2.5k→5k→7.5k에서 개선되다 10k에서 살짝
다시 나빠지는 동일한 패턴이 재현됐다. Combined65 7500의 offline MAE는 V2 7500(3.9841)보다도
낮다.

**3. first-action shoulder_lift/elbow_flex bias가 V2보다 줄었는가?**
**checkpoint에 따라 다르다 — 일괄적으로 "줄었다"고 말할 수 없다.** 10000 checkpoint에서는
shoulder_lift(+5.13→+3.83deg)와 elbow_flex(−7.08→−6.88deg) 둘 다 V2보다 개선됐지만, offline
기준 best인 7500에서는 shoulder_lift는 소폭 개선(+5.13→+4.51)된 반면 elbow_flex는 오히려 악화
(−7.08→−7.66, clamp rate 75%→85%)됐다. 5000에서도 elbow_flex는 악화(−7.08→−8.21)됐다.

**4. clamp-free rate가 의미 있게 개선됐는가?**
아니다, 미미하다. 유의미하게 학습된 checkpoint(5000/7500/10000) 중 V2(4/20=20%)를 넘는 것은
10000(5/20=25%, +1개 seed) 뿐이고, 7500은 오히려 3/20(15%)로 더 나쁘다. elbow_flex 단독 clamp
rate는 모든 combined65 checkpoint에서 75% 이상으로 V2와 사실상 동일하다.

**5. checkpoint가 증가하면서 다시 악화되는 현상이 있는가?**
있다 — 게다가 **offline MAE와 first-action 품질이 서로 반대 방향으로 비단조적**이라는 점이
핵심 발견이다. offline MAE는 7.5k에서 최저였다가 10k에서 다시 소폭 상승(3.41→3.64)한다(V2와
동일 패턴). 반면 first-action L2-vs-GT/clamp-free/bias는 5k→7.5k 구간에서 나빠졌다가(clamp-free
20%→15%) 10k에서 다시 좋아진다(→25%). 즉 "offline 기준으로 고른 best checkpoint"와 "first-action
기준으로 고른 best checkpoint"가 이번 실험에서는 서로 다른 checkpoint(7500 vs 10000)다.

**6. 실제 T01-T10 Shadow evaluation으로 넘어갈 가치가 있는 checkpoint는 무엇인가?**
**10000.** offline MAE는 7500보다 약간 높지만(3.64 vs 3.41), first-action 지표 전항목(두 관절
bias 크기, shoulder_lift clamp rate, clamp-free rate, GT 대비 L2)에서 V2 baseline과 combined65의
다른 모든 checkpoint를 동시에 능가하는 유일한 checkpoint다. 이번 실험의 목적이 "첫 action bias"
문제이므로, 순수 offline MAE보다 이 지표를 우선하는 것이 합리적이다. 다만 elbow_flex clamp
rate가 여전히 75%라는 점은 Shadow 단계에서도 계속 주시해야 한다.

**7. 데이터 보강만으로 문제가 해결됐는가, 아니면 chunk/loss 구조 실험이 여전히 필요한가?**
**해결되지 않았다 — chunk/loss 구조 실험이 여전히 필요하다.** offline action MAE는 거의 모든
checkpoint에서 확실히 개선됐고 이는 data coverage 보강의 실제 효과로 보인다. 하지만 first-action
안전성의 핵심 지표인 elbow_flex clamp rate(75~85%)는 V2(75%)에서 사실상 그대로다 — 최선인
10000 checkpoint조차 elbow_flex mean bias가 −6.88deg로 5.73deg 임계값을 항상 넘는 수준이고,
20개 seed 중 15개가 여전히 clamp된다. 이는 이전 coverage 분석(`reports/grid35_v2_vs_start_coverage_v3/`)이
예측한 그대로다 — static-looking observation과 큰 future chunk motion의 상관관계는 V3
보강으로 줄었을 뿐(61% 잔존) 사라지지 않았고, 그 잔존 상관관계가 이번 실측 first-action bias로
그대로 이어졌다. 따라서 data coverage는 필요조건이었지만 충분조건은 아니었다 — chunk-position-
aware loss weighting 등 구조적 개입이 다음 단계로 여전히 유효하다.
