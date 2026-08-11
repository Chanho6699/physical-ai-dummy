# Dataset Reweighting Ablation — `combined65_reweight_new3_old1_v1` (1:3)

Real GPU experiment (RTX 3050 8GB). Single-variable change vs `combined65_reweight_new2_old1_v1`:
**DataLoader sampling ratio only** (old35 : reinforcement30 = 1 : 3, i.e. reinforcement30 drawn
~3x as often, up from ~2x in the 2:1 run). Both source groups remain in the training pool. Original
(uniform) loss — **`early_action_loss_weights` intentionally NOT used in this run**. Model
architecture, chunk horizon, optimizer/scheduler, normalization/action representation, Safety Gate
thresholds all unchanged and identical to the 2:1 run. Fresh-started from `lerobot/smolvla_base`
(no resume from the 2:1 checkpoint). No real-robot writes. No dataset/checkpoint mutation. No git
commit/push.

## Implementation

Reused the existing `episode_group_sample_weights` / `build_episode_group_weighted_sampler()`
DataLoader path added for the 2:1 experiment, unchanged. **No new sampler code was written.**

Command: `--episode_group_sample_weights='{"0-34":1.0,"35-64":3.0}'` (episodes 0-34 = old35, 35-64 =
reinforcement30 in `so101_cube_pick_drop_combined65_v1`'s merged index space).

### Verification (before spending GPU time)

1. **Draw simulation** (no GPU): built the real sampler against the actual combined65 dataset
   (65 episodes / 21,327 frames) and drew 10 epochs' worth of indices (213,270 draws total):
   **old35 24.85%, reinforcement30 75.15%** (target 25.0%/75.0%) — matches to within sampling noise.
2. **5-step weighted smoke test**: ran cleanly, `episode_group_sample_weights` confirmed
   `early_action_loss_weights: None`, and the sampler's own startup log printed the correct group
   breakdown (`group 0-34: 35 episodes, 11505 frames, weight=1.000 -> expected sampling
   share=25.0%`; `group 35-64: 30 episodes, 9822 frames, weight=3.000 -> expected sampling
   share=75.0%`).
3. **Live training log**: the actual `combined65_reweight_new3_old1_v1` run's own startup log
   printed the identical 25.0%/75.0% breakdown — the configured ratio is what actually ran.

## Training

Fresh-started from `lerobot/smolvla_base`, otherwise identical command/hyperparameters to the 2:1
run (batch_size=4, seed=1000, AdamW lr=1e-4/betas=[0.9,0.95]/wd=1e-10/grad_clip=10.0,
cosine_decay_with_warmup warmup=1000/decay_steps=30000/decay_lr=2.5e-6, chunk_size=50,
n_action_steps=50, rename_map `{workspace->camera1, wrist->camera2}`, empty_cameras=1, use_amp=false,
`early_action_loss_weights=null`).

| metric | value |
|---|---|
| started / ended | 2026-08-09 20:17:56 → 21:24:08 |
| total wall time | 66.2 min (comparable to the 2:1 run's 66.9 min — the higher ratio adds no per-step overhead) |
| checkpoints saved | 002500, 005000, 007500, 010000 (all 4, confirmed) |
| errors / OOM | none |

---

## 4-way comparison (checkpoint 10000)

**A. Combined65 uniform baseline · B. reinforcement30-only · C. reweight 2:1 · D. reweight 3:1**

| metric | A baseline | B r30-only | C 2:1 | **D 3:1** |
|---|---:|---:|---:|---:|
| heldout MAE | 3.640 | 3.976 | 3.528 | **3.679** |
| train/heldout gap | n/a (no in-sample eval run) | 1.753 | 0.631 | **0.667** |
| clamp-free | 5/20 (25%) | 5/20 (25%) | 8/20 (40%) | **6/20 (30%)** |
| shoulder_lift clamp | 25% | 10% | 25% | **25%** |
| elbow_flex clamp | 75% | 75% | 60% | **70%** |
| L2 vs GT | 4.15 | 4.97 | 3.31 | **3.67** |

(A/B numbers pulled from the same `checkpoint_metrics.csv` / `first_action_diagnostics.csv`
sources used for the 2:1 report — see that file for the 2:1-vs-baseline narrative; A's in-sample
eval was never run so its train/heldout gap is not available.)

**At checkpoint 10000, 3:1 is worse than 2:1 on every one of these headline metrics** — it does
not clamp-free/L2/elbow_flex-clamp better than 2:1, and both heldout MAE and the train/heldout gap
regress slightly.

### Full checkpoint sweep — offline heldout MAE + train/heldout gap

| checkpoint | A baseline | B r30-only | C 2:1 | **D 3:1** |
|---:|---:|---:|---:|---:|
| 2500 | 4.716 | 5.474 (gap 1.248) | 4.867 (gap −0.039) | **4.710 (gap 0.018)** |
| 5000 | 4.340 | 4.145 (gap 1.390) | 4.109 (gap 0.270) | **3.634 (gap 0.130)** |
| 7500 | 3.412 | 3.881 (gap 1.632) | 3.588 (gap 0.548) | **3.567 (gap 0.491)** |
| 10000 | 3.640 | 3.976 (gap 1.753) | 3.528 (gap 0.631) | **3.680 (gap 0.667)** |

3:1's offline-best checkpoint is **7500** (3.567), not 10000 — the only experiment among the four
whose offline-best isn't its final checkpoint. 10000 is actually slightly *worse* than 7500 on
heldout MAE, unlike every prior experiment in this family.

### First-action diagnostic (same T01 reference observation, seeds 0-19)

| checkpoint | shoulder_lift mean/clamp | elbow_flex mean/clamp | clamp-free | L2 vs GT |
|---:|---:|---:|---:|---:|
| 2500 | +7.21/65% | −10.33/90% | 0/20 (0%) | 10.15 |
| 5000 | −0.16/20% | −3.50/25% | **11/20 (55%)** | 7.34 |
| 7500 | +4.18/30% | −7.95/85% | 3/20 (15%) | 4.91 |
| 10000 | +3.33/25% | −6.27/70% | 6/20 (30%) | **3.67** |

Unlike 2:1 (where clamp-free, elbow_flex-clamp-floor, and lowest-L2 all landed together at
checkpoint 10000), **3:1's first-action metrics do not converge on one checkpoint**: clamp-free
peaks at 5000 (55%, best of any checkpoint across every experiment in this whole ablation family),
while lowest L2 is at 10000 (3.67) and offline-best is at 7500. This 3-way split across checkpoints
is itself a notable finding — more aggressive resampling made the run's checkpoint trajectory
noisier, not cleaner.

### Temporal (chunk-position) error — key-joint MAE by bucket

| checkpoint | A baseline | B r30-only | C 2:1 | **D 3:1** |
|---:|---|---|---|---|
| 7500 step0 | 2.183 | 1.940 | **1.641** | 2.021 |
| 7500 step1-2 | 1.988 | 2.520 | 1.823 | **1.841** |
| 7500 step3+ | 11.213 | 10.661 | 8.254 | **7.377** |
| 10000 step0 | 1.806 | 1.991 | **1.617** | 1.707 |
| 10000 step1-2 | 1.812 | 2.077 | 1.859 | **1.846** |
| 10000 step3+ | 6.826 | 7.402 | 7.435 | **6.112** |

3:1's step0 (immediate-action) error is slightly worse than 2:1's at both 7500 and 10000, but its
step3+ (long-horizon, tail of the 50-step chunk) error is the **best of all four experiments at
both checkpoints** — 7.377 vs 2:1's 8.254 at 7500, and 6.112 vs 2:1's 7.435 at 10000. Full table:
`temporal_chunk_error.csv`.

---

## Answers

**1. 실제 sampling이 목표 25%/75%에 근접했는가?**
예. 213,270회 draw 테스트에서 old35 24.85% / reinforcement30 75.15%(목표 25.0%/75.0%)로 확인,
5-step weighted smoke test와 실제 훈련 로그 모두 동일한 25.0%/75.0% expected share를 출력했다.

**2. 3:1에서 clamp-free가 2:1의 40%보다 증가했는가?**
**아니다.** checkpoint 10000 기준 3:1의 clamp-free는 30%(6/20)로, 2:1의 40%(8/20)보다 오히려
낮다. (3:1의 checkpoint별 clamp-free 최고치는 5000에서의 55%지만, 이는 offline-best/L2-best
checkpoint와 다른 지점이라 직접 비교 대상인 "checkpoint 10000" 기준으로는 개선되지 않았다.)

**3. elbow_flex clamp가 60%보다 감소했는가?**
**아니다.** checkpoint 10000 기준 3:1의 elbow_flex clamp는 70%로, 2:1의 60%보다 오히려 높다
(악화).

**4. shoulder_lift clamp가 25% 이하로 유지/개선됐는가?**
**유지됐으나 개선되지는 않았다.** checkpoint 10000 기준 3:1의 shoulder_lift clamp는 25%로 2:1과
정확히 동일 — 목표(≤15%, reinforcement30-only의 10%)에는 여전히 도달하지 못했다.

**5. L2 vs GT가 3.31보다 좋아졌는가?**
**아니다.** checkpoint 10000 기준 3:1의 L2는 3.67로, 2:1의 3.31보다 악화됐다(+0.36deg). 이는 이
전체 reweighting 계열 실험(5개) 중 checkpoint 10000에서 2:1 다음으로 두 번째로 낮은 L2이지만,
목표였던 "2:1보다 개선"에는 미치지 못한다.

**6. heldout MAE 3.53 대비 generalization이 유지되는가?**
**소폭 악화됐다.** checkpoint 10000 기준 3:1의 heldout MAE는 3.68로 2:1의 3.53보다 +0.15deg
높다. 다만 3:1의 offline-best checkpoint(7500, 3.567)를 기준으로 보면 2:1의 최고치(3.53)에 거의
근접하며, combined65 baseline(3.41-3.64 범위)이나 reinforcement30-only(3.88-5.47 범위)보다는
여전히 낫다.

**7. train/heldout gap 0.63 대비 악화되는가?**
**소폭 악화됐다.** checkpoint 10000 기준 3:1의 gap은 0.667로 2:1의 0.631보다 +0.036 크다. 증가
폭 자체는 작지만(2:1 대비 약 6% 증가), gap 추이(0.018→0.130→0.491→0.667)도 2:1의 추이
(−0.039→0.270→0.548→0.631)와 거의 같은 기울기로 커져, "더 공격적인 resampling이 generalization
여유를 계속 개선해줄 것"이라는 2:1 보고서의 가설은 이번 실험에서 지지되지 않았다.

**8. offline-best와 first-action-best가 여전히 같은 checkpoint인가?**
**아니다.** 3:1에서는 세 기준이 서로 다른 checkpoint를 가리킨다 — offline heldout MAE 최저는
**7500**(3.567), first-action L2 최저는 **10000**(3.67), clamp-free 최고는 **5000**(55%). 2:1은
이 세 기준이 모두 10000에서 일치했던 유일한 실험이었는데, 3:1은 그 "깨끗한" 패턴이 깨졌다.

**9. 2:1과 3:1 중 어느 비율이 더 좋은 trade-off인가?**
**2:1이 더 낫다.** checkpoint 10000에서 3:1은 heldout MAE, train/heldout gap, clamp-free,
elbow_flex clamp, L2 vs GT — 이 다섯 지표 중 넷(모두 shoulder_lift clamp만 동률)에서 2:1보다
나쁘다. 3:1 자신의 최적 checkpoint(7500)로 비교해도 first-action 지표(clamp-free 15% vs 2:1의
35%, elbow_flex clamp 85% vs 65%, L2 4.91 vs 3.57)가 2:1의 7500보다 뚜렷이 나쁘다. 유일하게 3:1이
우위인 지표는 temporal step3+ (장기 chunk 오차) — 7500/10000 모두 4개 실험 중 최저치를 기록했다.
즉 3:1은 "chunk 뒷부분 정확도"라는 좁은 영역에서만 이득이 있고, 나머지 전 영역(즉시 행동 안전성,
generalization, L2)에서는 2:1보다 못하다. **resampling ratio를 2:1에서 3:1로 더 밀어붙이는 것은
수확체감을 넘어 일부 지표에서 역전(regression)까지 관찰된 지점으로 보이며, 2:1이 이 계열 실험의
sweet spot에 더 가깝다.**

**10. 다음 단계에서 early_action_loss_weights를 결합할 가치가 있는가?**
**있다 — 단, 3:1이 아니라 2:1 위에 결합해야 한다.** 이번 실험은 "resampling을 더 강하게 밀면
계속 좋아질 것"이라는 가설을 반증했다: 3:1은 2:1보다 대부분의 축에서 나쁘다. 따라서 다음 실험은
`combined65_reweight_new2_old1_v1`(2:1) 데이터셋 비율 위에 `early_action_loss_weights`를 추가하는
2-factor 실험을 권장한다. 3:1의 유일한 이점(step3+ 장기 오차 개선)은 early-weight 실험이 이미
checkpoint 7500에서 보여준 "step0/step3+ 동시 개선" 패턴과 방향이 겹치므로, reweighting 자체를
3:1까지 밀어붙이기보다는 2:1 + early-weight 조합으로 그 효과를 얻는 편이 더 안전해 보인다.

---

## Case 판정

**2:1 대비 명확한 regression** — 3:1은 이번 ablation에서 "더 많이 reweight할수록 좋다"는 단조적
개선을 반증하는 반례다.

- ❌ clamp-free: 2:1의 40% → 3:1의 30% (checkpoint 10000, 악화)
- ❌ elbow_flex clamp: 2:1의 60% → 3:1의 70% (악화)
- ➖ shoulder_lift clamp: 25% → 25% (동률, 목표 미달 유지)
- ❌ L2 vs GT: 2:1의 3.31 → 3:1의 3.67 (악화)
- ❌ heldout MAE: 2:1의 3.53 → 3:1의 3.68 (소폭 악화)
- ❌ train/heldout gap: 2:1의 0.63 → 3:1의 0.67 (소폭 악화)
- ✅ temporal step3+ (장기 chunk 오차): 4개 실험 중 최저 (유일한 우위 지표)
- ⚠️ offline-best(7500)와 first-action-best(10000)가 처음으로 divergence — checkpoint 선택
  자체가 더 애매해짐

**결론**: 데이터 reweighting은 2:1 부근에서 최적점에 도달한 것으로 보이며, 3:1로 더 밀어붙이는
것은 순이익이 없다. **2:1을 이 계열의 최종 채택 비율로 유지**하고, 다음 단계는 비율을 더 올리는
대신 2:1 위에 `early_action_loss_weights`를 결합하는 실험으로 전환할 것을 권장한다.

## 안전

Safety threshold 변경 없음. 실물 로봇 write 없음. 이번 3:1 checkpoint 10000/7500 모두 elbow_flex
clamp가 70-85%로 2:1보다 높아, Shadow evaluation 후보로는 2:1의 checkpoint 10000이 여전히 더
우선순위가 높다.
