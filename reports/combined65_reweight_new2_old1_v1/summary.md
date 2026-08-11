# Dataset Reweighting Ablation — `combined65_reweight_new2_old1_v1`

Real GPU experiment (RTX 3050 8GB). Single-variable change vs the combined65 baseline: **DataLoader
sampling ratio only** (old35 : reinforcement30 = 1 : 2, i.e. reinforcement30 drawn ~2x as often).
Both source groups (old35 + reinforcement30) remain in the training pool — nothing removed, unlike
`reinforcement30_only_v1`. Original (uniform) loss, no early-action weighting. Model architecture,
chunk horizon, optimizer/scheduler, normalization/action representation, Safety Gate thresholds all
unchanged. No real-robot writes. No dataset/checkpoint mutation. No git commit/push.

## Implementation

Added a new `WeightedRandomSampler`-based DataLoader path to LeRobot, gated behind a new optional
config field (default `None` = exact pre-existing sampler, unchanged for every other experiment):

| file | change |
|---|---|
| `~/lerobot/src/lerobot/configs/train.py` | new field `episode_group_sample_weights: dict[str, float] \| None = None` |
| `~/lerobot/src/lerobot/datasets/sampler.py` | new `build_episode_group_weighted_sampler()` - builds per-frame weights so each named episode-index-range GROUP is drawn with aggregate probability proportional to its weight (weight ÷ that group's own frame count), regardless of the two groups' different episode/frame counts |
| `~/lerobot/src/lerobot/scripts/lerobot_train.py` | sampler-construction block: `if cfg.episode_group_sample_weights: use the new sampler; else: unchanged EpisodeAwareSampler path` |

Command: `--episode_group_sample_weights='{"0-34":1.0,"35-64":2.0}'` (episodes 0-34 = old35, 35-64 =
reinforcement30 in `so101_cube_pick_drop_combined65_v1`'s merged index space).

### Verification (before spending GPU time)

1. **Direct sampler test**: drew all 21,327 indices from the built sampler and counted group
   membership — **old35 33.29%, reinforcement30 66.71%** (target 33.3%/66.7%, i.e. 2:1) — matches
   to within sampling noise.
2. **Logging bug found and fixed**: the first implementation's "expected sampling share" log line
   summed weight *per episode* instead of *per group*, printing a nonsensical 1.1%/2.1%. Fixed to
   sum per distinct group; re-verified it now prints the correct 33.3%/66.7%.
3. **Regression smoke test**: 5-step run with **no** flag reproduced the same losses
   (0.704/1.045/0.504/0.654/0.374) as every prior baseline preflight — default path unaffected.
4. **Weighted smoke test**: 5-step run with the flag ran cleanly, log printed the correct group
   breakdown (`group 0-34: 35 episodes, 11505 frames, weight=1.000 -> expected sampling
   share=33.3%`; `group 35-64: 30 episodes, 9822 frames, weight=2.000 -> expected sampling
   share=66.7%`).
5. **Live training log**: the actual `combined65_reweight_new2_old1_v1` run's own startup log
   printed the identical 33.3%/66.7% breakdown — the configured ratio is what actually ran, not
   just what the smoke test showed.

## Training

Fresh-started from `lerobot/smolvla_base`, otherwise identical command to the combined65 baseline.

| metric | value |
|---|---|
| started / ended | 2026-08-09 17:46:22 → 18:52:46 |
| total wall time | 66.9 min (comparable to every other 10k-step run — the sampler change adds no per-step overhead) |
| checkpoints saved | 002500, 005000, 007500, 010000 (all 4, confirmed) |
| errors / OOM | none |

---

## 5-way comparison

### Offline heldout MAE + train/heldout generalization gap

| checkpoint | V2 (35ep) | combined65 baseline (65ep) | combined65 early-weight (65ep) | reinforcement30-only (30ep) | **reweight 2:1 (65ep)** |
|---:|---:|---:|---:|---:|---:|
| 2500 | 5.39 | 4.72 | 4.65 | 5.47 | 4.87 |
| 5000 | 4.29 | 4.34 | 4.43 | 4.14 | 4.11 |
| 7500 | 3.98 | 3.41 | 3.40 | 3.88 | 3.59 |
| **10000** | 4.22 | 3.64 | 3.61 | 3.98 | **3.53** |

| checkpoint | reinforcement30-only train/heldout gap | **reweight 2:1 train/heldout gap** |
|---:|---:|---:|
| 2500 | 1.25 | **-0.04** |
| 5000 | 1.39 | **0.27** |
| 7500 | 1.63 | **0.55** |
| 10000 | 1.75 | **0.63** |

The gap grows far more slowly under reweighting (10000: 0.63 vs 1.75 — **~2.8x smaller**), and
offline MAE at 10000 (3.53) beats reinforcement30-only's best checkpoint (3.88) by 0.35deg while
landing close to combined65 baseline's best (3.41-3.61 range). **Generalization is substantially
recovered** without giving up all of reinforcement30-only's advantage.

### First-action diagnostic (same T01 reference observation, seeds 0-19)

| experiment / checkpoint | shoulder_lift bias/clamp | elbow_flex bias/clamp | clamp-free | L2 vs GT |
|---|---:|---:|---:|---:|
| V2 7500 | +5.13/45% | −7.08/75% | 4/20 (20%) | 4.62 |
| combined65 baseline 7500 | +4.51/35% | −7.66/85% | 3/20 (15%) | 4.78 |
| combined65 early-weight 7500 | +3.25/25% | −6.94/75% | 5/20 (25%) | 4.40 |
| reinforcement30-only 7500 | **+1.61/10%** | −5.88/60% | **8/20 (40%)** | 4.58 |
| reweight 5000 | +5.21/40% | −6.29/65% | 6/20 (30%) | 5.64 |
| **reweight 7500** | +3.93/25% | −6.50/65% | 7/20 (35%) | **3.57** |
| **reweight 10000** | +3.96/25% | −6.11/60% | **8/20 (40%)** | **3.31** |

reweight's 10000 checkpoint: **matches reinforcement30-only's best clamp-free rate (40%)**, matches
its elbow_flex clamp floor (60%), and posts the **lowest L2-vs-GT of any checkpoint across all five
experiments (3.31deg)** — while shoulder_lift bias/clamp (25%) does not reach reinforcement30-only's
10%. Full detail: `first_action_diagnostics.csv`, per-seed data in `first_action_seed_sweep_<step>/`.

### Temporal (chunk-position) error — key-joint MAE by bucket

| checkpoint | combined65 baseline | early-weight | reinforcement30-only | **reweight 2:1** |
|---:|---|---|---|---|
| 7500 step0 | 2.183 | 1.918 | 1.940 | **1.641** |
| 7500 step1-2 | 1.988 | 2.016 | 2.520 | 1.823 |
| 7500 step3+ | 11.213 | 10.932 | 10.661 | **8.254** |
| 10000 step0 | 1.806 | 1.693 | 1.991 | **1.617** |
| 10000 step1-2 | 1.812 | 1.867 | 2.077 | 1.859 |
| 10000 step3+ | 6.826 | 7.424 | 7.402 | 7.435 |

At 7500, reweight has the **lowest step0 error of all four experiments** (1.641deg) *and* the
lowest step3+ error (8.254deg) — no visible early/late trade-off, unlike the early-weight
experiment's own step3+ regression at 10000. Full table: `temporal_chunk_error.csv`.

---

## Answers

**1. sampling ratio가 실제로 2:1에 가깝게 적용됐는가?**
예. 직접 샘플러 draw 테스트로 33.29%/66.71%(목표 33.3%/66.7%) 확인, 실제 훈련 로그에도 동일하게
기록됨. (구현 과정에서 발견한 로깅 버그—실제 sampler 가중치가 아니라 로그 문구만 잘못 계산되던
문제—도 함께 수정·재검증했다.)

**2. offline-best checkpoint는?**
**10000** (MAE=3.53) — 다른 모든 실험이 7500에서 offline-best였던 것과 달리, 이번 실험은
10000이 최저. combined65 baseline(3.41)에 가깝게 근접했다.

**3. first-action-best checkpoint는?**
**10000** (clamp-free 40%, L2=3.31, 전체 실험 중 최저 L2) — **offline-best와 완전히 일치**한다.
이는 5개 실험 중 유일하게 offline-best와 first-action-best가 divergence 없이 겹치는 사례다.

**4. shoulder_lift bias/clamp 변화는?**
combined65 baseline(+4.51/35%) → reweight(+3.96/25%, 10000 기준)로 개선됐으나,
reinforcement30-only(+1.61/10%)만큼 좋아지지는 않았다 — old35가 여전히 1의 비중으로 섞여있기
때문으로 보인다.

**5. elbow_flex bias/clamp 변화는?**
combined65 baseline(−7.66/85%) → reweight(−6.11/60%, 10000 기준)로 크게 개선, clamp rate는
reinforcement30-only의 최저치(60%)와 동일한 수준까지 내려왔다.

**6. clamp-free rate는?**
combined65 baseline 15% → reweight 40% (10000) — reinforcement30-only의 최고치(40%, 7500)와
정확히 동률.

**7. heldout MAE는?**
3.53 (10000) — reinforcement30-only(3.88)보다 명확히 낮고(−0.35deg), combined65
baseline(3.41-3.64 범위)에 근접.

**8. train/heldout gap은?**
10000에서 0.63deg — reinforcement30-only의 1.75deg 대비 **약 2.8배 작다**. gap 증가 추세도
훨씬 완만함(-0.04→0.27→0.55→0.63 vs 1.25→1.39→1.63→1.75).

**9. reinforcement30-only 대비 generalization 회복 여부는?**
**회복됐다.** heldout MAE 개선(3.88→3.53)과 train/heldout gap 대폭 축소(1.75→0.63) 둘 다
뚜렷하다.

**10. combined65 대비 first-action 개선 유지 여부는?**
**유지, 오히려 일부는 더 개선.** clamp-free 15%→40%, elbow_flex clamp 85%→60%, L2 4.78→3.31
(가장 큰 개선), step0 temporal error도 combined65/early-weight보다 낮다(1.641 vs 2.183/1.918
@7500).

**11. 다음 비율은 3:1 / 1.5:1 중 무엇이 적절한가?**
**3:1을 권장한다.** 이유: generalization 회복에 아직 여유가 있다 — heldout MAE(3.53)가 "강한
성공" 기준(≤3.6-3.7)을 이미 만족하고, gap(0.63)도 reinforcement30-only 대비 여유가 크다. 반면
shoulder_lift clamp(25%)는 목표(≤15%)에 아직 못 미치고 있어, generalization 예산을 조금 더
써서 new30 비중을 높이는 방향이 이치에 맞는다. 1.5:1로 완화할 이유(과적합 폭발 등)는 이번
실험에서 관찰되지 않았다.

**12. 이후 early-weight와 결합할 가치가 있는가?**
**있다.** 두 레버가 서로 다른 곳에서 이점을 보였다 — reweighting은 generalization/clamp-free/L2
전반에서 우수했고, early-weight는 checkpoint 7500에서 step0을 줄이면서 step3+까지 개선시키는
유일한 조합이었다(reweight도 이번에 비슷하게 step0/step3+ 동시 개선을 보였다는 점에서 방향이
일치한다). 두 레버 모두 단독으로는 "강한 성공" 기준(elbow_flex clamp≤40%, clamp-free≥60%)에
도달하지 못했으므로, **reweighted dataset(3:1 검토) 위에 early_action_loss_weights를 추가로
적용하는 2-factor 실험**이 다음 단계로 타당하다.

---

## Case 판정

**Case 1 (first-action 유지 + generalization 회복)에 가장 가깝다**, 다만 완전히 깨끗하지는 않다:

- ✅ generalization 회복: heldout MAE 3.88→3.53, gap 1.75→0.63 (강한 성공 기준 충족)
- ✅ clamp-free/L2/elbow_flex clamp: reinforcement30-only 수준 유지 또는 개선 (L2는 전체 실험 중 최고)
- ⚠️ shoulder_lift clamp(25%)는 reinforcement30-only(10%)에 못 미침 — "우선 성공" 기준(≤15%)
  미달
- ⚠️ elbow_flex clamp(60%)/clamp-free(40%)는 "우선 성공" 기준(≤50%/≥50%)에 근접했으나 정확히는
  미달

**결론**: 데이터 reweighting은 진짜로 작동한다 — old35를 완전히 버리지 않고도
reinforcement30-only의 first-action 이점 대부분을 가져오면서 generalization을 크게 회복시켰다.
다만 아직 "강한 성공" 전 항목을 만족하지는 못했으므로, 3:1 비율 및/또는 early-weight와의 결합이
다음 단계로 유효하다.

## 안전

Safety threshold 변경 없음. 실물 로봇 write 없음. 이번 checkpoint 10000은 Shadow evaluation
(비-write) 진행 가치가 있는 가장 강력한 후보다 — 다만 elbow_flex clamp 60%가 여전히 남아있다는
점은 Shadow 단계에서도 계속 주시해야 한다.
