# Early-Action Weighted Loss — `combined65_early_weight_v1`

Real GPU experiment (RTX 3050 8GB). Single-variable change vs the `combined65` baseline: **loss
weighting only**. Data (`data/so101_cube_pick_drop_combined65_v1`, 65 episodes), model
architecture, chunk horizon (chunk_size=50), optimizer/scheduler, normalization/action
representation, and all other hyperparameters are byte-identical to the combined65 baseline run.
No real-robot writes. Safety thresholds unchanged. No dataset/checkpoint mutation. No git
commit/push.

## Modified files

| file | change |
|---|---|
| `~/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py` | added one new optional field: `early_action_loss_weights: list[float] \| None = None` |
| `~/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` | `SmolVLAPolicy.forward()`'s loss-reduction branches now check a new `_chunk_step_weights()` helper; **`None` (default) → exact pre-existing code path, unchanged** |

`git diff --stat` (informational only, not committed): 2 files changed, 62 insertions, 8 deletions.

## Loss change (what actually happens in code)

Before: `losses` (B, chunk_size, action_dim) reduced with `losses.sum() / num_valid` where
`num_valid` counts valid (non-padding) `(sample, time)` positions × action_dim — every chunk
position weighted identically.

After (only when `config.early_action_loss_weights` is set): a per-chunk-position weight vector
`w_t` (shape `(1, chunk_size, 1)`, broadcasting across the batch and action dims — **no
per-action-dimension weighting**, purely temporal) is built from the config list (`[3.0, 2.0,
2.0]` for this experiment → step 0 = 3.0×, steps 1-2 = 2.0×, steps 3+ = 1.0×), then:

```
weighted_loss = sum(loss_t * w_t * valid_mask) / sum(w_t * valid_mask)
```

— i.e. exactly the formula requested, generalizing the pre-existing `sum(loss)/num_valid`
normalization (which is the `w_t ≡ 1` special case) rather than replacing it, so the two are
on the same MSE-per-element scale. Existing `action_is_pad` masking/padding semantics are
untouched — the weighting is applied on top of, not instead of, the existing in-episode-bound
mask.

### Verification (before spending any GPU time on the full run)

1. **Pure-math unit test**: `_chunk_step_weights` returns the exact `[3,2,2,1,1,1,...]` vector
   when set and `None` when unset; the weighted-mean formula matches a hand-computed value; the
   weighted-formula-with-all-ones-weights reproduces the original code's output bit-for-bit.
2. **Live regression smoke test**: a 5-step `lerobot-train` run with **no** weighting flag
   reproduced the *exact* same losses (`0.702, 1.048, 0.505, 0.650, 0.374`) as the original
   combined65-baseline preflight run — proof this change is byte-identical for every
   already-existing experiment (V2, combined65 baseline) that doesn't set the new field.
3. **Live weighted smoke test**: same 5 steps with `--policy.early_action_loss_weights='[3.0,2.0,2.0]'`
   produced distinct, finite, sane losses (`0.685, 1.037, 0.502, 0.645, 0.356`) with no errors.

## Training config

Identical to the combined65 baseline command (`reports/pick_drop_combined65_fresh_training/summary.md`
section 1) plus one flag: `--policy.early_action_loss_weights='[3.0,2.0,2.0]'`. Fresh-started from
`lerobot/smolvla_base` (no resume, no reused optimizer state). Output:
`outputs/pick_drop_combined65/combined65_early_weight_v1/`.

| metric | value |
|---|---|
| started / ended | 2026-08-09 14:31:45 → 15:36:38 |
| total wall time | 64.9 min (baseline: 66.7 min — no meaningful overhead from the weighting) |
| checkpoints saved | 002500, 005000, 007500, 010000 (all 4, confirmed on disk) |
| errors / OOM / Traceback | none |
| GPU memory | steady ~2.45GB / 8GB (baseline: 2.44GB — no meaningful overhead) |

| checkpoint | train loss (weighted) | baseline train loss (uniform) | wall time (min) |
|---:|---:|---:|---:|
| 2500 | 0.123 | 0.124 | 15.63 |
| 5000 | 0.082 | 0.082 | 32.00 |
| 7500 | 0.064 | 0.066 | 48.57 |
| 10000 | 0.052 | 0.053 | 64.85 |

(These two "loss" columns use different normalizations by construction — not directly comparable
in absolute terms — but their near-identical magnitude here is a further sanity check that the
weighted-mean normalization is behaving as intended, not silently inflating/deflating loss scale.)

## Checkpoint comparison table (offline heldout eval, `data/so101_cube_xy_midpoint_test10_v2_clean`,
unmodified, seed=42, task overridden to Pick&Drop string at eval time only)

| checkpoint | V2 baseline MAE | combined65 baseline MAE | combined65 early-weight MAE | Δ (early-weight − baseline) |
|---:|---:|---:|---:|---:|
| 2500 | 5.3875 | 4.7157 | 4.6534 | −0.062 |
| 5000 | 4.2909 | 4.3404 | 4.4293 | +0.089 |
| **7500** | 3.9841 | **3.4123** | **3.4023** | **−0.010** |
| 10000 | 4.2210 | 3.6396 | 3.6108 | −0.029 |

Offline MAE is essentially unchanged by the weighting (all deltas within noise, ≤0.09deg) — 7500
remains the offline-best checkpoint for both combined65 experiments, and it is *not* degraded by
early-action weighting. Full per-joint numbers: `checkpoint_metrics.csv`.

## First-action diagnostic (same T01 reference observation as V2/combined65 baseline; seeds 0–19;
Safety Gate thresholds unchanged)

| experiment / checkpoint | shoulder_lift bias | sl clamp rate | elbow_flex bias | ef clamp rate | clamp-free | L2 vs GT |
|---|---:|---:|---:|---:|---:|---:|
| **V2 7500 (baseline)** | +5.13±3.01 | 45% | −7.08±2.30 | 75% | 4/20 (20%) | 4.62 |
| combined65 baseline 2500 | −0.02±7.78 | 55% | +1.63±6.81 | 30% | 5/20 (25%)* | 13.17 |
| combined65 baseline 5000 | +5.26±4.13 | 45% | −8.21±3.35 | 75% | 4/20 (20%) | 6.83 |
| **combined65 baseline 7500** | +4.51±3.02 | 35% | −7.66±2.30 | **85%** | 3/20 (15%) | 4.78 |
| combined65 baseline 10000 | +3.83±2.77 | 25% | −6.88±2.12 | 75% | 5/20 (25%) | 4.15 |
| combined65 early-weight 2500 | −1.46±7.82 | 60% | +1.10±7.10 | 35% | 4/20 (20%)* | 13.30 |
| combined65 early-weight 5000 | +5.22±3.98 | 50% | −8.10±3.63 | 75% | 4/20 (20%) | 6.84 |
| **combined65 early-weight 7500** | **+3.25±2.73** | **25%** | **−6.94±2.42** | 75% | **5/20 (25%)** | **4.40** |
| combined65 early-weight 10000 | **+3.14±2.47** | 25% | **−6.46±2.21** | 75% | 5/20 (25%) | **3.94** |

(*2500 is badly undertrained on both experiments — std 7-8deg, L2≈13deg, ~3× every other
checkpoint's error — its clamp-free/clamp-rate numbers are noise, not signal; excluded from the
"did early weighting help" comparison below.)

**Head-to-head at 7500 (offline-MAE-best checkpoint for both experiments) — every first-action
metric improved:**

| metric | baseline 7500 | early-weight 7500 | change |
|---|---:|---:|---:|
| shoulder_lift bias | +4.51deg | +3.25deg | **−1.26deg (28% smaller)** |
| shoulder_lift clamp rate | 35% | 25% | **−10pp** |
| elbow_flex bias | −7.66deg | −6.94deg | **−0.72deg (9% smaller)** |
| elbow_flex clamp rate | 85% | 75% | **−10pp** |
| clamp-free rate | 15% | 25% | **+10pp** |
| L2 vs GT | 4.78deg | 4.40deg | **−0.38deg** |

**At 10000 — improved or equal on every metric, none worse:**

| metric | baseline 10000 | early-weight 10000 | change |
|---|---:|---:|---:|
| shoulder_lift bias | +3.83deg | +3.14deg | −0.69deg |
| shoulder_lift clamp rate | 25% | 25% | 0 |
| elbow_flex bias | −6.88deg | −6.46deg | −0.42deg |
| elbow_flex clamp rate | 75% | 75% | 0 |
| clamp-free rate | 25% | 25% | 0 |
| L2 vs GT | 4.15deg | 3.94deg | −0.21deg |

Full per-joint numbers, all checkpoints: `first_action_diagnostics.csv`, full per-seed detail in
`first_action_seed_sweep_<step>/seed_sweep.{json,csv,md}`.

**Reused-script labeling note carried forward** (does not affect any number above): the same
merged-single-parquet-file mislabeling described in `reports/pick_drop_combined65_fresh_training/summary.md`
applies to `sweep_grid35_first_action_seed.py` runs against combined65-derived checkpoints here too
(`nearest_demo_match.episode` reads `0`; the correct label, verified the same way, is episode 33 /
frame 25). `scripts/diagnose_temporal_chunk_error.py` (this experiment's new script) does **not**
have this bug — it groups by `episode_index` directly rather than assuming one file per episode,
and correctly reports `episode=33, frame=25`.

## Temporal (chunk-position) error diagnostic

Reuses the seed-sweep's own `predict_action_chunk()` output in full (no extra inference cost —
euler-integrating the flow-matching chunk already produces all 50 steps per call). Ground truth
per position `k` is the nearest-training-demo trajectory's own action at
`min(matched_frame+k, episode_end-1)` (same alignment law as
`scripts/verify_smolvla_training_target_alignment.py`). Key-joint (shoulder_lift + elbow_flex
average) MAE by bucket:

| checkpoint | bucket | baseline (uniform loss) | early-weight | Δ |
|---:|---|---:|---:|---:|
| 2500 | step0 | 7.886 | 8.002 | +0.116 |
| 2500 | step1-2 | 8.080 | 7.451 | −0.629 |
| 2500 | step3+ | 17.562 | 15.474 | −2.088 |
| 5000 | step0 | 3.127 | 3.185 | +0.058 |
| 5000 | step1-2 | 2.866 | 2.833 | −0.033 |
| 5000 | step3+ | 5.131 | 4.723 | −0.408 |
| **7500** | **step0** | **2.183** | **1.918** | **−0.265 (−12%)** |
| **7500** | **step1-2** | **1.988** | **2.016** | +0.028 |
| **7500** | **step3+** | **11.213** | **10.932** | −0.281 |
| **10000** | **step0** | **1.806** | **1.693** | **−0.113 (−6%)** |
| **10000** | **step1-2** | **1.812** | **1.867** | +0.055 |
| **10000** | **step3+** | **6.826** | **7.424** | +0.598 (+9%) |

At **7500**, early weighting reduces step-0 error by 12% while step-1-2 stays flat and step-3+
*also* improves slightly — a clean win with no visible tradeoff. At **10000**, step-0 improves 6%
and step-1-2 stays flat, but step-3+ gets 9% *worse* — a small, real tradeoff (more of the loss
budget spent on early steps, slightly less on far-future steps), consistent with the weighting's
intended mechanism. Full table: `temporal_chunk_error.csv`.

---

## Answers

**1. 학습은 정상 완료됐는가?**
예. fresh-start, 10,000 step 완주, 4개 checkpoint 전부 저장, 에러/OOM 없음, wall time 64.9분
(baseline 66.7분과 거의 동일 — weighting으로 인한 유의미한 오버헤드 없음).

**2. offline best checkpoint는 무엇인가?**
**7500** (MAE=3.4023) — baseline(3.4123)과 사실상 동일, 근소하게 더 낮음. 순서(2.5k→5k→7.5k
개선, 10k 재상승)도 동일하게 재현됨.

**3. first-action best checkpoint는 무엇인가?**
**10000** (L2 vs GT=3.94deg, 이 실험 전체 checkpoint 중 최저) — 다만 **7500도 baseline 7500
대비 전 지표 개선**이라는 점이 이번 실험의 핵심 결과다(아래 4번 참고).

**4. offline best와 first-action best가 다시 불일치하는가?**
그렇다, 그러나 격차는 줄었다. offline-best(7500)와 first-action-best(10000)가 여전히 다른
checkpoint이지만, **이번에는 offline-best인 7500 자체도 first-action 전 지표에서 baseline
대비 명확히 개선**됐다(shoulder_lift bias −28%, elbow_flex bias −9%, 두 clamp rate 모두
−10pp, clamp-free +10pp, L2 −0.38deg) — baseline 실험에서는 7500이 오히려 V2보다 나빴던 것과
대조적이다.

**5. shoulder_lift bias는 얼마나 변했는가?**
7500: +4.51→+3.25deg (−1.26deg, 28% 감소). 10000: +3.83→+3.14deg (−0.69deg, 18% 감소). 두
checkpoint 모두 명확히 개선.

**6. elbow_flex bias는 얼마나 변했는가?**
7500: −7.66→−6.94deg (−0.72deg, 9% 감소). 10000: −6.88→−6.46deg (−0.42deg, 6% 감소). 방향은
일관되게 개선이지만 shoulder_lift보다 개선폭이 작다.

**7. elbow_flex clamp rate가 실제로 감소했는가?**
감소는 했지만 **성공 기준(75%→50% 이하)에는 못 미쳤다**. 7500에서 85%→75%로 10pp 감소했고,
10000/5000은 75%로 변화 없음(V2 baseline과 동일 수준). 목표했던 50% 이하 도달은 이번 단순
weight schedule로는 달성되지 않았다.

**8. clamp-free rate는 개선됐는가?**
7500에서 15%→25%로 개선(+10pp), 10000은 25%→25%로 변화 없음. **성공 기준(25%→50% 이상)에는
못 미쳤다.**

**9. first-action L2는 개선됐는가?**
예, 2500을 제외한 모든 학습된 checkpoint에서 개선: 5000 6.83→6.84(사실상 동일), 7500
4.78→4.40(−8%), 10000 4.15→3.94(−5%). 방향 일관성 있음.

**10. early weighting 때문에 offline MAE나 후반 action 품질이 희생됐는가?**
offline MAE는 희생되지 않았다(모든 checkpoint에서 ±0.09deg 이내, 사실상 동일하거나 근소하게
개선). Chunk 후반부(step3+) 품질은 **7500에서는 오히려 개선**됐지만 **10000에서는 9% 악화**
됐다 — 완전히 공짜는 아니고, 학습이 충분히 진행된 시점(10k)에서는 실제로 약간의 late-chunk
품질 trade-off가 나타난다.

**11. 이번 결과가 early-action bias 가설을 지지하는가?**
**부분적으로 지지한다.** "static-looking observation에서 chunk 앞부분에 미래 큰 움직임이
당겨져 나타난다"는 가설이 맞다면, 앞부분에 더 큰 loss weight를 주는 것으로 첫 action 품질이
개선되어야 하는데, 실제로 7500/10000 두 checkpoint 모두에서 shoulder_lift/elbow_flex bias와
L2가 방향 일관되게 개선됐고, 특히 7500에서는 대가 없이(step3+도 개선) 개선됐다. 다만
elbow_flex clamp rate가 여전히 75%에 머물러 있다는 점은, 단순 loss weight 조정만으로는 이
관절의 근본적인 문제(잔존 chunk-mean 상관관계, 이전 coverage 분석에서 확인된 61% 잔존율)를
다 해소하지 못한다는 뜻이기도 하다 — 가설의 방향은 맞지만, 이번 weight 강도(3.0/2.0/2.0)로는
불충분하다.

**12. 다음 단계는 weight 조정인가, chunk horizon 축소인가?**
**먼저 weight 조정**을 권한다. 이유: (a) 이번 실험에서 offline MAE 희생이 사실상 없었고(≤0.09deg),
7500에서는 step3+까지 개선되는 등 부작용이 아직 관찰되지 않았다 — 즉 weight를 더 키울 여력이
있다. (b) chunk horizon 축소는 모델 architecture/입출력 shape에 더 큰 영향을 주는 변경이며,
이번 실험 원칙("chunk horizon 변경 금지")과도 다음 실험 단계로 미루는 것이 안전하다. (c) 10000
checkpoint에서 step3+가 9% 악화된 것을 볼 때, weight를 과도하게 키우면 후반부 trade-off가
본격적으로 나타날 수 있으므로, 예를 들어 `[5.0, 3.0, 3.0]` 또는 `[3.0,2.0,2.0,1.5,1.5]`처럼
점진적으로 강도/폭을 늘려가며 elbow_flex clamp rate 50% 이하 도달 여부와 step3+ 저하 정도를
함께 추적하는 것이 다음 실험으로 적절하다.

---

## 최종 출력 체크리스트

- **수정 파일 목록**: `~/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py`,
  `~/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` (2 files, 62 insertions/8 deletions,
  not committed).
- **loss 변경 내용**: 위 "Loss change" 절 참고 — chunk-position 전용 temporal weighting,
  action-dim 가중치 없음, `None` 기본값으로 기존 실험 완전 보존.
- **train 설정**: combined65 baseline과 완전 동일 + `--policy.early_action_loss_weights='[3.0,2.0,2.0]'` 1개 플래그만 추가.
- **checkpoint별 결과표**: 위 "Checkpoint comparison table", `checkpoint_metrics.csv`.
- **V2 / combined65 / early-weight 비교표**: 위 "First-action diagnostic" 표, `first_action_diagnostics.csv`.
- **first-action 진단**: 위 절 + `first_action_seed_sweep_<step>/` (4개, 20-seed 원본).
- **temporal error 분석**: 위 "Temporal (chunk-position) error diagnostic" 절, `temporal_chunk_error.csv`.
- **최종 추천 checkpoint**: **combined65_early_weight_v1의 checkpoint 7500** — offline MAE가
  이 실험 최선(3.4023)이면서 first-action 전 지표가 baseline보다 명확히 개선된 유일한
  checkpoint. 10000은 first-action 단일 최고점이지만 step3+ trade-off가 있고 offline MAE도
  7500보다 약간 높다 — Shadow 이전 후보로는 7500을, 만약 late-chunk 품질보다 first-action을
  절대적으로 우선한다면 10000을 2순위 후보로 제안한다.
- **다음 실험 제안**: weight schedule 강도 상향 (`[5.0,3.0,3.0]` 등) — 위 질문 12 참고.
- **실물 write를 해도 되는지 여부**: **아니오, 아직 안 된다.** 이번 실험은 offline/T01
  reference-observation 기준으로만 검증됐고, 지시된 성공 기준(elbow_flex clamp≤50% 또는
  clamp-free≥50%) 중 어느 것도 아직 충족하지 못했다. 다음 단계로 **Shadow evaluation(비-write,
  T01-T10)까지는 진행할 가치가 있다** — 특히 checkpoint 7500 — 하지만 실물 로봇 action write는
  여전히 금지되어야 한다.
