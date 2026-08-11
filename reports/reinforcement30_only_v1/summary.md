# reinforcement30-only Ablation — `reinforcement30_only_v1`

Real GPU experiment (RTX 3050 8GB). Single-variable change vs the combined65 baseline: **data
composition only**. Trained on `data/so101_cube_pick_drop_start_coverage_v3_clean` (the
"reinforcement30" / V3 Pick&Drop dataset) **alone** — old35 (V2) completely excluded. Original
(uniform) loss, no early-action weighting. Model architecture, chunk horizon (chunk_size=50),
optimizer/scheduler, normalization/action representation, and Safety Gate thresholds unchanged
from every prior experiment. No real-robot writes. No dataset/checkpoint mutation. No git
commit/push.

## Preflight (before spending any GPU time)

| check | result |
|---|---|
| reinforcement30 episode count | **30** (confirmed via `meta/episodes` parquet, and independently via the real training pipeline's own `dataset.num_episodes=30` log line) |
| old35 mixed in? | **No** - `data/so101_cube_pick_drop_start_coverage_v3_clean` was built (prior task) by copying+retasking only the original V3 source directory; never touched or merged with V2 data. Episode indices are exactly `0..29`, continuous, no duplicates (verified against both `meta/episodes` and every `data/*.parquet` file's own `episode_index` column - identical sets). |
| frame count | **9822** (exact match to spec, verified via both the raw parquet and the training pipeline's own `dataset.num_frames=9822` log line) |
| feature schema vs combined65 | `features_equal_for_merge(combined65, reinforcement30) = True`, fps equal (30), robot_type equal (`so_follower`) - same normalization/policy config applies to both without modification |
| NaN / Inf | none found |
| task string | `"Pick up the cube and drop it into the bin."` (single task, matches all other Pick&Drop experiments) |
| first-action seed/env | same T01 Shadow reference observation (`reports/grid35_v2_shadow_T01/shadow_20260808_211555.json`), same seeds 0-19, same `configs/safety_gate.yaml` thresholds, same `evaluate_smolvla_midpoint.py`/`sweep_grid35_first_action_seed.py`/`diagnose_temporal_chunk_error.py` scripts unmodified as every prior experiment |
| 5-step smoke test | ran cleanly through the real `lerobot-train` pipeline, finite losses, no errors |

**Preflight PASSED** - full training launched.

## Training

Identical command to the combined65 baseline (no `--policy.early_action_loss_weights` flag - this
is deliberately the *original* loss), only `--dataset.repo_id`/`--dataset.root` point at
reinforcement30 and `--output_dir`/`--job_name` are new. Fresh-started from `lerobot/smolvla_base`.

| metric | value |
|---|---|
| started / ended | 2026-08-09 15:57:52 → 17:04:33 |
| total wall time | 65.7 min (combined65 baseline: 66.7 min, early-weight: 64.9 min - no meaningful difference despite 30 vs 65 episodes, since wall time is step-count-bound, not dataset-size-bound) |
| checkpoints saved | 002500, 005000, 007500, 010000 (all 4, confirmed on disk) |
| errors / OOM / Traceback | none |
| GPU memory | steady, no pressure |

## Checkpoint comparison — offline MAE, in-sample fit, and the train/heldout gap (the overfitting
concern flagged for a 30-episode dataset)

| checkpoint | heldout MAE | in-sample (train-set) MAE | gap (heldout − in-sample) |
|---:|---:|---:|---:|
| 2500 | 5.4743 | 4.2267 | 1.25 |
| 5000 | 4.1449 | 2.7544 | 1.39 |
| **7500** | **3.8812** | 2.2492 | 1.63 |
| 10000 | 3.9760 | 2.2234 | 1.75 |

The gap **widens monotonically through training** (1.25→1.39→1.63→1.75deg) while in-sample loss
plateaus after 7500 (2.25→2.22, essentially flat) but heldout MAE gets *worse* 7500→10000
(3.88→3.98) — the textbook train/val divergence signature of overfitting on a small (30-episode)
dataset. This is a real, measured cost of dropping to 30 episodes, not a hypothetical one.

## 4-way offline MAE comparison

| checkpoint | V2 baseline (35ep) | combined65 baseline (65ep) | combined65 early-weight (65ep) | **reinforcement30-only (30ep)** |
|---:|---:|---:|---:|---:|
| 2500 | 5.3875 | 4.7157 | 4.6534 | 5.4743 |
| 5000 | 4.2909 | 4.3404 | 4.4293 | 4.1449 |
| **7500** | 3.9841 | **3.4123** | **3.4023** | 3.8812 |
| 10000 | 4.2210 | 3.6396 | 3.6108 | 3.9760 |

reinforcement30-only's offline heldout MAE is **worse than both combined65 variants at every
checkpoint** (fewer episodes → less generalization headroom, as expected) but **better than V2** at
5000/7500/10000 despite having 5 fewer episodes than V2's 35 — i.e. per-episode data quality
(diverse start poses, varied hold time) outweighs raw episode count here, but doesn't fully close
the gap to the 65-episode datasets.

## First-action diagnostic (same T01 reference observation, seeds 0-19, unchanged Safety Gate
thresholds)

| experiment / checkpoint | shoulder_lift bias | sl clamp | elbow_flex bias | ef clamp | clamp-free | L2 vs GT |
|---|---:|---:|---:|---:|---:|---:|
| V2 7500 | +5.13±3.01 | 45% | −7.08±2.30 | 75% | 4/20 (20%) | 4.62 |
| combined65 baseline 7500 | +4.51±3.02 | 35% | −7.66±2.30 | 85% | 3/20 (15%) | 4.78 |
| combined65 early-weight 7500 | +3.25±2.73 | 25% | −6.94±2.42 | 75% | 5/20 (25%) | 4.40 |
| **reinforcement30-only 2500** | +3.29±6.97 | 35% | −9.76±5.67 | 75% | 2/20 (10%)* | 12.14 |
| **reinforcement30-only 5000** | +1.79±3.79 | 20% | −7.59±2.39 | 70% | 4/20 (20%) | 6.28 |
| **reinforcement30-only 7500** | **+1.61±2.79** | **10%** | **−5.88±1.79** | **60%** | **8/20 (40%)** | **4.58** |
| **reinforcement30-only 10000** | +1.41±2.72 | 10% | −6.91±1.68 | 75% | 5/20 (25%) | 4.97 |

(*2500 undertrained on every experiment - std 5-8deg, L2>10deg - excluded from comparison as noise.)

**reinforcement30-only 7500 is, by a clear margin, the best first-action checkpoint across all
four experiments run so far:**

| metric | best of {V2, combined65 baseline, early-weight} | reinforcement30-only 7500 | improvement |
|---|---:|---:|---:|
| shoulder_lift bias | +3.25deg (early-weight) | **+1.61deg** | 50% smaller |
| shoulder_lift clamp rate | 25% (early-weight) | **10%** | 15pp lower |
| elbow_flex bias | −6.94deg (early-weight) | **−5.88deg** | 15% smaller |
| elbow_flex clamp rate | 75% (V2/early-weight) | **60%** | 15pp lower |
| clamp-free rate | 25% (early-weight/reinf.30@10k) | **40%** | 15pp higher |
| L2 vs GT | 4.40deg (early-weight) | 4.58deg | slightly worse |

Every metric except L2-vs-GT improves, several substantially. **However**, elbow_flex clamp rate
(60%) still does not cross the 50% "방향성 성공" bar from the prior early-weight experiment, and
checkpoint 10000 **regresses** back to 75% elbow_flex clamp (matching V2/combined65-baseline
levels) — consistent with the overfitting signature above: 7500 is a narrow sweet spot, not a
stable plateau.

Full detail: `first_action_diagnostics.csv`, per-seed raw data in
`first_action_seed_sweep_<step>/seed_sweep.{json,csv,md}`.

**Nearest-demo note**: reinforcement30-only's seed sweeps match a *different* nearest training demo
(episode 2, frame 57, L2=3.245deg) than the V2/combined65 experiments (episode 33, frame 25,
L2=2.217deg) — expected and correct, since reinforcement30 doesn't contain V2's episode 33 at all.
The two experiments' L2-vs-GT numbers are each internally consistent but compare against a
genuinely different reference trajectory - a mild caveat on direct L2 comparability across the
old35/reinforcement30 boundary (does not affect the bias-mean/clamp-rate comparisons, which don't
depend on the nearest-demo match).

## Temporal (chunk-position) error diagnostic

Key-joint (shoulder_lift + elbow_flex average) MAE by bucket:

| checkpoint | step0 | step1-2 | step3+ |
|---:|---:|---:|---:|
| 2500 | 5.206 | 4.482 | 11.314 |
| 5000 | 2.744 | 2.674 | 10.524 |
| **7500** | **1.940** | 2.520 | 10.661 |
| 10000 | 1.991 | 2.077 | **7.402** |

At 7500 (the first-action/offline-best checkpoint), step0 error (1.940) is the lowest of any
checkpoint in this experiment and comparable to combined65-early-weight-7500's step0 (1.918) -
achieved with the **original, unweighted loss** on 30 episodes alone, vs. early-weight's 65
episodes + temporal reweighting needed to reach a similar number. step3+ stays elevated
(10.661deg) across 2500-7500, only dropping at 10000 (7.402) - but 10000 is also where elbow_flex
clamp rate regresses, so this isn't a straightforward "wait longer" fix. Full table:
`temporal_chunk_error.csv`.

## old35 vs reinforcement30: direct distributional comparison (reused from
`reports/grid35_v2_vs_start_coverage_v3/` - not recomputed, same underlying state/action values)

| metric | old35 (V2) | reinforcement30 (V3) |
|---|---:|---:|
| start-pose pairwise L2 median | 3.85deg | **17.88deg** (4.6x more diverse) |
| static/low-motion frame fraction | 7.6% | 13.5% |
| static segment length median | 25 frames (0.83s) | 38.5 frames (1.28s) |
| state/action first-movement median frame | 25 / 21 | 38.5 / 33.5 |
| immediate (static-seg) shoulder_lift \|delta\| mean | 0.614deg | 0.551deg |
| immediate (static-seg) elbow_flex \|delta\| mean | 2.245deg | 2.089deg |
| immediate shoulder_lift/elbow_flex WOULD_CLAMP frac | 0% / 0.11% | 0% / 0% |
| chunk-mean \|delta\| shoulder_lift/elbow_flex | 20.50 / 20.97deg | **9.53 / 10.02deg** |
| **fraction of static frames where chunk-mean exceeds WOULD_CLAMP (shoulder_lift/elbow_flex)** | **99.7% / 99.7%** | **60.7% / 61.7%** |

**The key structural fact**: even reinforcement30 *alone*, with zero old35 contamination, still has
**60-62% of its own static-looking observations paired with a large-mean future chunk** (vs old35's
near-total 99.7%). This is direct evidence the "static observation → large future chunk motion"
correlation is not an old35-specific artifact — it's inherent to how *both* datasets were
collected (pick-then-move-to-bin-then-drop, from a hold, is inherently followed by a big motion
regardless of hold-length diversity). What differs is *degree*, not *presence*.

A plausible mechanism tying this together with the diversity numbers: old35's very low start-pose
diversity (median L2 3.85deg — 35 nearly-identical starts) makes the "static-look → big future
motion" mapping easy to memorize as a near-constant rule, because the model sees the *same*
starting context paired with the *same* large-motion target 35 times over. reinforcement30's much
higher start diversity (17.88deg) means the same underlying "hold then move" structure is paired
with 30 more varied contexts, which the checkpoint-7500 results suggest makes the spurious
correlation harder (though not impossible - 60% clamp rate remains) to learn as a blanket rule.

---

## Answers

**1. reinforcement30-only 학습은 정상 완료됐는가?**
예. Preflight 전 항목 통과(30ep/9822 frames/old35 미혼입/schema 일치 확인), fresh-start, 10,000
step 완주, 4개 checkpoint 전부 저장, 에러/OOM 없음, wall time 65.7분(다른 실험들과 동일 수준).

**2. offline-best checkpoint는 무엇인가?**
**7500** (heldout MAE=3.8812) - V2/combined65/early-weight와 동일한 순위 패턴(2.5k→7.5k 개선,
10k 재상승) 재현.

**3. first-action-best checkpoint는 무엇인가?**
**7500** (clamp-free 8/20=40%, L2=4.58deg) - **offline-best와 완전히 일치한다.** 이는 이전 두
실험(combined65 baseline/early-weight)에서 offline-best(7500)와 first-action-best(10000)가
서로 달랐던 것과 대조적인, 이번 실험만의 특징이다.

**4. old35 제거만으로 shoulder bias가 개선됐는가?**
**예, 크게 개선됐다.** 7500 기준 +4.51deg(combined65 baseline) → +1.61deg(reinforcement30-only),
지금까지의 최선이었던 early-weight의 +3.25deg보다도 훨씬 낮다. clamp rate도 35%→10%로 감소.

**5. old35 제거만으로 elbow bias가 개선됐는가?**
**예, 개선됐지만 shoulder_lift만큼 극적이지는 않다.** 7500 기준 −7.66deg(combined65
baseline) → −5.88deg(reinforcement30-only), 이전 최선이었던 early-weight의 −6.94deg보다도
낮다. 하지만 여전히 5.73deg 임계값을 크게 넘는 수준이다.

**6. elbow_flex clamp rate는 얼마나 변했는가?**
7500 기준 85%(combined65 baseline) → 60%(reinforcement30-only), 25pp 감소 - 지금까지 모든
실험의 checkpoint 중 최저치. 다만 10000에서는 75%로 다시 올라간다(재악화, 아래 9번 참고).

**7. clamp-free rate는 얼마나 변했는가?**
7500 기준 15%(combined65 baseline) → 40%(reinforcement30-only), 25pp 증가 - 지금까지 모든
실험의 모든 checkpoint 중 최고치(2배 이상).

**8. first-action L2는 개선됐는가?**
소폭 개선: V2 baseline 4.62deg → reinforcement30-only 7500 4.58deg. combined65-early-weight의
4.40deg보다는 약간 높다 - L2는 이번 실험에서 유일하게 "완벽한 최고"는 아닌 지표다.

**9. offline MAE/generalization은 얼마나 희생됐는가?**
**뚜렷하게 희생됐다.** heldout MAE 자체는 combined65(3.41~3.44)보다 명확히 높다(3.88~3.98,
V2 수준으로 회귀). 더 중요하게는 **train/heldout gap이 학습할수록 계속 확대**된다(1.25deg→
1.75deg), 그리고 10000에서 in-sample loss는 정체(2.25→2.22)인데 heldout은 오히려 악화
(3.88→3.98)하는 명백한 overfitting 신호가 관찰됐다 - 30-episode 데이터셋의 좁은 coverage가
실제 대가로 나타난 것이다.

**10. old35가 주요 원인이라는 증거가 생겼는가?**
**강한 증거가 생겼다.** old35를 완전히 제거하고 reinforcement30만으로 학습한 checkpoint 7500이
first-action 거의 모든 지표(bias 크기, clamp rate, clamp-free rate)에서 지금까지의 세 실험(V2,
combined65 baseline, early-weight) 전체를 능가했다 - 특히 shoulder_lift는 압도적으로
개선됐다. 이는 "old35가 bias를 유지시키는 주요 원인 중 하나"라는 가설을 직접 뒷받침한다.

**11. 데이터 구조 문제와 loss/chunk 구조 문제 중 어느 쪽 가설이 더 강해졌는가?**
**둘 다 부분적으로 강화됐다 - 순수 Case A도 순수 Case B도 아니다.** old35 제거가 first-action을
가장 크게 개선시켰다는 점(질문 10)은 데이터 구성 가설을 강하게 지지한다. 그러나 (a)
reinforcement30 자체도 static frame의 60~62%가 여전히 chunk-mean WOULD_CLAMP을 초과하는
구조(old35의 99.7%보다는 훨씬 낮지만 0%가 아님)를 갖고 있고, (b) elbow_flex clamp rate가
60%에 머물러 목표(50% 이하)에 못 미치며, (c) 10000에서 75%로 재악화된다는 점은 순수 chunk/loss
구조 가설도 완전히 배제하지 못한다는 뜻이다. 종합하면: **old35의 낮은 start-pose 다양성(median
L2 3.85deg, 35개 거의 동일한 시작 pose)이 "static-looking → 큰 future motion" 상관관계를
암기하기 쉽게 만든 것으로 보이며, reinforcement30의 훨씬 높은 다양성(17.88deg)이 이 암기를
어렵게 만들어 bias를 줄였다**는 것이 가장 설득력 있는 해석이다 - 즉 "data 구성"과 "chunk
구조"는 서로 배타적인 가설이 아니라, data 구성(특히 start-pose 다양성)이 chunk 구조 문제의
심각도를 결정하는 조절 변수(moderator)라는 통합된 그림이다.

**12. 다음 실험은 dataset reweighting인가 stronger early weighting인가 chunk horizon 축소인가?**
**dataset reweighting/전략 조정을 최우선으로 권한다** (Case C 방향, old35 완전 제거보다는):
- reinforcement30-only는 first-action에서 최고 성능을 냈지만 offline MAE/generalization을
  희생했다(질문 9) - old35를 아예 배제하는 것은 답이 아니다.
- 다음 실험으로 제안: **combined65 안에서 reinforcement30을 oversampling하거나 old35를
  downweight/undersampling**하는 sample-weighting 실험 (예: old35:reinforcement30 = 1:2 또는
  1:3 비율로 재샘플링한 뒤 fresh training) - old35의 coverage/episode-수 이점은 유지하면서
  reinforcement30의 다양성이 학습에 더 크게 반영되도록 하는 절충안.
- 이것이 stronger early weighting(`[5.0,3.0,3.0]`)이나 chunk horizon 축소보다 우선인 이유:
  이번 실험이 이미 "data 구성이 loss weighting보다 더 큰 단일 레버"임을 보여줬으므로
  (질문 4-7의 개선폭이 early-weight 실험보다 컸다), 같은 축(data)을 더 정교하게 조정하는 것이
  다음으로 시도할 가장 유망한 단일 변수다. stronger early weighting은 reweighted-data
  checkpoint에도 나중에 추가로 결합해볼 수 있는 보완 레버로 남겨둔다.

---

## Case 판정

주어진 판정 기준(Case A/B/C)에 정확히 들어맞지 않는, **A와 C의 혼합**이다:

- **Case A 방향 증거**: 4개 실험 중 first-action 전 지표 최고(shoulder_lift clamp 10%, clamp-free
  40%, 두 관절 bias 크기 최소) - old35가 주요 기여 요인이라는 명확한 신호.
- **Case A 미달**: elbow_flex clamp 60%는 방향성 성공 기준(≤50%)에 못 미치고, checkpoint
  10000에서 75%로 재악화 - old35 제거만으로 완전히 해결되지는 않았다.
- **Case C 증거**: heldout MAE가 combined65보다 뚜렷이 높고(3.88 vs 3.41), train/heldout gap이
  학습할수록 계속 확대되며(1.25→1.75deg), 10000에서 명백한 overfitting 징후(in-sample 정체,
  heldout 악화) 관찰 - reinforcement30만으로는 coverage가 좁다는 우려가 실측으로 확인됐다.

**결론**: old35가 bias의 유의미한 원인이라는 것은 확인됐지만, 완전 제거는 generalization을
희생시킨다. 다음 단계는 old35를 유지하되 reinforcement30의 비중을 높이는 **reweighting** 실험이다.

---

## 최종 출력 체크리스트

- **정상 완료 여부**: 예 (질문 1).
- **비교표**: 위 "4-way offline MAE comparison", "First-action diagnostic" 절, `checkpoint_metrics.csv`, `first_action_diagnostics.csv`.
- **offline-best/first-action-best**: 둘 다 **7500** (질문 2-3).
- **temporal chunk error**: 위 절, `temporal_chunk_error.csv`.
- **old35 vs reinforcement30 통계 비교**: 위 절, `old35_vs_reinforcement30_distributional_stats.csv` (기존 `grid35_v2_vs_start_coverage_v3` 분석 재사용, 재계산 없음).
- **Case 판정**: A/C 혼합 (위 "Case 판정" 절).
- **다음 실험 제안**: old35:reinforcement30 sample reweighting (질문 12).
- **실물 write 여부**: **아니오, 여전히 금지.** Shadow evaluation(비-write)까지는 checkpoint 7500을
  후보로 진행할 가치가 있다 - 지금까지 나온 checkpoint 중 first-action 안전성이 가장 높지만,
  elbow_flex clamp 60%·좁은 generalization이라는 두 가지 실측 리스크를 반드시 함께 보고해야 한다.
