# V3:V4 = 2:1 Sampling Ablation — `pick_drop_v3_v4_reweight2_v1`

Goal: on top of `data/so101_cube_pick_drop_v3_v4_combined69_v1` (reused unmodified, no re-merge),
reweight V3:V4 sampling 2:1 to see if V3+V4 uniform's big safety-clamp gains can be kept while
recovering some of the accuracy/L2/gap it gave up vs the current-best V2+V3 reweight2:1. Uniform
sampling not used, 3:1 not used, no loss tricks. Fresh `lerobot/smolvla_base`, identical
hyperparameters to every prior run. No real-robot writes, no git commit/push.

**Result: the goal was not achieved.** No checkpoint recovers accuracy toward the V2+V3 baseline,
and no checkpoint cleanly preserves V3+V4 uniform's safety profile either — the ratio produces an
unstable, checkpoint-dependent trade-off instead.

## Disk preflight
813G/1007G free (15%) — no cleanup needed.

## Sampling verification

Reused `data/so101_cube_pick_drop_v3_v4_combined69_v1` as-is (episodes 0-29=V3, 30-68=V4, confirmed
against source frame counts, no re-merge performed). `--episode_group_sample_weights='{"0-29":2.0,"30-68":1.0}'`.
Draw test (22,617 draws): **V3 66.44% / V4 33.56%** (target 66.7%/33.3%) — matches 2:1. Training log
confirmed the identical breakdown. Preflight 5-step smoke test passed before the full run.

## Training

Fresh, 10000/10000 steps, **66.5 min**, 4 checkpoints saved, no errors. Otherwise byte-identical
command/hyperparameters to the uniform run.

## Full checkpoint results (this experiment)

| ckpt | historical MAE | V4 heldout6 MAE | in-sample MAE | gap | clamp-free | sl clamp | ef clamp | L2 vs GT | step0 | step1-2 | step3+ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 5.479 | 5.653 | 5.373 | 0.106 | 35% | 30% | 30% | 10.07 | 5.17 | 5.86 | 8.92 |
| 5000 | 4.210 | 4.414 | 3.670 | 0.539 | 30% | 20% | 55% | 6.11 | 2.88 | 2.63 | 3.72 |
| **7500** | 3.986 | 3.590 | 2.741 | 1.245 | **70%** | **15%** | **25%** | 4.90 | 2.59 | 3.30 | **11.13** |
| **10000** | 3.843 | 3.609 | 2.701 | 1.142 | 20% | 25% | **80%** | 4.66 | 1.75 | 1.85 | 8.59 |

Two very different checkpoints stand out for opposite reasons: **7500 has the best first-action
safety profile of the entire project** (clamp-free 70%, sl 15%, ef 25% — all meet or beat every
target) but **the worst step3+ chunk error of the entire project (11.13deg)**. **10000** is the
offline-best (lowest historical MAE) but has the **worst elbow_flex clamp rate ever recorded
(80%)** and the worst clamp-free rate of any V3+V4-family checkpoint (20%).

## Direct comparison vs the two reference points @10000

| metric | A: V2+V3 reweight2:1 | F: V3+V4 uniform | **G: this (2:1) @10000** | **G @7500 (safety-best)** |
|---|---:|---:|---:|---:|
| historical MAE | **3.53** | 3.78 | 3.84 (worse than both) | 3.99 |
| gap | **0.63** | 1.16 | 1.14 (~tied w/ F) | 1.25 (worse) |
| clamp-free | 40% | 65% | 20% (worst) | **70% (best ever)** |
| shoulder_lift clamp | 25% | **10%** | 25% (=A) | 15% (meets target) |
| elbow_flex clamp | 60% | 30% | **80% (worst ever)** | **25% (best)** |
| L2 vs GT | **3.31** | 4.38 | 4.66 (worst) | 4.90 (worst) |
| step3+ | 7.435 | **7.231** | 8.59 (worse) | **11.13 (worst ever)** |
| V4 heldout6 | — | **3.52** | 3.61 (worse) | 3.59 (worse) |

At 10000, G loses to F on every single metric except gap (negligibly). At 7500, G's first-action
numbers beat everything else in the project, but its step3+ is catastrophically worse than every
other checkpoint ever measured — since the deployed policy executes the predicted chunk beyond
just index 0, this checkpoint's excellent T01-single-step safety numbers do not translate into a
trustworthy full-trajectory candidate.

## 12가지 핵심 판정

**1. 실제 V3:V4 draw ratio는?** 66.44%/33.56% (목표 66.7%/33.3%) — 2:1 정확히 구현됨.

**2. historical MAE 개선?** 아니오. 모든 checkpoint에서 F(uniform)보다도 나쁨 (10000: 3.84 vs 3.78).

**3. V4 heldout6 성능 유지?** 아니오, 소폭 악화. 10000: 3.61 vs F의 3.52.

**4. clamp-free 60% 이상 유지?** checkpoint마다 다름 — **10000은 20%로 완전히 무너짐**, **7500은
70%로 오히려 F(65%)보다 좋음**. 단일 checkpoint로 일관되게 유지되지 않음.

**5. elbow clamp 40% 이하 유지?** 마찬가지로 불안정. 10000=80%(최악), 7500=25%(최고). 하나의
checkpoint가 두 목표를 동시에 만족하지 못함.

**6. lift clamp 15% 이하 유지?** 7500만 15%로 정확히 달성. 10000은 25%로 A와 동일(목표 미달).

**7. L2가 4.38보다 개선?** 아니오. 모든 checkpoint에서 F의 4.38보다 나쁨 (10000: 4.66, 7500: 4.90).

**8. gap이 1.16보다 개선?** 근소하게. 10000: 1.142 (−0.022, −1.9%) — 사실상 동률, 의미 있는
개선은 아님.

**9. step3+가 유지/개선?** 아니오. 10000=8.59(F의 7.231보다 악화), **7500=11.13로 프로젝트 전체
역대 최악** — 개선은커녕 심각하게 악화.

**10. 가장 좋은 checkpoint는?** 단일 후보를 고를 수 없음. **offline-best=10000**(historical MAE
최저)이지만 safety가 최악. **safety-best=7500**(clamp 지표 역대 최고)이지만 step3+가 역대 최악.
둘 다 "가장 좋은 checkpoint"라고 부르기 어렵다 — 서로 다른 축에서 극단적으로 갈린다.

**11. offline-best와 safety-best가 같은가?** **전혀 다르다** (10000 vs 7500), 그리고 이번
실험에서는 그 격차가 이전 어느 실험보다도 크다 — safety-best(7500)는 step3+가 파탄 수준이라
실제로 안전하다고 보기도 어렵다(첫 액션만 안전하고 그 이후 궤적은 신뢰 불가).

**12. V3+V4 uniform과 reweight2:1 중 어떤 모델을 MuJoCo full rollout에 보낼 것인가?**
**V3+V4 uniform(F)을 보낸다. 이번 reweight2:1(G)의 어떤 checkpoint도 보내지 않는다.** 10000은
F보다 모든 면에서 나쁘고, 7500은 첫 액션 지표만 보면 매력적이지만 step3+ 파탄으로 실제 rollout에서
신뢰할 수 없다 — Safety Gate가 매 스텝을 검사한다 해도, chunk 후반부가 이렇게 크게 틀어지면
경로 자체가 발산할 위험이 크다.

## 다음 단계

지시대로 **추가 ratio sweep은 자동으로 하지 않는다.** MuJoCo full rollout benchmark로 넘어갈
후보는 원래 계획한 3개가 아니라 **2개**로 확정한다:

- **accuracy-oriented**: V2+V3 reweight2:1 (`outputs/reweight_ablation/combined65_reweight_new2_old1_v1/checkpoints/010000`)
- **safety-oriented**: V3+V4 uniform (`outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/010000`)
- **compromise candidate**: 없음 — 이번 실험(V3:V4 2:1)은 목표(정확도 회복 + 안전성 유지)를
  달성하지 못했으므로 후보에서 제외.

## 산출물

`reports/pick_drop_v3_v4_reweight2_v1/`: `checkpoint_metrics.csv`, `first_action_diagnostics.csv`,
`temporal_chunk_error.csv`, `comparison_vs_reweight2.csv`, `summary.json`,
`first_action_seed_sweep_<step>/`, `temporal_chunk_error/v3_v4_reweight2_<step>/`.
Offline eval 원본: `reports/pick_drop_v3_v4_reweight2_v1_historical_offline_eval/`,
`reports/pick_drop_v3_v4_reweight2_v1_v4heldout6_offline_eval/`.

**Cleanup**: 디스크 여유 충분해서 아무것도 삭제하지 않음. 보호 대상 dataset/checkpoint 전부
원본 그대로 확인. 실물 follower write 없음. Git commit/push 없음.
