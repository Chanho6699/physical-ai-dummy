# Combined65 Reweight 2:1 + Early-Action Loss — `combined65_reweight2_early_action_v1`

Real GPU experiment (RTX 3050 8GB, `~/lerobot` venv). Fresh 10k ablation on top of the current
best strategy — **combined65 (65ep) with old35:reinforcement30 sampling reweighted 1:2 (`combined65_reweight_new2_old1_v1`)**
— adding exactly one new variable: `early_action_loss_weights`. Sampling ratio, dataset
(`data/so101_cube_pick_drop_combined65_v1`), task string, model architecture, chunk horizon
(chunk_size=50), optimizer/scheduler, seed (1000), `empty_cameras=1`, camera rename map,
Safety Gate thresholds — all byte-identical to the `combined65_reweight_new2_old1_v1` baseline.
**V4 was not mixed into this training.** No real-robot writes at any point. No dataset/checkpoint
mutation. No git commit/push.

## 0. Disk preflight

`df -h` on the project's filesystem (`/dev/sdd` mounted at `/`): **823G free of 1007G (14% used)**
before this run started. Largest existing output dirs (`du -sh outputs/*`) topped out at 9.9G
(`outputs/pick_drop_combined65`, mostly older combined65 runs). This training's own checkpoints
(4 x ~1.2GB) fit trivially inside the free margin. **No cleanup was necessary and none was
performed** — nothing was deleted, moved, or overwritten. All protected datasets/checkpoints listed
in the task remain untouched (verified present, unmodified timestamps, before and after this run).

## 1. `early_action_loss_weights` — investigated, reused verbatim (no new implementation)

Searched the repo for an existing, already-verified implementation before writing anything new.
Found one: `combined65_early_weight_v1` (`reports/pick_drop_combined65_early_weight_v1/summary.md`),
a prior single-variable experiment that added this exact feature and validated it with a unit test
+ two live smoke tests before spending GPU time. Reused as-is:

| aspect | value | source |
|---|---|---|
| implementation | `SmolVLAPolicy.forward()`'s loss reduction calls a new `_chunk_step_weights()` helper in `~/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`; config field `early_action_loss_weights: list[float] \| None` in `configuration_smolvla.py` | already present in `~/lerobot` working tree (`git diff --stat`: 2 files, +62/-8, uncommitted, same as the original experiment) |
| applies to | chunk-position (temporal) only, **no per-action-dimension weighting** — a `(1, chunk_size, 1)` weight vector broadcasts over batch/action dims | `modeling_smolvla.py` |
| weight values used | `[3.0, 2.0, 2.0]` → step 0 = 3.0x, steps 1-2 = 2.0x, steps 3..49 = 1.0x (unweighted) | same list as `combined65_early_weight_v1`, reused verbatim, no new sweep |
| loss normalization | generalizes the pre-existing `sum(loss)/num_valid` to a weighted mean `sum(loss_t * w_t * mask)/sum(w_t * mask)` — the `w_t≡1` case reproduces the original formula bit-for-bit | verified by the original experiment's unit test |
| default (`None`) behavior | byte-identical to every experiment that never sets this field (V2, combined65 baseline, reweight2:1, reweight3:1) | verified by the original experiment's regression smoke test (5-step run reproduced identical losses with the flag unset) |
| prior standalone result | on top of **uniform** combined65 sampling: offline MAE unchanged (±0.09deg), first-action bias/clamp improved at 7500 (no step3+ cost) and at 10000 (small step3+ cost, +9%) | `reports/pick_drop_combined65_early_weight_v1/summary.md` |

Per the task instructions, since a validated implementation and config already existed, it was
reused as-is — **no new loss implementation, no weight sweep** in this experiment.

## 2. Dataset / sampling — 2:1 kept, draw-tested before spending GPU time

`data/so101_cube_pick_drop_combined65_v1` (65 episodes, 21,327 frames), task
`"Pick up the cube and drop it into the bin."`. Reused the existing
`build_episode_group_weighted_sampler()` (`~/lerobot/src/lerobot/datasets/sampler.py`) and
`--episode_group_sample_weights='{"0-34":1.0,"35-64":2.0}'` flag unchanged from
`combined65_reweight_new2_old1_v1` — **not 3:1**.

Direct sampler draw test (all 21,327 indices, seed=1000) before training:

| group | episodes | frames | target share | measured share |
|---|---:|---:|---:|---:|
| old35 (ep 0-34) | 35 | 11,505 | 33.3% | **33.29%** |
| reinforcement30 (ep 35-64) | 30 | 9,822 | 66.7% | **66.71%** |

Matches 2:1 to within sampling noise. Live training log printed the identical breakdown
(`group 0-34: ... expected sampling share=33.3%`, `group 35-64: ... expected sampling
share=66.7%`), confirming the configured ratio is what actually ran.

## 3. Preflight

5-step `lerobot-train` smoke run (`--steps=5 --save_checkpoint=false`) with **both** flags
(`--episode_group_sample_weights` + `--policy.early_action_loss_weights`) together through the
real training pipeline: losses `0.651 → 1.192 → 0.604 → 0.642 → 2.317` (finite), sampler log
confirmed 33.3%/66.7%, GPU mem ~2.44GB, no errors. **PASS** — full training launched.

## 4. Training

Fresh-started from `lerobot/smolvla_base` (no resume, no reused optimizer state). Identical
command to `combined65_reweight_new2_old1_v1` plus one flag:

```
lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --policy.early_action_loss_weights='[3.0,2.0,2.0]' \
  --rename_map='{"observation.images.workspace":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --dataset.repo_id=local/so101_cube_pick_drop_combined65_v1 \
  --dataset.root=.../data/so101_cube_pick_drop_combined65_v1 \
  --episode_group_sample_weights='{"0-34":1.0,"35-64":2.0}' \
  --output_dir=outputs/pick_drop_combined65_reweight2_early/smolvla_pick_drop_combined65_reweight2_early_fresh \
  --job_name=smolvla_pick_drop_combined65_reweight2_early_fresh \
  --batch_size=4 --steps=10000 \
  --save_checkpoint=true --save_freq=2500 --log_freq=100 --wandb.enable=false
```

batch_size=4, lr=1e-4 (AdamW, betas 0.9/0.95, weight_decay=1e-10), cosine scheduler
(warmup 1000→333 auto-scaled), chunk_size=50, seed=1000 (default, not overridden) — all identical
to the 2:1 baseline.

| metric | value |
|---|---|
| started / ended | 2026-08-10 07:23:40 → 08:29:48 |
| total wall time | 66.1 min (2:1 baseline: 65.2 min — no meaningful overhead) |
| checkpoints saved | 002500, 005000, 007500, 010000 (all 4, confirmed on disk) |
| errors / OOM / Traceback | none |

## 5. Offline evaluation (historical heldout10, `data/so101_cube_xy_midpoint_test10_v2_clean`, unmodified, seed=42, task overridden to Pick&Drop string at eval time only)

| checkpoint | train loss | wall (min) | **B: reweight2:1 heldout MAE** | **E: reweight2:1+early heldout MAE** | Δ | E in-sample MAE | E gap (heldout−insample) | B gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 0.121 | 15.8 | 4.867 | 5.000 | +0.133 | 4.856 | 0.144 | -0.039 |
| 5000 | 0.079 | 32.3 | 4.109 | 4.114 | +0.005 | 3.751 | 0.363 | 0.270 |
| 7500 | 0.058 | 48.6 | 3.588 | 3.608 | +0.021 | 3.033 | 0.575 | 0.548 |
| **10000** | 0.053 | 65.1 | **3.528** | **3.541** | **+0.014** | 2.885 | **0.656** | **0.631** |

Heldout MAE is essentially unchanged by adding early-action weighting on top of reweight2:1 (all
deltas ≤0.13deg, shrinking with more training) — **10000 remains offline-best for both**. The
train/heldout gap is very slightly larger with early-weighting at every checkpoint (+0.03-0.18),
i.e. early-action weighting does **not** further shrink the gap the sampling fix already achieved;
if anything it costs a hair of it back. Full per-joint numbers: `checkpoint_metrics.csv`.

## 6. First-action diagnostic (same real T01 reference observation, seeds 0-19, Safety Gate thresholds unchanged)

| checkpoint | metric | B: reweight2:1 | E: reweight2:1+early | Δ |
|---:|---|---:|---:|---:|
| 7500 | shoulder_lift bias / clamp | +3.93deg / 25% | **+3.03deg / 15%** | better |
| 7500 | elbow_flex bias / clamp | −6.50deg / 65% | −6.67deg / 65% | slightly worse |
| 7500 | clamp-free | 35% | 35% | same |
| 7500 | L2 vs GT | **3.57** | 3.91 | worse |
| **10000** | shoulder_lift bias / clamp | +3.96deg / 25% | +3.80deg / **25%** | ~same |
| **10000** | elbow_flex bias / clamp | **−6.11deg / 60%** | −6.61deg / 65% | worse |
| **10000** | clamp-free | **40%** | 35% | worse |
| **10000** | L2 vs GT | **3.31** | 3.54 | worse |

At the offline-best checkpoint (10000), stacking early-action weighting on top of the already-
reweighted sampler is a **net regression on every first-action safety metric**: clamp-free rate
drops (40%→35%), elbow_flex clamp rate rises (60%→65%), elbow_flex bias magnitude grows
(−6.11→−6.61deg), L2 vs GT grows (3.31→3.54deg). Only shoulder_lift is flat-to-slightly-better.
At 7500, shoulder_lift genuinely improves (25%→15% clamp) but elbow_flex/L2/clamp-free do not.
Full detail: `first_action_diagnostics.csv`, per-seed data in `first_action_seed_sweep_<step>/`.

## 7. Temporal (chunk-position) diagnostic — 5-way comparison A-E

Key-joint (shoulder_lift + elbow_flex average) MAE by chunk-position bucket, all against the same
nearest-training-demo ground truth (episode 33 / frame 25 for A/B/C/E — trained on combined65;
episode 28 / frame 34 for D — trained on the disjoint V4 dataset):

**A** = combined65 uniform · **B** = reweight 2:1 (current best sampling) · **C** = reweight 3:1 ·
**D** = V4 fresh · **E** = reweight 2:1 + early-action loss (this experiment)

| checkpoint | bucket | A uniform | B rw2:1 | C rw3:1 | D V4 fresh | **E rw2:1+early** |
|---:|---|---:|---:|---:|---:|---:|
| 7500 | step0 | 2.183 | **1.641** | 2.021 | 2.136 | 1.653 |
| 7500 | step1-2 | 1.988 | **1.823** | 1.841 | 1.932 | 1.984 |
| 7500 | step3+ | 11.213 | 8.254 | 7.377 | **6.077** | 7.807 |
| **10000** | **step0** | 1.806 | 1.617 | 1.707 | 2.101 | **1.511** |
| **10000** | **step1-2** | **1.812** | 1.859 | 1.846 | 1.931 | 1.826 |
| **10000** | **step3+** | 6.826 | 7.435 | **6.112** | 6.265 | 7.536 |

**At 10000, E has the single lowest step0 error of all five experiments (1.511deg)** — the early-
action weighting's intended effect does show up, and it stacks on top of (not just repeats) the
2:1 sampling fix's own step0 gain. step1-2 is essentially flat vs A/B/C (1.826, mid-pack). **But
step3+ is the worst of all five at 10000 (7.536deg)** — the early-position weighting is pulling
loss budget away from the tail of the chunk more visibly here than in the standalone early-weight
experiment (which only showed a step3+ cost at 10000, not at 7500). At 7500 the pattern is
similar but smaller: step0 near-best (1.653, essentially tied with B's 1.641), step3+ mid-pack
(7.807, between C/D and A/B). Full table: `temporal_chunk_error.csv`.

Answering the diagnostic questions directly:
- **step0만 좋아지는가?** Largely yes — step0 is where the entire benefit concentrates (best of
  all 5 at 10000).
- **step1~2도 좋아지는가?** No meaningfully — flat/mid-pack at both checkpoints, no clear gain
  over B or even A.
- **step3+도 실제로 감소하는가?** No — it is the **worst of all 5** at 10000, and mid-pack (not
  best) at 7500. The tradeoff the standalone early-weight experiment showed only at 10000 is
  present at both mature checkpoints here.
- **elbow bias가 줄어드는가?** No — elbow_flex bias magnitude is *larger* with E than B at both
  7500 (−6.67 vs −6.50) and 10000 (−6.61 vs −6.11). This is the opposite of what the standalone
  early-weight-on-uniform-sampling experiment found (there, elbow bias shrank). The mechanism does
  not stack additively once the sampler has already partially addressed the same bias.

## 8. 최종 판정

**1. 2:1 sampling이 유지됐는가?**
예. draw-test 33.29%/66.71% (목표 33.3%/66.7%), 훈련 로그도 동일하게 확인. 3:1이 아님.

**2. exact early_action_loss_weights는 무엇인가?**
`[3.0, 2.0, 2.0]` (step0=3.0x, steps1-2=2.0x, steps3+=1.0x, action-dim 가중치 없음) — 이미
검증된 `combined65_early_weight_v1` 실험의 설정을 그대로 재사용, 새 sweep 없음.

**3. heldout MAE 3.53 유지/개선?**
사실상 유지(noise 수준). 10000: 3.528→3.541 (+0.014, +0.4%) — 개선은 아니지만 유의미한 악화도
아니다.

**4. gap 0.63 유지/개선?**
소폭 악화. 10000: 0.631→0.656 (+0.026, +4%). reinforcement30-only의 1.75보다는 여전히 훨씬
작지만, sampling만으로 얻은 gap 축소 효과를 early-weighting이 더 개선하지는 못했다.

**5. clamp-free 40% 초과?**
아니오. 10000: 40%→35% (−5pp) — 오히려 하락. E 실험 전체에서 최고 clamp-free는 35%(7500,
10000 동률)로 baseline의 40%에 못 미친다.

**6. elbow clamp 60% 미만?**
아니오. 10000: 60%→65% (+5pp) — 목표에서 더 멀어짐. 7500도 65%로 baseline 7500(65%)과 동일,
개선 없음.

**7. lift clamp 25% 미만?**
7500에서는 충족(25%→15%), 10000에서는 동률(25%→25%, "미만"은 아님). offline-best인 10000
기준으로는 엄밀히 미충족.

**8. L2 3.31 미만?**
아니오. 10000: 3.31→3.54deg (+0.23, +7%) — 악화. 7500도 3.57→3.91deg로 악화.

**9. step3+ error 개선?**
아니오, 오히려 10000에서 이번 실험 전체 A-E 중 최악(7.536deg, baseline 7.435 대비 +1.4%). 7500은
mid-pack (7.807, baseline 8.254 대비는 개선이지만 C/D보다는 나쁨).

**10. offline-best와 safety-best checkpoint가 일치?**
대체로 일치. offline-best=10000 (heldout MAE 최저). E 자체 checkpoint 중 L2 vs GT도 10000이
최저(3.54)이고 clamp-free도 7500과 동률 최고(35%)라 10000이 사실상 safety-best이기도 하다 —
유일한 예외는 shoulder_lift clamp rate가 7500(15%)이 10000(25%)보다 낮다는 점.

**11. 이 checkpoint를 MuJoCo full rollout benchmark에 투입할 가치가 있는가?**
**아니오, 이 checkpoint(10000)를 우선 투입할 근거는 부족하다.** heldout MAE는 noise 수준으로
유지됐지만, first-action safety 지표(clamp-free, elbow clamp rate, L2 vs GT) 전부와 gap이
기존 reweight2:1 baseline보다 소폭 악화됐고, step3+ chunk error도 A-E 중 최악이다. step0만
놓고 보면 A-E 중 최고치(1.511deg)를 기록해 "early-action bias 완화"라는 원래 가설의 방향성은
다시 한번 확인됐지만, 이번 실험에서는 그 효과가 뒷부분 chunk 품질과 elbow_flex safety 지표를
희생하는 방식으로만 나타났다. **기존 reweight2:1 checkpoint 10000이 여전히 최선의 MuJoCo 투입
후보**이며, 이번 checkpoint를 그 자리에 대체 투입할 이유는 없다.

**12. 다음 후보로 V4+V3 조합을 실험할 가치가 있는가?**
**예, 가치가 있다.** 이번 실험은 "같은 데이터 풀(combined65) 위에서 loss weighting을 더
쌓는" 접근이 이미 reweight2:1로 상당 부분 해소된 문제에 대해 추가 이득 없이 부작용(step3+
저하, elbow clamp 상승)만 낳을 수 있음을 보여준다 — loss-engineering lever가 이 데이터 조합
에서는 거의 소진된 신호다. 반면 D(V4 fresh)는 step3+에서 A-E 중 최상위권(6.265, C 다음)이면서
step0은 오히려 가장 나쁜(2.101) 서로 다른 특성 프로파일을 보였다 — V4의 generalization
데이터가 late-chunk 품질에는 도움이 되지만 first-action bias에는 별도 해법이 필요하다는
뜻이다. V3(start-coverage 다양성)를 V4와 결합하면 V4의 late-chunk 이점을 유지하면서 V3의
시작 자세 다양성으로 first-action bias를 보완할 수 있는지 확인할 수 있다 — 이는 이번 실험과
달리 데이터 구성(data composition) 레버를 쓰는 것이라 loss-weighting 레버가 소진된 지금
시점에 시도해 볼 가치가 충분하다.

---

## 최종 출력 체크리스트

- **디스크 preflight**: 823G/1007G 여유 (14% 사용), cleanup 불필요 — **아무것도 삭제하지 않음**.
  보호 대상 dataset/checkpoint 전부 수정 없이 확인됨 (사전 목록: combined65_v1, grid35_v2_clean,
  start_coverage_v3_clean, generalization_v4, v4_heldout10, historical heldout test dataset,
  reweight2:1 checkpoint/report, V4 fresh best checkpoint/report, 현재 baseline report/CSV —
  모두 read-only 접근만 수행).
- **early_action_loss_weights 조사 결과**: 기존 검증된 구현/설정(`[3.0,2.0,2.0]`) 발견, 그대로
  재사용, 새 구현/새 sweep 없음. 근거: 위 1절.
- **sampling 검증**: draw-test 33.29%/66.71%, 훈련 로그로 재확인. 위 2절.
- **checkpoint별 결과표**: `checkpoint_metrics.csv`, `first_action_diagnostics.csv`.
- **temporal/chunk 비교(A-E)**: `temporal_chunk_error.csv`, 위 7절.
- **reweight2 대비 직접 비교**: `comparison_vs_reweight2.csv`.
- **최종 판정**: 위 8절, 질문 1-12 전부 답변.
- **실물 write 여부**: 이번 실험에서 실물 follower write는 전혀 수행하지 않았음(오프라인
  평가 + T01 reference 기반 diagnostic만). 이 checkpoint의 실물/MuJoCo 투입은 위 11번 답변대로
  권장하지 않음 — 기존 reweight2:1 checkpoint 10000이 여전히 최선 후보.
- **Git commit/push**: 수행하지 않음.
