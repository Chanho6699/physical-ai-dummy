# V3+V4 Dataset-Composition Ablation — `pick_drop_v3_v4_combined69_uniform_v1`

Real GPU experiment. Question: does replacing V2 with the more-diverse V4 in the current-best pool
(V2+V3 reweight2:1) help, **holding everything else fixed** (uniform sampling, no loss tricks,
fresh `lerobot/smolvla_base`, identical hyperparameters)? V2 not used. Historical test10 and V4's
own heldout6 not used in training. No real-robot writes. No git commit/push.

## Disk preflight
818G/1007G free (15% used) before starting — no cleanup needed, nothing deleted.

## Merged dataset: `data/so101_cube_pick_drop_v3_v4_combined69_v1`

Built via LeRobot's official `lerobot.datasets.merge_datasets` (same function `combined65_v1` was
built with) directly from the two **original** V3-clean/V4 directories — no retask needed since
both already store the canonical task string. Checksummed before/after: **both sources proven
byte-unchanged**. Full script: `scripts/build_pick_drop_v3_v4_combined69_dataset.py`.

| check | result |
|---|---|
| episodes | 30 (V3) + 39 (V4) = **69** ✓ |
| frames | 9822 + 12795 = **22617** ✓ |
| task string | `"Pick up the cube and drop it into the bin."` (all episodes) ✓ |
| state/action dim | 6D / 6D ✓ |
| cameras | `observation.images.{workspace,wrist}`, per-episode video files present ✓ |
| episode/frame index continuity | 0..68 continuous, `dataset_from/to_index` chain verified ✓ |
| NaN/Inf | none ✓ |
| data/video file references | all resolve, row counts match episode lengths ✓ |
| **`all_ok`** | **True** (full detail: `reports/pick_drop_v3_v4_combined69_dataset_build/summary.json`) |

## Training

Fresh from `lerobot/smolvla_base`, uniform sampling (no `episode_group_sample_weights`), no
`early_action_loss_weights` — otherwise byte-identical command/hyperparameters to every prior
V2/V3-family run (batch=4, AdamW lr=1e-4, cosine schedule, chunk_size=50, seed=1000,
`empty_cameras=1`, camera rename). Preflight 5-step smoke test passed first (69ep/22617 frames
loaded, uniform sampler confirmed, finite losses). Full run: 10000/10000 steps, **65.9 min**, 4
checkpoints saved, **no errors**.

## Checkpoint performance

| ckpt | train loss | historical test10 MAE | V4 heldout6 MAE | in-sample MAE | gap (hist.) |
|---:|---:|---:|---:|---:|---:|
| 2500 | 0.143 | 5.063 | 5.160 | 4.427 | 0.636 |
| 5000 | 0.083 | 4.299 | 4.218 | 3.229 | 1.071 |
| 7500 | 0.059 | 3.779 | 3.709 | 2.789 | 0.990 |
| **10000** | 0.053 | **3.780** | **3.524** | 2.616 | **1.164** |

Historical-MAE curve **plateaus/flattens between 7500→10000** (3.779→3.780, essentially tied) —
unlike every V2/V3-family experiment's monotonic improvement to 10000. V4-heldout6 keeps improving
monotonically and is the better of the two heldout numbers throughout.

## Comparison vs current best (B = reweight2:1) — full table: `comparison_vs_reweight2.csv`

| metric @10000 | B (reweight2:1) | **F (V3+V4 uniform)** | verdict |
|---|---:|---:|---|
| historical test10 MAE | **3.528** | 3.780 | worse (+7%) |
| train/heldout gap | **0.631** | 1.164 | worse (+84%) |
| clamp-free rate | 40% | **65%** | **much better** |
| shoulder_lift clamp | 25% | **10%** | **much better** |
| elbow_flex clamp | 60% | **30%** | **much better** |
| L2 vs GT (first-action) | **3.31** | 4.38 | worse |
| step0 chunk MAE | **1.617** | 2.573 | worse |
| step1-2 chunk MAE | **1.859** | 2.793 | worse |
| step3+ chunk MAE | 7.435 | **7.231** | slightly better |

**F vs D (V4 standalone, no V3)**: F beats D on *both* heldout sets — historical test10
3.780 vs D's 4.064 (−7%), V4-heldout6 3.524 vs D's 3.563 (−1%) — so V3+V4 is unambiguously better
than V4 alone.

## 6-way temporal/chunk comparison (A-F, key-joint MAE, `temporal_chunk_error.csv`)

A=V2+V3 uniform · B=reweight2:1 · C=reweight3:1 · D=V4 standalone · E=reweight2:1+early ·
**F=V3+V4 uniform (this)**

| @10000 | A | B | C | D | E | **F** |
|---|---:|---:|---:|---:|---:|---:|
| step0 | 1.806 | 1.617 | 1.707 | 2.101 | 1.511 | **2.573 (worst)** |
| step1-2 | 1.812 | 1.859 | 1.846 | 1.931 | 1.826 | **2.793 (worst)** |
| step3+ | 6.826 | 7.435 | **6.112** | 6.265 | 7.536 | 7.231 |

F is the **worst of all 6 on raw chunk-accuracy vs nearest-demo ground truth** (step0/step1-2), yet
simultaneously has by far the **best safety-clamp metrics of all 6** (table above). Reconciliation:
clamp rate measures delta-from-*current-state* against a safety threshold, not accuracy vs a
ground-truth trajectory — V4's broader training diversity appears to make the model predict
smaller, more conservative first-action deltas at this T01 reference pose (safer, less clamping)
even though those deltas track the *specific* nearest-demo's own (possibly larger) trajectory jump
less closely (worse GT-MAE). Both readings are computed correctly; they answer different questions
(safety-gate pass rate vs trajectory-fidelity).

## 10가지 핵심 판정

**1. historical test10 MAE가 reweight2:1(3.53)보다 좋은가?**
아니오. F 최선(7500=3.779, 10000=3.780, 사실상 동률)도 3.53보다 나쁨 (+7%).

**2. V4 heldout6에서 일반화가 좋은가?**
예, 이 실험의 강점. 10000에서 3.524, 단조 개선, D(V4 standalone)의 3.563보다도 좋음 — V3를
더해도 V4 자체 일반화가 나빠지지 않고 오히려 소폭 개선.

**3. clamp-free가 40%를 넘는가?**
예. 10000=65%, 5000부터 60%대 유지 — 이번 프로젝트 전체 실험(A-E) 중 최고치.

**4. elbow clamp가 60% 아래인가?**
예. 10000=30% (60% 기준 절반) — A-E 어느 실험도 60% 밑으로 못 내려온 지표를 처음 돌파.

**5. shoulder_lift clamp가 25% 아래인가?**
예. 5000부터 10%로 유지 — B(25%)의 절반 이하.

**6. L2가 3.31보다 낮은가?**
아니오. 10000=4.38로 오히려 악화. F의 첫 액션은 안전하지만(clamp 기준) GT 궤적과의 절대 오차는
더 크다 — 위 "6-way 비교" 절의 reconciliation 참고.

**7. step3+가 7.435보다 좋아지는가?**
예, 소폭. 10000: 7.231 < 7.435 (−2.6%). C/D보다는 나쁘지만 B/E보다는 낫다.

**8. offline-best와 safety-best checkpoint가 같은가?**
대체로 같다. offline-best(historical test10)는 7500/10000 사실상 동률(3.779/3.780). safety상
clamp-free는 5000/10000 동률(65%)이지만 L2는 10000이 더 낮다(4.38<5.41) — 종합하면 **10000이
합리적 단일 후보**이나, elbow_flex clamp만 보면 5000(20%)이 10000(30%)보다 근소하게 낫다는
예외가 있다.

**9. V4가 실제로 V2를 대체할 가치가 있는가?**
**조건부로 그렇다.** uniform sampling 단독으로는 현재 최선(B, reweight2:1)의 offline MAE/gap/L2를
이기지 못한다 — 이 자체는 대체 근거가 안 된다. 그러나 F는 이 프로젝트 전체에서 **처음으로**
clamp-free 40% 초과·elbow clamp 60% 미만·shoulder_lift clamp 25% 미만을 **동시에** 달성했고,
V4 standalone(D)보다 두 heldout 지표 모두 우수하다 — V4의 다양성이 safety-gate 관련 지표에
실질적 가치가 있다는 강한 신호다. 즉 "V4가 V2를 완전히 대체"보다는 "V4를 조합에 넣는
것 자체가 이번 프로젝트가 계속 못 넘던 safety 임계값을 넘게 해준다"는 결론이 더 정확하다.

**10. 다음에 V3:V4 reweight를 실험할 가치가 있는가?**
**예, 이번 결과가 그 근거다.** B(reweight2:1)가 이미 증명한 것: sampling reweighting은 uniform
대비 gap/offline MAE를 크게 개선한다. F(V3+V4 uniform)가 이번에 증명한 것: V4 조합은 uniform
상태에서도 safety-clamp 지표를 크게 개선한다. 두 레버가 서로 다른 축(정확도 vs 안전성)을
개선했으므로, **V3:V4를 reweight(예: 2:1 또는 그 역방향)하면 두 이득을 동시에 취할 가능성이
있다** — 특히 F의 약점(historical MAE, gap, step0/1-2 chunk accuracy)이 B의 강점과 정확히
겹치는 지점이라 결합 여지가 크다.

---

## 산출물

`checkpoint_metrics.csv` · `first_action_diagnostics.csv` · `temporal_chunk_error.csv` ·
`comparison_vs_reweight2.csv` · `summary.json` · `first_action_seed_sweep_<step>/` ·
`temporal_chunk_error/v3_v4_uniform_<step>/` — 이 디렉터리.
데이터셋 빌드 증적: `reports/pick_drop_v3_v4_combined69_dataset_build/summary.json`.
Historical/V4-heldout6 offline eval 원본: `reports/pick_drop_v3_v4_combined69_uniform_v1_historical_offline_eval/`,
`reports/pick_drop_v3_v4_combined69_uniform_v1_v4heldout6_offline_eval/`.

**Cleanup**: 디스크 여유 충분(818G)해서 아무것도 삭제하지 않음. 보호 대상 V2/V3/V4/heldout
dataset, reweight2:1 checkpoint/report, V4 fresh checkpoint/report 전부 원본 그대로 확인.
실물 follower write 없음. Git commit/push 없음.
