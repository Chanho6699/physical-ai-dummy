# Grid35 V2 clean SmolVLA — checkpoint 005000 vs 007500 vs 010000 first-action Shadow stability comparison

Pure comparison run on top of the existing pipeline (`scripts/sweep_grid35_multi_scene.py` →
`scripts/analyze_seed_mitigation_strategies.py` → `scripts/aggregate_t01_t10_seed_sweep_report.py`),
against the **actual (real SO-101 hardware) T01-T10 Shadow observations**
(`reports/grid35_v2_shadow_T0N_real_final/shadow_patched.json`, one self-consistent capture
session) — the same source the existing 007500 `real_final` result already used. **No synthetic /
past same-scene results are mixed in.**

- Dataset: `data/so101_cube_xy_grid35_v2_clean`
- Scenes: T01–T10 (all real hardware, `evaluation_mode == "midpoint-shadow"`)
- Seeds: 0–19 (identical set across all 3 checkpoints, 20 live `predict_action_chunk()` calls per
  scene per checkpoint = 200 seed×scene draws per checkpoint, 600 total)
- Safety Gate thresholds: read from `configs/safety_gate.yaml`, **not modified**
  (shoulder_pan 4.58°, shoulder_lift 5.16°, elbow_flex 5.73°, wrist_flex 4.01°, wrist_roll 1.15°,
  gripper 9.17°)
- **No training, no robot writes, no LeRobot source changes, no git operations.**

New artifacts produced by this comparison (same schema as the existing 007500 `real_final`
artifacts, only the checkpoint differs):

| checkpoint | per-scene sweep | per-scene mitigation | T01–T10 aggregate |
|---|---|---|---|
| 005000 | `reports/grid35_v2_T0N_ckpt005000_real_final_seed_sweep/` | `reports/grid35_v2_T0N_ckpt005000_real_final_seed_mitigation/` | `reports/grid35_v2_T01_T10_ckpt005000_real_final_seed_sweep_summary/` |
| 007500 *(pre-existing, reused as-is)* | `reports/grid35_v2_T0N_real_final_seed_sweep/` | `reports/grid35_v2_T0N_real_final_seed_mitigation/` | `reports/grid35_v2_T01_T10_real_final_seed_sweep_summary/` |
| 010000 | `reports/grid35_v2_T0N_ckpt010000_real_final_seed_sweep/` | `reports/grid35_v2_T0N_ckpt010000_real_final_seed_mitigation/` | `reports/grid35_v2_T01_T10_ckpt010000_real_final_seed_sweep_summary/` |

## Final comparison table (T01–T10 average, seeds 0–19, pooled over 200 scene×seed draws per checkpoint)

| checkpoint | single-sample clamp-free | resample≤5 success rate | shoulder_lift clamp rate | elbow_flex clamp rate | GT L2 mean / median / p95 (deg) |
|---|---:|---:|---:|---:|---:|
| **005000** | **5.0%** | **25.3%** | 52.0% | 95.0% | 9.12 / 8.59 / 15.11 |
| **007500** | 2.5% | 12.6% | 84.0% | 85.5% | 6.05 / 5.18 / 11.30 |
| **010000** | 1.0% | 5.1% | 54.0% | 97.0% | 5.88 / 5.24 / 10.61 |

(All 200 draws per checkpoint = 10 scenes × 20 seeds; "clamp-free" = zero of the 6 joints exceed
their WOULD_CLAMP threshold on the raw single-sample chunk[0] action.)

## Supporting detail

### Scene-level single clamp-free % (10/10 scenes, seeds 0–19)

| checkpoint | T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08 | T09 | T10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 005000 | 5% | 5% | 5% | 5% | 5% | 5% | 5% | 5% | 5% | 5% |
| 007500 | 5% | 5% | 0% | 5% | 0% | 0% | 0% | 5% | 0% | 5% |
| 010000 | 0% | 5% | 0% | 0% | 5% | 0% | 0% | 0% | 0% | 0% |

005000 is the only checkpoint where every one of the 10 scenes lands at exactly the same 5%
(1/20) clamp-free rate — a flat, low floor, not scene-dependent noise. 007500 and 010000 both show
scene-to-scene swings between 0% and 5%, i.e. several scenes with a **zero**-clamp-free single-seed
pool.

### Per-joint clamp-rate breakdown (all 6 joints, T01–T10 average)

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---:|---:|---:|---:|---:|---:|
| 005000 | 13.0% | 52.0% | 95.0% | 5.0% | 0.0% | 0.0% |
| 007500 | 0.0% | 84.0% | 85.5% | 0.0% | 0.0% | 0.0% |
| 010000 | 0.0% | 54.0% | 97.0% | 0.0% | 0.0% | 0.0% |

`elbow_flex` clamps on **85–97% of every single-sample draw at every checkpoint tested** — this is
not a checkpoint-specific defect, it is a standing property of this training run's flow-matching
output at this reference-state family. `shoulder_lift` varies more by checkpoint (52% → 84% → 54%,
non-monotonic with training step). Only 005000 leaks clamp risk into `shoulder_pan` (13%) and
`wrist_flex` (5%) as well — the other two checkpoints keep those two joints and `wrist_roll`/
`gripper` at a clean 0%.

### Seed-to-seed action variance (std of chunk[0] delta across the 20 seeds, averaged per-scene)

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---:|---:|---:|---:|---:|---:|
| 005000 | 1.41° | 4.39° | 3.26° | 1.35° | 0.23° | 0.80° |
| 007500 | 0.76° | 2.81° | 2.18° | 0.89° | 0.14° | 0.51° |
| 010000 | 0.64° | 2.49° | 1.78° | 0.76° | 0.12° | 0.44° |

Seed-to-seed variance **decreases monotonically with training step** on every joint (005000 →
007500 → 010000). More training tightens the flow-matching sampling distribution — but a tighter
distribution centered further from the safe zone (see `shoulder_lift`/`elbow_flex` clamp rates
above) does not by itself buy a higher clamp-free rate; 007500's `shoulder_lift` clamp rate
(84.0%) is actually the *worst* of the three despite lower variance than 005000, because its
distribution's mean sits closer to (or past) the threshold.

### GT immediate-action L2 (pooled, single-sample, 200 draws)

| checkpoint | mean | median | p95 |
|---|---:|---:|---:|
| 005000 | 9.12° | 8.59° | 15.11° |
| 007500 | 6.05° | 5.18° | 11.30° |
| 010000 | 5.88° | 5.24° | 10.61° |

007500 and 010000 are close on GT-L2 (within ~3% of each other, both clearly better than 005000);
005000 is materially worse (~50% higher mean/median L2 than the other two) — consistent with it
being the least-trained checkpoint of the three.

### Safety-pass resampling≤5 (offline-simulated from the same 20 observed seeds, 20000 draw-order permutations, no GT in the stopping rule)

| checkpoint | resample≤5 success rate | resample≤5 GT L2 mean / median / p95 | avg inference calls used |
|---|---:|---:|---:|
| 005000 | 25.3% | 4.20° / 3.78° / 6.56° | 4.50 |
| 007500 | 12.6% | 2.80° / 2.63° / 4.45° | 4.75 |
| 010000 | 5.1% | 2.61° / 2.45° / 4.15° | 4.90 |

Resampling roughly doubles clamp-free odds over single-sample at every checkpoint (5.0%→25.3%,
2.5%→12.6%, 1.0%→5.1%), and improves delivered GT-L2 substantially in all three cases — but the
*residual failure rate at cap=5* is 74.7% / 87.4% / 94.9% respectively. None of the three
checkpoints gets anywhere near a resampling-rescued deploy-ready state.

## Judgment

### Q1 — 5k/10k가 7.5k보다 실제 first-action stability가 유의미하게 좋은가?

**아니요, 유의미하게 좋지 않습니다.** 오히려 방향이 checkpoint마다 반대로 갈립니다:

- **005000**은 clamp-free(5.0%)와 resample≤5(25.3%)가 세 checkpoint 중 수치상 **가장 높습니다** —
  하지만 이는 "더 안정적"이라기보다 GT-L2가 가장 나쁜(9.12°, 다른 두 checkpoint 대비 ~50% 더 큼)
  아직 덜 수렴한 상태에서 나온 결과이고, `shoulder_pan`(13%)·`wrist_flex`(5%)까지 clamp가 새는 유일한
  checkpoint입니다. seed 분산도 세 checkpoint 중 가장 큽니다(shoulder_lift std 4.39° vs 2.81°/2.49°).
  즉 "안정적"이 아니라 "아직 학습이 덜 되어 threshold 근처에서 덜 뭉쳐 있는" 상태에 가깝습니다.
- **010000**은 세 checkpoint 중 clamp-free(1.0%)·resample≤5(5.1%)가 **가장 낮습니다** — GT-L2는
  가장 좋지만(5.88°), `elbow_flex` clamp rate가 97%로 최고치이고 10/10 scene 중 8개가 단일 시드
  clamp-free 0%입니다. 학습이 더 진행될수록 분포가 좁아지면서(seed variance는 감소) 그 좁아진
  분포의 중심 자체가 안전 영역 밖으로 더 쏠린 결과로 보입니다.

세 checkpoint 모두 clamp-free rate가 5% 이하이고 resample≤5로도 25.3%를 넘지 못하므로, "5k/10k가
7.5k보다 실전 배포 가능한 수준으로 안정적"이라고 말할 근거는 없습니다.

### Q2 — offline MAE 최저였던 7.5k와 Shadow stability 사이 trade-off가 있는가?

**있습니다, 다만 007500과 010000 사이는 미미하고, 005000 쪽이 더 뚜렷합니다.**

- 007500 vs 010000: GT-L2(6.05° vs 5.88°, ~3% 차이)는 거의 같은데, clamp-free(2.5% vs 1.0%)와
  resample≤5(12.6% vs 5.1%)는 010000이 뚜렷이 나쁩니다 — **정확도상 이득이 거의 없이 안정성만
  더 잃는** 구간입니다. `shoulder_lift`만 84%→54%로 개선되지만 `elbow_flex`가 85.5%→97%로 더
  악화되어 상쇄됩니다.
- 005000 vs 007500: 005000은 clamp-free/resample≤5가 더 높지만 GT-L2가 ~50% 나쁩니다 — 이쪽은
  "정확도-안정성" 교환이라기보다 단순 미수렴(underfit) 상태의 부작용에 가깝습니다.

결론적으로 007500이 offline MAE 최저 지점이라는 사실과 이번 Shadow 결과 사이에 **뚜렷한 상충은
아니지만, 010000으로 더 학습을 진행해도 Shadow 안정성 개선은 없고 오히려 후퇴**한다는 점은
분명한 trade-off 신호입니다 — "더 학습 = 더 안정"이라는 가정이 이 체크포인트 구간에서는 성립하지
않습니다.

### Q3 — 어떤 checkpoint도 충분히 안정적이지 않다

**맞습니다 — 세 checkpoint 모두 배포 기준을 충족하지 못합니다.**

- 단일 샘플 clamp-free rate: 1.0% ~ 5.0% (세 checkpoint 모두 10% 미만)
- resample≤5 성공률: 5.1% ~ 25.3% (세 checkpoint 모두 residual failure ≥ 74.7%)
- `elbow_flex`는 **세 checkpoint 전부에서 85% 이상 clamp** — 이번 비교로 checkpoint 선택만으로는
  해결되지 않는, training-step에 무관하게 고정된 실패 모드임이 재확인되었습니다.
- 10개 scene 중 다수(007500 5/10, 010000 8/10)가 단일 시드 clamp-free 0%인 "safe seed 없음"
  상태이며, 이는 seed sweep 범위(0–19) 안에서 clamp를 피하는 시드가 아예 존재하지 않을 수 있다는
  뜻입니다.

**판단: 이 세 checkpoint 중 어느 것도 Shadow first-action 안정성 기준을 만족하지 못하며, 이는
checkpoint 선택(005000/007500/010000 중 택1)으로 해결될 문제가 아닙니다.** `elbow_flex`가 전
checkpoint·전 scene에서 일관되게 threshold를 초과하는 것은 sampling-seed 문제도, 특정 training
step의 문제도 아니라 **학습 데이터 자체(grid35_v2_clean)가 이 관절 동작 구간을 충분히 커버하지
못하고 있음을 가리키는 신호**입니다. 다음 단계로 (a) `elbow_flex`/`shoulder_lift` 근방 training
demo 커버리지 검토 및 보강, (b) 필요 시 재학습(이번 비교 범위 밖), (c) 그 전까지는 real Shadow
경로에서 resample≤5만으로 운용하지 말고 잔여 실패율(74.7~94.9%)을 명시적으로 모니터링되는
Safety-Gate-clip 비율로 취급할 것을 권고합니다. (재학습·threshold 변경·robot write는 이번 비교의
범위 밖이며 실행하지 않았습니다.)
