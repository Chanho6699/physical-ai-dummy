# V4 Pick & Drop generalization39 — SmolVLA fresh training & 평가

실물 GPU 실행 (RTX 3050 8GB, `~/lerobot` venv). `lerobot/smolvla_base`에서 fresh-start (resume 없음,
optimizer state 재사용 없음). 학습 데이터는 **`data/so101_cube_pick_drop_generalization_v4`
단독** (39 episodes, 12,795 frames, task `"Pick up the cube and drop it into the bin."`).
reweight/early-action-loss/sampler 변경 **전부 사용하지 않음** — V4 dataset 자체의 효과만 측정.
실물 로봇 write 없음(학습은 offline, 평가/진단은 기록된 데이터에 대한 순수 inference).
Safety threshold 변경 없음. dataset/checkpoint 수정 없음. git 작업 없음.

## 0. 재사용한 것 / 새로 만든 것

- **training 설정**: `outputs/pick_drop_combined65/smolvla_pick_drop_combined65_fresh/checkpoints/010000/pretrained_model/train_config.json`
  (실제 기록된 config)과 `~/.bash_history`의 V2 fresh-training 호출을 대조해 그대로 재사용 —
  `--dataset.repo_id`/`--dataset.root`/`--output_dir`/`--job_name`만 변경.
- **평가/진단 스크립트**: [`scripts/evaluate_smolvla_midpoint.py`](../../scripts/evaluate_smolvla_midpoint.py),
  [`scripts/sweep_grid35_first_action_seed.py`](../../scripts/sweep_grid35_first_action_seed.py),
  [`scripts/diagnose_temporal_chunk_error.py`](../../scripts/diagnose_temporal_chunk_error.py) 전부
  수정 없이 재사용, `--task`만 Pick&Drop 문자열로 비파괴 override.
- **비교 baseline 수치**: V2 baseline(`reports/grid35_v2_midpoint_eval`, `reports/grid35_v2_T01_seed_sweep`),
  Combined65 uniform(`reports/pick_drop_combined65_offline_eval`, `reports/pick_drop_combined65_fresh_training`),
  reinforcement30-only(`reports/reinforcement30_only_v1*`), reweight2:1/3:1
  (`reports/combined65_reweight_new2_old1_v1`, `reports/combined65_reweight_new3_old1_v1`) —
  모두 기존 리포트에서 그대로 읽음, 재계산하지 않음.
- **새로 만든 것**: `scripts/build_pick_drop_v4_fresh_training_report.py` (본 리포트 조립),
  `scripts/compute_v4_heldout6_per_episode_mae.py` (heldout6 episode별 MAE — 기존
  `evaluate_smolvla_midpoint.py` 출력의 `records`에서 재계산 없이 그룹화만 수행).

## 1. Training pipeline

```
lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.workspace":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --dataset.repo_id=local/so101_cube_pick_drop_generalization_v4 \
  --dataset.root=.../data/so101_cube_pick_drop_generalization_v4 \
  --output_dir=outputs/pick_drop_v4/smolvla_pick_drop_v4_fresh \
  --job_name=smolvla_pick_drop_v4_fresh --seed=1000 \
  --batch_size=4 --steps=10000 \
  --save_checkpoint=true --save_freq=2500 --log_freq=100 --wandb.enable=false
```

| hyperparameter | value | 비고 |
|---|---|---|
| base checkpoint | `lerobot/smolvla_base` | V2/Combined65와 동일 |
| batch size | 4 | 동일 |
| optimizer | AdamW, lr=1e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=1e-10, grad_clip_norm=10.0 | 동일 |
| scheduler | cosine_decay_with_warmup, warmup 1000→333(auto-scaled), decay 30000→10000, peak_lr=1e-4, decay_lr=2.5e-6 | 동일 |
| chunk_size | 50 | 동일 |
| seed | 1000 | 동일 (명시적으로 `--seed=1000` 전달) |
| empty_cameras | 1 | 동일 |
| rename_map | workspace→camera1, wrist→camera2 | 동일 |
| reweight / sampler / early-action loss | **사용 안 함** | 이번 실험의 핵심 제약 |
| save frequency | 2500 | 동일 |

## 2. Preflight

| check | 결과 |
|---|---|
| episode 수 | **39** (`dataset.num_episodes=39` 학습 로그 자체 확인) |
| frame 수 | **12,795** |
| task string | `"Pick up the cube and drop it into the bin."` ✓ |
| state/action | 6D, `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` 순서 ✓ |
| camera1(workspace)/camera2(wrist) | `[3,480,640]` float32, 정상 로드 ✓ |
| NaN/Inf | 없음 (샘플 5 프레임 검사) |
| 5-step smoke run | losses `0.576→1.407→0.749→0.383→0.438` (모두 finite), GPU mem 2.44GB, 에러 없음 |

**PASS → full training 실행.**

## 3. Training 결과

| metric | value |
|---|---|
| started | 2026-08-09 23:02:31 |
| ended (`End of training`) | 2026-08-10 00:07:40 |
| 총 wall time | **65.15분** (V2 ~65.5분, Combined65 66.7분, reinforcement30-only 65.7분과 거의 동일 — step 수에 비례, dataset 크기와 무관) |
| checkpoints | 002500 / 005000 / 007500 / 010000 (4개 모두 정상 저장) |
| 에러/OOM/Traceback | **없음** |
| GPU 메모리 | 2.44GB로 안정 (8GB 중) |

| checkpoint | wall time(분) | train loss | lr | grad norm |
|---:|---:|---:|---:|---:|
| 2500 | 15.72 | 0.115 | 8.6e-05 | 3.437 |
| 5000 | 32.17 | 0.070 | 5.2e-05 | 2.345 |
| 7500 | 48.62 | 0.044 | 1.7e-05 | 1.637 |
| 10000 | 65.08 | 0.041 | 2.5e-06 | 1.454 |

Train loss가 0.115→0.041까지 **단조 감소**, V2/Combined65에서 이따금 보였던 후반 재상승 없음.

## 4. Historical test10 offline 평가 (`data/so101_cube_xy_midpoint_test10_v2_clean`, 10 episodes/3288 frames, seed=42, task는 평가시에만 Pick&Drop 문자열로 override) — [`historical_test10_metrics.csv`](historical_test10_metrics.csv)

| checkpoint | V4 action MAE | shoulder_lift MAE | elbow_flex MAE | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT | V2 MAE | Combined65 uniform MAE | reinforcement30-only MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 5.0486 | 8.107 | 6.810 | 153 | 3059 | 76 | 5.3875 | 4.7157 | 5.4743 |
| 5000 | 4.5662 | 7.963 | 6.769 | 550 | 2669 | 69 | 4.2909 | 4.3404 | 4.1449 |
| 7500 | 4.1282 | 6.827 | 6.468 | 734 | 2504 | 50 | 3.9841 | 3.4123 | 3.8812 |
| **10000** | **4.0643** | **6.711** | **6.502** | **803** | **2451** | **34** | 4.2210 | 3.6396 | 3.9760 |

V4는 4개 checkpoint 전부에서 **오차가 단조 감소**한다 — V2/Combined65/reinforcement30-only가
모두 보였던 "7500이 최적, 10000에서 살짝 재상승" 패턴이 V4에서는 나타나지 않고 10000이 그대로
최적이다. 다만 절대 MAE 수치 자체는 Combined65 uniform(3.41~3.64)보다 전 checkpoint에서 나쁘고,
V2(35ep)와 비슷하거나 근소하게 나은 수준이다 — **데이터 다양성이 늘었지만 episode 수(39)가
Combined65(65)보다 적어 offline MAE 자체는 Combined65에 못 미친다.**

## 5. V4 heldout6 평가 (`data/so101_cube_pick_drop_v4_heldout10`, 실제 **6 episodes**/1968 frames — 디렉터리명은 heldout10이지만 본 보고서 전체에서 **heldout6**으로 표기) — [`v4_heldout6_metrics.csv`](v4_heldout6_metrics.csv)

historical test10과 **절대 하나의 평균으로 합치지 않았음** — 별도 eval 실행, 별도 CSV/summary 블록.

### checkpoint 단위

| checkpoint | action MAE | shoulder_lift MAE | elbow_flex MAE | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |
|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 5.0403 | 7.418 | 7.895 | 79 | 1584 | 305 |
| 5000 | 4.1940 | 6.914 | 6.377 | 270 | 1557 | 141 |
| 7500 | 3.6034 | 5.879 | 5.512 | 347 | 1486 | 135 |
| **10000** | **3.5634** | **5.695** | **5.520** | 342 | 1503 | 123 |

test10과 동일하게 **단조 개선**, 10000이 최적. WOULD_REJECT(gross 위반)도 305→123으로 계속 감소.

### episode 단위 (checkpoint 10000, 6개 episode 모두 표시)

| episode | action MAE | n_frames |
|---:|---:|---:|
| 0 | 3.884 | 328 |
| 1 | 3.869 | 328 |
| 2 | 3.274 | 328 |
| 3 | 2.835 | 328 |
| 4 | 4.122 | 328 |
| 5 | 3.396 | 328 |

episode 3이 가장 쉬움(2.84~4.01, checkpoint 전체에서 최저), episode 4가 가장 어려움
(4.12~6.11) — 4개 checkpoint 모두 동일한 순서를 보여 특정 시작 위치(episode 4)가 구조적으로
어렵다는 신호(정확한 cube x/y는 metadata에 없으므로 추정하지 않음).

## 6. First-action diagnostic (동일 T01 Shadow reference observation, seeds 0-19, Safety threshold 불변) — [`first_action_diagnostics.csv`](first_action_diagnostics.csv)

| 실험/checkpoint | shoulder_lift bias/clamp | elbow_flex bias/clamp | wrist_roll bias/clamp | clamp-free | L2 vs GT |
|---|---:|---:|---:|---:|---:|
| V2 7500 (baseline) | +5.13/45% | −7.08/75% | +0.42/0% | 4/20 (20%) | 4.62 |
| Combined65 uniform 7500 | +4.51/35% | −7.66/85% | +0.23/0% | 3/20 (15%) | 4.78 |
| Combined65 uniform 10000 | +3.83/25% | −6.88/75% | +0.21/0% | 5/20 (25%) | 4.15 |
| reinforcement30-only 7500 | +1.61/10% | −5.88/60% | +0.18/0% | 8/20 (40%) | 4.58 |
| **reweight 2:1 10000** | +3.96/25% | −6.11/60% | +0.32/0% | **8/20 (40%)** | **3.31** |
| reweight 3:1 10000 | +3.33/25% | −6.27/70% | +0.27/0% | 6/20 (30%) | 3.67 |
| V4 fresh 2500* | +4.17±6.55/40% | −10.58±4.96/80% | +1.19±1.16/55% | 1/20 (5%) | 9.93 |
| V4 fresh 5000 | +6.07±4.08/50% | −6.46±2.91/65% | +0.18±0.55/5% | 4/20 (20%) | 5.61 |
| **V4 fresh 7500** | +2.38±2.99/**20%** | −6.35±2.37/**65%** | +0.39±0.37/5% | 6/20 (30%) | **4.44** |
| **V4 fresh 10000** | +2.13±2.88/**10%** | −6.88±2.22/70% | +0.19±0.35/0% | 6/20 (30%) | 4.70 |

(*2500은 모든 실험에서 undertrained noise로 취급 — std 4.9~6.6deg, L2>9deg, 비교에서 제외.)

- **shoulder_lift**: V4가 5000/7500/10000 어디서도 V2(45%)/Combined65(25~35%)보다 clamp rate가
  낮거나(7500=20%, 10000=**10%, 전체 실험 중 최저**) 비슷하다 — **분명한 개선**.
- **elbow_flex**: V4 best(7500/5000=65%)는 V2(75%)/Combined65(75~85%)보다는 낮지만
  reinforcement30-only(60%)/reweight2:1(60%)보다는 못하다. 10000에서는 70%로 다시 V2 수준에
  가까워진다 — **개선은 있지만 부분적이고, checkpoint에 따라 되돌아간다**.
- **wrist_roll**: V4 5000/7500/10000에서 clamp rate 0~5% — V2/Combined65/reweight 계열과
  동일하게 0% 수준. 2500(55%)만 예외인데, 이는 std 1.16deg로 이 checkpoint 자체가 모든 관절에서
  noise인 것과 일치 — **wrist_roll variation 확대가 학습이 진행된 checkpoint에서 새로운
  instability를 만들지는 않았다.**
- **clamp-free**: V4 best(7500/10000 공동 30%)는 reweight2:1/reinforcement30-only의 40%에
  **못 미친다.**
- **L2 vs GT**: V4 best(7500=4.44)는 reweight2:1(3.31)/reweight3:1(3.67)보다 **명확히 나쁘다.**

## 7. Temporal(chunk-position) 오차 진단 — [`temporal_chunk_error.csv`](temporal_chunk_error.csv)

key-joint(shoulder_lift+elbow_flex 평균) MAE, chunk_size=50:

| checkpoint | step0 | step1-2 | step3+ |
|---:|---:|---:|---:|
| 2500 | 4.957 | 3.875 | 4.838 |
| 5000 | 3.146 | 2.584 | 3.587 |
| **7500** | **2.136** | **1.932** | **6.077** |
| 10000 | 2.101 | 1.931 | **6.265** |

**핵심 발견 — V4의 데이터셋 분석에서 확인된 ~85% chunk future-motion 문제가 실제 policy에서
그대로, 오히려 checkpoint가 진행될수록 더 뚜렷하게 병목으로 나타난다:**

- step0(=`select_action()`이 실제로 내보내는 action)은 7500/10000에서 **전체 실험 중 가장
  낮은 값(2.10~2.14deg)** — immediate action 자체는 매우 정확하게 학습됐다.
- 그런데 step3+(chunk의 나머지 47 스텝)는 7500/10000에서 **6.08~6.27deg로, 오히려 5000(3.59)
  이나 2500(4.84)보다 나쁘다.** 즉 학습이 진행될수록 "가까운 미래(step0~2)는 정확해지지만 먼
  미래(step3+)는 더 부정확해지는" 뚜렷한 trade-off가 나타난다.
- 이는 데이터셋 분석(`reports/pick_drop_v2_v3_v4_dataset_analysis/summary.md`)에서 측정한
  "static observation의 85%가 큰 평균 future chunk와 짝지어짐" (V2 99.7%, V3 61%보다 나쁨)
  구조적 문제가 **실제로 학습된 policy의 chunk 뒷부분 오차로 직결됨**을 보여주는 직접적 증거다.

## 8. 판정 질문

**1. V4 fresh training은 정상 완료됐는가?**
예. resume 없이 `smolvla_base`에서 fresh-start, preflight 통과, 10,000 step 전부 완주, 4개
checkpoint 모두 정상 저장, 에러/OOM 없음, loss 0.115→0.041 단조 감소, wall time 65.15분(기존
실험들과 동일 페이스).

**2. historical test10 기준 best checkpoint는?**
**10000** (action MAE=4.0643, 4개 중 최저). V2/Combined65/reinforcement30-only가 보인
"7500 peak, 10000 소폭 재상승" 패턴과 달리 V4는 10000까지 단조 개선.

**3. V4 heldout6 기준 best checkpoint는?**
**10000** (action MAE=3.5634, 4개 중 최저, WOULD_REJECT도 최저인 123).

**4. 두 test set에서 best checkpoint가 일치하는가?**
**예, 10000으로 일치한다.** 다만 first-action 안전성 지표(6번 문항)에서는 7500이 10000보다
낫거나 동등한 항목이 많아 "offline MAE 최적"과 "first-action 안전 최적"이 이번에도 완전히는
일치하지 않는다 — 7500/10000 둘 다 후속 단계(Shadow 등)에서 함께 볼 가치가 있다.

**5. V4의 start diversity 증가가 first-action safety 개선으로 이어졌는가?**
**부분적으로 그렇다.** shoulder_lift는 명확히 개선됐다(V2 45%→V4 10000의 10%, 전체 실험 중
최저). 그러나 elbow_flex는 checkpoint에 따라 다르다(7500=65%로 V2의 75%보다 낫지만,
10000=70%로 다시 악화) — "개선됐다"고 일괄적으로 말할 수 없다.

**6. elbow_flex clamp는 V2/Combined65보다 줄었는가?**
**V4 최선(65%, 5000/7500)은 V2(75%)·Combined65(75~85%)보다 줄었지만, reinforcement30-only/
reweight2:1의 60%에는 못 미치고, 10000 checkpoint에서는 70%로 다시 V2에 근접한다.** 즉 방향은
맞지만 폭이 작고 불안정하다.

**7. wrist_roll variation 증가가 새로운 instability를 만들었는가?**
**아니다.** 학습이 진행된 checkpoint(5000/7500/10000)에서 wrist_roll clamp rate는 0~5%로 V2/
Combined65/reweight 계열의 0% 수준과 동일하다. 2500의 55%는 이 checkpoint 자체가 모든 지표에서
undertrained noise(std 1.16deg, L2>9deg)인 것과 일치하며 wrist_roll 고유의 문제가 아니다.

**8. clamp-free는 reweight2:1의 40%를 넘는가?**
**아니다.** V4 최선(7500/10000 공동 30%, 6/20)은 reweight2:1과 reinforcement30-only의
40%(8/20)에 못 미친다.

**9. GT first-action L2는 3.31보다 좋아지는가?**
**아니다.** V4 최선(7500=4.44)은 reweight2:1(3.31)보다 명확히 나쁘고, reweight3:1(3.67)보다도
나쁘다.

**10. V4의 85% chunk future-motion 문제가 policy 성능에서 실제 병목으로 나타나는가?**
**예, 명확히 나타난다.** 7500/10000에서 step0 오차는 전체 실험 중 최저(2.10~2.14deg)이지만
step3+ 오차는 6.08~6.27deg로 오히려 5000(3.59deg)보다 크게 나쁘다 — "가까운 미래는 정확해지고
먼 미래는 더 부정확해지는" trade-off가 실측됐고, 이는 데이터셋 분석에서 측정한 85% future-chunk
문제(V2 99.7%보다 낫지만 V3 61%보다 나쁨)와 정확히 같은 방향이다.

**11. V4 단독이 V2+V3 2:1보다 낫나?**
**아니다.** first-action 안전성의 핵심 지표(L2 vs GT, clamp-free rate, elbow_flex clamp) 전부
에서 reweight2:1이 V4 단독보다 낫다 (L2 3.31 vs 4.44, clamp-free 40% vs 30%, elbow_flex clamp
60% vs 65~70%). V4가 유일하게 이긴 지표는 shoulder_lift clamp rate(10% vs 25%)뿐이다.

**12. 다음 단계 추천: A) V4+V3 결합 / B) V4+V3 reweight / C) V2+V3 2:1 + early-action loss**

**C를 추천한다.**

이유:
- V4 단독은 이미 검증된 reweight2:1보다 대부분의 first-action 지표에서 못하다(11번 답변). 즉
  "새 데이터만으로 기존 최선을 대체"하는 전략은 이번 실험으로 부정됐다.
- V4의 chunk future-motion 문제(~85%)는 V3(~61%)보다 나쁘다 — A)처럼 V4를 그대로 V3와
  섞으면, 이미 reweight2:1이 어렵게 눌러놓은 chunk-motion 문제가 V4의 비중만큼 다시 악화될
  위험이 있다. B)는 이 위험을 완화하지만, "V4를 얼마나 가중치를 줄여 섞을지"를 새로 검증해야
  하는 미검증 레버다.
- 반면 C(early-action loss)는 `combined65_early_weight_v1`에서 **reweight와 독립적으로 이미
  방향성 있는 효과가 확인된 레버**(`reports/reinforcement30_only_v1/summary.md`의 "두 레버가
  서로 다른 곳에서 이점을 보였다" 결론)이고, reweight2:1(현재 L2=3.31, clamp-free=40%인 이미
  최고 config) 위에 쌓는 것이므로 새 데이터 수집/새 sampler 구현 없이 바로 검증 가능하다.
- V4 데이터 자체(diversity, elbow_flex/wrist_roll coverage)는 여전히 가치가 있으므로 폐기하자는
  뜻은 아니다 — C를 먼저 검증한 뒤, 그 결과가 한계에 도달하면 B(V4를 chunk-motion 낮은
  episode만 선별 reweight)를 다음 후보로 남겨둔다.

## 산출물

- `summary.md` (본 문서), `summary.json`
- `historical_test10_metrics.csv` — checkpoint별 offline MAE/joint MAE/WOULD_* + V2/Combined65/reinforcement30-only 비교열
- `v4_heldout6_metrics.csv` — checkpoint 단위 + episode 단위(0~5) 행을 `row_type` 컬럼으로 구분, historical test10과 절대 평균하지 않음
- `first_action_diagnostics.csv` — V2/Combined65/reinforcement30-only/reweight2:1/reweight3:1/V4(4 checkpoint) 전체
- `temporal_chunk_error.csv` — V4 4개 checkpoint의 step0/step1-2/step3+ 버킷
- 원시 데이터: `train.log`, `first_action_seed_sweep_<step>/`, `temporal_chunk_error/v4_fresh_<step>/`,
  `v4_heldout6_per_episode_mae.json`

## 하지 않은 것 (지시대로)

- reweight / early-action loss / sampler 변경 없음 — V4 dataset 자체 효과만 측정.
- 실물 SO-101/MuJoCo에 어떤 write도 없음 (학습은 offline, 평가/진단은 순수 inference).
- 기존 checkpoint에서 resume 없음 — 매번 `lerobot/smolvla_base`에서 fresh start.
- historical test10과 V4 heldout6 결과를 하나의 평균으로 합치지 않음.
- git commit/push 없음.
