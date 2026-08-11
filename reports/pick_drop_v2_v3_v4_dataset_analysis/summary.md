# Pick & Drop V2 / V3 / V4(train39) / V4(heldout6) 데이터셋 분석

읽기 전용(read-only) 분석. 원본 dataset 수정 없음, git commit/push 없음, merge/reweight/training
없음. 생성 스크립트: [`scripts/analyze_pick_drop_v2_v3_v4_dataset_analysis.py`](../../scripts/analyze_pick_drop_v2_v3_v4_dataset_analysis.py)
(정의 재사용원: [`scripts/analyze_v2_vs_v3_start_coverage.py`](../../scripts/analyze_v2_vs_v3_start_coverage.py)).

## 대상 데이터셋

| 태그 | 경로 | expected episodes | 실제 episodes | expected videos | 실제 videos |
|---|---|---|---|---|---|
| `v2_clean` | `data/so101_cube_pick_drop_grid35_v2_clean` | 35 | 35 | 70 | 70 |
| `v3_clean` | `data/so101_cube_pick_drop_start_coverage_v3_clean` | 30 | 30 | 60 | 60 |
| `v4_train39` | `data/so101_cube_pick_drop_generalization_v4` | 39 | 39 | 78 | 78 |
| `v4_heldout6` | `data/so101_cube_pick_drop_v4_heldout10` (디렉터리명은 그대로 유지, 실제 6 episodes) | 6 | 6 | 12 | 12 |

모든 4개 데이터셋의 `meta/tasks.parquet`은 정확히 `"Pick up the cube and drop it into the bin."` 한
줄만 가지고 있음 (task string 불일치 없음).

---

## 1. Dataset integrity — [`dataset_integrity.csv`](dataset_integrity.csv)

4개 데이터셋 모두 다음을 통과:

- episode 수 / video 수가 기대값과 정확히 일치 (V4 train 39/78, V4 heldout 6/12 포함).
- `state`/`action` 모두 6D, feature 순서 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll, gripper`로 4개 데이터셋 동일.
- workspace + wrist 카메라 존재, 해상도 480x640x3 동일.
- NaN/Inf 없음 (`episodes_with_nan`/`episodes_with_inf` 모두 빈 리스트).
- episode_index 0..N-1 연속, 각 episode의 frame_index 0..len-1 연속, 전역 `index` 컬럼 연속.
- meta(episodes parquet) ↔ data(parquet) ↔ video(mp4) 3자 참조 무결성: 누락된 data/video 파일 0건,
  참조되지 않는 orphan video 파일 0건, meta에 선언된 episode length와 실제 로드된 length 불일치 0건.
- fps 메타데이터 모두 30.

episode length: V2 328.7±(15~34 static 구간 별도), V3 327.4, V4 train 328.1, V4 heldout 328.0
frame(약 10.9s) — 4개 데이터셋 모두 비슷한 episode 길이.

**결론: 4개 데이터셋 모두 integrity 이상 없음.**

---

## 2. FPS / frame timing 품질 검사 — [`timing_quality.csv`](timing_quality.csv)

### 중요 발견: 저장된 `timestamp`는 실측값이 아니라 합성값

`~/lerobot/src/lerobot/datasets/dataset_writer.py`의 `add_frame()`을 직접 확인한 결과:

```python
frame_index = self.episode_buffer["size"]
timestamp = frame_index / self._meta.fps
```

즉 LeRobot의 dataset writer는 매 프레임의 `timestamp`를 **실제 wall-clock 캡처 시각이 아니라
`frame_index / fps`로 그 자리에서 계산**해서 저장한다. 실제로 4개 데이터셋 전체에서 저장된
`timestamp`는 이상적인 1/30s 그리드에서 최대 `4.45e-7s`(float32 라운딩 오차 수준)만 벗어난다 —
즉 **분산이 0에 가까워서, 이 컬럼만으로는 기록 중 발생한 실제 loop 지연/저속을 원천적으로 알 수
없다.** `meta/episodes`의 `from_timestamp`/`to_timestamp` (`length/fps`)와 비디오 인코딩 fps
(`encode_video_frames`가 이미 캡처된 이미지 시퀀스를 고정 fps로 인코딩)도 마찬가지로 합성값이다.

이는 "console warning만으로 판정하지 말고 실제 저장 데이터로 판정하라"는 요구와 정직하게 배치되는
사실 관계다: **저장된 timestamp/frame 메타데이터 자체는 실제 loop 저속 이벤트를 담고 있지 않다.**

### 대신 사용한, 저장 데이터에 근거한 proxy: 완전 동일(exact-duplicate) 연속 프레임 검사

workspace 카메라 비디오를 직접 디코드(grayscale, 1/4 다운샘플, 프레임당 ~0.2s)해서, **연속한 두
프레임이 픽셀 단위로 완전히 동일한 경우**를 계산했다. 카메라 캡처 루프가 새 프레임을 시간 내에
받지 못하면 이전 프레임 버퍼가 그대로 재사용/인코딩되어 "완전 동일 프레임"으로 남는다 — 실제
정지 상태에서도 센서 노이즈 때문에 프레임이 완전히 동일해지는 경우는 거의 없으므로, 이 지표는
"영상 encode/decode 관점에서" 저속 캡처의 실제 흔적을 잡아낸다 (video container `nb_frames`도
parquet `frame_count`와 4개 데이터셋 전체 110개 episode 모두 정확히 일치 — 프레임 드롭으로 인한
개수 불일치는 없음).

| dataset | mean duplicate_frame_fraction | max_duplicate_run_length (전체) | n_big_gaps (run≥3) |
|---|---|---|---|
| v2_clean | 6.07% | 1 | 0 |
| v3_clean | 8.81% | 1 | 0 |
| v4_train39 | 6.53% | 1 | 0 |
| **v4_heldout6** | **5.66%** | 1 | 0 |

- 4개 데이터셋 전체 110 episode 중 어디에도 3연속 이상 동일 프레임(진짜 "gap")은 없음
  (`n_big_gaps` 합계 0, `max_duplicate_run_length` 항상 1 — 즉 산발적 단발 중복뿐).
- **V4 heldout6의 평균 중복 프레임 비율(5.66%)은 4개 데이터셋 중 가장 낮다** — V3(8.81%)보다도
  낮고 V4 train39(6.53%)와 비슷한 수준. 즉 저장된 영상 데이터를 기준으로는 heldout6이 다른
  데이터셋보다 눈에 띄게 나쁜 timing/frame-drop 징후를 보이지 않는다.
- heldout6 6개 episode 개별 값(3.06%~6.73%)도 서로 비슷해서, 특정 episode 하나만 뚜렷하게
  나쁜 경우는 없다.

**결론: 저장된 데이터(영상 프레임 수/exact-duplicate 비율) 기준으로는 heldout6에 재수집이
필요한 episode가 없다.** 다만 이 proxy는 콘솔에서 실제로 보고된 "30Hz 미달 loop"의 진짜 크기를
확정적으로 반증하는 것도 아니다 — LeRobot 저장 포맷 자체가 실제 wall-clock 타이밍을 보존하지
않기 때문에, "그 warning이 실제로 얼마나 심했는지"는 이 데이터셋만으로는 원천적으로 검증 불가능
하다는 점을 명시한다. V4 train39도 동일 방식으로 검사했고 heldout6과 같은 범위의 값을 보인다.

---

## 3. Start/static coverage 비교 — [`start_static_coverage.csv`](start_static_coverage.csv)

정의: 1deg 이상 cumulative displacement가 3프레임 연속 지속된 첫 시점(state[0]/action[0] 각각
자기 기준), static_length = state 기준 첫 movement frame (없으면 전체 길이).

| dataset | static fraction | static_length mean/median/p10/p90 (frames) | state first-move mean/median (s) | action first-move mean/median (s) |
|---|---|---|---|---|
| v2_clean | 7.61% | 25.0 / 25.0 / 20.0 / 30.6 | 0.833 / 0.833 | 0.714 / 0.700 |
| v3_clean | 13.52% | 44.3 / 38.5 / 20.9 / 73.4 | 1.476 / 1.283 | 1.351 / 1.117 |
| v4_train39 | 8.95% | 29.4 / 30.0 / 15.8 / 40.8 | 0.979 / 1.000 | 0.840 / 0.867 |
| v4_heldout6 | 9.76% | 32.0 / 30.5 / 25.5 / 40.0 | 1.067 / 1.017 | 0.806 / 0.750 |

4개 데이터셋 모두 "한 번도 안 움직인" episode는 없음 (`n_never_moved`=0).

- V4 train39의 static fraction(8.95%)은 V2(7.61%)보다 살짝 늘었지만 V3(13.52%)만큼 늘지는
  않았다 — hold-time 다양화라는 V3의 의도적 개선을 V4가 완전히 물려받지는 않았고, V2와 V3의
  중간 정도.
- static segment 길이의 흩어짐(p10~p90)도 V4 train39(15.8~40.8)가 V2(20.0~30.6)보다는 넓지만
  V3(20.9~73.4)보다는 좁다 — 마찬가지로 중간 수준.

---

## 4. Start pose diversity — [`start_pose_diversity.csv`](start_pose_diversity.csv)

frame-0 state 기준 pairwise L2 (6-joint, deg):

| dataset | n_ep | median | mean | p95 | max |
|---|---|---|---|---|---|
| v2_clean | 35 | 3.85 | 4.13 | 7.93 | 11.76 |
| v3_clean | 30 | 17.88 | 16.44 | 27.62 | 31.85 |
| v4_train39 | 39 | **16.65** | **16.24** | 27.65 | 37.29 |
| v4_heldout6 | 6 | 11.96 | 11.54 | 16.34 | 17.39 |

per-joint start std (deg):

| joint | v2_clean | v3_clean | v4_train39 |
|---|---|---|---|
| shoulder_pan | 0.85 | 9.56 | 7.78 |
| shoulder_lift | 1.19 | 3.46 | 4.95 |
| **elbow_flex** | 0.32 | **0.24** | **0.90** |
| wrist_flex | 1.73 | 7.59 | 7.96 |
| **wrist_roll** | 2.18 | **0.04** | **2.02** |
| gripper | 0.68 | 0.39 | 0.33 |

- **V4 train39는 V2보다 훨씬 자연스러운 start diversity를 확보했다** — pairwise L2 median이
  3.85deg(V2) → 16.65deg(V4)로 약 4.3배, V3(17.88deg)와 거의 동급.
- **elbow_flex**: V2 0.32deg → V3 0.24deg(오히려 더 나빠짐, V3는 elbow_flex를 다양화하지
  못했음) → **V4 0.90deg로 V2/V3 모두보다 명확히 개선**.
- **wrist_roll**: V2 2.18deg → V3 0.04deg(거의 고정, V3의 알려진 결함) → **V4 2.02deg로 V3
  대비 약 50배 개선, V2 수준을 거의 회복**.
- V4 train 마지막 4개 episode(35~38, 손목틀기 variation 포함)의 wrist_roll 시작값은
  -3.65~-1.45deg 범위로 나머지 35개(-5.93~1.63deg) 범위 안에 들어가며, 이 4개를 포함한 채로도
  wrist_roll std가 V2 수준을 회복한 것이지 이 4개만으로 만들어진 인위적 결과가 아니다.

**결론: V4 train39는 V2보다 start diversity가 뚜렷이 개선되었고, V3에서 부족했던
elbow_flex/wrist_roll coverage도 명확히 개선되었다.**

---

## 5. Immediate GT safety — [`immediate_action_safety.csv`](immediate_action_safety.csv)

정의: static 구간 프레임에서 `action(t) - state(t)`, Safety Gate `excessive_step_deg`
(`configs/safety_gate.yaml`) 기준 WOULD_CLAMP.

| dataset | shoulder_lift mean/p95/max (deg) | elbow_flex mean/p95/max | wrist_roll mean/p95/max | WOULD_CLAMP rate (모든 joint) |
|---|---|---|---|---|
| v2_clean | 0.61 / 2.46 / 4.66 | 2.25 / 4.00 / 6.02 | 0.24 / 0.53 / 0.70 | elbow_flex만 0.11% |
| v3_clean | 0.55 / 1.32 / 3.96 | 2.09 / 3.30 / 5.41 | 0.17 / 0.35 / 0.44 | 0% |
| v4_train39 | 0.59 / 1.93 / 4.22 | 2.21 / 3.82 / 6.29 | 0.21 / 0.44 / 0.70 | elbow_flex만 0.09% |
| v4_heldout6 | 0.58 / 1.01 / 3.34 | 2.66 / 4.00 / 4.62 | 0.22 / 0.48 / 0.53 | 0% |

- WOULD_CLAMP threshold: shoulder_lift 5.16deg, elbow_flex 5.73deg, wrist_roll 1.15deg
  (`configs/safety_gate.yaml`).
- 4개 데이터셋 모두 REJECT/과도 위반은 없고, elbow_flex의 WOULD_CLAMP도 V2(0.11%)와 V4
  train(0.09%)에서만 극소수(threshold를 살짝 넘는 outlier 몇 프레임) 존재 — V3/heldout은 0%.
- 손목틀기 variation episode(35~38)만 따로 봐도 shoulder_lift/elbow_flex/wrist_roll 모두
  WOULD_CLAMP 0%, max delta도 나머지 35개 episode 범위 안.

**결론: V4는 안전한 immediate target을 제공한다 — V2와 동급으로 극소수 outlier만 있고
gross 위반은 없음.**

---

## 6. Chunk future-motion (chunk_size=50) — [`chunk_future_motion.csv`](chunk_future_motion.csv)

정의: static 구간 각 프레임에서 50-step action chunk와 그 프레임의 state 차이, chunk-mean이
WOULD_CLAMP threshold를 넘는 static-frame 비율.

| dataset | shoulder_lift: chunk-mean exceeds WOULD_CLAMP | elbow_flex: chunk-mean exceeds WOULD_CLAMP |
|---|---|---|
| v2_clean | **99.66%** | **99.66%** |
| v3_clean | **60.69%** | **61.75%** |
| v4_train39 | **85.33%** | **84.54%** |
| v4_heldout6 | 81.25% | 90.10% |

- **V4 train39는 V2의 ~99.7% 문제보다는 뚜렷이 개선(85% 전후)됐지만, V3의 ~61%보다는
  명확히 나쁘다 (약 24~30%p 더 나쁨).** 즉 V4는 "V2보다 낫고 V3보다 못한" 중간 지점이며,
  task에서 우려한 "V3만큼 좋아졌는가"에 대해서는 **아니다(V3보다 후퇴)**가 정답이다.
  이는 3번 항목의 static-coverage 결과(V4의 static fraction/hold-time 다양성이 V2와 V3
  사이라는 것)와 정합적이다 — static일 때 immediate action 자체는 안전(섹션5)하지만, static
  observation과 짝지어진 50-step 미래 chunk는 여전히 V3보다 더 크게 움직인다.
- heldout6(81~90%)도 V4 train39와 비슷한 수준으로, 학습 목표와 평가 목표가 이 지표에서는
  일관되어 있다.

**결론: chunk future-motion 문제는 V2보다는 개선됐지만 V3 대비로는 개선이 아니라 후퇴다.**

---

## 7. Conflicting-label 후보 — [`conflicting_label_candidates.csv`](conflicting_label_candidates.csv) (CSV는 dataset별 상위 100건만 저장, 전체 통계는 summary.json)

정의: 같은 dataset 내에서 static 구간(최대 90프레임) 프레임 쌍 중 state L2 ≤ 3.0deg(6-joint)이면서
shoulder_lift 또는 elbow_flex의 `action(t)-state(t)` 차이가 ≥ 2.0deg인 쌍.

| dataset | close state pairs | conflict candidates | conflict rate |
|---|---|---|---|
| v2_clean | 137,313 | 19,044 | 13.87% |
| v3_clean | 75,486 | 4,982 | 6.60% |
| **v4_train39** | 18,011 | 2,334 | **12.96%** |
| v4_heldout6 | 0 | 0 | N/A (episode끼리 static-구간 state가 서로 3deg 이내로 가까운 경우가 아예 없음 — heldout 6개는 서로 충분히 다른 시작 위치를 목표로 수집됨을 시사) |

- **V4 train39의 conflict rate(12.96%)는 V3(6.60%)의 약 2배로, V3보다 명확히 나쁘다** — V2
  (13.87%)와 거의 같은 수준. 6번 항목(chunk future-motion)과 같은 방향의 결과: V4가 V3의
  static/hold-time 다양화 이점을 충분히 물려받지 못해 "가까운 시작 상태에 다른 즉시 라벨"이
  붙는 경우가 V3보다 늘었다.
- 다만 절대 수치로는 심각한 수준(예: 50%+)은 아니고, V2와 동급이라는 점에서 "V4가 V2 대비
  새로운 결함을 만든 것"은 아니다 — V3의 개선이 V4에서 유지되지 못했다는 것이 정확한 진단.

**결론: conflict label 문제는 심각한 수준(V2 대비 악화)은 아니지만, V3의 개선을 상실했다.**

---

## 8. V4 heldout6 leakage sanity check — [`heldout_leakage_check.csv`](heldout_leakage_check.csv)

cube 실제 x/y 좌표는 어느 데이터셋에도 메타데이터로 없으므로 위치를 추정하지 않고, train39×heldout6
전체 234쌍에 대해 3가지 저비용 체크만 수행:

1. **start-state(frame 0) L2** (near-duplicate threshold 1.0deg): 최솟값 1.21deg (heldout ep0
   ↔ train ep36), threshold 미만 0쌍. → **exact/near-duplicate 시작 자세 없음.**
2. **trajectory mean L2** (공통 길이 전체 state 평균 L2): 최솟값 10.55deg, median 32.95deg —
   가장 가까운 쌍도 자세가 계속 다른 궤적임.
3. **비디오 파일 hash(md5) 전건 비교** (workspace+wrist, train 78개 + heldout 12개 = 90개
   파일): exact match 0건.
4. **저비용 영상 유사도** (workspace cam에서 균등 추출 6프레임, 1/8 다운샘플 grayscale,
   L2 거리 — 모델 추론 없음): 최솟값 3467, median 6407 — 가장 가까운 쌍도 median과 크게
   다르지 않아 특별히 겹치는 영상이 없음.

가장 가까운 pair(heldout ep0 ↔ train ep36, start-state L2=1.21deg)도 전체 trajectory
L2(49.8deg)와 영상 특징 거리(7330)는 오히려 median급으로, 시작 자세만 우연히 비슷했을 뿐
동일/near-duplicate 수집이 아님.

**결론: train39와 heldout6 사이에 직접적인 복제/중복 증거는 없다.**

---

## 9. 최종 판정

1. **V4 train39 integrity는 정상인가?**
   → **예.** 39 episodes/78 videos 기대치와 정확히 일치, NaN/Inf/index 불연속/meta-data-video
   참조 불일치 모두 0건 ([섹션 1](#1-dataset-integrity--dataset_integritycsv)).

2. **heldout6 timing/FPS 품질은 평가용으로 충분한가?**
   → **저장된 데이터 기준으로는 충분하다.** 저장 timestamp는 합성값이라 실제 loop 저속을
   직접 증명/반증할 수 없지만, 유일하게 가용한 저장-데이터 proxy(영상 exact-duplicate 프레임
   비율)로는 heldout6(평균 5.66%)이 4개 데이터셋 중 **가장 낮은** 비율을 보이고, 6개 episode
   전부 3연속 이상 동일 프레임(진짜 gap)이 전혀 없다. 재수집이 필요하다고 판단할 episode는
   없음 ([섹션 2](#2-fps--frame-timing-품질-검사--timing_qualitycsv)).

3. **V4 static coverage는 V2/V3 대비 어떤가?**
   → **V2와 V3 중간.** static fraction 8.95%(V2 7.61%, V3 13.52%), static length 흩어짐도
   중간 수준 ([섹션 3](#3-startstatic-coverage-비교--start_static_coveragecsv)).

4. **V4 start pose diversity는 V2보다 개선됐는가?**
   → **예, 명확히.** pairwise L2 median 3.85deg(V2)→16.65deg(V4), 약 4.3배
   ([섹션 4](#4-start-pose-diversity--start_pose_diversitycsv)).

5. **elbow_flex / wrist_roll diversity는 V3보다 개선됐는가?**
   → **예, 명확히.** elbow_flex std 0.24deg(V3)→0.90deg(V4), wrist_roll std
   0.04deg(V3)→2.02deg(V4, V2의 2.18deg 수준을 거의 회복) ([섹션 4](#4-start-pose-diversity--start_pose_diversitycsv)).

6. **immediate GT action은 안전한가?**
   → **예.** WOULD_CLAMP rate가 V2와 동급으로 매우 낮음(elbow_flex 0.09%, 나머지 0%), 손목틀기
   variation episode(35~38)도 이상 없음 ([섹션 5](#5-immediate-gt-safety--immediate_action_safetycsv)).

7. **chunk future-motion 문제는 V2/V3 대비 어떤가?**
   → **V2보다는 개선(99.7%→85%), 그러나 V3보다는 후퇴(61%→85%).** V3에서 달성한 개선을
   V4가 온전히 유지하지 못함 ([섹션 6](#6-chunk-future-motion-chunk_size50--chunk_future_motioncsv)).

8. **conflict label 문제가 심각한가?**
   → **심각하지는 않지만(V2와 동급 수준), V3의 개선(6.60%)을 상실하고 12.96%로 되돌아감.**
   ([섹션 7](#7-conflicting-label-후보--conflicting_label_candidatescsv-csv는-dataset별-상위-100건만-저장-전체-통계는-summaryjson)).

9. **V4 train39를 다음 fresh training에 사용할 가치가 있는가?**
   → **가치는 있으나, 있는 그대로 쓰면 V3에서 확인됐던 완화 효과(chunk future-motion,
   conflicting-label)가 되돌아간다는 점을 인지하고 써야 한다.** Start-pose diversity(특히
   elbow_flex/wrist_roll)와 V2 대비 전반적 안전성은 V4가 명확히 더 낫다는 점에서 순수 교체
   용도로는 유의미한 개선이지만, "V2/V3 문제를 모두 해결한 데이터셋"은 아니다. (본 분석은
   merge/reweight/재학습 실험을 수행하지 않았음 — 이 판단은 온전히 정적 데이터 특성 비교에
   근거함.)

10. **heldout6을 그대로 봉인 테스트셋으로 사용할 수 있는가?**
    → **예.** train39와의 leakage sanity check(start-state/trajectory/video-hash/영상
    유사도) 전부에서 직접적 복제·중복 증거가 없고, integrity·timing 품질도 이상이 없다
    ([섹션 8](#8-v4-heldout6-leakage-sanity-check--heldout_leakage_checkcsv)). 디렉터리명은
    지시대로 변경하지 않았다 (`so101_cube_pick_drop_v4_heldout10`, 실제 6 episodes).

---

## 산출물

- `summary.md` (본 문서)
- `summary.json` — 위 모든 절의 원시 수치 (dataset_integrity 전체, 각 데이터셋의 timing/segment
  /diversity/safety/chunk 결과, conflicting-label summary, heldout leakage summary 포함)
- `dataset_integrity.csv`
- `timing_quality.csv`
- `start_static_coverage.csv`
- `start_pose_diversity.csv`
- `immediate_action_safety.csv`
- `chunk_future_motion.csv`
- `conflicting_label_candidates.csv` (dataset별 상위 100건, `max_key_joint_diff_deg` 내림차순)
- `heldout_leakage_check.csv` (train39×heldout6 전체 234쌍)

## 하지 않은 것 (지시대로)

- training / merge / reweight 실험 없음.
- 원본 dataset(`data/...`) 수정 없음 — 모두 read-only 로드.
- `data/so101_cube_pick_drop_v4_heldout10` 디렉터리명 변경 없음.
- git commit/push 없음.
- cube 실제 x/y 좌표 추정/생성 없음 (메타데이터에 없으므로 전 구간에서 proprioceptive
  state/action 및 영상 자체만 사용).
