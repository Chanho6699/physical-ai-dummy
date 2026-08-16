# V1~V3 Blue Cube (blurry-camera) 실험 archival 요약

> 이 문서는 2026-08-16에 진행한 archival cleanup의 기록이다. 데이터/checkpoint를
> 삭제했을 뿐, 아래 나열된 reports/scripts/설정 파일의 내용은 이 정리 과정에서
> 변경하지 않았다 (git history도 그대로).

## 1. 대상 실험 흐름

"Pick up the blue cube, place it inside the blue rectangle labeled 'BLUE', then
return to the starting pose." task를 목표로 진행했던 데이터 수집/학습 캠페인.
두 대의 webcam(workspace/wrist)의 **물리적 초점이 흐린 상태**로 전체 캠페인이
진행됐다.

| 단계 | 데이터셋 | episodes | 목적 |
|---|---|---|---|
| V1 | `data/so101_blue_cube_place_return_v1` (+ `..._v1_backup`, 10-episode 부분 백업) | 41 | 최초 place & return demonstration |
| V2 | `data/so101_blue_cube_place_return_v2` | 61 | V1 대비 demonstration 추가/보강 |
| V3 | `data/so101_blue_cube_grasp_precision_v3` | 52 | grasp 단계 정밀도 보강에 집중한 추가 수집 |
| V1+V3 merged | (별도 디렉터리로 로컬에 존재한 적 없음) | 93 (41+52) | V1과 V3를 합쳐 학습한 checkpoint. `reports/real_pick_drop_merged93_7500*`가 이 checkpoint(step 7500)로 실물에서 돌린 기록 |

checkpoint(`outputs/blue_cube_place_return_v1/...`, `outputs/blue_cube_place_return_v2/...`,
V1+V3 merged 등)는 학습이 GPU를 가진 Desktop 머신에서 이루어졌고, 이 저장소(laptop
쪽)에는 애초에 로컬 사본이 없었다 - 그래서 이번 cleanup에서 로컬에서 실제로 지운 것은
`data/` 아래의 원본 데이터셋 4개뿐이다 (§3 참고).

## 2. 폐기(archival) 이유

- 두 webcam(workspace/wrist)의 **물리적 초점이 흐린 상태**로 V1~V3 전체가 촬영됐다.
- 이 조건에서도 **coarse localization(대략적인 물체/타겟 위치 파악)은 어느 정도
  동작**했다 - 완전히 못 쓰는 데이터는 아니었다.
- 하지만 **final grasp precision(마지막 grasp 단계의 정밀도)이 계속 좋지 않았다** -
  V3("grasp_precision")를 별도로 더 모으고 V1과 merge(93 episode, checkpoint 7500)해
  봐도 이 문제가 해소되지 않았다.
- **주의**: blur를 이 실패의 유일한 원인으로 단정하지 않는다. grasp precision 저하에는
  다른 요인(캘리브레이션, demonstration 품질/다양성, 카메라 프레이밍 등)이 같이 얽혀
  있었을 가능성이 있고, 이번 조사에서 blur 하나만으로 인과관계를 확정하지 못했다.
  이 archival은 "새 sharp-camera 하드웨어로 전환하면서 이 시기 데이터를 더 이상
  학습에 쓰지 않기로 한다"는 운영상의 결정이지, "blur가 근본 원인으로 확인됐다"는
  결론을 의미하지 않는다.

## 3. 이번 cleanup에서 실제로 지운 것

로컬(`data/`)에 있던 원본 LeRobot 데이터셋 4개 디렉터리:

- `data/so101_blue_cube_place_return_v1`
- `data/so101_blue_cube_place_return_v1_backup`
- `data/so101_blue_cube_place_return_v2`
- `data/so101_blue_cube_grasp_precision_v3`

checkpoint/`outputs/` 쪽은 로컬에 애초에 사본이 없어(§1 참고) 지운 대상이 없었다.
단, `outputs/pick_drop_v3_v4_combined69`는 2026-08-16 후속 점검에서 삭제 대상으로
추가 확인되어 삭제했다 - §7 참고.

## 4. 보존한 것

- `reports/` 전체(예: `real_pick_drop_merged93_7500*`, `real_pick_drop_v3_7500_*`,
  `real_pick_drop_realtime_v1`, `real_pick_drop_realtime_canonical_a_7500`,
  `v1_canonicalized_training_view`, `v1_initial_target_semantics_analysis`,
  `v1_7500_grasp_phase_policy_vs_gt_seed20260815`,
  `v1_7500_counterfactual_4way_seed20260815` 등) - 원본 데이터 없이도 읽을 수 있는
  분석 결과/로그/이미지는 그대로 둔다.
- 이 데이터셋들을 다루던 분석/빌드 스크립트(`scripts/analyze_v1_grasp_phase_policy_vs_gt.py`,
  `scripts/design_v1_canonicalized_training_view.py`,
  `scripts/analyze_motion_guard_recalibration.py` 등) - 원본 데이터가 없어 재실행은
  안 되지만, 당시 분석 로직의 기록으로 그대로 둔다.
- `configs/intent_gross_outlier.yaml`, `configs/motion_guard_tracking.yaml`의
  `provenance`/`generated_from` 블록에 남아 있는 `data/so101_blue_cube_place_return_v1`,
  `..._v2` 경로 - 이 값들은 코드에서 파일을 실제로 여는 데 쓰이지 않고(순수 metadata,
  §5 참고) 캘리브레이션 수치가 어떤 데이터로부터 유도됐는지 보여주는 provenance 기록이라
  archival cleanup 취지(기존 문서 보존)에 맞춰 손대지 않았다.
- Git history 전체 - 이번 정리는 파일시스템 삭제만이며 git commit/push는 하지 않았다.

## 5. 삭제 후 stale reference 점검 결과

`data/so101_blue_cube_place_return_v1`, `..._v1_backup`, `..._v2`,
`data/so101_blue_cube_grasp_precision_v3` 경로를 참조하는 코드를 전수 점검했다:

- **분석/빌드 스크립트**(`scripts/analyze_v1_grasp_phase_policy_vs_gt.py`,
  `scripts/design_v1_canonicalized_training_view.py`,
  `scripts/analyze_motion_guard_recalibration.py`) - 원본 데이터를 직접 읽는 one-off
  분석 도구. 재실행하면 지금은 실패하지만, 이는 "당시 분석 기록 보존" 목적과 archival
  cleanup의 자연스러운 결과이며 §번 요구사항("scripts 보존")대로 코드는 수정하지
  않았다.
- **`scripts/run_real_pick_drop_realtime.py`, `scripts/run_real_follower_staged_safety_test.py`**
  (현재 실행용 실물 로봇 제어 스크립트) - `DEFAULT_CHECKPOINT`가
  `outputs/blue_cube_place_return_v1/.../checkpoints/010000/pretrained_model`을
  가리킨다. 다만 이 경로는 **로컬 파일로 열리지 않는다** - `_checkpoint_signature()`가
  경로의 마지막 4개 구성요소만 문자열로 잘라 Desktop이 `/health`로 보고하는
  `model_id`와 부분 일치하는지만 검사하는 용도이고, 애초에 이 checkpoint의 로컬
  사본은 이번 삭제 이전에도 존재하지 않았다(§1). 즉 이번 삭제로 새로 깨진 동작은
  없다. 다만 이 기본값 자체는 이미 archival 대상이 된 캠페인을 가리키고 있고, 아직
  이를 대체할 검증된 sharp-camera checkpoint가 없으므로(§2, "fresh baseline을
  아직 새로 시작하는 단계") 실제로 이 스크립트를 다시 돌릴 때는 `--checkpoint`를
  **반드시 명시적으로 지정**해서 Desktop이 로딩한 checkpoint와 일치하는지 확인해야
  한다. 이 상수를 어떤 값으로 바꿀지는 다음 sharp-camera 학습이 나온 뒤에 결정할
  문제라 판단해 이번 cleanup에서는 코드를 건드리지 않았다(추측성 기본값 변경을 피함).
- **`configs/intent_gross_outlier.yaml`, `configs/motion_guard_tracking.yaml`** - 위
  경로들을 `provenance`/`generated_from.datasets` 필드에 문자열로만 갖고 있고,
  로딩 코드(`runtime/laptop/intent_validation.py`)는 이 블록을 그대로 opaque
  dict로만 보관할 뿐 파일을 열지 않는다(`tests/test_intent_validation.py`,
  `tests/test_motion_guard.py`로 확인). 실제로 여는 코드가 없으므로 기능상
  깨지는 부분이 없어 수정하지 않았다.
- 그 외 `configs/`, `hardware/`, `runtime/`의 어떤 실행용 default/설정도 이
  4개 삭제된 경로를 직접 참조하지 않는다.

**결론: 이번 삭제로 인해 실제로 크래시하거나 잘못된 동작을 하게 되는 실행용
config/default는 없었다.** 위 checkpoint 기본값 건은 코드를 고치기보다 "명시적으로
`--checkpoint`를 지정할 것"이라는 운영 주의사항으로 남긴다.

## 6. 새 sharp-camera setup

물리적으로 초점을 맞춘 새 webcam 두 대로 교체한 뒤에는, 이 V1~V3 blurry-camera
데이터를 새 데이터에 섞지 않고 **fresh baseline**으로 새로 시작한다. 즉 V1~V3는
새 학습 데이터의 일부로 재사용하지 않으며, 이 문서와 `reports/`에 남은 기록으로만
참조한다.

## 7. 후속 정리 (2026-08-16) - `outputs/pick_drop_v3_v4_combined69` 삭제

Desktop에서 provenance를 재확인한 결과 `pick_drop_v3_v4_combined69`는 새
sharp-camera V4가 아니라 **구형 blurry-camera V3 + 구형 V4 데이터를 합친 과거
모델**로 확인됐다 (checkpoint `train_config.json`의 `dataset.repo_id` =
`local/so101_cube_pick_drop_v3_v4_combined69_v1`, `reports/pick_drop_v2_v3_v4_dataset_analysis`가
분석한 `so101_cube_pick_drop_start_coverage_v3_clean`(30 episode, 구형 V3) +
`so101_cube_pick_drop_generalization_v4`(39 episode, 구형 V4) = 69 episode와 정확히
일치). §1의 blue-cube place-return V1~V3(다른 task 계열)와는 별개지만, 마찬가지로
구형/blurry-camera 계열 데이터로 학습된 checkpoint였다. Desktop에서 이미 삭제됐고,
laptop 쪽 사본(`outputs/pick_drop_v3_v4_combined69/smolvla_pick_drop_v3_v4_combined69_uniform_fresh/checkpoints/010000/pretrained_model`,
865MB)도 동일 provenance를 확인한 뒤 삭제했다.

이로써 **현재 sharp-camera V4는 아직 촬영되지 않았고, `data/`와 `outputs/` 모두 이
저장소에 비어 있는 상태가 최종 기준**이다.
