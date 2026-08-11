# Grid35 SmolVLA - Held-out Midpoint10 Offline Evaluation

- eval dataset (held-out, 학습에 미사용): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_midpoint_test10_v2_clean`
- train dataset (provenance 확인용, 이 평가에는 사용 안 함): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_pick_drop_combined65_v1`
- task: Pick up the cube and drop it into the bin.
- seed: 42
- Safety Gate: 재사용됨 (`runtime/laptop/safety_gate.py`, write 없음)

> **중요**: 이 offline metric(action MAE/delta)은 실제 task success를 단정하지 않는다. train35(학습 데이터) 성능은 이번 checkpoint 선택 기준이 아니며, 아래는 전부 unseen midpoint10 기준이다. 이 표의 목적은 Shadow Mode에 올릴 후보 1~2개를 거르는 것이다.

## 핵심 비교 표 (unseen midpoint10 기준)

| checkpoint | frames | episodes | action MAE | delta median | delta p95 | delta max | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |
|---|---|---|---|---|---|---|---|---|---|
| 2500 | 3288 | 10 | 4.7157 | 3.3673 | 14.0368 | 43.0896 | 256 | 3011 | 21 |
| 5000 | 3288 | 10 | 4.3404 | 3.0473 | 17.5524 | 42.8774 | 375 | 2809 | 104 |
| 7500 | 3288 | 10 | 3.4123 | 2.3932 | 13.5816 | 36.9583 | 688 | 2590 | 10 |
| 10000 | 3288 | 10 | 3.6396 | 2.4188 | 15.3465 | 38.2296 | 659 | 2568 | 61 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.7912, p95=6.6374, max=15.1702

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 4.7625 | 7.4651 | 6.8734 | 3.4937 | 0.3154 | 5.3839 |
| 5000 | 3.9951 | 5.8459 | 7.2078 | 3.8694 | 0.2138 | 4.9105 |
| 7500 | 2.8530 | 4.7004 | 5.3052 | 2.9273 | 0.1774 | 4.5103 |
| 10000 | 2.8123 | 5.0518 | 6.0095 | 3.4248 | 0.1692 | 4.3700 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 1572 | 1863 | 1324 | 1191 | 16 | 1181 |
| 5000 | 1140 | 1499 | 1967 | 992 | 0 | 845 |
| 7500 | 632 | 1289 | 1724 | 844 | 0 | 517 |
| 10000 | 711 | 1346 | 1770 | 961 | 0 | 529 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

