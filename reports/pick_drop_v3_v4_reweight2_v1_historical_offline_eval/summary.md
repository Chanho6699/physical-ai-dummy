# Grid35 SmolVLA - Held-out Midpoint10 Offline Evaluation

- eval dataset (held-out, 학습에 미사용): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_midpoint_test10_v2_clean`
- train dataset (provenance 확인용, 이 평가에는 사용 안 함): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_pick_drop_v3_v4_combined69_v1`
- task: Pick up the cube and drop it into the bin.
- seed: 42
- Safety Gate: 재사용됨 (`runtime/laptop/safety_gate.py`, write 없음)

> **중요**: 이 offline metric(action MAE/delta)은 실제 task success를 단정하지 않는다. train35(학습 데이터) 성능은 이번 checkpoint 선택 기준이 아니며, 아래는 전부 unseen midpoint10 기준이다. 이 표의 목적은 Shadow Mode에 올릴 후보 1~2개를 거르는 것이다.

## 핵심 비교 표 (unseen midpoint10 기준)

| checkpoint | frames | episodes | action MAE | delta median | delta p95 | delta max | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |
|---|---|---|---|---|---|---|---|---|---|
| 2500 | 3288 | 10 | 5.4790 | 3.3322 | 16.7973 | 35.5991 | 108 | 3052 | 128 |
| 5000 | 3288 | 10 | 4.2098 | 2.3651 | 16.1043 | 36.8720 | 617 | 2600 | 71 |
| 7500 | 3288 | 10 | 3.9863 | 2.0639 | 14.8806 | 38.8011 | 1070 | 2169 | 49 |
| 10000 | 3288 | 10 | 3.8431 | 2.1117 | 14.7032 | 37.8210 | 956 | 2312 | 20 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.7912, p95=6.6374, max=15.1702

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 5.1443 | 8.4558 | 9.4404 | 2.7577 | 1.5069 | 5.5689 |
| 5000 | 3.2840 | 6.7106 | 6.6594 | 2.4273 | 0.6562 | 5.5213 |
| 7500 | 3.0653 | 6.3921 | 6.5429 | 2.4840 | 0.4686 | 4.9651 |
| 10000 | 2.8076 | 6.0187 | 6.2180 | 2.5332 | 0.4078 | 5.0735 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 1515 | 1961 | 1789 | 713 | 1889 | 626 |
| 5000 | 822 | 1491 | 1700 | 553 | 382 | 657 |
| 7500 | 553 | 1216 | 1324 | 553 | 73 | 582 |
| 10000 | 529 | 1232 | 1449 | 612 | 59 | 596 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

