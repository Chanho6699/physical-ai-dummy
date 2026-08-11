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
| 2500 | 3288 | 10 | 5.0635 | 2.8308 | 15.8607 | 39.3725 | 453 | 2785 | 50 |
| 5000 | 3288 | 10 | 4.2991 | 2.3775 | 16.0320 | 43.6393 | 713 | 2547 | 28 |
| 7500 | 3288 | 10 | 3.7791 | 2.0933 | 13.7795 | 38.0122 | 863 | 2424 | 1 |
| 10000 | 3288 | 10 | 3.7802 | 2.0263 | 13.9795 | 35.9813 | 966 | 2321 | 1 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.7912, p95=6.6374, max=15.1702

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 4.1877 | 7.8131 | 7.9092 | 2.8200 | 1.0082 | 6.6426 |
| 5000 | 3.3401 | 6.2731 | 6.1387 | 2.8221 | 0.4778 | 6.7431 |
| 7500 | 2.8585 | 5.7259 | 5.8712 | 2.5949 | 0.4789 | 5.1449 |
| 10000 | 2.7793 | 5.8936 | 6.0071 | 2.5201 | 0.4079 | 5.0731 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 997 | 1715 | 1504 | 678 | 935 | 660 |
| 5000 | 710 | 1416 | 1429 | 774 | 142 | 802 |
| 7500 | 495 | 1322 | 1470 | 651 | 117 | 594 |
| 10000 | 489 | 1239 | 1383 | 647 | 61 | 552 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

