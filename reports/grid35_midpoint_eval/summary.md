# Grid35 SmolVLA - Held-out Midpoint10 Offline Evaluation

- eval dataset (held-out, 학습에 미사용): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_midpoint_test10_v1`
- train dataset (provenance 확인용, 이 평가에는 사용 안 함): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_xy_grid35_v1`
- task: Pick up the cube and place it in the target area.
- seed: 42
- Safety Gate: 재사용됨 (`runtime/laptop/safety_gate.py`, write 없음)

> **중요**: 이 offline metric(action MAE/delta)은 실제 task success를 단정하지 않는다. train35(학습 데이터) 성능은 이번 checkpoint 선택 기준이 아니며, 아래는 전부 unseen midpoint10 기준이다. 이 표의 목적은 Shadow Mode에 올릴 후보 1~2개를 거르는 것이다.

## 핵심 비교 표 (unseen midpoint10 기준)

| checkpoint | frames | episodes | action MAE | delta median | delta p95 | delta max | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |
|---|---|---|---|---|---|---|---|---|---|
| 2500 | 2980 | 10 | 11.4432 | 8.9284 | 37.2705 | 85.1573 | 0 | 1883 | 1097 |
| 5000 | 2980 | 10 | 8.8611 | 5.9423 | 32.8580 | 81.7899 | 4 | 2255 | 721 |
| 7500 | 2980 | 10 | 8.4944 | 5.6487 | 32.4383 | 80.2716 | 3 | 2403 | 574 |
| 10000 | 2980 | 10 | 8.3882 | 5.6547 | 32.2876 | 80.8875 | 6 | 2410 | 564 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.9670, p95=8.3077, max=16.0319

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 10.4157 | 15.5666 | 19.0799 | 14.0492 | 0.5962 | 8.9515 |
| 5000 | 6.8654 | 12.7010 | 15.7021 | 9.2668 | 0.5061 | 8.1248 |
| 7500 | 6.8128 | 12.8401 | 14.3734 | 8.7652 | 0.4906 | 7.6845 |
| 10000 | 6.4097 | 12.3478 | 14.2284 | 9.3523 | 0.4951 | 7.4960 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 2007 | 1558 | 1825 | 2614 | 145 | 974 |
| 5000 | 1484 | 1406 | 1898 | 2739 | 141 | 730 |
| 7500 | 1404 | 1513 | 1875 | 2641 | 141 | 816 |
| 10000 | 1335 | 1448 | 1877 | 2641 | 141 | 778 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

