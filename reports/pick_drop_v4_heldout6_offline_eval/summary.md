# Grid35 SmolVLA - Held-out Midpoint10 Offline Evaluation

- eval dataset (held-out, 학습에 미사용): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_pick_drop_v4_heldout10`
- train dataset (provenance 확인용, 이 평가에는 사용 안 함): `/home/rlack/Projects/physical-ai-dummy/physical-ai-dummy/data/so101_cube_pick_drop_generalization_v4`
- task: Pick up the cube and drop it into the bin.
- seed: 42
- Safety Gate: 재사용됨 (`runtime/laptop/safety_gate.py`, write 없음)

> **중요**: 이 offline metric(action MAE/delta)은 실제 task success를 단정하지 않는다. train35(학습 데이터) 성능은 이번 checkpoint 선택 기준이 아니며, 아래는 전부 unseen midpoint10 기준이다. 이 표의 목적은 Shadow Mode에 올릴 후보 1~2개를 거르는 것이다.

## 핵심 비교 표 (unseen midpoint10 기준)

| checkpoint | frames | episodes | action MAE | delta median | delta p95 | delta max | WOULD_PASS | WOULD_CLAMP | WOULD_REJECT |
|---|---|---|---|---|---|---|---|---|---|
| 2500 | 1968 | 6 | 5.0403 | 3.3044 | 16.7152 | 42.4463 | 79 | 1584 | 305 |
| 5000 | 1968 | 6 | 4.1940 | 2.2661 | 14.9608 | 44.5700 | 270 | 1557 | 141 |
| 7500 | 1968 | 6 | 3.6034 | 2.0078 | 13.2792 | 39.6422 | 347 | 1486 | 135 |
| 10000 | 1968 | 6 | 3.5634 | 1.9720 | 13.0393 | 39.0674 | 342 | 1503 | 123 |

## Ground-truth demonstration state->action delta (checkpoint 무관, 참고용 baseline)

- median=0.7912, p95=5.8901, max=10.8942

## Joint별 action MAE

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 5.2617 | 7.4178 | 7.8953 | 2.9502 | 2.5046 | 4.2123 |
| 5000 | 3.4565 | 6.9139 | 6.3773 | 2.6312 | 1.2734 | 4.5115 |
| 7500 | 2.9702 | 5.8787 | 5.5122 | 2.3830 | 1.1140 | 3.7621 |
| 10000 | 2.7945 | 5.6948 | 5.5197 | 2.4027 | 1.1386 | 3.8300 |

## Joint별 WOULD_CLAMP count

| checkpoint | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| 2500 | 1073 | 910 | 1060 | 447 | 837 | 514 |
| 5000 | 414 | 714 | 799 | 403 | 615 | 463 |
| 7500 | 381 | 621 | 850 | 371 | 458 | 341 |
| 10000 | 342 | 613 | 870 | 368 | 465 | 296 |

## 한계 (읽는 사람이 반드시 알아야 함)

- teacher forcing 평가다: 매 프레임 policy에게 넣는 state/이미지는 데이터셋에 실제로 기록된 값이지, policy 자신이 이전 스텝에서 만든 action을 따라간 결과가 아니다. 즉 진짜 closed-loop rollout이 아니라 "기록된 궤적을 그대로 재생하면서 각 시점에서 policy라면 무엇을 예측했을까"를 보는 것이다.
- SmolVLA는 action chunk(＝chunk_size)를 한 번에 생성하고 재사용한다 - 실제로 매 프레임마다 새로 forward하지 않고, chunk가 소진될 때만(이번 설정 기준 매 50프레임마다) 새로 샘플링한다. 이는 실제 배포(`runtime/desktop/vla_server.py`)와 동일한 동작이다.
- action MAE/delta는 offline metric이다 - 실제 task success(큐브를 집어서 target에 놓았는지)를 이 수치만으로 단정할 수 없다.

