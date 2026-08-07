# SO-101 Control Profile Candidate v1

- status: **CANDIDATE_ONLY**
- source: instrumented_teleop_6runs
- run_count: 6
- apply_automatically: **False**
- source_aggregate_path: `reports/instrumented_teleop/aggregate_6runs_20260807_094532.json`
- generated_at: 2026-08-07T10:09:37.174140+00:00

> 이 문서는 실측 6-run instrumented teleop 통계를 VLA/follower 제어용 후보 기준표로 정리한 것입니다. **어떤 값도 실제 robot runtime에 적용되지 않았습니다.**

## 현재 확보한 것

- joint별 historical operating range (observed min/max, p01/p99)
- joint별 frame delta 분포 (p50/p95/p99/max) + candidate soft limit
- joint별 velocity 분포 (p50/p95/p99/max) + candidate soft limit
- joint별 tracking error 분포 (MAE/p95/p99/max) + warning/severe candidate
- wrist_roll deadband candidate (0~5 tick NO_RESPONSE, 6+ tick TRANSITION, HIGH_RESPONSE는 미확립)
- local instrumented teleop 계측 latency (leader command -> follower actual, end-to-end 아님)

## 향후 VLA에서 어디에 사용할지 (계획만 - 이번 작업에서 코드에 적용하지 않음)

```text
VLA action
   |
   v
Action Adapter
   |
   v
candidate deadband / rate profile   <- 이 문서/JSON (CANDIDATE_ONLY)
   |
   v
Follower command
   |
   v
Execution Monitor
   |
   v
tracking error / latency monitoring
```

## Joint별 candidate 값 (pooled 6-run aggregate)

| joint | unit | confidence | historical inner range (p01~p99) | frame_delta p95/p99 | velocity p95/p99 | tracking_error p95/p99 |
|---|---|---|---|---|---|---|
| shoulder_pan | degree | MEDIUM | [-89.714, 81.104] | 1.055 / 1.407 | 62.965 / 83.795 | 7.473 / 9.670 |
| shoulder_lift | degree | MEDIUM | [-98.593, 58.251] | 0.879 / 1.143 | 52.478 / 68.134 | 5.626 / 7.473 |
| elbow_flex | degree | HIGH | [-71.516, 96.571] | 0.967 / 1.319 | 57.699 / 78.443 | 6.813 / 9.275 |
| wrist_flex | degree | MEDIUM | [-66.330, 84.352] | 1.407 / 1.846 | 82.967 / 109.073 | 8.571 / 11.473 |
| wrist_roll | degree | LOW | [-71.165, 90.507] | 1.231 / 1.758 | 73.396 / 104.370 | 7.209 / 10.375 |
| gripper | percent_0_100 | MEDIUM | [1.173, 72.050] | 1.242 / 2.001 | 73.659 / 118.620 | 7.808 / 14.586 |

gripper는 arm 5개 joint(degree)와 달리 percent_0_100 semantics를 씁니다 (hardware/state_server/readonly_so101_reader.py, configs/follower_safe_mapper.yaml 확인). gripper의 range/frame_delta/velocity candidate 값을 degree로 해석하지 마세요.

## wrist_roll deadband candidate

- no_response_region_ticks: [0, 5] (confidence: HIGH)
- transition_region_start_ticks: 6 (confidence: MEDIUM)
- transition_region_aggregate_response_fraction: 0.7060 (runs_with_response: 4/6)
- **high_response_region: NOT_ESTABLISHED**
  - 6+ tick 버킷의 aggregate response_fraction은 약 70.6%로, 이 저장소가 쓰는 HIGH_RESPONSE_REGION 판정 기준(>80%)에 못 미친다. 게다가 6개 run 중 2개(20260807_093136, 20260807_093204)는 이 구간에서 실제 움직임 표본 자체가 없어 판단에서 사실상 제외된다. '6 ticks = guaranteed motion'이라고 단정할 수 없다.

| tick | degree equivalent |
|---|---|
| 0 | 0.0000° |
| 1 | 0.0879° |
| 2 | 0.1758° |
| 3 | 0.2637° |
| 4 | 0.3516° |
| 5 | 0.4396° |
| 6 | 0.5275° |

> 이 구간 분류는 분석 편의용이며 hardware safety threshold로 확정된 것이 아닙니다. 6 ticks가 '항상 반응하는 지점'을 보장하지 않습니다.

## Control timing profile

- nominal_control_hz: 59.13 (confidence: HIGH)
- observed_latency_ms: median=92.90 min=67.69 max=101.57 std=13.97 valid_runs=4/6 (confidence: MEDIUM)
- **latency_scope: local_instrumented_teleop_command_to_actual**
  - 이 latency는 leader->follower local instrumented teleop 계측에서 얻은 'command to actual' 값입니다. 아직 Desktop VLA -> network -> Laptop -> robot의 end-to-end latency가 아닙니다.

## Confidence legend

- **HIGH**: 6개 run에서 반복적으로 안정적으로 관측됨 (run-to-run cv < 0.3, 또는 6-run 전체에서 반복 확인된 정성적 패턴).
- **MEDIUM**: 데이터는 있으나 run 간 변동(cv)이 상당하거나, 일부 run에서만 유효한 표본이 관측됨.
- **LOW**: 일부 run에서만 관찰되었거나 run 간 변동(cv)이 매우 큼.
- **INSUFFICIENT_DATA**: candidate를 계산할 근거 표본이 부족하거나 없음.

## 기존 follower-safe mapper (configs/follower_safe_mapper.yaml, read-only 비교)

| joint | 기존 rate_limit_deg_per_sec | 6-run observed velocity p95 | candidate soft limit (p99) | verdict |
|---|---|---|---|---|
| shoulder_pan | 20.00 | 62.96 | 83.80 | CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP |
| shoulder_lift | 15.00 | 52.48 | 68.13 | CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP |
| elbow_flex | 20.00 | 57.70 | 78.44 | CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP |
| wrist_flex | 15.00 | 82.97 | 109.07 | CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP |
| wrist_roll | 25.00 | 73.40 | 104.37 | CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP |
| gripper | 30.00 | 73.66 | 118.62 | CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP |

이 표는 `configs/follower_safe_mapper.yaml`을 **읽기만** 했습니다 - 수정하지 않았습니다. `CURRENT_LIMIT_MORE_CONSERVATIVE_THAN_TELEOP`은 기존에 미검증 상태로 넣어둔 rate limit이 실제 6-run teleop에서 관측된 velocity p95보다 낮다(더 보수적이다)는 뜻일 뿐, 지금 기존 설정이 틀렸다거나 바꿔야 한다는 의미는 아닙니다.

## 적용 제한

- 이 candidate 값은 실제 로봇 제어 코드에 자동 적용되지 않습니다 (apply_automatically=false).
- 이 파일 생성 과정에서 leader/follower connect, teleop 실행, servo write는 발생하지 않았습니다.
- safety threshold로 확정된 값이 아니며, production config가 아닙니다.

---
direct_register_write_count=0, hardware_execution_count=0, git_commit_count=0