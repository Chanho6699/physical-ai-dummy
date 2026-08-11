# Real follower staged safety test (Candidate B) — 사용 안내

이 저장소가 실제 SO-101 follower에 처음으로 write하는 경로다. 반드시 **로봇 옆에 물리적으로
있으면서 즉시 개입(전원 차단 등)할 수 있는 상태**에서만 실행한다.

이 세션(cloud/원격 환경)에는 시리얼 장치가 전혀 연결되어 있지 않아 직접 검증하지 못했다 -
아래 코드는 fake follower 기반 단위/통합 테스트(`tests/test_staged_follower_writer.py`,
`tests/test_staged_real_rollout.py`, 17개 전부 통과)로만 검증되었다. 실제 하드웨어 검증은
사용자가 로봇이 연결된 머신에서 직접 실행해야 한다.

## 절대 불변식

- Safety Gate 임계값(`configs/safety_gate.yaml`, `configs/follower_safe_mapper.yaml`)은 바뀌지 않는다.
- `ACCEPT` action만 실제로 write된다. `WOULD_CLAMP`/`REJECT`는 절대 write하지 않는다.
- 첫 번째 non-ACCEPT에서 그 stage는 즉시 멈춘다.
- stage별 hard step 상한: stage1=1(고정), stage2=3~5, stage3=1~15. CLI로 못 늘림.
- stage는 자동으로 다음 단계로 안 넘어간다 - 각 stage는 별도 실행이고, stage 2/3는 바로 전
  stage의 PASS 영수증이 없으면 시작을 거부한다.

## 실행 순서 (반드시 이 순서로, 매 단계 로봇을 직접 확인한 뒤)

```bash
source ~/lerobot/.venv/bin/activate   # 로봇이 연결된 실제 laptop에서

# 0) 먼저 dry-run으로 계획만 확인 (하드웨어 미접근 - 포트를 열지 않음)
python scripts/run_real_follower_staged_safety_test.py --stage 1 --dry-run \
  --hardware-config configs/hardware.local.json \
  --follower-port /dev/serial/by-id/<실제 팔로워 포트> --follower-id <실제 follower id>

# 1) stage 1: single accepted action - 로봇을 보면서 실행
python scripts/run_real_follower_staged_safety_test.py --stage 1 \
  --hardware-config configs/hardware.local.json \
  --follower-port /dev/serial/by-id/<...> --follower-id <...> \
  --confirm-physically-present
# -> "STAGE 1: PASS"가 나오고 robot이 예상대로 움직였는지 직접 확인.
#    이상하면(PASS라도) 여기서 멈추고 다음 stage로 넘어가지 말 것.

# 2) stage 1이 PASS면 stage 2 (3~5 step)
python scripts/run_real_follower_staged_safety_test.py --stage 2 --steps 3 \
  --hardware-config configs/hardware.local.json \
  --follower-port /dev/serial/by-id/<...> --follower-id <...> \
  --confirm-physically-present

# 3) stage 2가 PASS면 stage 3 (short chunk, 최대 15 step)
python scripts/run_real_follower_staged_safety_test.py --stage 3 --steps 10 \
  --hardware-config configs/hardware.local.json \
  --follower-port /dev/serial/by-id/<...> --follower-id <...> \
  --confirm-physically-present
```

각 실행은 `--confirm-physically-present` 플래그 + 실행 중 타이핑 확인
(`I AM PHYSICALLY PRESENT AND WATCHING THE ROBOT`) 둘 다 요구한다.

## 산출물

- `reports/real_follower_staged_safety_test_v1/reports/stageN_<timestamp>.json` — 매 step의
  before/after state, delta, safety decision, write 여부 전체 기록.
- `reports/real_follower_staged_safety_test_v1/receipts/stageN_receipt.json` — 그 stage가
  PASS했다는 영수증. 다음 stage 실행에 필요하다. `ABNORMAL_STOPPED`면 만들어지지 않는다.

## ABNORMAL_STOPPED가 나오면

절대 같은 stage를 바로 재시도하지 말 것. 리포트 JSON에서 `stop_reason`과 마지막 step의
`safety_decision`/`safety_reasons`를 먼저 읽고, 왜 non-ACCEPT가 나왔는지(관측 문제/action
자체가 과격함/그 외) 파악한 뒤에 판단한다.

## 아직 하지 않은 것 (요청 범위 밖)

cube 접근 테스트는 이 3-stage가 전부 PASS한 뒤에만, 별도 요청 시 진행한다 - 이 세션은
cube-접근 코드를 만들지 않았다.
