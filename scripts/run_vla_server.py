#!/usr/bin/env python3
"""Desktop SmolVLA FastAPI 서버 실행 CLI.

실물 SO-101에 어떤 write도 하지 않는다 (이 스크립트는 GPU에서 추론만 한다).

Fake backend (체크포인트/GPU 없이 통신 계약만 검증):
    python scripts/run_vla_server.py --fake --host 0.0.0.0 --port 9200

Real backend (실제 SmolVLA 체크포인트):
    python scripts/run_vla_server.py --checkpoint /path/to/checkpoint --host 0.0.0.0 --port 9200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.desktop.vla_server import FakePolicyRunner, SmolVLAPolicyRunner, create_app

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9200
API_TOKEN_ENV_VAR = "VLA_SERVER_TOKEN"


def build_policy_runner(args: argparse.Namespace):
    """Build the runner whose effective seed is reported by /health."""
    if args.fake:
        return FakePolicyRunner()
    return SmolVLAPolicyRunner(
        args.checkpoint, policy_type=args.policy_type, device=args.device,
        inference_seed=args.inference_seed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Desktop SmolVLA FastAPI 서버")
    parser.add_argument("--fake", action="store_true", help="체크포인트/GPU 없이 Fake backend로 기동 (통신 계약 검증용)")
    parser.add_argument("--checkpoint", default=None, help="SmolVLA 체크포인트 경로 또는 HF repo id (--fake와 배타적)")
    parser.add_argument("--policy-type", default="smolvla")
    parser.add_argument("--device", default=None, help="cuda/cpu (미지정 시 자동 감지)")
    parser.add_argument("--inference-seed", type=int, default=None, help="선택: flow-matching noise seed; 미지정 시 stochastic")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-token", default=None)
    parser.add_argument(
        "--input-diagnostic-dir",
        default=None,
        help="선택: /predict_chunk 수신 JPEG hash/checksum/raw chunk JSONL 저장 디렉터리",
    )
    args = parser.parse_args(argv)

    if not args.fake and not args.checkpoint:
        parser.error("--fake 또는 --checkpoint 중 하나는 반드시 지정해야 합니다.")
    if args.fake and args.checkpoint:
        parser.error("--fake와 --checkpoint는 동시에 지정할 수 없습니다.")
    if args.inference_seed is not None and args.inference_seed < 0:
        parser.error("--inference-seed는 0 이상의 정수여야 합니다.")
    if args.fake and args.inference_seed is not None:
        parser.error("--inference-seed는 checkpoint backend에서만 사용할 수 있습니다.")
    return args


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)

    print("MODE=VLA_SERVER")
    print(f"BACKEND={'FAKE' if args.fake else 'SMOLVLA'}")
    print("REAL_SO101_WRITE=DISABLED (이 서버는 로봇에 어떤 명령도 보내지 않습니다)")

    policy_runner = build_policy_runner(args)
    if not args.fake:
        if not policy_runner.is_ready():
            print(f"[경고] 체크포인트 로딩에 실패했습니다 - /health가 degraded로 응답합니다: {policy_runner._load_error}")

    api_token = args.api_token or os.environ.get(API_TOKEN_ENV_VAR)
    app = create_app(
        policy_runner=policy_runner, api_token=api_token,
        input_diagnostic_dir=args.input_diagnostic_dir,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
