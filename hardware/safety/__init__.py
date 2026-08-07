"""실물 SO-101 하드웨어에 안전하게(읽기 전용/dry-run) 접근하기 위한 도구 모음.

이 패키지의 모듈은 명시적으로 "이번 단계에서 구현/실행 금지"로 표시된 부분(write,
armed 실행)을 제외하면 모두 읽기 전용이거나 순수 계산이다. 자세한 설계는
``hardware/safety/single_joint_test_planner.py``의 모듈 docstring과
``docs/single_joint_hardware_test.md``(있다면)를 참고한다.
"""
