"""실제 SO-101에서 측정한 Control Profile Candidate를 이용해 MuJoCo 앞에 선택적으로
붙일 수 있는 "Realistic SO-101 Control Layer" 패키지.

이 패키지는 순수 파이썬 로직만 담는다 (``mujoco`` 모듈을 import하지 않는다) - MuJoCo API
호출은 항상 호출자(``simulation/mujoco/*``)쪽에 남겨둔다. 실물 팔로워에 쓰는 코드도
전혀 없다 (``hardware/safety/*`` 등 serial 접근 클래스를 import하지 않는다).
"""
