"""Desktop(RTX GPU) 측 SmolVLA FastAPI 추론 서버.

이 패키지는 실물 SO-101에 어떤 write도 하지 않는다 (serial/모터 SDK를 import하지
않는다). GPU에서 SmolVLA 체크포인트를 로딩해 관측(observation)을 받아 action을
반환하는 순수 추론 서버 역할만 한다.
"""
