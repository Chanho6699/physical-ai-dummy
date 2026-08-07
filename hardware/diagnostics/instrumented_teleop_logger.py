"""Instrumented Teleop Diagnostic CSV/JSON 기록 - 파일 직렬화만 담당한다.

``hardware/diagnostics/instrumented_teleop.py``의 데이터 구조(``TeleopCycleSample``)를
CSV 행으로, 분석 결과 dict를 JSON으로 저장하는 순수 I/O 헬퍼다. 하드웨어 접근이 없어
``tests/test_instrumented_teleop_logger.py``에서 ``tmp_path``만으로 테스트한다.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from hardware.diagnostics.instrumented_teleop import CSV_FIELDNAMES, TeleopCycleSample

__all__ = [
    "CSV_FILENAME_PREFIX",
    "build_csv_path",
    "build_json_report_path",
    "CsvSampleWriter",
    "write_json_report",
]

CSV_FILENAME_PREFIX = "instrumented_wrist_roll"


def build_csv_path(directory: Path, *, timestamp: str | None = None) -> Path:
    """섹션 13: ``instrumented_wrist_roll_YYYYMMDD_HHMMSS.csv`` 경로를 만든다."""
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return directory / f"{CSV_FILENAME_PREFIX}_{ts}.csv"


def build_json_report_path(csv_path: Path) -> Path:
    """CSV와 같은 timestamp를 공유하는 ``..._report.json`` 경로를 만든다."""
    return csv_path.with_name(csv_path.stem + "_report.json")


class CsvSampleWriter:
    """CSV 파일을 열고 헤더를 쓴 뒤, ``TeleopCycleSample``마다 한 행씩 append한다.

    write 계열 하드웨어 메서드와는 무관한 순수 파일 I/O 클래스다 - 이름의 "writer"는
    ``csv.DictWriter``를 가리킨다.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: TextIO = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=list(CSV_FIELDNAMES))
        self._writer.writeheader()

    def write_sample(self, sample: TeleopCycleSample) -> None:
        self._writer.writerow(sample.to_csv_row())

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
