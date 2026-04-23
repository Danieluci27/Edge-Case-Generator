"""Console and structured metric logging."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any


def configure_logging(output_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes both to console and file."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("edge_case_generator")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_path / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class MetricsLogger:
    """Structured metrics sink backed by JSONL and CSV."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self._csv_fieldnames: list[str] | None = None

    def log(self, metrics: dict[str, Any]) -> None:
        """Append metrics to JSONL and CSV."""

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

        if self._csv_fieldnames is None:
            self._csv_fieldnames = list(metrics.keys())
            with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_fieldnames)
                writer.writeheader()
                writer.writerow(metrics)
            return

        row = {field: metrics.get(field) for field in self._csv_fieldnames}
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_fieldnames)
            writer.writerow(row)

