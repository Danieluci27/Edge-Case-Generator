"""I/O helpers for JSON and JSONL payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path, ignore_errors: bool = False) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if ignore_errors:
                    continue
                raise ValueError(f"Malformed JSONL record at line {line_number} in {path}")
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records to JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
