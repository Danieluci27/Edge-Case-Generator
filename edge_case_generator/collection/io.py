"""Collection-specific config and output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edge_case_generator.config import load_config
from edge_case_generator.utils.io import write_json, write_jsonl


def load_collection_config(path: str | Path) -> dict[str, Any]:
    """Load a collection config and resolve the output directory."""

    config = load_config(path)
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config["_config_path"] = str(path)
    config["_output_dir"] = str(output_dir)
    return config


def write_records(path: str | Path, records: list[Any]) -> str:
    """Write dataclass-like records as JSONL."""

    serialized = [record.to_dict() if hasattr(record, "to_dict") else record for record in records]
    write_jsonl(path, serialized)
    return str(path)


def write_summary(path: str | Path, payload: dict[str, Any]) -> str:
    """Write a JSON summary."""

    write_json(path, payload)
    return str(path)
