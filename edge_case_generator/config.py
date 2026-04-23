"""Configuration loading utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly in this environment
    yaml = None


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this project."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, raw_value = raw_line.strip().partition(":")
        value = raw_value.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value == "":
            new_dict: dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _parse_scalar(value)
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    contents = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        config = yaml.safe_load(contents) or {}
    else:
        config = _load_simple_yaml(contents)
    return config


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Apply dotted-key overrides to a configuration dictionary."""

    result = deepcopy(config)
    if not overrides:
        return result

    for key, value in overrides.items():
        cursor: dict[str, Any] = result
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return result
