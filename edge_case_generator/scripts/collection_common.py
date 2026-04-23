"""Common helpers for collection scripts."""

from __future__ import annotations

import argparse

from edge_case_generator.collection.io import load_collection_config


def parse_collection_args(description: str, default_config: str | None = None) -> argparse.Namespace:
    """Parse CLI args for collection scripts."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, required=default_config is None, help="Path to YAML config")
    return parser.parse_args()


def load_script_config(description: str, default_config: str | None = None) -> dict:
    """Load a collection config for CLI scripts."""

    args = parse_collection_args(description, default_config=default_config)
    return load_collection_config(args.config)
