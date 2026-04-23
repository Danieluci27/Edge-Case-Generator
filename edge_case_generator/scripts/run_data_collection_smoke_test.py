"""Run a tiny end-to-end collection pipeline smoke test."""

from __future__ import annotations

import pprint

from edge_case_generator.collection.pipeline import run_collection_pipeline
from edge_case_generator.scripts.collection_common import load_script_config


def main() -> None:
    config = load_script_config(
        "Run the packaged data-collection smoke test.",
        default_config="edge_case_generator/configs/collection_debug.yaml",
    )
    summary = run_collection_pipeline(config)
    pprint.pprint(summary)


if __name__ == "__main__":
    main()
