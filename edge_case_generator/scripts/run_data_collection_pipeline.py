"""Run the full collection pipeline."""

from __future__ import annotations

import pprint

from edge_case_generator.collection.pipeline import run_collection_pipeline
from edge_case_generator.scripts.collection_common import load_script_config


def main() -> None:
    config = load_script_config("Run the full data-collection pipeline.")
    summary = run_collection_pipeline(config)
    pprint.pprint(summary)


if __name__ == "__main__":
    main()
