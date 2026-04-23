"""Collect normalized problem records."""

from __future__ import annotations

from edge_case_generator.collection.pipeline import collect_problems
from edge_case_generator.collection.io import write_summary
from edge_case_generator.scripts.collection_common import load_script_config


def main() -> None:
    config = load_script_config("Collect normalized problem records.")
    problems, summary, _ = collect_problems(config)
    write_summary(
        f"{config['_output_dir']}/collect_problems_summary.json",
        {"problems_collected": len(problems), **summary},
    )


if __name__ == "__main__":
    main()
