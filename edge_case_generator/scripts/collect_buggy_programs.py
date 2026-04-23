"""Collect real buggy programs from dataset adapters."""

from __future__ import annotations

from pathlib import Path

from edge_case_generator.collection.buggy import collect_real_buggy_solutions
from edge_case_generator.collection.catalog import load_problem_bundles
from edge_case_generator.collection.io import write_records, write_summary
from edge_case_generator.scripts.collection_common import load_script_config


def main() -> None:
    config = load_script_config("Collect buggy programs from public dataset files.")
    bundles = load_problem_bundles(config)
    buggy = collect_real_buggy_solutions(
        bundles,
        target_language=config.get("target_language", "python"),
        max_buggy_per_problem=config.get("buggy_collection", {}).get("max_real_buggy_per_problem"),
    )
    write_records(Path(config["_output_dir"]) / "buggy_programs.jsonl", buggy)
    write_summary(
        Path(config["_output_dir"]) / "collect_buggy_programs_summary.json",
        {"real_buggy_solutions": len(buggy)},
    )


if __name__ == "__main__":
    main()
