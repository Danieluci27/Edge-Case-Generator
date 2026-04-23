"""Generate synthetic buggy variants from GT code."""

from __future__ import annotations

from pathlib import Path

from edge_case_generator.collection.catalog import load_problem_bundles
from edge_case_generator.collection.io import write_records, write_summary
from edge_case_generator.collection.mutations import generate_synthetic_buggy_solutions
from edge_case_generator.collection.selection import select_reference_solutions
from edge_case_generator.scripts.collection_common import load_script_config


def main() -> None:
    config = load_script_config("Generate synthetic buggy programs from GT code.")
    bundles = load_problem_bundles(config)
    references = select_reference_solutions(
        bundles,
        target_language=config.get("target_language", "python"),
        keep_multiple=config.get("gt_selection", {}).get("keep_multiple", False),
        max_refs_per_problem=config.get("gt_selection", {}).get("max_refs_per_problem", 1),
    )
    buggy = generate_synthetic_buggy_solutions(
        references,
        per_reference_limit=config.get("synthetic_mutation", {}).get("per_reference_limit", 2),
        seed=config.get("seed", 0),
    )
    write_records(Path(config["_output_dir"]) / "buggy_programs.jsonl", buggy)
    write_summary(
        Path(config["_output_dir"]) / "generate_synthetic_bugs_summary.json",
        {"synthetic_buggy_solutions": len(buggy)},
    )


if __name__ == "__main__":
    main()
