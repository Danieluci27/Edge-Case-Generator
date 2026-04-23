"""Select GT/reference solutions."""

from __future__ import annotations

from edge_case_generator.collection.catalog import load_problem_bundles
from edge_case_generator.collection.io import write_summary
from edge_case_generator.collection.pipeline import select_gt_solutions
from edge_case_generator.scripts.collection_common import load_script_config


def main() -> None:
    config = load_script_config("Select GT/reference solutions.")
    bundles = load_problem_bundles(config)
    references = select_gt_solutions(config, bundles)
    write_summary(
        f"{config['_output_dir']}/select_gt_solutions_summary.json",
        {"gt_solutions": len(references)},
    )


if __name__ == "__main__":
    main()
