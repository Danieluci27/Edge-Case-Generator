"""Generate mismatch-only supervised edge cases."""

from __future__ import annotations

from edge_case_generator.collection.pipeline import collect_buggy_programs, collect_problems, generate_supervised_edge_cases, select_gt_solutions
from edge_case_generator.collection.io import write_summary
from edge_case_generator.scripts.collection_common import load_script_config


def main() -> None:
    config = load_script_config("Generate mismatch-only supervised edge cases.")
    problems, _, bundles = collect_problems(config)
    references = select_gt_solutions(config, bundles)
    buggy = collect_buggy_programs(config, bundles, references)
    accepted, rejected, summary = generate_supervised_edge_cases(config, problems, references, buggy)
    write_summary(
        f"{config['_output_dir']}/generate_supervised_edge_cases_summary.json",
        {
            "accepted_edge_cases": len(accepted),
            "rejected_cases": len(rejected),
            **summary,
        },
    )


if __name__ == "__main__":
    main()
