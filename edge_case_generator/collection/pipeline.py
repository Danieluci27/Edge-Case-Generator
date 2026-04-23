"""End-to-end collection pipeline orchestration."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from edge_case_generator.collection.buggy import collect_real_buggy_solutions
from edge_case_generator.collection.catalog import extract_problem_records, load_problem_bundles, summarize_bundles
from edge_case_generator.collection.execution import accept_or_reject_case, run_program_pair
from edge_case_generator.collection.io import write_records, write_summary
from edge_case_generator.collection.mutations import generate_synthetic_buggy_solutions
from edge_case_generator.collection.sampling import infer_constraints_from_problem, sample_valid_input
from edge_case_generator.collection.selection import select_reference_solutions
from edge_case_generator.collection.types import BuggySolutionRecord, EdgeCaseRecord, ProblemRecord, ReferenceSolutionRecord, RejectedCaseRecord
from edge_case_generator.utils.randomness import seed_everything
from edge_case_generator.verifier.executor import CodeExecutor


def collect_problems(config: dict[str, Any]) -> tuple[list[ProblemRecord], dict[str, Any], list[Any]]:
    """Load enabled sources and write normalized problem records."""

    bundles = load_problem_bundles(config)
    problems = extract_problem_records(bundles)
    summary = summarize_bundles(bundles)
    output_path = Path(config["_output_dir"]) / "problems.jsonl"
    write_records(output_path, problems)
    return problems, summary, bundles


def select_gt_solutions(config: dict[str, Any], bundles: list[Any]) -> list[ReferenceSolutionRecord]:
    """Select GT/reference programs and write them."""

    references = select_reference_solutions(
        bundles,
        target_language=config.get("target_language", "python"),
        keep_multiple=config.get("gt_selection", {}).get("keep_multiple", False),
        max_refs_per_problem=config.get("gt_selection", {}).get("max_refs_per_problem", 1),
    )
    write_records(Path(config["_output_dir"]) / "references.jsonl", references)
    return references


def collect_buggy_programs(config: dict[str, Any], bundles: list[Any], references: list[ReferenceSolutionRecord]) -> list[BuggySolutionRecord]:
    """Collect real buggy programs and optionally add synthetic variants."""

    real_buggy = collect_real_buggy_solutions(
        bundles,
        target_language=config.get("target_language", "python"),
        max_buggy_per_problem=config.get("buggy_collection", {}).get("max_real_buggy_per_problem"),
    )
    synthetic: list[BuggySolutionRecord] = []
    synthetic_cfg = config.get("synthetic_mutation", {})
    if synthetic_cfg.get("enabled", True):
        synthetic = generate_synthetic_buggy_solutions(
            references,
            per_reference_limit=synthetic_cfg.get("per_reference_limit", 2),
            seed=config.get("seed", 0),
        )
    buggy_programs = real_buggy + synthetic
    write_records(Path(config["_output_dir"]) / "buggy_programs.jsonl", buggy_programs)
    return buggy_programs


def generate_supervised_edge_cases(
    config: dict[str, Any],
    problems: list[ProblemRecord],
    references: list[ReferenceSolutionRecord],
    buggy_programs: list[BuggySolutionRecord],
) -> tuple[list[EdgeCaseRecord], list[RejectedCaseRecord], dict[str, Any]]:
    """Generate mismatch-only supervised edge cases."""

    execution_cfg = config.get("execution", {})
    sampling_cfg = config.get("sampling", {})
    compare_cfg = config.get("comparison", {})
    executor = CodeExecutor(
        timeout_sec=execution_cfg.get("timeout_sec", 1.0),
        python_executable=execution_cfg.get("python_executable", "python3"),
    )
    rng = random.Random(config.get("seed", 0))

    reference_by_problem: dict[str, list[ReferenceSolutionRecord]] = defaultdict(list)
    buggy_by_problem: dict[str, list[BuggySolutionRecord]] = defaultdict(list)
    for reference in references:
        reference_by_problem[reference.problem_id].append(reference)
    for buggy in buggy_programs:
        buggy_by_problem[buggy.problem_id].append(buggy)

    accepted_records: list[EdgeCaseRecord] = []
    rejected_records: list[RejectedCaseRecord] = []
    mismatch_per_problem: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    min_keep = sampling_cfg.get("min_mismatch_cases_per_problem", 1)

    for problem in problems:
        parsed = infer_constraints_from_problem(problem)
        gt_candidates = reference_by_problem.get(problem.problem_id, [])
        buggy_candidates = buggy_by_problem.get(problem.problem_id, [])
        if not gt_candidates or not buggy_candidates:
            continue
        gt = gt_candidates[0]
        attempts = 0
        while attempts < sampling_cfg.get("max_attempts_per_problem", 20):
            if mismatch_per_problem[problem.problem_id] >= min_keep and attempts >= sampling_cfg.get("num_samples_per_problem", 10):
                break
            sampled = sample_valid_input(problem, parsed, rng)
            for buggy in buggy_candidates:
                gt_result, buggy_result, comparison = run_program_pair(
                    executor,
                    gt_code=gt.gt_code,
                    buggy_code=buggy.buggy_code,
                    input_data=sampled.input_data,
                    comparison_mode=compare_cfg.get("mode", "normalized"),
                )
                accepted, rejected = accept_or_reject_case(
                    problem=problem,
                    gt_solution_id=gt.solution_id,
                    buggy_id=buggy.buggy_id,
                    bug_origin=buggy.bug_origin,
                    input_data=sampled.input_data,
                    gt_result=gt_result,
                    buggy_result=buggy_result,
                    comparison=comparison,
                    generator_metadata={
                        "seed": config.get("seed", 0),
                        "sample_metadata": sampled.metadata,
                        "comparison_mode": compare_cfg.get("mode", "normalized"),
                    },
                )
                if accepted is not None:
                    accepted_records.append(accepted)
                    mismatch_per_problem[problem.problem_id] += 1
                elif rejected is not None:
                    rejection_reasons[rejected.rejection_reason] += 1
                    if sampling_cfg.get("save_rejected_diagnostics", False):
                        rejected_records.append(rejected)
            attempts += 1

    write_records(Path(config["_output_dir"]) / "supervised_edge_cases.jsonl", accepted_records)
    if sampling_cfg.get("save_rejected_diagnostics", False):
        write_records(Path(config["_output_dir"]) / "rejected_cases_diagnostics.jsonl", rejected_records)

    summary = {
        "edge_case_count": len(accepted_records),
        "rejected_case_count": len(rejected_records),
        "mismatch_cases_per_problem": dict(mismatch_per_problem),
        "rejection_reason_breakdown": dict(rejection_reasons),
    }
    return accepted_records, rejected_records, summary


def run_collection_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full data-collection pipeline and emit normalized outputs."""

    seed_everything(config.get("seed", 0))
    problems, problem_summary, bundles = collect_problems(config)
    references = select_gt_solutions(config, bundles)
    buggy_programs = collect_buggy_programs(config, bundles, references)
    edge_cases, rejected, edge_summary = generate_supervised_edge_cases(config, problems, references, buggy_programs)

    source_breakdown: Counter[str] = Counter(problem.source for problem in problems)
    summary = {
        "problems_collected": len(problems),
        "gt_solutions": len(references),
        "real_buggy_solutions": sum(1 for item in buggy_programs if item.bug_origin == "real_submission"),
        "synthetic_buggy_solutions": sum(1 for item in buggy_programs if item.bug_origin == "synthetic_mutation"),
        "mismatch_edge_cases": len(edge_cases),
        "source_breakdown": dict(source_breakdown),
        "problem_summary": problem_summary,
        "edge_case_summary": edge_summary,
        "output_files": {
            "problems": str(Path(config["_output_dir"]) / "problems.jsonl"),
            "references": str(Path(config["_output_dir"]) / "references.jsonl"),
            "buggy_programs": str(Path(config["_output_dir"]) / "buggy_programs.jsonl"),
            "supervised_edge_cases": str(Path(config["_output_dir"]) / "supervised_edge_cases.jsonl"),
            "rejected_cases_diagnostics": str(Path(config["_output_dir"]) / "rejected_cases_diagnostics.jsonl"),
        },
    }
    write_summary(Path(config["_output_dir"]) / "collection_summary.json", summary)
    return summary
