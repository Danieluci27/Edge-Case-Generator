"""Ground-truth solution selection."""

from __future__ import annotations

from statistics import median_low
from typing import Any

from edge_case_generator.collection.types import RawProblemBundle, ReferenceSolutionRecord


def _score_solution_length(code: str) -> int:
    return len(code.strip())


def select_reference_solutions(
    bundles: list[RawProblemBundle],
    *,
    target_language: str = "python",
    keep_multiple: bool = False,
    max_refs_per_problem: int = 1,
) -> list[ReferenceSolutionRecord]:
    """Select deterministic GT solutions from accepted/reference submissions."""

    records: list[ReferenceSolutionRecord] = []
    language = target_language.lower()
    for bundle in bundles:
        candidates = [item for item in bundle.accepted_solutions if item.language == language and item.code.strip()]
        if not candidates:
            continue
        if keep_multiple:
            ordered = sorted(candidates, key=lambda item: (_score_solution_length(item.code), item.solution_id))
            selected = ordered[:max_refs_per_problem]
        else:
            lengths = sorted(_score_solution_length(item.code) for item in candidates)
            target = median_low(lengths)
            selected = [
                min(
                    candidates,
                    key=lambda item: (abs(_score_solution_length(item.code) - target), _score_solution_length(item.code), item.solution_id),
                )
            ]
        for solution in selected:
            records.append(
                ReferenceSolutionRecord(
                    problem_id=bundle.problem.problem_id,
                    solution_id=solution.solution_id,
                    language=solution.language,
                    gt_code=solution.code,
                    source_label=solution.source_label,
                    metadata={"selection_strategy": "median_length", **solution.metadata},
                )
            )
    return records
