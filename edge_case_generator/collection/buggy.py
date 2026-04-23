"""Buggy program collection helpers."""

from __future__ import annotations

from typing import Any

from edge_case_generator.collection.types import BuggySolutionRecord, RawProblemBundle


def collect_real_buggy_solutions(
    bundles: list[RawProblemBundle],
    *,
    target_language: str = "python",
    max_buggy_per_problem: int | None = None,
) -> list[BuggySolutionRecord]:
    """Collect publicly available incorrect solutions exposed by dataset adapters."""

    language = target_language.lower()
    results: list[BuggySolutionRecord] = []
    for bundle in bundles:
        selected = [
            item for item in bundle.incorrect_solutions if item.language == language and item.code.strip()
        ]
        if max_buggy_per_problem is not None:
            selected = selected[:max_buggy_per_problem]
        for item in selected:
            results.append(
                BuggySolutionRecord(
                    problem_id=bundle.problem.problem_id,
                    buggy_id=item.solution_id,
                    language=item.language,
                    buggy_code=item.code,
                    bug_origin="real_submission",
                    original_verdict=item.verdict,
                    mutation_type=None,
                    metadata=item.metadata,
                )
            )
    return results
