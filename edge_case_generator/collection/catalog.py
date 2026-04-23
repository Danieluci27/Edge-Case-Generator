"""Adapter registry and dataset loading helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from edge_case_generator.collection.adapters import APPSAdapter, CodeContestsAdapter, CodeNetAdapter
from edge_case_generator.collection.types import ProblemRecord, RawProblemBundle


ADAPTERS = {
    "codenet": CodeNetAdapter,
    "codecontests": CodeContestsAdapter,
    "apps": APPSAdapter,
}


def load_problem_bundles(config: dict[str, Any]) -> list[RawProblemBundle]:
    """Load enabled sources according to the collection config."""

    target_language = str(config.get("target_language", "python")).lower()
    max_problems = config.get("max_problems")
    bundles: list[RawProblemBundle] = []
    for source_name, source_cfg in (config.get("sources") or {}).items():
        if not source_cfg.get("enabled"):
            continue
        adapter_cls = ADAPTERS[source_name]
        adapter = adapter_cls(
            path=source_cfg.get("path"),
            target_language=target_language,
            max_problems=source_cfg.get("max_problems", max_problems),
            source_config=source_cfg,
        )
        bundles.extend(adapter.load())
    return bundles


def summarize_bundles(bundles: list[RawProblemBundle]) -> dict[str, Any]:
    """Summarize loaded bundles."""

    source_breakdown: dict[str, int] = defaultdict(int)
    accepted = 0
    incorrect = 0
    for bundle in bundles:
        source_breakdown[bundle.problem.source] += 1
        accepted += len(bundle.accepted_solutions)
        incorrect += len(bundle.incorrect_solutions)
    return {
        "problem_count": len(bundles),
        "accepted_solution_candidates": accepted,
        "incorrect_solution_candidates": incorrect,
        "source_breakdown": dict(source_breakdown),
    }


def extract_problem_records(bundles: list[RawProblemBundle]) -> list[ProblemRecord]:
    """Return normalized problems only."""

    return [bundle.problem for bundle in bundles]
