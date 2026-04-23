"""Adapter for local APPS-style dataset files."""

from __future__ import annotations

from typing import Any

from edge_case_generator.collection.adapters.base import DatasetAdapter
from edge_case_generator.collection.types import RawProblemBundle


class APPSAdapter(DatasetAdapter):
    """Normalize local APPS-style records."""

    source_name = "apps"

    def normalize_row(self, row: dict[str, Any]) -> RawProblemBundle:
        question = str(row.get("question") or row.get("problem_statement") or "")
        samples = list(row.get("samples") or [])
        public_tests = list(row.get("public_tests") or [])
        input_output = row.get("input_output")
        if input_output and not public_tests:
            public_tests = [
                {"input": inp, "output": out}
                for inp, out in zip(input_output.get("inputs", []), input_output.get("outputs", []))
            ]
        metadata = dict(row.get("metadata") or {})
        metadata["difficulty"] = row.get("difficulty")
        problem, parsed = self.build_problem(
            source_problem_id=str(row.get("problem_id") or row.get("id")),
            title=str(row.get("title") or row.get("name") or row.get("problem_id")),
            statement=question,
            source_url=row.get("url"),
            samples=samples,
            public_tests=public_tests,
            time_limit=row.get("time_limit"),
            memory_limit=row.get("memory_limit"),
            metadata=metadata,
        )
        accepted = [
            self.build_solution(item, fallback_prefix=f"{problem.problem_id}_accepted", index=index)
            for index, item in enumerate(row.get("solutions") or row.get("accepted_solutions") or [], start=1)
        ]
        incorrect = [
            self.build_solution(item, fallback_prefix=f"{problem.problem_id}_incorrect", index=index)
            for index, item in enumerate(row.get("incorrect_solutions") or [], start=1)
        ]
        return RawProblemBundle(problem=problem, parsed_constraints=parsed, accepted_solutions=accepted, incorrect_solutions=incorrect)
