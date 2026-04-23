"""Adapter for local Project CodeNet-style files."""

from __future__ import annotations

from typing import Any

from edge_case_generator.collection.adapters.base import DatasetAdapter
from edge_case_generator.collection.types import RawProblemBundle


class CodeNetAdapter(DatasetAdapter):
    """Normalize local Project CodeNet-style records."""

    source_name = "codenet"

    def normalize_row(self, row: dict[str, Any]) -> RawProblemBundle:
        statement_parts = [str(row.get("statement") or row.get("problem_statement") or "")]
        constraints = row.get("constraints")
        if constraints:
            statement_parts.append(f"Constraints\n{constraints}")
        if row.get("input_description"):
            statement_parts.append(f"Input\n{row['input_description']}")
        if row.get("output_description"):
            statement_parts.append(f"Output\n{row['output_description']}")
        statement = "\n\n".join(part for part in statement_parts if part)
        samples = list(row.get("samples") or [])
        public_tests = list(row.get("public_tests") or [])
        metadata = dict(row.get("metadata") or {})
        problem, parsed = self.build_problem(
            source_problem_id=str(row.get("problem_id") or row.get("id")),
            title=str(row.get("title") or row.get("name") or row.get("problem_id")),
            statement=statement,
            source_url=row.get("url"),
            samples=samples,
            public_tests=public_tests,
            time_limit=row.get("time_limit"),
            memory_limit=row.get("memory_limit"),
            metadata=metadata,
        )
        accepted = [
            self.build_solution(item, fallback_prefix=f"{problem.problem_id}_accepted", index=index)
            for index, item in enumerate(row.get("accepted_submissions") or row.get("accepted_solutions") or [], start=1)
        ]
        incorrect = [
            self.build_solution(item, fallback_prefix=f"{problem.problem_id}_incorrect", index=index)
            for index, item in enumerate(row.get("wrong_submissions") or row.get("incorrect_solutions") or [], start=1)
        ]
        return RawProblemBundle(problem=problem, parsed_constraints=parsed, accepted_solutions=accepted, incorrect_solutions=incorrect)
