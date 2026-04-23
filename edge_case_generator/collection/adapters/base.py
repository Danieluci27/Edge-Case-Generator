"""Base adapter utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from edge_case_generator.collection.parsing import parse_problem_assets
from edge_case_generator.collection.types import ProblemRecord, RawProblemBundle, RawSolution
from edge_case_generator.utils.io import read_jsonl


class DatasetAdapter:
    """Base class for local packaged dataset adapters."""

    source_name: str = "unknown"

    def __init__(
        self,
        path: str | Path | None,
        target_language: str = "python",
        max_problems: int | None = None,
        source_config: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.target_language = target_language.lower()
        self.max_problems = max_problems
        self.source_config = dict(source_config or {})

    def load(self) -> list[RawProblemBundle]:
        """Load and normalize dataset rows."""

        rows = self._load_rows()
        bundles: list[RawProblemBundle] = []
        for row in rows:
            bundle = self.normalize_row(row)
            bundles.append(bundle)
            if self.max_problems is not None and len(bundles) >= self.max_problems:
                break
        return bundles

    def _load_rows(self) -> Iterable[dict[str, Any]]:
        if self.path is None:
            raise ValueError(f"{self.source_name} adapter requires either a local path or a source-specific loader config")
        if self.path.suffix == ".jsonl":
            return read_jsonl(self.path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "records" in data:
            return data["records"]
        raise ValueError(f"Unsupported dataset format for {self.path}")

    def normalize_row(self, row: dict[str, Any]) -> RawProblemBundle:
        """Normalize a raw row into the common bundle type."""

        raise NotImplementedError

    def build_problem(
        self,
        *,
        source_problem_id: str,
        title: str,
        statement: str,
        source_url: str | None,
        samples: list[dict[str, str]],
        public_tests: list[dict[str, Any]],
        time_limit: str | None,
        memory_limit: str | None,
        metadata: dict[str, Any],
    ) -> tuple[ProblemRecord, Any]:
        parsed = parse_problem_assets(statement, samples=samples, metadata=metadata)
        problem_id = f"{self.source_name}:{source_problem_id}"
        record = ProblemRecord(
            problem_id=problem_id,
            source=self.source_name,
            source_problem_id=source_problem_id,
            source_url=source_url,
            title=title,
            problem_statement=statement,
            input_constraints=parsed.input_constraints,
            output_constraints=parsed.output_constraints,
            raw_constraint_text=parsed.raw_constraint_text,
            sample_inputs=[sample.get("input", "") for sample in samples],
            sample_outputs=[sample.get("output", "") for sample in samples],
            public_tests=public_tests,
            language=self.target_language,
            time_limit=time_limit,
            memory_limit=memory_limit,
            metadata=metadata,
        )
        return record, parsed

    def build_solution(self, row: dict[str, Any], *, fallback_prefix: str, index: int) -> RawSolution:
        """Normalize a solution-like payload."""

        language = str(row.get("language", self.target_language)).lower()
        return RawSolution(
            solution_id=str(row.get("id") or f"{fallback_prefix}_{index}"),
            language=language,
            code=str(row.get("code") or row.get("solution") or ""),
            source_label=str(row.get("source_label") or self.source_name),
            verdict=row.get("verdict"),
            metadata=dict(row.get("metadata") or {}),
        )
