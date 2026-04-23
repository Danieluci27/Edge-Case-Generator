"""Adapter for local CodeContests dataset files."""

from __future__ import annotations

from typing import Any, Iterable

from edge_case_generator.collection.adapters.base import DatasetAdapter
from edge_case_generator.collection.types import RawProblemBundle


LANGUAGE_ID_TO_NAME = {
    0: "unknown",
    1: "python",
    2: "cpp",
    3: "java",
    4: "python",
    5: "go",
    6: "rust",
}


class CodeContestsAdapter(DatasetAdapter):
    """Normalize local CodeContests-style records."""

    source_name = "codecontests"

    def _load_dataset_api(self):
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Hugging Face dataset loading requires the 'datasets' package. "
                "Install it to use sources.codecontests.hf_dataset."
            ) from exc
        return load_dataset

    def _load_rows(self) -> Iterable[dict[str, Any]]:
        dataset_name = self.source_config.get("hf_dataset")
        if dataset_name:
            return self._load_huggingface_rows(dataset_name)
        return super()._load_rows()

    def _load_huggingface_rows(self, dataset_name: str) -> Iterable[dict[str, Any]]:
        load_dataset = self._load_dataset_api()
        dataset = load_dataset(
            dataset_name,
            name=self.source_config.get("hf_name"),
            split=self.source_config.get("hf_split", "train"),
            revision=self.source_config.get("hf_revision"),
            cache_dir=self.source_config.get("hf_cache_dir"),
            streaming=self.source_config.get("hf_streaming", False),
        )
        return dataset

    def _normalize_test_pairs(self, payload: Any) -> list[dict[str, str]]:
        if not payload:
            return []
        if isinstance(payload, list):
            normalized: list[dict[str, str]] = []
            for item in payload:
                if isinstance(item, dict) and ("input" in item or "output" in item):
                    normalized.append(
                        {
                            "input": str(item.get("input", "")),
                            "output": str(item.get("output", "")),
                        }
                    )
            return normalized
        if isinstance(payload, dict):
            input_values = payload.get("input") or payload.get("inputs") or []
            output_values = payload.get("output") or payload.get("outputs") or []
            return [
                {"input": str(inp), "output": str(out)}
                for inp, out in zip(input_values, output_values)
            ]
        return []

    def _language_name(self, value: Any) -> str:
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, int):
            return LANGUAGE_ID_TO_NAME.get(value, f"lang_{value}")
        return self.target_language

    def _normalize_solution_bundle(self, payload: Any, key_prefix: str) -> list[dict[str, Any]]:
        if not payload:
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []

        languages = payload.get("language") or payload.get("languages") or payload.get("language_id") or payload.get("language_ids") or []
        codes = payload.get("solution") or payload.get("solutions") or payload.get("code") or payload.get("codes") or []
        if isinstance(codes, str):
            codes = [codes]
        if isinstance(languages, (str, int)):
            languages = [languages] * len(codes)
        rows: list[dict[str, Any]] = []
        for index, code in enumerate(codes, start=1):
            language = self._language_name(languages[index - 1]) if index - 1 < len(languages) else self.target_language
            rows.append(
                {
                    "id": f"{key_prefix}_{index}",
                    "language": language,
                    "code": code,
                }
            )
        return rows

    def normalize_row(self, row: dict[str, Any]) -> RawProblemBundle:
        statement = str(row.get("description") or row.get("problem_statement") or "")
        public_tests = self._normalize_test_pairs(row.get("public_tests"))
        private_tests = self._normalize_test_pairs(row.get("private_tests"))
        generated_tests = self._normalize_test_pairs(row.get("generated_tests"))
        samples = list(row.get("samples") or public_tests[: min(2, len(public_tests))])
        metadata = dict(row.get("metadata") or {})
        metadata["hf_dataset"] = self.source_config.get("hf_dataset")
        if private_tests:
            metadata["private_tests_count"] = len(private_tests)
        if generated_tests:
            metadata["generated_tests_count"] = len(generated_tests)
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
        accepted_payload = row.get("accepted_solutions") or row.get("solutions")
        incorrect_payload = row.get("incorrect_solutions")
        accepted = [
            self.build_solution(item, fallback_prefix=f"{problem.problem_id}_accepted", index=index)
            for index, item in enumerate(self._normalize_solution_bundle(accepted_payload, f"{problem.problem_id}_accepted"), start=1)
        ]
        incorrect = [
            self.build_solution(item, fallback_prefix=f"{problem.problem_id}_incorrect", index=index)
            for index, item in enumerate(self._normalize_solution_bundle(incorrect_payload, f"{problem.problem_id}_incorrect"), start=1)
        ]
        return RawProblemBundle(problem=problem, parsed_constraints=parsed, accepted_solutions=accepted, incorrect_solutions=incorrect)
