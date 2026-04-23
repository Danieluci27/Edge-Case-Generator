"""Typed records used by the data-collection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ConstraintVariable:
    """Structured scalar constraint for a named variable."""

    name: str
    type: str = "int"
    min_value: int | None = None
    max_value: int | None = None
    length_min: int | None = None
    length_max: int | None = None
    alphabet: str | None = None
    relation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArrayConstraint:
    """Structured array/list constraint."""

    name: str
    length_ref: str | None = None
    length_min: int | None = None
    length_max: int | None = None
    elem_min: int | None = None
    elem_max: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StringConstraint:
    """Structured string constraint."""

    name: str
    length_ref: str | None = None
    length_min: int | None = None
    length_max: int | None = None
    alphabet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedConstraints:
    """Structured and raw constraint representation."""

    raw_constraint_text: str
    input_constraints: list[str] = field(default_factory=list)
    output_constraints: list[str] = field(default_factory=list)
    variables: list[ConstraintVariable] = field(default_factory=list)
    arrays: list[ArrayConstraint] = field(default_factory=list)
    strings: list[StringConstraint] = field(default_factory=list)
    multiple_test_cases: bool = False
    test_case_count_var: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_constraint_text": self.raw_constraint_text,
            "input_constraints": list(self.input_constraints),
            "output_constraints": list(self.output_constraints),
            "variables": [item.to_dict() for item in self.variables],
            "arrays": [item.to_dict() for item in self.arrays],
            "strings": [item.to_dict() for item in self.strings],
            "multiple_test_cases": self.multiple_test_cases,
            "test_case_count_var": self.test_case_count_var,
        }


@dataclass
class ProblemRecord:
    """Normalized problem record."""

    problem_id: str
    source: str
    source_problem_id: str
    source_url: str | None
    title: str
    problem_statement: str
    input_constraints: list[str]
    output_constraints: list[str]
    raw_constraint_text: str
    sample_inputs: list[str]
    sample_outputs: list[str]
    public_tests: list[dict[str, Any]]
    language: str
    time_limit: str | None = None
    memory_limit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceSolutionRecord:
    """Ground-truth reference program."""

    problem_id: str
    solution_id: str
    language: str
    gt_code: str
    source_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuggySolutionRecord:
    """Buggy program record."""

    problem_id: str
    buggy_id: str
    language: str
    buggy_code: str
    bug_origin: str
    original_verdict: str | None = None
    mutation_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeCaseRecord:
    """Accepted semantic mismatch edge case."""

    problem_id: str
    edge_case_id: str
    input_data: str
    gt_output: str
    buggy_output: str
    output_equal: bool
    gt_solution_id: str
    buggy_id: str
    bug_origin: str
    generator_metadata: dict[str, Any] = field(default_factory=dict)
    validation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RejectedCaseRecord:
    """Rejected sampled input with diagnostics."""

    problem_id: str
    input_data: str
    gt_status: str
    buggy_status: str
    rejection_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RawSolution:
    """Adapter-level solution payload."""

    solution_id: str
    language: str
    code: str
    source_label: str
    verdict: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawProblemBundle:
    """Adapter-level normalized problem plus attached solutions."""

    problem: ProblemRecord
    parsed_constraints: ParsedConstraints
    accepted_solutions: list[RawSolution] = field(default_factory=list)
    incorrect_solutions: list[RawSolution] = field(default_factory=list)

