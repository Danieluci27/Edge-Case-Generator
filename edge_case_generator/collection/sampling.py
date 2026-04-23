"""Constraint-aware input sampling."""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from typing import Any

from edge_case_generator.collection.types import ParsedConstraints, ProblemRecord


@dataclass
class SampledInput:
    """Generated input and validation metadata."""

    input_data: str
    metadata: dict[str, Any]


def _safe_int_bounds(low: int | None, high: int | None) -> tuple[int, int]:
    return (low if low is not None else -10, high if high is not None else 10)


def _sample_int(low: int | None, high: int | None, rng: random.Random) -> int:
    lower, upper = _safe_int_bounds(low, high)
    if lower > upper:
        lower, upper = upper, lower
    return rng.randint(lower, upper)


def _infer_shape_from_sample(sample: str) -> list[list[str]]:
    return [line.split() for line in sample.strip().splitlines() if line.strip()]


def infer_constraints_from_problem(problem: ProblemRecord) -> ParsedConstraints:
    """Create a minimal parsed constraint object from stored problem fields."""

    from edge_case_generator.collection.parsing import parse_problem_assets

    samples = [{"input": inp, "output": out} for inp, out in zip(problem.sample_inputs, problem.sample_outputs)]
    return parse_problem_assets(problem.problem_statement, samples=samples, metadata=problem.metadata)


def sample_valid_input(problem: ProblemRecord, parsed: ParsedConstraints, rng: random.Random) -> SampledInput:
    """Sample an input using parsed constraints and conservative fallbacks."""

    values: dict[str, Any] = {}
    lines: list[str] = []

    for variable in parsed.variables:
        if variable.type == "int":
            values[variable.name] = _sample_int(variable.min_value, variable.max_value, rng)

    multiple_case_count = None
    if parsed.multiple_test_cases:
        count_var = parsed.test_case_count_var or "t"
        multiple_case_count = values.get(count_var)
        if multiple_case_count is None:
            multiple_case_count = max(1, _sample_int(1, 3, rng))
            values[count_var] = multiple_case_count
        lines.append(str(multiple_case_count))

    payload_lines: list[str] = []
    for variable in parsed.variables:
        if parsed.multiple_test_cases and variable.name == parsed.test_case_count_var:
            continue
        payload_lines.append(str(values[variable.name]))

    for array in parsed.arrays:
        length = values.get(array.length_ref or "")
        if length is None:
            length = _sample_int(array.length_min or 1, array.length_max or 5, rng)
            if array.length_ref:
                values[array.length_ref] = length
                if not any(item == str(length) for item in payload_lines):
                    payload_lines.insert(0, str(length))
        array_values = [_sample_int(array.elem_min, array.elem_max, rng) for _ in range(max(0, int(length)))]
        payload_lines.append(" ".join(str(item) for item in array_values))

    for text_constraint in parsed.strings:
        length = values.get(text_constraint.length_ref or "")
        if length is None:
            length = _sample_int(text_constraint.length_min or 1, text_constraint.length_max or 8, rng)
        alphabet = string.ascii_lowercase if text_constraint.alphabet == "ascii_lowercase" else string.ascii_letters
        payload_lines.append("".join(rng.choice(alphabet) for _ in range(max(0, int(length)))))

    if not payload_lines:
        if problem.sample_inputs:
            inferred = _infer_shape_from_sample(problem.sample_inputs[0])
            for row in inferred:
                if all(re.fullmatch(r"-?\d+", token) for token in row):
                    payload_lines.append(" ".join(str(rng.randint(-10, 10)) for _ in row))
                else:
                    payload_lines.append(" ".join("a" for _ in row))
        else:
            payload_lines.append(str(rng.randint(-10, 10)))

    if multiple_case_count is not None and payload_lines:
        # Repeat a simple payload pattern for each case.
        case_block = payload_lines[:]
        payload_lines = []
        for _ in range(multiple_case_count):
            payload_lines.extend(case_block)

    lines.extend(payload_lines)
    return SampledInput(input_data="\n".join(lines).rstrip() + "\n", metadata={"values": values})
