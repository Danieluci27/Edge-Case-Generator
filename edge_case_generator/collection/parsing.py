"""Problem statement parsing and constraint extraction utilities."""

from __future__ import annotations

import re
from typing import Any

from edge_case_generator.collection.types import ArrayConstraint, ConstraintVariable, ParsedConstraints, StringConstraint

SECTION_PATTERNS = {
    "constraints": re.compile(r"(?ims)^constraints?\s*:?\s*(.+?)(?=^\w[\w ]{0,24}:?\s*$|\Z)"),
    "input": re.compile(r"(?ims)^input\s*:?\s*(.+?)(?=^\w[\w ]{0,24}:?\s*$|\Z)"),
    "output": re.compile(r"(?ims)^output\s*:?\s*(.+?)(?=^\w[\w ]{0,24}:?\s*$|\Z)"),
}

SAMPLE_PATTERN = re.compile(
    r"(?ims)^sample input\s*\d*\s*:?\s*(.+?)^\s*sample output\s*\d*\s*:?\s*(.+?)(?=^sample input|\Z)"
)
RANGE_PATTERN = re.compile(r"(?P<low>-?\d+)\s*<=\s*(?P<var>[a-zA-Z_][a-zA-Z0-9_]*)\s*<=\s*(?P<high>-?\d+)")
UPPER_BOUND_PATTERN = re.compile(r"(?P<var>[a-zA-Z_][a-zA-Z0-9_]*)\s*<=\s*(?P<high>-?\d+)")
LOWER_BOUND_PATTERN = re.compile(r"(?P<low>-?\d+)\s*<=\s*(?P<var>[a-zA-Z_][a-zA-Z0-9_]*)")
ARRAY_PATTERN = re.compile(
    r"(?P<length_var>[a-zA-Z_][a-zA-Z0-9_]*)\s+integers?\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\s+where\s+|\s*,\s*)?(?P<low>-?\d+)\s*<=\s*[a-zA-Z_][a-zA-Z0-9_]*\s*<=\s*(?P<high>-?\d+)",
    re.IGNORECASE,
)
STRING_PATTERN = re.compile(
    r"string\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)[^.\n]*length\s+(?:is\s+)?between\s+(?P<low>\d+)\s+and\s+(?P<high>\d+)",
    re.IGNORECASE,
)


def _extract_section(text: str, name: str) -> str:
    match = SECTION_PATTERNS[name].search(text)
    return match.group(1).strip() if match else ""


def _normalize_lines(block: str) -> list[str]:
    lines = [line.strip(" -\t") for line in block.splitlines() if line.strip()]
    return [line for line in lines if line]


def parse_problem_assets(statement: str, samples: list[dict[str, str]] | None = None, metadata: dict[str, Any] | None = None) -> ParsedConstraints:
    """Parse constraints, input/output sections, and samples from a problem statement."""

    del metadata
    constraints_block = _extract_section(statement, "constraints")
    input_block = _extract_section(statement, "input")
    output_block = _extract_section(statement, "output")

    sample_pairs = list(samples or [])
    if not sample_pairs:
        for match in SAMPLE_PATTERN.finditer(statement):
            sample_pairs.append({"input": match.group(1).strip(), "output": match.group(2).strip()})

    raw_constraint_parts = [part for part in [constraints_block, input_block, output_block] if part]
    raw_constraint_text = "\n\n".join(raw_constraint_parts).strip()
    parsed = ParsedConstraints(
        raw_constraint_text=raw_constraint_text,
        input_constraints=_normalize_lines(constraints_block or input_block),
        output_constraints=_normalize_lines(output_block),
    )

    searchable_text = "\n".join([constraints_block, input_block, output_block])
    for match in RANGE_PATTERN.finditer(searchable_text):
        parsed.variables.append(
            ConstraintVariable(
                name=match.group("var"),
                min_value=int(match.group("low")),
                max_value=int(match.group("high")),
            )
        )

    for match in UPPER_BOUND_PATTERN.finditer(searchable_text):
        name = match.group("var")
        if any(item.name == name for item in parsed.variables):
            continue
        parsed.variables.append(ConstraintVariable(name=name, max_value=int(match.group("high"))))

    for match in LOWER_BOUND_PATTERN.finditer(searchable_text):
        name = match.group("var")
        existing = next((item for item in parsed.variables if item.name == name), None)
        if existing is None:
            parsed.variables.append(ConstraintVariable(name=name, min_value=int(match.group("low"))))
        elif existing.min_value is None:
            existing.min_value = int(match.group("low"))

    for match in ARRAY_PATTERN.finditer(searchable_text):
        parsed.arrays.append(
            ArrayConstraint(
                name=match.group("name"),
                length_ref=match.group("length_var"),
                elem_min=int(match.group("low")),
                elem_max=int(match.group("high")),
            )
        )

    if parsed.arrays:
        parsed.variables = [
            item
            for item in parsed.variables
            if not re.fullmatch(r"[a-z](?:_)?i", item.name, flags=re.IGNORECASE)
        ]

    for match in STRING_PATTERN.finditer(searchable_text):
        parsed.strings.append(
            StringConstraint(
                name=match.group("name"),
                length_min=int(match.group("low")),
                length_max=int(match.group("high")),
                alphabet="ascii_lowercase",
            )
        )

    lowered = searchable_text.lower()
    if "multiple test cases" in lowered or "number of test cases" in lowered:
        parsed.multiple_test_cases = True
        if any(item.name == "t" for item in parsed.variables):
            parsed.test_case_count_var = "t"

    return parsed


def infer_samples_from_statement(statement: str) -> list[dict[str, str]]:
    """Extract sample pairs when they were not provided explicitly."""

    return [{"input": match.group(1).strip(), "output": match.group(2).strip()} for match in SAMPLE_PATTERN.finditer(statement)]
