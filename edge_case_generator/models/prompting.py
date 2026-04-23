"""Prompt construction for LLM-backed edge-case generation."""

from __future__ import annotations

from edge_case_generator.types import ProblemExample


def build_generation_prompt(example: ProblemExample, include_gt_code: bool = False) -> str:
    """Create the conditioning prompt for an edge-case generator LLM."""

    sections = [
        "You generate compact edge-case inputs for a code task.",
        "Return only the candidate input string with no explanation.",
        "",
        f"Problem ID: {example.problem_id}",
        "Task:",
        example.prompt.strip(),
        "",
        "Program under test:",
        example.input_code.strip(),
    ]

    if example.public_tests:
        sections.extend(
            [
                "",
                "Known public tests:",
                "\n".join(
                    f"- input={test.get('input', '')!r} output={test.get('output', '')!r}"
                    for test in example.public_tests
                ),
            ]
        )

    if include_gt_code:
        sections.extend(
            [
                "",
                "Reference implementation:",
                example.gt_code.strip(),
            ]
        )

    sections.extend(
        [
            "",
            "Goal: propose a valid input likely to expose a behavioral difference.",
            "Prefer compact, high-signal edge cases.",
        ]
    )
    return "\n".join(sections).strip()

