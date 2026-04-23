"""Execution and output comparison helpers for the collection pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from edge_case_generator.collection.types import EdgeCaseRecord, ProblemRecord, RejectedCaseRecord
from edge_case_generator.types import ExecutionResult, ExecutionStatus
from edge_case_generator.verifier.executor import CodeExecutor


@dataclass
class OutputComparison:
    """Comparison result for two outputs."""

    comparable: bool
    equal: bool
    normalized_left: str | None
    normalized_right: str | None
    reason: str


def normalize_output(output: str | bytes, mode: str = "normalized") -> str | None:
    """Normalize output conservatively for comparison."""

    if output is None:
        return None
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    stripped = output.strip()
    if stripped == "":
        return ""
    lines = [" ".join(line.strip().split()) for line in stripped.splitlines()]
    if mode == "exact":
        return "\n".join(line.rstrip() for line in stripped.splitlines())
    return "\n".join(lines)


def compare_outputs(left: str | None, right: str | None, mode: str = "normalized") -> OutputComparison:
    """Compare two stdout payloads."""

    normalized_left = normalize_output(left, mode=mode) if left is not None else None
    normalized_right = normalize_output(right, mode=mode) if right is not None else None
    if normalized_left is None or normalized_right is None:
        return OutputComparison(False, False, normalized_left, normalized_right, "missing_output")
    return OutputComparison(True, normalized_left == normalized_right, normalized_left, normalized_right, "ok")


def run_program_pair(
    executor: CodeExecutor,
    *,
    gt_code: str,
    buggy_code: str,
    input_data: str,
    comparison_mode: str = "normalized",
) -> tuple[ExecutionResult, ExecutionResult, OutputComparison]:
    """Execute GT and buggy programs and compare their outputs."""

    gt_result = executor.run(gt_code, input_data, valid_input=True)
    buggy_result = executor.run(buggy_code, input_data, valid_input=True)
    comparison = compare_outputs(gt_result.stdout, buggy_result.stdout, mode=comparison_mode)
    return gt_result, buggy_result, comparison


def accept_or_reject_case(
    *,
    problem: ProblemRecord,
    gt_solution_id: str,
    buggy_id: str,
    bug_origin: str,
    input_data: str,
    gt_result: ExecutionResult,
    buggy_result: ExecutionResult,
    comparison: OutputComparison,
    generator_metadata: dict[str, Any],
) -> tuple[EdgeCaseRecord | None, RejectedCaseRecord | None]:
    """Apply mismatch-only acceptance rules."""

    if gt_result.status != ExecutionStatus.SUCCESS:
        reason = f"gt_{gt_result.status.value}"
    elif buggy_result.status != ExecutionStatus.SUCCESS:
        reason = f"buggy_{buggy_result.status.value}"
    elif not comparison.comparable:
        reason = "non_comparable_output"
    elif comparison.equal:
        reason = "equal_output"
    else:
        reason = ""

    if reason:
        rejected = RejectedCaseRecord(
            problem_id=problem.problem_id,
            input_data=input_data,
            gt_status=gt_result.status.value,
            buggy_status=buggy_result.status.value,
            rejection_reason=reason,
            metadata={
                "gt_stdout": gt_result.stdout,
                "buggy_stdout": buggy_result.stdout,
                "comparison_reason": comparison.reason,
            },
        )
        return None, rejected

    digest = hashlib.sha1(f"{problem.problem_id}|{gt_solution_id}|{buggy_id}|{input_data}".encode("utf-8")).hexdigest()[:16]
    accepted = EdgeCaseRecord(
        problem_id=problem.problem_id,
        edge_case_id=f"{problem.problem_id}__{digest}",
        input_data=input_data,
        gt_output=comparison.normalized_left or "",
        buggy_output=comparison.normalized_right or "",
        output_equal=False,
        gt_solution_id=gt_solution_id,
        buggy_id=buggy_id,
        bug_origin=bug_origin,
        generator_metadata=generator_metadata,
        validation_metadata={
            "gt_runtime_sec": gt_result.runtime_sec,
            "buggy_runtime_sec": buggy_result.runtime_sec,
            "comparison_mode": generator_metadata.get("comparison_mode", "normalized"),
        },
    )
    return accepted, None
