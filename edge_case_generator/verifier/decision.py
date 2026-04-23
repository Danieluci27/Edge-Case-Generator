"""Verification and failure categorization logic."""

from __future__ import annotations

from collections import Counter

from edge_case_generator.datasets.candidates import build_candidate
from edge_case_generator.types import (
    CandidateInput,
    ExecutionResult,
    ExecutionStatus,
    FailureCategory,
    ProblemExample,
    VerificationDecision,
)
from edge_case_generator.verifier.executor import CodeExecutor


def categorize_failure(candidate: CandidateInput, gt_result: ExecutionResult, input_result: ExecutionResult) -> FailureCategory | None:
    """Map raw execution outcomes to a stable failure category."""

    if not candidate.valid or gt_result.status == ExecutionStatus.INVALID_INPUT or input_result.status == ExecutionStatus.INVALID_INPUT:
        return FailureCategory.INVALID_FORMAT
    if input_result.status == ExecutionStatus.TIMEOUT:
        return FailureCategory.TIMEOUT
    if input_result.status == ExecutionStatus.ASSERTION_FAILURE:
        return FailureCategory.ASSERTION_FAILURE
    if input_result.status == ExecutionStatus.CRASH:
        return FailureCategory.CRASH
    if input_result.status == ExecutionStatus.RUNTIME_ERROR:
        return FailureCategory.RUNTIME_ERROR
    if gt_result.succeeded and input_result.succeeded and gt_result.parsed_output != input_result.parsed_output:
        return FailureCategory.WRONG_ANSWER
    if not input_result.succeeded:
        return FailureCategory.OTHER
    return None


def split_candidate_output(example: ProblemExample, candidate: CandidateInput) -> list[CandidateInput]:
    """Split one generated output into one or more candidate test cases."""

    metadata = example.metadata or {}
    separator = metadata.get("candidate_separator")
    if not separator:
        return [candidate]

    parts = [part.strip() for part in candidate.canonical_text.split(separator)]
    subcases = [build_candidate(part, example) for part in parts if part]
    return subcases or [candidate]


def is_semantic_mismatch(candidate: CandidateInput, gt_result: ExecutionResult, input_result: ExecutionResult) -> bool:
    """Return whether this candidate is a verified semantic mismatch."""

    return bool(
        candidate.valid
        and gt_result.succeeded
        and input_result.succeeded
        and gt_result.parsed_output is not None
        and input_result.parsed_output is not None
        and gt_result.parsed_output != input_result.parsed_output
    )


def canonical_mismatch_case_id(problem_id: str, candidate: CandidateInput, gt_result: ExecutionResult, input_result: ExecutionResult) -> str | None:
    """Build a stable canonical mismatch case identity."""

    if not is_semantic_mismatch(candidate, gt_result, input_result):
        return None
    from edge_case_generator.rewards.components import canonical_case_hash_for_payload

    return canonical_case_hash_for_payload(
        problem_id=problem_id,
        candidate_text=candidate.canonical_text,
        gt_output=gt_result.parsed_output or "",
        input_output=input_result.parsed_output or "",
    )


class EdgeCaseVerifier:
    """Verify candidate edge cases by comparing target and reference behavior."""

    def __init__(self, executor: CodeExecutor) -> None:
        self.executor = executor

    def verify(self, example: ProblemExample, candidate: CandidateInput) -> VerificationDecision:
        """Run both programs and decide whether the candidate is accepted."""

        subcases = split_candidate_output(example, candidate)
        first_candidate = subcases[0]
        first_gt_result = self.executor.run(example.gt_code, first_candidate.canonical_text, valid_input=first_candidate.valid)
        first_input_result = self.executor.run(example.input_code, first_candidate.canonical_text, valid_input=first_candidate.valid)
        verified_case_ids: list[str] = []
        gt_output = first_gt_result.parsed_output
        input_output = first_input_result.parsed_output
        last_failure_category = categorize_failure(first_candidate, first_gt_result, first_input_result)
        first_valid_failure = last_failure_category
        any_semantic_mismatch = False

        for subcase in subcases:
            gt_result = self.executor.run(example.gt_code, subcase.canonical_text, valid_input=subcase.valid)
            input_result = self.executor.run(example.input_code, subcase.canonical_text, valid_input=subcase.valid)
            failure_category = categorize_failure(subcase, gt_result, input_result)
            if first_valid_failure is None and failure_category is not None:
                first_valid_failure = failure_category
            if is_semantic_mismatch(subcase, gt_result, input_result):
                any_semantic_mismatch = True
                gt_output = gt_result.parsed_output
                input_output = input_result.parsed_output
                case_id = canonical_mismatch_case_id(example.problem_id, subcase, gt_result, input_result)
                if case_id:
                    verified_case_ids.append(case_id)
            last_failure_category = failure_category

        intra_output_duplicate_count = sum(count - 1 for count in Counter(verified_case_ids).values() if count > 1)
        distinct_verified_case_ids = sorted(set(verified_case_ids))

        if len(subcases) == 1 and not candidate.valid:
            return VerificationDecision(
                problem_id=example.problem_id,
                candidate=candidate,
                accepted=False,
                failure_category=FailureCategory.INVALID_FORMAT,
                reason=candidate.validation_error or "Candidate failed validation",
                gt_result=first_gt_result,
                input_result=first_input_result,
                gt_output=gt_output,
                input_output=input_output,
                verified_case_ids=[],
                distinct_verified_case_count=0,
                intra_output_duplicate_count=0,
            )

        if len(subcases) == 1 and not first_gt_result.succeeded:
            return VerificationDecision(
                problem_id=example.problem_id,
                candidate=candidate,
                accepted=False,
                failure_category=None,
                reason="Reference implementation failed",
                gt_result=first_gt_result,
                input_result=first_input_result,
                gt_output=gt_output,
                input_output=input_output,
                verified_case_ids=[],
                distinct_verified_case_count=0,
                intra_output_duplicate_count=0,
            )

        category = FailureCategory.WRONG_ANSWER if any_semantic_mismatch else first_valid_failure
        accepted = any_semantic_mismatch
        reason = "Candidate is not a verified semantic mismatch"

        if any_semantic_mismatch:
            reason = "Input program output differs semantically from reference output"

        return VerificationDecision(
            problem_id=example.problem_id,
            candidate=candidate,
            accepted=accepted,
            failure_category=category,
            reason=reason,
            gt_result=first_gt_result,
            input_result=first_input_result,
            gt_output=gt_output,
            input_output=input_output,
            verified_case_ids=distinct_verified_case_ids,
            distinct_verified_case_count=len(distinct_verified_case_ids),
            intra_output_duplicate_count=intra_output_duplicate_count,
        )
