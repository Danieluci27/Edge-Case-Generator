"""Shared data types for the edge-case generator pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    MALFORMED_RESULT = "malformed_result"
    CRASH = "crash"
    ASSERTION_FAILURE = "assertion_failure"
    OTHER = "other"


class FailureCategory(str, Enum):
    WRONG_ANSWER = "wrong_answer"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    CRASH = "crash"
    INVALID_FORMAT = "invalid_format"
    ASSERTION_FAILURE = "assertion_failure"
    OTHER = "other"


@dataclass
class ProblemExample:
    """Single dataset example describing a code task."""

    problem_id: str
    prompt: str
    input_code: str
    gt_code: str
    starter_code: str | None = None
    public_tests: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateInput:
    """Canonical representation for a candidate edge-case input."""

    raw_text: str
    canonical_text: str
    candidate_hash: str
    valid: bool = True
    validation_error: str | None = None
    structured_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionResult:
    """Result of running a candidate against a program."""

    status: ExecutionStatus
    stdout: str
    stderr: str
    return_code: int | None
    runtime_sec: float
    timed_out: bool
    valid_input: bool
    parsed_output: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass
class VerificationDecision:
    """Decision about whether a candidate is a verified edge case."""

    problem_id: str
    candidate: CandidateInput
    accepted: bool
    failure_category: FailureCategory | None
    reason: str
    gt_result: ExecutionResult
    input_result: ExecutionResult
    gt_output: str | None = None
    input_output: str | None = None
    verified_case_ids: list[str] = field(default_factory=list)
    distinct_verified_case_count: int = 0
    intra_output_duplicate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_category"] = self.failure_category.value if self.failure_category else None
        payload["gt_result"]["status"] = self.gt_result.status.value
        payload["input_result"]["status"] = self.input_result.status.value
        return payload


@dataclass
class SampledCandidate:
    """Policy sample output."""

    problem_id: str
    action_id: str
    candidate: CandidateInput
    logprob: float
    probability: float
    group_index: int
    conditioning_prompt: str | None = None


@dataclass
class PolicyRolloutSample:
    """Single rollout sample used for policy updates."""

    problem: ProblemExample
    sample: SampledCandidate
    reward: "RewardBreakdown"
    reward_signal: float


@dataclass
class RewardBreakdown:
    """Per-sample reward decomposition."""

    total_reward: float
    verification_reward: float = 0.0
    new_frequency_reward: float = 0.0
    length_efficiency_reward: float = 0.0
    duplicate_penalty: float = 0.0
    raw_advantage: float = 0.0
    normalized_advantage: float = 0.0
    case_ids: list[str] = field(default_factory=list)
    globally_new_case_ids: list[str] = field(default_factory=list)
    within_group_frequency: dict[str, int] = field(default_factory=dict)
    is_duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayRecord:
    """Persistent accepted edge-case record."""

    problem_id: str
    candidate_input: str
    canonical_input: str
    candidate_hash: str
    case_ids: list[str]
    reward: float
    reward_components: dict[str, float]
    verifier_outcome: dict[str, Any]
    failure_category: str | None
    iteration: int
    timestamp: str
    gt_output: str | None = None
    input_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
