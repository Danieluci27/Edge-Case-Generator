"""Abstract generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from edge_case_generator.types import PolicyRolloutSample, ProblemExample, SampledCandidate


class BaseGeneratorPolicy(ABC):
    """Abstract interface for candidate generators."""

    @abstractmethod
    def sample_candidates(
        self,
        example: ProblemExample,
        num_samples: int,
        temperature: float,
        top_p: float,
        max_len: int,
    ) -> list[SampledCandidate]:
        """Sample candidates for a problem."""

    @abstractmethod
    def logprob(self, problem_id: str, action_id: str) -> float:
        """Return the log-probability of an action."""

    @abstractmethod
    def update(self, problem_id: str, action_rewards: list[tuple[str, float]], learning_rate: float) -> None:
        """Apply a policy-gradient style update."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model state."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseGeneratorPolicy":
        """Load model state."""

    @abstractmethod
    def train(self) -> None:
        """Switch to training mode."""

    @abstractmethod
    def eval(self) -> None:
        """Switch to eval mode."""

    def supports_rollout_updates(self) -> bool:
        """Whether this policy can consume full rollout samples directly."""

        return False

    def update_from_rollouts(self, rollouts: list[PolicyRolloutSample], learning_rate: float) -> dict[str, float] | None:
        """Optional direct policy update hook for gradient-based backends."""

        return None
