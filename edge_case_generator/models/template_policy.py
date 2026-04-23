"""A lightweight trainable baseline policy over templated candidates."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from edge_case_generator.datasets.candidates import build_candidate
from edge_case_generator.models.base import BaseGeneratorPolicy
from edge_case_generator.types import ProblemExample, SampledCandidate


@dataclass
class CandidateAction:
    """Discrete templated candidate action."""

    action_id: str
    text: str


def _softmax(logits: list[float], temperature: float) -> list[float]:
    scaled = [value / max(temperature, 1e-6) for value in logits]
    max_logit = max(scaled)
    exps = [math.exp(value - max_logit) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def _top_p_indices(probabilities: list[float], top_p: float) -> list[int]:
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    cumulative = 0.0
    kept: list[int] = []
    for index, probability in ranked:
        kept.append(index)
        cumulative += probability
        if cumulative >= top_p:
            break
    return kept


class TemplateGeneratorPolicy(BaseGeneratorPolicy):
    """Problem-specific categorical policy over templated candidate inputs."""

    def __init__(self, problem_actions: dict[str, list[CandidateAction]], seed: int = 0) -> None:
        self.problem_actions = problem_actions
        self.logits = {
            problem_id: {action.action_id: 0.0 for action in actions}
            for problem_id, actions in problem_actions.items()
        }
        self.rng = random.Random(seed)
        self.training = True

    @staticmethod
    def build_actions(example: ProblemExample, max_len: int) -> list[CandidateAction]:
        """Construct templated candidates from metadata and generic heuristics."""

        hints = example.metadata.get("generator_hints", {})
        candidates: list[str] = list(hints.get("candidate_pool", []))

        input_spec = (example.metadata.get("input_spec") or {}).get("type", "raw_text")
        if input_spec == "single_int":
            numbers = hints.get("integers", [-10, -1, 0, 1, 2, 10, 999999999])
            candidates.extend(str(value) for value in numbers)
            candidates.extend(["", " ", "000", "-0001"])
        elif input_spec == "int_list":
            sequences = hints.get(
                "sequences",
                [
                    "",
                    "0",
                    "1 1 1",
                    "-1 -2 -3",
                    "0 0 0 0",
                    "1000000 -1000000 0",
                    "5 4 3 2 1",
                ],
            )
            candidates.extend(sequences)
        elif input_spec == "int_with_count":
            sequences = hints.get(
                "sequences",
                [
                    "0\n",
                    "1\n0",
                    "2\n1 1",
                    "3\n-1 0 1",
                    "5\n1 2 3 4 5",
                    "4\n0 0 0 0",
                    "1\n999999999",
                ],
            )
            candidates.extend(sequences)

        deduped: list[CandidateAction] = []
        seen: set[str] = set()
        for index, candidate in enumerate(candidates):
            trimmed = candidate[:max_len]
            if trimmed in seen:
                continue
            seen.add(trimmed)
            deduped.append(CandidateAction(action_id=f"{example.problem_id}_action_{index}", text=trimmed))
        if not deduped:
            deduped.append(CandidateAction(action_id=f"{example.problem_id}_default", text=""))
        return deduped

    @classmethod
    def from_examples(cls, examples: list[ProblemExample], max_len: int, seed: int = 0) -> "TemplateGeneratorPolicy":
        """Build a template policy from dataset examples."""

        problem_actions = {example.problem_id: cls.build_actions(example, max_len) for example in examples}
        return cls(problem_actions=problem_actions, seed=seed)

    def _action_probabilities(self, problem_id: str, temperature: float, top_p: float) -> tuple[list[CandidateAction], list[float]]:
        actions = self.problem_actions[problem_id]
        logits = [self.logits[problem_id][action.action_id] for action in actions]
        probabilities = _softmax(logits, temperature=temperature)
        kept_indices = _top_p_indices(probabilities, top_p=top_p)
        mask_total = sum(probabilities[index] for index in kept_indices)
        filtered = [probabilities[index] / mask_total for index in kept_indices]
        filtered_actions = [actions[index] for index in kept_indices]
        return filtered_actions, filtered

    def sample_candidates(
        self,
        example: ProblemExample,
        num_samples: int,
        temperature: float,
        top_p: float,
        max_len: int,
    ) -> list[SampledCandidate]:
        actions, probabilities = self._action_probabilities(example.problem_id, temperature=temperature, top_p=top_p)
        samples: list[SampledCandidate] = []
        for group_index in range(num_samples):
            chosen = self.rng.choices(actions, weights=probabilities, k=1)[0]
            probability = probabilities[actions.index(chosen)]
            candidate = build_candidate(chosen.text[:max_len], example)
            samples.append(
                SampledCandidate(
                    problem_id=example.problem_id,
                    action_id=chosen.action_id,
                    candidate=candidate,
                    logprob=math.log(max(probability, 1e-12)),
                    probability=probability,
                    group_index=group_index,
                )
            )
        return samples

    def logprob(self, problem_id: str, action_id: str) -> float:
        actions, probabilities = self._action_probabilities(problem_id, temperature=1.0, top_p=1.0)
        for action, probability in zip(actions, probabilities):
            if action.action_id == action_id:
                return math.log(max(probability, 1e-12))
        raise KeyError(f"Unknown action_id {action_id!r} for problem {problem_id!r}")

    def update(self, problem_id: str, action_rewards: list[tuple[str, float]], learning_rate: float) -> None:
        actions = self.problem_actions[problem_id]
        logits = [self.logits[problem_id][action.action_id] for action in actions]
        probabilities = _softmax(logits, temperature=1.0)
        index_map = {action.action_id: index for index, action in enumerate(actions)}

        gradient = [0.0 for _ in actions]
        for action_id, reward in action_rewards:
            chosen_index = index_map[action_id]
            for index in range(len(actions)):
                indicator = 1.0 if index == chosen_index else 0.0
                gradient[index] += reward * (indicator - probabilities[index])

        for index, action in enumerate(actions):
            self.logits[problem_id][action.action_id] += learning_rate * gradient[index] / max(len(action_rewards), 1)

    def save(self, path: str) -> None:
        payload = {
            "problem_actions": {
                problem_id: [asdict(action) for action in actions]
                for problem_id, actions in self.problem_actions.items()
            },
            "logits": self.logits,
            "training": self.training,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TemplateGeneratorPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        problem_actions = {
            problem_id: [CandidateAction(**action) for action in actions]
            for problem_id, actions in payload["problem_actions"].items()
        }
        model = cls(problem_actions=problem_actions)
        model.logits = payload["logits"]
        model.training = payload.get("training", True)
        return model

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False
