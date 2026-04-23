"""End-to-end RL trainer."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from edge_case_generator.buffer.replay_buffer import ReplayBuffer
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.models.base import BaseGeneratorPolicy
from edge_case_generator.rewards.components import build_reward_breakdowns
from edge_case_generator.types import PolicyRolloutSample, RewardBreakdown, VerificationDecision
from edge_case_generator.utils.io import write_json
from edge_case_generator.utils.logging_utils import MetricsLogger
from edge_case_generator.verifier.decision import EdgeCaseVerifier


class RLTrainer:
    """Train the edge-case generator using REINFORCE or GRPO-style advantages."""

    def __init__(
        self,
        dataset: JSONLProblemDataset,
        generator: BaseGeneratorPolicy,
        verifier: EdgeCaseVerifier,
        replay_buffer: ReplayBuffer,
        metrics_logger: MetricsLogger,
        logger,
        config: dict[str, Any],
    ) -> None:
        self.dataset = dataset
        self.generator = generator
        self.verifier = verifier
        self.replay_buffer = replay_buffer
        self.metrics_logger = metrics_logger
        self.logger = logger
        self.config = config

    def train(self) -> dict[str, Any]:
        """Run the configured training loop."""

        rl_config = self.config["rl"]
        sampling = self.config["sampling"]
        reward_config = self.config.get("reward", self.config.get("rewards", {}))
        output_dir = Path(self.config["output"]["dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        iteration_summaries: list[dict[str, Any]] = []

        baseline = 0.0
        baseline_momentum = rl_config.get("baseline_momentum", 0.9)
        use_grpo = rl_config.get("algorithm", "reinforce").lower() == "grpo"
        self.generator.train()

        for iteration in range(1, rl_config["num_iterations"] + 1):
            per_problem_acceptance: dict[str, tuple[int, int]] = {}
            all_decisions: list[VerificationDecision] = []
            all_rewards: list[RewardBreakdown] = []
            rollout_updates: list[PolicyRolloutSample] = []
            action_updates: dict[str, list[tuple[str, float]]] = defaultdict(list)

            for example in self.dataset:
                samples = self.generator.sample_candidates(
                    example,
                    num_samples=sampling["group_size"],
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    max_len=sampling["max_len"],
                )
                decisions = [self.verifier.verify(example, sample.candidate) for sample in samples]
                breakdowns, _ = build_reward_breakdowns(
                    decisions,
                    buffer=self.replay_buffer,
                    config=self.config,
                )
                accepted_count = sum(1 for decision in decisions if decision.accepted)
                per_problem_acceptance[example.problem_id] = (accepted_count, len(decisions))

                for sample, decision, reward in zip(samples, decisions, breakdowns):
                    if decision.accepted:
                        self.replay_buffer.add(decision, reward, iteration=iteration)

                    reward_signal = reward.normalized_advantage if use_grpo else reward.total_reward - baseline
                    if self.generator.supports_rollout_updates():
                        rollout_updates.append(
                            PolicyRolloutSample(
                                problem=example,
                                sample=sample,
                                reward=reward,
                                reward_signal=reward_signal,
                            )
                        )
                    else:
                        action_updates[example.problem_id].append((sample.action_id, reward_signal))
                    all_decisions.append(decision)
                    all_rewards.append(reward)

                if not use_grpo and breakdowns:
                    reward_mean = mean(reward.total_reward for reward in breakdowns)
                    baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * reward_mean

            optimizer_metrics: dict[str, float] = {}
            if self.generator.supports_rollout_updates():
                optimizer_metrics = self.generator.update_from_rollouts(
                    rollout_updates,
                    learning_rate=rl_config["learning_rate"],
                ) or {}
            else:
                for problem_id, updates in action_updates.items():
                    self.generator.update(problem_id, updates, learning_rate=rl_config["learning_rate"])

            summary = self._summarize_iteration(
                iteration=iteration,
                decisions=all_decisions,
                rewards=all_rewards,
                per_problem_acceptance=per_problem_acceptance,
                group_size=sampling["group_size"],
                reward_config=reward_config,
            )
            summary.update(optimizer_metrics)
            iteration_summaries.append(summary)
            self.metrics_logger.log(summary)
            self.logger.info(
                "Iteration %s | accepted=%s unique=%s avg_reward=%.3f",
                iteration,
                summary["accepted_count"],
                summary["unique_accepted_count"],
                summary["avg_total_reward"],
            )

            if iteration % self.config["output"]["checkpoint_every"] == 0:
                self.generator.save(str(output_dir / f"checkpoints/generator_iter_{iteration}.json"))

        final_summary = {
            "model": self.config.get("model", {"type": "unknown"}),
            "iterations": iteration_summaries,
            "buffer": self.replay_buffer.stats(),
        }
        write_json(output_dir / "training_summary.json", final_summary)
        self.generator.save(str(output_dir / "checkpoints/generator_final.json"))
        return final_summary

    def _summarize_iteration(
        self,
        iteration: int,
        decisions: list[VerificationDecision],
        rewards: list[RewardBreakdown],
        per_problem_acceptance: dict[str, tuple[int, int]],
        group_size: int,
        reward_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Aggregate metrics for logging and reports."""

        accepted = [decision for decision in decisions if decision.accepted]
        failure_counts = Counter(
            decision.failure_category.value
            for decision in accepted
            if decision.failure_category is not None
        )
        candidate_hashes = [decision.candidate.candidate_hash for decision in accepted]
        exact_counts = Counter(decision.candidate.candidate_hash for decision in decisions)
        lengths = [len(decision.candidate.canonical_text) for decision in accepted]
        reward_values = [reward.total_reward for reward in rewards]
        normalized_advantages = [reward.normalized_advantage for reward in rewards]
        verified_breakdowns = [reward for reward in rewards if reward.verification_reward > 0]
        case_frequency = Counter(case_id for reward in verified_breakdowns for case_id in reward.case_ids)

        summary = {
            "iteration": iteration,
            "group_size": group_size,
            "candidate_count": len(decisions),
            "candidate_validity_rate": sum(1 for decision in decisions if decision.candidate.valid) / max(len(decisions), 1),
            "verification_success_rate": len(accepted) / max(len(decisions), 1),
            "accepted_count": len(accepted),
            "unique_accepted_count": len(set(candidate_hashes)),
            "duplicate_rate": sum(1 for count in exact_counts.values() if count > 1) / max(len(exact_counts), 1),
            "avg_total_reward": mean(reward_values) if reward_values else 0.0,
            "reward_std_proxy": max(reward_values) - min(reward_values) if reward_values else 0.0,
            "avg_verification_reward": mean(reward.verification_reward for reward in rewards) if rewards else 0.0,
            "avg_new_frequency_reward": mean(reward.new_frequency_reward for reward in rewards) if rewards else 0.0,
            "avg_length_efficiency_reward": mean(reward.length_efficiency_reward for reward in rewards) if rewards else 0.0,
            "avg_duplicate_penalty": mean(reward.duplicate_penalty for reward in rewards) if rewards else 0.0,
            "avg_relative_advantage": mean(normalized_advantages) if normalized_advantages else 0.0,
            "std_relative_advantage_proxy": max(normalized_advantages) - min(normalized_advantages) if normalized_advantages else 0.0,
            "timeout_rate": sum(1 for decision in decisions if decision.input_result.timed_out) / max(len(decisions), 1),
            "crash_rate": sum(1 for decision in decisions if decision.input_result.status.value == "crash") / max(len(decisions), 1),
            "group_verified_mismatch_count_mean": mean(count for count, _ in per_problem_acceptance.values()) if per_problem_acceptance else 0.0,
            "group_verified_mismatch_ratio_mean": mean(count / total for count, total in per_problem_acceptance.values()) if per_problem_acceptance else 0.0,
            "group_distinct_new_verified_cases": len({case_id for reward in rewards for case_id in reward.globally_new_case_ids}),
            "group_avg_within_frequency": mean(case_frequency.values()) if case_frequency else 0.0,
            "group_exact_duplicate_count": sum(count - 1 for count in exact_counts.values() if count > 1),
            "globally_new_case_count": len({case_id for reward in rewards for case_id in reward.globally_new_case_ids}),
            "accepted_length_mean": mean(lengths) if lengths else 0.0,
            "accepted_length_max": max(lengths) if lengths else 0,
            "failure_mode_counts": dict(failure_counts),
            "per_problem_acceptance_rate": {
                problem_id: accepted_count / max(total_count, 1)
                for problem_id, (accepted_count, total_count) in per_problem_acceptance.items()
            },
            "reward_verify_weight": reward_config.get("verify_weight"),
            "reward_newfreq_weight": reward_config.get("newfreq_weight"),
            "reward_length_efficiency_weight": reward_config.get("length_efficiency_weight"),
            "reward_duplicate_weight": reward_config.get("duplicate_weight"),
        }
        return summary
