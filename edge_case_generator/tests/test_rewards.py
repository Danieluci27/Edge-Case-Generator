import tempfile
import unittest
from pathlib import Path

from edge_case_generator.buffer.replay_buffer import ReplayBuffer
from edge_case_generator.datasets.candidates import build_candidate
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.rewards.components import (
    aggregate_reward,
    build_reward_breakdowns,
    grpo_advantages,
    is_verified_mismatch,
    new_frequency_allocation,
    verified_mismatch_reward,
)
from edge_case_generator.types import ProblemExample
from edge_case_generator.verifier.decision import EdgeCaseVerifier
from edge_case_generator.verifier.executor import CodeExecutor


def reward_config() -> dict:
    return {
        "reward": {
            "verify_weight": 5.0,
            "newfreq_weight": 2.0,
            "length_efficiency_weight": 0.2,
            "duplicate_weight": 1.5,
            "verify": {"mode": "binary", "cap": 1},
            "newfreq": {"allocation": "inverse_frequency"},
            "length_efficiency": {
                "enabled": True,
                "mode": "inverse_normalized_length",
                "min_length": 1,
                "target_length": 8,
                "max_length": 128,
                "eps": 1.0e-8,
            },
            "duplicate": {
                "check_group_duplicates": True,
                "check_replay_duplicates": True,
                "check_recent_history": True,
            },
        },
        "grpo": {
            "advantage_normalization": "std_normalized",
            "eps": 1.0e-8,
        },
    }


class RewardTests(unittest.TestCase):
    def setUp(self):
        dataset = JSONLProblemDataset.from_jsonl("edge_case_generator/data/demo_dataset.jsonl")
        self.examples = list(dataset)
        self.verifier = EdgeCaseVerifier(CodeExecutor(timeout_sec=0.5, python_executable="python3"))

    def test_verified_mismatch_reward(self):
        decision = self.verifier.verify(self.examples[0], build_candidate("-5", self.examples[0]))
        self.assertTrue(is_verified_mismatch(decision))
        self.assertEqual(verified_mismatch_reward(decision), 1.0)

    def test_repeated_new_case_splits_credit_correctly(self):
        decision = self.verifier.verify(self.examples[0], build_candidate("-5", self.examples[0]))
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = ReplayBuffer(Path(temp_dir) / "buffer.jsonl")
            breakdowns, _ = build_reward_breakdowns([decision, decision], buffer=buffer, config=reward_config())
        self.assertAlmostEqual(breakdowns[0].new_frequency_reward, 0.5)
        self.assertAlmostEqual(breakdowns[1].new_frequency_reward, 0.5)

    def test_old_verified_case_receives_zero_newfreq_reward(self):
        decision = self.verifier.verify(self.examples[0], build_candidate("-5", self.examples[0]))
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = ReplayBuffer(Path(temp_dir) / "buffer.jsonl")
            first_breakdown, _ = build_reward_breakdowns([decision], buffer=buffer, config=reward_config())
            buffer.add(decision, first_breakdown[0], iteration=1)
            second_breakdown, _ = build_reward_breakdowns([decision], buffer=buffer, config=reward_config())
        self.assertEqual(second_breakdown[0].new_frequency_reward, 0.0)

    def test_length_efficiency_rewards_only_verified_cases(self):
        verified = self.verifier.verify(self.examples[0], build_candidate("-5", self.examples[0]))
        invalid = self.verifier.verify(self.examples[1], build_candidate("x", self.examples[1]))
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = ReplayBuffer(Path(temp_dir) / "buffer.jsonl")
            breakdowns, _ = build_reward_breakdowns([verified, invalid], buffer=buffer, config=reward_config())
        self.assertGreater(breakdowns[0].length_efficiency_reward, 0.0)
        self.assertEqual(breakdowns[1].length_efficiency_reward, 0.0)

    def test_duplicate_penalty(self):
        decision = self.verifier.verify(self.examples[0], build_candidate("-5", self.examples[0]))
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = ReplayBuffer(Path(temp_dir) / "buffer.jsonl")
            breakdowns, _ = build_reward_breakdowns([decision, decision], buffer=buffer, config=reward_config())
        self.assertEqual(breakdowns[0].duplicate_penalty, 1.0)
        self.assertEqual(breakdowns[1].duplicate_penalty, 1.0)

    def test_multi_case_output_accumulates_newfreq_per_output(self):
        multi_example = ProblemExample(
            problem_id="multi_abs",
            prompt="Return absolute values.",
            input_code="import sys\nn=int(sys.stdin.read().strip())\nprint(n)\n",
            gt_code="import sys\nn=int(sys.stdin.read().strip())\nprint(abs(n))\n",
            metadata={"input_spec": {"type": "single_int"}, "candidate_separator": "||"},
        )
        decision = self.verifier.verify(multi_example, build_candidate("-5||-7||-5", multi_example))
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = ReplayBuffer(Path(temp_dir) / "buffer.jsonl")
            breakdowns, summary = build_reward_breakdowns([decision], buffer=buffer, config=reward_config())
        self.assertEqual(decision.distinct_verified_case_count, 2)
        self.assertAlmostEqual(breakdowns[0].new_frequency_reward, 2.0)
        self.assertEqual(summary["intra_output_duplicate_count"], 1)

    def test_final_reward_aggregation(self):
        total = aggregate_reward(
            verify_component=1.0,
            newfreq_component=0.5,
            length_component=2.0,
            duplicate_component=1.0,
            reward_config=reward_config()["reward"],
        )
        self.assertAlmostEqual(total, 5.0 + 2.0 * 0.5 + 0.2 * 2.0 - 1.5)

    def test_grpo_advantage_normalization(self):
        centered = grpo_advantages([1.0, 2.0, 3.0], mode="centered", eps=1.0e-8)
        self.assertEqual(centered, [-1.0, 0.0, 1.0])
        normalized = grpo_advantages([1.0, 2.0, 3.0], mode="std_normalized", eps=1.0e-8)
        self.assertAlmostEqual(sum(normalized), 0.0, places=6)

    def test_new_case_frequency_allocation_modes(self):
        self.assertAlmostEqual(new_frequency_allocation(4, "inverse_frequency"), 0.25)
        self.assertAlmostEqual(new_frequency_allocation(4, "inverse_sqrt_frequency"), 0.5)
        self.assertAlmostEqual(new_frequency_allocation(4, "inverse_max_frequency"), 0.25)

    def test_per_sample_rewards_include_expected_flags(self):
        decision = self.verifier.verify(self.examples[0], build_candidate("-5", self.examples[0]))
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = ReplayBuffer(Path(temp_dir) / "buffer.jsonl")
            breakdowns, summary = build_reward_breakdowns([decision], buffer=buffer, config=reward_config())
        self.assertEqual(len(breakdowns), 1)
        self.assertTrue(breakdowns[0].case_ids)
        self.assertTrue(breakdowns[0].globally_new_case_ids)
        self.assertEqual(summary["verified_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
