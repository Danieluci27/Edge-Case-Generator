import tempfile
import unittest
from pathlib import Path

from edge_case_generator.buffer.replay_buffer import ReplayBuffer
from edge_case_generator.datasets.candidates import build_candidate
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.rewards.components import build_reward_breakdowns
from edge_case_generator.verifier.decision import EdgeCaseVerifier
from edge_case_generator.verifier.executor import CodeExecutor


class BufferTests(unittest.TestCase):
    def test_replay_buffer_persists_records(self):
        dataset = JSONLProblemDataset.from_jsonl("edge_case_generator/data/demo_dataset.jsonl")
        example = list(dataset)[0]
        verifier = EdgeCaseVerifier(CodeExecutor(timeout_sec=0.5, python_executable="python3"))
        decision = verifier.verify(example, build_candidate("-5", example))
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer_path = Path(temp_dir) / "buffer.jsonl"
            buffer = ReplayBuffer(buffer_path)
            reward = build_reward_breakdowns(
                [decision],
                buffer=buffer,
                config={
                    "reward": {
                        "verify_weight": 5.0,
                        "newfreq_weight": 0.0,
                        "length_efficiency_weight": 0.0,
                        "duplicate_weight": 0.0,
                        "verify": {"mode": "binary", "cap": 1},
                        "newfreq": {"allocation": "inverse_frequency"},
                        "length_efficiency": {
                            "enabled": False,
                            "mode": "inverse_normalized_length",
                            "min_length": 1,
                            "target_length": 8,
                            "max_length": 128,
                            "eps": 1.0e-8,
                        },
                        "duplicate": {
                            "check_group_duplicates": False,
                            "check_replay_duplicates": False,
                            "check_recent_history": False,
                        },
                    },
                    "grpo": {"advantage_normalization": "centered", "eps": 1.0e-8},
                },
            )[0][0]
            buffer.add(decision, reward, iteration=1)

            reloaded = ReplayBuffer(buffer_path)
            self.assertTrue(reloaded.contains(decision.candidate.candidate_hash))
            self.assertTrue(reloaded.contains_case(reward.case_ids[0]))
            self.assertEqual(reloaded.stats()["accepted_count"], 1)


if __name__ == "__main__":
    unittest.main()
