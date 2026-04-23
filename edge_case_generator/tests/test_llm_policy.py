import tempfile
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.models.llm_policy import PretrainedLLMGeneratorPolicy


class LLMPolicyTests(unittest.TestCase):
    def test_pretrained_llm_policy_supports_rollout_updates(self):
        dataset = JSONLProblemDataset.from_jsonl("edge_case_generator/data/demo_dataset.jsonl")
        
        
        policy = PretrainedLLMGeneratorPolicy.from_examples(
            list(dataset),
            backend_name="huggingface",
            seed=5,
            backend_config={
                "pretrained_model_name_or_path": "Intel/tiny-random-distilgpt2",
                "use_safetensors": True,
                "adapter": {"enabled": False},
            },
        )

        self.assertTrue(policy.supports_rollout_updates())
        self.assertEqual(policy.logprob(list(dataset)[0].problem_id, "missing"), policy.logprob(list(dataset)[0].problem_id, "missing"))

    def test_pretrained_llm_policy_save_and_load(self):
        dataset = JSONLProblemDataset.from_jsonl("edge_case_generator/data/demo_dataset.jsonl")
        example = list(dataset)[0]
        policy = PretrainedLLMGeneratorPolicy.from_examples(
            list(dataset),
            backend_name="huggingface",
            seed=5,
            backend_config={
                "pretrained_model_name_or_path": "Intel/tiny-random-distilgpt2",
                "use_safetensors": True,
                "adapter": {"enabled": False},
            },
        )
        policy.update(example.problem_id, [("demo_action", 1.0)], learning_rate=0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "policy.json"
            policy.save(str(checkpoint))
            restored = PretrainedLLMGeneratorPolicy.load(str(checkpoint))
            self.assertTrue(restored.supports_rollout_updates())
            self.assertIn("demo_action", restored.adapter_biases[example.problem_id])


if __name__ == "__main__":
    unittest.main()
