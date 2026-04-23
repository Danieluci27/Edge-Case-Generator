import unittest

from edge_case_generator.config import load_config
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.models.factory import build_generator
from edge_case_generator.models.llm_policy import PretrainedLLMGeneratorPolicy


class HuggingFaceBackendConfigTests(unittest.TestCase):
    def test_factory_builds_huggingface_policy_without_eager_imports(self):
        config = load_config("edge_case_generator/configs/hf_lora_debug.yaml")
        dataset = JSONLProblemDataset.from_jsonl(config["dataset"]["path"])
        train_dataset = dataset.shuffled(config["seed"])
        generator = build_generator(train_dataset, config)

        self.assertIsInstance(generator, PretrainedLLMGeneratorPolicy)
        self.assertTrue(generator.supports_rollout_updates())


if __name__ == "__main__":
    unittest.main()
