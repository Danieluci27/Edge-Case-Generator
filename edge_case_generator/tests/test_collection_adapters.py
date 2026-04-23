import unittest
from unittest.mock import patch

from edge_case_generator.collection.catalog import load_problem_bundles
from edge_case_generator.collection.adapters.codecontests import CodeContestsAdapter
from edge_case_generator.collection.io import load_collection_config


class CollectionAdapterTests(unittest.TestCase):
    def test_adapters_normalize_fixture_sources(self):
        config = load_collection_config("edge_case_generator/configs/collection_debug.yaml")
        bundles = load_problem_bundles(config)
        self.assertEqual(len(bundles), 3)
        sources = {bundle.problem.source for bundle in bundles}
        self.assertEqual(sources, {"codenet", "codecontests", "apps"})
        for bundle in bundles:
            self.assertTrue(bundle.problem.problem_statement)
            self.assertGreaterEqual(len(bundle.accepted_solutions), 1)

    @patch.object(CodeContestsAdapter, "_load_dataset_api")
    def test_codecontests_adapter_supports_huggingface_dataset_config(self, mock_load_dataset_api):
        mock_load_dataset = lambda *args, **kwargs: [
            {
                "name": "HF Sum",
                "description": "Input\nOne integer n.\n\nConstraints\n0 <= n <= 10\n\nOutput\nPrint n + 1.\n",
                "public_tests": {"input": ["2\n"], "output": ["3\n"]},
                "private_tests": {"input": ["4\n"], "output": ["5\n"]},
                "solutions": {"language": ["python"], "solution": ["n = int(input())\nprint(n + 1)\n"]},
                "incorrect_solutions": {"language": ["python"], "solution": ["n = int(input())\nprint(n)\n"]},
                "cf_contest_id": 1234,
                "cf_index": "A",
                "difficulty": 1,
            }
        ]
        mock_load_dataset_api.return_value = mock_load_dataset
        adapter = CodeContestsAdapter(
            path=None,
            target_language="python",
            source_config={"hf_dataset": "deepmind/code_contests", "hf_split": "train"},
        )
        bundles = adapter.load()
        self.assertEqual(len(bundles), 1)
        bundle = bundles[0]
        self.assertEqual(bundle.problem.source, "codecontests")
        self.assertEqual(bundle.problem.title, "HF Sum")
        self.assertEqual(bundle.problem.sample_inputs, ["2\n"])
        self.assertEqual(len(bundle.accepted_solutions), 1)
        self.assertEqual(bundle.accepted_solutions[0].language, "python")
        self.assertEqual(len(bundle.incorrect_solutions), 1)
        mock_load_dataset_api.assert_called_once()


if __name__ == "__main__":
    unittest.main()
