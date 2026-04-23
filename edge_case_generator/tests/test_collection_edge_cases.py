import unittest

from edge_case_generator.collection.io import load_collection_config
from edge_case_generator.collection.pipeline import collect_buggy_programs, collect_problems, generate_supervised_edge_cases, select_gt_solutions


class CollectionEdgeCaseTests(unittest.TestCase):
    def test_pipeline_generates_mismatch_only_cases(self):
        config = load_collection_config("edge_case_generator/configs/collection_debug.yaml")
        problems, _, bundles = collect_problems(config)
        references = select_gt_solutions(config, bundles)
        buggy = collect_buggy_programs(config, bundles, references)
        accepted, rejected, summary = generate_supervised_edge_cases(config, problems, references, buggy)
        self.assertGreaterEqual(len(accepted), 1)
        self.assertTrue(all(item.output_equal is False for item in accepted))
        self.assertTrue(all(item.gt_output != item.buggy_output for item in accepted))
        self.assertIn("rejection_reason_breakdown", summary)


if __name__ == "__main__":
    unittest.main()
