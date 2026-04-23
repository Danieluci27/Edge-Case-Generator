import unittest

from edge_case_generator.collection.catalog import load_problem_bundles
from edge_case_generator.collection.io import load_collection_config
from edge_case_generator.collection.selection import select_reference_solutions


class CollectionSelectionTests(unittest.TestCase):
    def test_gt_selection_returns_one_python_reference_per_problem(self):
        config = load_collection_config("edge_case_generator/configs/collection_debug.yaml")
        bundles = load_problem_bundles(config)
        refs = select_reference_solutions(bundles, target_language="python")
        self.assertEqual(len(refs), 3)
        self.assertTrue(all(item.language == "python" for item in refs))


if __name__ == "__main__":
    unittest.main()
