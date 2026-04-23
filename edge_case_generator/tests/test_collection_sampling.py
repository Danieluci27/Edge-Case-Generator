import random
import unittest

from edge_case_generator.collection.catalog import load_problem_bundles
from edge_case_generator.collection.io import load_collection_config
from edge_case_generator.collection.sampling import infer_constraints_from_problem, sample_valid_input


class CollectionSamplingTests(unittest.TestCase):
    def test_sampler_respects_array_shape(self):
        config = load_collection_config("edge_case_generator/configs/collection_debug.yaml")
        problem = next(bundle.problem for bundle in load_problem_bundles(config) if bundle.problem.source == "codecontests")
        parsed = infer_constraints_from_problem(problem)
        sampled = sample_valid_input(problem, parsed, random.Random(5))
        lines = sampled.input_data.strip().splitlines()
        self.assertGreaterEqual(len(lines), 2)
        n = int(lines[0])
        self.assertEqual(len(lines[1].split()), n)


if __name__ == "__main__":
    unittest.main()
