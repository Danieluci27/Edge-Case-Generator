import unittest

from edge_case_generator.collection.mutations import apply_mutation, generate_synthetic_buggy_solutions
from edge_case_generator.collection.types import ReferenceSolutionRecord


class CollectionMutationTests(unittest.TestCase):
    def test_apply_mutation_changes_code(self):
        code = "n = 0\nfor i in range(3):\n    n += i\nprint(n)\n"
        mutated = apply_mutation(code, "off_by_one_range")
        self.assertIsNotNone(mutated)
        self.assertIn("range(3 + 1)", mutated.code)

    def test_generate_synthetic_buggy_solutions(self):
        reference = ReferenceSolutionRecord(
            problem_id="demo",
            solution_id="ref1",
            language="python",
            gt_code="x = int(input())\nprint(x if x > 0 else 0)\n",
            source_label="fixture",
        )
        buggy = generate_synthetic_buggy_solutions([reference], per_reference_limit=2, seed=3)
        self.assertGreaterEqual(len(buggy), 1)
        self.assertTrue(all(item.bug_origin == "synthetic_mutation" for item in buggy))


if __name__ == "__main__":
    unittest.main()
