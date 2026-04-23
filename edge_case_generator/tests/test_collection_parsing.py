import unittest

from edge_case_generator.collection.parsing import parse_problem_assets


STATEMENT = """Solve the task.

Input
The first line contains n.
The second line contains n integers arr.

Constraints
1 <= n <= 5
n integers arr where -9 <= ai <= 9
string s length between 1 and 4

Output
Print one integer.
"""


class CollectionParsingTests(unittest.TestCase):
    def test_constraint_parsing_extracts_structured_fields(self):
        parsed = parse_problem_assets(STATEMENT)
        names = {item.name for item in parsed.variables}
        self.assertIn("n", names)
        self.assertEqual(parsed.variables[0].min_value, 1)
        self.assertEqual(parsed.arrays[0].length_ref, "n")
        self.assertEqual(parsed.strings[0].name, "s")
        self.assertTrue(parsed.raw_constraint_text)


if __name__ == "__main__":
    unittest.main()
