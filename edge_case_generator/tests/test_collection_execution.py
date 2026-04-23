import unittest

from edge_case_generator.collection.execution import accept_or_reject_case, compare_outputs, run_program_pair
from edge_case_generator.collection.types import ProblemRecord
from edge_case_generator.types import ExecutionStatus
from edge_case_generator.verifier.executor import CodeExecutor


class CollectionExecutionTests(unittest.TestCase):
    def test_output_comparison_normalizes_whitespace(self):
        comparison = compare_outputs("1  2\n", "1 2\n")
        self.assertTrue(comparison.comparable)
        self.assertTrue(comparison.equal)

    def test_run_pair_and_accept_only_mismatch(self):
        executor = CodeExecutor(timeout_sec=2.0, python_executable="python3")
        gt = "x = int(input())\nprint(x + 1)\n"
        buggy = "x = int(input())\nprint(x)\n"
        gt_result, buggy_result, comparison = run_program_pair(executor, gt_code=gt, buggy_code=buggy, input_data="4\n")
        problem = ProblemRecord(
            problem_id="demo",
            source="fixture",
            source_problem_id="demo",
            source_url=None,
            title="Demo",
            problem_statement="Input\nOne integer.\nOutput\nOne integer.",
            input_constraints=[],
            output_constraints=[],
            raw_constraint_text="",
            sample_inputs=[],
            sample_outputs=[],
            public_tests=[],
            language="python",
        )
        accepted, rejected = accept_or_reject_case(
            problem=problem,
            gt_solution_id="gt",
            buggy_id="buggy",
            bug_origin="synthetic_mutation",
            input_data="4\n",
            gt_result=gt_result,
            buggy_result=buggy_result,
            comparison=comparison,
            generator_metadata={"comparison_mode": "normalized"},
        )
        self.assertIsNotNone(accepted)
        self.assertIsNone(rejected)
        self.assertEqual(gt_result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(buggy_result.status, ExecutionStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()
