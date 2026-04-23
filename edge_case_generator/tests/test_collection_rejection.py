import unittest

from edge_case_generator.collection.execution import accept_or_reject_case, compare_outputs
from edge_case_generator.collection.types import ProblemRecord
from edge_case_generator.types import ExecutionResult, ExecutionStatus


class CollectionRejectionTests(unittest.TestCase):
    def test_equal_output_is_rejected(self):
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            stdout="3\n",
            stderr="",
            return_code=0,
            runtime_sec=0.01,
            timed_out=False,
            valid_input=True,
            parsed_output="3",
        )
        problem = ProblemRecord(
            problem_id="demo",
            source="fixture",
            source_problem_id="demo",
            source_url=None,
            title="Demo",
            problem_statement="",
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
            bug_origin="real_submission",
            input_data="1\n",
            gt_result=result,
            buggy_result=result,
            comparison=compare_outputs("3\n", "3\n"),
            generator_metadata={"comparison_mode": "normalized"},
        )
        self.assertIsNone(accepted)
        self.assertEqual(rejected.rejection_reason, "equal_output")


if __name__ == "__main__":
    unittest.main()
