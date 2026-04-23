import unittest

from edge_case_generator.datasets.candidates import build_candidate
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.types import FailureCategory
from edge_case_generator.verifier.decision import EdgeCaseVerifier
from edge_case_generator.verifier.executor import CodeExecutor


class VerifierTests(unittest.TestCase):
    def test_verifier_accepts_wrong_answer_case(self):
        dataset = JSONLProblemDataset.from_jsonl("edge_case_generator/data/demo_dataset.jsonl")
        example = list(dataset)[0]
        verifier = EdgeCaseVerifier(CodeExecutor(timeout_sec=0.5, python_executable="python3"))
        decision = verifier.verify(example, build_candidate("-5", example))
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.failure_category, FailureCategory.WRONG_ANSWER)

    def test_verifier_rejects_invalid_candidate(self):
        dataset = JSONLProblemDataset.from_jsonl("edge_case_generator/data/demo_dataset.jsonl")
        example = list(dataset)[1]
        verifier = EdgeCaseVerifier(CodeExecutor(timeout_sec=0.5, python_executable="python3"))
        decision = verifier.verify(example, build_candidate("x", example))
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.failure_category, FailureCategory.INVALID_FORMAT)


if __name__ == "__main__":
    unittest.main()
