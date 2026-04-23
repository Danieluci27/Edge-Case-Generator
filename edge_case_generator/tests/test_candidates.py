import unittest

from edge_case_generator.datasets.candidates import build_candidate, canonicalize_text, deserialize_candidate, serialize_candidate
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset


class CandidateTests(unittest.TestCase):
    def test_candidate_canonicalization_and_serialization(self):
        dataset = JSONLProblemDataset.from_jsonl("edge_case_generator/data/demo_dataset.jsonl")
        example = next(iter(dataset))
        candidate = build_candidate(" -5 \n", example)
        self.assertEqual(canonicalize_text(" -5 \n"), "-5")
        self.assertTrue(candidate.valid)

        restored = deserialize_candidate(serialize_candidate(candidate))
        self.assertEqual(restored.candidate_hash, candidate.candidate_hash)

    def test_candidate_validation_failure(self):
        dataset = JSONLProblemDataset.from_jsonl("edge_case_generator/data/demo_dataset.jsonl")
        example = list(dataset)[1]
        candidate = build_candidate("2\n1 nope", example)
        self.assertFalse(candidate.valid)
        self.assertIsNotNone(candidate.validation_error)


if __name__ == "__main__":
    unittest.main()
