import unittest

from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset


class DatasetTests(unittest.TestCase):
    def test_dataset_loading_and_split(self):
        dataset = JSONLProblemDataset.from_jsonl("edge_case_generator/data/demo_dataset.jsonl")
        self.assertEqual(len(dataset), 3)

        shuffled_once = [example.problem_id for example in dataset.shuffled(7)]
        shuffled_twice = [example.problem_id for example in dataset.shuffled(7)]
        self.assertEqual(shuffled_once, shuffled_twice)

        splits = dataset.split(train_ratio=0.67, val_ratio=0.0, test_ratio=0.33, seed=3)
        self.assertEqual(len(splits["train"]) + len(splits["test"]), len(dataset))


if __name__ == "__main__":
    unittest.main()
