import tempfile
import unittest
from pathlib import Path

from edge_case_generator.collection.io import load_collection_config
from edge_case_generator.collection.pipeline import run_collection_pipeline
from edge_case_generator.utils.io import read_jsonl


class CollectionSmokeTests(unittest.TestCase):
    def test_end_to_end_collection_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "collection.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "seed: 13",
                        "target_language: python",
                        "sources:",
                        "  codenet:",
                        "    enabled: true",
                        "    path: edge_case_generator/data/collection_fixtures/codenet_fixture.jsonl",
                        "  codecontests:",
                        "    enabled: true",
                        "    path: edge_case_generator/data/collection_fixtures/codecontests_fixture.jsonl",
                        "  apps:",
                        "    enabled: false",
                        "    path: edge_case_generator/data/collection_fixtures/apps_fixture.jsonl",
                        "gt_selection:",
                        "  keep_multiple: false",
                        "  max_refs_per_problem: 1",
                        "buggy_collection:",
                        "  max_real_buggy_per_problem: 2",
                        "synthetic_mutation:",
                        "  enabled: true",
                        "  per_reference_limit: 2",
                        "execution:",
                        "  timeout_sec: 1.5",
                        "  python_executable: python3",
                        "sampling:",
                        "  num_samples_per_problem: 5",
                        "  max_attempts_per_problem: 8",
                        "  min_mismatch_cases_per_problem: 1",
                        "  save_rejected_diagnostics: true",
                        "comparison:",
                        "  mode: normalized",
                        "output:",
                        f"  dir: {temp_dir}/outputs",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_collection_config(config_path)
            summary = run_collection_pipeline(config)
            self.assertGreaterEqual(summary["problems_collected"], 2)
            self.assertGreaterEqual(summary["gt_solutions"], 2)
            self.assertGreaterEqual(summary["mismatch_edge_cases"], 1)

            problems = read_jsonl(Path(temp_dir) / "outputs" / "problems.jsonl")
            references = read_jsonl(Path(temp_dir) / "outputs" / "references.jsonl")
            buggy = read_jsonl(Path(temp_dir) / "outputs" / "buggy_programs.jsonl")
            edges = read_jsonl(Path(temp_dir) / "outputs" / "supervised_edge_cases.jsonl")

            self.assertTrue(problems)
            self.assertTrue(references)
            self.assertTrue(buggy)
            self.assertTrue(edges)
            self.assertTrue(all(edge["output_equal"] is False for edge in edges))


if __name__ == "__main__":
    unittest.main()
