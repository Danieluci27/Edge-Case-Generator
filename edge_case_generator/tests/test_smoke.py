import tempfile
import unittest
from pathlib import Path

from edge_case_generator.buffer.replay_buffer import ReplayBuffer
from edge_case_generator.export import export_augmented_dataset
from edge_case_generator.rl.trainer import RLTrainer
from edge_case_generator.scripts.common import build_runtime


class SmokeTests(unittest.TestCase):
    def test_end_to_end_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "seed: 7",
                        "debug: true",
                        "dataset:",
                        "  path: edge_case_generator/data/demo_dataset.jsonl",
                        "  split:",
                        "    train: 0.67",
                        "    val: 0.0",
                        "    test: 0.33",
                        "  subset: null",
                        "execution:",
                        "  timeout_sec: 0.5",
                        "  python_executable: python3",
                        "sampling:",
                        "  group_size: 4",
                        "  temperature: 1.0",
                        "  top_p: 1.0",
                        "  max_len: 128",
                        "rl:",
                        "  algorithm: grpo",
                        "  num_iterations: 2",
                        "  learning_rate: 0.5",
                        "  baseline_momentum: 0.8",
                        "grpo:",
                        "  advantage_normalization: std_normalized",
                        "  eps: 1.0e-8",
                        "reward:",
                        "  verify_weight: 5.0",
                        "  newfreq_weight: 2.0",
                        "  length_efficiency_weight: 0.2",
                        "  duplicate_weight: 1.5",
                        "  verify:",
                        "    mode: binary",
                        "    cap: 1",
                        "  newfreq:",
                        "    allocation: inverse_frequency",
                        "  length_efficiency:",
                        "    enabled: true",
                        "    mode: inverse_normalized_length",
                        "    min_length: 1",
                        "    target_length: 8",
                        "    max_length: 128",
                        "    eps: 1.0e-8",
                        "  duplicate:",
                        "    check_group_duplicates: true",
                        "    check_replay_duplicates: true",
                        "    check_recent_history: true",
                        "buffer:",
                        f"  path: {temp_path / 'replay_buffer.jsonl'}",
                        "  recent_window_size: 50",
                        "output:",
                        f"  dir: {temp_path / 'outputs'}",
                        "  checkpoint_every: 1",
                        "logging:",
                        "  level: INFO",
                    ]
                ),
                encoding="utf-8",
            )
            config, train_dataset, generator, verifier, replay_buffer, logger, metrics_logger = build_runtime(str(config_path))
            trainer = RLTrainer(train_dataset, generator, verifier, replay_buffer, metrics_logger, logger, config)
            summary = trainer.train()
            self.assertGreaterEqual(summary["buffer"]["accepted_count"], 1)

            refreshed = ReplayBuffer(path=config["buffer"]["path"], recent_window_size=50)
            export_summary = export_augmented_dataset(train_dataset, refreshed, config["output"]["dir"])
            self.assertTrue(Path(export_summary["jsonl_path"]).exists())
            self.assertGreaterEqual(export_summary["accepted_record_count"], 1)


if __name__ == "__main__":
    unittest.main()
