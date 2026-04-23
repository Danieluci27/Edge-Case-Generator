"""Shared script helpers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from edge_case_generator.buffer.replay_buffer import ReplayBuffer
from edge_case_generator.config import load_config
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.models.factory import build_generator
from edge_case_generator.utils.logging_utils import MetricsLogger, configure_logging
from edge_case_generator.utils.randomness import seed_everything
from edge_case_generator.verifier.decision import EdgeCaseVerifier
from edge_case_generator.verifier.executor import CodeExecutor


def parse_args(description: str) -> argparse.Namespace:
    """Common argument parser."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def build_runtime(config_path: str):
    """Build common runtime objects from config."""

    config = load_config(config_path)
    seed_everything(config["seed"])

    output_dir = Path(config["output"]["dir"])
    logger = configure_logging(output_dir, getattr(logging, config["logging"]["level"]))
    metrics_logger = MetricsLogger(output_dir)

    dataset = JSONLProblemDataset.from_jsonl(config["dataset"]["path"])
    dataset = dataset.shuffled(config["seed"]).subset(config["dataset"].get("subset"))
    splits = dataset.split(
        train_ratio=config["dataset"]["split"]["train"],
        val_ratio=config["dataset"]["split"]["val"],
        test_ratio=config["dataset"]["split"]["test"],
        seed=config["seed"],
    )
    train_dataset = splits["train"] if len(splits["train"]) > 0 else dataset

    generator = build_generator(train_dataset, config)
    verifier = EdgeCaseVerifier(
        CodeExecutor(
            timeout_sec=config["execution"]["timeout_sec"],
            python_executable=config["execution"].get("python_executable", "python"),
        )
    )
    replay_buffer = ReplayBuffer(
        path=config["buffer"]["path"],
        recent_window_size=config["buffer"]["recent_window_size"],
    )
    return config, train_dataset, generator, verifier, replay_buffer, logger, metrics_logger
