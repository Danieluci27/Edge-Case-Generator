"""Run a tiny end-to-end smoke experiment."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathlib import Path

from edge_case_generator.buffer.replay_buffer import ReplayBuffer
from edge_case_generator.export import export_augmented_dataset
from edge_case_generator.rl.trainer import RLTrainer
from edge_case_generator.scripts.common import build_runtime


def main() -> None:
    """Train on the debug config and export results."""

    config_path = "edge_case_generator/configs/debug.yaml"
    config, train_dataset, generator, verifier, replay_buffer, logger, metrics_logger = build_runtime(config_path)
    trainer = RLTrainer(
        dataset=train_dataset,
        generator=generator,
        verifier=verifier,
        replay_buffer=replay_buffer,
        metrics_logger=metrics_logger,
        logger=logger,
        config=config,
    )
    trainer.train()

    refreshed_buffer = ReplayBuffer(
        path=config["buffer"]["path"],
        recent_window_size=config["buffer"]["recent_window_size"],
    )
    summary = export_augmented_dataset(train_dataset, refreshed_buffer, config["output"]["dir"])
    logger.info("Smoke test export complete: %s", summary["jsonl_path"])
    assert Path(summary["jsonl_path"]).exists(), "Expected augmented dataset export to exist"


if __name__ == "__main__":
    main()
