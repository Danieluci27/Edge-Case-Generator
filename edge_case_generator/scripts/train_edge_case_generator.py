"""Train the edge-case generator."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge_case_generator.rl.trainer import RLTrainer
from edge_case_generator.scripts.common import build_runtime, parse_args


def main() -> None:
    """CLI entry point."""

    args = parse_args("Train the edge-case generator")
    config, train_dataset, generator, verifier, replay_buffer, logger, metrics_logger = build_runtime(args.config)
    trainer = RLTrainer(
        dataset=train_dataset,
        generator=generator,
        verifier=verifier,
        replay_buffer=replay_buffer,
        metrics_logger=metrics_logger,
        logger=logger,
        config=config,
    )
    summary = trainer.train()
    logger.info("Training complete. Accepted cases: %s", summary["buffer"]["accepted_count"])


if __name__ == "__main__":
    main()
