"""Export accepted edge cases into an augmented dataset."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge_case_generator.buffer.replay_buffer import ReplayBuffer
from edge_case_generator.config import load_config
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.export import export_augmented_dataset

from edge_case_generator.scripts.common import parse_args


def main() -> None:
    """CLI entry point."""

    args = parse_args("Export an augmented dataset from the replay buffer")
    config = load_config(args.config)
    dataset = JSONLProblemDataset.from_jsonl(config["dataset"]["path"])
    replay_buffer = ReplayBuffer(
        path=config["buffer"]["path"],
        recent_window_size=config["buffer"]["recent_window_size"],
    )
    export_augmented_dataset(
        dataset=dataset,
        replay_buffer=replay_buffer,
        output_dir=config["output"]["dir"],
    )


if __name__ == "__main__":
    main()
