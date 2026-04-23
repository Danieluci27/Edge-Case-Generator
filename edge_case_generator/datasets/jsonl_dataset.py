"""JSONL dataset loading and split helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from edge_case_generator.types import ProblemExample
from edge_case_generator.utils.io import read_jsonl


class JSONLProblemDataset:
    """Dataset wrapper with deterministic split support."""

    def __init__(self, examples: Iterable[ProblemExample]) -> None:
        self.examples = list(examples)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "JSONLProblemDataset":
        """Load dataset examples from JSONL."""

        rows = read_jsonl(path)
        examples = [ProblemExample(**row) for row in rows]
        return cls(examples)

    def subset(self, limit: int | None) -> "JSONLProblemDataset":
        """Return a capped subset for debugging."""

        if limit is None:
            return JSONLProblemDataset(self.examples)
        return JSONLProblemDataset(self.examples[:limit])

    def shuffled(self, seed: int) -> "JSONLProblemDataset":
        """Return a deterministically shuffled dataset."""

        rng = random.Random(seed)
        copied = list(self.examples)
        rng.shuffle(copied)
        return JSONLProblemDataset(copied)

    def split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> dict[str, "JSONLProblemDataset"]:
        """Split into train, validation, and test datasets."""

        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        shuffled = self.shuffled(seed).examples
        n_examples = len(shuffled)
        train_end = int(n_examples * train_ratio)
        val_end = train_end + int(n_examples * val_ratio)
        return {
            "train": JSONLProblemDataset(shuffled[:train_end]),
            "val": JSONLProblemDataset(shuffled[train_end:val_end]),
            "test": JSONLProblemDataset(shuffled[val_end:]),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

