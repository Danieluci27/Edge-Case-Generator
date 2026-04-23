"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed available random sources."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

