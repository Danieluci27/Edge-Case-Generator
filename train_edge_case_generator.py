"""Convenience wrapper for training the edge-case generator from repo root."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from edge_case_generator.scripts.train_edge_case_generator import main


if __name__ == "__main__":
    main()
