"""Verify candidate inputs from a JSONL file."""

from __future__ import annotations

import argparse

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge_case_generator.config import load_config
from edge_case_generator.datasets.candidates import build_candidate
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.utils.io import read_jsonl, write_jsonl
from edge_case_generator.verifier.decision import EdgeCaseVerifier
from edge_case_generator.verifier.executor import CodeExecutor


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Verify candidate inputs against problem implementations")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--candidates", required=True, help="JSONL file with problem_id and candidate_input")
    parser.add_argument("--output", required=True, help="Where to write verification results JSONL")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = JSONLProblemDataset.from_jsonl(config["dataset"]["path"])
    problems = {example.problem_id: example for example in dataset}
    verifier = EdgeCaseVerifier(
        CodeExecutor(
            timeout_sec=config["execution"]["timeout_sec"],
            python_executable=config["execution"].get("python_executable", "python"),
        )
    )

    results = []
    for row in read_jsonl(args.candidates):
        example = problems[row["problem_id"]]
        candidate = build_candidate(row["candidate_input"], example)
        decision = verifier.verify(example, candidate)
        results.append(decision.to_dict())
    write_jsonl(args.output, results)


if __name__ == "__main__":
    main()
