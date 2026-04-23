"""Augmented dataset export utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from edge_case_generator.buffer.replay_buffer import ReplayBuffer
from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.utils.io import write_json, write_jsonl


def export_augmented_dataset(
    dataset: JSONLProblemDataset,
    replay_buffer: ReplayBuffer,
    output_dir: str | Path,
) -> dict:
    """Export accepted cases merged with original problem metadata."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_by_problem = {example.problem_id: example for example in dataset}

    records = []
    failure_counts: Counter[str] = Counter()
    for record in replay_buffer.records:
        example = dataset_by_problem[record.problem_id]
        failure_counts[record.failure_category or "unknown"] += 1
        records.append(
            {
                "problem_id": record.problem_id,
                "prompt": example.prompt,
                "candidate_input": record.candidate_input,
                "canonical_input": record.canonical_input,
                "candidate_hash": record.candidate_hash,
                "gt_output": record.gt_output,
                "input_output": record.input_output,
                "failure_category": record.failure_category,
                "reward": record.reward,
                "reward_components": record.reward_components,
                "verifier_labels": record.verifier_outcome,
                "metadata": example.metadata,
            }
        )

    jsonl_path = destination / "augmented_dataset.jsonl"
    summary_path = destination / "augmented_summary.json"
    write_jsonl(jsonl_path, records)
    summary = {
        "accepted_record_count": len(records),
        "unique_problem_count": len({record["problem_id"] for record in records}),
        "failure_category_counts": dict(failure_counts),
        "jsonl_path": str(jsonl_path),
    }
    write_json(summary_path, summary)
    return summary
