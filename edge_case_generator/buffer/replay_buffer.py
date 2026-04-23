"""Persistent replay buffer for accepted edge cases."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from edge_case_generator.types import ReplayRecord, RewardBreakdown, VerificationDecision
from edge_case_generator.utils.io import read_jsonl, write_jsonl


class ReplayBuffer:
    """Persistent accepted-case store used for novelty and export."""

    def __init__(self, path: str | Path, recent_window_size: int = 100) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.recent_window_size = recent_window_size
        self.records: list[ReplayRecord] = []
        self.hashes: set[str] = set()
        self.case_ids: set[str] = set()
        self.recent_hashes: deque[str] = deque(maxlen=recent_window_size)
        self.recent_case_ids: deque[str] = deque(maxlen=recent_window_size)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        rows = read_jsonl(self.path, ignore_errors=True)
        normalized_rows = []
        for row in rows:
            if "case_ids" not in row:
                legacy_case_id = row.pop("case_id", None)
                row["case_ids"] = [legacy_case_id] if legacy_case_id else []
            normalized_rows.append(row)
        self.records = [ReplayRecord(**row) for row in normalized_rows]
        self.hashes = {record.candidate_hash for record in self.records}
        self.case_ids = {case_id for record in self.records for case_id in record.case_ids}
        self.recent_hashes = deque((record.candidate_hash for record in self.records[-self.recent_window_size :]), maxlen=self.recent_window_size)
        self.recent_case_ids = deque(
            (case_id for record in self.records[-self.recent_window_size :] for case_id in record.case_ids),
            maxlen=self.recent_window_size,
        )

    def contains(self, candidate_hash: str) -> bool:
        return candidate_hash in self.hashes

    def seen_recently(self, candidate_hash: str) -> bool:
        return candidate_hash in set(self.recent_hashes)

    def contains_case(self, case_id: str) -> bool:
        return case_id in self.case_ids

    def seen_case_recently(self, case_id: str) -> bool:
        return case_id in set(self.recent_case_ids)

    def add(self, decision: VerificationDecision, reward: RewardBreakdown, iteration: int) -> ReplayRecord:
        """Add an accepted decision to the buffer."""

        record = ReplayRecord(
            problem_id=decision.problem_id,
            candidate_input=decision.candidate.raw_text,
            canonical_input=decision.candidate.canonical_text,
            candidate_hash=decision.candidate.candidate_hash,
            case_ids=list(reward.case_ids),
            reward=reward.total_reward,
            reward_components=reward.to_dict(),
            verifier_outcome=decision.to_dict(),
            failure_category=decision.failure_category.value if decision.failure_category else None,
            iteration=iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            gt_output=decision.gt_output,
            input_output=decision.input_output,
        )
        self.records.append(record)
        self.hashes.add(record.candidate_hash)
        self.case_ids.update(record.case_ids)
        self.recent_hashes.append(record.candidate_hash)
        for case_id in record.case_ids:
            self.recent_case_ids.append(case_id)
        self.flush()
        return record

    def flush(self) -> None:
        """Persist buffer contents to disk."""

        write_jsonl(self.path, (record.to_dict() for record in self.records))

    def stats(self) -> dict[str, int]:
        """Return simple buffer summary stats."""

        return {
            "accepted_count": len(self.records),
            "unique_hash_count": len(self.hashes),
            "recent_window_size": self.recent_window_size,
        }
