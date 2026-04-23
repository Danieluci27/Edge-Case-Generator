"""Candidate representation helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from edge_case_generator.types import CandidateInput, ProblemExample


def canonicalize_text(text: str) -> str:
    """Normalize candidate text for hashing and deduplication."""

    normalized_lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    canonical = "\n".join(normalized_lines).strip()
    return canonical


def candidate_hash(canonical_text: str) -> str:
    """Stable hash for a canonical candidate."""

    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def _validate_regex(candidate: str, pattern: str) -> tuple[bool, str | None]:
    if re.fullmatch(pattern, candidate, flags=re.MULTILINE) is None:
        return False, f"Candidate does not match required regex: {pattern}"
    return True, None


def _validate_numeric_spec(candidate: str, spec: dict[str, Any]) -> tuple[bool, str | None]:
    lines = [line for line in candidate.splitlines() if line.strip() or len(candidate.splitlines()) == 1]
    mode = spec.get("type", "raw_text")
    if mode == "single_int":
        if len(lines) != 1:
            return False, "Expected a single-line integer input"
        try:
            int(lines[0].strip())
        except ValueError:
            return False, "Expected a valid integer"
    elif mode == "int_list":
        if len(lines) != 1:
            return False, "Expected a single-line list of integers"
        for token in lines[0].split():
            try:
                int(token)
            except ValueError:
                return False, "List contains a non-integer token"
    elif mode == "int_with_count":
        if len(lines) < 1:
            return False, "Missing count line"
        try:
            count = int(lines[0].strip())
        except ValueError:
            return False, "First line must be an integer count"
        if len(lines) < 2:
            return False, "Missing payload line"
        tokens = lines[1].split()
        if count != len(tokens):
            return False, "Count does not match token count"
        for token in tokens:
            try:
                int(token)
            except ValueError:
                return False, "Payload contains a non-integer token"
    return True, None


def build_candidate(raw_text: str, example: ProblemExample) -> CandidateInput:
    """Create a candidate with canonicalization, hashing, and validation."""

    canonical = canonicalize_text(raw_text)
    valid = True
    error: str | None = None
    metadata = example.metadata or {}

    regex = metadata.get("input_regex")
    if regex:
        valid, error = _validate_regex(canonical, regex)

    if valid and metadata.get("input_spec"):
        valid, error = _validate_numeric_spec(canonical, metadata["input_spec"])

    return CandidateInput(
        raw_text=raw_text,
        canonical_text=canonical,
        candidate_hash=candidate_hash(canonical),
        valid=valid,
        validation_error=error,
        structured_payload={"format": metadata.get("input_spec", {"type": "raw_text"})},
    )


def serialize_candidate(candidate: CandidateInput) -> str:
    """Serialize a candidate to JSON."""

    return json.dumps(candidate.to_dict(), sort_keys=True)


def deserialize_candidate(payload: str) -> CandidateInput:
    """Deserialize a candidate from JSON."""

    raw = json.loads(payload)
    return CandidateInput(**raw)

