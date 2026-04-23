"""Simplified per-sample GRPO reward components."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any

from edge_case_generator.buffer.replay_buffer import ReplayBuffer
from edge_case_generator.types import RewardBreakdown, VerificationDecision


def _reward_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("reward", config.get("rewards", config))


def is_verified_mismatch(decision: VerificationDecision) -> bool:
    """Return whether the sample is a verified semantic mismatch."""

    return decision.distinct_verified_case_count > 0


def verified_mismatch_reward(decision: VerificationDecision) -> float:
    """Anchor reward: one for verified semantic mismatches, zero otherwise."""

    return 1.0 if is_verified_mismatch(decision) else 0.0


def canonical_case_hash_for_payload(problem_id: str, candidate_text: str, gt_output: str, input_output: str) -> str:
    """Hash a canonical mismatch case payload into a stable identity."""

    payload = "\n".join([problem_id, candidate_text, gt_output, input_output])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_globally_new_case(case_id: str | None, buffer: ReplayBuffer) -> bool:
    """Check if the case is globally new relative to replay memory."""

    if case_id is None:
        return False
    return not buffer.contains_case(case_id)


def new_frequency_allocation(freq: int, allocation: str) -> float:
    """Allocate new-case novelty credit across repeated new cases in a group."""

    freq = max(freq, 1)
    if allocation in {"inverse_frequency", "inverse_max_frequency"}:
        return 1.0 / freq
    if allocation == "inverse_sqrt_frequency":
        return 1.0 / math.sqrt(freq)
    raise ValueError(f"Unsupported newfreq allocation mode: {allocation}")


def new_case_frequency_reward(
    case_ids: set[str],
    frequency_table: Counter[str],
    buffer: ReplayBuffer,
    allocation_mode: str,
) -> float:
    """Reward globally new verified cases, splitting credit by group frequency."""

    total = 0.0
    for case_id in case_ids:
        if is_globally_new_case(case_id, buffer):
            total += new_frequency_allocation(frequency_table[case_id], allocation_mode)
    return total


def length_efficiency_reward(
    decision: VerificationDecision,
    mode: str,
    min_length: int,
    target_length: int,
    max_length: int,
    eps: float,
) -> float:
    """Encourage concise verified mismatches without rewarding invalid short junk."""

    if not is_verified_mismatch(decision):
        return 0.0

    length = len(decision.candidate.canonical_text)
    verified_count = max(decision.distinct_verified_case_count, 1)
    clipped_length = max(length, max(min_length, 1))
    if mode == "inverse_normalized_length":
        normalized = clipped_length / max(max_length, 1)
        return verified_count / (normalized + eps)
    if mode == "target_band":
        if min_length <= length <= max_length:
            distance = abs(length - target_length)
            band = max(max_length - min_length, 1)
            return verified_count * max(0.0, 1.0 - (distance / band))
        return 0.0
    if mode == "residual":
        residual = abs(length - target_length)
        return verified_count / (1.0 + (residual / max(target_length, 1)))
    raise ValueError(f"Unsupported length-efficiency mode: {mode}")


def duplicate_penalty(
    case_ids: set[str],
    candidate_hash: str,
    group_case_output_counts: Counter[str],
    buffer: ReplayBuffer,
    duplicate_config: dict[str, Any],
    intra_output_duplicate_count: int,
) -> float:
    """Apply a unit duplicate penalty when a configured duplicate condition is met."""

    duplicate = False
    if intra_output_duplicate_count > 0:
        duplicate = True
    if duplicate_config.get("check_group_duplicates", True) and any(group_case_output_counts[case_id] > 1 for case_id in case_ids):
        duplicate = True
    if duplicate_config.get("check_replay_duplicates", True):
        duplicate = duplicate or buffer.contains(candidate_hash) or any(buffer.contains_case(case_id) for case_id in case_ids)
    if duplicate_config.get("check_recent_history", True):
        duplicate = duplicate or buffer.seen_recently(candidate_hash) or any(buffer.seen_case_recently(case_id) for case_id in case_ids)
    return 1.0 if duplicate else 0.0


def aggregate_reward(
    verify_component: float,
    newfreq_component: float,
    length_component: float,
    duplicate_component: float,
    reward_config: dict[str, Any],
) -> float:
    """Combine weighted per-sample reward components."""

    return (
        reward_config["verify_weight"] * verify_component
        + reward_config["newfreq_weight"] * newfreq_component
        + reward_config["length_efficiency_weight"] * length_component
        - reward_config["duplicate_weight"] * duplicate_component
    )


def grpo_advantages(rewards: list[float], mode: str, eps: float) -> list[float]:
    """Compute centered or std-normalized GRPO advantages."""

    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    centered = [reward - mean_reward for reward in rewards]
    if mode == "centered":
        return centered
    if mode == "std_normalized":
        variance = sum(value * value for value in centered) / len(centered)
        std = math.sqrt(variance)
        return [value / (std + eps) for value in centered]
    raise ValueError(f"Unsupported advantage normalization mode: {mode}")


def build_reward_breakdowns(
    decisions: list[VerificationDecision],
    buffer: ReplayBuffer,
    config: dict[str, Any],
) -> tuple[list[RewardBreakdown], dict[str, Any]]:
    """Build per-sample rewards and group-derived logging stats."""

    reward_config = _reward_config(config)
    grpo_config = config.get("grpo", {})

    output_case_sets = [set(decision.verified_case_ids) for decision in decisions]
    verified_flags = [is_verified_mismatch(decision) for decision in decisions]
    frequency_table = Counter(case_id for case_set in output_case_sets for case_id in case_set)
    group_candidate_counts = Counter(decision.candidate.candidate_hash for decision in decisions)

    breakdowns: list[RewardBreakdown] = []
    totals: list[float] = []
    for decision, case_ids in zip(decisions, output_case_sets):
        verify_mode = reward_config.get("verify", {}).get("mode", "binary")
        verify_cap = reward_config.get("verify", {}).get("cap", 1)
        if verify_mode == "binary":
            verify_component = 1.0 if is_verified_mismatch(decision) else 0.0
        elif verify_mode == "count":
            verify_component = float(decision.distinct_verified_case_count)
        elif verify_mode == "capped_count":
            verify_component = float(min(decision.distinct_verified_case_count, verify_cap))
        else:
            raise ValueError(f"Unsupported verify mode: {verify_mode}")
        newfreq_component = new_case_frequency_reward(
            case_ids=case_ids,
            frequency_table=frequency_table,
            buffer=buffer,
            allocation_mode=reward_config["newfreq"]["allocation"],
        )
        length_component = 0.0
        if reward_config["length_efficiency"].get("enabled", True):
            length_component = length_efficiency_reward(
                decision=decision,
                mode=reward_config["length_efficiency"]["mode"],
                min_length=reward_config["length_efficiency"]["min_length"],
                target_length=reward_config["length_efficiency"]["target_length"],
                max_length=reward_config["length_efficiency"]["max_length"],
                eps=reward_config["length_efficiency"]["eps"],
            )
        duplicate_component = duplicate_penalty(
            case_ids=case_ids,
            candidate_hash=decision.candidate.candidate_hash,
            group_case_output_counts=frequency_table,
            buffer=buffer,
            duplicate_config=reward_config["duplicate"],
            intra_output_duplicate_count=decision.intra_output_duplicate_count,
        )
        total = aggregate_reward(
            verify_component=verify_component,
            newfreq_component=newfreq_component,
            length_component=length_component,
            duplicate_component=duplicate_component,
            reward_config=reward_config,
        )
        totals.append(total)
        globally_new_case_ids = [case_id for case_id in sorted(case_ids) if is_globally_new_case(case_id, buffer)]
        breakdowns.append(
            RewardBreakdown(
                total_reward=total,
                verification_reward=verify_component,
                new_frequency_reward=newfreq_component,
                length_efficiency_reward=length_component,
                duplicate_penalty=duplicate_component,
                case_ids=sorted(case_ids),
                globally_new_case_ids=globally_new_case_ids,
                within_group_frequency={case_id: frequency_table[case_id] for case_id in sorted(case_ids)},
                is_duplicate=duplicate_component > 0,
            )
        )

    advantage_mode = grpo_config.get("advantage_normalization", "centered")
    eps = grpo_config.get("eps", 1e-8)
    advantages = grpo_advantages(totals, mode=advantage_mode, eps=eps)
    mean_total = sum(totals) / len(totals) if totals else 0.0
    for breakdown, advantage in zip(breakdowns, advantages):
        breakdown.raw_advantage = breakdown.total_reward - mean_total
        breakdown.normalized_advantage = advantage

    verified_lengths = [
        len(decision.candidate.canonical_text)
        for decision, verified in zip(decisions, verified_flags)
        if verified
    ]
    summary = {
        "group_size": len(decisions),
        "verified_mismatch_count": sum(verified_flags),
        "verified_mismatch_ratio": sum(verified_flags) / max(len(decisions), 1),
        "distinct_new_verified_cases": len({case_id for breakdown in breakdowns for case_id in breakdown.globally_new_case_ids}),
        "frequency_table": dict(frequency_table),
        "intra_output_duplicate_count": sum(decision.intra_output_duplicate_count for decision in decisions),
        "duplicate_count": sum(1 for breakdown in breakdowns if breakdown.is_duplicate),
        "globally_new_case_count": len({case_id for breakdown in breakdowns for case_id in breakdown.globally_new_case_ids}),
        "accepted_case_length_mean": (sum(verified_lengths) / len(verified_lengths)) if verified_lengths else 0.0,
        "verification_reward_mean": (sum(b.verification_reward for b in breakdowns) / len(breakdowns)) if breakdowns else 0.0,
        "verification_reward_std": _std([b.verification_reward for b in breakdowns]),
        "new_frequency_reward_mean": (sum(b.new_frequency_reward for b in breakdowns) / len(breakdowns)) if breakdowns else 0.0,
        "new_frequency_reward_std": _std([b.new_frequency_reward for b in breakdowns]),
        "length_efficiency_reward_mean": (sum(b.length_efficiency_reward for b in breakdowns) / len(breakdowns)) if breakdowns else 0.0,
        "length_efficiency_reward_std": _std([b.length_efficiency_reward for b in breakdowns]),
        "duplicate_penalty_mean": (sum(b.duplicate_penalty for b in breakdowns) / len(breakdowns)) if breakdowns else 0.0,
        "duplicate_penalty_std": _std([b.duplicate_penalty for b in breakdowns]),
        "total_reward_mean": (sum(totals) / len(totals)) if totals else 0.0,
        "total_reward_std": _std(totals),
        "advantage_mean": (sum(advantages) / len(advantages)) if advantages else 0.0,
        "advantage_std": _std(advantages),
    }
    return breakdowns, summary


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)
