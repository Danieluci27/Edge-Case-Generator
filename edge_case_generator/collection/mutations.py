"""Synthetic mutation operators for runnable semantic bugs."""

from __future__ import annotations

import ast
import copy
import random
from dataclasses import dataclass

from edge_case_generator.collection.types import BuggySolutionRecord, ReferenceSolutionRecord


def _clone_module(tree: ast.AST) -> ast.AST:
    return copy.deepcopy(tree)


@dataclass
class MutationResult:
    mutation_type: str
    code: str


class _SingleMutationTransformer(ast.NodeTransformer):
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.changed = False

    def visit_Compare(self, node: ast.Compare):  # type: ignore[override]
        self.generic_visit(node)
        if self.changed or self.mode != "comparator_flip":
            return node
        if not node.ops:
            return node
        first = node.ops[0]
        replacements = {
            ast.Gt: ast.GtE,
            ast.GtE: ast.Gt,
            ast.Lt: ast.LtE,
            ast.LtE: ast.Lt,
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
        }
        for source, target in replacements.items():
            if isinstance(first, source):
                node.ops[0] = target()
                self.changed = True
                return node
        return node

    def visit_Call(self, node: ast.Call):  # type: ignore[override]
        self.generic_visit(node)
        if self.changed or self.mode != "off_by_one_range":
            return node
        if isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
            last_arg = node.args[-1]
            node.args[-1] = ast.BinOp(left=last_arg, op=ast.Add(), right=ast.Constant(value=1))
            self.changed = True
        return node

    def visit_Assign(self, node: ast.Assign):  # type: ignore[override]
        self.generic_visit(node)
        if self.changed or self.mode != "wrong_initialization":
            return node
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            if node.value.value == 0:
                node.value = ast.Constant(value=1)
                self.changed = True
            elif node.value.value == 1:
                node.value = ast.Constant(value=0)
                self.changed = True
        return node

    def visit_AugAssign(self, node: ast.AugAssign):  # type: ignore[override]
        self.generic_visit(node)
        if self.changed or self.mode != "wrong_accumulator_update":
            return node
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
            self.changed = True
        elif isinstance(node.op, ast.Sub):
            node.op = ast.Add()
            self.changed = True
        return node

    def visit_BinOp(self, node: ast.BinOp):  # type: ignore[override]
        self.generic_visit(node)
        if self.changed or self.mode != "division_rounding_bug":
            return node
        if isinstance(node.op, ast.Div):
            node.op = ast.FloorDiv()
            self.changed = True
        elif isinstance(node.op, ast.FloorDiv):
            node.op = ast.Div()
            self.changed = True
        return node

    def visit_Return(self, node: ast.Return):  # type: ignore[override]
        self.generic_visit(node)
        if self.changed or self.mode != "wrong_default_return":
            return node
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            node.value = ast.Constant(value=node.value.value + 1)
            self.changed = True
        return node


MUTATION_ORDER = [
    "comparator_flip",
    "off_by_one_range",
    "wrong_initialization",
    "wrong_accumulator_update",
    "division_rounding_bug",
    "wrong_default_return",
]


def apply_mutation(code: str, mutation_type: str) -> MutationResult | None:
    """Apply a single AST-based mutation and return runnable code when successful."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    transformer = _SingleMutationTransformer(mutation_type)
    mutated = transformer.visit(_clone_module(tree))
    ast.fix_missing_locations(mutated)
    if not transformer.changed:
        return None
    return MutationResult(mutation_type=mutation_type, code=ast.unparse(mutated))


def generate_synthetic_buggy_solutions(
    references: list[ReferenceSolutionRecord],
    *,
    per_reference_limit: int = 2,
    seed: int = 0,
) -> list[BuggySolutionRecord]:
    """Generate deterministic synthetic runnable bug variants from GT code."""

    rng = random.Random(seed)
    results: list[BuggySolutionRecord] = []
    for reference in references:
        mutation_types = list(MUTATION_ORDER)
        rng.shuffle(mutation_types)
        emitted = 0
        for mutation_type in mutation_types:
            mutated = apply_mutation(reference.gt_code, mutation_type)
            if mutated is None or mutated.code.strip() == reference.gt_code.strip():
                continue
            results.append(
                BuggySolutionRecord(
                    problem_id=reference.problem_id,
                    buggy_id=f"{reference.solution_id}__{mutation_type}",
                    language=reference.language,
                    buggy_code=mutated.code,
                    bug_origin="synthetic_mutation",
                    original_verdict=None,
                    mutation_type=mutation_type,
                    metadata={"reference_solution_id": reference.solution_id},
                )
            )
            emitted += 1
            if emitted >= per_reference_limit:
                break
    return results
