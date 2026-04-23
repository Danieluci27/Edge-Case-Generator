"""Inspect baseline edge-case generation from a pretrained LLM backend."""

from __future__ import annotations

import argparse

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from edge_case_generator.models.llm_policy import PretrainedLLMGeneratorPolicy
from edge_case_generator.models.prompting import build_generation_prompt
from edge_case_generator.types import ProblemExample


def build_synthetic_examples() -> list[ProblemExample]:
    """Create a small synthetic set of buggy programs for baseline inspection."""

    return [
        ProblemExample(
            problem_id="abs_int_bug",
            prompt="Read a single integer and print its absolute value.",
            input_code=(
                "import sys\n"
                "n = int(sys.stdin.read().strip())\n"
                "print(n)\n"
            ),
            gt_code=(
                "import sys\n"
                "n = int(sys.stdin.read().strip())\n"
                "print(abs(n))\n"
            ),
            public_tests=[{"input": "5", "output": "5"}],
            metadata={"input_spec": {"type": "single_int"}},
        ),
        ProblemExample(
            problem_id="distinct_count_bug",
            prompt="Read N and then N integers. Print the count of distinct integers.",
            input_code=(
                "import sys\n"
                "parts = sys.stdin.read().strip().split()\n"
                "n = int(parts[0])\n"
                "nums = list(map(int, parts[1:1+n]))\n"
                "count = 0\n"
                "prev = None\n"
                "for x in nums:\n"
                "    if x != prev:\n"
                "        count += 1\n"
                "        prev = x\n"
                "print(count)\n"
            ),
            gt_code=(
                "import sys\n"
                "parts = sys.stdin.read().strip().split()\n"
                "n = int(parts[0])\n"
                "nums = list(map(int, parts[1:1+n]))\n"
                "print(len(set(nums)))\n"
            ),
            public_tests=[{"input": "3\n1 2 3", "output": "3"}],
            metadata={"input_spec": {"type": "int_with_count"}},
        ),
        ProblemExample(
            problem_id="sum_vs_average_bug",
            prompt="Read a space-separated list of integers and print their sum.",
            input_code=(
                "import sys\n"
                "nums = list(map(int, sys.stdin.read().strip().split()))\n"
                "print(sum(nums) / len(nums))\n"
            ),
            gt_code=(
                "import sys\n"
                "text = sys.stdin.read().strip()\n"
                "nums = list(map(int, text.split())) if text else []\n"
                "print(sum(nums))\n"
            ),
            public_tests=[{"input": "1 2 3", "output": "6"}],
            metadata={"input_spec": {"type": "int_list"}},
        ),
        ProblemExample(
            problem_id="min_bug",
            prompt="Read N and then N integers. Print the minimum value.",
            input_code=(
                "import sys\n"
                "parts = sys.stdin.read().strip().split()\n"
                "n = int(parts[0])\n"
                "nums = list(map(int, parts[1:1+n]))\n"
                "print(nums[0])\n"
            ),
            gt_code=(
                "import sys\n"
                "parts = sys.stdin.read().strip().split()\n"
                "n = int(parts[0])\n"
                "nums = list(map(int, parts[1:1+n]))\n"
                "print(min(nums))\n"
            ),
            public_tests=[{"input": "3\n4 2 5", "output": "2"}],
            metadata={"input_spec": {"type": "int_with_count"}},
        ),
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Inspect baseline edge-case candidates from a pretrained LLM.")
    parser.add_argument(
        "--model",
        default="Intel/tiny-random-distilgpt2",
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of candidate edge cases to sample per problem.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=32,
        help="Maximum generated candidate length in new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional existing adapter checkpoint to load.",
    )
    parser.add_argument(
        "--include-gt-code",
        action="store_true",
        help="Include reference code in the conditioning prompt.",
    )
    return parser.parse_args()


def main() -> None:
    """Run baseline LLM sampling on synthetic buggy tasks."""

    args = parse_args()
    examples = build_synthetic_examples()
    backend_config = {
        "pretrained_model_name_or_path": args.model,
        "use_safetensors": True,
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "padding_side": "left",
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "max_grad_norm": 1.0,
        "normalize_loss_by_tokens": True,
        "adapter": {
            "enabled": args.adapter_path is not None,
            "path": args.adapter_path,
            "name": "edge_case_lora",
        },
    }

    policy = PretrainedLLMGeneratorPolicy.from_examples(
        examples=examples,
        backend_name="huggingface",
        seed=7,
        include_gt_code=args.include_gt_code,
        backend_config=backend_config,
    )
    policy.eval()

    for example in examples:
        prompt = build_generation_prompt(example, include_gt_code=args.include_gt_code)
        samples = policy.sample_candidates(
            example=example,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_len=args.max_len,
        )

        print("=" * 100)
        print(f"Problem ID: {example.problem_id}")
        print(f"Task: {example.prompt}")
        print("-" * 100)
        print("Buggy Code:")
        print(example.input_code.rstrip())
        print("-" * 100)
        print("Prompt Sent To LLM:")
        print(prompt)
        print("-" * 100)
        print("LLM Candidate Outputs:")
        for index, sample in enumerate(samples, start=1):
            print(
                f"[{index}] candidate={sample.candidate.canonical_text!r} "
                f"valid={sample.candidate.valid} "
                f"logprob={sample.logprob:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
