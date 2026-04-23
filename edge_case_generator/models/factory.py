"""Generator model factory."""

from __future__ import annotations

from edge_case_generator.datasets.jsonl_dataset import JSONLProblemDataset
from edge_case_generator.models.base import BaseGeneratorPolicy
from edge_case_generator.models.llm_policy import PretrainedLLMGeneratorPolicy
from edge_case_generator.models.template_policy import TemplateGeneratorPolicy


def build_generator(dataset: JSONLProblemDataset, config: dict) -> BaseGeneratorPolicy:
    """Build the configured generator policy."""

    default_model_type = "template_policy" if config.get("debug") else "pretrained_llm"
    model_config = config.get("model", {"type": default_model_type})
    model_type = model_config.get("type", default_model_type)
    examples = list(dataset)

    if model_type == "pretrained_llm":
        backend_name = model_config.get("backend", "huggingface")
        return PretrainedLLMGeneratorPolicy.from_examples(
            examples=examples,
            backend_name=backend_name,
            seed=config["seed"],
            include_gt_code=model_config.get("include_gt_code", False),
            backend_config=model_config.get("huggingface", {}),
        )

    if model_type == "template_policy":
        return TemplateGeneratorPolicy.from_examples(
            examples=examples,
            max_len=config["sampling"]["max_len"],
            seed=config["seed"],
        )

    raise ValueError(f"Unsupported model.type: {model_type}")
