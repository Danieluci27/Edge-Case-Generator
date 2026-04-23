"""Pretrained-LLM-style policy wrappers with a Hugging Face backend."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_case_generator.datasets.candidates import build_candidate, candidate_hash, canonicalize_text
from edge_case_generator.models.base import BaseGeneratorPolicy
from edge_case_generator.models.prompting import build_generation_prompt
from edge_case_generator.types import PolicyRolloutSample, ProblemExample, SampledCandidate


@dataclass
class BackendGeneration:
    """Single backend candidate proposal before policy scoring."""

    text: str
    base_score: float
    logprob: float | None = None


class BaseLLMBackend(ABC):
    """Backend interface for pretrained generator models."""

    @abstractmethod
    def generate_candidates(
        self,
        example: ProblemExample,
        prompt: str,
        num_samples: int,
        max_len: int,
        temperature: float,
        top_p: float,
    ) -> list[BackendGeneration]:
        """Generate candidate strings and backend prior scores."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return backend metadata for checkpoints and logs."""

    def supports_policy_gradient(self) -> bool:
        """Whether the backend supports gradient updates."""

        return False

    def policy_gradient_step(self, rollouts: list[PolicyRolloutSample], learning_rate: float) -> dict[str, float]:
        """Run one weighted policy-gradient update."""

        raise NotImplementedError("This backend does not support policy-gradient updates")


class HuggingFaceCausalLMBackend(BaseLLMBackend):
    """Real Hugging Face causal LM backend with optional LoRA adapters."""

    def __init__(self, config: dict[str, Any], seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.device = "cpu"

    def _lazy_imports(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Hugging Face backend requires 'torch' and 'transformers'. "
                "Install them to use model.backend='huggingface'."
            ) from exc

        peft_symbols = {}
        if self.config.get("adapter", {}).get("enabled", False):
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "LoRA adapter support requires 'peft'. Install it to use adapter-style fine-tuning."
                ) from exc
            peft_symbols = {
                "LoraConfig": LoraConfig,
                "TaskType": TaskType,
                "get_peft_model": get_peft_model,
            }

        return torch, AutoModelForCausalLM, AutoTokenizer, peft_symbols

    def _resolve_dtype(self, torch_module):
        dtype_name = self.config.get("torch_dtype")
        if not dtype_name or dtype_name == "auto":
            return "auto"
        if not hasattr(torch_module, dtype_name):
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        return getattr(torch_module, dtype_name)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        torch, AutoModelForCausalLM, AutoTokenizer, peft_symbols = self._lazy_imports()
        model_name = self.config["pretrained_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.get("trust_remote_code", False),
            use_fast=self.config.get("use_fast_tokenizer", True),
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = self.config.get("padding_side", "left")

        model_kwargs = {
            "trust_remote_code": self.config.get("trust_remote_code", False),
        }
        dtype = self._resolve_dtype(torch)
        if dtype != "auto":
            model_kwargs["dtype"] = dtype
        elif self.config.get("torch_dtype") == "auto":
            model_kwargs["dtype"] = "auto"

        if self.config.get("device_map"):
            model_kwargs["device_map"] = self.config["device_map"]
        if self.config.get("low_cpu_mem_usage") is not None:
            model_kwargs["low_cpu_mem_usage"] = self.config["low_cpu_mem_usage"]
        if self.config.get("attn_implementation"):
            model_kwargs["attn_implementation"] = self.config["attn_implementation"]
        if self.config.get("use_safetensors") is not None:
            model_kwargs["use_safetensors"] = self.config["use_safetensors"]

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        adapter_config = self.config.get("adapter", {})
        if adapter_config.get("enabled", False):
            if adapter_config.get("path"):
                model.load_adapter(adapter_config["path"], adapter_name=adapter_config.get("name", "edge_case_lora"))
                model.set_adapter(adapter_config.get("name", "edge_case_lora"))
            else:
                lora_config = peft_symbols["LoraConfig"](
                    task_type=peft_symbols["TaskType"].CAUSAL_LM,
                    inference_mode=False,
                    r=adapter_config.get("r", 8),
                    lora_alpha=adapter_config.get("alpha", 16),
                    lora_dropout=adapter_config.get("dropout", 0.05),
                    target_modules=adapter_config.get("target_modules"),
                    modules_to_save=adapter_config.get("modules_to_save"),
                )
                model = peft_symbols["get_peft_model"](model, lora_config)

        trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_params:
            for parameter in model.parameters():
                parameter.requires_grad = True
            trainable_params = list(model.parameters())

        device = self.config.get("device")
        if not self.config.get("device_map") and device:
            model.to(device)
            self.device = device
        elif hasattr(model, "device"):
            self.device = str(model.device)

        optimizer_name = self.config.get("optimizer", "adamw").lower()
        if optimizer_name != "adamw":
            raise ValueError(f"Unsupported optimizer for Hugging Face backend: {optimizer_name}")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(self.config.get("learning_rate", 1e-5)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

        model.train()
        self.tokenizer = tokenizer
        self.model = model
        self._loaded = True

    def _tokenize_prompt_and_completion(self, prompt: str, completion: str):
        torch, _, _, _ = self._lazy_imports()
        self._ensure_loaded()
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        completion_ids = self.tokenizer(completion, add_special_tokens=False, return_tensors="pt")
        eos_token_id = self.tokenizer.eos_token_id
        completion_tensor = completion_ids["input_ids"]
        if eos_token_id is not None:
            completion_tensor = torch.cat(
                [completion_tensor, torch.tensor([[eos_token_id]], dtype=completion_tensor.dtype)],
                dim=1,
            )
        input_ids = torch.cat([prompt_ids["input_ids"], completion_tensor], dim=1)
        attention_mask = torch.ones_like(input_ids)
        prompt_length = prompt_ids["input_ids"].shape[1]
        return input_ids, attention_mask, prompt_length

    def _completion_logprob(self, prompt: str, completion: str) -> float:
        torch, _, _, _ = self._lazy_imports()
        self._ensure_loaded()
        input_ids, attention_mask, prompt_length = self._tokenize_prompt_and_completion(prompt, completion)

        if not self.config.get("device_map") and self.device:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        completion_start = max(prompt_length - 1, 0)
        completion_logprob = token_log_probs[:, completion_start:].sum().item()
        return float(completion_logprob)

    def generate_candidates(
        self,
        example: ProblemExample,
        prompt: str,
        num_samples: int,
        max_len: int,
        temperature: float,
        top_p: float,
    ) -> list[BackendGeneration]:
        del example
        torch, _, _, _ = self._lazy_imports()
        self._ensure_loaded()
        model = self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        if not self.config.get("device_map") and self.device:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                top_p=top_p,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_len,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False,
            )

        prompt_length = inputs["input_ids"].shape[1]
        generations: list[BackendGeneration] = []
        for sequence in outputs:
            completion_ids = sequence[prompt_length:]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            canonical = canonicalize_text(text)
            if not canonical:
                continue
            logprob = self._completion_logprob(prompt, canonical)
            generations.append(BackendGeneration(text=canonical, base_score=logprob, logprob=logprob))
        self.model.train(self.model.training)
        return generations or [BackendGeneration(text="", base_score=-1e9, logprob=-1e9)]

    def supports_policy_gradient(self) -> bool:
        return True

    def policy_gradient_step(self, rollouts: list[PolicyRolloutSample], learning_rate: float) -> dict[str, float]:
        torch, _, _, _ = self._lazy_imports()
        self._ensure_loaded()
        if self.optimizer is None:
            raise RuntimeError("Optimizer was not initialized for Hugging Face backend")

        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        losses = []
        logprob_values = []
        for rollout in rollouts:
            prompt = rollout.sample.conditioning_prompt or ""
            completion = rollout.sample.candidate.canonical_text
            if not completion:
                continue
            input_ids, attention_mask, prompt_length = self._tokenize_prompt_and_completion(prompt, completion)
            if not self.config.get("device_map") and self.device:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            completion_start = max(prompt_length - 1, 0)
            completion_logprob = token_log_probs[:, completion_start:].sum()
            logprob_values.append(float(completion_logprob.detach().cpu().item()))

            advantage = torch.tensor(rollout.reward_signal, dtype=completion_logprob.dtype, device=completion_logprob.device)
            if self.config.get("normalize_loss_by_tokens", True):
                token_count = max(token_log_probs[:, completion_start:].shape[1], 1)
                completion_logprob = completion_logprob / token_count
            losses.append(-(advantage * completion_logprob))

        if not losses:
            return {"policy_loss": 0.0, "mean_completion_logprob": 0.0, "updated_samples": 0.0}

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()

        max_grad_norm = self.config.get("max_grad_norm")
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": float(loss.detach().cpu().item()),
            "mean_completion_logprob": sum(logprob_values) / max(len(logprob_values), 1),
            "updated_samples": float(len(logprob_values)),
        }

    def save_pretrained(self, output_dir: str | Path) -> None:
        self._ensure_loaded()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination)
        self.model.save_pretrained(destination)

    def describe(self) -> dict[str, Any]:
        return {
            "backend_type": "huggingface_causal_lm",
            "config": self.config,
        }


class PretrainedLLMGeneratorPolicy(BaseGeneratorPolicy):
    """LLM-style policy with swappable pretrained backends."""

    def __init__(
        self,
        backend: BaseLLMBackend,
        examples: list[ProblemExample],
        seed: int = 0,
        include_gt_code: bool = False,
    ) -> None:
        self.backend = backend
        self.examples = {example.problem_id: example for example in examples}
        self.include_gt_code = include_gt_code
        self.training = True
        self.rng = random.Random(seed)
        self.adapter_biases: dict[str, dict[str, float]] = {example.problem_id: {} for example in examples}
        self.prompt_cache = {
            example.problem_id: build_generation_prompt(example, include_gt_code=include_gt_code)
            for example in examples
        }
        self.latest_action_metadata: dict[str, dict[str, dict[str, Any]]] = {example.problem_id: {} for example in examples}

    @classmethod
    def from_examples(
        cls,
        examples: list[ProblemExample],
        backend_name: str = "huggingface",
        seed: int = 0,
        include_gt_code: bool = False,
        backend_config: dict[str, Any] | None = None,
    ) -> "PretrainedLLMGeneratorPolicy":
        """Construct a policy from dataset examples and backend choice."""

        if backend_name == "huggingface":
            backend = HuggingFaceCausalLMBackend(config=backend_config or {}, seed=seed)
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
        return cls(backend=backend, examples=examples, seed=seed, include_gt_code=include_gt_code)

    def _proposal_distribution(
        self,
        example: ProblemExample,
        num_samples: int,
        max_len: int,
        temperature: float,
        top_p: float,
    ) -> tuple[list[tuple[str, float]], dict[str, str]]:
        prompt = self.prompt_cache[example.problem_id]
        proposals = self.backend.generate_candidates(
            example=example,
            prompt=prompt,
            num_samples=max(num_samples, 1),
            max_len=max_len,
            temperature=temperature,
            top_p=top_p,
        )
        scored: dict[str, float] = {}
        action_texts: dict[str, str] = {}
        metadata = self.latest_action_metadata.setdefault(example.problem_id, {})
        metadata.clear()
        for proposal in proposals:
            canonical = canonicalize_text(proposal.text[:max_len])
            key = candidate_hash(canonical)
            adapter_bias = self.adapter_biases[example.problem_id].get(key, 0.0)
            score = proposal.base_score + adapter_bias
            scored[key] = max(scored.get(key, float("-inf")), score)
            action_texts[key] = canonical
            metadata[key] = {
                "text": canonical,
                "prompt": prompt,
                "backend_logprob": proposal.logprob,
            }

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        logits = [score for _, score in ranked]
        if not logits:
            return [], {}
        max_logit = max(logits)
        exp_values = [math.exp((logit - max_logit) / max(temperature, 1e-6)) for logit in logits]
        total = sum(exp_values)
        probabilities = [value / total for value in exp_values]

        kept: list[tuple[str, float]] = []
        cumulative = 0.0
        for (candidate_key, _), probability in zip(ranked, probabilities):
            kept.append((candidate_key, probability))
            cumulative += probability
            if cumulative >= top_p:
                break
        renorm = sum(probability for _, probability in kept)
        normalized = [(candidate_key, probability / renorm) for candidate_key, probability in kept]
        for candidate_key, probability in normalized:
            metadata[candidate_key]["policy_probability"] = probability
            metadata[candidate_key]["policy_logprob"] = math.log(max(probability, 1e-12))
        return normalized, action_texts

    def sample_candidates(
        self,
        example: ProblemExample,
        num_samples: int,
        temperature: float,
        top_p: float,
        max_len: int,
    ) -> list[SampledCandidate]:
        distribution, action_texts = self._proposal_distribution(
            example=example,
            num_samples=num_samples,
            max_len=max_len,
            temperature=temperature,
            top_p=top_p,
        )
        if not distribution:
            candidate = build_candidate("", example)
            return [
                SampledCandidate(
                    problem_id=example.problem_id,
                    action_id=candidate.candidate_hash,
                    candidate=candidate,
                    logprob=0.0,
                    probability=1.0,
                    group_index=0,
                    conditioning_prompt=self.prompt_cache[example.problem_id],
                )
            ]

        action_ids = [action_id for action_id, _ in distribution]
        weights = [probability for _, probability in distribution]
        probability_map = dict(distribution)

        samples: list[SampledCandidate] = []
        for group_index in range(num_samples):
            chosen_action = self.rng.choices(action_ids, weights=weights, k=1)[0]
            raw_text = action_texts.get(chosen_action, "")
            candidate = build_candidate(raw_text, example)
            probability = probability_map[chosen_action]
            metadata = self.latest_action_metadata[example.problem_id].get(chosen_action, {})
            samples.append(
                SampledCandidate(
                    problem_id=example.problem_id,
                    action_id=chosen_action,
                    candidate=candidate,
                    logprob=metadata.get("policy_logprob", math.log(max(probability, 1e-12))),
                    probability=probability,
                    group_index=group_index,
                    conditioning_prompt=metadata.get("prompt"),
                )
            )
        return samples

    def logprob(self, problem_id: str, action_id: str) -> float:
        metadata = self.latest_action_metadata.get(problem_id, {}).get(action_id, {})
        probability = metadata.get("policy_probability")
        if probability is not None:
            return math.log(max(probability, 1e-12))
        backend_logprob = metadata.get("backend_logprob")
        if backend_logprob is not None:
            return float(backend_logprob)
        return math.log(1e-12)

    def update(self, problem_id: str, action_rewards: list[tuple[str, float]], learning_rate: float) -> None:
        biases = self.adapter_biases.setdefault(problem_id, {})
        for action_id, reward in action_rewards:
            biases[action_id] = biases.get(action_id, 0.0) + learning_rate * reward

    def supports_rollout_updates(self) -> bool:
        return self.backend.supports_policy_gradient()

    def update_from_rollouts(self, rollouts: list[PolicyRolloutSample], learning_rate: float) -> dict[str, float] | None:
        if not self.backend.supports_policy_gradient():
            return None
        return self.backend.policy_gradient_step(rollouts, learning_rate=learning_rate)

    def save(self, path: str) -> None:
        payload = {
            "backend": self.backend.describe(),
            "adapter_biases": self.adapter_biases,
            "examples": [example.to_dict() for example in self.examples.values()],
            "include_gt_code": self.include_gt_code,
            "training": self.training,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        if isinstance(self.backend, HuggingFaceCausalLMBackend):
            hf_dir = destination.with_suffix("")
            hf_dir = hf_dir.parent / f"{hf_dir.name}_hf"
            self.backend.save_pretrained(hf_dir)

    @classmethod
    def load(cls, path: str) -> "PretrainedLLMGeneratorPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        backend = payload["backend"]
        examples = [ProblemExample(**example) for example in payload.get("examples", [])]
        backend_type = backend.get("backend_type")
        if backend_type == "huggingface_causal_lm":
            model = cls(
                backend=HuggingFaceCausalLMBackend(config=backend.get("config", {})),
                examples=examples,
                include_gt_code=payload.get("include_gt_code", False),
            )
        else:
            raise ValueError(f"Unsupported backend type in checkpoint: {backend_type}")
        model.adapter_biases = payload.get("adapter_biases", {})
        model.training = payload.get("training", True)
        return model

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


if __name__ == "__main__":
    raise SystemExit(
        "llm_policy.py is a library module, not a standalone entry point. "
        "Run the package scripts instead, for example: "
        "`python3 -m edge_case_generator.scripts.train_edge_case_generator --config edge_case_generator/configs/debug.yaml`."
    )
