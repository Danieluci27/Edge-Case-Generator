"""Minimal Hugging Face runtime diagnostic for local segmentation faults."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from edge_case_generator.config import load_config


def main() -> None:
    """Import and load the configured HF stack step by step."""

    config = load_config("edge_case_generator/configs/hf_lora_debug.yaml")
    hf_config = config["model"]["huggingface"]
    model_name = hf_config["pretrained_model_name_or_path"]

    print("step=python")

    import torch

    print(f"step=torch version={torch.__version__}")

    import transformers

    print(f"step=transformers version={transformers.__version__}")

    import peft

    print(f"step=peft version={peft.__version__}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("step=imports_ok")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=hf_config.get("trust_remote_code", False),
        use_fast=hf_config.get("use_fast_tokenizer", True),
    )
    print(f"step=tokenizer_loaded model={model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=hf_config.get("trust_remote_code", False),
        use_safetensors=hf_config.get("use_safetensors", True),
        low_cpu_mem_usage=hf_config.get("low_cpu_mem_usage", False),
    )
    print(f"step=model_loaded class={model.__class__.__name__}")

    sample = tokenizer("Generate compact edge-case inputs.", return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **sample,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=8,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"step=generation_ok output={decoded!r}")


if __name__ == "__main__":
    main()
