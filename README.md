# Edge Case Generator

This repository implements an end-to-end research pipeline for generating, verifying, rewarding, and exporting edge-case inputs for code tasks.

## What is included

- Dataset loading from JSONL with deterministic splits and debug subsetting
- Plain-text candidate representation with canonicalization, hashing, and deduplication
- A modular generator interface centered on a pre-trained-LLM-style policy that can be fine-tuned with this pipeline
- A fallback template policy for smoke tests and local debugging when you do not want to load a real LLM
- An optional real Hugging Face causal LM backend with tokenizer/model loading and LoRA-style adapter fine-tuning hooks
- Deterministic subprocess-based verification against `gt_code` and `input_code`
- Configurable reward components for verification, novelty, duplication, group distinctness, verification ratio, and length control
- REINFORCE and GRPO-style relative-advantage training
- Persistent replay buffer for accepted edge cases
- Export pipeline for augmented JSONL datasets and summary reports
- Tests and a runnable smoke experiment on a tiny synthetic dataset

## Quick start

Install in editable mode:

```bash
python -m pip install -e .[dev]
```

Run a smoke experiment:

```bash
python -m edge_case_generator.scripts.run_smoke_test
```

Run training directly:

```bash
python -m edge_case_generator.scripts.train_edge_case_generator --config edge_case_generator/configs/debug.yaml
```

For a real Hugging Face causal LM backend with LoRA-style fine-tuning hooks:

```bash
python3 -m pip install -e .[hf]
python3 -m edge_case_generator.scripts.train_edge_case_generator --config edge_case_generator/configs/hf_lora_debug.yaml
```

Export accepted edge cases:

```bash
python -m edge_case_generator.scripts.export_augmented_dataset --config edge_case_generator/configs/debug.yaml
```

Run the data-collection smoke test:

```bash
python -m edge_case_generator.scripts.run_data_collection_smoke_test --config edge_case_generator/configs/collection_debug.yaml
```

Run the collection pipeline:

```bash
python -m edge_case_generator.scripts.run_data_collection_pipeline --config edge_case_generator/configs/collection_default.yaml
```

For CodeContests, the collection config can either point at a local `json`/`jsonl` file or load the public Hugging Face dataset directly with:

```yaml
sources:
  codecontests:
    enabled: true
    hf_dataset: deepmind/code_contests
    hf_split: train
    path: null
```

This requires the `datasets` Python package to be installed.

## Project structure

```text
edge_case_generator/
  buffer/
  collection/
  configs/
  data/
  datasets/
  models/
  rewards/
  rl/
  scripts/
  tests/
  utils/
  verifier/
```

## Notes

- The real pretrained-LLM path uses the Hugging Face backend.
- The generator interface now includes a real Hugging Face causal LM backend using `AutoTokenizer`, `AutoModelForCausalLM`, `generate()`, and optional LoRA adapters through PEFT.
- The default debug configs use `template_policy` so tests and smoke runs stay lightweight and reproducible. Use `edge_case_generator/configs/hf_lora_debug.yaml` for a real model-backed run.
- Verification uses isolated subprocess execution with configurable timeouts. This is practical rather than fully secure sandboxing.
- The data-collection subsystem supports local packaged dataset files for Project CodeNet, CodeContests, and APPS. It intentionally avoids direct scraping and focuses on mismatch-only semantic edge cases where both GT and buggy programs run successfully but disagree.
