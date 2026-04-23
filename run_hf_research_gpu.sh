#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

python3 train_edge_case_generator.py --config edge_case_generator/configs/hf_lora_research_gpu.yaml "$@"
