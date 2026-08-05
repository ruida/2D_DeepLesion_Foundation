#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FINETUNE_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:?Need MODEL_PATH set}"
DATA_DIR="${DEEPLESION_QWEN_JSON_DIR:?Need DEEPLESION_QWEN_JSON_DIR set}"

cd "${REPO_ROOT}"

export PYTHONPATH="${FINETUNE_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/ruida/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/ruida/hf_cache/hub
export TRITON_CACHE_DIR=/data/ruida/triton_cache
mkdir -p /data/ruida/hf_cache/hub /data/ruida/triton_cache /data/ruida/Qwen3VL_runs

python Qwen3-VL/qwen-vl-finetune/evaluation/evaluate_deeplesion.py \
  --model_path "${MODEL_PATH}" \
  --test_json "${DATA_DIR}/deeplesion_test.json" \
  --output_json /data/ruida/Qwen3VL_runs/deeplesion_dam_yolomoe_swinumamba_predictions_checkpoint16000.json \
  --max_new_tokens 64
