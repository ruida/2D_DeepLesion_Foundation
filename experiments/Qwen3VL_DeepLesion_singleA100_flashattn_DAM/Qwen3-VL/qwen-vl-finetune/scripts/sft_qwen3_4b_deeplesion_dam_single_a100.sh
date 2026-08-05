#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FINETUNE_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
DATA_DIR="${DEEPLESION_QWEN_JSON_DIR:?Need DEEPLESION_QWEN_JSON_DIR set}"

cd "${REPO_ROOT}"
export PYTHONPATH="${FINETUNE_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/ruida/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/ruida/hf_cache/hub
export TRITON_CACHE_DIR=/data/ruida/triton_cache
mkdir -p /data/ruida/hf_cache/hub /data/ruida/triton_cache

python -m qwenvl.train.train_qwen \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_use deeplesion_train \
  --output_dir /data/ruida/Qwen3VL_runs/output_deeplesion_qwen3vl_dam_four_image \
  --bf16 True \
  --num_train_epochs 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --logging_steps 1 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --report_to none \
  --tune_mm_vision True \
  --tune_mm_mlp True \
  --tune_mm_llm False \
  --gradient_checkpointing True \
  --max_pixels 262144
