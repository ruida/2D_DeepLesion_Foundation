#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="/data/ruida/LLM/MedGemma1.5/medgemma-1.5-4b-it"
ADAPTER_DIR="/data/ruida/LLM/MedGemma1.5/outputs/medgemma15_deeplesion_short"
ANN_JSON="/data/ruida/data/DeepLesion/DeepLesion/annotation_pretty_deeplesion_sentences_flat.json"
IMAGE_ROOT="/data/ruida/data/DeepLesion/Key_slices"
OUT_JSONL="/data/ruida/LLM/MedGemma1.5/outputs/medgemma15_deeplesion_short_test_all.jsonl"

python infer_medgemma_deeplesion_debug.py \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$ADAPTER_DIR" \
  --annotations_json "$ANN_JSON" \
  --image_root "$IMAGE_ROOT" \
  --split test \
  --output_jsonl "$OUT_JSONL" \
  --bf16 \
  --use_4bit \
  --local_files_only \
  --max_new_tokens 64 \
  --max_samples 999999

echo
echo "Saved predictions to: $OUT_JSONL"
echo "Line count:"
wc -l "$OUT_JSONL"
