#!/usr/bin/env bash
set -euo pipefail
cd /data/ruida/LLM/MedGemma1.5_DAM
python infer_medgemma_deeplesion_dam.py \
  --base_model /data/ruida/LLM/MedGemma1.5/medgemma-1.5-4b-it \
  --adapter_dir outputs/medgemma15_deeplesion_dam_four_image \
  --annotations_json /data/ruida/LLM/R2Gen-DAM/data/generated_annotations/deeplesion_dam_predmask_test/deeplesion_gttrainval_yolo_swinumamba_test.json \
  --image_root /data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D \
  --split test --output_jsonl outputs/medgemma15_dam_predmask_test.jsonl \
  --visual_mode four_image --crop_scale 3.0 --min_crop_size 48 \
  --bf16 --use_4bit --local_files_only --max_new_tokens 64
