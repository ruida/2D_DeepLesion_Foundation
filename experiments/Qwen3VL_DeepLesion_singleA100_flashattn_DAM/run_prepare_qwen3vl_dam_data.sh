#!/usr/bin/env bash
set -euo pipefail
cd /data/ruida/LLM/Qwen3VL_DeepLesion_singleA100_flashattn_DAM

python data_conversion/convert_deeplesion_dam_to_qwen3vl.py \
  --ann_path /data/ruida/LLM/R2Gen-DAM/data/generated_annotations/deeplesion_dam_segmask/deeplesion_oracle_report_segmask_bbox.json \
  --image_root /data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D \
  --output_dir /data/ruida/data/DeepLesion/qwen3vl_dam_four_image \
  --derived_image_dir /data/ruida/data/DeepLesion/qwen3vl_dam_four_image/derived_images \
  --visual_mode four_image \
  --crop_scale 3.0 \
  --min_crop_size 48
