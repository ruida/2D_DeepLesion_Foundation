#!/bin/bash
set -euo pipefail

cd /data/ruida/LLM/R2Gen-DAM

ORACLE=/data/ruida/LLM/R2Gen-Mamba-Merged_text/data/simple_pair_rag_annotations/04_oracle_simple_r2gen.json
DAM_JSON=data/generated_annotations/deeplesion_dam_segmask/deeplesion_oracle_report_segmask_bbox.json
CLS_JSON=data/generated_annotations/deeplesion_dam_segmask/deeplesion_dam_classifier_oracle04.json
IMAGE_DIR=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D
SAVE_DIR=results/dam_lesion_classifier_oracle04

python dam_lesion_classifier/prepare_dam_classifier_json.py \
  --oracle_json "$ORACLE" \
  --dam_json "$DAM_JSON" \
  --out_json "$CLS_JSON"

python dam_lesion_classifier/train_dam_lesion_classifier.py \
  --ann_path "$CLS_JSON" \
  --image_dir "$IMAGE_DIR" \
  --save_dir "$SAVE_DIR" \
  --backbone resnet18 \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --num_workers 4 \
  --attribute_lambda 0.5 \
  --image_size 224 \
  --crop_scale 3.0 \
  --min_crop_size 48 \
  --bbox_format xyxy \
  --adapter_dim 256 \
  --num_heads 8 \
  --patience 10 \
  --seed 9223
