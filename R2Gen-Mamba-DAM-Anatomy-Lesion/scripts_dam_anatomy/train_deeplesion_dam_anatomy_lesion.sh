#!/bin/bash
set -euo pipefail
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion

IMAGE_DIR=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D
ANN=data/generated_annotations/dam_anatomy_lesion/deeplesion_dam_merged_anatomy_oracle_lesion.json
SAVE_DIR=results/deeplesion_dam_anatomy_lesion
LOG=logs/deeplesion_dam_anatomy_lesion_train.log
mkdir -p "$SAVE_DIR" logs

python main_train.py \
  --image_dir "$IMAGE_DIR" \
  --ann_path "$ANN" \
  --dataset_name deeplesion \
  --use_dam \
  --dam_image_size 224 \
  --dam_crop_scale 3.0 \
  --dam_min_crop_size 48 \
  --bbox_format xyxy \
  --dam_adapter_dim 512 \
  --dam_num_heads 8 \
  --dam_output_mode local \
  --anatomy_source rough \
  --anatomy_encoding text \
  --include_lesion_type \
  --num_anatomy 174 \
  --max_anatomy 20 \
  --max_boxes 1 \
  --bbox_hidden 128 \
  --max_seq_length 60 \
  --threshold 3 \
  --num_workers 2 \
  --batch_size 8 \
  --epochs 100 \
  --save_dir "$SAVE_DIR" \
  --early_stop 50 \
  --step_size 50 \
  --gamma 0.1 \
  --seed 9223 \
  2>&1 | tee "$LOG"
