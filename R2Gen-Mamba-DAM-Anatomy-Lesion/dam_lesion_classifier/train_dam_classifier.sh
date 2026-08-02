#!/bin/bash
set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

PROJECT=/data/ruida/LLM/R2Gen-Mamba-Merged_text
DAM_REPO=/data/ruida/LLM/R2Gen-DAM

export PYTHONPATH=$DAM_REPO:${PYTHONPATH:-}

cd "$PROJECT/dam_lesion_classifier"

ANN=$PROJECT/data/simple_pair_rag_annotations/12_dam_classifier_oracle04.json
IMAGE_ROOT=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D
SAVE_DIR=$PROJECT/results/dam_lesion_classifier_oracle04
LOG_DIR=$PROJECT/logs/dam_lesion_classifier
LOG=$LOG_DIR/train_dam_lesion_classifier_oracle04.log

mkdir -p "$SAVE_DIR" "$LOG_DIR"

python train_dam_lesion_classifier.py \
  --ann_path "$ANN" \
  --image_dir "$IMAGE_ROOT" \
  --save_dir "$SAVE_DIR" \
  --backbone resnet18 \
  --batch_size 16 \
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
  --seed 9223 \
  2>&1 | tee "$LOG"
