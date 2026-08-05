#!/usr/bin/env bash
set -euo pipefail
cd /data/ruida/LLM/MedGemma1.5_DAM
python train_medgemma_deeplesion_dam.py \
  --model_name /data/ruida/LLM/MedGemma1.5/medgemma-1.5-4b-it \
  --annotations_json /data/ruida/LLM/R2Gen-DAM/data/generated_annotations/deeplesion_dam_segmask/deeplesion_oracle_report_segmask_bbox.json \
  --image_root /data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D \
  --output_dir outputs/medgemma15_deeplesion_dam_four_image \
  --visual_mode four_image --crop_scale 3.0 --min_crop_size 48 \
  --bf16 --tf32 --use_4bit --gradient_checkpointing --local_files_only \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 --learning_rate 1e-4 --num_train_epochs 3 \
  --logging_steps 10 --save_steps 500 --eval_steps 500 --dataloader_num_workers 0
