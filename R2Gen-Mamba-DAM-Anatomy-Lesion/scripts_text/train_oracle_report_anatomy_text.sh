#!/bin/bash

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

cd /data/ruida/LLM/R2Gen-Mamba-Merged_text

SAVE_DIR=results/step1_oracle_report_anatomy_text
ANN=data/generated_annotations/anatomy_text/deeplesion_oracle_report_anatomy_text.json

mkdir -p ${SAVE_DIR}

python main_train.py \
  --image_dir /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --ann_path ${ANN} \
  --dataset_name deeplesion \
  --max_seq_length 64 \
  --threshold 3 \
  --batch_size 16 \
  --epochs 100 \
  --save_dir ${SAVE_DIR} \
  --seed 9223 \
  --num_workers 2 \
  --anatomy_source rough \
  --anatomy_encoding text \
  --num_anatomy 14 \
  --max_anatomy 8 \
  --max_boxes 2
