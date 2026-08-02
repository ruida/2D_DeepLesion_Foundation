#!/bin/bash
set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

cd /data/ruida/LLM/R2Gen-Mamba-Merged_text

CKPT=results/oracle_simple_r2gen/model_best.pth
ANN=data/simple_pair_rag_annotations/11_anatomy_agent_plus_oracle_lesion_test_r2gen_ready.json
SAVE_DIR=results/anatomy_agent_plus_oracle_lesion_test
LOG=logs/simple_pair_rag/test_anatomy_agent_plus_oracle_lesion.log

mkdir -p "$SAVE_DIR" logs/simple_pair_rag

python main_test.py \
  --image_dir /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --ann_path "$ANN" \
  --dataset_name deeplesion \
  --max_seq_length 64 \
  --threshold 3 \
  --batch_size 16 \
  --save_dir "$SAVE_DIR" \
  --load "$CKPT" \
  --seed 9223 \
  --num_workers 4 \
  --anatomy_source rough \
  --anatomy_encoding text \
  --num_anatomy 14 \
  --max_anatomy 8 \
  --max_boxes 2 \
  2>&1 | tee "$LOG"
