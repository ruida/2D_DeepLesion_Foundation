#!/bin/bash
set -euo pipefail
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba
ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

PROJECT=${PROJECT:-/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text}
cd "$PROJECT"
ANN=${ANN:-data/oracle_pair_rag_annotations/02_pair_rag_oracle_style_r2gen.json}
IMAGE_DIR=${IMAGE_DIR:-/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear}
# Set this to the checkpoint from your high-performing oracle anatomy+lesion model.
CHECKPOINT=${CHECKPOINT:-results/REPLACE_WITH_ORACLE_MODEL/current_checkpoint.pth}
SAVE_DIR=${SAVE_DIR:-results/test_oracle_model_with_pair_rag}
LOG=${LOG:-logs/test_oracle_model_with_pair_rag.log}

if [[ "$CHECKPOINT" == *REPLACE_WITH_ORACLE_MODEL* ]] || [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: set CHECKPOINT to your oracle anatomy+lesion checkpoint."
  exit 1
fi
mkdir -p "$SAVE_DIR" logs

python main_test.py \
  --image_dir "$IMAGE_DIR" \
  --ann_path "$ANN" \
  --dataset_name deeplesion \
  --max_seq_length 64 \
  --threshold 3 \
  --batch_size 16 \
  --save_dir "$SAVE_DIR" \
  --load "$CHECKPOINT" \
  --seed 9223 \
  --num_workers 4 \
  --anatomy_source rough \
  --anatomy_encoding text \
  --num_anatomy 14 \
  --max_anatomy 12 \
  --max_boxes 2 \
  2>&1 | tee "$LOG"
