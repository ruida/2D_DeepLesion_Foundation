#!/bin/bash
set -euo pipefail

IMAGE_DIR=${IMAGE_DIR:-/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear}
ANN_PATH=${ANN_PATH:-data/generated_annotations/predbbox_anat_lesion_token/deeplesion_test_yolotlpmoe_p54_top2_totalseg_to_oracle_anat_generic_lesion_v3_trainval.json}
SAVE_DIR=${SAVE_DIR:-results/deeplesion_bbox_anatomy_text}

mkdir -p "${SAVE_DIR}"

python main_train.py \
  --dataset_name deeplesion \
  --image_dir "${IMAGE_DIR}" \
  --ann_path "${ANN_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --max_seq_length 64 \
  --threshold 3 \
  --batch_size 16 \
  --epochs 100 \
  --early_stop 50 \
  --seed 9223 \
  --num_workers 2 \
  --max_boxes 2 \
  --anatomy_source rough \
  --anatomy_encoding text \
  --max_anatomy 8
