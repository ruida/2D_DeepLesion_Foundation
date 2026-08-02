#!/bin/bash
set -euo pipefail

IMAGE_DIR=${IMAGE_DIR:-/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear}
ANN_PATH=${ANN_PATH:-data/generated_annotations/predbbox_anat_lesion_token/deeplesion_test_yolotlpmoe_p54_top2_totalseg_to_oracle_anat_generic_lesion_v3_trainval.json}
CKPT=${CKPT:-results/deeplesion_bbox_anatomy_text/model_best.pth}
SAVE_DIR=${SAVE_DIR:-results/deeplesion_bbox_anatomy_text_test}

mkdir -p "${SAVE_DIR}"

python main_test.py \
  --dataset_name deeplesion \
  --image_dir "${IMAGE_DIR}" \
  --ann_path "${ANN_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --load "${CKPT}" \
  --max_seq_length 64 \
  --threshold 3 \
  --batch_size 16 \
  --num_workers 2 \
  --max_boxes 2 \
  --anatomy_source rough \
  --anatomy_encoding text \
  --max_anatomy 8
