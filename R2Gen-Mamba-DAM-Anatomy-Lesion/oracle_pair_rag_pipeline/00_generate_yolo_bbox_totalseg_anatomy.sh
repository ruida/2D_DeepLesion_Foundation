#!/bin/bash
set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

cd /vf/users/ruida/LLM/R2Gen-Mamba-Merged_text

INPUT_JSON="data/generated_annotations/deeplesion_test_yolotlp_clearbbox_keep_detailed_ids_lesion_token.json"

OUT_DIR="data/oracle_pair_rag_annotations"
OUT_JSON="${OUT_DIR}/01_yolo_bbox_totalseg_raw_anatomy.json"
LOG="logs/01_generate_yolo_bbox_totalseg_raw_anatomy.log"

TRAIN_MASK_DIR="/data/ruida/object_detection/TotalSegmentator/DeepLesion_train_2d_masks_sitk"
VAL_MASK_DIR="/data/ruida/object_detection/TotalSegmentator/DeepLesion_val_2d_masks_sitk"
TEST_MASK_DIR="/data/ruida/object_detection/TotalSegmentator/DeepLesion_testdev_2d_masks_sitk"

mkdir -p "${OUT_DIR}" logs

test -f "${INPUT_JSON}" || {
    echo "ERROR: input JSON not found: ${INPUT_JSON}"
    exit 1
}

test -d "${TRAIN_MASK_DIR}" || {
    echo "ERROR: train mask directory not found: ${TRAIN_MASK_DIR}"
    exit 1
}

test -d "${VAL_MASK_DIR}" || {
    echo "ERROR: val mask directory not found: ${VAL_MASK_DIR}"
    exit 1
}

test -d "${TEST_MASK_DIR}" || {
    echo "ERROR: test mask directory not found: ${TEST_MASK_DIR}"
    exit 1
}

echo "Generating TotalSegmentator anatomy from bounding boxes"
echo "Input : ${INPUT_JSON}"
echo "Output: ${OUT_JSON}"

python build_r2gen_json_from_yolotlp_predbbox_totalseg_masks.py \
  --input_json "${INPUT_JSON}" \
  --output_json "${OUT_JSON}" \
  --train_mask_dir "${TRAIN_MASK_DIR}" \
  --val_mask_dir "${VAL_MASK_DIR}" \
  --test_mask_dir "${TEST_MASK_DIR}" \
  --bbox_format xywh_norm \
  --expand_ratio 0.25 \
  --min_label_pixels 3 \
  --use_nearest \
  --nearest_radius 30 \
  --max_anatomy_without_lesion 1 \
  --add_lesion_token \
  --lesion_token_id 13 \
  --lesion_token_name lesion \
  --max_anatomy 2 \
  --topk_debug 10 \
  2>&1 | tee "${LOG}"

echo
echo "Created:"
ls -lh "${OUT_JSON}"
