#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------
# Build Dataset711_DeepLesionCrop from DeepLesion full images, masks,
# and report.json bounding boxes.
#
# Run:
#   bash make_dataset711_deeplesion_crop.sh
# ---------------------------------------------------------------------

cd /data/ruida/segmentation/Swin-UMamba-DeepLesion-Clean

# nnU-Net paths
export nnUNet_raw=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw
export nnUNet_preprocessed=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed
export nnUNet_results=/data/ruida/segmentation/Swin-UMamba-Origin/results

# Input data paths
IMAGES_DIR=/data/ruida/SAM/sam3/data/images_filtered
LABELS_DIR=/data/ruida/SAM/sam3/data/labels_filtered
REPORT_JSON=/data/ruida/SAM/sam3/data/report.json

# Dataset settings
DATASET_ID=711
DATASET_NAME=DeepLesionCrop
SPLIT=all
MARGIN=16
MIN_BOX_SIZE=8
MAX_ROIS_PER_IMAGE=1

echo "============================================================"
echo "Building nnU-Net ROI crop dataset"
echo "Repo:        $(pwd)"
echo "Dataset:     Dataset${DATASET_ID}_${DATASET_NAME}"
echo "Images:      ${IMAGES_DIR}"
echo "Labels:      ${LABELS_DIR}"
echo "Report JSON: ${REPORT_JSON}"
echo "nnUNet_raw:  ${nnUNet_raw}"
echo "Split:       ${SPLIT}"
echo "Margin:      ${MARGIN}"
echo "Max ROIs:    ${MAX_ROIS_PER_IMAGE}"
echo "============================================================"

python roi_tight_cropping_scripts/roi_scripts/make_roi_dataset_tight.py \
  --dataset_id "${DATASET_ID}" \
  --dataset_name "${DATASET_NAME}" \
  --images_dir "${IMAGES_DIR}" \
  --labels_dir "${LABELS_DIR}" \
  --report_json "${REPORT_JSON}" \
  --nnunet_raw "${nnUNet_raw}" \
  --split "${SPLIT}" \
  --margin "${MARGIN}" \
  --min_box_size "${MIN_BOX_SIZE}" \
  --max_rois_per_image "${MAX_ROIS_PER_IMAGE}"

echo
echo "Done."
echo "Output dataset:"
echo "${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}"

echo
echo "Checking output:"
ls -lh "${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}" || true
find "${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}" -maxdepth 2 -type f | head -20 || true
