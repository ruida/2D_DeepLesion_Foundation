#!/bin/bash
# run_yolo_txt_roi_swinumamba_infer.sh
#
# End-to-end inference after YOLO-TLP already produced predicted label txt files:
#   1) YOLO txt labels -> tight ROI crops + mapping.json
#   2) nnUNetv2_predict with Swin-UMamba trainer
#   3) Stitch ROI predictions back to full-size 512x512 masks

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

IMAGES_DIR="/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images"
YOLO_LABELS_DIR="/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION-HEAD/predict_runs/DeepLesion_clear2_testdev_best_pred/labels"

ROI_ROOT="/data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs_top2"
ROI_IMAGES_TS="${ROI_ROOT}/roi_infer/imagesTs"
MAPPING_JSON="${ROI_ROOT}/roi_infer/mapping.json"
ROI_PREDS="${ROI_ROOT}/roi_infer/preds"
FULL_MASKS="${ROI_ROOT}/full_masks"

export nnUNet_raw="/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw"
export nnUNet_preprocessed="/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed"
export nnUNet_results="/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_results"

DATASET_ID=711
CONFIG="2d"
FOLD=all
TRAINER="nnUNetTrainerSwinUMamba"

CONF_TH=0.20
MARGIN=16
MIN_BOX_SIZE=8
MAX_ROIS_PER_IMAGE=2

mkdir -p "${ROI_IMAGES_TS}" "${ROI_PREDS}" "${FULL_MASKS}"

echo "[1/3] Creating ROI crops from YOLO txt labels..."
python make_roi_inference_crops_from_yolo_txt.py   --images_dir "${IMAGES_DIR}"   --yolo_labels_dir "${YOLO_LABELS_DIR}"   --out_imagesTs "${ROI_IMAGES_TS}"   --mapping_json "${MAPPING_JSON}"   --conf_th "${CONF_TH}"   --margin "${MARGIN}"   --min_box_size "${MIN_BOX_SIZE}"   --max_rois_per_image "${MAX_ROIS_PER_IMAGE}"

echo "[2/3] Running nnUNetv2/Swin-UMamba prediction on ROI crops..."
nnUNetv2_predict   -d "${DATASET_ID}"   -c "${CONFIG}"   -f "${FOLD}"   -i "${ROI_IMAGES_TS}"   -o "${ROI_PREDS}"   -tr "${TRAINER}" -npp 1 -nps 1 -npp 1 -nps 1

echo "[3/3] Stitching ROI masks back to full-size masks..."
python stitch_roi_masks.py   --mapping_json "${MAPPING_JSON}"   --pred_dir "${ROI_PREDS}"   --out_dir "${FULL_MASKS}"   --binarize

echo "Done."
echo "Full masks saved to: ${FULL_MASKS}"
