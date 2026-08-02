#!/bin/bash
set -euo pipefail

cd /vf/users/ruida/segmentation/Swin-UMamba-DeepLesion-Clean

source /data/ruida/conda2/etc/profile.d/conda.sh
set +u
conda activate acl_umamba
set -u

ml CUDA/12.1
ml gcc/11.3.0

BASE=data/sam_filtered_by_visdrone_split
IMAGE_DIR=${BASE}/test-dev/images

YOLO_LABEL_DIR=/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION-HEAD/predict_runs/DeepLesion_clear2_testdev_best_pred/labels

echo "Creating top-1 ROI crops..."
rm -rf roi_outputs_top1
mkdir -p roi_outputs_top1/imagesTs

python make_roi_inference_crops_from_yolo_txt.py \
  --images_dir ${IMAGE_DIR} \
  --yolo_labels_dir ${YOLO_LABEL_DIR} \
  --out_imagesTs roi_outputs_top1/imagesTs \
  --mapping_json roi_outputs_top1/mapping.json \
  --conf_th 0.05 \
  --iou_nms 0.7 \
  --margin 16 \
  --min_box_size 8 \
  --max_rois_per_image 1 \
  --image_ext .png

echo "Creating top-2 ROI crops..."
rm -rf roi_outputs_top2
mkdir -p roi_outputs_top2/imagesTs

python make_roi_inference_crops_from_yolo_txt.py \
  --images_dir ${IMAGE_DIR} \
  --yolo_labels_dir ${YOLO_LABEL_DIR} \
  --out_imagesTs roi_outputs_top2/imagesTs \
  --mapping_json roi_outputs_top2/mapping.json \
  --conf_th 0.05 \
  --iou_nms 0.7 \
  --margin 16 \
  --min_box_size 8 \
  --max_rois_per_image 2 \
  --image_ext .png

echo "Top-1 ROI crop count:"
find roi_outputs_top1/imagesTs -name "*.png" | wc -l

echo "Top-2 ROI crop count:"
find roi_outputs_top2/imagesTs -name "*.png" | wc -l

echo "Done."
