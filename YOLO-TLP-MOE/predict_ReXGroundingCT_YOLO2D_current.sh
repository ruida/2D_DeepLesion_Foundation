#!/bin/bash
set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

cd /data/ruida/object_detection/YOLO-TLP-MOE

MODEL=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/ReXGroundingCT_YOLO2D_tlp_esmoe_p5_gray/weights/best.pt
SOURCE=/data/ruida/data/CT-RATE/ReXGroundingCT_YOLO2D/images/val

echo "MODEL : ${MODEL}"
echo "SOURCE: ${SOURCE}"

yolo detect predict \
  model="${MODEL}" \
  source="${SOURCE}" \
  imgsz=512 \
  conf=0.05 \
  iou=0.7 \
  max_det=20 \
  device=0 \
  save=True \
  save_txt=True \
  save_conf=True \
  project=predict_runs \
  name=ReXGroundingCT_YOLO2D_val_pred
