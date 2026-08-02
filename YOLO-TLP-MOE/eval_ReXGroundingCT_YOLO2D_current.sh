#!/bin/bash
set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

cd /data/ruida/object_detection/YOLO-TLP-MOE

MODEL=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/ReXGroundingCT_YOLO2D_tlp_esmoe_p5_gray/weights/best.pt
DATA=/data/ruida/data/CT-RATE/ReXGroundingCT_YOLO2D/rexgroundingct_yolo2d.yaml

echo "MODEL: ${MODEL}"
echo "DATA : ${DATA}"

ls -lh "${MODEL}"
cat "${DATA}"

yolo detect val \
  model="${MODEL}" \
  data="${DATA}" \
  split=val \
  imgsz=512 \
  batch=8 \
  device=0 \
  workers=2 \
  iou=0.7 \
  conf=0.05 \
  max_det=20 \
  project=yoloTLP_runs_eval \
  name=ReXGroundingCT_YOLO2D_val
