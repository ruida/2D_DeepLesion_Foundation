#!/bin/bash
set -euo pipefail

cd /data/ruida/object_detection/YOLO-TLP-MOE

MODEL=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/best.pt
DATA=/data/ruida/object_detection/YOLO-TLP-MOE/deeplesion_clear.yaml

echo "========================================"
echo "MODEL: $MODEL"
echo "DATA : $DATA"
echo "========================================"

ls -lh "$MODEL"
cat "$DATA"

yolo detect val \
  model="$MODEL" \
  data="$DATA" \
  split=test \
  imgsz=768 \
  batch=8 \
  device=0 \
  workers=2 \
  save_json=True \
  project=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/eval \
  name=DeepLesion_tlp_esmoe_p54_eval \
  exist_ok=True \
  2>&1 | tee /data/ruida/object_detection/YOLO-TLP-MOE/YOLO-TLP-MOE_eval_DeepLesion_p54.log
