#!/bin/bash

# MODEL=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe3/weights/best.pt
MODEL=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/best.pt

# If your actual run folder is DeepLesion_tlp_esmoe2 or DeepLesion_tlp_esmoe3,
# change MODEL above to that weights/best.pt path.

yolo detect val \
  model="$MODEL" \
  data=deeplesion_clear.yaml \
  split=test \
  imgsz=768 \
  batch=8 \
  device=0 \
  workers=2 \
  save_json=True \
  project=yoloTLP_runs/eval \
  name=DeepLesion_tlp_esmoe_eval \
  2>&1 | tee YOLO-TLP-MOE_eval_DeepLesion_clear.log
