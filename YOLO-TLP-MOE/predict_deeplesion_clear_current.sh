#!/bin/bash

# MODEL=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe3/weights/best.pt
MODEL=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/best.pt
SOURCE=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images

# If your actual run folder is DeepLesion_tlp_esmoe2 or DeepLesion_tlp_esmoe3,
# change MODEL above to that weights/best.pt path.

yolo detect predict \
  model="$MODEL" \
  source="$SOURCE" \
  imgsz=768 \
  conf=0.25 \
  device=0 \
  workers=2 \
  save=True \
  save_txt=True \
  save_conf=True \
  project=yoloTLP_runs/predict \
  name=DeepLesion_tlp_esmoe_pred \
  2>&1 | tee YOLO-TLP-MOE_predict_DeepLesion_clear.log
