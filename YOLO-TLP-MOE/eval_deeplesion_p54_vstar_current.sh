#!/bin/bash
set -euo pipefail

cd /data/ruida/object_detection/YOLO-TLP-MOE

MODEL=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/best.pt
DATA=/data/ruida/object_detection/YOLO-TLP-MOE/deeplesion_clear.yaml
SOURCE=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images
GT_DIR=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/labels

OUT_DIR=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/eval/DeepLesion_tlp_esmoe_p54_vstar_eval
PRED_DIR=${OUT_DIR}/labels
OUT_CSV=${OUT_DIR}/vstar_eval_metrics.csv
LOG=/data/ruida/object_detection/YOLO-TLP-MOE/YOLO-TLP-MOE_eval_DeepLesion_p54_vstar.log

echo "========================================"
echo "V*-style YOLO-TLP-MOE evaluation"
echo "MODEL : $MODEL"
echo "DATA  : $DATA"
echo "SOURCE: $SOURCE"
echo "GT_DIR: $GT_DIR"
echo "OUT   : $OUT_DIR"
echo "========================================"

ls -lh "$MODEL"
cat "$DATA"

mkdir -p "$OUT_DIR"

echo ""
echo "========================================"
echo "Step 1: Run V*-style inference"
echo "========================================"

python vstar_yolotlp_infer.py \
  --weights "$MODEL" \
  --source "$SOURCE" \
  --out-dir "$OUT_DIR" \
  --imgsz 768 \
  --device 0 \
  --full-conf 0.01 \
  --crop-conf 0.03 \
  --final-conf 0.10 \
  --trigger-conf 0.15 \
  --pred-iou 0.70 \
  --merge-iou 0.40 \
  --max-depth 2 \
  --max-patches 12 \
  --topk-children 2 \
  --min-patch 128 \
  --overlap 0.20 \
  --max-final-det 2 \
  --save-vis \
  2>&1 | tee "$LOG"

echo ""
echo "========================================"
echo "Step 2: Evaluate V*-style saved txt predictions"
echo "========================================"

echo "GT label count:"
find "$GT_DIR" -name "*.txt" | wc -l

echo "Prediction label count:"
find "$PRED_DIR" -name "*.txt" | wc -l

python eval_vstar_yolo_txt.py \
  --gt-dir "$GT_DIR" \
  --pred-dir "$PRED_DIR" \
  --out-csv "$OUT_CSV" \
  --conf-thr 0.10 \
  --iou-thr 0.5 \
  2>&1 | tee -a "$LOG"

echo ""
echo "========================================"
echo "Done."
echo "V* output directory:"
echo "$OUT_DIR"
echo ""
echo "V* metrics CSV:"
echo "$OUT_CSV"
echo ""
echo "Log:"
echo "$LOG"
echo "========================================"
