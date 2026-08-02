# YOLO-TLP-MOE for DeepLesion Detection

This is a simplified README for training and testing YOLO-TLP-MOE on the DeepLesion/VisDrone-style dataset.

---

## 1. Goal

YOLO-TLP-MOE is used for 2D tiny lesion detection on DeepLesion images.

The basic workflow is:

```text
DeepLesion images + YOLO labels
        ↓
YOLO-TLP-MOE training
        ↓
test-dev prediction
        ↓
YOLO txt predicted boxes
        ↓
optional downstream Swin-UMamba ROI crop segmentation
```

---

## 2. Environment Setup

Go to the YOLO-TLP-MOE repo:

```bash
cd /data/ruida/object_detection/YOLO-TLP-MOE
```

Activate conda environment:

```bash
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe
```

Load CUDA and GCC if needed:

```bash
ml CUDA/12.1
ml gcc/11.3.0
```

Check environment:

```bash
which python
python --version

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

---

## 3. Dataset Layout

The DeepLesion detection dataset follows YOLO format:

```text
VisDroneDeepLesion_clear/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-test-dev/
│   ├── images/
│   └── labels/
└── split_membership/
```

Example image and label names:

```text
images/000001_01_01_109.png
labels/000001_01_01_109.txt

images/000001_01_01_109_000.png
labels/000001_01_01_109_000.txt
```

Each YOLO label file contains:

```text
class_id x_center y_center width height
```

For DeepLesion:

```text
0 lesion
```

---

## 4. Dataset YAML

Create or check:

```bash
data/VisDroneDeepLesion_clear.yaml
```

Example:

```yaml
path: /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear

train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/images

names:
  0: lesion
```

Check files:

```bash
ls /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-train/images | head
ls /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-train/labels | head

ls /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-val/images | head
ls /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-val/labels | head

ls /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images | head
ls /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/labels | head
```

---

## 5. Model Config

Example YOLO-TLP-MOE model config:

```bash
ultralytics/cfg/models/v8/yoloTLP_esmoe_p5.yaml
```

Check available model configs:

```bash
find ultralytics/cfg/models -name "*tlp*" -o -name "*moe*" -o -name "*esmoe*"
```

---

## 6. Train

Basic training command:

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/yoloTLP_esmoe_p5.yaml \
  data=data/VisDroneDeepLesion_clear.yaml \
  imgsz=768 \
  batch=8 \
  epochs=300 \
  workers=8 \
  device=0 \
  project=yoloTLP_runs \
  name=DeepLesion_tlp_esmoe_p54
```

Expected output:

```text
yoloTLP_runs/DeepLesion_tlp_esmoe_p54/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
└── args.yaml
```

Resume training:

```bash
yolo detect train \
  model=yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/last.pt \
  resume=True
```

---

## 7. Validate

```bash
yolo detect val \
  model=yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/best.pt \
  data=data/VisDroneDeepLesion_clear.yaml \
  imgsz=768 \
  batch=8 \
  device=0 \
  split=val \
  project=yoloTLP_runs \
  name=DeepLesion_tlp_esmoe_p54_val
```

Main metrics:

```text
Precision
Recall
mAP50
mAP50-95
```

---

## 8. Predict on Test-dev

```bash
yolo detect predict \
  model=yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/best.pt \
  source=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images \
  imgsz=768 \
  conf=0.05 \
  iou=0.7 \
  save=True \
  save_txt=True \
  save_conf=True \
  device=0 \
  project=yoloTLP_runs \
  name=DeepLesion_tlp_esmoe_p54_testdev_pred
```

Prediction output:

```text
yoloTLP_runs/DeepLesion_tlp_esmoe_p54_testdev_pred/
├── labels/
│   ├── 000004_03_02_136.txt
│   └── ...
└── predicted images
```

Each predicted txt file contains:

```text
class_id x_center y_center width height confidence
```

---

## 9. Evaluate on Test-dev

If test-dev labels are available, run:

```bash
yolo detect val \
  model=yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/best.pt \
  data=data/VisDroneDeepLesion_clear.yaml \
  imgsz=768 \
  batch=8 \
  device=0 \
  split=test \
  project=yoloTLP_runs \
  name=DeepLesion_tlp_esmoe_p54_testdev_eval
```

Example reported performance from the DeepLesion experiment:

```text
mAP50:    70.1%
mAP50-95: 46.4%
```

---

## 10. Use YOLO Predictions for Swin-UMamba ROI Segmentation

The predicted YOLO txt files can be used to generate ROI crops for Swin-UMamba.

Example:

```bash
YOLO_LABEL_DIR=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe_p54_testdev_pred/labels
IMAGE_DIR=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images

cd /data/ruida/segmentation/Swin-UMamba-DeepLesion-Clean
```

Top-1 ROI crop generation:

```bash
python make_roi_inference_crops_from_yolo_txt.py \
  --images_dir $IMAGE_DIR \
  --yolo_labels_dir $YOLO_LABEL_DIR \
  --out_imagesTs roi_outputs_top1/imagesTs \
  --mapping_json roi_outputs_top1/mapping.json \
  --conf_th 0.05 \
  --iou_nms 0.7 \
  --margin 16 \
  --min_box_size 8 \
  --max_rois_per_image 1 \
  --image_ext .png
```

Top-2 ROI crop generation:

```bash
python make_roi_inference_crops_from_yolo_txt.py \
  --images_dir $IMAGE_DIR \
  --yolo_labels_dir $YOLO_LABEL_DIR \
  --out_imagesTs roi_outputs_top2/imagesTs \
  --mapping_json roi_outputs_top2/mapping.json \
  --conf_th 0.05 \
  --iou_nms 0.7 \
  --margin 16 \
  --min_box_size 8 \
  --max_rois_per_image 2 \
  --image_ext .png
```

The crop coordinates are saved in:

```text
roi_outputs_top1/mapping.json
roi_outputs_top2/mapping.json
```

---

## 11. Useful Checks

Check number of prediction label files:

```bash
find yoloTLP_runs/DeepLesion_tlp_esmoe_p54_testdev_pred/labels -name "*.txt" | wc -l
```

Check predicted box count:

```bash
python - <<'PY'
from pathlib import Path

label_dir = Path("yoloTLP_runs/DeepLesion_tlp_esmoe_p54_testdev_pred/labels")

n_files = 0
n_boxes = 0

for p in label_dir.glob("*.txt"):
    n_files += 1
    with open(p) as f:
        n_boxes += sum(1 for line in f if line.strip())

print("label files:", n_files)
print("predicted boxes:", n_boxes)
PY
```

Check trained weights:

```bash
ls -lh yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/
```

---

## 12. Notes

- Use `best.pt` for final evaluation and prediction.
- Use `last.pt` only for resuming training.
- Use a low prediction threshold such as `conf=0.05` when downstream segmentation needs high recall.
- The `_000` suffix may appear in some DeepLesion filenames. Normalize filenames when matching with SAM masks or Swin-UMamba segmentation outputs.
- Do not commit trained weights, prediction outputs, logs, or large datasets to GitHub.
