# Realistic Testing Pipeline for R2Gen-Mamba-DAM-Anatomy-Lesion

This README describes the finalized realistic testing workflow for DeepLesion short radiology report generation.

```text
YOLO-TLP-MOE predicted bounding box
        ↓
Swin-UMamba predicted lesion mask
        ↓
TotalSegmentator anatomy lookup
        ↓
Precomputed mapping into oracle-style anatomy terms
        ↓
Dual-head predicted lesion type and attribute
        ↓
Conditioning fields written into the test JSON
        ↓
R2Gen-Mamba-DAM generates the report
```

No retrieval-augmented generation is performed during final report-generation inference. Anatomy mapping and lesion-type prediction are preprocessing steps, and their outputs are stored in the test JSON before `main_test.py` is run.

## 1. Project root

```bash
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba
```

Final realistic test annotation:

```text
data/generated_annotations/dam_anatomy_lesion_realistic/
deeplesion_gttrainval_yolomoe_swinumamba_predanatomy_predlesion_test.json
```

The final realistic test split contains 1,850 matched cases.

## 2. Step 1 — Predict lesion bounding boxes with YOLO-TLP-MOE

Run YOLO-TLP-MOE on the DeepLesion test images.

```bash
cd /data/ruida/object_detection/YOLO-TLP-MOE-HEAD

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

MODEL=/data/ruida/object_detection/YOLO-TLP-MOE-HEAD/yoloTLP_runs/DeepLesion_tlp_esmoe_p5_p5moe_clear/weights/best.pt
SOURCE=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images

yolo detect predict \
  model="$MODEL" \
  source="$SOURCE" \
  imgsz=768 \
  conf=0.05 \
  iou=0.7 \
  max_det=3 \
  device=0 \
  workers=2 \
  save=True \
  save_txt=True \
  save_conf=True \
  project=predict_runs \
  name=DeepLesion_clear_testdev_best_pred
```

Expected output:

```text
predict_runs/DeepLesion_clear_testdev_best_pred/labels/
```

Each output line has:

```text
class_id center_x center_y width height confidence
```

These boxes are used for crop extraction, segmentation, anatomy lookup, and lesion-type classification.

## 3. Step 2 — Predict lesion masks with Swin-UMamba

The segmentation workflow is:

```text
YOLO predicted box
        ↓
ROI crop
        ↓
Swin-UMamba segmentation
        ↓
Predicted ROI mask
        ↓
Stitch mask into original image space
```

Typical scripts:

```text
make_roi_inference_crops_tight.py
stitch_roi_masks.py
```

Example stitching command:

```bash
python stitch_roi_masks.py \
  --mapping_json /data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs/roi_infer/mapping.json \
  --pred_dir /data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs/roi_infer/preds \
  --out_dir /data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs/full_masks \
  --binarize
```

Expected stitched masks:

```text
/data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs/full_masks/
```

The top-1 predicted lesion mask is used for each matched realistic test case.

## 4. Step 3 — Perform TotalSegmentator anatomy lookup

For each predicted lesion box, inspect the corresponding TotalSegmentator label map and identify structures overlapping the lesion region.

```text
YOLO predicted bounding box
+ TotalSegmentator label map
        ↓
Detailed anatomical labels
        ↓
Coarse anatomical category
```

Example detailed labels:

```text
lung_upper_lobe_left
liver
kidney_left
adrenal_gland_right
vertebra_L2
```

Example coarse labels:

```text
lung
liver
kidney
adrenal
abdomen
pelvis
chest
brain_head_neck
spine
bone
unknown
```

Useful intermediate JSON fields include:

```text
totalseg_anatomy_name
totalseg_anatomy_names
totalseg_detail_names
totalseg_detail_hits
totalseg_source
```

## 5. Step 4 — Map TotalSegmentator anatomy to oracle-style anatomy

Map the raw TotalSegmentator labels into the vocabulary used by the report-generation model:

```text
lung
liver
kidney
adrenal
abdomen
pelvis
chest
brain_head_neck
spine
bone
unknown
```

Example:

```text
TotalSegmentator label:
lung_upper_lobe_left

Mapped oracle-style anatomy:
lung
```

Typical mapping fields:

```text
predicted_oracle_anatomy_name
anatomy_mapping_source
anatomy_mapping_detail
anatomy_mapping_train_count
anatomy_mapping_train_purity
safe_anatomy_mapping_source
safe_mapping_accepted
```

This is a preprocessing mapping step. It is not runtime RAG.

## 6. Step 5 — Predict lesion type and attribute

Run the dual-head lesion classifier on lesion-centered crops.

Primary lesion classes:

```text
lymph_node
mass
nodule
opacity
consolidation
cystic
other_lesion
```

Attribute classes:

```text
low_density
enhancing
hyperdense
calcified
none
```

Prediction command:

```bash
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion

python simple_dualhead_lesion_classifier/02_predict_dualhead_classifier.py \
  --dataset_json data/simple_dualhead_classifier/dualhead_crop_dataset.json \
  --image_root /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --checkpoint results/simple_dualhead_classifier/model_best.pth \
  --out_json data/simple_dualhead_classifier/dualhead_predictions.json \
  --batch_size 64 \
  --num_workers 2
```

Evaluate:

```bash
python simple_dualhead_lesion_classifier/03_evaluate_dualhead_predictions.py \
  --pred_json data/simple_dualhead_classifier/dualhead_predictions.json
```

Output fields include:

```text
predicted_primary_lesion_type
predicted_primary_confidence
predicted_lesion_attribute
predicted_attribute_confidence
```

Previous reproduced test results:

```text
Primary lesion accuracy: 57.55%
Attribute accuracy:      79.71%
Joint accuracy:          49.16%
```

## 7. Step 6 — Build the realistic test JSON

Merge the predicted inputs by case ID:

```text
YOLO predicted bounding box
Swin-UMamba predicted mask
Mapped oracle-style anatomy
Predicted primary lesion type
Predicted lesion attribute
```

The final test entries should contain fields such as:

```text
id
image_path
report
bboxes
bbox_format
bbox_source
mask_path
mask_source
predicted_oracle_anatomy_name
predicted_primary_lesion_type
predicted_lesion_attribute
```

Final annotation:

```text
data/generated_annotations/dam_anatomy_lesion_realistic/
deeplesion_gttrainval_yolomoe_swinumamba_predanatomy_predlesion_test.json
```

Scenario definition:

```text
Training:
ground-truth lesion mask
ground-truth bounding box
ground-truth anatomy
ground-truth lesion type
ground-truth report target

Validation:
ground-truth lesion mask
ground-truth bounding box
ground-truth anatomy
ground-truth lesion type
ground-truth report target

Realistic testing:
YOLO-TLP-MOE predicted bounding box
Swin-UMamba predicted lesion mask
mapped predicted anatomy
dual-head predicted lesion type and attribute
ground-truth report used only as the evaluation reference
```

Verify split sizes:

```bash
python - <<'PY'
import json

p = (
    "data/generated_annotations/dam_anatomy_lesion_realistic/"
    "deeplesion_gttrainval_yolomoe_swinumamba_predanatomy_predlesion_test.json"
)

with open(p, "r") as f:
    data = json.load(f)

for split in ["train", "val", "test"]:
    print(split, len(data[split]))
PY
```

Expected realistic test count:

```text
test 1850
```

## 8. Step 7 — Run R2Gen-Mamba-DAM realistic testing

Final testing script:

```text
scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh
```

Run:

```bash
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion

bash scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh
```

The model receives:

```text
CT image
predicted lesion mask
predicted bounding box
predicted anatomy conditioning
predicted lesion-type conditioning
```

The model generates a short report autoregressively.

No ground-truth localization or semantic conditioning is supplied to the model during realistic testing.

No nearest-neighbor retrieval, pair retrieval, or report retrieval is performed by `main_test.py`.

## 9. Step 8 — Evaluate generated reports

Compare generated reports with the ground-truth short reports after inference.

Typical metrics:

```text
BLEU-1
BLEU-2
BLEU-3
BLEU-4
METEOR
ROUGE-L
Precision
Recall
F1
```

The ground-truth report is used only as the evaluation reference.

## 10. End-to-end checklist

Check YOLO checkpoint:

```bash
ls -lh \
/data/ruida/object_detection/YOLO-TLP-MOE-HEAD/yoloTLP_runs/DeepLesion_tlp_esmoe_p5_p5moe_clear/weights/best.pt
```

Check predicted masks:

```bash
ls -ld \
/data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs/full_masks
```

Check dual-head checkpoint:

```bash
ls -lh \
results/simple_dualhead_classifier/model_best.pth
```

Check final realistic JSON:

```bash
ls -lh \
data/generated_annotations/dam_anatomy_lesion_realistic/deeplesion_gttrainval_yolomoe_swinumamba_predanatomy_predlesion_test.json
```

Check realistic test size:

```bash
python - <<'PY'
import json

p = (
    "data/generated_annotations/dam_anatomy_lesion_realistic/"
    "deeplesion_gttrainval_yolomoe_swinumamba_predanatomy_predlesion_test.json"
)

data = json.load(open(p))
print("Realistic test cases:", len(data["test"]))
PY
```

Run testing:

```bash
bash scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh
```

## Important distinction

```text
TotalSegmentator anatomy
        ↓
oracle-style anatomy mapping
```

is a preprocessing transformation, not retrieval-augmented generation.

The final R2Gen-Mamba-DAM testing runtime reads prepared anatomy and lesion-type fields directly from the JSON and generates the report without RAG.
