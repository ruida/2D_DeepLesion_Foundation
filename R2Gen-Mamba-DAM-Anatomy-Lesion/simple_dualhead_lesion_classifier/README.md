# Dual-Head Lesion Classifier

This directory contains the lesion-type classifier used in the
`R2Gen-Mamba-DAM-Anatomy-Lesion` pipeline.

The classifier uses one shared ImageNet-pretrained ResNet backbone and two
classification heads:

- **Primary lesion head**: predicts the main lesion category.
- **Attribute head**: predicts a complementary lesion attribute.

The intended workflow is:

```text
Build classifier dataset
        ↓
Train dual-head classifier
        ↓
Select model_best.pth
        ↓
Run prediction
        ↓
Evaluate primary, attribute, and joint accuracy
        ↓
Optionally merge predictions into the older RAG-input JSON
```

---

## 1. Project location

Run all commands from:

```bash
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion
```

Activate the environment:

```bash
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba
```

On the cluster, the SLURM scripts load:

```bash
ml purge
ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12
```

The classifier scripts are located in:

```text
simple_dualhead_lesion_classifier/
```

---

## 2. Directory contents

```text
simple_dualhead_lesion_classifier/
├── 00_build_dualhead_dataset.py
├── 01_train_dualhead_classifier.py
├── 02_predict_dualhead_classifier.py
├── 03_evaluate_dualhead_predictions.py
├── 04_merge_dualhead_predictions_into_rag_input.py
├── train_dualhead_classifier.slurm
├── predict_dualhead_classifier.slurm
└── README.md
```

---

## 3. Required data and output paths

Classifier dataset:

```text
data/simple_dualhead_classifier/dualhead_crop_dataset.json
```

Dataset summary:

```text
data/simple_dualhead_classifier/dualhead_crop_dataset_summary.json
```

Image root:

```text
/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear
```

Best checkpoint:

```text
results/simple_dualhead_classifier/model_best.pth
```

Prediction output:

```text
data/simple_dualhead_classifier/dualhead_predictions.json
```

Optional RAG merge input:

```text
data/oracle_pair_rag_annotations/
02_totalseg_plus_predicted_lesion_complete.json
```

Optional RAG merge output:

```text
data/simple_pair_rag_annotations/
01b_raw_simple_inputs_dualhead.json
```

Check the required files:

```bash
ls -lh data/simple_dualhead_classifier/dualhead_crop_dataset.json
ls -ld /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear
```

If a trained model already exists:

```bash
ls -lh results/simple_dualhead_classifier/model_best.pth
```

---

# Step 1: Build the classifier dataset

Run this step when the classifier dataset must be regenerated.

```bash
python simple_dualhead_lesion_classifier/00_build_dualhead_dataset.py
```

Expected outputs:

```text
data/simple_dualhead_classifier/dualhead_crop_dataset.json
data/simple_dualhead_classifier/dualhead_crop_dataset_summary.json
```

Inspect the generated dataset:

```bash
python - <<'PY'
import json

path = "data/simple_dualhead_classifier/dualhead_crop_dataset.json"

with open(path, "r") as f:
    data = json.load(f)

for split in ["train", "val", "test"]:
    print(split, len(data.get(split, [])))
    if data.get(split):
        print(json.dumps(data[split][0], indent=2))
PY
```

Each case should contain fields such as:

```text
id
image_path
bboxes
bbox_format
primary_label
attribute_label
report
```

The classifier crop is generated from the lesion bounding box, enlarged by
25%, and resized to 224 × 224 pixels.

---

# Step 2: Train the classifier

## Recommended: SLURM training

Check the SLURM script:

```bash
cat simple_dualhead_lesion_classifier/train_dualhead_classifier.slurm
```

Validate its syntax:

```bash
bash -n simple_dualhead_lesion_classifier/train_dualhead_classifier.slurm
```

Submit the training job:

```bash
mkdir -p slurm_logs logs/simple_dualhead_classifier
sbatch simple_dualhead_lesion_classifier/train_dualhead_classifier.slurm
```

Check job status:

```bash
squeue -u "$USER"
```

Monitor the training log:

```bash
tail -f logs/simple_dualhead_classifier/train.log
```

The training job saves:

```text
results/simple_dualhead_classifier/current_checkpoint.pth
results/simple_dualhead_classifier/model_best.pth
```

Use this checkpoint for inference:

```text
results/simple_dualhead_classifier/model_best.pth
```

## Direct training

Check the available arguments:

```bash
python simple_dualhead_lesion_classifier/01_train_dualhead_classifier.py --help
```

The SLURM file contains the exact training command and hyperparameters used for
the final model.

---

# Step 3: Run lesion-type prediction

## Recommended: SLURM prediction

Check the script:

```bash
cat simple_dualhead_lesion_classifier/predict_dualhead_classifier.slurm
```

Validate it:

```bash
bash -n simple_dualhead_lesion_classifier/predict_dualhead_classifier.slurm
```

Submit:

```bash
mkdir -p slurm_logs logs/simple_dualhead_classifier
sbatch simple_dualhead_lesion_classifier/predict_dualhead_classifier.slurm
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/simple_dualhead_classifier/predict.log
```

The prediction script uses:

```text
Dataset:
data/simple_dualhead_classifier/dualhead_crop_dataset.json

Images:
/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear

Checkpoint:
results/simple_dualhead_classifier/model_best.pth
```

The prediction output is:

```text
data/simple_dualhead_classifier/dualhead_predictions.json
```

## Direct prediction

```bash
python simple_dualhead_lesion_classifier/02_predict_dualhead_classifier.py \
  --dataset_json data/simple_dualhead_classifier/dualhead_crop_dataset.json \
  --image_root /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --checkpoint results/simple_dualhead_classifier/model_best.pth \
  --out_json data/simple_dualhead_classifier/dualhead_predictions.json \
  --batch_size 64 \
  --num_workers 8
```

Check the output:

```bash
ls -lh data/simple_dualhead_classifier/dualhead_predictions.json
```

Inspect one prediction:

```bash
python - <<'PY'
import json

path = "data/simple_dualhead_classifier/dualhead_predictions.json"

with open(path, "r") as f:
    data = json.load(f)

for split in ["train", "val", "test"]:
    print(split, len(data.get(split, [])))

print(json.dumps(data["test"][0], indent=2))
PY
```

Typical prediction fields include:

```text
predicted_primary_lesion_type
predicted_primary_confidence
predicted_lesion_attribute
predicted_attribute_confidence
```

---

# Step 4: Evaluate classifier predictions

Run:

```bash
python simple_dualhead_lesion_classifier/03_evaluate_dualhead_predictions.py \
  --pred_json data/simple_dualhead_classifier/dualhead_predictions.json
```

The evaluator reports:

```text
primary_accuracy
attribute_accuracy
joint_accuracy
```

The finalized classifier results were:

```text
Validation:
  primary_accuracy   = 0.6084
  attribute_accuracy = 0.7974
  joint_accuracy     = 0.5019

Test:
  primary_accuracy   = 0.5755
  attribute_accuracy = 0.7971
  joint_accuracy     = 0.4916
```

Joint accuracy requires both classifier heads to be correct for the same case.

---

# Step 5: Optional merge into the older RAG-input JSON

This step is optional for the finalized non-RAG report-generation pipeline.

The script:

```text
04_merge_dualhead_predictions_into_rag_input.py
```

merges:

```text
data/simple_dualhead_classifier/dualhead_predictions.json
```

with:

```text
data/oracle_pair_rag_annotations/
02_totalseg_plus_predicted_lesion_complete.json
```

and creates:

```text
data/simple_pair_rag_annotations/
01b_raw_simple_inputs_dualhead.json
```

Check that the required base JSON exists:

```bash
ls -lh \
data/oracle_pair_rag_annotations/02_totalseg_plus_predicted_lesion_complete.json
```

Check ID coverage before merging:

```bash
python - <<'PY'
import json

pred_path = "data/simple_dualhead_classifier/dualhead_predictions.json"
base_path = (
    "data/oracle_pair_rag_annotations/"
    "02_totalseg_plus_predicted_lesion_complete.json"
)

with open(pred_path, "r") as f:
    pred = json.load(f)

with open(base_path, "r") as f:
    base = json.load(f)

for split in ["train", "val", "test"]:
    pred_ids = {str(x["id"]) for x in pred[split]}
    base_ids = {str(x["id"]) for x in base[split]}

    print(
        split,
        "pred:", len(pred_ids),
        "base:", len(base_ids),
        "missing:", len(base_ids - pred_ids),
        "extra:", len(pred_ids - base_ids),
    )
PY
```

Run the merge:

```bash
python \
simple_dualhead_lesion_classifier/04_merge_dualhead_predictions_into_rag_input.py
```

Check the output:

```bash
ls -lh \
data/simple_pair_rag_annotations/01b_raw_simple_inputs_dualhead.json
```

This merge script does not itself perform retrieval. It only prepares an
intermediate structured JSON for the older RAG workflow.

The finalized R2Gen-Mamba-DAM report generator does not run RAG during training
or testing.

---

# Step 6: Full classifier workflow

To rebuild everything from the beginning:

```bash
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

# 1. Build dataset
python simple_dualhead_lesion_classifier/00_build_dualhead_dataset.py

# 2. Train
sbatch simple_dualhead_lesion_classifier/train_dualhead_classifier.slurm

# 3. Predict, evaluate, and optionally merge
sbatch simple_dualhead_lesion_classifier/predict_dualhead_classifier.slurm
```

If the dataset and checkpoint already exist, run only:

```bash
python simple_dualhead_lesion_classifier/02_predict_dualhead_classifier.py \
  --dataset_json data/simple_dualhead_classifier/dualhead_crop_dataset.json \
  --image_root /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --checkpoint results/simple_dualhead_classifier/model_best.pth \
  --out_json data/simple_dualhead_classifier/dualhead_predictions.json \
  --batch_size 64 \
  --num_workers 8

python simple_dualhead_lesion_classifier/03_evaluate_dualhead_predictions.py \
  --pred_json data/simple_dualhead_classifier/dualhead_predictions.json
```

Run the optional merge separately:

```bash
python \
simple_dualhead_lesion_classifier/04_merge_dualhead_predictions_into_rag_input.py
```

---

# Step 7: Verify project paths

The active project root should be:

```text
/data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion
```

Check for obsolete paths:

```bash
grep -RIn \
"/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text" \
simple_dualhead_lesion_classifier
```

The command should return no active code references.

Check all important script paths:

```bash
grep -RInE \
'PROJECT|cd |dataset_json|image_root|checkpoint|out_json' \
simple_dualhead_lesion_classifier
```

---

# Step 8: Connection to R2Gen-Mamba-DAM

During report-generator training, ground-truth lesion-type tokens are used.

During realistic report-generator testing:

```text
YOLO-TLP-MOE predicted bounding box
        ↓
lesion-centered crop
        ↓
dual-head classifier
        ↓
predicted primary lesion category
+ predicted imaging attribute
        ↓
lesion-type conditioning token
        ↓
R2Gen-Mamba-DAM
```

The classifier predictions provide realistic lesion-type conditioning during
short-report generation.
