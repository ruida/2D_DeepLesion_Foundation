# Dual-Head Lesion Classifier

This directory contains the dual-head lesion classifier used to generate lesion-type conditioning for the R2Gen-Mamba-DAM-Anatomy-Lesion pipeline.

The classifier uses a shared image backbone with two output heads:

- **Primary lesion head**: predicts the main lesion category.
- **Attribute head**: predicts a complementary imaging attribute.

The classifier is trained from lesion-centered image crops and can be applied to predicted YOLO-TLP-MOE bounding boxes during realistic testing.

---

## Directory structure

```text
simple_dualhead_lesion_classifier/
├── 00_build_dualhead_dataset.py
├── 01_train_dualhead_classifier.py
├── 02_predict_dualhead_classifier.py
├── 03_evaluate_dualhead_predictions.py
├── 04_merge_dualhead_predictions_into_rag_input.py
├── train_dualhead_classifier.slurm
└── predict_dualhead_classifier.slurm
```

---

## Project directory

```bash
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion
```

## Conda environment

```bash
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba
```

On the cluster, the provided SLURM scripts use:

```bash
ml purge
ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12
```

---

## Required files

### Classifier dataset

```text
data/simple_dualhead_classifier/dualhead_crop_dataset.json
```

### Dataset summary

```text
data/simple_dualhead_classifier/dualhead_crop_dataset_summary.json
```

### Best trained checkpoint

```text
results/simple_dualhead_classifier/model_best.pth
```

### Prediction output

```text
data/simple_dualhead_classifier/dualhead_predictions.json
```

Check the files:

```bash
ls -lh data/simple_dualhead_classifier/
ls -lh results/simple_dualhead_classifier/model_best.pth
```

---

# 1. Build the classifier dataset

Run this step only when the classifier dataset must be regenerated.

```bash
python simple_dualhead_lesion_classifier/00_build_dualhead_dataset.py
```

Expected output:

```text
data/simple_dualhead_classifier/dualhead_crop_dataset.json
data/simple_dualhead_classifier/dualhead_crop_dataset_summary.json
```

Inspect the split sizes:

```bash
python - <<'PY'
import json

path = "data/simple_dualhead_classifier/dualhead_crop_dataset.json"

with open(path, "r") as f:
    data = json.load(f)

for split in ["train", "val", "test"]:
    print(split, len(data.get(split, [])))
    if data.get(split):
        print(data[split][0])
PY
```

The dataset records should contain fields such as:

```text
id
image_path
bboxes
bbox_format
primary_label
attribute_label
report
```

---

# 2. Train the classifier

## SLURM training

Check the training script:

```bash
cat simple_dualhead_lesion_classifier/train_dualhead_classifier.slurm
```

Submit training:

```bash
sbatch simple_dualhead_lesion_classifier/train_dualhead_classifier.slurm
```

Check the job:

```bash
squeue -u "$USER"
```

Monitor the training log:

```bash
tail -f logs/simple_dualhead_classifier/train.log
```

The best checkpoint is saved to:

```text
results/simple_dualhead_classifier/model_best.pth
```

The latest checkpoint is saved to:

```text
results/simple_dualhead_classifier/current_checkpoint.pth
```

Use `model_best.pth` for prediction.

## Direct training

To inspect the exact arguments used by the Python script:

```bash
python simple_dualhead_lesion_classifier/01_train_dualhead_classifier.py --help
```

The SLURM script should call:

```text
simple_dualhead_lesion_classifier/01_train_dualhead_classifier.py
```

with the classifier dataset, image root, output directory, and training hyperparameters.

---

# 3. Run prediction

The prediction script applies the trained classifier to the samples in:

```text
data/simple_dualhead_classifier/dualhead_crop_dataset.json
```

and loads:

```text
results/simple_dualhead_classifier/model_best.pth
```

## SLURM prediction and evaluation

Submit:

```bash
sbatch simple_dualhead_lesion_classifier/predict_dualhead_classifier.slurm
```

The current prediction SLURM workflow runs:

```text
02_predict_dualhead_classifier.py
03_evaluate_dualhead_predictions.py
04_merge_dualhead_predictions_into_rag_input.py
```

Monitor:

```bash
tail -f logs/simple_dualhead_classifier/predict.log
```

Prediction output:

```text
data/simple_dualhead_classifier/dualhead_predictions.json
```

## Direct prediction

```bash
python simple_dualhead_lesion_classifier/02_predict_dualhead_classifier.py   --dataset_json data/simple_dualhead_classifier/dualhead_crop_dataset.json   --image_root /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear   --checkpoint results/simple_dualhead_classifier/model_best.pth   --out_json data/simple_dualhead_classifier/dualhead_predictions.json   --batch_size 64   --num_workers 8
```

---

# 4. Evaluate predictions

Run:

```bash
python simple_dualhead_lesion_classifier/03_evaluate_dualhead_predictions.py   --pred_json data/simple_dualhead_classifier/dualhead_predictions.json
```

The evaluator reports classifier performance for the available validation and test samples, including the primary-head, attribute-head, and joint prediction performance supported by the script.

---

# 5. Use predictions in the report-generation pipeline

The classifier prediction JSON contains the predicted lesion category and attribute for each case.

Typical output fields include:

```text
predicted_primary_lesion_type
predicted_primary_confidence
predicted_lesion_attribute
predicted_attribute_confidence
```

These predictions are converted into lesion-type conditioning tokens for the realistic R2Gen-Mamba-DAM test annotation.

The final realistic report-generation annotation is:

```text
data/generated_annotations/dam_anatomy_lesion_realistic/
deeplesion_gttrainval_yolomoe_swinumamba_predanatomy_predlesion_test.json
```

The report generator then uses:

```text
YOLO-TLP-MOE predicted bounding box
+ Swin-UMamba predicted lesion mask
+ predicted anatomy tokens
+ predicted lesion-type tokens
```

during realistic testing.

---

## Important note about script 04

```text
04_merge_dualhead_predictions_into_rag_input.py
```

was written for an older RAG-oriented intermediate annotation.

The finalized R2Gen-Mamba-DAM-Anatomy-Lesion report generator does not perform RAG during training or testing.

Therefore:

- Run script `04` only when reproducing the older RAG intermediate data.
- Do not treat script `04` as the final DAM annotation builder.
- For the final DAM pipeline, merge the predicted lesion category and attribute into the realistic DAM test annotation used by `test_deeplesion_dam_anatomy_lesion_yolomoe.sh`.

---

# 6. Recommended end-to-end order

```text
1. Build dual-head classifier dataset
2. Train the classifier
3. Select model_best.pth
4. Predict lesion categories and attributes
5. Evaluate classifier predictions
6. Merge predicted lesion semantics into the realistic DAM test annotation
7. Run R2Gen-Mamba-DAM realistic testing
```

Commands:

```bash
# 1. Build dataset
python simple_dualhead_lesion_classifier/00_build_dualhead_dataset.py

# 2. Train
sbatch simple_dualhead_lesion_classifier/train_dualhead_classifier.slurm

# 3–5. Predict and evaluate
sbatch simple_dualhead_lesion_classifier/predict_dualhead_classifier.slurm

# 6–7. Run the final report-generation test after the realistic
# annotation has been prepared
bash scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh
```

---

# 7. Path checks before running

Check for obsolete project paths:

```bash
grep -RIn "/data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion" simple_dualhead_lesion_classifier
```

For this project, the working directory should be:

```text
/data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion
```

Update old paths when necessary:

```bash
sed -i 's#/data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion#/data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion#g' simple_dualhead_lesion_classifier/*.slurm
```

Verify:

```bash
grep -RInE 'cd |dataset_json|image_root|checkpoint|out_json' simple_dualhead_lesion_classifier/*.slurm
```

---

# 8. GitHub release notes

Keep the classifier source code in the repository.

Do not commit large generated files or checkpoints unless they are intentionally released:

```gitignore
data/simple_dualhead_classifier/*.json
results/simple_dualhead_classifier/
logs/simple_dualhead_classifier/
*.pth
```

For a public release, provide download instructions for the trained checkpoint and generated prediction JSON separately.
