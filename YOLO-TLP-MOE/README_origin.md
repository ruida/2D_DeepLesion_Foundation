# YOLO-TLP-MOE DeepLesion Bounding-Box Detection

This repository trains and evaluates a YOLO-TLP-MOE model for single-class
lesion bounding-box detection on the DeepLesion dataset.

The workflow covered by this README is:

```text
Check dataset and environment
        ↓
Train with train_DeepLesion_clear.slurm
        ↓
Resume with resume_clear.slurm, if needed
        ↓
Evaluate the best checkpoint with eval_deeplesion_clear_current.sh
```

---

## 1. Important files

```text
train_DeepLesion_clear.slurm
resume_clear.slurm
eval_deeplesion_clear_current.sh
deeplesion_clear.yaml
ultralytics/cfg/models/v12/yoloTLP_esmoe_p5.yaml
```

The dataset contains one detection class:

```text
0: lesion
```

---

## 2. Environment

The supplied scripts use:

```text
Conda environment: yolo_tlp_moe
CUDA module:       CUDA/12.1
GCC module:        gcc/11.3.0
GPU:               one NVIDIA A100
```

Activate the environment manually when running commands outside SLURM:

```bash
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0
```

Check the installation:

```bash
which yolo
yolo version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 3. DeepLesion dataset

The supplied `deeplesion_clear.yaml` contains:

```yaml
path: /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/images

names:
  0: lesion

nc: 1
```

Expected structure:

```text
/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
└── VisDrone2019-DET-test-dev/
    ├── images/
    └── labels/
```

Verify the dataset before training:

```bash
ls -ld \
/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear

ls -ld \
/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-train/images \
/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-val/images \
/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images
```

Check the image and label counts:

```bash
for split in train val test-dev; do
    case "$split" in
        train)    folder=VisDrone2019-DET-train ;;
        val)      folder=VisDrone2019-DET-val ;;
        test-dev) folder=VisDrone2019-DET-test-dev ;;
    esac

    echo "$folder"
    find \
    /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/$folder/images \
    -type f | wc -l

    find \
    /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/$folder/labels \
    -type f -name '*.txt' | wc -l
done
```

YOLO label format:

```text
class_id center_x center_y width height
```

The coordinates are normalized to `[0, 1]`.

---

## 4. Important path consistency check

The supplied scripts do not currently point to one consistent experiment.

### Training script

`train_DeepLesion_clear.slurm` changes to:

```text
/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION
```

and creates the run:

```text
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc
```

### Resume and evaluation scripts

`resume_clear.slurm` and `eval_deeplesion_clear_current.sh` currently point to:

```text
/data/ruida/object_detection/YOLO-TLP-MOE
```

and use the older run:

```text
yoloTLP_runs/DeepLesion_tlp_esmoe_p54
```

Therefore, the attached resume and evaluation scripts do **not** automatically
resume or evaluate the checkpoint produced by the attached training script.

Before running, decide which experiment is intended.

For the model trained by `train_DeepLesion_clear.slurm`, use:

```text
Project:
/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION

Run:
DeepLesion_tlp_esmoe_p5_loc

Last checkpoint:
/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION/
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/last.pt

Best checkpoint:
/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION/
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/best.pt
```

If the actual repository has been moved to another directory, replace the
project root consistently in all three scripts.

---

# Step 1: Train the model

The supplied training configuration is:

```text
Model:       ultralytics/cfg/models/v12/yoloTLP_esmoe_p5.yaml
Dataset:     deeplesion_clear.yaml
Epochs:      300
Image size:  768
Batch size:  8
Workers:     2
Multi-scale: enabled
GPU:         device 0
Run name:    DeepLesion_tlp_esmoe_p5_loc
```

## 1.1 Inspect the training script

```bash
cat train_DeepLesion_clear.slurm
```

The command should not contain a space after `data=`. Use:

```text
data=deeplesion_clear.yaml
```

rather than:

```text
data= deeplesion_clear.yaml
```

## 1.2 Recommended corrected training script

For the attached training experiment, replace the script with:

```bash
cat > train_DeepLesion_clear.slurm <<'EOF'
#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=64g
#SBATCH --job-name="yolo_tlp_moe_DeepLesion_clear"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm_logs/train_deeplesion_clear_%j.out
#SBATCH --error=slurm_logs/train_deeplesion_clear_%j.err

set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

PROJECT=/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION

cd "$PROJECT"
mkdir -p slurm_logs

yolo detect train \
  model=ultralytics/cfg/models/v12/yoloTLP_esmoe_p5.yaml \
  data=deeplesion_clear.yaml \
  epochs=300 \
  imgsz=768 \
  batch=8 \
  device=0 \
  workers=2 \
  multi_scale=True \
  project=yoloTLP_runs \
  name=DeepLesion_tlp_esmoe_p5_loc \
  2>&1 | tee yolo_tlp_moe_train_DeepLesion_p5_clear.log
EOF
```

If the active repository is actually:

```text
/data/ruida/object_detection/YOLO-TLP-MOE
```

change only:

```bash
PROJECT=/data/ruida/object_detection/YOLO-TLP-MOE
```

and keep the same project root in the resume and evaluation scripts.

## 1.3 Validate and submit

```bash
mkdir -p slurm_logs
bash -n train_DeepLesion_clear.slurm
sbatch train_DeepLesion_clear.slurm
```

Check the job:

```bash
squeue -u "$USER"
```

Monitor the training log:

```bash
tail -f yolo_tlp_moe_train_DeepLesion_p5_clear.log
```

## 1.4 Training outputs

Expected run directory:

```text
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/
```

Important outputs:

```text
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/best.pt
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/last.pt
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/results.csv
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/results.png
```

Check them:

```bash
ls -lh \
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/
```

Use:

```text
best.pt
```

for final testing and inference.

Use:

```text
last.pt
```

only to resume interrupted training.

---

# Step 2: Resume training

Resume only after the original training job has created:

```text
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/last.pt
```

## 2.1 Recommended corrected resume script

The supplied `resume_clear.slurm` points to the older `p54` experiment. To
resume the run created by `train_DeepLesion_clear.slurm`, use:

```bash
cat > resume_clear.slurm <<'EOF'
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=64g
#SBATCH --job-name="yolo_tlp_moe_resume"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm_logs/resume_deeplesion_clear_%j.out
#SBATCH --error=slurm_logs/resume_deeplesion_clear_%j.err

set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

PROJECT=/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION
LAST=$PROJECT/yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/last.pt

cd "$PROJECT"
mkdir -p slurm_logs

if [[ ! -f "$LAST" ]]; then
    echo "ERROR: resume checkpoint not found: $LAST"
    exit 1
fi

yolo detect train \
  resume \
  model="$LAST" \
  device=0 \
  workers=2 \
  2>&1 | tee resume_deeplesion_clear.log
EOF
```

With Ultralytics resume mode, the training state stored in `last.pt` is restored,
including the prior optimizer, scheduler, epoch, and saved training arguments.

Do not change the total epoch target casually when using resume mode. First
inspect the saved run configuration:

```bash
cat \
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/args.yaml
```

## 2.2 Validate and submit

```bash
bash -n resume_clear.slurm
sbatch resume_clear.slurm
```

Monitor:

```bash
squeue -u "$USER"
tail -f resume_deeplesion_clear.log
```

---

# Step 3: Evaluate the test split

The supplied evaluation script uses:

```text
split=test
imgsz=768
batch=8
workers=2
save_json=True
```

It reports the standard detection metrics:

```text
Precision
Recall
mAP50
mAP50-95
```

## 3.1 Recommended evaluation script for the trained P5-location run

```bash
cat > eval_deeplesion_clear_current.sh <<'EOF'
#!/bin/bash

set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

PROJECT=/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION
MODEL=$PROJECT/yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/best.pt
DATA=$PROJECT/deeplesion_clear.yaml

cd "$PROJECT"

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: best checkpoint not found: $MODEL"
    exit 1
fi

if [[ ! -f "$DATA" ]]; then
    echo "ERROR: dataset YAML not found: $DATA"
    exit 1
fi

yolo detect val \
  model="$MODEL" \
  data="$DATA" \
  split=test \
  imgsz=768 \
  batch=8 \
  device=0 \
  workers=2 \
  save_json=True \
  plots=True \
  project=yoloTLP_runs/eval \
  name=DeepLesion_tlp_esmoe_p5_loc_test \
  exist_ok=True \
  2>&1 | tee YOLO-TLP-MOE_eval_DeepLesion_clear.log
EOF
```

Make it executable:

```bash
chmod +x eval_deeplesion_clear_current.sh
```

Run on a GPU compute node:

```bash
./eval_deeplesion_clear_current.sh
```

Alternatively, place the same command in a SLURM script and submit it with
`sbatch`.

## 3.2 Evaluation outputs

Expected directory:

```text
yoloTLP_runs/eval/DeepLesion_tlp_esmoe_p5_loc_test/
```

Typical files include:

```text
confusion_matrix.png
confusion_matrix_normalized.png
F1_curve.png
P_curve.png
PR_curve.png
R_curve.png
predictions.json
```

Inspect the final metric lines:

```bash
grep -E "all|Precision|Recall|mAP" \
YOLO-TLP-MOE_eval_DeepLesion_clear.log | tail -20
```

---

# Step 4: Evaluate the existing p54 checkpoint instead

The original supplied evaluation script evaluates:

```text
/data/ruida/object_detection/YOLO-TLP-MOE/
yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/best.pt
```

To reproduce that specific older test, use:

```bash
cd /data/ruida/object_detection/YOLO-TLP-MOE

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

bash eval_deeplesion_clear_current.sh
```

This is a different experiment from the P5-location training script unless the
training paths and run names are intentionally changed to match.

---

# Step 5: Generate prediction files for downstream pipelines

`yolo detect val` computes evaluation metrics. To export bounding-box TXT files
and rendered images, run prediction separately.

For the P5-location experiment:

```bash
PROJECT=/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION
MODEL=$PROJECT/yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/best.pt
SOURCE=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images

cd "$PROJECT"

yolo detect predict \
  model="$MODEL" \
  source="$SOURCE" \
  imgsz=768 \
  conf=0.05 \
  device=0 \
  workers=2 \
  save=True \
  save_txt=True \
  save_conf=True \
  project=yoloTLP_runs/predict \
  name=DeepLesion_tlp_esmoe_p5_loc_pred
```

Prediction labels are saved in:

```text
yoloTLP_runs/predict/DeepLesion_tlp_esmoe_p5_loc_pred/labels/
```

Prediction label format:

```text
class_id center_x center_y width height confidence
```

These labels can be used by downstream lesion-crop, segmentation, anatomy, and
report-generation pipelines.

---

# Step 6: Quick command summary

## Train

```bash
cd /data/ruida/object_detection/YOLO-TLP-MOE-LOCATION
mkdir -p slurm_logs
bash -n train_DeepLesion_clear.slurm
sbatch train_DeepLesion_clear.slurm
```

## Resume

```bash
cd /data/ruida/object_detection/YOLO-TLP-MOE-LOCATION
bash -n resume_clear.slurm
sbatch resume_clear.slurm
```

## Evaluate

```bash
cd /data/ruida/object_detection/YOLO-TLP-MOE-LOCATION
chmod +x eval_deeplesion_clear_current.sh
./eval_deeplesion_clear_current.sh
```

## Monitor

```bash
squeue -u "$USER"
tail -f yolo_tlp_moe_train_DeepLesion_p5_clear.log
tail -f resume_deeplesion_clear.log
tail -f YOLO-TLP-MOE_eval_DeepLesion_clear.log
```

---

## Final reproducibility checklist

Before training:

```bash
test -f ultralytics/cfg/models/v12/yoloTLP_esmoe_p5.yaml
test -f deeplesion_clear.yaml
test -d /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear
```

Before resuming:

```bash
test -f \
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/last.pt
```

Before testing:

```bash
test -f \
yoloTLP_runs/DeepLesion_tlp_esmoe_p5_loc/weights/best.pt
```

The project directory, run name, and checkpoint path must match across the
training, resume, and evaluation scripts.
