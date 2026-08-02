# R2Gen-Mamba-DAM-Anatomy-Lesion

This repository trains and tests the **R2Gen-Mamba-DAM-Anatomy-Lesion** model for short report generation on DeepLesion.

The final model uses:

```text
CT image
+ DAM lesion-localized visual feature
+ bounding-box conditioning
+ anatomy conditioning
+ lesion-type conditioning
→ short report generation
```

---

## 1. Environment

Run all commands from the project root:

```bash
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion
```

Activate the environment:

```bash
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba
```

Optional cluster modules:

```bash
ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12
```

---

## 2. Main scripts

```text
scripts_dam_anatomy/train_deeplesion_dam_anatomy_lesion.slurm
scripts_dam_anatomy/train_deeplesion_dam_anatomy_lesion.sh
scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh
```

Use these scripts for the final training and realistic testing workflow.

---

## 3. Training

The main training script is:

```text
scripts_dam_anatomy/train_deeplesion_dam_anatomy_lesion.slurm
```

Check the required training annotation:

```bash
ls -lh \
data/generated_annotations/dam_anatomy_lesion/deeplesion_dam_merged_anatomy_oracle_lesion.json
```

Submit training:

```bash
mkdir -p logs slurm_logs results

sbatch scripts_dam_anatomy/train_deeplesion_dam_anatomy_lesion.slurm
```

The best checkpoint is saved to:

```text
results/deeplesion_dam_anatomy_lesion/model_best.pth
```

Check the checkpoint:

```bash
ls -lh results/deeplesion_dam_anatomy_lesion/model_best.pth
```

---

## 4. Realistic testing

The final realistic testing script is:

```text
scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh
```

This test uses predicted upstream inputs:

```text
YOLO-TLP-MOE predicted bounding boxes
+ Swin-UMamba predicted lesion masks
+ predicted anatomy tokens
+ predicted lesion-type tokens
```

Check the required realistic testing annotation and trained checkpoint:

```bash
ls -lh \
data/generated_annotations/dam_anatomy_lesion_realistic/deeplesion_gttrainval_yolomoe_swinumamba_predanatomy_predlesion_test.json \
results/deeplesion_dam_anatomy_lesion/model_best.pth
```

Run realistic testing:

```bash
chmod +x scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh

./scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh
```

The realistic testing output directory is:

```text
results/deeplesion_dam_anatomy_lesion_yolomoe_test
```

The realistic testing log is:

```text
logs/deeplesion_dam_anatomy_lesion_yolomoe_test.log
```

Check generated result files:

```bash
find \
results/deeplesion_dam_anatomy_lesion_yolomoe_test \
-maxdepth 2 -type f | sort
```

Check final metric lines:

```bash
grep -E \
'BLEU|METEOR|ROUGE|precision|recall|F1|test_' \
logs/deeplesion_dam_anatomy_lesion_yolomoe_test.log | tail -50
```

---

## 5. Simple command summary

### Train

```bash
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

mkdir -p logs slurm_logs results

sbatch scripts_dam_anatomy/train_deeplesion_dam_anatomy_lesion.slurm
```

### Test

```bash
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

chmod +x scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh

./scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion_yolomoe.sh
```

---

## 6. Expected key files

Training annotation:

```text
data/generated_annotations/dam_anatomy_lesion/
deeplesion_dam_merged_anatomy_oracle_lesion.json
```

Realistic testing annotation:

```text
data/generated_annotations/dam_anatomy_lesion_realistic/
deeplesion_gttrainval_yolomoe_swinumamba_predanatomy_predlesion_test.json
```

Best trained checkpoint:

```text
results/deeplesion_dam_anatomy_lesion/model_best.pth
```

Realistic testing output:

```text
results/deeplesion_dam_anatomy_lesion_yolomoe_test
```

---

## 7. Notes

- Use `results/deeplesion_dam_anatomy_lesion/model_best.pth` for final testing.
- The realistic annotation contains the matched cases for the final test.
- The ground-truth report is used only as the evaluation reference during testing.
- No RAG is run during final R2Gen-Mamba-DAM training or testing.
- The `simple_dualhead_lesion_classifier/` directory is a supporting component for lesion-type prediction.
- The YOLO-TLP-MOE detector and Swin-UMamba segmentation outputs are upstream inputs for the realistic test.
