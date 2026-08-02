#!/bin/bash
#SBATCH --job-name=lesion10cls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-lesion10cls-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

cd /vf/users/ruida/LLM/R2Gen-Mamba-Merged_text

DATA_ROOT=data/lesion_type_cls_expand025_10class
SAVE_DIR=results/lesion_type_cls_expand025_10class_resnet18
LOG=logs/train_lesion_type_10class.log

mkdir -p "${SAVE_DIR}" logs

test -f "${DATA_ROOT}/labels/train.csv" || {
    echo "ERROR: missing train CSV:"
    echo "${DATA_ROOT}/labels/train.csv"
    exit 1
}

test -f "${DATA_ROOT}/labels/val.csv" || {
    echo "ERROR: missing val CSV:"
    echo "${DATA_ROOT}/labels/val.csv"
    exit 1
}

test -f "${DATA_ROOT}/label_map.json" || {
    echo "ERROR: missing label map:"
    echo "${DATA_ROOT}/label_map.json"
    exit 1
}

test -f train_lesion_type_classifier.py || {
    echo "ERROR: missing train_lesion_type_classifier.py"
    exit 1
}

echo "Training configuration:"
echo "  data root : ${DATA_ROOT}"
echo "  save dir  : ${SAVE_DIR}"
echo "  epochs    : 50"
echo "  batch size: 64"
echo "  lr        : 1e-4"
echo "  weights   : sqrt inverse frequency"

python train_lesion_type_classifier.py \
  --data_root "${DATA_ROOT}" \
  --save_dir "${SAVE_DIR}" \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-4 \
  --num_workers 4 \
  --pretrained \
  --seed 9223 \
  2>&1 | tee "${LOG}"
