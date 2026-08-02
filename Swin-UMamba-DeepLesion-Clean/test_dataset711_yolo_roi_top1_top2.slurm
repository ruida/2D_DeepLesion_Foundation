#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=64g
#SBATCH --job-name=test_711_yolo_roi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/test_711_yolo_roi_%j.out
#SBATCH --error=logs/test_711_yolo_roi_%j.err

set -euo pipefail

cd /vf/users/ruida/segmentation/Swin-UMamba-DeepLesion-Clean
mkdir -p logs

echo "============================================================"
echo "Start YOLO ROI Swin-UMamba testing"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo "============================================================"

# Conda environment
source /data/ruida/conda2/etc/profile.d/conda.sh

set +u
conda activate acl_umamba
set -u

# HPC modules
ml CUDA/12.1
ml gcc/11.3.0

# nnU-Net paths
export nnUNet_raw=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw
export nnUNet_preprocessed=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed
export nnUNet_results=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_results

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
python --version

echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"

echo
echo "[Check checkpoint]"
CKPT=${nnUNet_results}/Dataset711_DeepLesionCrop/nnUNetTrainerSwinUMamba__nnUNetPlans__2d/fold_all/checkpoint_best.pth
ls -lh "${CKPT}"

echo
echo "[Check ROI crop inputs]"
echo "Top-1 input crops:"
find roi_outputs_top1/imagesTs -name "*.png" | wc -l

echo "Top-2 input crops:"
find roi_outputs_top2/imagesTs -name "*.png" | wc -l

echo
echo "============================================================"
echo "Top-1 ROI prediction"
echo "============================================================"

rm -rf roi_outputs_top1/preds
mkdir -p roi_outputs_top1/preds

nnUNetv2_predict \
  -i roi_outputs_top1/imagesTs \
  -o roi_outputs_top1/preds \
  -d 711 \
  -c 2d \
  -f all \
  -tr nnUNetTrainerSwinUMamba \
  -chk checkpoint_best.pth

echo
echo "============================================================"
echo "Top-2 ROI prediction"
echo "============================================================"

rm -rf roi_outputs_top2/preds
mkdir -p roi_outputs_top2/preds

nnUNetv2_predict \
  -i roi_outputs_top2/imagesTs \
  -o roi_outputs_top2/preds \
  -d 711 \
  -c 2d \
  -f all \
  -tr nnUNetTrainerSwinUMamba \
  -chk checkpoint_best.pth

echo
echo "============================================================"
echo "Prediction finished"
echo "============================================================"

echo "Top-1 predicted ROI masks:"
find roi_outputs_top1/preds -name "*.png" | wc -l

echo "Top-2 predicted ROI masks:"
find roi_outputs_top2/preds -name "*.png" | wc -l

echo "Finished at: $(date)"
