#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=64g
#SBATCH --job-name=train_2D_deeplesion_crop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/train_2D_deeplesion_crop_%j.out
#SBATCH --error=logs/train_2D_deeplesion_crop_%j.err

cd /data/ruida/segmentation/Swin-UMamba-DeepLesion-Clean
mkdir -p logs

source /data/ruida/conda2/etc/profile.d/conda.sh

set +u
conda activate acl_umamba
set -u

ml CUDA/12.1
ml gcc/11.3.0

# IMPORTANT:
# These paths must match where preprocessing saved Dataset711_DeepLesionCrop.
export nnUNet_raw=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw
export nnUNet_preprocessed=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed
export nnUNet_results=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_results

echo "============================================================"
echo "Training Dataset711_DeepLesionCrop"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "============================================================"

ls -lh ${nnUNet_preprocessed}/Dataset711_DeepLesionCrop
ls -lh ${nnUNet_preprocessed}/Dataset711_DeepLesionCrop/nnUNetPlans.json

cd /vf/users/ruida/segmentation/Swin-UMamba-Origin

nnUNetv2_train 711 2d all -tr nnUNetTrainerSwinUMamba --c
