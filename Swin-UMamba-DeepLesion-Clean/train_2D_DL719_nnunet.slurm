#!/bin/bash
#SBATCH --job-name=DL719_2D_nnUNet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/DL719_2D_nnUNet_%j.out
#SBATCH --error=logs/DL719_2D_nnUNet_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

cd /data/ruida/segmentation/Swin-UMamba-Origin
mkdir -p logs

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate acl_umamba

ml CUDA/12.1
ml gcc/11.3

export nnUNet_raw=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw
export nnUNet_preprocessed=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed
export nnUNet_results=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run this only if preprocessing has not been completed yet.
# nnUNetv2_plan_and_preprocess -d 719 --verify_dataset_integrity -npfp 1 -np 1

echo "Start 2D nnUNet training"
nnUNetv2_train 719 2d all \
  -tr nnUNetTrainer \
  2>&1 | tee train_719_2d_nnunet.log

echo "Training finished"
