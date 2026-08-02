#!/bin/bash
#SBATCH --job-name=DL719_2D_KEN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/DL719_2D_KEN_%j.out
#SBATCH --error=logs/DL719_2D_KEN_%j.err
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

# nnUNetv2_plan_and_preprocess -d 719 --verify_dataset_integrity

echo "Start 2D Swin-UMamba training"
nnUNetv2_train 719 2d all \
  -tr nnUNetTrainerSwinUMamba \
  2>&1 | tee train_719_2d_ken.log

echo "Training finished"
