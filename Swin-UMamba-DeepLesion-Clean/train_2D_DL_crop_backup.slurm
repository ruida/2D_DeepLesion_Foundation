#!/bin/bash
###
#SBATCH --time=72:00:00
#SBATCH --mem=64g
#SBATCH --job-nam="train_2D_deeplesion_crop"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,ALL
 

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate acl_umamba

ml CUDA/12.1
ml gcc/11.3

export nnUNet_raw="/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw"
export nnUNet_preprocessed="/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed"
export nnUNet_results="/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_results"


cd "/data/ruida/segmentation/Swin-UMamba-Origin" \
&&  nnUNetv2_train 711 2d all -tr nnUNetTrainerSwinUMamba --c 2>&1| tee ${PWD##*/}_train_2D_deeplesion_crop.log
