#!/bin/bash
#SBATCH --job-name=DL719_2D_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/DL719_2D_test_%j.out
#SBATCH --error=logs/DL719_2D_test_%j.err
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

INPUT_DIR=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D/imagesTs
OUTPUT_DIR=/data/ruida/segmentation/Swin-UMamba-Origin/results/Dataset719_DeepLesion2D/pred_2d

mkdir -p ${OUTPUT_DIR}

echo "Start prediction Dataset719 DeepLesion2D"

nnUNetv2_predict \
  -i ${INPUT_DIR} \
  -o ${OUTPUT_DIR} \
  -d 719 \
  -c 2d \
  -tr nnUNetTrainerSwinUMamba \
  -f all \
  2>&1 | tee test_719_2d_predict.log

echo "Prediction finished"

