#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=64g
#SBATCH --job-name="yolo_tlp_moe_cv1"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,ALL

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

cd /data/ruida/object_detection/YOLO-TLP-MOE || exit 1

yolo detect train \
  model=ultralytics/cfg/models/v12/yoloTLP_esmoe_p5.yaml \
  data=/data/ruida/object_detection/YOLO-TLP-MOE/cv_folds/fold_2/deeplesion_cv.yaml \
  epochs=300 \
  imgsz=768 \
  batch=8 \
  device=0 \
  workers=2 \
  multi_scale=True \
  project=yoloTLP_runs/cv \
  name=DeepLesion_tlp_esmoe_p54_fold2 \
  2>&1 | tee cv_fold2_train.log
