#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=64g
#SBATCH --job-name="smoke_ReXCT_2D"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=smoke_ReXGroundingCT_YOLO2D_%j.out
#SBATCH --error=smoke_ReXGroundingCT_YOLO2D_%j.err

set -eo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

cd /data/ruida/object_detection/YOLO-TLP-MOE

DATA_YAML=/data/ruida/data/CT-RATE/ReXGroundingCT_YOLO2D/rexgroundingct_yolo2d.yaml
MODEL_CFG=ultralytics/cfg/models/v12/yoloTLP_esmoe_p5.yaml

yolo detect train \
  model="${MODEL_CFG}" \
  data="${DATA_YAML}" \
  epochs=1 \
  imgsz=512 \
  batch=8 \
  device=0 \
  workers=2 \
  multi_scale=False \
  project=yoloTLP_runs \
  name=ReXGroundingCT_YOLO2D_tlp_esmoe_p5_gray_smoke \
  2>&1 | tee YOLO-TLP-MOE_smoke_train_ReXGroundingCT_YOLO2D.log
