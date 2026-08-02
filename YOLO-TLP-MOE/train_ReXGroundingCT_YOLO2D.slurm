#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=128g
#SBATCH --job-name="yolo_tlp_moe_ReXCT_2D"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm_ReXGroundingCT_YOLO2D_%j.out
#SBATCH --error=slurm_ReXGroundingCT_YOLO2D_%j.err


source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

cd /data/ruida/object_detection/YOLO-TLP-MOE

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(which python)"
echo "YOLO: $(which yolo)"

DATA_YAML=/data/ruida/data/CT-RATE/ReXGroundingCT_YOLO2D/rexgroundingct_yolo2d.yaml
MODEL_CFG=ultralytics/cfg/models/v12/yoloTLP_esmoe_p5.yaml

echo "DATA_YAML=${DATA_YAML}"
echo "MODEL_CFG=${MODEL_CFG}"

cat "${DATA_YAML}"

yolo detect train \
  model="${MODEL_CFG}" \
  data="${DATA_YAML}" \
  epochs=300 \
  imgsz=512 \
  batch=8 \
  device=0 \
  workers=2 \
  multi_scale=True \
  project=yoloTLP_runs \
  name=ReXGroundingCT_YOLO2D_tlp_esmoe_p5_gray \
  2>&1 | tee YOLO-TLP-MOE_train_ReXGroundingCT_YOLO2D.log

echo "Job ended at: $(date)"
