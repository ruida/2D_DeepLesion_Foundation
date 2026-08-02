#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=64g
#SBATCH --job-name="yolo_tlp_moe_resume500"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,ALL

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0

cd /data/ruida/object_detection/YOLO-TLP-MOE

yolo detect train \
  resume \
  model=/data/ruida/object_detection/YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/last.pt \
  epochs=500 \
  device=0 \
  workers=2 \
  2>&1 | tee resume_to_500_cont.log
