#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=64g
#SBATCH --job-name="eval_ReXCT_2D"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=eval_ReXGroundingCT_YOLO2D_%j.out
#SBATCH --error=eval_ReXGroundingCT_YOLO2D_%j.err

set -eo pipefail

cd /data/ruida/object_detection/YOLO-TLP-MOE
bash eval_ReXGroundingCT_YOLO2D_current.sh 2>&1 | tee YOLO-TLP-MOE_eval_ReXGroundingCT_YOLO2D.log
