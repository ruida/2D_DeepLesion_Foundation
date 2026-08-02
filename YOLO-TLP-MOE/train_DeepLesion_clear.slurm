#!/bin/bash
###
#SBATCH --time=96:00:00
#SBATCH --mem=64g
#SBATCH --job-name="yolo_tlp_moe_DeepLesion_clear"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,ALL 

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe 

ml CUDA/12.1
ml gcc/11.3.0

cd "/data/ruida/object_detection/YOLO-TLP-MOE-LOCATION" \
&& yolo detect train   model=ultralytics/cfg/models/v12/yoloTLP_esmoe_p5.yaml   data= deeplesion_clear.yaml   epochs=300   imgsz=768   batch=8   device=0   workers=2   multi_scale=True   project=yoloTLP_runs   name=DeepLesion_tlp_esmoe_p5_loc  2>&1| tee ${PWD##*/}_yolo_tlp_moe_train_DeepLesion_p5_clear.log
