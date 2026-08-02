#!/bin/bash
#SBATCH --job-name=dualhead_pred
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/dualhead_pred_%j.out
#SBATCH --error=slurm_logs/dualhead_pred_%j.err
set -euo pipefail
cd /vf/users/ruida/LLM/R2Gen-Mamba-Merged_text
mkdir -p slurm_logs logs/simple_dualhead_classifier
ml purge
ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba
python simple_dualhead_lesion_classifier/02_predict_dualhead_classifier.py   --dataset_json data/simple_dualhead_classifier/dualhead_crop_dataset.json   --image_root /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear   --checkpoint results/simple_dualhead_classifier/model_best.pth   --out_json data/simple_dualhead_classifier/dualhead_predictions.json   --batch_size 64   --num_workers 8   2>&1 | tee logs/simple_dualhead_classifier/predict.log
python simple_dualhead_lesion_classifier/03_evaluate_dualhead_predictions.py   --pred_json data/simple_dualhead_classifier/dualhead_predictions.json
python simple_dualhead_lesion_classifier/04_merge_dualhead_predictions_into_rag_input.py
