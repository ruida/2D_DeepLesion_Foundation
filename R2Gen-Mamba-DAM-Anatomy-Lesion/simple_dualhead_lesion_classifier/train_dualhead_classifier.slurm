#!/bin/bash
#SBATCH --job-name=dualhead_lesion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/dualhead_lesion_%j.out
#SBATCH --error=slurm_logs/dualhead_lesion_%j.err
set -euo pipefail
cd /vf/users/ruida/LLM/R2Gen-Mamba-Merged_text
mkdir -p slurm_logs logs/simple_dualhead_classifier results/simple_dualhead_classifier
ml purge
ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba
python simple_dualhead_lesion_classifier/01_train_dualhead_classifier.py   --dataset_json data/simple_dualhead_classifier/dualhead_crop_dataset.json   --image_root /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear   --save_dir results/simple_dualhead_classifier   --backbone resnet18   --batch_size 64   --epochs 50   --lr 3e-4   --weight_decay 1e-4   --num_workers 8   --crop_size 224   --expand_ratio 0.25   --attribute_loss_weight 0.5   --patience 12   --seed 9223   2>&1 | tee logs/simple_dualhead_classifier/train.log
