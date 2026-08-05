#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=96g
#SBATCH --job-name="qwen3vl_dl_dam_test"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

set -euo pipefail
ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate qwen3vl_dl
cd /data/ruida/LLM/Qwen3VL_DeepLesion_DAM

export MODEL_PATH=/data/ruida/Qwen3VL_runs/output_deeplesion_qwen3vl_dam_four_image/checkpoint-REPLACE
export DEEPLESION_QWEN_JSON_DIR=/data/ruida/data/DeepLesion/qwen3vl_dam_four_image
export QWEN_ATTN_BACKEND=sdpa

bash Qwen3-VL/qwen-vl-finetune/evaluation/run_deeplesion_test_single_a100.sh
