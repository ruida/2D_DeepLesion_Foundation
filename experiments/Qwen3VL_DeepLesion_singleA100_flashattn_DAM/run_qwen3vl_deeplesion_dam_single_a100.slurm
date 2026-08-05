#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=128g
#SBATCH --job-name="qwen3vl_dl_dam"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm-%j.out

set -euo pipefail

ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate qwen3vl_dl

cd /data/ruida/LLM/Qwen3VL_DeepLesion_singleA100_flashattn_DAM

export MODEL_PATH=/data/ruida/LLM/Qwen3VL_DeepLesion_singleA100_flashattn/Qwen3-VL-4B-Instruct
export DEEPLESION_QWEN_JSON_DIR=/data/ruida/data/DeepLesion/qwen3vl_dam_four_image
export QWEN_ATTN_BACKEND=sdpa

mkdir -p /data/ruida/Qwen3VL_runs

bash Qwen3-VL/qwen-vl-finetune/scripts/sft_qwen3_4b_deeplesion_dam_single_a100.sh \
  2>&1 | tee /data/ruida/Qwen3VL_runs/qwen3vl_deeplesion_dam_four_image.log
