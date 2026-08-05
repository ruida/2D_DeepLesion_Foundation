#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=64g
#SBATCH --job-name="qwen3vl_dl_test"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL

ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate qwen3vl_dl

cd /vf/users/ruida/LLM/Qwen3VL_DeepLesion_singleA100_flashattn

export PYTHONPATH=$PWD/Qwen3-VL/qwen-vl-finetune:${PYTHONPATH:-}

export MODEL_PATH=/data/ruida/Qwen3VL_runs/output_deeplesion_qwen3vl_single_a100
export DEEPLESION_QWEN_JSON_DIR=/data/ruida/data/DeepLesion/qwen3vl_converted
export QWEN_ATTN_BACKEND=sdpa

export HF_HOME=/data/ruida/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/ruida/hf_cache/hub
mkdir -p /data/ruida/hf_cache/hub

export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/data/ruida/triton_cache
mkdir -p /data/ruida/triton_cache

bash Qwen3-VL/qwen-vl-finetune/evaluation/run_deeplesion_test_single_a100.sh \
  2>&1 | tee /data/ruida/Qwen3VL_runs/qwen3vl_deeplesion_test_single_a100.log
