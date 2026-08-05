#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=128g
#SBATCH --job-name="qwen3vl_dl_fa2"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL

ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate qwen3vl_dl

cd /vf/users/ruida/LLM/Qwen3VL_DeepLesion_singleA100_flashattn

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0"
export DEEPLESION_QWEN_JSON_DIR=/data/ruida/data/DeepLesion/qwen3vl_converted
export QWEN_ATTN_BACKEND=flash_attention_2

bash Qwen3-VL/qwen-vl-finetune/scripts/sft_qwen3_4b_deeplesion_single_a100.sh   2>&1 | tee qwen3vl_deeplesion_flashattn_train.log
