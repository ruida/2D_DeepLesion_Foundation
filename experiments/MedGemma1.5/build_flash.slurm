#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=128g
#SBATCH --job-name="build_flash"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL

ml CUDA/12.1
ml gcc/11.3
ml cuDNN/8.9.2/CUDA-12

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate medgemma15

export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="8.0"

cd /data/ruida/LLM/MedGemma1.5

pip install flash-attn --no-build-isolation --no-cache-dir -v 2>&1 | tee build_flash.log

