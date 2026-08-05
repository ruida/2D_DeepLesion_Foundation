#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=64g
#SBATCH --job-name="medgemma15_deeplesion_infer"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate medgemma15

ml CUDA/12.1
ml gcc/11.3.0

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/ruida/LLM/MedGemma1.5

sh infer_short.sh 2>&1 | tee infer_deeplesion_short.log
