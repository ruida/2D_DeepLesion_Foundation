#!/bin/bash
#SBATCH --job-name=r2gen_rag
#SBATCH --output=slurm-r2gen-rag-%j.out
#SBATCH --error=slurm-r2gen-rag-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

set -euo pipefail

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

cd /vf/users/ruida/LLM/R2Gen-Mamba-Merged_text

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results/r2gen_rag_conditioning records

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
echo

nvidia-smi

./train_r2gen_rag_conditioning.sh

echo
echo "Finished: $(date)"
