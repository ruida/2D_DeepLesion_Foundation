#!/bin/bash
#SBATCH --job-name=pair_rag
#SBATCH --output=slurm-pair-rag-%j.out
#SBATCH --error=slurm-pair-rag-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba
cd /vf/users/ruida/LLM/R2Gen-Mamba-Merged_text
./oracle_pair_rag_pipeline/run_pair_rag.sh
