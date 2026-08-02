#!/bin/bash
#SBATCH --job-name=r2gen_anat_lesion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd /vf/users/ruida/LLM/R2Gen-Mamba-Merged_text

bash train_merged_anatomy_plus_oracle_lesion.sh
