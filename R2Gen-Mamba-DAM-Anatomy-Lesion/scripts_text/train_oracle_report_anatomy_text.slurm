#!/bin/bash
#SBATCH --job-name=r2gen_oracle_text
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/r2gen_oracle_text_%j.out
#SBATCH --error=slurm_logs/r2gen_oracle_text_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

cd /data/ruida/LLM/R2Gen-Mamba-Merged_text

bash scripts_text/train_oracle_report_anatomy_text.sh \
  2>&1 | tee logs_text/train_oracle_report_anatomy_text_${SLURM_JOB_ID}.log
