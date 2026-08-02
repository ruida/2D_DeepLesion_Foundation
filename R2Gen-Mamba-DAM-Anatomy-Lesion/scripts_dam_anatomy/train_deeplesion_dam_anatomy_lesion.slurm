#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=128g
#SBATCH --job-name="r2gen_dam_anat_lesion"
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
conda activate r2gen_mamba
cd /data/ruida/LLM/R2Gen-Mamba-DAM-Anatomy-Lesion
bash scripts_dam_anatomy/train_deeplesion_dam_anatomy_lesion.sh
