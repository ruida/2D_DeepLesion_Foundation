#!/bin/bash
#SBATCH --job-name=oracle_simple_r2gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/oracle_simple_r2gen_%j.out
#SBATCH --error=slurm_logs/oracle_simple_r2gen_%j.err

set -euo pipefail

PROJECT=/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text
ANN=data/simple_pair_rag_annotations/04_oracle_simple_r2gen.json
IMAGE_DIR=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear
SAVE_DIR=results/oracle_simple_r2gen
LOG=logs/simple_pair_rag/train_oracle_simple_r2gen.log

cd "${PROJECT}"

mkdir -p \
    slurm_logs \
    logs/simple_pair_rag \
    results/oracle_simple_r2gen \
    records

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Project: $(pwd)"
echo "Annotation: ${ANN}"
echo "Save directory: ${SAVE_DIR}"
echo "========================================"

# Load software modules
ml purge
ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

# Activate conda environment
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

echo "Python: $(which python)"
python --version

echo "Loaded modules:"
ml list

echo "GPU information:"
nvidia-smi

# Verify required files
if [[ ! -f "${ANN}" ]]; then
    echo "ERROR: Annotation file not found:"
    echo "${PROJECT}/${ANN}"
    exit 1
fi

if [[ ! -f main_train.py ]]; then
    echo "ERROR: main_train.py not found in $(pwd)"
    exit 1
fi

echo "Starting oracle-simple R2Gen training..."

python main_train.py \
    --image_dir "${IMAGE_DIR}" \
    --ann_path "${ANN}" \
    --dataset_name deeplesion \
    --max_seq_length 64 \
    --threshold 3 \
    --batch_size 16 \
    --epochs 100 \
    --save_dir "${SAVE_DIR}" \
    --record_dir records \
    --seed 9223 \
    --num_workers 8 \
    --anatomy_source rough \
    --anatomy_encoding text \
    --num_anatomy 14 \
    --max_anatomy 8 \
    --max_boxes 2 \
    --early_stop 50 \
    --monitor_mode max \
    --monitor_metric BLEU_4 \
    --lr_ve 5e-5 \
    --lr_ed 1e-4 \
    --weight_decay 5e-5 \
    --lr_scheduler StepLR \
    --step_size 30 \
    --gamma 0.1 \
    2>&1 | tee "${LOG}"

status=${PIPESTATUS[0]}

echo "========================================"
echo "Training exit status: ${status}"
echo "Finish: $(date)"
echo "========================================"

if [[ ${status} -ne 0 ]]; then
    echo "Training failed. Check:"
    echo "${LOG}"
    exit ${status}
fi

echo "Generated checkpoint files:"

find "${SAVE_DIR}" -maxdepth 2 -type f \
    \( -name "*.pth" \
       -o -name "*.pth.tar" \
       -o -name "*checkp*" \) \
    -ls || true

echo "Training completed successfully."
