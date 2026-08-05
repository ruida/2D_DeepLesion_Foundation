#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=96g
#SBATCH --job-name="medgemma15_r2gen_short"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/medgemma15_r2gen_short_%j.out
#SBATCH --error=logs/medgemma15_r2gen_short_%j.err


source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate medgemma15

ml CUDA/12.1
ml gcc/11.3.0

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/data/ruida/LLM/MedGemma1.5
ANN_JSON=/data/ruida/LLM/R2Gen-Mamba/data/visdrone_deeplesion_two_image_short_safe.json
IMAGE_ROOT=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear
OUTPUT_DIR=${PROJECT_DIR}/outputs/medgemma15_r2gen_two_image_short_safe

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "Host: $(hostname)"
echo "Annotation: ${ANN_JSON}"
echo "Image root: ${IMAGE_ROOT}"
echo "Output: ${OUTPUT_DIR}"

test -f "${ANN_JSON}" || {
    echo "ERROR: annotation JSON not found: ${ANN_JSON}"
    exit 1
}

test -d "${IMAGE_ROOT}" || {
    echo "ERROR: image root not found: ${IMAGE_ROOT}"
    exit 1
}

python train_medgemma_deeplesion.py \
  --model_name "${PROJECT_DIR}/medgemma-1.5-4b-it" \
  --annotations_json "${ANN_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --bf16 \
  --tf32 \
  --use_4bit \
  --gradient_checkpointing \
  --local_files_only \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --dataloader_num_workers 0
