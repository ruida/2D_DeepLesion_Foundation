#!/bin/bash
#SBATCH --job-name=medgemma_v3_anat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/medgemma_v3_anat_%j.out
#SBATCH --error=logs/medgemma_v3_anat_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL



cd /data/ruida/LLM/MedGemma1.5
mkdir -p logs

source /vf/users/ruida/conda2/etc/profile.d/conda.sh
conda activate medgemma15

module load CUDA/12.1
module load gcc/11.3.0
module load cuDNN/8.9.2/CUDA-12

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="/data/ruida/LLM/MedGemma1.5/medgemma-1.5-4b-it"

ANN_JSON="/data/ruida/LLM/R2Gen-Mamba-Merged_old/data/generated_annotations/fresh_oracle_cropcls/deeplesion_totalseg_hybrid_v3_report_assisted_oracle_anat_cropcls_lesion_type.json"

IMAGE_ROOT="/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear"

OUT_DIR="outputs/medgemma15_deeplesion_short_v3_anat_cropcls"

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "MedGemma 1.5 DeepLesion training"
echo "V3 report-assisted anatomy + cropcls lesion type"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Repo: $(pwd)"
echo "Model: ${MODEL_NAME}"
echo "Annotation: ${ANN_JSON}"
echo "Image root: ${IMAGE_ROOT}"
echo "Output: ${OUT_DIR}"
echo "=========================================="

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

python - <<PY
import json
from collections import Counter

p = "${ANN_JSON}"
data = json.load(open(p))

cnt = Counter()
lens = Counter()

for split in ["train", "val", "test"]:
    print(split, len(data[split]))
    for x in data[split]:
        ids = x.get("rough_anatomy_ids", [])
        lens[len(ids)] += 1
        for i in ids:
            cnt[int(i)] += 1

print("min_id:", min(cnt))
print("max_id:", max(cnt))
print("num tokens:", len(cnt))
print("length distribution:", lens)
print("top token ids:")
for k, v in cnt.most_common(20):
    print(k, v)

print("example:")
import json as _json
print(_json.dumps(data["train"][0], indent=2)[:1500])
PY

python train_medgemma_deeplesion_anat_lesion.py \
  --model_name "${MODEL_NAME}" \
  --annotations_json "${ANN_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --use_structured_context \
  --condition_style names \
  --bf16 \
  --tf32 \
  --use_4bit \
  --gradient_checkpointing \
  --local_files_only \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --save_total_limit 2 \
  --max_length 512 \
  --dataloader_num_workers 0 \
  --seed 9223

echo "=========================================="
echo "Finished time: $(date)"
echo "Output folder:"
ls -lh "${OUT_DIR}"
echo "=========================================="
