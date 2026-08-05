#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=96g
#SBATCH --job-name="medgemma15_deeplesion"
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

python train_medgemma_deeplesion.py \
  --model_name /data/ruida/LLM/MedGemma1.5/medgemma-1.5-4b-it \
  --annotations_json /data/ruida/data/DeepLesion/DeepLesion/annotation_pretty_deeplesion_sentences_flat.json \
  --image_root /data/ruida/data/DeepLesion/Key_slices \
  --output_dir /data/ruida/LLM/MedGemma1.5/outputs/medgemma15_deeplesion_short \
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
