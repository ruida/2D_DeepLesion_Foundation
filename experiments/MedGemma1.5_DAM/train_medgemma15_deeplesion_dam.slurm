#!/bin/bash
#SBATCH --job-name=medgemma15_dam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


module purge
ml CUDA/12.1
ml gcc/11.3.0
ml cuDNN/8.9.2/CUDA-12

source /vf/users/ruida/conda2/etc/profile.d/conda.sh
conda activate medgemma15

cd /data/ruida/LLM/MedGemma1.5_DAM

mkdir -p outputs logs

python train_medgemma_deeplesion_dam.py \
  --model_name /data/ruida/LLM/MedGemma1.5/medgemma-1.5-4b-it \
  --annotations_json /data/ruida/LLM/R2Gen-DAM/data/generated_annotations/deeplesion_dam_segmask/deeplesion_oracle_report_segmask_bbox.json \
  --image_root /data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D \
  --output_dir outputs/medgemma15_deeplesion_dam_four_image \
  --visual_mode four_image \
  --crop_scale 3.0 \
  --min_crop_size 48 \
  --bf16 \
  --tf32 \
  --use_4bit \
  --gradient_checkpointing \
  --local_files_only \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --num_train_epochs 100 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --save_steps 250 \
  --eval_steps 250 \
  --save_total_limit 3 \
  --dataloader_num_workers 2 \
  2>&1 | tee logs/medgemma15_deeplesion_dam_four_image_train.log
