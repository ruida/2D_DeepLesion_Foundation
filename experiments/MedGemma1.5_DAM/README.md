# MedGemma 1.5 DeepLesion long-report starter

This starter trains MedGemma 1.5 on the long-report DeepLesion JSON file:

- annotations: `/data/ruida/data/DeepLesion/annotation_pretty_deeplesion_single.json`
- image root: `/data/ruida/data/DeepLesion/Key_slices`
- local base model: `/data/ruida/LLM/MedGemma1.5/medgemma-1.5-4b-it`

## Train

```bash
python train_medgemma_deeplesion_long.py \
  --model_name /data/ruida/LLM/MedGemma1.5/medgemma-1.5-4b-it \
  --annotations_json /data/ruida/data/DeepLesion/annotation_pretty_deeplesion_single.json \
  --image_root /data/ruida/data/DeepLesion/Key_slices \
  --output_dir ./outputs/medgemma15_deeplesion_long \
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
  --max_length 1024
```

## Test inference

```bash
python infer_medgemma_deeplesion_long.py \
  --base_model /data/ruida/LLM/MedGemma1.5/medgemma-1.5-4b-it \
  --adapter_dir ./outputs/medgemma15_deeplesion_long \
  --annotations_json /data/ruida/data/DeepLesion/annotation_pretty_deeplesion_single.json \
  --image_root /data/ruida/data/DeepLesion/Key_slices \
  --split test \
  --output_jsonl ./outputs/medgemma15_deeplesion_long_test.jsonl \
  --bf16 \
  --use_4bit \
  --local_files_only \
  --max_new_tokens 256
```

## Notes

- This is a direct long-report baseline.
- The reports may contain more whole-study text than a single key slice strictly supports, so hallucination review matters.
- Start small first with `--max_train_samples 128 --max_eval_samples 64` to sanity check the pipeline.
