# DeepLesion Qwen3-VL single-A100 package

This package patches the original FLARE/Qwen3-VL fine-tuning code for DeepLesion report generation on a single A100 GPU.

## Main changes
- Adds `data_conversion/convert_deeplesion_to_qwen3vl.py`
- Registers `deeplesion_train`, `deeplesion_val`, `deeplesion_test` using `DEEPLESION_QWEN_JSON_DIR`
- Makes FlashAttention optional
- Fixes Qwen3-VL module paths (`model.model.visual`, `model.model.language_model`)
- Adds single-GPU training and testing scripts
- Defaults to training vision tower + merger while freezing the LLM for memory stability on 1x A100

## Convert the dataset
```bash
python data_conversion/convert_deeplesion_to_qwen3vl.py \
  --ann_path /data/ruida/data/DeepLesion/DeepLesion/deeplesion_mamba_final.json \
  --image_root /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --output_dir /data/ruida/data/DeepLesion/qwen3vl_converted \
  --image_mode first
```

## Train on one A100
```bash
export DEEPLESION_QWEN_JSON_DIR=/data/ruida/data/DeepLesion/qwen3vl_converted
bash Qwen3-VL/qwen-vl-finetune/scripts/sft_qwen3_4b_deeplesion_single_a100.sh
```

## Test
```bash
export DEEPLESION_QWEN_JSON_DIR=/data/ruida/data/DeepLesion/qwen3vl_converted
export MODEL_PATH=/data/ruida/LLM/Qwen3VL_runs/deeplesion_qwen3vl_single_a100
bash Qwen3-VL/qwen-vl-finetune/evaluation/run_deeplesion_test_single_a100.sh
```
