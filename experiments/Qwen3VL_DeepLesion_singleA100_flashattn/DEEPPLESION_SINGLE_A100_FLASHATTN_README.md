# DeepLesion Qwen3-VL single-A100 with FlashAttention-2

This package is patched for **single A100** training and testing on DeepLesion.

## Attention backend
- Default backend in the provided scripts: `flash_attention_2`
- Override to SDPA if needed:
  ```bash
  export QWEN_ATTN_BACKEND=sdpa
  ```
- Force FlashAttention-2:
  ```bash
  export QWEN_ATTN_BACKEND=flash_attention_2
  ```

If `flash_attn` is not importable, `train_qwen.py` will raise a clear error when `QWEN_ATTN_BACKEND=flash_attention_2`.

## Train
```bash
export DEEPLESION_QWEN_JSON_DIR=/data/ruida/data/DeepLesion/qwen3vl_converted
export QWEN_ATTN_BACKEND=flash_attention_2
bash Qwen3-VL/qwen-vl-finetune/scripts/sft_qwen3_4b_deeplesion_single_a100.sh
```

## Test
```bash
export DEEPLESION_QWEN_JSON_DIR=/data/ruida/data/DeepLesion/qwen3vl_converted
export MODEL_PATH=/data/ruida/LLM/Qwen3VL_runs/deeplesion_qwen3vl_single_a100
export QWEN_ATTN_BACKEND=flash_attention_2
bash Qwen3-VL/qwen-vl-finetune/evaluation/run_deeplesion_test_single_a100.sh
```
