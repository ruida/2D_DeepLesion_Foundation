# Qwen3-VL DeepLesion DAM integration

This integration uses the existing Qwen3-VL multi-image interface. It does not alter the internal Qwen3-VL vision transformer. Each sample is encoded as either:

- `four_image`: full CT, full binary mask, focal CT crop, focal crop mask.
- `two_overlay`: full CT with mask overlay, focal crop with mask overlay.

The focal crop is expanded from the annotation bbox with `crop_scale=3.0` and `min_crop_size=48`, matching the R2Gen-DAM experiment.

## 1. Prepare oracle-mask training data

```bash
bash run_prepare_qwen3vl_dam_data.sh
```

This creates:

```text
/data/ruida/data/DeepLesion/qwen3vl_dam_four_image/deeplesion_train.json
/data/ruida/data/DeepLesion/qwen3vl_dam_four_image/deeplesion_val.json
/data/ruida/data/DeepLesion/qwen3vl_dam_four_image/deeplesion_test.json
```

## 2. Smoke conversion

Add `--max_samples_per_split 8` to the converter command for a quick check.

## 3. Train

```bash
sbatch run_qwen3vl_deeplesion_dam_single_a100.slurm
```

The integration deliberately uses `QWEN_ATTN_BACKEND=sdpa`, preserving the prior decision to avoid FlashAttention as a blocker.

## 4. Realistic predicted-mask test

Run the same converter with:

```text
/data/ruida/LLM/R2Gen-DAM/data/generated_annotations/deeplesion_dam_predmask_test/deeplesion_gttrainval_yolo_swinumamba_test.json
```

Use a separate output directory such as:

```text
/data/ruida/data/DeepLesion/qwen3vl_dam_predmask_test
```

The converted test split will then use YOLO-TLP-MOE boxes and Swin-UMamba masks while retaining the oracle train/validation splits from that combined JSON.

## Design note

This is a DAM-style multi-image prompting baseline. A true DAM localized vision backbone would require modifying Qwen3-VL's vision encoder to add local self-attention and gated global-to-local cross-attention. The supplied version preserves the pretrained Qwen3-VL architecture and is the safest first experiment.
