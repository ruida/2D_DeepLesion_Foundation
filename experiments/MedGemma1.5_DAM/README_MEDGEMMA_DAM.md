# MedGemma 1.5 + DeepLesion DAM-style focal prompting

This integration keeps MedGemma's pretrained vision tower unchanged and supplies four visual inputs per sample:

1. full CT slice
2. full binary lesion mask
3. 3x expanded focal CT crop
4. focal binary mask crop

The dataset JSON must contain `image_path`, `mask_path`, pixel-space `bboxes` in `xyxy`, and `report`. The R2Gen-DAM JSONs already have this format.

## Important distinction

This is a practical multi-image DAM-style integration. It does **not** insert the R2Gen-DAM gated cross-attention adapter into MedGemma's internal vision transformer. Doing that requires a custom MedGemma model class tied to the exact Transformers implementation and checkpoint layout. The four-image version is safer, preserves pretrained weights, works with LoRA/QLoRA, and is the recommended first experiment.

## Smoke test

Add `--max_train_samples 32 --max_eval_samples 16` to the training command.

## Oracle-mask training

```bash
bash train_medgemma_deeplesion_dam.sh
```

## Predicted-mask inference

```bash
bash infer_medgemma_deeplesion_dam.sh
```

For a fair comparison, also infer on `deeplesion_oracle_masks_same_1850_test.json` using the same adapter.

## Memory

Four images increase vision tokens and GPU memory substantially. Start with batch size 1 and gradient accumulation 16. If OOM occurs, use `--visual_mode two_overlay`, which sends only full and focal masked overlays.
