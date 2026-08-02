# R2Gen-Mamba DeepLesion anatomy-name text encoding

This patch adds an anatomy-name text-token conditioning path:

```text
visual tokens + bbox tokens + anatomy-name text tokens -> Mamba encoder -> short report decoder
```

The original ID embedding path is preserved.

## New argument

```bash
--anatomy_encoding id      # original behavior: anatomy IDs -> nn.Embedding
--anatomy_encoding text    # new behavior: anatomy names -> tokenizer IDs -> nn.Embedding
```

For the text mode, use the rough anatomy fields in your JSON:

```json
"rough_anatomy_ids": [13, 1],
"rough_anatomy_names": ["lesion", "lung"]
```

If `rough_anatomy_names` is missing, the code falls back to the fixed rough-ID map:

```text
1 lung, 2 liver, 3 kidney, 4 adrenal, 5 lymph node, 6 bone,
7 soft tissue, 8 abdomen, 9 pelvis, 10 chest, 11 brain head neck,
12 spine, 13 lesion
```

## Recommended training command

```bash
python main_train.py \
  --dataset_name deeplesion \
  --image_dir /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --ann_path data/generated_annotations/predbbox_anat_lesion_token/deeplesion_test_yolotlpmoe_p54_top2_totalseg_to_oracle_anat_generic_lesion_v3_trainval.json \
  --save_dir results/deeplesion_bbox_anatomy_text \
  --max_seq_length 64 \
  --threshold 3 \
  --batch_size 16 \
  --epochs 100 \
  --early_stop 50 \
  --seed 9223 \
  --num_workers 2 \
  --max_boxes 2 \
  --anatomy_source rough \
  --anatomy_encoding text \
  --max_anatomy 8
```

`--max_anatomy 8` now means maximum number of anatomy-name text tokens, not number of anatomy ID slots.
For examples:

```text
["lesion", "lung"]       -> lesion lung        -> 2 tokens
["lesion", "lymph_node"] -> lesion lymph node  -> 3 tokens
["brain_head_neck"]      -> brain head neck    -> 3 tokens
```

## Test command

Use the same `--anatomy_encoding text` setting when testing/loading this checkpoint.

```bash
python main_test.py \
  --dataset_name deeplesion \
  --image_dir /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --ann_path data/generated_annotations/predbbox_anat_lesion_token/deeplesion_test_yolotlpmoe_p54_top2_totalseg_to_oracle_anat_generic_lesion_v3_trainval.json \
  --save_dir results/deeplesion_bbox_anatomy_text_test \
  --load results/deeplesion_bbox_anatomy_text/model_best.pth \
  --max_seq_length 64 \
  --threshold 3 \
  --batch_size 16 \
  --num_workers 2 \
  --max_boxes 2 \
  --anatomy_source rough \
  --anatomy_encoding text \
  --max_anatomy 8
```

## What changed

- `modules/datasets.py`
  - adds `--anatomy_encoding text` behavior.
  - converts `rough_anatomy_names` or rough IDs into text token IDs.

- `modules/lesion_encoder.py`
  - adds `AnatomyTextEmbedder`.
  - `LesionFusion` can now use either ID embedding or anatomy-name text-token embedding.

- `models/r2gen_deeplesion.py`
  - passes `anatomy_encoding` and tokenizer vocabulary size into `LesionFusion`.

- `main_train.py`, `main_test.py`
  - add the new CLI argument.

## Important note

This is a clean text-token replacement for the anatomy-ID input. It is not a pretrained text encoder. The anatomy-name tokens are embedded by a learnable embedding table using the report tokenizer vocabulary. This keeps the change small and compatible with your current R2Gen-Mamba code.
