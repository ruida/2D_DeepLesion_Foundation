# R2Gen-Mamba: DAM + Anatomy + Lesion-Type Integration

This package merges the latest features of the two supplied projects:

1. **R2Gen-DAM visual localization**
   - full CT image plus aligned lesion mask;
   - expanded focal CT crop plus aligned focal mask;
   - shared ResNet-101 with zero-initialized fourth mask channel;
   - zero-gated global-to-local cross-attention and gated FFN.

2. **R2Gen-Mamba semantic conditioning**
   - normalized bounding-box token;
   - anatomy ID or anatomy-name text tokens;
   - explicit lesion-type ID appended with `--include_lesion_type`;
   - original R2Gen-Mamba report decoder and DeepLesion trainer/tester.

## Merged feature flow

```text
full RGB + full mask ------> shared ResNet ------> global tokens
focal RGB + focal mask ----> shared ResNet ------> local tokens
                                         |
                          gated global-to-local cross-attention
                                         |
                              localized visual tokens
                                         |
             + normalized bbox token + anatomy token + lesion-type token
                                         |
                              R2Gen-Mamba decoder
                                         |
                              short lesion report
```

The model accepts a single merged annotation JSON. The DAM fields and semantic
fields must coexist in each row.

Required visual fields:

```json
{
  "id": "000002_02_01_050",
  "image_path": ["imagesTr/000002_02_01_050_0000.png"],
  "mask_path": "labelsTr/000002_02_01_050.png",
  "bboxes": [[293, 251, 314, 270]],
  "bbox_format": "xyxy",
  "report": "abdomen, nodule, retroperitoneum"
}
```

Supported semantic fields:

```json
{
  "rough_anatomy_ids": [8, 4],
  "rough_anatomy_names": ["abdomen", "adrenal"],
  "lesion_type_id": 105,
  "lesion_type_merged": "mass_or_nodule"
}
```

## 1. Build the merged JSON

Use the included merger. The DAM JSON supplies image, mask, and pixel bbox. The
anatomy/lesion JSON supplies semantic metadata and report targets.

```bash
python tools/merge_dam_with_anatomy_lesion_json.py \
  --dam_json /data/ruida/LLM/R2Gen-DAM/data/generated_annotations/deeplesion_dam_segmask/deeplesion_oracle_report_segmask_bbox.json \
  --metadata_json /path/to/anatomy_plus_lesion_type.json \
  --out_json data/generated_annotations/dam_anatomy_lesion/deeplesion_dam_anatomy_lesion.json \
  --summary_json data/generated_annotations/dam_anatomy_lesion/merge_summary.json
```

The merger uses exact sample IDs and omits unmatched rows. Review the summary
before training.

## 2. Train

Edit paths if the extracted project directory differs, then run:

```bash
bash scripts_dam_anatomy/train_deeplesion_dam_anatomy_lesion.sh
```

or:

```bash
sbatch scripts_dam_anatomy/train_deeplesion_dam_anatomy_lesion.slurm
```

Important flags:

```text
--use_dam
--bbox_format xyxy
--anatomy_source rough
--anatomy_encoding id
--include_lesion_type
--num_anatomy 174
--max_anatomy 4
```

`num_anatomy=174` covers the existing anatomy vocabulary and lesion-type IDs
such as 101-108. Increase it if a new JSON contains a larger ID.

## 3. Test

```bash
bash scripts_dam_anatomy/test_deeplesion_dam_anatomy_lesion.sh
```

## 4. Realistic predicted-localization test

Merge the YOLO-TLP-MOE + Swin-UMamba predicted-mask JSON with the same anatomy
and lesion-type metadata JSON:

```bash
python tools/merge_dam_with_anatomy_lesion_json.py \
  --dam_json /data/ruida/LLM/R2Gen-DAM/data/generated_annotations/deeplesion_dam_predmask_test/deeplesion_gttrainval_yolo_swinumamba_test.json \
  --metadata_json /path/to/anatomy_plus_lesion_type.json \
  --out_json data/generated_annotations/dam_anatomy_lesion/deeplesion_gttrainval_predmask_test_anatomy_lesion.json \
  --summary_json data/generated_annotations/dam_anatomy_lesion/predmask_merge_summary.json
```

Then change `ANN=` in the test shell script. The test rows will use predicted
YOLO boxes and predicted Swin-UMamba masks, while retaining the selected
anatomy/lesion-type metadata.

## Implementation notes

- In DAM mode, crop bboxes are interpreted with `--bbox_format` and converted
  internally to normalized `[cx, cy, w, h]` before bbox-token encoding.
- Absolute and image-root-relative image/mask paths are both supported.
- If `mask_path` is absent, the loader creates a rectangular mask from the bbox.
- The visual adapter implements gated cross-attention and gated FFN. It does not
  add a separate new local self-attention layer before cross-attention.
- Existing non-DAM DeepLesion, IU X-Ray, and MIMIC-CXR code is retained.
