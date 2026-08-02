#!/usr/bin/env python3
"""Stitch ROI predictions back to full-size masks.

Assumes nnUNet predicted PNG masks for each ROI crop with same stem as roi_image.
Example ROI image: 000001_01_01_109__roi00_0000.png
Expected pred file (common nnUNet png writer): 000001_01_01_109__roi00.png

This script:
- Reads mapping.json created by make_roi_inference_crops_tight.py
- For each ROI pred, pastes into full canvas at (x1,y1,x2,y2)
- Combines overlaps via pixelwise max
- Writes full-size binary masks as PNG

Example:
  python stitch_roi_masks.py \
    --mapping_json /data/ruida/tmp/roi_infer/mapping.json \
    --pred_dir /data/ruida/tmp/roi_infer/preds \
    --out_dir /data/ruida/tmp/full_masks \
    --binarize
"""

import os, json, argparse
import numpy as np
from PIL import Image


def roi_pred_name_from_roi_image(roi_image: str) -> str:
    # roi_image: <stem>__roiXX_0000.png -> <stem>__roiXX.png
    base = roi_image.replace('_0000.png', '').replace('.png', '')
    return f"{base}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapping_json', required=True)
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--binarize', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.mapping_json, 'r') as f:
        mapping = json.load(f)

    # group items by orig_image
    groups = {}
    for it in mapping.get('items', []):
        groups.setdefault(it['orig_image'], []).append(it)

    for orig, items in groups.items():
        W,H = items[0]['orig_size']
        canvas = np.zeros((H, W), dtype=np.uint8)

        for it in items:
            pred_name = roi_pred_name_from_roi_image(it['roi_image'])
            pred_path = os.path.join(args.pred_dir, pred_name)
            if not os.path.exists(pred_path):
                # try alternate: maybe nnUNet outputs same as input name without _0000
                continue
            pred = Image.open(pred_path).convert('L')
            pred_arr = np.array(pred, dtype=np.uint8)
            if args.binarize:
                pred_arr = (pred_arr > 0).astype(np.uint8) * 255

            x1,y1,x2,y2 = it['xyxy']
            # resize pred to match ROI (in case writer changed size)
            roi_w = x2 - x1
            roi_h = y2 - y1
            if pred_arr.shape[0] != roi_h or pred_arr.shape[1] != roi_w:
                pred = pred.resize((roi_w, roi_h), resample=Image.NEAREST)
                pred_arr = np.array(pred, dtype=np.uint8)
                if args.binarize:
                    pred_arr = (pred_arr > 0).astype(np.uint8) * 255

            canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], pred_arr)

        out_path = os.path.join(args.out_dir, orig)
        Image.fromarray(canvas).save(out_path)

    print(f"[OK] stitched masks -> {args.out_dir} (images={len(groups)})")


if __name__ == '__main__':
    main()
