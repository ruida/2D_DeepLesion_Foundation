#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping_json", required=True)
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--binarize", action="store_true")
    return p.parse_args()


def find_pred_file(pred_dir, roi_image):
    roi_path = Path(roi_image)
    stem = roi_path.stem

    candidates = []

    # input: xxx__roi00_0000.png
    # nnUNet output usually: xxx__roi00.png
    if stem.endswith("_0000"):
        candidates.append(pred_dir / (stem[:-5] + ".png"))

    candidates.append(pred_dir / roi_path.name)
    candidates.append(pred_dir / (stem + ".png"))

    for p in candidates:
        if p.exists():
            return p

    return None


def main():
    args = parse_args()

    mapping_json = Path(args.mapping_json)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_json, "r") as f:
        mapping = json.load(f)

    grouped = {}
    for item in mapping["items"]:
        grouped.setdefault(item["orig_image"], []).append(item)

    saved = 0
    missing = 0
    nonzero_roi = 0
    nonzero_full = 0

    for orig_image, roi_items in grouped.items():
        W, H = roi_items[0]["orig_size"]
        full_mask = np.zeros((H, W), dtype=np.uint8)

        for item in roi_items:
            pred_path = find_pred_file(pred_dir, item["roi_image"])

            if pred_path is None:
                print(f"[WARN] missing pred for {item['roi_image']}")
                missing += 1
                continue

            x1, y1, x2, y2 = item["xyxy"]
            roi_w = x2 - x1
            roi_h = y2 - y1

            pred = Image.open(pred_path).convert("L")

            if pred.size != (roi_w, roi_h):
                pred = pred.resize((roi_w, roi_h), resample=Image.NEAREST)

            pred_arr = np.array(pred)

            # IMPORTANT:
            # nnUNet PNG prediction may be 0/1, not 0/255.
            # So binarize by >0, not >127.
            if args.binarize:
                pred_arr = (pred_arr > 0).astype(np.uint8) * 255
            else:
                if pred_arr.max() <= 1:
                    pred_arr = pred_arr.astype(np.uint8) * 255
                else:
                    pred_arr = pred_arr.astype(np.uint8)

            if pred_arr.max() > 0:
                nonzero_roi += 1

            region = full_mask[y1:y2, x1:x2]
            full_mask[y1:y2, x1:x2] = np.maximum(region, pred_arr)

        if full_mask.max() > 0:
            nonzero_full += 1

        Image.fromarray(full_mask).save(out_dir / orig_image)
        saved += 1

    print("========================================")
    print("Stitch ROI masks complete")
    print(f"Full masks saved        : {saved}")
    print(f"Nonzero ROI predictions : {nonzero_roi}")
    print(f"Nonzero full masks      : {nonzero_full}")
    print(f"Missing predictions     : {missing}")
    print(f"Output dir              : {out_dir}")
    print("========================================")


if __name__ == "__main__":
    main()
