#!/usr/bin/env python3
"""Create a nnUNetv2 2D dataset of *tight* ROI crops from DeepLesion PNGs.

- Uses GT bounding boxes from report.json ("Bounding_boxes")
- Crops image+mask with identical ROI (variable-size) = bbox expanded by `--margin`
- Enforces `--min_box_size` (min width/height) after expansion
- Writes nnUNet_raw/DatasetXXX_NAME/{imagesTr,labelsTr,dataset.json}

Your inputs:
  images_dir: /data/ruida/SAM/sam3/data/images_filtered
  labels_dir: /data/ruida/SAM/sam3/data/labels_filtered   (binary 0/1 mask)
  report_json: /data/ruida/SAM/sam3/data/report.json

Example:
  python make_roi_dataset_tight.py \
    --dataset_id 711 --dataset_name DeepLesionROI \
    --images_dir .../images_filtered --labels_dir .../labels_filtered \
    --report_json .../report.json --nnunet_raw $nnUNet_raw \
    --split train --margin 16 --min_box_size 8
"""

import os, json, argparse, math
from typing import Dict, Any, List, Tuple
from PIL import Image


def parse_bbox_string(b: str) -> Tuple[int,int,int,int]:
    # "x1, y1, x2, y2" floats -> ints
    parts = [p.strip() for p in b.split(',')]
    if len(parts) != 4:
        raise ValueError(f"Bad Bounding_boxes string: {b}")
    x1,y1,x2,y2 = [int(round(float(p))) for p in parts]
    # ensure proper ordering
    if x2 < x1:
        x1,x2 = x2,x1
    if y2 < y1:
        y1,y2 = y2,y1
    return x1,y1,x2,y2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_and_fix_box(x1,y1,x2,y2, margin, W, H, min_box_size):
    # Expand by margin on all sides
    x1 -= margin; y1 -= margin; x2 += margin; y2 += margin

    # Clamp to bounds (note: right/bottom are exclusive for PIL crop)
    x1 = clamp(x1, 0, W-1)
    y1 = clamp(y1, 0, H-1)
    x2 = clamp(x2, 1, W)
    y2 = clamp(y2, 1, H)

    # Enforce minimum width/height (expand around center if needed)
    w = x2 - x1
    h = y2 - y1
    if w < min_box_size:
        cx = (x1 + x2) / 2.0
        x1 = int(math.floor(cx - min_box_size/2.0))
        x2 = x1 + min_box_size
    if h < min_box_size:
        cy = (y1 + y2) / 2.0
        y1 = int(math.floor(cy - min_box_size/2.0))
        y2 = y1 + min_box_size

    # Re-clamp after enforcing min size
    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    if x2 > W:
        x1 -= (x2 - W)
        x2 = W
    if y2 > H:
        y1 -= (y2 - H)
        y2 = H

    x1 = clamp(x1, 0, W-1)
    y1 = clamp(y1, 0, H-1)
    x2 = clamp(x2, 1, W)
    y2 = clamp(y2, 1, H)

    # Final sanity
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid ROI after processing: {(x1,y1,x2,y2)}")
    return int(x1),int(y1),int(x2),int(y2)


def write_dataset_json(out_dir: str, dataset_name: str):
    # minimal nnUNetv2 dataset.json for 2D PNG
    ds = {
        "name": dataset_name,
        "description": "DeepLesion ROI crops (tight bbox+margin)",
        "tensorImageSize": "2D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "modality": {"0": "CT"},
        "labels": {"background": 0, "lesion": 1},
        "channel_names": {"0": "CT"},
        "file_ending": ".png"
    }
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(ds, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=int, required=True)
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--labels_dir", type=str, required=True)
    ap.add_argument("--report_json", type=str, required=True)
    ap.add_argument("--nnunet_raw", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train","val","test","all"])
    ap.add_argument("--margin", type=int, default=16)
    ap.add_argument("--min_box_size", type=int, default=8)
    ap.add_argument("--max_rois_per_image", type=int, default=50)
    args = ap.parse_args()

    with open(args.report_json, "r") as f:
        rep = json.load(f)

    ds_dir = os.path.join(args.nnunet_raw, f"Dataset{args.dataset_id:03d}_{args.dataset_name}")
    imagesTr = os.path.join(ds_dir, "imagesTr")
    labelsTr = os.path.join(ds_dir, "labelsTr")
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)

    def split_ok(entry: Dict[str, Any]) -> bool:
        # DeepLesion uses Train_Val_Test field; user showed 3.
        # We'll interpret: 1=train, 2=val, 3=test (common in some DeepLesion preprocessed sets).
        tvt = entry.get("Train_Val_Test", None)
        if args.split == "all" or tvt is None:
            return True
        if args.split == "train":
            return int(tvt) == 1
        if args.split == "val":
            return int(tvt) == 2
        if args.split == "test":
            return int(tvt) == 3
        return True

    written = 0
    skipped_no_bbox = 0

    for fname, lesions in rep.items():
        # fname is like 000001_01_01_109.png
        img_path = os.path.join(args.images_dir, fname)
        lab_path = os.path.join(args.labels_dir, fname)
        if not (os.path.exists(img_path) and os.path.exists(lab_path)):
            continue

        # Some keys map to list of lesions; we crop per lesion_idx
        img = Image.open(img_path).convert("L")
        lab = Image.open(lab_path).convert("L")
        W,H = img.size

        rois_for_img = 0
        for li, entry in enumerate(lesions):
            if rois_for_img >= args.max_rois_per_image:
                break
            if not split_ok(entry):
                continue
            bb = entry.get("Bounding_boxes", None)
            if not bb:
                skipped_no_bbox += 1
                continue

            x1,y1,x2,y2 = parse_bbox_string(bb)
            x1,y1,x2,y2 = expand_and_fix_box(x1,y1,x2,y2, args.margin, W, H, args.min_box_size)

            img_roi = img.crop((x1,y1,x2,y2))
            lab_roi = lab.crop((x1,y1,x2,y2))

            stem = os.path.splitext(fname)[0]
            out_case = f"{stem}__roi{li:02d}"
            out_img = os.path.join(imagesTr, f"{out_case}_0000.png")
            out_lab = os.path.join(labelsTr, f"{out_case}.png")

            img_roi.save(out_img)
            lab_roi.save(out_lab)

            written += 1
            rois_for_img += 1

    write_dataset_json(ds_dir, args.dataset_name)

    print(f"[OK] Dataset written: {ds_dir}")
    print(f"[OK] ROIs saved: {written}")
    if skipped_no_bbox:
        print(f"[WARN] Skipped lesions without Bounding_boxes: {skipped_no_bbox}")


if __name__ == "__main__":
    main()
