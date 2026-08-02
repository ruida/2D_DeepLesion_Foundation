#!/usr/bin/env python3
"""
make_roi_inference_crops_from_yolo_txt.py

Create tight ROI crops for nnUNet/Swin-UMamba inference from YOLO-TLP predicted label txt files.

Input YOLO label format per line:
    class x_center y_center width height confidence
where x_center, y_center, width, height are normalized YOLO coordinates.

Example:
    0 0.241736 0.508267 0.0727046 0.0834989 0.313698

Outputs:
    out_imagesTs/*.png      nnUNet inference ROI crops, named <case>__roiXX_0000.png
    mapping.json            needed by stitch_roi_masks.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def yolo_norm_to_xyxy(xc, yc, bw, bh, W, H):
    cx = xc * W
    cy = yc * H
    w = bw * W
    h = bh * H
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    return x1, y1, x2, y2


def expand_and_clamp_box(x1, y1, x2, y2, W, H, margin=16, min_box_size=8):
    x1 = int(round(x1 - margin))
    y1 = int(round(y1 - margin))
    x2 = int(round(x2 + margin))
    y2 = int(round(y2 + margin))

    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    x2 = clamp(x2, 1, W)
    y2 = clamp(y2, 1, H)

    if x2 - x1 < min_box_size:
        cx = (x1 + x2) / 2.0
        x1 = int(round(cx - min_box_size / 2.0))
        x2 = x1 + min_box_size
        if x1 < 0:
            x1 = 0
            x2 = min_box_size
        if x2 > W:
            x2 = W
            x1 = max(0, W - min_box_size)

    if y2 - y1 < min_box_size:
        cy = (y1 + y2) / 2.0
        y1 = int(round(cy - min_box_size / 2.0))
        y2 = y1 + min_box_size
        if y1 < 0:
            y1 = 0
            y2 = min_box_size
        if y2 > H:
            y2 = H
            y1 = max(0, H - min_box_size)

    return int(x1), int(y1), int(x2), int(y2)


def read_yolo_txt(txt_path):
    dets = []
    if not txt_path.exists():
        return dets

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            dets.append({
                "class_id": int(float(parts[0])),
                "xc": float(parts[1]),
                "yc": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
                "conf": float(parts[5]) if len(parts) >= 6 else 1.0,
            })

    return dets


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    return inter / (area_a + area_b - inter + 1e-6)


def nms(boxes, scores, iou_th):
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1].tolist()
    keep = []

    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if box_iou(boxes[i], boxes[j]) < iou_th]

    return keep


def center_fallback_rois(W, H, scales):
    out = []
    cx, cy = W // 2, H // 2
    for s in scales:
        s = int(s)
        x1 = clamp(cx - s // 2, 0, W - 1)
        y1 = clamp(cy - s // 2, 0, H - 1)
        x2 = clamp(x1 + s, 1, W)
        y2 = clamp(y1 + s, 1, H)
        x1 = max(0, x2 - s)
        y1 = max(0, y2 - s)
        out.append((x1, y1, x2, y2, -1.0, f"fallback_center_{s}"))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--yolo_labels_dir", required=True)
    parser.add_argument("--out_imagesTs", required=True)
    parser.add_argument("--mapping_json", required=True)
    parser.add_argument("--conf_th", type=float, default=0.25)
    parser.add_argument("--iou_nms", type=float, default=0.70)
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--min_box_size", type=int, default=8)
    parser.add_argument("--max_rois_per_image", type=int, default=20)
    parser.add_argument("--image_ext", default=".png")
    parser.add_argument("--keep_empty_images", action="store_true")
    parser.add_argument("--fallback_center_scales", type=int, nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    yolo_labels_dir = Path(args.yolo_labels_dir)
    out_imagesTs = Path(args.out_imagesTs)
    mapping_json = Path(args.mapping_json)

    out_imagesTs.mkdir(parents=True, exist_ok=True)
    mapping_json.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob(f"*{args.image_ext}"))

    mapping = {
        "images_dir": str(images_dir),
        "yolo_labels_dir": str(yolo_labels_dir),
        "out_imagesTs": str(out_imagesTs),
        "items": [],
        "empty_images": []
    }

    total_rois = 0
    total_images_with_rois = 0
    total_empty = 0

    for img_path in image_paths:
        img_name = img_path.name
        stem = img_path.stem
        txt_path = yolo_labels_dir / f"{stem}.txt"

        img = Image.open(img_path).convert("L")
        W, H = img.size

        dets = read_yolo_txt(txt_path)

        boxes = []
        scores = []

        for d in dets:
            if d["conf"] < args.conf_th:
                continue

            x1, y1, x2, y2 = yolo_norm_to_xyxy(d["xc"], d["yc"], d["w"], d["h"], W, H)

            rx1, ry1, rx2, ry2 = expand_and_clamp_box(
                x1, y1, x2, y2,
                W=W, H=H,
                margin=args.margin,
                min_box_size=args.min_box_size
            )

            boxes.append((rx1, ry1, rx2, ry2))
            scores.append(d["conf"])

        if boxes:
            keep = nms(boxes, scores, args.iou_nms)
            keep = keep[:args.max_rois_per_image]
            rois = [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], "yolo_txt") for i in keep]
        else:
            rois = []

        if len(rois) == 0 and args.fallback_center_scales:
            rois = center_fallback_rois(W, H, args.fallback_center_scales)
            rois = rois[:args.max_rois_per_image]

        if len(rois) == 0:
            total_empty += 1
            if args.keep_empty_images:
                mapping["empty_images"].append({
                    "orig_image": img_name,
                    "orig_size": [W, H],
                    "reason": "no_yolo_box_after_conf_threshold"
                })
            continue

        total_images_with_rois += 1

        for roi_idx, (x1, y1, x2, y2, conf, source) in enumerate(rois):
            roi_name = f"{stem}__roi{roi_idx:02d}_0000.png"
            roi_path = out_imagesTs / roi_name

            crop = img.crop((x1, y1, x2, y2))
            crop.save(roi_path)

            mapping["items"].append({
                "orig_image": img_name,
                "roi_image": roi_name,
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(conf),
                "source": source,
                "orig_size": [int(W), int(H)]
            })
            total_rois += 1

    with open(mapping_json, "w") as f:
        json.dump(mapping, f, indent=2)

    print("========================================")
    print("YOLO txt -> tight ROI crops complete")
    print(f"Images scanned          : {len(image_paths)}")
    print(f"Images with ROI crops   : {total_images_with_rois}")
    print(f"Images with no ROI      : {total_empty}")
    print(f"Total ROI crops         : {total_rois}")
    print(f"ROI imagesTs            : {out_imagesTs}")
    print(f"Mapping JSON            : {mapping_json}")
    print("========================================")


if __name__ == "__main__":
    main()
