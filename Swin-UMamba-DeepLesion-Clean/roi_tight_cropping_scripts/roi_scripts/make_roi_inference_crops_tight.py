#!/usr/bin/env python3
"""Create *tight* ROI crops for inference from YOLO boxes (variable-size).

- Reads yolo_boxes.json from detect_yolo_boxes.py
- Applies NMS (optional) + conf_th filtering
- Expands each bbox by margin
- Enforces min_box_size
- Writes nnUNet-style imagesTs crops + mapping.json for stitching

Output crop naming:
  <stem>__roi00_0000.png, <stem>__roi01_0000.png, ...
Mapping contains xyxy of crop in original image.

Example:
  python make_roi_inference_crops_tight.py \
    --images_dir /data/ruida/SAM/sam3/data/images_filtered \
    --yolo_boxes_json /data/ruida/SAM/sam3/data/yolo_boxes.json \
    --out_imagesTs /data/ruida/tmp/roi_infer/imagesTs \
    --mapping_json /data/ruida/tmp/roi_infer/mapping.json \
    --conf_th 0.25 --iou_nms 0.7 --margin 16 --min_box_size 8 --max_rois_per_image 20
"""

import os, json, argparse
from typing import List, Tuple
import numpy as np
from PIL import Image


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    area_a = max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b = max(0,bx2-bx1)*max(0,by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)


def nms_xyxy(boxes: List[Tuple[int,int,int,int]], scores: List[float], iou_th: float):
    idxs = np.argsort(scores)[::-1].tolist()
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou(boxes[i], boxes[j]) < iou_th]
    return keep


def expand_and_fix_box(x1,y1,x2,y2, margin, W, H, min_box_size):
    x1 -= margin; y1 -= margin; x2 += margin; y2 += margin
    x1 = clamp(x1, 0, W-1); y1 = clamp(y1, 0, H-1)
    x2 = clamp(x2, 1, W);   y2 = clamp(y2, 1, H)

    w = x2 - x1
    h = y2 - y1
    if w < min_box_size:
        cx = (x1+x2)/2
        x1 = int(round(cx - min_box_size/2))
        x2 = x1 + min_box_size
    if h < min_box_size:
        cy = (y1+y2)/2
        y1 = int(round(cy - min_box_size/2))
        y2 = y1 + min_box_size

    # reclamp
    if x1 < 0:
        x2 += -x1; x1 = 0
    if y1 < 0:
        y2 += -y1; y1 = 0
    if x2 > W:
        x1 -= (x2-W); x2 = W
    if y2 > H:
        y1 -= (y2-H); y2 = H

    x1 = clamp(x1, 0, W-1); y1 = clamp(y1, 0, H-1)
    x2 = clamp(x2, 1, W);   y2 = clamp(y2, 1, H)
    return int(x1),int(y1),int(x2),int(y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--yolo_boxes_json", required=True)
    ap.add_argument("--out_imagesTs", required=True)
    ap.add_argument("--mapping_json", required=True)
    ap.add_argument("--conf_th", type=float, default=0.25)
    ap.add_argument("--iou_nms", type=float, default=0.7)
    ap.add_argument("--margin", type=int, default=16)
    ap.add_argument("--min_box_size", type=int, default=8)
    ap.add_argument("--max_rois_per_image", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_imagesTs, exist_ok=True)

    with open(args.yolo_boxes_json, 'r') as f:
        det = json.load(f)

    mapping = {"items": []}

    img_names = sorted([n for n in os.listdir(args.images_dir) if n.lower().endswith('.png')])
    for name in img_names:
        img_path = os.path.join(args.images_dir, name)
        img = Image.open(img_path).convert('L')
        W,H = img.size

        dets = det.get(name, [])
        boxes = []
        scores = []
        for d in dets:
            c = float(d.get('conf', d.get('score', 1.0)))
            if c < args.conf_th:
                continue
            bb = d.get('bbox', d.get('box', None))
            if bb is None or len(bb) != 4:
                continue
            x1,y1,x2,y2 = [int(round(float(v))) for v in bb]
            if x2 < x1:
                x1,x2 = x2,x1
            if y2 < y1:
                y1,y2 = y2,y1
            if x2-x1 < 1 or y2-y1 < 1:
                continue
            boxes.append((x1,y1,x2,y2))
            scores.append(c)

        if boxes:
            keep = nms_xyxy(boxes, scores, args.iou_nms)
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]

        boxes = boxes[:args.max_rois_per_image]
        scores = scores[:args.max_rois_per_image]

        stem = os.path.splitext(name)[0]
        for idx, (b, sc) in enumerate(zip(boxes, scores)):
            x1,y1,x2,y2 = expand_and_fix_box(*b, args.margin, W, H, args.min_box_size)
            crop = img.crop((x1,y1,x2,y2))
            roi_name = f"{stem}__roi{idx:02d}_0000.png"
            crop.save(os.path.join(args.out_imagesTs, roi_name))
            mapping["items"].append({
                "orig_image": name,
                "roi_image": roi_name,
                "xyxy": [x1,y1,x2,y2],
                "score": float(sc),
                "orig_size": [W,H]
            })

    with open(args.mapping_json, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"[OK] crops -> {args.out_imagesTs}")
    print(f"[OK] mapping -> {args.mapping_json} (items={len(mapping['items'])})")


if __name__ == '__main__':
    main()
