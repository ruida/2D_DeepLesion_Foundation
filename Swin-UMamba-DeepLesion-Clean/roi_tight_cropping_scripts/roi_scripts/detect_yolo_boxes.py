#!/usr/bin/env python3
"""Run YOLO (Ultralytics) on a folder of PNGs and export detections to JSON.

Output JSON format:
  {
    "000001_01_01_109.png": [
      {"bbox": [x1,y1,x2,y2], "conf": 0.87, "cls": 0},
      ...
    ],
    ...
  }

Example:
  python detect_yolo_boxes.py \
    --weights runs/detect/train/weights/best.pt \
    --images_dir /data/ruida/SAM/sam3/data/images_filtered \
    --out_json /data/ruida/SAM/sam3/data/yolo_boxes.json \
    --conf 0.25 --iou 0.7 --imgsz 512 --device 0
"""

import os, json, argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    model = YOLO(args.weights)
    imgs = sorted([f for f in os.listdir(args.images_dir) if f.lower().endswith('.png')])

    out = {}
    for f in imgs:
        path = os.path.join(args.images_dir, f)
        res = model.predict(path, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)
        r = res[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
                dets.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(c),
                    "cls": int(k)
                })
        out[f] = dets

    with open(args.out_json, 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f"[OK] wrote {args.out_json} (images={len(out)})")


if __name__ == '__main__':
    main()
