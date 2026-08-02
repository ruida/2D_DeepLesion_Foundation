#!/usr/bin/env python3
"""
Evaluate V*-style YOLO-TLP-MOE txt predictions.

Prediction txt format:
class cx cy w h conf

GT txt format:
class cx cy w h

Outputs:
- precision
- recall
- F1
- AP50
- mAP50-95
- per-IoU AP table
- CSV summary
"""

import argparse
import csv
from pathlib import Path
import numpy as np


def read_gt_txt(path):
    boxes = []
    if not path.exists():
        return boxes

    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes.append(
            {
                "cls": cls,
                "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                "matched": False,
            }
        )

    return boxes


def read_pred_txt(path):
    boxes = []
    if not path.exists():
        return boxes

    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue

        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        conf = float(parts[5])

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes.append(
            {
                "cls": cls,
                "conf": conf,
                "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            }
        )

    return boxes


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def compute_pr_at_threshold(gt_by_img, pred_by_img, iou_thr=0.5, conf_thr=0.05):
    total_gt = sum(len(v) for v in gt_by_img.values())

    tp = 0
    fp = 0
    fn = 0

    for stem, gt_boxes_orig in gt_by_img.items():
        gt_boxes = []
        for g in gt_boxes_orig:
            q = dict(g)
            q["matched"] = False
            gt_boxes.append(q)

        preds = [
            p for p in pred_by_img.get(stem, [])
            if p["conf"] >= conf_thr
        ]
        preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

        for p in preds:
            best_iou = 0.0
            best_idx = -1

            for i, g in enumerate(gt_boxes):
                if g["matched"]:
                    continue
                if p["cls"] != g["cls"]:
                    continue

                cur_iou = iou(p["box"], g["box"])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = i

            if best_iou >= iou_thr and best_idx >= 0:
                tp += 1
                gt_boxes[best_idx]["matched"] = True
            else:
                fp += 1

        fn += sum(1 for g in gt_boxes if not g["matched"])

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": total_gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_ap(gt_by_img, pred_by_img, iou_thr=0.5):
    total_gt = sum(len(v) for v in gt_by_img.values())

    all_preds = []
    for stem, preds in pred_by_img.items():
        for p in preds:
            q = dict(p)
            q["image"] = stem
            all_preds.append(q)

    all_preds = sorted(all_preds, key=lambda x: x["conf"], reverse=True)

    matched = {}
    for stem, gt_boxes in gt_by_img.items():
        matched[stem] = np.zeros(len(gt_boxes), dtype=bool)

    tp = np.zeros(len(all_preds), dtype=np.float32)
    fp = np.zeros(len(all_preds), dtype=np.float32)

    for pi, p in enumerate(all_preds):
        stem = p["image"]
        gt_boxes = gt_by_img.get(stem, [])

        best_iou = 0.0
        best_idx = -1

        for gi, g in enumerate(gt_boxes):
            if matched[stem][gi]:
                continue
            if p["cls"] != g["cls"]:
                continue

            cur_iou = iou(p["box"], g["box"])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_idx = gi

        if best_iou >= iou_thr and best_idx >= 0:
            tp[pi] = 1
            matched[stem][best_idx] = True
        else:
            fp[pi] = 1

    if total_gt == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / (total_gt + 1e-16)
    precision = tp_cum / (tp_cum + fp_cum + 1e-16)

    # COCO-style interpolation over 101 recall points.
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0

    for r in recall_points:
        p_at_r = precision[recall >= r]
        ap += np.max(p_at_r) if p_at_r.size > 0 else 0.0

    ap /= 101.0
    return float(ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--conf-thr", type=float, default=0.05)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    gt_files = sorted(gt_dir.glob("*.txt"))
    pred_files = sorted(pred_dir.glob("*.txt"))

    stems = sorted(set([p.stem for p in gt_files]) | set([p.stem for p in pred_files]))

    gt_by_img = {}
    pred_by_img = {}

    for stem in stems:
        gt_by_img[stem] = read_gt_txt(gt_dir / f"{stem}.txt")
        pred_by_img[stem] = read_pred_txt(pred_dir / f"{stem}.txt")

    pr = compute_pr_at_threshold(
        gt_by_img,
        pred_by_img,
        iou_thr=args.iou_thr,
        conf_thr=args.conf_thr,
    )

    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    ap_by_iou = {}

    for t in iou_thresholds:
        ap_by_iou[round(float(t), 2)] = compute_ap(gt_by_img, pred_by_img, iou_thr=float(t))

    ap50 = ap_by_iou[0.5]
    map5095 = float(np.mean(list(ap_by_iou.values())))

    n_images = len(stems)
    n_gt = sum(len(v) for v in gt_by_img.values())
    n_pred = sum(len(v) for v in pred_by_img.values())

    print("")
    print("Evaluation results")
    print("==================")
    print(f"Images:      {n_images}")
    print(f"GT boxes:    {n_gt}")
    print(f"Pred boxes:  {n_pred}")
    print("")
    print(f"Conf thr:    {args.conf_thr}")
    print(f"IoU thr:     {args.iou_thr}")
    print(f"TP:          {pr['tp']}")
    print(f"FP:          {pr['fp']}")
    print(f"FN:          {pr['fn']}")
    print(f"Precision:   {pr['precision']:.6f}")
    print(f"Recall:      {pr['recall']:.6f}")
    print(f"F1:          {pr['f1']:.6f}")
    print(f"AP50:        {ap50:.6f}")
    print(f"mAP50-95:    {map5095:.6f}")
    print("")
    print("AP by IoU")
    for k, v in ap_by_iou.items():
        print(f"AP@{k:.2f}:     {v:.6f}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["images", n_images])
        writer.writerow(["gt_boxes", n_gt])
        writer.writerow(["pred_boxes", n_pred])
        writer.writerow(["conf_thr", args.conf_thr])
        writer.writerow(["iou_thr", args.iou_thr])
        writer.writerow(["tp", pr["tp"]])
        writer.writerow(["fp", pr["fp"]])
        writer.writerow(["fn", pr["fn"]])
        writer.writerow(["precision", pr["precision"]])
        writer.writerow(["recall", pr["recall"]])
        writer.writerow(["f1", pr["f1"]])
        writer.writerow(["AP50", ap50])
        writer.writerow(["mAP50_95", map5095])
        for k, v in ap_by_iou.items():
            writer.writerow([f"AP@{k:.2f}", v])

    print("")
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
