#!/usr/bin/env python3
"""
Official-style evaluation for saved YOLO txt predictions.

Reports:
P
R
mAP50
mAP50-95

Prediction txt:
class cx cy w h conf

GT txt:
class cx cy w h
"""

import argparse
from pathlib import Path
import numpy as np


def read_gt(path):
    boxes = []
    if not path.exists():
        return boxes

    txt = path.read_text().strip()
    if not txt:
        return boxes

    for line in txt.splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, w, h = map(float, p[1:5])
        boxes.append({
            "cls": cls,
            "box": np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32),
        })
    return boxes


def read_pred(path):
    boxes = []
    if not path.exists():
        return boxes

    txt = path.read_text().strip()
    if not txt:
        return boxes

    for line in txt.splitlines():
        p = line.strip().split()
        if len(p) < 6:
            continue
        cls = int(float(p[0]))
        cx, cy, w, h = map(float, p[1:5])
        conf = float(p[5])
        boxes.append({
            "cls": cls,
            "conf": conf,
            "box": np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32),
        })
    return boxes


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_pr_at_conf(gt_by_img, pred_by_img, conf_thr, iou_thr=0.5):
    tp = 0
    fp = 0
    fn = 0

    for stem, gts in gt_by_img.items():
        matched = np.zeros(len(gts), dtype=bool)

        preds = [p for p in pred_by_img.get(stem, []) if p["conf"] >= conf_thr]
        preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

        for pred in preds:
            best_iou = 0.0
            best_idx = -1

            for i, gt in enumerate(gts):
                if matched[i]:
                    continue
                if pred["cls"] != gt["cls"]:
                    continue

                cur_iou = iou(pred["box"], gt["box"])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = i

            if best_iou >= iou_thr and best_idx >= 0:
                tp += 1
                matched[best_idx] = True
            else:
                fp += 1

        fn += int((~matched).sum())

    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

    return p, r, f1, tp, fp, fn


def compute_ap(gt_by_img, pred_by_img, iou_thr):
    total_gt = sum(len(x) for x in gt_by_img.values())

    preds_all = []
    for stem, preds in pred_by_img.items():
        for p in preds:
            q = dict(p)
            q["stem"] = stem
            preds_all.append(q)

    preds_all = sorted(preds_all, key=lambda x: x["conf"], reverse=True)

    matched = {
        stem: np.zeros(len(gts), dtype=bool)
        for stem, gts in gt_by_img.items()
    }

    tp = np.zeros(len(preds_all), dtype=np.float32)
    fp = np.zeros(len(preds_all), dtype=np.float32)

    for pi, pred in enumerate(preds_all):
        stem = pred["stem"]
        gts = gt_by_img.get(stem, [])

        best_iou = 0.0
        best_idx = -1

        for gi, gt in enumerate(gts):
            if matched[stem][gi]:
                continue
            if pred["cls"] != gt["cls"]:
                continue

            cur_iou = iou(pred["box"], gt["box"])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_idx = gi

        if best_iou >= iou_thr and best_idx >= 0:
            tp[pi] = 1.0
            matched[stem][best_idx] = True
        else:
            fp[pi] = 1.0

    if total_gt == 0 or len(preds_all) == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / (total_gt + 1e-16)
    precision = tp_cum / (tp_cum + fp_cum + 1e-16)

    # 101-point COCO-style AP interpolation
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        vals = precision[recall >= r]
        ap += vals.max() if vals.size else 0.0

    return float(ap / 101.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-txt", default=None)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    stems = sorted(set(p.stem for p in gt_dir.glob("*.txt")) | set(p.stem for p in pred_dir.glob("*.txt")))

    gt_by_img = {s: read_gt(gt_dir / f"{s}.txt") for s in stems}
    pred_by_img = {s: read_pred(pred_dir / f"{s}.txt") for s in stems}

    n_img = len(stems)
    n_gt = sum(len(v) for v in gt_by_img.values())
    n_pred = sum(len(v) for v in pred_by_img.values())

    # Ultralytics-style displayed P/R are usually near the best-F1 operating point.
    confs = []
    for preds in pred_by_img.values():
        confs.extend([p["conf"] for p in preds])

    if confs:
        grid = sorted(set(np.linspace(0.001, 0.999, 300).tolist() + confs))
    else:
        grid = [0.25]

    best = None
    for c in grid:
        p, r, f1, tp, fp, fn = compute_pr_at_conf(gt_by_img, pred_by_img, c, iou_thr=0.5)
        if best is None or f1 > best["f1"]:
            best = {
                "conf": c,
                "p": p,
                "r": r,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

    ap50 = compute_ap(gt_by_img, pred_by_img, 0.50)
    aps = [compute_ap(gt_by_img, pred_by_img, t) for t in np.arange(0.50, 0.96, 0.05)]
    map5095 = float(np.mean(aps))

    lines = []
    lines.append("")
    lines.append("Official-style V* evaluation")
    lines.append("============================")
    lines.append(f"Images:     {n_img}")
    lines.append(f"Instances:  {n_gt}")
    lines.append(f"Pred boxes: {n_pred}")
    lines.append("")
    lines.append("                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)")
    lines.append(
        f"{'all':>20} {n_img:10d} {n_gt:10d} "
        f"{best['p']:10.3f} {best['r']:10.3f} {ap50:10.3f} {map5095:10.3f}"
    )
    lines.append("")
    lines.append(f"Best-F1 confidence threshold: {best['conf']:.6f}")
    lines.append(f"F1: {best['f1']:.6f}")
    lines.append(f"TP: {best['tp']}  FP: {best['fp']}  FN: {best['fn']}")

    text = "\n".join(lines)
    print(text)

    if args.out_txt:
        out = Path(args.out_txt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
