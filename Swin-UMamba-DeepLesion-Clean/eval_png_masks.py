#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import csv

def read_mask_png(path: str, thresh: int = 0) -> np.ndarray:
    """
    Read a PNG mask and return boolean array.
    Works for 0/255, 0/1, grayscale, or RGB (will convert to grayscale).
    thresh: pixels > thresh are treated as foreground.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return arr > thresh

def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    # pred/gt are boolean
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    eps = 1e-8
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou  = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "dice": float(dice), "iou": float(iou),
        "precision": float(prec), "recall": float(rec),
        "specificity": float(spec),
        "pred_sum": int(pred.sum()),
        "gt_sum": int(gt.sum()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Dir containing predicted PNG masks")
    ap.add_argument("--gt_dir", required=True, help="Dir containing GT PNG masks")
    ap.add_argument("--out_dir", default="eval_out", help="Output directory")
    ap.add_argument("--pred_thresh", type=int, default=0, help="Foreground if pixel > pred_thresh")
    ap.add_argument("--gt_thresh", type=int, default=0, help="Foreground if pixel > gt_thresh")
    ap.add_argument("--strict_shape", action="store_true", help="Error if shapes differ (default: skip)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.lower().endswith(".png")])
    gt_set = set([f for f in os.listdir(args.gt_dir) if f.lower().endswith(".png")])

    pairs = [(f, f) for f in pred_files if f in gt_set]
    missing_gt = [f for f in pred_files if f not in gt_set]

    rows = []
    skipped = []

    for pf, gf in pairs:
        p_path = os.path.join(args.pred_dir, pf)
        g_path = os.path.join(args.gt_dir, gf)

        pred = read_mask_png(p_path, thresh=args.pred_thresh)
        gt   = read_mask_png(g_path, thresh=args.gt_thresh)

        if pred.shape != gt.shape:
            msg = f"{pf}: shape pred {pred.shape} != gt {gt.shape}"
            if args.strict_shape:
                raise RuntimeError(msg)
            skipped.append(msg)
            continue

        m = compute_metrics(pred, gt)
        m["case"] = pf
        rows.append(m)

    # write per-case csv
    csv_path = os.path.join(args.out_dir, "metrics_per_case.csv")
    fieldnames = ["case","dice","iou","precision","recall","specificity","tp","fp","fn","tn","pred_sum","gt_sum"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    # summary
    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    summary = {
        "n_pred_png": len(pred_files),
        "n_gt_png": len(gt_set),
        "n_matched": len(pairs),
        "n_evaluated": len(rows),
        "n_missing_gt": len(missing_gt),
        "n_skipped_shape": len(skipped),
        "mean_dice": mean("dice"),
        "mean_iou": mean("iou"),
        "mean_precision": mean("precision"),
        "mean_recall": mean("recall"),
        "mean_specificity": mean("specificity"),
    }

    txt_path = os.path.join(args.out_dir, "metrics_summary.txt")
    with open(txt_path, "w") as f:
        for k,v in summary.items():
            f.write(f"{k}: {v}\n")
        if missing_gt:
            f.write("\nMissing GT for (first 50):\n")
            for x in missing_gt[:50]:
                f.write(f"  {x}\n")
        if skipped:
            f.write("\nSkipped due to shape mismatch (first 50):\n")
            for x in skipped[:50]:
                f.write(f"  {x}\n")

    print("=== Summary ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("Wrote:", csv_path)
    print("Wrote:", txt_path)

if __name__ == "__main__":
    main()

