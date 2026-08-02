#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


def load_mask(path, thresh):
    arr = np.array(Image.open(path).convert("L"))
    return arr > thresh


def dice_score(pred, gt):
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    denom = pred_sum + gt_sum
    if denom == 0:
        return 0.0

    inter = np.logical_and(pred, gt).sum()
    return 2.0 * inter / denom


def iou_score(pred, gt):
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    inter = np.logical_and(pred, gt).sum()
    return inter / union


def surface(mask):
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def hd95(pred, gt):
    """
    2D Hausdorff95 in pixel units.
    If both empty: 0.
    If one empty: NaN.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0

    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    pred_surf = surface(pred)
    gt_surf = surface(gt)

    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return np.nan

    # Distance from every pixel to nearest surface pixel.
    dt_pred = distance_transform_edt(~pred_surf)
    dt_gt = distance_transform_edt(~gt_surf)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    if all_dists.size == 0:
        return np.nan

    return float(np.percentile(all_dists, 95))


def norm_name(name):
    """
    Match common nnUNet and ROI naming.
    Examples:
      xxx_0000.png -> xxx.png
      xxx.png      -> xxx.png
    """
    p = Path(name)
    stem = p.stem
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem + ".png"


def summarize(values):
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]

    if valid.size == 0:
        return {
            "mean": None,
            "std": None,
            "n_valid": 0,
            "n_nan": int(np.isnan(arr).sum()),
        }

    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=0)),
        "n_valid": int(valid.size),
        "n_nan": int(np.isnan(arr).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pred_thresh", type=int, default=0)
    parser.add_argument("--gt_thresh", type=int, default=0)
    parser.add_argument("--strict_shape", action="store_true")
    parser.add_argument(
        "--include_missing_pred_as_empty",
        action="store_true",
        help="Evaluate GT files without prediction as empty prediction masks.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = {}
    for p in pred_dir.glob("*.png"):
        pred_files[norm_name(p.name)] = p

    gt_files = {}
    for p in gt_dir.glob("*.png"):
        gt_files[norm_name(p.name)] = p

    rows = []

    # Default: evaluate predictions that have GT.
    names = sorted(set(pred_files.keys()) & set(gt_files.keys()))

    # Optional: include all GT names and treat missing predictions as empty.
    if args.include_missing_pred_as_empty:
        names = sorted(gt_files.keys())

    skipped_missing_gt = sorted(set(pred_files.keys()) - set(gt_files.keys()))
    skipped_missing_pred = sorted(set(gt_files.keys()) - set(pred_files.keys()))

    for name in names:
        gt_path = gt_files[name]
        gt = load_mask(gt_path, args.gt_thresh)

        pred_path = pred_files.get(name)

        if pred_path is None:
            pred = np.zeros_like(gt, dtype=bool)
            pred_path_str = ""
            missing_pred = True
        else:
            pred = load_mask(pred_path, args.pred_thresh)
            pred_path_str = str(pred_path)
            missing_pred = False

        if pred.shape != gt.shape:
            if args.strict_shape:
                raise RuntimeError(f"Shape mismatch for {name}: pred={pred.shape}, gt={gt.shape}")
            rows.append({
                "name": name,
                "pred_path": pred_path_str,
                "gt_path": str(gt_path),
                "status": "skipped_shape_mismatch",
                "pred_sum": int(pred.sum()),
                "gt_sum": int(gt.sum()),
                "dice": np.nan,
                "iou": np.nan,
                "hd95": np.nan,
                "missing_pred": missing_pred,
            })
            continue

        d = dice_score(pred, gt)
        j = iou_score(pred, gt)
        h = hd95(pred, gt)

        rows.append({
            "name": name,
            "pred_path": pred_path_str,
            "gt_path": str(gt_path),
            "status": "ok",
            "pred_sum": int(pred.sum()),
            "gt_sum": int(gt.sum()),
            "dice": d,
            "iou": j,
            "hd95": h,
            "missing_pred": missing_pred,
        })

    csv_path = out_dir / "per_case_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "name",
            "pred_path",
            "gt_path",
            "status",
            "pred_sum",
            "gt_sum",
            "dice",
            "iou",
            "hd95",
            "missing_pred",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    ok_rows = [r for r in rows if r["status"] == "ok"]

    dice_values = [r["dice"] for r in ok_rows]
    iou_values = [r["iou"] for r in ok_rows]
    hd95_values = [r["hd95"] for r in ok_rows]

    summary = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "num_pred_files": len(pred_files),
        "num_gt_files": len(gt_files),
        "num_evaluated": len(ok_rows),
        "num_rows_total": len(rows),
        "num_missing_gt_skipped": len(skipped_missing_gt),
        "num_missing_pred_skipped": 0 if args.include_missing_pred_as_empty else len(skipped_missing_pred),
        "include_missing_pred_as_empty": bool(args.include_missing_pred_as_empty),
        "dice": summarize(dice_values),
        "iou": summarize(iou_values),
        "hd95": summarize(hd95_values),
    }

    summary_path = out_dir / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = out_dir / "summary_metrics.txt"
    with open(txt_path, "w") as f:
        f.write("Evaluation summary\n")
        f.write("==================\n")
        f.write(f"pred_dir: {pred_dir}\n")
        f.write(f"gt_dir:   {gt_dir}\n")
        f.write(f"num_pred_files: {len(pred_files)}\n")
        f.write(f"num_gt_files:   {len(gt_files)}\n")
        f.write(f"num_evaluated:  {len(ok_rows)}\n")
        f.write("\n")
        f.write(f"Mean Dice: {summary['dice']['mean']} ± {summary['dice']['std']}\n")
        f.write(f"Mean IoU:  {summary['iou']['mean']} ± {summary['iou']['std']}\n")
        f.write(f"HD95:      {summary['hd95']['mean']} ± {summary['hd95']['std']}\n")
        f.write("\n")
        f.write(f"HD95 valid n: {summary['hd95']['n_valid']}\n")
        f.write(f"HD95 NaN n:   {summary['hd95']['n_nan']}\n")

    print("============================================================")
    print("Evaluation complete")
    print("============================================================")
    print(f"Pred dir: {pred_dir}")
    print(f"GT dir:   {gt_dir}")
    print(f"Pred files: {len(pred_files)}")
    print(f"GT files:   {len(gt_files)}")
    print(f"Evaluated:  {len(ok_rows)}")
    print()
    print(f"Mean Dice: {summary['dice']['mean']:.6f} ± {summary['dice']['std']:.6f}")
    print(f"Mean IoU:  {summary['iou']['mean']:.6f} ± {summary['iou']['std']:.6f}")

    hd_mean = summary["hd95"]["mean"]
    hd_std = summary["hd95"]["std"]
    if hd_mean is None:
        print("HD95:      None")
    else:
        print(f"HD95:      {hd_mean:.6f} ± {hd_std:.6f}")

    print()
    print(f"Saved CSV:     {csv_path}")
    print(f"Saved JSON:    {summary_path}")
    print(f"Saved summary: {txt_path}")


if __name__ == "__main__":
    main()
