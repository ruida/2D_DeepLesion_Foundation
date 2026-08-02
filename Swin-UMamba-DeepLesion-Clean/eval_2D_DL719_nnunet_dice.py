from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


GT_DIR = Path("/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D/labelsTs")
PRED_DIR = Path("/data/ruida/segmentation/Swin-UMamba-Origin/results/Dataset719_DeepLesion2D/pred_2d_nnunet")
OUT_CSV = Path("/data/ruida/segmentation/Swin-UMamba-Origin/results/Dataset719_DeepLesion2D/pred_2d_nnunet_metrics.csv")


def load_mask(path):
    arr = np.array(Image.open(path))
    return arr > 0


def dice_score(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()

    if denom == 0:
        return 1.0

    return 2.0 * inter / denom


def iou_score(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 1.0

    return inter / union


def surface_distances(mask_a, mask_b):
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    if not mask_a.any() and not mask_b.any():
        return np.array([0.0])

    if not mask_a.any() or not mask_b.any():
        return None

    structure = np.ones((3, 3), dtype=bool)

    surface_a = np.logical_xor(
        mask_a,
        binary_erosion(mask_a, structure=structure, border_value=0)
    )
    surface_b = np.logical_xor(
        mask_b,
        binary_erosion(mask_b, structure=structure, border_value=0)
    )

    dt_to_b = distance_transform_edt(~surface_b)
    dt_to_a = distance_transform_edt(~surface_a)

    dist_a_to_b = dt_to_b[surface_a]
    dist_b_to_a = dt_to_a[surface_b]

    return np.concatenate([dist_a_to_b, dist_b_to_a])


def hd95_score(gt, pred):
    distances = surface_distances(gt, pred)

    if distances is None:
        return np.nan

    return float(np.percentile(distances, 95))


rows = []

gt_paths = sorted(GT_DIR.glob("*.png"))

for gt_path in gt_paths:
    case = gt_path.stem
    pred_path = PRED_DIR / f"{case}.png"

    if not pred_path.exists():
        print("Missing prediction:", pred_path)
        continue

    gt = load_mask(gt_path)
    pred = load_mask(pred_path)

    dice = dice_score(gt, pred)
    iou = iou_score(gt, pred)
    hd95 = hd95_score(gt, pred)

    rows.append({
        "case": case,
        "dice": dice,
        "iou": iou,
        "hausdorff95": hd95,
        "gt_pixels": int(gt.sum()),
        "pred_pixels": int(pred.sum()),
        "empty_gt": int(gt.sum() == 0),
        "empty_pred": int(pred.sum() == 0),
        "both_empty": int((gt.sum() == 0) and (pred.sum() == 0)),
    })


df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)


def print_metric(metric_name, column_name):
    values = df[column_name].dropna()

    print(f"\n{metric_name}")
    print(f"Mean {metric_name}: {values.mean():.6f}")
    print(f"Std {metric_name}: {values.std():.6f}")
    print(f"{metric_name} = {values.mean():.4f} ± {values.std():.4f}")


print("Cases evaluated:", len(df))
print("Total GT cases:", len(gt_paths))
print("Missing predictions:", len(gt_paths) - len(df))

print("Empty GT cases:", int((df["gt_pixels"] == 0).sum()))
print("Empty Pred cases:", int((df["pred_pixels"] == 0).sum()))
print("Both empty cases:", int(df["both_empty"].sum()))

print("Hausdorff95 valid cases:", int(df["hausdorff95"].notna().sum()))
print("Hausdorff95 NaN cases:", int(df["hausdorff95"].isna().sum()))

print_metric("Dice", "dice")
print_metric("IoU", "iou")
print_metric("Hausdorff95", "hausdorff95")

print("\nSaved CSV:", OUT_CSV)
