import os
import shutil
from pathlib import Path

pred_dir = Path("/data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs/full_masks")
gt_dir   = Path("/data/ruida/SAM/sam3/data/labels_filtered")
out_dir  = Path("/data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs/gt_masks_matched")

out_dir.mkdir(parents=True, exist_ok=True)

def normalize_name(name):
    """
    000022_03_01_063_000.png -> 000022_03_01_063.png
    000022_03_01_063.png     -> 000022_03_01_063.png
    """
    stem = Path(name).stem
    if stem.endswith("_000"):
        stem = stem[:-4]
    return stem + ".png"

copied = 0
missing = 0
seen = set()

for pred_path in sorted(pred_dir.glob("*.png")):
    gt_name = normalize_name(pred_path.name)

    # avoid duplicate copy for xxx.png and xxx_000.png
    if gt_name in seen:
        continue
    seen.add(gt_name)

    gt_path = gt_dir / gt_name

    if gt_path.exists():
        shutil.copy2(gt_path, out_dir / gt_name)
        copied += 1
    else:
        print(f"[MISSING GT] pred={pred_path.name} expected_gt={gt_name}")
        missing += 1

print("====================================")
print(f"Prediction files checked : {len(list(pred_dir.glob('*.png')))}")
print(f"Unique GT names expected : {len(seen)}")
print(f"GT masks copied          : {copied}")
print(f"Missing GT masks         : {missing}")
print(f"Output dir               : {out_dir}")
print("====================================")
