#!/usr/bin/env python3

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


YOLO_IMAGE_DIR = Path(
    "/data/ruida/object_detection/YOLO-TLP/data/"
    "VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images"
)

PRED_DIR = Path(
    "/data/ruida/segmentation/Swin-UMamba-Origin/"
    "roi_outputs_top1/full_masks"
)

NNUNET_IMAGE_DIR = Path(
    "/data/ruida/segmentation/Swin-UMamba-Origin/"
    "data/nnUNet_raw/Dataset719_DeepLesion2D/imagesTs"
)

NNUNET_GT_DIR = Path(
    "/data/ruida/segmentation/Swin-UMamba-Origin/"
    "data/nnUNet_raw/Dataset719_DeepLesion2D/labelsTs"
)

OUT_GT_DIR = Path(
    "/data/ruida/segmentation/Swin-UMamba-Origin/"
    "roi_outputs_top1/gt_masks_content_matched"
)

OUT_MAPPING = Path(
    "/data/ruida/segmentation/Swin-UMamba-Origin/"
    "roi_outputs_top1/content_match_mapping.json"
)


def pixel_hash(path: Path) -> str:
    """Hash normalized grayscale pixel content rather than encoded file bytes."""
    arr = np.asarray(Image.open(path).convert("L"))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def case_id_from_nnunet_image(path: Path) -> str:
    stem = path.stem

    # nnUNet image modality suffix.
    if stem.endswith("_0000"):
        stem = stem[:-5]

    return stem


def find_gt(case_id: str) -> Path | None:
    candidates = [
        NNUNET_GT_DIR / f"{case_id}.png",
        NNUNET_GT_DIR / f"{case_id}_000.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def main():
    OUT_GT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale matched masks from a previous attempt.
    for old in OUT_GT_DIR.glob("*.png"):
        old.unlink()

    print("Indexing Dataset719 test images by pixel content...")

    hash_to_nnunet = defaultdict(list)

    nnunet_images = sorted(
        p for p in NNUNET_IMAGE_DIR.iterdir()
        if p.is_file()
    )

    for index, image_path in enumerate(nnunet_images, start=1):
        try:
            digest = pixel_hash(image_path)
            hash_to_nnunet[digest].append(image_path)
        except Exception as exc:
            print(f"Could not hash Dataset719 image {image_path}: {exc}")

        if index % 500 == 0:
            print(f"Indexed {index}/{len(nnunet_images)} Dataset719 images")

    pred_paths = sorted(PRED_DIR.glob("*.png"))

    records = []
    matched = 0
    unmatched = 0
    ambiguous = 0
    missing_source = 0
    missing_gt = 0

    print(f"Matching {len(pred_paths)} top-1 predictions...")

    for index, pred_path in enumerate(pred_paths, start=1):
        source_image = YOLO_IMAGE_DIR / pred_path.name

        if not source_image.exists():
            missing_source += 1
            records.append({
                "prediction": pred_path.name,
                "status": "missing_source_image",
            })
            continue

        digest = pixel_hash(source_image)
        candidates = hash_to_nnunet.get(digest, [])

        if len(candidates) == 0:
            unmatched += 1
            records.append({
                "prediction": pred_path.name,
                "source_image": str(source_image),
                "status": "no_pixel_match",
            })
            continue

        if len(candidates) > 1:
            ambiguous += 1
            records.append({
                "prediction": pred_path.name,
                "source_image": str(source_image),
                "status": "ambiguous_pixel_match",
                "candidates": [str(x) for x in candidates],
            })
            continue

        nnunet_image = candidates[0]
        case_id = case_id_from_nnunet_image(nnunet_image)
        gt_path = find_gt(case_id)

        if gt_path is None:
            missing_gt += 1
            records.append({
                "prediction": pred_path.name,
                "source_image": str(source_image),
                "nnunet_image": str(nnunet_image),
                "case_id": case_id,
                "status": "missing_gt",
            })
            continue

        # Rename the copied GT mask to exactly match the prediction filename.
        out_gt = OUT_GT_DIR / pred_path.name
        shutil.copy2(gt_path, out_gt)

        matched += 1
        records.append({
            "prediction": pred_path.name,
            "source_image": str(source_image),
            "nnunet_image": str(nnunet_image),
            "gt_source": str(gt_path),
            "gt_output": str(out_gt),
            "status": "matched",
        })

        if index % 500 == 0:
            print(f"Processed {index}/{len(pred_paths)} predictions")

    summary = {
        "prediction_count": len(pred_paths),
        "dataset719_image_count": len(nnunet_images),
        "matched": matched,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "missing_source": missing_source,
        "missing_gt": missing_gt,
        "output_gt_dir": str(OUT_GT_DIR),
    }

    OUT_MAPPING.write_text(
        json.dumps(
            {
                "summary": summary,
                "records": records,
            },
            indent=2,
        )
    )

    print("\n========================================")
    for key, value in summary.items():
        print(f"{key:24s}: {value}")
    print(f"Mapping JSON            : {OUT_MAPPING}")
    print("========================================")


if __name__ == "__main__":
    main()
