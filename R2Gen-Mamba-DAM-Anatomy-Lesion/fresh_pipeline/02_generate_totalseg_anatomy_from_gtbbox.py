#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# Final anatomy-location label space
# ============================================================

ANATOMY_NAME_TO_ID = {
    "unknown": 0,
    "lung": 1,
    "liver": 2,
    "kidney": 3,
    "adrenal": 4,
    "abdomen": 5,
    "pelvis": 6,
    "chest": 7,
    "brain_head_neck": 8,
    "spine": 9,
    "bone": 10,
}


# ============================================================
# TotalSegmentator class IDs
#
# This table follows the TotalSegmentator label ordering used
# by your previously generated 2-D mask files.
# ============================================================

TOTALSEG_ID_TO_NAME = {
    0: "background",
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
}

# Common rib IDs used by this TotalSegmentator label layout.
for label_id in range(92, 116):
    TOTALSEG_ID_TO_NAME[label_id] = f"rib_{label_id - 91}"


MASK_DIRS = {
    "train": Path(
        "/data/ruida/object_detection/TotalSegmentator/"
        "DeepLesion_train_2d_masks_sitk"
    ),
    "val": Path(
        "/data/ruida/object_detection/TotalSegmentator/"
        "DeepLesion_val_2d_masks_sitk"
    ),
    "test": Path(
        "/data/ruida/object_detection/TotalSegmentator/"
        "DeepLesion_testdev_2d_masks_sitk"
    ),
}


def unique_keep_order(values):
    result = []

    for value in values:
        if value not in result:
            result.append(value)

    return result


def totalseg_name_to_anatomy(name):
    """
    Convert a detailed TotalSegmentator structure into the final
    anatomy-location label space.

    This is a deterministic structural mapping. It is not yet the
    learned TotalSeg-to-oracle correction mapper.
    """
    name = str(name).lower()

    if name in {"", "background", "unknown"}:
        return "unknown"

    if "lung" in name:
        return "lung"

    if "liver" in name:
        return "liver"

    if "kidney" in name:
        return "kidney"

    if "adrenal" in name:
        return "adrenal"

    if "brain" in name or "skull" in name or "thyroid" in name:
        return "brain_head_neck"

    if (
        "vertebra" in name
        or "spinal_cord" in name
        or "sacrum" in name
    ):
        return "spine"

    if (
        "rib" in name
        or "humerus" in name
        or "scapula" in name
        or "clavicula" in name
        or "femur" in name
    ):
        return "bone"

    if (
        "hip" in name
        or "iliac_" in name
        or "urinary_bladder" in name
        or "prostate" in name
        or "gluteus" in name
        or "iliopsoas" in name
    ):
        return "pelvis"

    if (
        "spleen" in name
        or "gallbladder" in name
        or "stomach" in name
        or "pancreas" in name
        or "bowel" in name
        or "duodenum" in name
        or "colon" in name
        or "portal_vein" in name
        or "inferior_vena_cava" in name
    ):
        return "abdomen"

    if (
        "heart" in name
        or "aorta" in name
        or "pulmonary_vein" in name
        or "brachiocephalic" in name
        or "subclavian" in name
        or "carotid" in name
        or "superior_vena_cava" in name
        or "atrial_appendage" in name
        or "esophagus" in name
        or "trachea" in name
    ):
        return "chest"

    return "unknown"


def load_mask(path):
    array = np.asarray(Image.open(path))

    if array.ndim == 3:
        array = array[..., 0]

    return array.astype(np.int32)


def yolo_xywh_to_xyxy(box, width, height):
    """
    Input box:
        normalized YOLO [center_x, center_y, width, height]

    Output:
        absolute [x1, y1, x2, y2]
    """
    cx, cy, bw, bh = [float(value) for value in box]

    x1 = int(round((cx - bw / 2.0) * width))
    y1 = int(round((cy - bh / 2.0) * height))
    x2 = int(round((cx + bw / 2.0) * width))
    y2 = int(round((cy + bh / 2.0) * height))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    return [x1, y1, x2, y2]


def expand_xyxy(box, width, height, ratio):
    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    dx = int(round(box_width * ratio))
    dy = int(round(box_height * ratio))

    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(width, x2 + dx),
        min(height, y2 + dy),
    ]


def get_mask_candidates(
    mask,
    bbox,
    expand_ratio,
    min_pixels,
    max_detail_hits,
):
    height, width = mask.shape[:2]

    bbox_xyxy = yolo_xywh_to_xyxy(
        bbox,
        width,
        height,
    )

    expanded_xyxy = expand_xyxy(
        bbox_xyxy,
        width,
        height,
        expand_ratio,
    )

    x1, y1, x2, y2 = expanded_xyxy
    crop = mask[y1:y2, x1:x2]

    pixel_counts = Counter(crop.reshape(-1).tolist())
    pixel_counts.pop(0, None)

    detail_hits = []

    for label_id, pixel_count in pixel_counts.most_common(
        max_detail_hits
    ):
        if pixel_count < min_pixels:
            continue

        detail_name = TOTALSEG_ID_TO_NAME.get(
            int(label_id),
            f"totalseg_label_{int(label_id)}",
        )

        anatomy_name = totalseg_name_to_anatomy(detail_name)

        detail_hits.append(
            {
                "totalseg_label_id": int(label_id),
                "totalseg_label_name": detail_name,
                "mapped_anatomy_id": ANATOMY_NAME_TO_ID[
                    anatomy_name
                ],
                "mapped_anatomy_name": anatomy_name,
                "pixel_count": int(pixel_count),
                "bbox_xyxy_abs": bbox_xyxy,
                "bbox_xyxy_expanded": expanded_xyxy,
            }
        )

    return detail_hits


def find_mask(mask_dir, image_path):
    image_name = Path(image_path).name

    candidates = [
        mask_dir / image_name,
        mask_dir / f"{Path(image_name).stem}.png",
    ]

    stem = Path(image_name).stem

    if stem.endswith("_000"):
        candidates.append(
            mask_dir / f"{stem[:-4]}.png"
        )
    else:
        candidates.append(
            mask_dir / f"{stem}_000.png"
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def process_example(
    example,
    split,
    expand_ratio,
    min_pixels,
    max_detail_hits,
    topk_anatomy,
):
    result = deepcopy(example)

    image_paths = result.get("image_path", [])
    bboxes = result.get("bboxes", [])

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    if not isinstance(bboxes, list):
        bboxes = []

    # DeepLesion normally has two corresponding image/bbox entries.
    pair_count = min(len(image_paths), len(bboxes))

    all_detail_hits = []
    per_image_debug = []
    missing_masks = []

    for index in range(pair_count):
        image_path = image_paths[index]
        bbox = bboxes[index]

        mask_path = find_mask(
            MASK_DIRS[split],
            image_path,
        )

        debug_record = {
            "image_path": image_path,
            "bbox": bbox,
        }

        if mask_path is None:
            debug_record["status"] = "missing_mask"
            missing_masks.append(image_path)
            per_image_debug.append(debug_record)
            continue

        mask = load_mask(mask_path)

        detail_hits = get_mask_candidates(
            mask=mask,
            bbox=bbox,
            expand_ratio=expand_ratio,
            min_pixels=min_pixels,
            max_detail_hits=max_detail_hits,
        )

        debug_record["status"] = "ok"
        debug_record["mask_path"] = str(mask_path)
        debug_record["detail_hits"] = detail_hits
        per_image_debug.append(debug_record)

        for hit in detail_hits:
            hit_copy = dict(hit)
            hit_copy["image_path"] = image_path
            hit_copy["mask_path"] = str(mask_path)
            hit_copy["bbox_index"] = index
            all_detail_hits.append(hit_copy)

    # Aggregate TotalSeg overlap by the final anatomy class.
    anatomy_pixel_counts = Counter()

    for hit in all_detail_hits:
        anatomy_name = hit["mapped_anatomy_name"]

        if anatomy_name == "unknown":
            continue

        anatomy_pixel_counts[anatomy_name] += hit["pixel_count"]

    top_anatomy = [
        name
        for name, _ in anatomy_pixel_counts.most_common(topk_anatomy)
    ]

    if top_anatomy:
        primary_anatomy = top_anatomy[0]
    else:
        primary_anatomy = "unknown"

    result["totalseg_anatomy_id"] = (
        ANATOMY_NAME_TO_ID[primary_anatomy]
    )
    result["totalseg_anatomy_name"] = primary_anatomy

    result["totalseg_anatomy_ids"] = [
        ANATOMY_NAME_TO_ID[name]
        for name in top_anatomy
    ]

    result["totalseg_anatomy_names"] = top_anatomy

    result["totalseg_detail_ids"] = [
        hit["totalseg_label_id"]
        for hit in all_detail_hits[:max_detail_hits]
    ]

    result["totalseg_detail_names"] = [
        hit["totalseg_label_name"]
        for hit in all_detail_hits[:max_detail_hits]
    ]

    result["totalseg_detail_hits"] = all_detail_hits
    result["totalseg_anatomy_pixel_counts"] = dict(
        anatomy_pixel_counts
    )
    result["totalseg_bbox_debug"] = per_image_debug
    result["totalseg_missing_masks"] = missing_masks
    result["totalseg_source"] = "gt_bbox_plus_totalseg_mask"

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

    parser.add_argument(
        "--summary_json",
        required=True,
    )

    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--min_pixels",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max_detail_hits",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--topk_anatomy",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    with open(args.input_json) as file:
        source = json.load(file)

    output = {}
    summary = {}

    for split in ["train", "val", "test"]:
        output[split] = []

        anatomy_counts = Counter()
        missing_mask_samples = 0
        empty_bbox_samples = 0

        print(f"\nProcessing {split}: {len(source[split])}")

        for index, example in enumerate(source[split]):
            processed = process_example(
                example=example,
                split=split,
                expand_ratio=args.expand_ratio,
                min_pixels=args.min_pixels,
                max_detail_hits=args.max_detail_hits,
                topk_anatomy=args.topk_anatomy,
            )

            output[split].append(processed)

            anatomy_counts[
                processed["totalseg_anatomy_name"]
            ] += 1

            if processed["totalseg_missing_masks"]:
                missing_mask_samples += 1

            if not processed.get("bboxes"):
                empty_bbox_samples += 1

            if (index + 1) % 2000 == 0:
                print(
                    f"  {split}: "
                    f"{index + 1}/{len(source[split])}"
                )

        summary[split] = {
            "samples": len(output[split]),
            "missing_mask_samples": missing_mask_samples,
            "empty_bbox_samples": empty_bbox_samples,
            "anatomy_distribution": anatomy_counts.most_common(),
        }

        print(f"[{split}] missing masks:", missing_mask_samples)
        print(f"[{split}] empty bboxes:", empty_bbox_samples)
        print(
            f"[{split}] anatomy:",
            anatomy_counts.most_common(),
        )

    output_path = Path(args.output_json)
    summary_path = Path(args.summary_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(output, file, indent=2)

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    print("\nSaved JSON:", output_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
