#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


CLASSES = [
    "lymph_node",
    "soft_tissue",
    "nodule",
    "mass",
    "opacity",
    "cystic",
    "enhancing",
    "calcified",
    "metastatic",
    "other_lesion",
]

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(CLASSES)
}


def normalize_label(value):
    value = str(value or "").strip().lower()
    value = value.replace("-", "_")
    value = "_".join(value.split())

    aliases = {
        "lymphnode": "lymph_node",
        "lymph_node": "lymph_node",
        "soft_tissue": "soft_tissue",
        "nodule": "nodule",
        "mass": "mass",
        "opacity": "opacity",
        "cyst": "cystic",
        "cystic": "cystic",
        "enhancement": "enhancing",
        "enhancing": "enhancing",
        "calcification": "calcified",
        "calcified": "calcified",
        "metastasis": "metastatic",
        "metastases": "metastatic",
        "metastatic": "metastatic",
        "lesion": "other_lesion",
        "other": "other_lesion",
        "other_lesion": "other_lesion",
    }

    return aliases.get(value, "other_lesion")


def resolve_image(image_root, image_path):
    image_root = Path(image_root)
    image_path = Path(str(image_path))

    candidates = [
        image_path,
        image_root / image_path,
        image_root / image_path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def xywh_norm_to_xyxy(bbox, width, height, expand_ratio):
    cx, cy, bw, bh = map(float, bbox)

    cx *= width
    cy *= height
    bw *= width
    bh *= height

    bw *= 1.0 + 2.0 * expand_ratio
    bh *= 1.0 + 2.0 * expand_ratio

    x1 = max(0, int(round(cx - bw / 2.0)))
    y1 = max(0, int(round(cy - bh / 2.0)))
    x2 = min(width, int(round(cx + bw / 2.0)))
    y2 = min(height, int(round(cy + bh / 2.0)))

    if x2 <= x1:
        x2 = min(width, x1 + 1)

    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_root", required=True)

    parser.add_argument(
        "--expand_ratio",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=224,
    )

    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    output_root = Path(args.output_root)
    labels_dir = output_root / "labels"
    crops_dir = output_root / "crops"

    labels_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    with open(output_root / "label_map.json", "w") as f:
        json.dump(LABEL_TO_ID, f, indent=2)

    for split in ["train", "val", "test"]:
        split_crop_dir = crops_dir / split
        split_crop_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        distribution = Counter()
        missing_images = 0
        invalid_boxes = 0

        for example in data[split]:
            sample_id = str(example["id"])

            label = normalize_label(
                example.get(
                    "oracle_lesion_type_name",
                    example.get(
                        "lesion_type_name",
                        "other_lesion",
                    ),
                )
            )

            image_paths = example.get("image_path", [])
            bboxes = example.get("bboxes", [])

            if isinstance(image_paths, str):
                image_paths = [image_paths]

            if isinstance(bboxes, list) and len(bboxes) == 4:
                if all(
                    isinstance(x, (int, float))
                    for x in bboxes
                ):
                    bboxes = [bboxes]

            for view_index, (image_path, bbox) in enumerate(
                zip(image_paths, bboxes)
            ):
                resolved = resolve_image(
                    args.image_root,
                    image_path,
                )

                if resolved is None:
                    missing_images += 1
                    continue

                if not isinstance(bbox, list) or len(bbox) != 4:
                    invalid_boxes += 1
                    continue

                try:
                    image = Image.open(resolved).convert("RGB")
                    width, height = image.size

                    x1, y1, x2, y2 = xywh_norm_to_xyxy(
                        bbox,
                        width,
                        height,
                        args.expand_ratio,
                    )

                    crop = image.crop((x1, y1, x2, y2))
                    crop = crop.resize(
                        (args.crop_size, args.crop_size)
                    )
                except Exception as exc:
                    print(
                        "WARNING:",
                        sample_id,
                        image_path,
                        exc,
                    )
                    invalid_boxes += 1
                    continue

                crop_path = (
                    split_crop_dir /
                    f"{sample_id}_view{view_index}.png"
                )

                crop.save(crop_path)

                rows.append({
                    "id": sample_id,
                    "view_index": view_index,
                    "source_image": str(resolved),
                    "crop_path": str(crop_path.resolve()),
                    "label_name": label,
                    "label_id": LABEL_TO_ID[label],
                })

                distribution[label] += 1

        csv_path = labels_dir / f"{split}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "view_index",
                    "source_image",
                    "crop_path",
                    "label_name",
                    "label_id",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n[{split}]")
        print("samples:", len(data[split]))
        print("crops:", len(rows))
        print("missing images:", missing_images)
        print("invalid boxes:", invalid_boxes)
        print("distribution:", distribution.most_common())
        print("saved:", csv_path)

    print("\nSaved label map:", output_root / "label_map.json")


if __name__ == "__main__":
    main()
