#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


FULL_IMAGE_BOX = [0.5, 0.5, 1.0, 1.0]


def parse_yolo_label_file(label_path):
    """
    Supported YOLO prediction formats:

        class cx cy width height
        class cx cy width height confidence

    Coordinates are expected to be normalized xywh.
    """
    predictions = []

    if not label_path.exists():
        return predictions

    with open(label_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(
                    f"WARNING: malformed line skipped: "
                    f"{label_path}:{line_number}: {line}"
                )
                continue

            try:
                class_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                confidence = (
                    float(parts[5])
                    if len(parts) >= 6
                    else 1.0
                )
            except ValueError:
                print(
                    f"WARNING: non-numeric line skipped: "
                    f"{label_path}:{line_number}: {line}"
                )
                continue

            # Clamp to a valid normalized YOLO range.
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            width = min(max(width, 1e-6), 1.0)
            height = min(max(height, 1e-6), 1.0)

            predictions.append(
                {
                    "class_id": class_id,
                    "bbox": [cx, cy, width, height],
                    "confidence": confidence,
                    "label_path": str(label_path),
                }
            )

    predictions.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    return predictions


def label_path_for_image(label_dir, image_path):
    image_name = Path(image_path).name
    return label_dir / f"{Path(image_name).stem}.txt"


def paired_image_stem(stem):
    """
    Map:
        case.png       <-> case_000.png
    """
    if stem.endswith("_000"):
        return stem[:-4]

    return f"{stem}_000"


def paired_label_path(label_dir, image_path):
    stem = Path(image_path).stem
    return label_dir / f"{paired_image_stem(stem)}.txt"


def select_prediction(
    label_dir,
    image_path,
    class_id,
):
    """
    Select the highest-confidence prediction for the requested image.

    Fallback order:
      1. Exact image label file.
      2. Paired image label file.
      3. Full-image normalized box.
    """
    exact_path = label_path_for_image(label_dir, image_path)
    paired_path = paired_label_path(label_dir, image_path)

    exact_predictions = parse_yolo_label_file(exact_path)

    if class_id is not None:
        exact_predictions = [
            item
            for item in exact_predictions
            if item["class_id"] == class_id
        ]

    if exact_predictions:
        selected = exact_predictions[0]
        return {
            **selected,
            "prediction_source": "exact_label_file",
            "requested_image_path": image_path,
        }

    paired_predictions = parse_yolo_label_file(paired_path)

    if class_id is not None:
        paired_predictions = [
            item
            for item in paired_predictions
            if item["class_id"] == class_id
        ]

    if paired_predictions:
        selected = paired_predictions[0]
        return {
            **selected,
            "prediction_source": "paired_label_file",
            "requested_image_path": image_path,
        }

    return {
        "class_id": class_id if class_id is not None else 0,
        "bbox": list(FULL_IMAGE_BOX),
        "confidence": 0.0,
        "label_path": None,
        "prediction_source": "full_image_fallback",
        "requested_image_path": image_path,
    }


def process_test_sample(
    sample,
    label_dir,
    class_id,
    max_boxes,
):
    output = copy.deepcopy(sample)

    image_paths = output.get("image_path", [])

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    if not image_paths:
        raise ValueError(
            f"Sample {output.get('id')} has no image_path."
        )

    # R2Gen-Mamba normally uses two paired images.
    image_paths = image_paths[:max_boxes]

    selected_predictions = [
        select_prediction(
            label_dir=label_dir,
            image_path=image_path,
            class_id=class_id,
        )
        for image_path in image_paths
    ]

    output["bboxes"] = [
        item["bbox"]
        for item in selected_predictions
    ]

    output["bbox_confs"] = [
        item["confidence"]
        for item in selected_predictions
    ]

    output["bbox_classes"] = [
        item["class_id"]
        for item in selected_predictions
    ]

    output["bbox_prediction_sources"] = [
        item["prediction_source"]
        for item in selected_predictions
    ]

    output["bbox_label_paths"] = [
        item["label_path"]
        for item in selected_predictions
    ]

    output["bbox_source"] = "yolotlpmoe_prediction"
    output["bbox_format"] = "xywh_norm"

    # Keep the GT report only for evaluation.
    # It must not be used to construct predicted conditioning.
    output["report_usage"] = "evaluation_reference_only"

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--template_json",
        required=True,
        help="Original DeepLesion R2Gen annotation JSON.",
    )

    parser.add_argument(
        "--label_dir",
        required=True,
        help="Directory containing YOLO prediction txt files.",
    )

    parser.add_argument(
        "--out_json",
        required=True,
    )

    parser.add_argument(
        "--summary_json",
        default=None,
    )

    parser.add_argument(
        "--max_boxes",
        type=int,
        default=2,
        help=(
            "Maximum paired image boxes per sample. "
            "Use 2 for the current R2Gen-Mamba dataset."
        ),
    )

    parser.add_argument(
        "--class_id",
        type=int,
        default=0,
        help="YOLO class to retain. DeepLesion lesion class is 0.",
    )

    args = parser.parse_args()

    template_path = Path(args.template_json)
    label_dir = Path(args.label_dir)
    output_path = Path(args.out_json)

    if not template_path.exists():
        raise FileNotFoundError(template_path)

    if not label_dir.exists():
        raise FileNotFoundError(label_dir)

    with open(template_path, "r") as f:
        source = json.load(f)

    output = {
        "train": copy.deepcopy(source.get("train", [])),
        "val": copy.deepcopy(source.get("val", [])),
        "test": [],
    }

    source_counts = Counter()
    box_count_distribution = Counter()
    confidence_zero_samples = 0

    for index, sample in enumerate(source["test"]):
        processed = process_test_sample(
            sample=sample,
            label_dir=label_dir,
            class_id=args.class_id,
            max_boxes=args.max_boxes,
        )

        output["test"].append(processed)

        box_count_distribution[
            len(processed["bboxes"])
        ] += 1

        for prediction_source in processed[
            "bbox_prediction_sources"
        ]:
            source_counts[prediction_source] += 1

        if all(
            confidence == 0.0
            for confidence in processed["bbox_confs"]
        ):
            confidence_zero_samples += 1

        if (index + 1) % 500 == 0:
            print(
                f"Processed test: "
                f"{index + 1}/{len(source['test'])}"
            )

    summary = {
        "train_samples": len(output["train"]),
        "val_samples": len(output["val"]),
        "test_samples": len(output["test"]),
        "box_count_distribution": dict(
            box_count_distribution
        ),
        "prediction_source_counts": dict(source_counts),
        "samples_with_no_prediction_for_either_image": (
            confidence_zero_samples
        ),
        "class_id": args.class_id,
        "max_boxes": args.max_boxes,
        "fallback_box": FULL_IMAGE_BOX,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        summary_path = output_path.with_suffix(
            ".summary.json"
        )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nSaved JSON:", output_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
