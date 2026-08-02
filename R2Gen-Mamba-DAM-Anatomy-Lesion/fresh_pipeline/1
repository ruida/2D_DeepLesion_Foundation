#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms


FALLBACK_LABEL = "other_lesion"


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
    cx, cy, box_width, box_height = map(float, bbox)

    cx *= width
    cy *= height
    box_width *= width
    box_height *= height

    box_width *= 1.0 + 2.0 * expand_ratio
    box_height *= 1.0 + 2.0 * expand_ratio

    x1 = max(0, int(round(cx - box_width / 2.0)))
    y1 = max(0, int(round(cy - box_height / 2.0)))
    x2 = min(width, int(round(cx + box_width / 2.0)))
    y2 = min(height, int(round(cy + box_height / 2.0)))

    if x2 <= x1:
        x2 = min(width, x1 + 1)

    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )
    return model


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in [
        "model_state_dict",
        "state_dict",
        "model",
        "model_state",
    ]:
        if key in checkpoint:
            return checkpoint[key]

    # The checkpoint itself may already be a state_dict.
    if all(
        isinstance(key, str)
        for key in checkpoint.keys()
    ):
        return checkpoint

    raise ValueError(
        "Could not find model state_dict in checkpoint."
    )


def clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def load_label_map(checkpoint, label_map_path):
    if isinstance(checkpoint, dict):
        for key in [
            "label_to_id",
            "class_to_idx",
            "label_map",
        ]:
            if key in checkpoint:
                return checkpoint[key]

    if label_map_path is None:
        raise ValueError(
            "Checkpoint does not contain label mapping. "
            "Provide --label_map."
        )

    with open(label_map_path, "r") as f:
        return json.load(f)


def normalize_label_map(label_to_id):
    normalized = {}

    for label, index in label_to_id.items():
        normalized[str(label)] = int(index)

    return normalized


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
    )

    parser.add_argument(
        "--image_root",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--label_map",
        default=None,
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

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

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    label_to_id = normalize_label_map(
        load_label_map(
            checkpoint,
            args.label_map,
        )
    )

    id_to_label = {
        int(index): label
        for label, index in label_to_id.items()
    }

    print("label map:", label_to_id)

    model = build_model(
        num_classes=len(label_to_id)
    )

    state_dict = clean_state_dict(
        extract_state_dict(checkpoint)
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(
            (args.crop_size, args.crop_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    prediction_distribution = Counter()
    source_distribution = Counter()

    missing_images = 0
    invalid_boxes = 0
    fallback_samples = 0
    samples_with_one_crop = 0
    samples_with_two_crops = 0

    confidence_values = []

    with torch.no_grad():
        for example in tqdm(
            data["test"],
            desc="Predict lesion type",
        ):
            image_paths = example.get(
                "image_path",
                [],
            )

            bboxes = example.get(
                "bboxes",
                [],
            )

            if isinstance(image_paths, str):
                image_paths = [image_paths]

            if (
                isinstance(bboxes, list)
                and len(bboxes) == 4
                and all(
                    isinstance(value, (int, float))
                    for value in bboxes
                )
            ):
                bboxes = [bboxes]

            crop_tensors = []

            for image_path, bbox in zip(
                image_paths,
                bboxes,
            ):
                resolved = resolve_image(
                    args.image_root,
                    image_path,
                )

                if resolved is None:
                    missing_images += 1
                    continue

                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                ):
                    invalid_boxes += 1
                    continue

                try:
                    image = Image.open(
                        resolved
                    ).convert("RGB")

                    width, height = image.size

                    x1, y1, x2, y2 = (
                        xywh_norm_to_xyxy(
                            bbox=bbox,
                            width=width,
                            height=height,
                            expand_ratio=args.expand_ratio,
                        )
                    )

                    crop = image.crop(
                        (x1, y1, x2, y2)
                    )

                    crop_tensors.append(
                        transform(crop)
                    )

                except Exception as error:
                    print(
                        "WARNING:",
                        example.get("id"),
                        image_path,
                        error,
                    )
                    invalid_boxes += 1

            if len(crop_tensors) == 1:
                samples_with_one_crop += 1

            if len(crop_tensors) >= 2:
                samples_with_two_crops += 1

            if not crop_tensors:
                predicted_label = FALLBACK_LABEL
                confidence = 0.0
                source = "fallback_no_valid_crop"
                fallback_samples += 1

            else:
                batch = torch.stack(
                    crop_tensors
                ).to(device)

                logits = model(batch)

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                # Average paired-view probabilities.
                mean_probabilities = probabilities.mean(
                    dim=0
                )

                confidence_tensor, class_tensor = (
                    mean_probabilities.max(dim=0)
                )

                predicted_class_id = int(
                    class_tensor.item()
                )

                predicted_label = id_to_label[
                    predicted_class_id
                ]

                confidence = float(
                    confidence_tensor.item()
                )

                source = (
                    "yolo_bbox_crop_resnet18"
                )

            example[
                "predicted_lesion_type_name"
            ] = predicted_label

            example[
                "predicted_lesion_type_confidence"
            ] = confidence

            example[
                "lesion_type_prediction_source"
            ] = source

            example[
                "lesion_type_num_valid_crops"
            ] = len(crop_tensors)

            prediction_distribution[
                predicted_label
            ] += 1

            source_distribution[source] += 1

            confidence_values.append(confidence)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    mean_confidence = (
        sum(confidence_values) /
        len(confidence_values)
        if confidence_values
        else 0.0
    )

    print("\nPrediction summary")
    print("test samples:", len(data["test"]))
    print("missing images:", missing_images)
    print("invalid boxes:", invalid_boxes)
    print("fallback samples:", fallback_samples)
    print("samples with one crop:", samples_with_one_crop)
    print("samples with two crops:", samples_with_two_crops)
    print("mean confidence:", mean_confidence)
    print(
        "prediction sources:",
        source_distribution.most_common(),
    )
    print(
        "class distribution:",
        prediction_distribution.most_common(),
    )
    print("saved:", output_path)


if __name__ == "__main__":
    main()
