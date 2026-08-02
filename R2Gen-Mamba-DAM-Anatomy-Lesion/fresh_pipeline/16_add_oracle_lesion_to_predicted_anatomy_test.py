#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


VALID_LESION_TYPES = {
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
}


def normalize_label(value):
    value = str(value).strip().lower()
    value = value.replace("-", "_")
    value = "_".join(value.split())
    return value


def readable_label(value):
    mapping = {
        "lymph_node": "lymph node",
        "soft_tissue": "soft tissue",
        "other_lesion": "other lesion",
    }

    return mapping.get(value, value.replace("_", " "))


def get_oracle_lesion_type(example):
    for field in [
        "oracle_lesion_type_name",
        "lesion_type_name",
        "lesion_type",
    ]:
        if field in example and example[field] not in (None, ""):
            return normalize_label(example[field]), field

    return None, None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predicted_anatomy_json",
        required=True,
        help="JSON containing realistic predicted anatomy in test split.",
    )

    parser.add_argument(
        "--oracle_lesion_json",
        required=True,
        help="JSON containing oracle lesion type labels.",
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

    args = parser.parse_args()

    with open(args.predicted_anatomy_json, "r") as f:
        predicted_data = json.load(f)

    with open(args.oracle_lesion_json, "r") as f:
        oracle_data = json.load(f)

    oracle_by_id = {}
    detected_fields = Counter()

    for example in oracle_data["test"]:
        lesion_type, field = get_oracle_lesion_type(example)

        if lesion_type is None:
            continue

        oracle_by_id[str(example["id"])] = lesion_type
        detected_fields[field] += 1

    missing_oracle = 0
    invalid_oracle = 0
    token_lengths = Counter()
    lesion_distribution = Counter()

    for example in predicted_data["test"]:
        sample_id = str(example["id"])
        lesion_type = oracle_by_id.get(sample_id)

        if lesion_type is None:
            missing_oracle += 1
            lesion_type = "other_lesion"

        if lesion_type not in VALID_LESION_TYPES:
            invalid_oracle += 1
            lesion_type = "other_lesion"

        anatomy_text = str(
            example.get(
                "predicted_merged_anatomy_text",
                example.get("merged_anatomy_text", ""),
            )
        ).strip().lower()

        if not anatomy_text:
            anatomy_text = "lesion unknown"

        lesion_text = readable_label(lesion_type)

        combined_text = (
            f"{anatomy_text} lesion type {lesion_text}"
        )
        combined_text = " ".join(combined_text.split())

        tokens = combined_text.split()

        example["oracle_lesion_type_name"] = lesion_type
        example["oracle_lesion_type_text"] = lesion_text

        example["combined_anatomy_lesion_text"] = combined_text
        example["combined_predicted_anatomy_oracle_lesion_text"] = (
            combined_text
        )

        # Fields read by anatomy_source=rough, anatomy_encoding=text
        example["rough_anatomy_name"] = combined_text
        example["rough_anatomy_names"] = tokens
        example["anatomy_text"] = combined_text

        example["anatomy_source"] = (
            "yolo_predbbox_safe_mapped_totalseg"
        )
        example["lesion_type_source"] = "oracle_report"
        example["conditioning_source"] = (
            "predicted_anatomy_plus_oracle_lesion"
        )

        lesion_distribution[lesion_type] += 1
        token_lengths[len(tokens)] += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(predicted_data, f, indent=2)

    print("Test samples:", len(predicted_data["test"]))
    print("Oracle lesion records:", len(oracle_by_id))
    print("Detected oracle fields:", dict(detected_fields))
    print("Missing oracle lesion:", missing_oracle)
    print("Invalid oracle lesion:", invalid_oracle)
    print("Lesion distribution:", lesion_distribution.most_common())
    print("Token lengths:", dict(token_lengths))
    print("Maximum tokens:", max(token_lengths))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
