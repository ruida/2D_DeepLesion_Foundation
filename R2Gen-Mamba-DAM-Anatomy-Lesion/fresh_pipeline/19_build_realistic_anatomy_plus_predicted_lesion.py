#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


def normalize_text(value):
    """Convert underscores to spaces and normalize whitespace."""
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = value.replace("_", " ")
    return " ".join(value.split())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine realistic predicted anatomy with predicted lesion type "
            "using the same R2Gen conditioning format as the training JSON."
        )
    )

    parser.add_argument(
        "--input_json",
        required=True,
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

    parser.add_argument(
        "--label_map",
        default=(
            "data/lesion_type_cls_expand025_10class/"
            "label_map.json"
        ),
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        source_data = json.load(f)

    with open(args.label_map, "r") as f:
        label_to_id = {
            str(label): int(index)
            for label, index in json.load(f).items()
        }

    if "test" not in source_data:
        raise KeyError("Input JSON does not contain a test split.")

    output_data = copy.deepcopy(source_data)

    lesion_distribution = Counter()
    token_length_distribution = Counter()

    missing_anatomy = 0
    missing_lesion = 0

    for example in output_data["test"]:
        # Prefer the final predicted merged anatomy field.
        anatomy_text = normalize_text(
            example.get("predicted_merged_anatomy_text")
            or example.get("merged_anatomy_text")
            or example.get("anatomy_text")
            or example.get("rough_anatomy_name")
        )

        lesion_label = str(
            example.get(
                "predicted_lesion_type_name",
                "other_lesion",
            )
        ).strip()

        if not anatomy_text:
            anatomy_text = "lesion unknown"
            missing_anatomy += 1

        if not lesion_label:
            lesion_label = "other_lesion"
            missing_lesion += 1

        lesion_text = normalize_text(lesion_label)

        if lesion_label not in label_to_id:
            print(
                "WARNING: lesion label is absent from label map:",
                lesion_label,
                "for sample",
                example.get("id"),
            )
            lesion_label = "other_lesion"
            lesion_text = "other lesion"

        lesion_type_id = label_to_id[lesion_label]

        combined_text = (
            f"{anatomy_text} lesion type {lesion_text}"
        )

        combined_tokens = combined_text.split()

        # Match the exact field meanings used in the training JSON.
        example["anatomy_text"] = anatomy_text

        example["lesion_type_id"] = lesion_type_id
        example["lesion_type_name"] = lesion_label
        example["lesion_type_text"] = lesion_text
        example["lesion_type_source"] = (
            example.get(
                "lesion_type_prediction_source",
                "predicted_lesion_classifier",
            )
        )

        example["combined_anatomy_lesion_text"] = (
            combined_text
        )

        # These are the fields consumed by the current R2Gen loader.
        example["rough_anatomy_name"] = combined_text
        example["rough_anatomy_names"] = combined_tokens

        # Preserve an existing anatomy ID when available.
        existing_id = example.get("rough_anatomy_id")

        if existing_id is None:
            anatomy_ids = example.get("anatomy_ids", [])
            if isinstance(anatomy_ids, list) and anatomy_ids:
                existing_id = anatomy_ids[0]
            else:
                existing_id = 0

        example["rough_anatomy_id"] = int(existing_id)
        example["rough_anatomy_ids"] = [
            int(existing_id)
        ]

        example["conditioning_source"] = (
            "yolotlpmoe_predbbox_plus_totalseg_"
            "mapped_anatomy_plus_resnet18_"
            "predicted_lesion"
        )

        lesion_distribution[lesion_label] += 1
        token_length_distribution[
            len(combined_tokens)
        ] += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Saved:", output_path)
    print("test samples:", len(output_data["test"]))
    print("missing anatomy:", missing_anatomy)
    print("missing lesion:", missing_lesion)
    print(
        "lesion distribution:",
        lesion_distribution.most_common(),
    )
    print(
        "token-length distribution:",
        sorted(token_length_distribution.items()),
    )

    print("\nFirst five examples:")
    for example in output_data["test"][:5]:
        print(
            example.get("id"),
            "|",
            example["rough_anatomy_name"],
        )
        print(
            "  rough_anatomy_names:",
            example["rough_anatomy_names"],
        )
        print(
            "  lesion_type_id:",
            example["lesion_type_id"],
        )


if __name__ == "__main__":
    main()
