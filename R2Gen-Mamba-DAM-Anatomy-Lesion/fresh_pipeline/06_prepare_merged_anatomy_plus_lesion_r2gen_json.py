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


def normalize_label(text):
    text = str(text).strip().lower()
    text = text.replace("-", "_")
    text = "_".join(text.split())
    return text


def readable_lesion_type(name):
    name = normalize_label(name)

    readable = {
        "lymph_node": "lymph node",
        "soft_tissue": "soft tissue",
        "other_lesion": "other lesion",
    }

    return readable.get(name, name.replace("_", " "))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
        help=(
            "Input JSON containing merged_anatomy_text and "
            "oracle_lesion_type_name."
        ),
    )

    parser.add_argument(
        "--output_json",
        required=True,
        help="R2Gen-Mamba-compatible output JSON.",
    )

    parser.add_argument(
        "--separator_text",
        default="lesion type",
        help=(
            "Text inserted between anatomy and lesion type. "
            "Default: 'lesion type'."
        ),
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    summary = {}

    for split in ["train", "val", "test"]:
        if split not in data:
            continue

        missing_anatomy = 0
        missing_lesion = 0
        invalid_lesion = 0
        empty_combined = 0

        token_lengths = Counter()
        lesion_distribution = Counter()

        for ex in data[split]:
            anatomy_text = str(
                ex.get("merged_anatomy_text", "")
            ).strip().lower()

            if not anatomy_text:
                missing_anatomy += 1
                anatomy_text = "lesion unknown"

            lesion_type = normalize_label(
                ex.get(
                    "oracle_lesion_type_name",
                    ex.get("lesion_type_name", "other_lesion"),
                )
            )

            if not lesion_type:
                missing_lesion += 1
                lesion_type = "other_lesion"

            if lesion_type not in VALID_LESION_TYPES:
                invalid_lesion += 1
                lesion_type = "other_lesion"

            lesion_text = readable_lesion_type(lesion_type)

            combined_text = (
                f"{anatomy_text} "
                f"{args.separator_text.strip().lower()} "
                f"{lesion_text}"
            )

            combined_text = " ".join(combined_text.split())

            if not combined_text:
                empty_combined += 1
                combined_text = (
                    "lesion unknown lesion type other lesion"
                )

            combined_tokens = combined_text.split()

            # Existing text-anatomy loader reads this field.
            ex["rough_anatomy_names"] = combined_tokens
            ex["rough_anatomy_name"] = combined_text

            # Explicit audit fields.
            ex["anatomy_text"] = anatomy_text
            ex["lesion_type_text"] = lesion_text
            ex["combined_anatomy_lesion_text"] = combined_text

            ex["anatomy_source"] = (
                "oracle_coarse_plus_totalseg_gtbbox_top1_strict"
            )
            ex["lesion_type_source"] = "oracle_report"
            ex["conditioning_source"] = (
                "merged_anatomy_plus_oracle_lesion_type"
            )

            token_lengths[len(combined_tokens)] += 1
            lesion_distribution[lesion_type] += 1

        summary[split] = {
            "samples": len(data[split]),
            "missing_anatomy": missing_anatomy,
            "missing_lesion_type": missing_lesion,
            "invalid_lesion_type": invalid_lesion,
            "empty_combined_text": empty_combined,
            "token_length_distribution": dict(token_lengths),
            "maximum_token_length": (
                max(token_lengths) if token_lengths else 0
            ),
            "lesion_type_distribution": dict(
                lesion_distribution
            ),
        }

        print(f"\n[{split}]")
        print("samples:", len(data[split]))
        print("missing anatomy:", missing_anatomy)
        print("missing lesion type:", missing_lesion)
        print("invalid lesion type:", invalid_lesion)
        print("empty combined text:", empty_combined)
        print("token lengths:", dict(token_lengths))
        print(
            "maximum tokens:",
            summary[split]["maximum_token_length"],
        )
        print(
            "lesion types:",
            lesion_distribution.most_common(),
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    summary_path = output_path.with_suffix(".summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved JSON:", output_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
