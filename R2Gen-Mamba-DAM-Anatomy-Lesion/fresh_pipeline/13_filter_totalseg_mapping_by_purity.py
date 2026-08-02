#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


VALID_CLASSES = {
    "lung",
    "liver",
    "kidney",
    "adrenal",
    "abdomen",
    "pelvis",
    "chest",
    "brain_head_neck",
    "spine",
    "bone",
    "unknown",
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)

    parser.add_argument(
        "--min_purity",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    source_counts = Counter()
    final_distribution = Counter()
    changed_pairs = Counter()
    rejected_pairs = Counter()

    for ex in data["test"]:
        original = ex.get(
            "original_totalseg_anatomy_name",
            ex.get("totalseg_anatomy_name", "unknown"),
        )

        mapped = ex.get(
            "predicted_oracle_anatomy_name",
            original,
        )

        purity = float(
            ex.get("anatomy_mapping_train_purity", 0.0)
        )

        count = int(
            ex.get("anatomy_mapping_train_count", 0)
        )

        mapping_source = ex.get(
            "anatomy_mapping_source",
            "missing",
        )

        accept = (
            mapped in VALID_CLASSES
            and count >= args.min_count
            and purity >= args.min_purity
        )

        if accept:
            final_class = mapped
            final_source = "accepted_train_mapping"
        else:
            final_class = original
            final_source = "original_totalseg_fallback"

            if mapped != original:
                rejected_pairs[(original, mapped)] += 1

        if final_class not in VALID_CLASSES:
            final_class = "unknown"
            final_source = "unknown_fallback"

        ex["unfiltered_predicted_oracle_anatomy_name"] = mapped

        ex["predicted_oracle_anatomy_name"] = final_class

        ex["safe_anatomy_mapping_source"] = final_source

        ex["safe_mapping_original_source"] = mapping_source

        ex["safe_mapping_accepted"] = accept

        ex["safe_mapping_min_purity"] = args.min_purity
        ex["safe_mapping_min_count"] = args.min_count

        source_counts[final_source] += 1
        final_distribution[final_class] += 1
        changed_pairs[(original, final_class)] += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Test samples:", len(data["test"]))

    print("\nFinal mapping sources:")
    for name, count in source_counts.most_common():
        print(f"  {name:35s} {count}")

    print("\nFinal anatomy distribution:")
    for name, count in final_distribution.most_common():
        print(f"  {name:25s} {count}")

    print("\nMost common final mappings:")
    for (old, new), count in changed_pairs.most_common(30):
        print(f"  {old:20s} -> {new:20s} {count}")

    print("\nRejected low-confidence mappings:")
    for (old, new), count in rejected_pairs.most_common(30):
        print(f"  {old:20s} -> {new:20s} {count}")

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
