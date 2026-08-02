#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
        help="JSON containing merged_anatomy_text.",
    )

    parser.add_argument(
        "--output_json",
        required=True,
        help="R2Gen-Mamba-compatible output JSON.",
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    summary = {}

    for split in ["train", "val", "test"]:
        missing = 0
        empty = 0
        token_lengths = Counter()

        for ex in data[split]:
            merged_text = str(
                ex.get("merged_anatomy_text", "")
            ).strip().lower()

            if "merged_anatomy_text" not in ex:
                missing += 1

            if not merged_text:
                empty += 1
                merged_text = "lesion unknown"

            # The current DeepLesion loader joins rough_anatomy_names,
            # cleans the text, then tokenizes it word-by-word.
            tokens = merged_text.split()

            ex["rough_anatomy_names"] = tokens
            ex["rough_anatomy_name"] = " ".join(tokens)

            # Keep these aliases for clarity and auditing.
            ex["anatomy_text"] = " ".join(tokens)
            ex["anatomy_source"] = (
                "oracle_coarse_plus_totalseg_gtbbox_top1_strict"
            )

            token_lengths[len(tokens)] += 1

        summary[split] = {
            "samples": len(data[split]),
            "missing_merged_anatomy_text": missing,
            "empty_merged_anatomy_text": empty,
            "token_length_distribution": dict(token_lengths),
            "max_token_length": max(token_lengths)
            if token_lengths else 0,
        }

        print(f"\n[{split}]")
        print("samples:", len(data[split]))
        print("missing:", missing)
        print("empty:", empty)
        print("token lengths:", dict(token_lengths))
        print("maximum tokens:", summary[split]["max_token_length"])

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    summary_path = output_path.with_suffix(".summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:", output_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
