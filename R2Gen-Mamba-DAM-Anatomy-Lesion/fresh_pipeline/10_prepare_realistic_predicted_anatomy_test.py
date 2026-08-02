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
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    lengths = Counter()
    fallback = 0
    missing = 0

    for ex in data["test"]:
        if "merged_anatomy_text" not in ex:
            missing += 1

        text = str(
            ex.get("merged_anatomy_text", "")
        ).strip().lower()

        if not text:
            text = "lesion unknown"
            fallback += 1

        tokens = text.split()

        ex["predicted_merged_anatomy_text"] = text
        ex["rough_anatomy_names"] = tokens
        ex["rough_anatomy_name"] = text
        ex["anatomy_text"] = text

        ex["anatomy_source"] = (
            "yolotlpmoe_predbbox_plus_totalseg_top1"
        )

        ex["conditioning_source"] = (
            "realistic_predicted_anatomy_only"
        )

        lengths[len(tokens)] += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Saved:", output_path)
    print("test:", len(data["test"]))
    print("missing merged field:", missing)
    print("fallback:", fallback)
    print("token lengths:", dict(lengths))
    print("maximum:", max(lengths) if lengths else 0)


if __name__ == "__main__":
    main()
