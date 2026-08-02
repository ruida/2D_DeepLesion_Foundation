#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trainval_json",
        required=True,
    )

    parser.add_argument(
        "--realistic_test_json",
        required=True,
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

    args = parser.parse_args()

    with open(args.trainval_json, "r") as f:
        trainval = json.load(f)

    with open(args.realistic_test_json, "r") as f:
        realistic = json.load(f)

    output = {
        "train": trainval["train"],
        "val": trainval["val"],
        "test": realistic["test"],
    }

    print("train:", len(output["train"]))
    print("val:", len(output["val"]))
    print("test:", len(output["test"]))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
