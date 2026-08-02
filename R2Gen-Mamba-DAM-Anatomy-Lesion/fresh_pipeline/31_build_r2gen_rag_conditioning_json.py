#!/usr/bin/env python3

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path


def normalize_text(value):
    value = (
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
    )

    value = re.sub(r"[^a-z0-9\s]", " ", value)

    return " ".join(value.split())


def tokenize_text(value):
    value = normalize_text(value)

    if not value:
        return []

    return value.split()


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

    parser.add_argument(
        "--max_condition_tokens",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    if not input_path.is_file():
        raise FileNotFoundError(
            input_path.resolve()
        )

    with open(input_path, "r") as f:
        data = json.load(f)

    output = copy.deepcopy(data)

    summary = {}

    for split in ["train", "val", "test"]:
        rows = output[split]

        lengths_before_truncation = []
        lengths_after_truncation = []

        missing_query = 0
        missing_retrieved = 0
        truncated = 0

        source_distribution = Counter()

        for example in rows:
            query_text = normalize_text(
                example.get("rag_query_text")
            )

            retrieved_report = normalize_text(
                example.get("rag_retrieved_report")
                or example.get("retrieved_report")
            )

            if not query_text:
                missing_query += 1

            if not retrieved_report:
                missing_retrieved += 1

            # Preserve the previous R2Gen conditioning fields.
            example[
                "pre_rag_rough_anatomy_names"
            ] = copy.deepcopy(
                example.get(
                    "rough_anatomy_names",
                    [],
                )
            )

            example[
                "pre_rag_anatomy_text"
            ] = example.get(
                "anatomy_text",
                "",
            )

            # Explicit markers distinguish structured input from
            # retrieved report prior.
            conditioning_text = (
                f"{query_text} "
                f"retrieved report {retrieved_report}"
            ).strip()

            conditioning_tokens = tokenize_text(
                conditioning_text
            )

            lengths_before_truncation.append(
                len(conditioning_tokens)
            )

            if (
                len(conditioning_tokens)
                > args.max_condition_tokens
            ):
                truncated += 1
                conditioning_tokens = (
                    conditioning_tokens[
                        :args.max_condition_tokens
                    ]
                )

            lengths_after_truncation.append(
                len(conditioning_tokens)
            )

            # Existing dataset loader reads this field when:
            # --anatomy_source rough
            # --anatomy_encoding text
            example["rough_anatomy_names"] = (
                conditioning_tokens
            )

            example["anatomy_text"] = " ".join(
                conditioning_tokens
            )

            example[
                "r2gen_rag_conditioning_text"
            ] = " ".join(conditioning_tokens)

            example[
                "r2gen_rag_conditioning_tokens"
            ] = conditioning_tokens

            example[
                "r2gen_rag_conditioning_source"
            ] = (
                "mapped_anatomy_predicted_lesion_"
                "plus_train_only_retrieved_report"
            )

            example[
                "r2gen_rag_max_condition_tokens"
            ] = args.max_condition_tokens

            source_distribution[
                example[
                    "r2gen_rag_conditioning_source"
                ]
            ] += 1

        summary[split] = {
            "rows": len(rows),
            "missing_query": missing_query,
            "missing_retrieved_report":
                missing_retrieved,
            "truncated_samples": truncated,
            "minimum_tokens": (
                min(lengths_after_truncation)
                if lengths_after_truncation
                else 0
            ),
            "mean_tokens_before_truncation": (
                sum(lengths_before_truncation)
                / len(lengths_before_truncation)
                if lengths_before_truncation
                else 0.0
            ),
            "mean_tokens_after_truncation": (
                sum(lengths_after_truncation)
                / len(lengths_after_truncation)
                if lengths_after_truncation
                else 0.0
            ),
            "maximum_tokens_before_truncation": (
                max(lengths_before_truncation)
                if lengths_before_truncation
                else 0
            ),
            "maximum_tokens_after_truncation": (
                max(lengths_after_truncation)
                if lengths_after_truncation
                else 0
            ),
            "source_distribution":
                dict(source_distribution),
        }

        print("\n" + "=" * 80)
        print("split:", split)
        print(json.dumps(
            summary[split],
            indent=2,
        ))

        first = rows[0]

        print("first ID:", first.get("id"))
        print(
            "first conditioning:",
            first.get(
                "r2gen_rag_conditioning_text"
            ),
        )
        print(
            "first target report:",
            first.get("report"),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    summary_path = output_path.with_name(
        output_path.stem + "_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:", output_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
