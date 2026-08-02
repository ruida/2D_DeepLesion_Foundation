#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def normalize(value):
    return " ".join(
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
        .split()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--anatomy_json",
        required=True,
    )
    parser.add_argument(
        "--lesion_csv",
        required=True,
    )
    parser.add_argument(
        "--output_json",
        required=True,
    )

    args = parser.parse_args()

    anatomy_path = Path(args.anatomy_json)
    lesion_path = Path(args.lesion_csv)
    output_path = Path(args.output_json)

    if not anatomy_path.is_file():
        raise FileNotFoundError(anatomy_path.resolve())

    if not lesion_path.is_file():
        raise FileNotFoundError(lesion_path.resolve())

    data = json.load(open(anatomy_path))

    lesion_df = pd.read_csv(
        lesion_path,
        dtype={"id": str},
    )

    required_columns = {
        "id",
        "predicted",
        "confidence",
    }

    missing_columns = (
        required_columns - set(lesion_df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Missing CSV columns: {sorted(missing_columns)}"
        )

    if lesion_df["id"].duplicated().any():
        duplicate_ids = lesion_df.loc[
            lesion_df["id"].duplicated(),
            "id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate lesion prediction IDs: "
            f"{duplicate_ids[:20]}"
        )

    lesion_by_id = {}

    for row in lesion_df.to_dict("records"):
        sid = str(row["id"])

        lesion_by_id[sid] = {
            "predicted": normalize(row["predicted"]),
            "confidence": float(row["confidence"]),
            "oracle": normalize(row.get("oracle")),
            "correct": (
                int(row["correct"])
                if "correct" in row
                and pd.notna(row["correct"])
                else None
            ),
        }

    anatomy_ids = {
        str(x["id"])
        for x in data["test"]
    }

    lesion_ids = set(lesion_by_id)

    missing = sorted(anatomy_ids - lesion_ids)
    extra = sorted(lesion_ids - anatomy_ids)

    print(
        "Anatomy test IDs:",
        len(anatomy_ids),
        flush=True,
    )
    print(
        "Lesion prediction IDs:",
        len(lesion_ids),
        flush=True,
    )
    print(
        "Missing lesion predictions:",
        len(missing),
        flush=True,
    )
    print(
        "Extra lesion predictions:",
        len(extra),
        flush=True,
    )

    if missing:
        print(
            "First missing IDs:",
            missing[:20],
            flush=True,
        )
        raise RuntimeError(
            "Some anatomy test samples have no lesion prediction."
        )

    output_test = []

    lesion_distribution = Counter()
    anatomy_distribution = Counter()
    mapping_sources = Counter()

    for source in data["test"]:
        x = copy.deepcopy(source)
        sid = str(x["id"])

        lesion = lesion_by_id[sid]

        anatomy_name = normalize(
            x.get("predicted_oracle_anatomy_name")
        )

        if not anatomy_name:
            anatomy_name = "unknown"

        lesion_name = lesion["predicted"]

        if not lesion_name:
            lesion_name = "other lesion"

        rag_query_text = (
            f"anatomy {anatomy_name} "
            f"lesion type {lesion_name}"
        )

        x["statmapped_anatomy_name"] = anatomy_name
        x["crop_predicted_lesion_type"] = lesion_name
        x["crop_predicted_lesion_confidence"] = (
            lesion["confidence"]
        )
        x["crop_lesion_prediction_source"] = (
            "predicted_bbox_crop_classifier"
        )

        # These fields are for evaluation only.
        # They must not be used as RAG query features.
        x["crop_lesion_oracle_for_evaluation"] = (
            lesion["oracle"]
        )
        x["crop_lesion_correct_for_evaluation"] = (
            lesion["correct"]
        )

        x["rag_query_anatomy"] = anatomy_name
        x["rag_query_lesion_type"] = lesion_name
        x["rag_query_text"] = rag_query_text
        x["rag_query_source"] = (
            "statmapped_anatomy_plus_crop_predicted_lesion"
        )

        anatomy_distribution[anatomy_name] += 1
        lesion_distribution[lesion_name] += 1
        mapping_sources[
            x.get("safe_anatomy_mapping_source")
        ] += 1

        output_test.append(x)

    output = copy.deepcopy(data)
    output["test"] = output_test

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    summary = {
        "test_count": len(output_test),
        "anatomy_distribution":
            dict(anatomy_distribution),
        "lesion_distribution":
            dict(lesion_distribution),
        "mapping_sources":
            dict(mapping_sources),
    }

    summary_path = output_path.with_name(
        output_path.stem + "_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        "\nSaved:",
        output_path,
        flush=True,
    )
    print(
        "Saved summary:",
        summary_path,
        flush=True,
    )
    print(
        "Anatomy distribution:",
        dict(anatomy_distribution),
        flush=True,
    )
    print(
        "Lesion distribution:",
        dict(lesion_distribution),
        flush=True,
    )
    print(
        "Mapping sources:",
        dict(mapping_sources),
        flush=True,
    )

    first = output_test[0]

    print(
        "\nFirst merged example:",
        flush=True,
    )

    for key in [
        "id",
        "predicted_oracle_anatomy_name",
        "safe_mapping_accepted",
        "safe_anatomy_mapping_source",
        "crop_predicted_lesion_type",
        "crop_predicted_lesion_confidence",
        "rag_query_text",
    ]:
        print(
            key,
            ":",
            first.get(key),
            flush=True,
        )


if __name__ == "__main__":
    main()
