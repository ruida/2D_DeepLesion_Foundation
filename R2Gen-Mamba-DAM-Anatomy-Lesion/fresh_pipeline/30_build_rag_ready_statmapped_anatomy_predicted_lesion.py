#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


EXPECTED_COUNTS = {
    "train": 16667,
    "val": 2058,
    "test": 2026,
}


def normalize_text(value):
    return " ".join(
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
        .split()
    )


def load_split(path, split):
    with open(path, "r") as f:
        data = json.load(f)

    if split not in data:
        raise KeyError(
            f"Split '{split}' is missing from {path}. "
            f"Available keys: {list(data.keys())}"
        )

    return data[split]


def prepare_rows(rows, split):
    output = []

    anatomy_distribution = Counter()
    lesion_distribution = Counter()
    anatomy_sources = Counter()
    lesion_sources = Counter()

    missing_anatomy = []
    missing_lesion = []
    missing_report = []

    for source in rows:
        x = copy.deepcopy(source)
        sid = str(x.get("id"))

        anatomy = normalize_text(
            x.get("predicted_oracle_anatomy_name")
        )

        lesion_type = normalize_text(
            x.get("predicted_lesion_type_name")
        )

        if not anatomy:
            missing_anatomy.append(sid)
            anatomy = "unknown"

        if not lesion_type:
            missing_lesion.append(sid)
            lesion_type = "other lesion"

        report = str(x.get("report", "") or "").strip()

        if not report:
            missing_report.append(sid)

        query = (
            f"anatomy {anatomy} "
            f"lesion type {lesion_type}"
        )

        x["rag_query_anatomy"] = anatomy
        x["rag_query_lesion_type"] = lesion_type
        x["rag_query_text"] = query

        x["rag_query_source"] = (
            "statmapped_totalseg_anatomy_"
            "plus_crop_predicted_lesion_type"
        )

        x["rag_bbox_regime"] = (
            "gt_bbox"
            if split in {"train", "val"}
            else "yolo_predicted_bbox"
        )

        x["rag_split"] = split

        anatomy_distribution[anatomy] += 1
        lesion_distribution[lesion_type] += 1

        anatomy_sources[
            x.get("safe_anatomy_mapping_source")
        ] += 1

        lesion_sources[
            x.get("lesion_type_prediction_source")
        ] += 1

        output.append(x)

    summary = {
        "rows": len(output),
        "missing_anatomy": len(missing_anatomy),
        "missing_lesion": len(missing_lesion),
        "missing_report": len(missing_report),
        "anatomy_distribution": dict(
            anatomy_distribution
        ),
        "lesion_distribution": dict(
            lesion_distribution
        ),
        "anatomy_sources": dict(anatomy_sources),
        "lesion_sources": dict(lesion_sources),
        "first_missing_anatomy_ids":
            missing_anatomy[:20],
        "first_missing_lesion_ids":
            missing_lesion[:20],
        "first_missing_report_ids":
            missing_report[:20],
    }

    return output, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_json",
        required=True,
    )

    parser.add_argument(
        "--val_json",
        required=True,
    )

    parser.add_argument(
        "--test_json",
        required=True,
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

    args = parser.parse_args()

    input_paths = {
        "train": Path(args.train_json),
        "val": Path(args.val_json),
        "test": Path(args.test_json),
    }

    for split, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{split}: {path.resolve()}"
            )

    combined = {}
    summaries = {}

    for split in ["train", "val", "test"]:
        rows = load_split(
            input_paths[split],
            split,
        )

        expected = EXPECTED_COUNTS[split]

        if len(rows) != expected:
            raise RuntimeError(
                f"{split}: expected {expected} rows, "
                f"found {len(rows)}"
            )

        prepared, summary = prepare_rows(
            rows,
            split,
        )

        combined[split] = prepared
        summaries[split] = summary

        print("\n" + "=" * 80)
        print("split:", split)
        print("rows:", summary["rows"])
        print(
            "missing anatomy:",
            summary["missing_anatomy"],
        )
        print(
            "missing lesion:",
            summary["missing_lesion"],
        )
        print(
            "missing report:",
            summary["missing_report"],
        )
        print(
            "anatomy sources:",
            summary["anatomy_sources"],
        )
        print(
            "lesion sources:",
            summary["lesion_sources"],
        )

        first = prepared[0]

        print("first ID:", first.get("id"))
        print(
            "first query:",
            first.get("rag_query_text"),
        )
        print(
            "first report:",
            first.get("report"),
        )

    output_path = Path(args.output_json)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    summary_path = output_path.with_name(
        output_path.stem + "_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\nSaved:", output_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
