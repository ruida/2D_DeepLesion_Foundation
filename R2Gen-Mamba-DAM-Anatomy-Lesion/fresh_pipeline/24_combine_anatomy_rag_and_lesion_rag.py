#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


def clean(value):
    return " ".join(
        str(value or "")
        .replace("_", " ")
        .strip()
        .lower()
        .split()
    )


def clean_list(values):
    if isinstance(values, str):
        values = [values]

    output = []
    seen = set()

    for value in values or []:
        value = clean(value)

        if not value or value in seen:
            continue

        output.append(value)
        seen.add(value)

    return output


def remove_existing_lesion_type(names):
    names = clean_list(names)

    output = []
    i = 0

    while i < len(names):
        if names[i] == "lesion type":
            i += 2
            continue

        if (
            i + 2 < len(names)
            and names[i] == "lesion"
            and names[i + 1] == "type"
        ):
            i += 3
            continue

        output.append(names[i])
        i += 1

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--anatomy_rag_json",
        required=True,
    )

    parser.add_argument(
        "--lesion_rag_predictions",
        required=True,
    )

    parser.add_argument(
        "--oracle_trainval_json",
        required=True,
    )

    parser.add_argument(
        "--lesion_threshold",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--out_json",
        required=True,
    )

    args = parser.parse_args()

    for path_string in [
        args.anatomy_rag_json,
        args.lesion_rag_predictions,
        args.oracle_trainval_json,
    ]:
        path = Path(path_string)

        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path.resolve()}"
            )

    anatomy_data = json.load(
        open(args.anatomy_rag_json)
    )

    lesion_predictions = json.load(
        open(args.lesion_rag_predictions)
    )

    oracle_trainval = json.load(
        open(args.oracle_trainval_json)
    )

    lesion_by_id = {
        str(x["id"]): x
        for x in lesion_predictions
    }

    anatomy_test_ids = {
        str(x["id"])
        for x in anatomy_data["test"]
    }

    lesion_ids = set(lesion_by_id)

    missing = sorted(anatomy_test_ids - lesion_ids)
    extra = sorted(lesion_ids - anatomy_test_ids)

    print("Anatomy RAG test:", len(anatomy_data["test"]))
    print("Lesion RAG predictions:", len(lesion_predictions))
    print("Missing lesion predictions:", len(missing))
    print("Extra lesion predictions:", len(extra))

    if missing:
        print("First missing IDs:", missing[:20])
        raise RuntimeError(
            "Some anatomy-RAG test samples do not have "
            "lesion-RAG predictions."
        )

    output_test = []
    counts = Counter()
    anatomy_sources = Counter()
    lesion_distribution = Counter()

    for source in anatomy_data["test"]:
        x = copy.deepcopy(source)
        sid = str(x["id"])

        lesion_pred = lesion_by_id[sid]

        names = remove_existing_lesion_type(
            x.get("rough_anatomy_names", [])
        )

        if not names:
            fallback_text = clean(
                x.get("anatomy_text")
            )

            if fallback_text:
                names = fallback_text.split()

        if "lesion" not in names:
            names = ["lesion"] + names

        predicted_lesion = clean(
            lesion_pred.get("predicted")
        )

        confidence = float(
            lesion_pred.get("confidence", 0.0)
        )

        use_lesion = (
            bool(predicted_lesion)
            and confidence >= args.lesion_threshold
        )

        original_anatomy_source = x.get(
            "conditioning_source",
            "unknown_anatomy_source",
        )

        anatomy_sources[original_anatomy_source] += 1

        if use_lesion:
            names.extend([
                "lesion type",
                predicted_lesion,
            ])

            counts["rag_lesion_used"] += 1
            lesion_distribution[predicted_lesion] += 1

            final_source = (
                "gated_anatomy_rag_plus_"
                "gated_lesion_rag"
            )
        else:
            counts[
                "anatomy_only_low_lesion_confidence"
            ] += 1

            final_source = (
                "gated_anatomy_rag_plus_"
                "anatomy_only_lesion_fallback"
            )

        names = clean_list(names)
        anatomy_text = " ".join(names)

        # Fields consumed by the R2Gen text-conditioning loader.
        x["rough_anatomy_names"] = names
        x["rough_anatomy_name"] = anatomy_text
        x["anatomy_text"] = anatomy_text

        # Preserve anatomy-RAG information.
        x["joint_rag_anatomy_source"] = (
            original_anatomy_source
        )

        # Add lesion-RAG information.
        x["rag_lesion_type"] = predicted_lesion
        x["rag_lesion_confidence"] = confidence
        x["rag_lesion_threshold"] = (
            args.lesion_threshold
        )
        x["rag_lesion_used"] = use_lesion

        x["rag_lesion_vote_share"] = (
            lesion_pred.get("vote_share")
        )
        x["rag_lesion_vote_margin"] = (
            lesion_pred.get("vote_margin")
        )
        x["rag_lesion_top_similarity"] = (
            lesion_pred.get("top_similarity")
        )

        retrieved = lesion_pred.get(
            "retrieved",
            [],
        )

        x["rag_lesion_retrieved_ids"] = [
            r.get("id")
            for r in retrieved
        ]

        x["rag_lesion_retrieved_types"] = [
            r.get("lesion_type")
            for r in retrieved
        ]

        x["rag_lesion_retrieved_similarities"] = [
            r.get("similarity")
            for r in retrieved
        ]

        x["conditioning_source"] = final_source

        output_test.append(x)

    counts["total"] = len(output_test)

    output = {
        "train": copy.deepcopy(
            oracle_trainval["train"]
        ),
        "val": copy.deepcopy(
            oracle_trainval["val"]
        ),
        "test": output_test,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    summary = {
        "lesion_threshold":
            args.lesion_threshold,
        "source_counts":
            dict(counts),
        "anatomy_source_counts":
            dict(anatomy_sources),
        "lesion_distribution":
            dict(lesion_distribution),
        "train_count":
            len(output["train"]),
        "val_count":
            len(output["val"]),
        "test_count":
            len(output["test"]),
    }

    summary_path = out_path.with_name(
        out_path.stem + "_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved joint JSON:", out_path)
    print("Saved summary:", summary_path)
    print("Source counts:", dict(counts))
    print(
        "Anatomy source counts:",
        dict(anatomy_sources),
    )
    print(
        "Lesion distribution:",
        dict(lesion_distribution),
    )

    first = output_test[0]

    print("\nFirst test example:")

    for key in [
        "id",
        "bbox_source",
        "joint_rag_anatomy_source",
        "rag_final_anatomy_names",
        "rag_lesion_type",
        "rag_lesion_confidence",
        "rag_lesion_used",
        "rough_anatomy_names",
        "anatomy_text",
        "conditioning_source",
    ]:
        print(key, ":", first.get(key))


if __name__ == "__main__":
    main()
