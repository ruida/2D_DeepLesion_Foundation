#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


VALID_COARSE = {
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


def normalize(value):
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "_")
    )


def display_name(value):
    return normalize(value).replace("_", " ")


def get_totalseg_coarse(example):
    for key in [
        "totalseg_anatomy_names",
        "totalseg_anatomy_name",
        "predicted_totalseg_anatomy_names",
        "predicted_totalseg_anatomy_name",
    ]:
        value = example.get(key)

        if isinstance(value, list):
            for item in value:
                item = normalize(item)
                if item in VALID_COARSE:
                    return item

        elif value is not None:
            value = normalize(value)
            if value in VALID_COARSE:
                return value

    return "unknown"


def get_totalseg_details(example):
    for key in [
        "totalseg_detail_names",
        "predicted_totalseg_detail_names",
    ]:
        value = example.get(key)

        if isinstance(value, list):
            return [
                normalize(x)
                for x in value
                if normalize(x)
            ]

        if isinstance(value, str) and value.strip():
            return [normalize(value)]

    return []


def choose_mapping(
    coarse,
    details,
    mapping,
):
    pair_map = mapping["coarse_plus_detail_mapping"]
    detail_map = mapping["detail_mapping"]
    coarse_map = mapping["coarse_mapping"]

    candidates = []

    for detail in details:
        key = f"{coarse}|||{detail}"

        if key in pair_map:
            entry = pair_map[key]
            candidates.append((
                int(entry["total_count"]),
                float(entry["purity"]),
                "coarse+detail",
                detail,
                entry,
            ))

    if candidates:
        candidates.sort(
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )

        _, _, source, detail, entry = candidates[0]
        return source, detail, entry

    detail_candidates = []

    for detail in details:
        if detail in detail_map:
            entry = detail_map[detail]
            detail_candidates.append((
                int(entry["total_count"]),
                float(entry["purity"]),
                "detail-only",
                detail,
                entry,
            ))

    if detail_candidates:
        detail_candidates.sort(
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )

        _, _, source, detail, entry = detail_candidates[0]
        return source, detail, entry

    if coarse in coarse_map:
        return (
            "coarse-only",
            None,
            coarse_map[coarse],
        )

    return (
        "no-mapping",
        None,
        {
            "mapped_class": coarse,
            "majority_count": 0,
            "total_count": 0,
            "purity": 0.0,
        },
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
    )

    parser.add_argument(
        "--mapping_json",
        required=True,
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

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

    with open(args.input_json) as f:
        data = json.load(f)

    with open(args.mapping_json) as f:
        mapping = json.load(f)

    rows = data[args.split]

    output = copy.deepcopy(data)

    accepted = 0
    rejected = 0

    final_distribution = Counter()
    unfiltered_distribution = Counter()
    source_distribution = Counter()
    safe_source_distribution = Counter()

    for example in output[args.split]:
        coarse = get_totalseg_coarse(example)
        details = get_totalseg_details(example)

        source, detail, entry = choose_mapping(
            coarse=coarse,
            details=details,
            mapping=mapping,
        )

        mapped = normalize(
            entry.get("mapped_class", coarse)
        )

        count = int(
            entry.get(
                "total_count",
                entry.get("majority_count", 0),
            )
        )

        purity = float(entry.get("purity", 0.0))

        mapping_accepted = (
            source != "no-mapping"
            and count >= args.min_count
            and purity >= args.min_purity
        )

        if mapping_accepted:
            final = mapped
            safe_source = "accepted_train_mapping"
            accepted += 1
        else:
            final = coarse
            safe_source = "original_totalseg_fallback"
            rejected += 1

        example[
            "original_totalseg_anatomy_name"
        ] = display_name(coarse)

        example[
            "unfiltered_predicted_oracle_anatomy_name"
        ] = display_name(mapped)

        example[
            "predicted_oracle_anatomy_name"
        ] = display_name(final)

        example[
            "anatomy_mapping_source"
        ] = source

        example[
            "anatomy_mapping_detail"
        ] = detail

        example[
            "anatomy_mapping_train_count"
        ] = count

        example[
            "anatomy_mapping_train_purity"
        ] = purity

        example[
            "safe_mapping_accepted"
        ] = mapping_accepted

        example[
            "safe_anatomy_mapping_source"
        ] = safe_source

        example[
            "safe_mapping_original_source"
        ] = source

        example[
            "safe_mapping_min_count"
        ] = args.min_count

        example[
            "safe_mapping_min_purity"
        ] = args.min_purity

        final_distribution[
            display_name(final)
        ] += 1

        unfiltered_distribution[
            display_name(mapped)
        ] += 1

        source_distribution[source] += 1
        safe_source_distribution[safe_source] += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("split:", args.split)
    print("rows:", len(rows))
    print("accepted:", accepted)
    print("rejected:", rejected)
    print(
        "mapping sources:",
        dict(source_distribution),
    )
    print(
        "safe sources:",
        dict(safe_source_distribution),
    )
    print(
        "unfiltered distribution:",
        dict(unfiltered_distribution),
    )
    print(
        "final distribution:",
        dict(final_distribution),
    )
    print("saved:", output_path)


if __name__ == "__main__":
    main()
