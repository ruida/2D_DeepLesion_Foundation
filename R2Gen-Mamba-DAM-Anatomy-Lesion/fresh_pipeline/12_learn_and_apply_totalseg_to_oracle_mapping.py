#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
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


def normalize(value):
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = value.replace("-", "_")
    value = "_".join(value.split())

    if value.startswith("lesion_"):
        value = value[len("lesion_"):]

    return value


def first_existing(example, fields, default=None):
    for field in fields:
        if field in example:
            value = example[field]

            if value not in (None, "", []):
                return value, field

    return default, None


def get_oracle_anatomy(example):
    value, field = first_existing(
        example,
        [
            "oracle_anatomy_name",
            "oracle_anatomy",
            "anatomy_name",
        ],
    )

    value = normalize(value)

    if value in VALID_CLASSES:
        return value, field

    return None, field


def get_totalseg_coarse(example):
    """
    Read only raw TotalSegmentator coarse anatomy.

    Prefer the plural list field used by the GT-box and predicted-box
    TotalSeg JSON files. Do not use rough_anatomy_name because that
    field may already contain mapped/oracle-style conditioning.
    """
    value, field = first_existing(
        example,
        [
            "totalseg_anatomy_names",
            "totalseg_anatomy_name",
            "predicted_totalseg_anatomy_names",
            "predicted_totalseg_anatomy_name",
        ],
        default="unknown",
    )

    if isinstance(value, (list, tuple)):
        normalized = []

        for item in value:
            item = normalize(item)

            if item in VALID_CLASSES and item not in normalized:
                normalized.append(item)

        # Script 12 learns one coarse class per sample.
        # Use the first raw TotalSeg coarse class, matching JSON order.
        value = normalized[0] if normalized else "unknown"
    else:
        value = normalize(value)

        if value not in VALID_CLASSES:
            value = "unknown"

    return value, field


def get_totalseg_details(example):
    value, field = first_existing(
        example,
        [
            "totalseg_detail_names",
            "merged_anatomy_detail_names",
            "totalseg_anatomy_detail_names",
            "detail_names",
        ],
        default=[],
    )

    if isinstance(value, str):
        value = [value]

    details = []

    for item in value or []:
        if isinstance(item, dict):
            item = (
                item.get("name")
                or item.get("label_name")
                or item.get("anatomy_name")
                or ""
            )

        item = normalize(item)

        if item and item not in details:
            details.append(item)

    return details, field


def majority_row(counter):
    if not counter:
        return None

    mapped_class, majority_count = counter.most_common(1)[0]
    total_count = sum(counter.values())

    return {
        "mapped_class": mapped_class,
        "majority_count": majority_count,
        "total_count": total_count,
        "purity": majority_count / total_count,
        "distribution": dict(counter),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json", required=True)
    parser.add_argument("--realistic_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--mapping_json", required=True)

    parser.add_argument(
        "--min_pair_count",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min_detail_count",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    with open(args.train_json, "r") as f:
        train_data = json.load(f)

    with open(args.realistic_json, "r") as f:
        realistic_data = json.load(f)

    pair_counts = defaultdict(Counter)
    detail_counts = defaultdict(Counter)
    coarse_counts = defaultdict(Counter)

    oracle_distribution = Counter()

    oracle_fields = Counter()
    coarse_fields = Counter()
    detail_fields = Counter()

    used_train = 0
    missing_oracle = 0

    # Learn mapping only from training split.
    for example in train_data["train"]:
        oracle, oracle_field = get_oracle_anatomy(example)
        coarse, coarse_field = get_totalseg_coarse(example)
        details, detail_field = get_totalseg_details(example)

        if oracle_field:
            oracle_fields[oracle_field] += 1

        if coarse_field:
            coarse_fields[coarse_field] += 1

        if detail_field:
            detail_fields[detail_field] += 1

        if oracle is None:
            missing_oracle += 1
            continue

        used_train += 1
        oracle_distribution[oracle] += 1

        coarse_counts[coarse][oracle] += 1

        for detail in details:
            detail_counts[detail][oracle] += 1
            pair_counts[(coarse, detail)][oracle] += 1

    coarse_mapping = {
        coarse: majority_row(counter)
        for coarse, counter in coarse_counts.items()
    }

    detail_mapping = {
        detail: majority_row(counter)
        for detail, counter in detail_counts.items()
    }

    pair_mapping = {
        pair: majority_row(counter)
        for pair, counter in pair_counts.items()
    }

    output_data = json.loads(json.dumps(realistic_data))

    source_counts = Counter()
    original_distribution = Counter()
    mapped_distribution = Counter()
    changed_pairs = Counter()

    # Apply learned mapping to realistic test split.
    for example in output_data["test"]:
        coarse, _ = get_totalseg_coarse(example)
        details, _ = get_totalseg_details(example)

        original_distribution[coarse] += 1

        predicted = None
        source = None
        selected_detail = None
        train_count = 0
        purity = 0.0

        # 1. Coarse + detail mapping.
        for detail in details:
            row = pair_mapping.get((coarse, detail))

            if (
                row is not None
                and row["total_count"] >= args.min_pair_count
            ):
                predicted = row["mapped_class"]
                source = "train_coarse_plus_detail"
                selected_detail = detail
                train_count = row["total_count"]
                purity = row["purity"]
                break

        # 2. Detail-only mapping.
        if predicted is None:
            for detail in details:
                row = detail_mapping.get(detail)

                if (
                    row is not None
                    and row["total_count"] >= args.min_detail_count
                ):
                    predicted = row["mapped_class"]
                    source = "train_detail_only"
                    selected_detail = detail
                    train_count = row["total_count"]
                    purity = row["purity"]
                    break

        # 3. Coarse-only mapping.
        if predicted is None:
            row = coarse_mapping.get(coarse)

            if row is not None:
                predicted = row["mapped_class"]
                source = "train_coarse_only"
                train_count = row["total_count"]
                purity = row["purity"]

        # 4. Keep original TotalSeg coarse anatomy.
        if predicted is None:
            predicted = coarse
            source = "fixed_totalseg_fallback"

        if predicted not in VALID_CLASSES:
            predicted = "unknown"
            source = "unknown_fallback"

        example["original_totalseg_anatomy_name"] = coarse
        example["predicted_oracle_anatomy_name"] = predicted
        example["anatomy_mapping_source"] = source
        example["anatomy_mapping_detail"] = selected_detail
        example["anatomy_mapping_train_count"] = train_count
        example["anatomy_mapping_train_purity"] = purity

        source_counts[source] += 1
        mapped_distribution[predicted] += 1
        changed_pairs[(coarse, predicted)] += 1

    mapping_output = {
        "training_rows_used": used_train,
        "training_rows_missing_oracle": missing_oracle,
        "detected_oracle_fields": dict(oracle_fields),
        "detected_totalseg_coarse_fields": dict(coarse_fields),
        "detected_totalseg_detail_fields": dict(detail_fields),
        "oracle_distribution": dict(oracle_distribution),

        "coarse_mapping": coarse_mapping,

        "detail_mapping": detail_mapping,

        "coarse_plus_detail_mapping": {
            f"{coarse}|||{detail}": row
            for (coarse, detail), row in pair_mapping.items()
        },

        "application_summary": {
            "test_samples": len(output_data["test"]),
            "mapping_sources": dict(source_counts),
            "original_totalseg_distribution": dict(
                original_distribution
            ),
            "mapped_distribution": dict(mapped_distribution),
            "changed_pairs": {
                f"{old}->{new}": count
                for (old, new), count in changed_pairs.items()
            },
        },
    }

    output_path = Path(args.output_json)
    mapping_path = Path(args.mapping_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    with open(mapping_path, "w") as f:
        json.dump(mapping_output, f, indent=2)

    print("Training rows used:", used_train)
    print("Training rows missing oracle:", missing_oracle)
    print("Oracle fields:", dict(oracle_fields))
    print("TotalSeg coarse fields:", dict(coarse_fields))
    print("TotalSeg detail fields:", dict(detail_fields))

    print("\nMapping sources:")
    for name, count in source_counts.most_common():
        print(f"  {name:30s} {count}")

    print("\nOriginal TotalSeg distribution:")
    for name, count in original_distribution.most_common():
        print(f"  {name:25s} {count}")

    print("\nMapped anatomy distribution:")
    for name, count in mapped_distribution.most_common():
        print(f"  {name:25s} {count}")

    print("\nMost common mappings:")
    for (old, new), count in changed_pairs.most_common(30):
        print(f"  {old:20s} -> {new:20s} {count}")

    print("\nSaved mapped JSON:", output_path)
    print("Saved mapping JSON:", mapping_path)


if __name__ == "__main__":
    main()
