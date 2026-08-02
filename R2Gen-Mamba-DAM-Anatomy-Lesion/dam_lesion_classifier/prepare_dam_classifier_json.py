#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


PRIMARY_CLASSES = {
    "other_lesion",
    "lymph_node",
    "mass",
    "nodule",
    "cystic",
    "opacity",
    "consolidation",
}

ATTRIBUTE_CLASSES = {
    "none",
    "low_density",
    "enhancing",
    "calcified",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_json", required=True)
    parser.add_argument("--dam_json", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument(
        "--missing_json",
        default=None,
        help="Optional JSON containing unmatched oracle IDs.",
    )
    return parser.parse_args()


def build_index(rows):
    index = {}

    for row in rows:
        sample_id = str(row.get("id", "")).strip()

        if not sample_id:
            continue

        if sample_id in index:
            raise ValueError(f"Duplicate sample ID: {sample_id}")

        index[sample_id] = row

    return index


def main():
    args = parse_args()

    with open(args.oracle_json) as f:
        oracle_data = json.load(f)

    with open(args.dam_json) as f:
        dam_data = json.load(f)

    output = {}
    missing_output = {}
    label_conflicts = {}

    for split in ["train", "val", "test"]:
        oracle_rows = oracle_data[split]
        dam_rows = dam_data[split]

        oracle_index = build_index(oracle_rows)
        dam_index = build_index(dam_rows)

        merged_rows = []
        missing_ids = []
        conflicts = []

        primary_counts = Counter()
        attribute_counts = Counter()

        for oracle_row in oracle_rows:
            sample_id = str(oracle_row["id"])

            dam_row = dam_index.get(sample_id)

            if dam_row is None:
                missing_ids.append(sample_id)
                continue

            primary = oracle_row.get(
                "oracle_primary_lesion_type"
            )
            attribute = oracle_row.get(
                "oracle_lesion_attribute"
            )
            anatomy = oracle_row.get(
                "oracle_simple_anatomy"
            )
            oracle_tokens = oracle_row.get(
                "rough_anatomy_names",
                oracle_row.get(
                    "oracle_simple_tokens",
                    [],
                ),
            )

            if primary not in PRIMARY_CLASSES:
                raise ValueError(
                    f"{split}/{sample_id}: "
                    f"invalid primary label {primary!r}"
                )

            if attribute not in ATTRIBUTE_CLASSES:
                raise ValueError(
                    f"{split}/{sample_id}: "
                    f"invalid attribute label {attribute!r}"
                )

            # Preserve the complete DAM record.
            merged = copy.deepcopy(dam_row)

            # Rename older DAM labels so they cannot be mistaken
            # for the oracle-04 classifier targets.
            for old_key in [
                "oracle_anatomy_id",
                "oracle_anatomy_name",
                "rough_anatomy_id",
                "rough_anatomy_name",
                "rough_anatomy_ids",
                "rough_anatomy_names",
                "oracle_anatomy_all_matches",
                "oracle_anatomy_matched_keywords",
                "anatomy_source",
                "oracle_lesion_type_id",
                "oracle_lesion_type_name",
                "lesion_type_id",
                "lesion_type_name",
                "oracle_lesion_type_all_matches",
                "oracle_lesion_type_matched_keywords",
                "lesion_type_source",
            ]:
                if old_key in merged:
                    merged[f"dam_original_{old_key}"] = merged.pop(
                        old_key
                    )

            # Canonical oracle-04 fields.
            merged["oracle_simple_anatomy"] = anatomy
            merged["oracle_primary_lesion_type"] = primary
            merged["oracle_lesion_attribute"] = attribute

            merged["rough_anatomy_names"] = list(
                oracle_tokens
            )

            # Explicit classifier targets.
            merged["classifier_primary_target"] = primary
            merged["classifier_attribute_target"] = attribute

            old_primary = dam_row.get(
                "lesion_type_name",
                dam_row.get("oracle_lesion_type_name"),
            )

            if (
                old_primary is not None
                and old_primary != primary
            ):
                conflicts.append(
                    {
                        "id": sample_id,
                        "dam_old_lesion_type": old_primary,
                        "oracle04_primary": primary,
                    }
                )

            primary_counts[primary] += 1
            attribute_counts[attribute] += 1
            merged_rows.append(merged)

        output[split] = merged_rows
        missing_output[split] = missing_ids
        label_conflicts[split] = conflicts

        print("\n" + "=" * 72)
        print("split:", split)
        print("oracle rows:", len(oracle_rows))
        print("DAM rows:", len(dam_rows))
        print("merged rows:", len(merged_rows))
        print("missing oracle IDs:", len(missing_ids))
        print("old-DAM/oracle04 label conflicts:", len(conflicts))

        print("\nPrimary distribution:")
        for label, count in primary_counts.most_common():
            print(f"  {label:20s} {count:6d}")

        print("\nAttribute distribution:")
        for label, count in attribute_counts.most_common():
            print(f"  {label:20s} {count:6d}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    if args.missing_json:
        missing_path = Path(args.missing_json)
    else:
        missing_path = out_path.with_name(
            out_path.stem + "_audit.json"
        )

    audit = {
        "missing_oracle_ids": missing_output,
        "dam_old_vs_oracle04_label_conflicts": label_conflicts,
    }

    with open(missing_path, "w") as f:
        json.dump(audit, f, indent=2)

    print("\nSaved merged JSON:", out_path)
    print("Saved audit JSON:", missing_path)


if __name__ == "__main__":
    main()
