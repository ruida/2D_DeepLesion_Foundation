#!/usr/bin/env python3

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_image_entries(item):
    """
    Return the image paths as a list while remembering whether the original
    field was a string or a list.
    """
    value = item.get("image_path")

    if isinstance(value, str):
        return [value], "string"

    if isinstance(value, list):
        return value, "list"

    raise TypeError(
        f"Unsupported image_path value for sample {item.get('id')}: {value!r}"
    )


def build_image_index(image_root):
    """
    Index all images below image_root by basename.

    Example:
      000004_01_01_007.png
        -> VisDrone2019-DET-train/images/000004_01_01_007.png
    """
    index = defaultdict(list)

    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.name].append(path)

    return index


def preferred_split_terms(split):
    """
    Terms used to select the appropriate image when the same basename occurs
    in more than one dataset split.
    """
    if split == "train":
        return ["train"]

    if split in {"val", "valid", "validation"}:
        return ["val", "valid"]

    if split == "test":
        return ["test-dev", "test_dev", "testdev", "test"]

    return [split]


def choose_candidate(candidates, split):
    if len(candidates) == 1:
        return candidates[0], "unique"

    terms = preferred_split_terms(split)

    split_matches = [
        path
        for path in candidates
        if any(term in str(path).lower() for term in terms)
    ]

    if len(split_matches) == 1:
        return split_matches[0], "split_match"

    return None, "ambiguous"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite DeepLesion annotation image_path fields so they point "
            "to matching images under a new image root."
        )
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--keep_all_images",
        action="store_true",
        help=(
            "Map every image in image_path. Without this flag, map only the "
            "first image because the current MedGemma loader uses [0]."
        ),
    )
    parser.add_argument(
        "--drop_unmatched",
        action="store_true",
        help="Drop samples whose image cannot be matched.",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json).resolve()
    image_root = Path(args.image_root).resolve()
    output_json = Path(args.output_json).resolve()

    if not input_json.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root not found: {image_root}")

    print(f"Reading annotation: {input_json}")
    with input_json.open("r", encoding="utf-8") as f:
        original_data = json.load(f)

    if not isinstance(original_data, dict):
        raise TypeError(
            "Expected a dictionary containing train/val/test split lists."
        )

    print(f"Indexing images under: {image_root}")
    image_index = build_image_index(image_root)

    total_indexed = sum(len(paths) for paths in image_index.values())
    duplicated_names = {
        name: paths
        for name, paths in image_index.items()
        if len(paths) > 1
    }

    print(f"Indexed image files: {total_indexed}")
    print(f"Unique basenames:     {len(image_index)}")
    print(f"Duplicate basenames:  {len(duplicated_names)}")

    converted_data = copy.deepcopy(original_data)
    global_missing = []
    global_ambiguous = []

    for split, records in original_data.items():
        if not isinstance(records, list):
            converted_data[split] = records
            continue

        converted_records = []
        found_samples = 0
        unmatched_samples = 0
        ambiguous_samples = 0
        mapped_image_count = 0

        for item_index, original_item in enumerate(records):
            item = copy.deepcopy(original_item)
            image_entries, original_type = get_image_entries(original_item)

            if not image_entries:
                unmatched_samples += 1
                global_missing.append(
                    {
                        "split": split,
                        "index": item_index,
                        "id": item.get("id"),
                        "reason": "empty image_path",
                    }
                )

                if not args.drop_unmatched:
                    converted_records.append(item)
                continue

            entries_to_map = (
                image_entries
                if args.keep_all_images
                else image_entries[:1]
            )

            new_paths = []
            sample_failed = False
            sample_ambiguous = False

            for old_path in entries_to_map:
                filename = Path(str(old_path)).name
                candidates = image_index.get(filename, [])

                if not candidates:
                    sample_failed = True
                    global_missing.append(
                        {
                            "split": split,
                            "index": item_index,
                            "id": item.get("id"),
                            "old_path": old_path,
                            "filename": filename,
                            "reason": "basename not found",
                        }
                    )
                    continue

                selected, selection_reason = choose_candidate(
                    candidates,
                    split,
                )

                if selected is None:
                    sample_failed = True
                    sample_ambiguous = True
                    global_ambiguous.append(
                        {
                            "split": split,
                            "index": item_index,
                            "id": item.get("id"),
                            "old_path": old_path,
                            "filename": filename,
                            "candidates": [str(p) for p in candidates],
                        }
                    )
                    continue

                relative_path = selected.relative_to(image_root)
                new_paths.append(str(relative_path))
                mapped_image_count += 1

            if sample_failed:
                unmatched_samples += 1

                if sample_ambiguous:
                    ambiguous_samples += 1

                if args.drop_unmatched:
                    continue

                # Preserve original sample unchanged when not dropping.
                converted_records.append(item)
                continue

            found_samples += 1

            # Keep a list because train_medgemma_deeplesion.py accesses [0].
            item["image_path"] = new_paths

            converted_records.append(item)

        converted_data[split] = converted_records

        print()
        print(f"{split}:")
        print(f"  original samples:   {len(records)}")
        print(f"  output samples:     {len(converted_records)}")
        print(f"  matched samples:    {found_samples}")
        print(f"  unmatched samples:  {unmatched_samples}")
        print(f"  ambiguous samples:  {ambiguous_samples}")
        print(f"  mapped images:      {mapped_image_count}")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    missing_log = output_json.with_suffix(".missing.json")
    ambiguous_log = output_json.with_suffix(".ambiguous.json")

    with missing_log.open("w", encoding="utf-8") as f:
        json.dump(global_missing, f, indent=2)

    with ambiguous_log.open("w", encoding="utf-8") as f:
        json.dump(global_ambiguous, f, indent=2)

    print()
    print(f"Saved converted JSON: {output_json}")
    print(f"Missing log:           {missing_log}")
    print(f"Ambiguous log:         {ambiguous_log}")
    print(f"Total missing entries: {len(global_missing)}")
    print(f"Total ambiguous:       {len(global_ambiguous)}")

    if global_missing or global_ambiguous:
        print()
        print(
            "WARNING: Some samples were not converted. Review the two log "
            "files before training."
        )


if __name__ == "__main__":
    main()
