#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def normalize_term(text):
    text = str(text).lower().strip()
    text = text.replace("-", "_")
    text = re.sub(r"_+", "_", text)
    return text


def readable_totalseg_name(name):
    name = normalize_term(name)

    direct = {
        "lung_upper_lobe_left": "left upper lung",
        "lung_lower_lobe_left": "left lower lung",
        "lung_upper_lobe_right": "right upper lung",
        "lung_middle_lobe_right": "right middle lung",
        "lung_lower_lobe_right": "right lower lung",

        "kidney_left": "left kidney",
        "kidney_right": "right kidney",
        "kidney_cyst_left": "left kidney cyst",
        "kidney_cyst_right": "right kidney cyst",

        "adrenal_gland_left": "left adrenal gland",
        "adrenal_gland_right": "right adrenal gland",

        "iliac_artery_left": "left iliac artery",
        "iliac_artery_right": "right iliac artery",
        "iliac_vena_left": "left iliac vein",
        "iliac_vena_right": "right iliac vein",

        "humerus_left": "left humerus",
        "humerus_right": "right humerus",
        "scapula_left": "left scapula",
        "scapula_right": "right scapula",
        "clavicula_left": "left clavicle",
        "clavicula_right": "right clavicle",
        "femur_left": "left femur",
        "femur_right": "right femur",
        "hip_left": "left hip",
        "hip_right": "right hip",

        "gluteus_maximus_left": "left gluteus maximus",
        "gluteus_maximus_right": "right gluteus maximus",
        "gluteus_medius_left": "left gluteus medius",
        "gluteus_medius_right": "right gluteus medius",
        "gluteus_minimus_left": "left gluteus minimus",
        "gluteus_minimus_right": "right gluteus minimus",

        "iliopsoas_left": "left iliopsoas",
        "iliopsoas_right": "right iliopsoas",

        "autochthon_left": "left paraspinal muscle",
        "autochthon_right": "right paraspinal muscle",

        "portal_vein_and_splenic_vein": "portal and splenic vein",
        "superior_vena_cava": "superior vena cava",
        "inferior_vena_cava": "inferior vena cava",
        "pulmonary_vein": "pulmonary vein",

        "brachiocephalic_trunk": "brachiocephalic trunk",
        "brachiocephalic_vein_left": "left brachiocephalic vein",
        "brachiocephalic_vein_right": "right brachiocephalic vein",

        "common_carotid_artery_left": "left common carotid artery",
        "common_carotid_artery_right": "right common carotid artery",

        "subclavian_artery_left": "left subclavian artery",
        "subclavian_artery_right": "right subclavian artery",

        "atrial_appendage_left": "left atrial appendage",
        "urinary_bladder": "bladder",
        "thyroid_gland": "thyroid",
        "spinal_cord": "spinal cord",
    }

    if name in direct:
        return direct[name]

    if name.startswith("vertebrae_"):
        level = name.replace("vertebrae_", "").upper()
        return f"{level} vertebra"

    if name.startswith("rib_"):
        return "rib"

    return name.replace("_", " ")


def detail_group(name):
    name = normalize_term(name)

    if "lung" in name:
        return "lung"

    if "liver" in name:
        return "liver"

    if "kidney" in name:
        return "kidney"

    if "adrenal" in name:
        return "adrenal"

    if (
        "vertebra" in name
        or "spinal_cord" in name
        or "autochthon" in name
    ):
        return "spine"

    if (
        "rib" in name
        or "humerus" in name
        or "scapula" in name
        or "clavicula" in name
        or "femur" in name
        or "sacrum" in name
        or "skull" in name
    ):
        return "bone"

    if (
        "brain" in name
        or "thyroid" in name
        or "carotid" in name
        or "mandible" in name
        or "maxilla" in name
    ):
        return "brain_head_neck"

    if (
        "hip" in name
        or "iliac_" in name
        or "bladder" in name
        or "prostate" in name
        or "gluteus" in name
        or "iliopsoas" in name
    ):
        return "pelvis"

    if (
        "spleen" in name
        or "gallbladder" in name
        or "stomach" in name
        or "pancreas" in name
        or "bowel" in name
        or "duodenum" in name
        or "colon" in name
        or "portal_vein" in name
        or "inferior_vena_cava" in name
    ):
        return "abdomen"

    if (
        "heart" in name
        or "aorta" in name
        or "pulmonary_vein" in name
        or "brachiocephalic" in name
        or "subclavian" in name
        or "superior_vena_cava" in name
        or "atrial_appendage" in name
        or "esophagus" in name
        or "trachea" in name
    ):
        return "chest"

    return "unknown"


def is_allowed_for_unknown(raw_name):
    raw_name = normalize_term(raw_name)

    blocked_terms = [
        "aorta",
        "vena_cava",
        "portal_vein",
        "splenic_vein",
        "pulmonary_vein",
        "brachiocephalic",
        "subclavian",
        "carotid",
        "iliac_artery",
        "iliac_vena",
        "atrial_appendage",
    ]

    return not any(term in raw_name for term in blocked_terms)


COMPATIBLE_GROUPS = {
    "lung": {"lung"},
    "liver": {"liver"},
    "kidney": {"kidney"},
    "adrenal": {"adrenal"},
    "abdomen": {
        "abdomen",
        "liver",
        "kidney",
        "adrenal",
    },
    "pelvis": {"pelvis"},
    "chest": {
        "chest",
        "lung",
        "bone",
    },
    "brain_head_neck": {"brain_head_neck"},
    "spine": {"spine"},
    "bone": {
        "bone",
        "spine",
    },
    "unknown": {
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
    },
}


def choose_detail_hits(example, min_pixels):
    hits = example.get("totalseg_detail_hits", [])
    aggregated = {}

    if isinstance(hits, list) and hits:
        for hit in hits:
            name = hit.get("totalseg_label_name")

            if not name:
                continue

            name = normalize_term(name)
            pixels = int(hit.get("pixel_count", 0))

            if pixels < min_pixels:
                continue

            aggregated[name] = aggregated.get(name, 0) + pixels
    else:
        for name in example.get("totalseg_detail_names", []):
            name = normalize_term(name)

            if name not in aggregated:
                aggregated[name] = 1

    candidates = [
        {
            "name": name,
            "pixel_count": pixels,
        }
        for name, pixels in aggregated.items()
    ]

    candidates.sort(
        key=lambda item: item["pixel_count"],
        reverse=True,
    )

    return candidates


def merge_example(
    example,
    coarse_field,
    max_details,
    min_pixels,
    add_lesion_token,
):
    coarse_name = example.get(coarse_field, "unknown")

    if coarse_name is None:
        coarse_name = "unknown"

    coarse_name = normalize_term(coarse_name)

    allowed_groups = COMPATIBLE_GROUPS.get(
        coarse_name,
        COMPATIBLE_GROUPS["unknown"],
    )

    candidates = choose_detail_hits(
        example,
        min_pixels=min_pixels,
    )

    exact_candidates = []
    compatible_candidates = []
    debug_records = []

    for candidate in candidates:
        raw_name = candidate["name"]
        pixel_count = candidate["pixel_count"]

        group = detail_group(raw_name)
        readable = readable_totalseg_name(raw_name)

        debug = {
            "raw_name": raw_name,
            "readable_name": readable,
            "detail_group": group,
            "pixel_count": pixel_count,
            "compatible": group in allowed_groups,
            "selected": False,
            "reason": None,
        }

        if group == "unknown":
            debug["reason"] = "unknown_group"
            debug_records.append(debug)
            continue

        if group not in allowed_groups:
            debug["reason"] = "incompatible_group"
            debug_records.append(debug)
            continue

        if (
            coarse_name == "unknown"
            and not is_allowed_for_unknown(raw_name)
        ):
            debug["reason"] = "blocked_vessel_for_unknown"
            debug_records.append(debug)
            continue

        candidate_record = {
            "raw_name": raw_name,
            "readable": readable,
            "group": group,
            "pixel_count": pixel_count,
            "debug": debug,
        }

        if group == coarse_name:
            exact_candidates.append(candidate_record)
        else:
            compatible_candidates.append(candidate_record)

    exact_candidates.sort(
        key=lambda item: item["pixel_count"],
        reverse=True,
    )

    compatible_candidates.sort(
        key=lambda item: item["pixel_count"],
        reverse=True,
    )

    ranked_candidates = exact_candidates + compatible_candidates

    selected_details = []
    seen_readable = set()

    for candidate in ranked_candidates:
        readable = candidate["readable"]
        debug = candidate["debug"]

        if readable in seen_readable:
            debug["reason"] = "duplicate"
            debug_records.append(debug)
            continue

        selected_details.append(readable)
        seen_readable.add(readable)

        debug["selected"] = True
        debug["reason"] = (
            "exact_group"
            if candidate["group"] == coarse_name
            else "compatible_fallback"
        )

        debug_records.append(debug)

        if len(selected_details) >= max_details:
            break

    merged_names = []

    if coarse_name != "unknown":
        merged_names.append(
            coarse_name.replace("_", " ")
        )

    for detail in selected_details:
        if detail not in merged_names:
            merged_names.append(detail)

    if not merged_names:
        merged_names = ["unknown"]

    text_tokens = []

    if add_lesion_token:
        text_tokens.append("lesion")

    text_tokens.extend(merged_names)

    example["merged_anatomy_names"] = merged_names
    example["merged_anatomy_text"] = " ".join(text_tokens)

    example["merged_anatomy_coarse_name"] = coarse_name
    example["merged_anatomy_detail_names"] = selected_details
    example["merged_anatomy_debug"] = debug_records

    example["merged_anatomy_source"] = (
        f"{coarse_field}_plus_totalseg_gtbbox_detail_"
        f"top{max_details}_strict_no_unknown_vessels"
    )

    return example


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
        "--summary_json",
        required=True,
    )

    parser.add_argument(
        "--coarse_field",
        default="oracle_anatomy_name",
    )

    parser.add_argument(
        "--max_details",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--min_pixels",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--add_lesion_token",
        action="store_true",
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    summary = {}

    for split in ["train", "val", "test"]:
        if split not in data:
            continue

        text_counts = Counter()
        detail_count_distribution = Counter()

        incompatible_count = 0
        unknown_group_count = 0
        blocked_vessel_count = 0
        exact_group_count = 0
        fallback_count = 0

        for example in data[split]:
            merge_example(
                example=example,
                coarse_field=args.coarse_field,
                max_details=args.max_details,
                min_pixels=args.min_pixels,
                add_lesion_token=args.add_lesion_token,
            )

            text_counts[
                example["merged_anatomy_text"]
            ] += 1

            detail_count_distribution[
                len(example["merged_anatomy_detail_names"])
            ] += 1

            for item in example["merged_anatomy_debug"]:
                reason = item.get("reason")

                if reason == "incompatible_group":
                    incompatible_count += 1
                elif reason == "unknown_group":
                    unknown_group_count += 1
                elif reason == "blocked_vessel_for_unknown":
                    blocked_vessel_count += 1
                elif reason == "exact_group":
                    exact_group_count += 1
                elif reason == "compatible_fallback":
                    fallback_count += 1

        summary[split] = {
            "samples": len(data[split]),
            "coarse_field": args.coarse_field,
            "max_details": args.max_details,
            "min_pixels": args.min_pixels,
            "detail_count_distribution": dict(
                detail_count_distribution
            ),
            "incompatible_candidates_rejected": incompatible_count,
            "unknown_group_candidates_rejected": unknown_group_count,
            "unknown_vessels_rejected": blocked_vessel_count,
            "exact_group_details_selected": exact_group_count,
            "compatible_fallback_details_selected": fallback_count,
            "most_common_merged_text": text_counts.most_common(30),
        }

        print(f"\n[{split}]")
        print("samples:", len(data[split]))
        print(
            "detail count distribution:",
            dict(detail_count_distribution),
        )
        print(
            "incompatible candidates rejected:",
            incompatible_count,
        )
        print(
            "unknown-group candidates rejected:",
            unknown_group_count,
        )
        print(
            "unknown vessels rejected:",
            blocked_vessel_count,
        )
        print(
            "exact-group details selected:",
            exact_group_count,
        )
        print(
            "compatible fallback details selected:",
            fallback_count,
        )

        print("top merged anatomy text:")

        for text, count in text_counts.most_common(20):
            print(f"  {text:45s} {count}")

    output_path = Path(args.output_json)
    summary_path = Path(args.summary_json)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved JSON:", output_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
