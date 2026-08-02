#!/usr/bin/env python3

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path


# ============================================================
# Anatomy-location label space
# ============================================================

ANATOMY_NAME_TO_ID = {
    "unknown": 0,
    "lung": 1,
    "liver": 2,
    "kidney": 3,
    "adrenal": 4,
    "abdomen": 5,
    "pelvis": 6,
    "chest": 7,
    "brain_head_neck": 8,
    "spine": 9,
    "bone": 10,
}

ANATOMY_KEYWORDS = {
    "lung": [
        "lung",
        "lungs",
        "pulmonary",
        "right lung",
        "left lung",
        "upper lung",
        "lower lung",
        "upper lobe",
        "middle lobe",
        "lower lobe",
        "right upper lobe",
        "right middle lobe",
        "right lower lobe",
        "left upper lobe",
        "left lower lobe",
        "hilar",
        "hilum",
        "pleura",
        "pleural",
    ],

    "liver": [
        "liver",
        "hepatic",
        "right hepatic lobe",
        "left hepatic lobe",
    ],

    "kidney": [
        "kidney",
        "kidneys",
        "renal",
        "right kidney",
        "left kidney",
        "renal cortex",
        "renal pole",
    ],

    "adrenal": [
        "adrenal",
        "adrenal gland",
        "right adrenal",
        "left adrenal",
        "right adrenal gland",
        "left adrenal gland",
    ],

    "spine": [
        "spine",
        "spinal",
        "vertebra",
        "vertebral",
        "vertebral body",
        "cervical spine",
        "thoracic spine",
        "lumbar spine",
        "sacral spine",
    ],

    "bone": [
        "bone",
        "osseous",
        "rib",
        "ribs",
        "sternum",
        "clavicle",
        "scapula",
        "humerus",
        "femur",
        "ilium",
        "iliac bone",
        "sacrum",
        "acetabulum",
        "skull",
    ],

    "brain_head_neck": [
        "brain",
        "intracranial",
        "head",
        "neck",
        "head and neck",
        "thyroid",
        "parotid",
        "submandibular",
        "mandible",
        "maxilla",
        "nasopharynx",
        "oropharynx",
        "larynx",
        "pharynx",
    ],

    "pelvis": [
        "pelvis",
        "pelvic",
        "prostate",
        "uterus",
        "uterine",
        "ovary",
        "ovarian",
        "adnexa",
        "adnexal",
        "bladder",
        "rectum",
        "cervix",
        "inguinal",
        "iliac chain",
    ],

    "abdomen": [
        "abdomen",
        "abdominal",
        "retroperitoneum",
        "retroperitoneal",
        "pancreas",
        "pancreatic",
        "spleen",
        "splenic",
        "gallbladder",
        "bowel",
        "small bowel",
        "colon",
        "colonic",
        "stomach",
        "gastric",
        "mesentery",
        "mesenteric",
        "peritoneum",
        "peritoneal",
        "omentum",
        "omental",
        "porta hepatis",
        "peripancreatic",
        "para-aortic",
        "periaortic",
    ],

    "chest": [
        "chest",
        "thorax",
        "thoracic",
        "mediastinum",
        "mediastinal",
        "subcarinal",
        "paratracheal",
        "axillary",
        "breast",
        "chest wall",
    ],
}


# More specific organs should be selected before broad regions.
ANATOMY_PRIORITY = [
    "lung",
    "liver",
    "kidney",
    "adrenal",
    "spine",
    "bone",
    "brain_head_neck",
    "pelvis",
    "abdomen",
    "chest",
]


# ============================================================
# Lesion-type label space
# ============================================================

LESION_TYPE_NAME_TO_ID = {
    "other_lesion": 0,
    "lymph_node": 1,
    "soft_tissue": 2,
    "nodule": 3,
    "mass": 4,
    "opacity": 5,
    "cystic": 6,
    "enhancing": 7,
    "calcified": 8,
    "metastatic": 9,
}

LESION_TYPE_KEYWORDS = {
    "lymph_node": [
        "lymph node",
        "lymph nodes",
        "lymphadenopathy",
        "nodal",
        "adenopathy",
    ],

    "soft_tissue": [
        "soft tissue",
        "soft-tissue",
        "subcutaneous",
        "intramuscular",
        "muscular",
        "muscle lesion",
    ],

    "nodule": [
        "nodule",
        "nodules",
        "nodular",
    ],

    "mass": [
        "mass",
        "masses",
        "mass-like",
        "masslike",
    ],

    "opacity": [
        "opacity",
        "opacities",
        "ground glass",
        "ground-glass",
        "consolidation",
    ],

    "cystic": [
        "cyst",
        "cysts",
        "cystic",
        "hypoattenuating",
        "hypoattenuation",
        "hypodense",
        "low density",
        "low-density",
        "fluid collection",
    ],

    "enhancing": [
        "enhancing",
        "enhancement",
        "hyperenhancing",
        "hyperattenuating",
        "hyperattenuation",
        "hyperdense",
    ],

    "calcified": [
        "calcified",
        "calcification",
        "calcifications",
    ],

    "metastatic": [
        "metastatic",
        "metastasis",
        "metastases",
        "metastatic disease",
    ],
}


# Used only when several lesion-type words occur in one report.
LESION_TYPE_PRIORITY = [
    "metastatic",
    "lymph_node",
    "mass",
    "nodule",
    "cystic",
    "enhancing",
    "calcified",
    "opacity",
    "soft_tissue",
]


def normalize_text(text):
    text = str(text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s,.;:/()+-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_keyword(text, keyword):
    pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def get_report(example):
    for key in [
        "report",
        "findings",
        "impression",
        "text",
        "caption",
    ]:
        value = example.get(key)

        if value:
            return str(value)

    return ""


def find_matches(text, keyword_map, priority):
    matched_classes = []
    matched_keywords = {}

    for class_name in priority:
        found = []

        for keyword in keyword_map[class_name]:
            if contains_keyword(text, keyword):
                found.append(keyword)

        if found:
            matched_classes.append(class_name)
            matched_keywords[class_name] = found

    return matched_classes, matched_keywords


def infer_anatomy(report):
    text = normalize_text(report)

    matches, matched_keywords = find_matches(
        text,
        ANATOMY_KEYWORDS,
        ANATOMY_PRIORITY,
    )

    if matches:
        selected = matches[0]
    else:
        selected = "unknown"

    return selected, matches, matched_keywords


def infer_lesion_type(report):
    text = normalize_text(report)

    matches, matched_keywords = find_matches(
        text,
        LESION_TYPE_KEYWORDS,
        LESION_TYPE_PRIORITY,
    )

    if matches:
        selected = matches[0]
    else:
        selected = "other_lesion"

    return selected, matches, matched_keywords


def make_anatomy_json(source_data):
    output = copy.deepcopy(source_data)

    for split in ["train", "val", "test"]:
        if split not in output:
            continue

        distribution = Counter()

        for example in output[split]:
            report = get_report(example)

            selected, all_matches, matched_keywords = infer_anatomy(
                report
            )

            example["oracle_anatomy_id"] = ANATOMY_NAME_TO_ID[selected]
            example["oracle_anatomy_name"] = selected

            # Compatibility aliases for anatomy-guided R2Gen-Mamba.
            example["rough_anatomy_id"] = ANATOMY_NAME_TO_ID[selected]
            example["rough_anatomy_name"] = selected
            example["rough_anatomy_ids"] = [
                ANATOMY_NAME_TO_ID[selected]
            ]
            example["rough_anatomy_names"] = [selected]

            example["oracle_anatomy_all_matches"] = all_matches
            example["oracle_anatomy_matched_keywords"] = matched_keywords
            example["anatomy_source"] = "oracle_report_anatomy_only"

            distribution[selected] += 1

        print(f"\n[{split}] anatomy-location distribution")

        for class_name, count in distribution.most_common():
            print(f"{class_name:20s} {count}")

    return output


def make_lesion_type_json(source_data):
    output = copy.deepcopy(source_data)

    for split in ["train", "val", "test"]:
        if split not in output:
            continue

        distribution = Counter()

        for example in output[split]:
            report = get_report(example)

            selected, all_matches, matched_keywords = infer_lesion_type(
                report
            )

            example["oracle_lesion_type_id"] = (
                LESION_TYPE_NAME_TO_ID[selected]
            )
            example["oracle_lesion_type_name"] = selected

            # Shorter aliases.
            example["lesion_type_id"] = (
                LESION_TYPE_NAME_TO_ID[selected]
            )
            example["lesion_type_name"] = selected

            # Preserve every lesion-type match for auditing.
            example["oracle_lesion_type_all_matches"] = all_matches
            example["oracle_lesion_type_matched_keywords"] = (
                matched_keywords
            )
            example["lesion_type_source"] = (
                "oracle_report_lesion_type"
            )

            distribution[selected] += 1

        print(f"\n[{split}] lesion-type distribution")

        for class_name, count in distribution.most_common():
            print(f"{class_name:20s} {count}")

    return output


def validate_anatomy(data):
    valid_names = set(ANATOMY_NAME_TO_ID)
    errors = []

    for split in ["train", "val", "test"]:
        for example in data.get(split, []):
            name = example.get("oracle_anatomy_name")
            anatomy_id = example.get("oracle_anatomy_id")

            if name not in valid_names:
                errors.append(
                    (split, example.get("id"), "invalid name", name)
                )
                continue

            expected_id = ANATOMY_NAME_TO_ID[name]

            if anatomy_id != expected_id:
                errors.append(
                    (
                        split,
                        example.get("id"),
                        "ID mismatch",
                        anatomy_id,
                        expected_id,
                    )
                )

    return errors


def validate_lesion_type(data):
    valid_names = set(LESION_TYPE_NAME_TO_ID)
    errors = []

    for split in ["train", "val", "test"]:
        for example in data.get(split, []):
            name = example.get("oracle_lesion_type_name")
            lesion_id = example.get("oracle_lesion_type_id")

            if name not in valid_names:
                errors.append(
                    (split, example.get("id"), "invalid name", name)
                )
                continue

            expected_id = LESION_TYPE_NAME_TO_ID[name]

            if lesion_id != expected_id:
                errors.append(
                    (
                        split,
                        example.get("id"),
                        "ID mismatch",
                        lesion_id,
                        expected_id,
                    )
                )

    return errors


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
    )

    parser.add_argument(
        "--anatomy_output_json",
        required=True,
    )

    parser.add_argument(
        "--lesion_output_json",
        required=True,
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as file:
        source_data = json.load(file)

    anatomy_data = make_anatomy_json(source_data)
    lesion_data = make_lesion_type_json(source_data)

    anatomy_errors = validate_anatomy(anatomy_data)
    lesion_errors = validate_lesion_type(lesion_data)

    if anatomy_errors:
        print("\nAnatomy validation errors:")
        for error in anatomy_errors[:20]:
            print(error)
        raise RuntimeError(
            f"Found {len(anatomy_errors)} anatomy validation errors."
        )

    if lesion_errors:
        print("\nLesion-type validation errors:")
        for error in lesion_errors[:20]:
            print(error)
        raise RuntimeError(
            f"Found {len(lesion_errors)} lesion validation errors."
        )

    anatomy_output = Path(args.anatomy_output_json)
    lesion_output = Path(args.lesion_output_json)

    anatomy_output.parent.mkdir(parents=True, exist_ok=True)
    lesion_output.parent.mkdir(parents=True, exist_ok=True)

    with open(anatomy_output, "w") as file:
        json.dump(anatomy_data, file, indent=2)

    with open(lesion_output, "w") as file:
        json.dump(lesion_data, file, indent=2)

    print("\nValidation successful.")
    print("Anatomy JSON:", anatomy_output)
    print("Lesion-type JSON:", lesion_output)


if __name__ == "__main__":
    main()
