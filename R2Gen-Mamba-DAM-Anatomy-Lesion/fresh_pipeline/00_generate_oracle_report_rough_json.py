#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROUGH_NAME_TO_ID = {
    "unknown": 0,
    "lung": 1,
    "liver": 2,
    "kidney": 3,
    "adrenal": 4,
    "lymph_node": 5,
    "bone": 6,
    "soft_tissue": 7,
    "abdomen": 8,
    "pelvis": 9,
    "chest": 10,
    "brain_head_neck": 11,
    "spine": 12,
    "lesion": 13,
}

LESION_TOKEN_ID = 13
LESION_TOKEN_NAME = "lesion"

KEYWORDS = {
    "lung": [
        "lung",
        "pulmonary",
        "lobe",
        "upper lobe",
        "lower lobe",
        "middle lobe",
        "hilum",
        "hilar",
        "pleura",
        "pleural",
        "right upper lobe",
        "right lower lobe",
        "left upper lobe",
        "left lower lobe",
    ],
    "liver": [
        "liver",
        "hepatic",
        "right hepatic",
        "left hepatic",
        "hepatic lobe",
    ],
    "kidney": [
        "kidney",
        "renal",
        "left kidney",
        "right kidney",
        "upper pole",
        "lower pole",
    ],
    "adrenal": [
        "adrenal",
        "adrenal gland",
        "left adrenal",
        "right adrenal",
    ],
    "lymph_node": [
        "lymph node",
        "lymph nodes",
        "lymphadenopathy",
        "nodal",
        "node",
        "nodes",
        "mediastinal",
        "axillary",
        "inguinal",
        "retroperitoneal",
        "paratracheal",
        "subcarinal",
        "porta hepatis",
        "mesenteric",
        "para-aortic",
        "periaortic",
        "peripancreatic",
        "iliac chain",
    ],
    "bone": [
        "bone",
        "osseous",
        "rib",
        "ribs",
        "femur",
        "sacrum",
        "iliac",
        "ilium",
        "humerus",
        "clavicle",
        "sternum",
        "skull",
        "acetabulum",
    ],
    "spine": [
        "spine",
        "spinal",
        "cervical spine",
        "thoracic spine",
        "lumbar spine",
        "vertebral",
        "vertebra",
        "vertebral body",
    ],
    "soft_tissue": [
        "soft tissue",
        "muscle",
        "subcutaneous",
        "skin",
        "abdominal wall",
        "chest wall",
        "gluteal",
        "psoas",
    ],
    "abdomen": [
        "abdomen",
        "abdominal",
        "pancreas",
        "spleen",
        "gallbladder",
        "bowel",
        "colon",
        "stomach",
        "mesentery",
        "peritoneum",
        "retroperitoneum",
        "omental",
        "omentum",
    ],
    "pelvis": [
        "pelvis",
        "pelvic",
        "prostate",
        "uterus",
        "ovary",
        "ovarian",
        "bladder",
        "rectum",
        "cervix",
        "adnexa",
        "adnexal",
    ],
    "chest": [
        "chest",
        "thorax",
        "thoracic",
        "mediastinum",
        "breast",
    ],
    "brain_head_neck": [
        "brain",
        "head",
        "neck",
        "cervical",
        "thyroid",
        "parotid",
        "mandible",
        "maxillary",
        "nasopharynx",
        "oropharynx",
    ],
}

# Priority matters when a report contains multiple anatomy words.
# Example: "mediastinal lymph node in chest" should be classified as
# lymph_node rather than chest.
PRIORITY = [
    "lymph_node",
    "soft_tissue",
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


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s,.;:/()+-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_report_field(sample):
    for key in ["report", "findings", "impression", "text", "caption"]:
        if key in sample and sample[key]:
            return str(sample[key])
    return ""


def find_rough_anatomy(report):
    normalized_report = normalize_text(report)

    matched_classes = []
    matched_keywords = []

    for anatomy_class in PRIORITY:
        for keyword in KEYWORDS[anatomy_class]:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

            if re.search(pattern, normalized_report):
                matched_classes.append(anatomy_class)
                matched_keywords.append((anatomy_class, keyword))
                break

    if not matched_classes:
        return "unknown", [], []

    chosen_class = matched_classes[0]

    return chosen_class, matched_classes, matched_keywords


def process_split(samples, split, add_lesion_token=True):
    stats = Counter()

    for example in samples:
        report = find_report_field(example)

        anatomy_name, all_matches, matched_keywords = find_rough_anatomy(
            report
        )

        anatomy_id = ROUGH_NAME_TO_ID.get(anatomy_name, 0)

        # Single-value fields.
        example["rough_anatomy_id"] = anatomy_id
        example["rough_anatomy_name"] = anatomy_name

        # List fields used by R2Gen-Mamba.
        if add_lesion_token:
            example["rough_anatomy_ids"] = [
                LESION_TOKEN_ID,
                anatomy_id,
            ]
            example["rough_anatomy_names"] = [
                LESION_TOKEN_NAME,
                anatomy_name,
            ]
        else:
            example["rough_anatomy_ids"] = [anatomy_id]
            example["rough_anatomy_names"] = [anatomy_name]

        # Audit/debug information.
        example["rough_anatomy_all_matches"] = all_matches
        example["report_rough_matched_keywords"] = matched_keywords
        example["anatomy_source"] = (
            "oracle_report_keyword_keep_lymph_soft"
        )

        stats[anatomy_name] += 1

    print(f"\n[{split}] distribution")

    for anatomy_name, count in stats.most_common():
        print(f"{anatomy_name:20s} {count}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
        help="Input R2Gen-Mamba annotation JSON.",
    )

    parser.add_argument(
        "--output_json",
        required=True,
        help="Output JSON with oracle rough-anatomy fields.",
    )

    parser.add_argument(
        "--add_lesion_token",
        action="store_true",
        help="Prepend lesion ID 13 to rough_anatomy_ids.",
    )

    args = parser.parse_args()

    with open(args.input_json, "r") as file:
        data = json.load(file)

    for split in ["train", "val", "test"]:
        if split in data:
            process_split(
                data[split],
                split,
                add_lesion_token=args.add_lesion_token,
            )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(data, file, indent=2)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
