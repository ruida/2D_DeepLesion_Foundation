#!/usr/bin/env python3
"""Merge DAM image/mask/bbox rows with anatomy + lesion-type metadata by ID.

The DAM JSON supplies image_path, mask_path, and pixel-space bbox used for the
full/focal visual prompt. The metadata JSON supplies anatomy/lesion fields and,
by default, the report target. Unmatched rows are omitted and summarized.
"""
import argparse
import copy
import json
from pathlib import Path

META_FIELDS = [
    "anatomy_ids", "anatomy_text", "rough_anatomy_id", "rough_anatomy_ids",
    "rough_anatomy_name", "rough_anatomy_names", "lesion_type_id",
    "lesion_type_ids", "lesion_type", "lesion_type_merged",
    "structured_target", "template_report", "main_bbox_xywh_norm",
    "retrieved_report", "conditioning_text",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dam_json", required=True)
    p.add_argument("--metadata_json", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--summary_json", required=True)
    p.add_argument("--report_source", choices=["metadata", "dam"], default="metadata")
    return p.parse_args()


def load(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    a = parse_args()
    dam, meta = load(a.dam_json), load(a.metadata_json)
    out, summary = {}, {}

    for split in ("train", "val", "test"):
        dam_rows = dam.get(split, [])
        meta_rows = meta.get(split, [])
        meta_by_id = {str(x["id"]): x for x in meta_rows}
        merged, missing = [], []

        for d in dam_rows:
            sid = str(d["id"])
            m = meta_by_id.get(sid)
            if m is None:
                missing.append(sid)
                continue
            row = copy.deepcopy(d)
            for key in META_FIELDS:
                if key in m:
                    row[key] = copy.deepcopy(m[key])
            if a.report_source == "metadata" and m.get("report"):
                row["report"] = m["report"]
            row["bbox_format"] = row.get("bbox_format", "xyxy")
            row["visual_conditioning"] = "dam_full_mask_focal_mask"
            row["semantic_conditioning"] = "anatomy_plus_lesion_type"
            merged.append(row)

        out[split] = merged
        summary[split] = {
            "dam_input": len(dam_rows),
            "metadata_input": len(meta_rows),
            "merged": len(merged),
            "missing_metadata": len(missing),
            "first_missing_metadata_ids": missing[:50],
        }

    out_path = Path(a.out_json); out_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path = Path(a.summary_json); sum_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    sum_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Saved:", out_path)
    print("Summary:", sum_path)


if __name__ == "__main__":
    main()
