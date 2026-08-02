#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

PROJECT = Path("/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text")
ORACLE_JSON = PROJECT / "data/simple_pair_rag_annotations/00_oracle_simple_labels.json"
SOURCE_JSON = PROJECT / "data/oracle_pair_rag_annotations/02_totalseg_plus_predicted_lesion_complete.json"
OUTPUT_JSON = PROJECT / "data/simple_dualhead_classifier/dualhead_crop_dataset.json"
SUMMARY_JSON = PROJECT / "data/simple_dualhead_classifier/dualhead_crop_dataset_summary.json"

PRIMARY_CLASSES = ["lymph_node","mass","nodule","opacity","consolidation","cystic","other_lesion"]
ATTRIBUTE_CLASSES = ["low_density","enhancing","hyperdense","calcified","none"]

def paths(x):
    v = x.get("image_path", x.get("image_paths", x.get("images", [])))
    return [v] if isinstance(v, str) else (v if isinstance(v, list) else [])

def boxes(x):
    v = x.get("bboxes", x.get("bbox", []))
    if isinstance(v, list) and len(v) == 4 and all(isinstance(z, (int,float)) for z in v):
        return [v]
    return v if isinstance(v, list) else []

oracle = json.load(open(ORACLE_JSON))
source = json.load(open(SOURCE_JSON))
out, summary = {}, {}

for split in ["train","val","test"]:
    oi = {str(x["id"]): x for x in oracle[split]}
    rows = []
    cp, ca = Counter(), Counter()
    miss_o = miss_i = miss_b = 0

    for x in source[split]:
        sid = str(x["id"])
        o = oi.get(sid)
        if o is None:
            miss_o += 1
            continue
        im = paths(x)
        bb = boxes(x)
        if not im:
            miss_i += 1
            continue
        if not bb:
            miss_b += 1
            continue

        p = o["oracle_primary_lesion_type"]
        a = o["oracle_lesion_attribute"]
        if p not in PRIMARY_CLASSES or a not in ATTRIBUTE_CLASSES:
            raise ValueError((sid, p, a))

        rows.append({
            "id": sid,
            "image_path": im,
            "bboxes": bb,
            "bbox_format": x.get("bbox_format", "xyxy"),
            "primary_label": p,
            "attribute_label": a,
            "report": o.get("report", ""),
        })
        cp[p] += 1
        ca[a] += 1

    out[split] = rows
    summary[split] = {
        "n": len(rows),
        "missing_oracle": miss_o,
        "missing_image": miss_i,
        "missing_bbox": miss_b,
        "primary_distribution": cp.most_common(),
        "attribute_distribution": ca.most_common(),
    }

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUTPUT_JSON, "w"), indent=2, ensure_ascii=False)
json.dump(summary, open(SUMMARY_JSON, "w"), indent=2, ensure_ascii=False)
print("Saved:", OUTPUT_JSON)
print(json.dumps(summary, indent=2))
