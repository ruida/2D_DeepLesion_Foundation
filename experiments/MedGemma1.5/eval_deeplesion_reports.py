#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from statistics import mean


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DeepLesion short-report predictions.")
    p.add_argument("--predictions_jsonl", required=True)
    return p.parse_args()


def normalize_phrase(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ,.;:")


def split_tags(s: str):
    tags = [normalize_phrase(x) for x in s.split(",")]
    return [t for t in tags if t]


def prf1(pred_tags, ref_tags):
    pred_set = set(pred_tags)
    ref_set = set(ref_tags)
    tp = len(pred_set & ref_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(ref_set) if ref_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    jacc = len(pred_set & ref_set) / len(pred_set | ref_set) if (pred_set | ref_set) else 0.0
    exact = 1.0 if pred_set == ref_set else 0.0
    return prec, rec, f1, jacc, exact


def main():
    args = parse_args()
    rows = []
    with open(args.predictions_jsonl, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise RuntimeError("No predictions found.")

    precisions, recalls, f1s, jaccs, exacts = [], [], [], [], []
    for row in rows:
        pred_tags = split_tags(row["prediction"])
        ref_tags = split_tags(row["reference"])
        p, r, f1, j, e = prf1(pred_tags, ref_tags)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        jaccs.append(j)
        exacts.append(e)

    metrics = {
        "n": len(rows),
        "macro_precision": mean(precisions),
        "macro_recall": mean(recalls),
        "macro_f1": mean(f1s),
        "mean_jaccard": mean(jaccs),
        "exact_set_match": mean(exacts),
    }
    print(json.dumps(metrics, indent=2))

    out_path = Path(args.predictions_jsonl).with_suffix(".metrics.json")
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
