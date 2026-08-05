#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate BLEU-1 to BLEU-4, METEOR, and ROUGE-L."
    )
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional output metrics JSON path.",
    )
    return parser.parse_args()


def normalize_text(text):
    text = str(text).lower().strip()

    # Treat comma-separated tags as a normal token sequence.
    text = re.sub(r"\s*,\s*", " ", text)
    text = re.sub(r"[.;:]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    args = parse_args()

    references = {}
    predictions = {}

    with open(args.predictions_jsonl, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            prediction = normalize_text(row.get("prediction", ""))

            reference = row.get("reference")
            if reference is None:
                reference = row.get("reference_report", "")
            reference = normalize_text(reference)

            if not prediction or not reference:
                print(
                    f"Warning: skipping line {line_number} because "
                    "prediction or reference is empty."
                )
                continue

            key = str(len(references))
            references[key] = [reference]
            predictions[key] = [prediction]

    if not references:
        raise RuntimeError("No valid prediction/reference pairs were found.")

    metrics = {
        "n": len(references),
    }

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(
        references,
        predictions,
    )

    for i, score in enumerate(bleu_scores, start=1):
        metrics[f"BLEU_{i}"] = float(score)

    meteor_scorer = Meteor()
    try:
        meteor_score, _ = meteor_scorer.compute_score(
            references,
            predictions,
        )
        metrics["METEOR"] = float(meteor_score)
    finally:
        # Close the Java METEOR subprocess when supported.
        if hasattr(meteor_scorer, "close"):
            meteor_scorer.close()

    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(
        references,
        predictions,
    )
    metrics["ROUGE_L"] = float(rouge_score)

    print(json.dumps(metrics, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        input_path = Path(args.predictions_jsonl)
        output_path = input_path.with_name(
            input_path.stem + "_nlg_metrics.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
