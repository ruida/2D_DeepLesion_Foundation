#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def clean_text(text):
    return " ".join(str(text).strip().split())


def main():
    args = parse_args()

    references = {}
    predictions = {}

    with open(args.predictions_jsonl, encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue

            row = json.loads(line)

            reference = (
                row.get("reference")
                or row.get("reference_report")
                or row.get("report")
                or row.get("target")
            )
            prediction = row.get("prediction")

            if reference is None:
                raise KeyError(
                    f"Missing reference at row {index}. "
                    f"Available keys: {list(row.keys())}"
                )

            if prediction is None:
                raise KeyError(
                    f"Missing prediction at row {index}. "
                    f"Available keys: {list(row.keys())}"
                )

            key = str(row.get("id", index))
            references[key] = [clean_text(reference)]
            predictions[key] = [clean_text(prediction)]

    bleu_score, _ = Bleu(4).compute_score(references, predictions)
    meteor_score, _ = Meteor().compute_score(references, predictions)
    rouge_score, _ = Rouge().compute_score(references, predictions)

    metrics = {
        "n": len(references),
        "BLEU_1": float(bleu_score[0]),
        "BLEU_2": float(bleu_score[1]),
        "BLEU_3": float(bleu_score[2]),
        "BLEU_4": float(bleu_score[3]),
        "METEOR": float(meteor_score),
        "ROUGE_L": float(rouge_score),
    }

    print(json.dumps(metrics, indent=2))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
