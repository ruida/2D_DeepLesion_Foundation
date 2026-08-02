import argparse
import json
import re
from pathlib import Path

import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def clean_text(x):
    x = str(x).lower().strip()
    x = x.replace(",", " ")
    x = x.replace(";", " ")
    x = x.replace(":", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def parse_structured(text):
    out = {
        "rough_anatomy": "",
        "fine_anatomy": "",
        "lesion_type": "",
    }

    text = str(text).strip().lower()

    for part in text.split(";"):
        if ":" not in part:
            continue

        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()

        if k in out:
            out[k] = v

    return out


def structured_to_template_report(text):
    fields = parse_structured(text)

    rough = fields["rough_anatomy"]
    fine = fields["fine_anatomy"]
    lesion = fields["lesion_type"]

    terms = []

    if rough and rough != "unknown":
        terms.append(rough)

    if lesion and lesion != "unknown":
        terms.append(lesion)

    if fine and fine != "unknown" and fine not in terms:
        terms.append(fine)

    return " ".join(terms)


def load_single_col_csv(path, col_name):
    df = pd.read_csv(path, header=None, names=[col_name])
    return df[col_name].astype(str).tolist()


def compute_metrics(gts_list, pred_list):
    gts = {}
    res = {}

    for i, (gt, pred) in enumerate(zip(gts_list, pred_list)):
        gts[i] = [clean_text(gt)]
        res[i] = [clean_text(pred)]

    scores = {}

    bleu_score, _ = Bleu(4).compute_score(gts, res)
    scores["BLEU_1"] = bleu_score[0]
    scores["BLEU_2"] = bleu_score[1]
    scores["BLEU_3"] = bleu_score[2]
    scores["BLEU_4"] = bleu_score[3]

    meteor_score, _ = Meteor().compute_score(gts, res)
    scores["METEOR"] = meteor_score

    rouge_score, _ = Rouge().compute_score(gts, res)
    scores["ROUGE_L"] = rouge_score

    return scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--structured_res_csv",
        default="results/deeplesion_structured_target_test/res.csv",
    )

    parser.add_argument(
        "--original_report_json",
        default="data/generated_annotations/original_report_anat_lesion_token/deeplesion_original_report_anat_lesion_token_all.json",
    )

    parser.add_argument(
        "--out_csv",
        default="results/deeplesion_structured_target_test/structured_template_vs_short_report.csv",
    )

    args = parser.parse_args()

    structured_preds = load_single_col_csv(args.structured_res_csv, "structured_pred")

    data = json.load(open(args.original_report_json))
    gt_reports = [
        x.get("original_report", x.get("report", ""))
        for x in data["test"]
    ]

    if len(structured_preds) != len(gt_reports):
        raise ValueError(f"Length mismatch: pred={len(structured_preds)}, gt={len(gt_reports)}")

    converted_preds = [structured_to_template_report(x) for x in structured_preds]

    metrics = compute_metrics(gt_reports, converted_preds)

    print("Mode: template")
    print("Num samples:", len(gt_reports))
    print("\nMetrics against original short report:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    rows = []
    for i, (gt, sp, cp) in enumerate(zip(gt_reports, structured_preds, converted_preds)):
        rows.append({
            "idx": i,
            "gt_short_report": gt,
            "structured_prediction": sp,
            "converted_prediction": cp,
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
