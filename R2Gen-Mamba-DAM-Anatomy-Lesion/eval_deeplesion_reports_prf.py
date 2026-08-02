import argparse
import re
from pathlib import Path

import pandas as pd


STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with",
    "without", "is", "are", "was", "were", "be", "been", "there",
    "this", "that", "these", "those",
}


def normalize_text(text):
    text = str(text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s,.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_report(text, remove_stopwords=True):
    text = normalize_text(text)

    # Split on comma and whitespace.
    tokens = re.split(r"[\s,;]+", text)
    tokens = [t.strip() for t in tokens if t.strip()]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    return set(tokens)


def compute_prf(pred_text, gt_text):
    pred_set = tokenize_report(pred_text)
    gt_set = tokenize_report(gt_text)

    if len(pred_set) == 0 and len(gt_set) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0

    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(gt_set)

    if len(gt_set) == 0:
        return 0.0, 0.0, 0.0, 0, len(pred_set), 0

    inter = pred_set & gt_set

    precision = len(inter) / len(pred_set)
    recall = len(inter) / len(gt_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, len(inter), len(pred_set), len(gt_set)


def guess_text_col(df):
    candidates = [
        "report",
        "prediction",
        "pred",
        "res",
        "caption",
        "text",
        "reference",
        "gt",
        "ground_truth",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    # fallback: last column
    return df.columns[-1]


def guess_id_col(df):
    candidates = ["id", "image_id", "image_name", "filename", "file_name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_csv_auto(path):
    path = Path(path)

    # Try normal header CSV first
    try:
        df = pd.read_csv(path)
        if len(df.columns) >= 1:
            return df
    except Exception:
        pass

    # Fallback no-header CSV
    df = pd.read_csv(path, header=None)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--gt_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_metrics", required=True)
    args = parser.parse_args()

    pred_df = read_csv_auto(args.pred_csv)
    gt_df = read_csv_auto(args.gt_csv)

    pred_text_col = guess_text_col(pred_df)
    gt_text_col = guess_text_col(gt_df)

    pred_id_col = guess_id_col(pred_df)
    gt_id_col = guess_id_col(gt_df)

    print("Prediction columns:", list(pred_df.columns))
    print("GT columns:", list(gt_df.columns))
    print("Pred text col:", pred_text_col)
    print("GT text col:", gt_text_col)
    print("Pred ID col:", pred_id_col)
    print("GT ID col:", gt_id_col)

    if pred_id_col is not None and gt_id_col is not None:
        p = pred_df[[pred_id_col, pred_text_col]].copy()
        g = gt_df[[gt_id_col, gt_text_col]].copy()

        p.columns = ["id", "prediction"]
        g.columns = ["id", "ground_truth"]

        df = g.merge(p, on="id", how="inner")
    else:
        n = min(len(pred_df), len(gt_df))
        df = pd.DataFrame({
            "id": list(range(n)),
            "ground_truth": gt_df[gt_text_col].astype(str).iloc[:n].tolist(),
            "prediction": pred_df[pred_text_col].astype(str).iloc[:n].tolist(),
        })

    rows = []

    total_inter = 0
    total_pred = 0
    total_gt = 0

    for _, row in df.iterrows():
        precision, recall, f1, inter_n, pred_n, gt_n = compute_prf(
            row["prediction"],
            row["ground_truth"],
        )

        total_inter += inter_n
        total_pred += pred_n
        total_gt += gt_n

        rows.append({
            "id": row["id"],
            "ground_truth": row["ground_truth"],
            "prediction": row["prediction"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_keywords": inter_n,
            "pred_keywords": pred_n,
            "gt_keywords": gt_n,
        })

    out_df = pd.DataFrame(rows)

    macro_precision = out_df["precision"].mean()
    macro_recall = out_df["recall"].mean()
    macro_f1 = out_df["f1"].mean()

    micro_precision = total_inter / total_pred if total_pred > 0 else 0.0
    micro_recall = total_inter / total_gt if total_gt > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0 else 0.0
    )

    exact_set_match = (
        (out_df["precision"] == 1.0)
        & (out_df["recall"] == 1.0)
    ).mean()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    metrics = {
        "n": len(out_df),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "exact_set_match": exact_set_match,
        "total_matched_keywords": total_inter,
        "total_pred_keywords": total_pred,
        "total_gt_keywords": total_gt,
    }

    out_metrics = Path(args.out_metrics)
    with open(out_metrics, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print()
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k:24s}: {v}")

    print()
    print("Saved per-sample CSV:", out_csv)
    print("Saved metrics:", out_metrics)


if __name__ == "__main__":
    main()
