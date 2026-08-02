#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


CLASSES = [
    "lymph_node",
    "soft_tissue",
    "nodule",
    "mass",
    "opacity",
    "cystic",
    "enhancing",
    "calcified",
    "metastatic",
    "other_lesion",
]


def get_oracle_label(example):
    for key in [
        "lesion_type_name",
        "oracle_lesion_type_name",
        "lesion_type_text",
    ]:
        value = example.get(key)
        if value:
            value = str(value).strip().lower().replace(" ", "_")
            return value

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--oracle_json",
        required=True,
    )
    parser.add_argument(
        "--pred_json",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        required=True,
    )

    args = parser.parse_args()

    oracle_data = json.load(open(args.oracle_json))
    pred_data = json.load(open(args.pred_json))

    oracle_by_id = {
        example["id"]: example
        for example in oracle_data["test"]
    }

    pred_by_id = {
        example["id"]: example
        for example in pred_data["test"]
    }

    oracle_ids = set(oracle_by_id)
    pred_ids = set(pred_by_id)
    common_ids = sorted(oracle_ids & pred_ids)

    print("oracle test:", len(oracle_ids))
    print("predicted test:", len(pred_ids))
    print("matched IDs:", len(common_ids))
    print("missing predictions:", len(oracle_ids - pred_ids))
    print("extra predictions:", len(pred_ids - oracle_ids))

    rows = []
    missing_oracle = []

    for sample_id in common_ids:
        oracle_example = oracle_by_id[sample_id]
        pred_example = pred_by_id[sample_id]

        oracle_label = get_oracle_label(oracle_example)
        predicted_label = pred_example.get(
            "predicted_lesion_type_name"
        )

        if oracle_label is None:
            missing_oracle.append(sample_id)
            continue

        predicted_label = str(
            predicted_label
        ).strip().lower().replace(" ", "_")

        confidence = float(
            pred_example.get(
                "predicted_lesion_type_confidence",
                0.0,
            )
        )

        rows.append(
            {
                "id": sample_id,
                "oracle": oracle_label,
                "predicted": predicted_label,
                "confidence": confidence,
                "correct": int(
                    oracle_label == predicted_label
                ),
                "anatomy": pred_example.get(
                    "predicted_merged_anatomy_text",
                    pred_example.get("anatomy_text"),
                ),
            }
        )

    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        out_dir / "lesion_predictions_vs_oracle.csv",
        index=False,
    )

    y_true = df["oracle"].tolist()
    y_pred = df["predicted"].tolist()

    labels = [
        label
        for label in CLASSES
        if label in set(y_true) | set(y_pred)
    ]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        out_dir / "classification_report.csv"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"oracle_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )

    matrix_df.to_csv(
        out_dir / "confusion_matrix.csv"
    )

    oracle_distribution = Counter(y_true)
    predicted_distribution = Counter(y_pred)

    with open(
        out_dir / "summary.txt",
        "w",
    ) as f:
        f.write(f"matched samples: {len(df)}\n")
        f.write(
            f"missing oracle labels: "
            f"{len(missing_oracle)}\n"
        )
        f.write(f"accuracy: {accuracy:.6f}\n")
        f.write(
            f"balanced_accuracy: "
            f"{balanced_accuracy:.6f}\n"
        )
        f.write(f"macro_f1: {macro_f1:.6f}\n")
        f.write(
            f"weighted_f1: {weighted_f1:.6f}\n\n"
        )

        f.write("Oracle distribution:\n")
        for label, count in oracle_distribution.most_common():
            f.write(f"{label}: {count}\n")

        f.write("\nPredicted distribution:\n")
        for label, count in predicted_distribution.most_common():
            f.write(f"{label}: {count}\n")

        f.write("\nClassification report:\n")
        f.write(report_text)

        f.write("\nConfusion matrix:\n")
        f.write(matrix_df.to_string())

    print("\nOverall metrics")
    print(f"accuracy          : {accuracy:.4f}")
    print(
        f"balanced accuracy : "
        f"{balanced_accuracy:.4f}"
    )
    print(f"macro F1          : {macro_f1:.4f}")
    print(f"weighted F1       : {weighted_f1:.4f}")

    print("\nOracle distribution:")
    print(oracle_distribution.most_common())

    print("\nPredicted distribution:")
    print(predicted_distribution.most_common())

    print("\nClassification report:")
    print(report_text)

    print("\nConfusion matrix:")
    print(matrix_df.to_string())

    print("\nConfidence analysis:")
    confidence_bins = [
        (0.0, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.01),
    ]

    for low, high in confidence_bins:
        subset = df[
            (df["confidence"] >= low)
            & (df["confidence"] < high)
        ]

        if len(subset) == 0:
            continue

        print(
            f"{low:.1f}-{min(high, 1.0):.1f}: "
            f"n={len(subset)}, "
            f"accuracy={subset['correct'].mean():.4f}"
        )

    print("\nMost frequent errors:")
    errors = df[df["correct"] == 0]

    error_pairs = Counter(
        zip(errors["oracle"], errors["predicted"])
    )

    for (oracle, predicted), count in error_pairs.most_common(20):
        print(
            f"{oracle:15s} -> "
            f"{predicted:15s}: {count}"
        )

    print("\nSaved:", out_dir)


if __name__ == "__main__":
    main()
