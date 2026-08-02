#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.neighbors import NearestNeighbors


DEFAULT_CLASSES = [
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


def clean_text(value):
    if value is None:
        return ""

    return " ".join(
        str(value)
        .replace("_", " ")
        .strip()
        .lower()
        .split()
    )


def normalize_lesion_type(value):
    value = clean_text(value)

    aliases = {
        "lymph node": "lymph_node",
        "lymph_node": "lymph_node",
        "soft tissue": "soft_tissue",
        "soft_tissue": "soft_tissue",
        "other lesion": "other_lesion",
        "other_lesion": "other_lesion",
        "calcified lesion": "calcified",
        "metastatic lesion": "metastatic",
        "enhancing lesion": "enhancing",
        "cystic lesion": "cystic",
        "mass lesion": "mass",
        "nodule lesion": "nodule",
    }

    return aliases.get(value, value.replace(" ", "_"))


def get_list(ex, plural_field, singular_field=None):
    raw = ex.get(plural_field, [])

    if isinstance(raw, str):
        raw = [raw]

    if not raw and singular_field:
        value = ex.get(singular_field)
        raw = [value] if value else []

    output = []
    seen = set()

    for value in raw:
        value = clean_text(value)

        if not value or value in seen:
            continue

        output.append(value)
        seen.add(value)

    return output


def get_rough_anatomy_names(ex):
    names = get_list(
        ex,
        "rough_anatomy_names",
        "rough_anatomy_name",
    )

    if not names:
        anatomy_text = clean_text(ex.get("anatomy_text"))
        if anatomy_text:
            names = anatomy_text.split()

    return names


def get_totalseg_names(ex):
    return get_list(
        ex,
        "totalseg_anatomy_names",
        "totalseg_anatomy_name",
    )


def get_totalseg_detail_names(ex):
    names = get_list(
        ex,
        "totalseg_detail_names",
    )

    if names:
        return names

    hits = ex.get("totalseg_detail_hits", [])
    recovered = []

    if isinstance(hits, list):
        for hit in hits:
            if not isinstance(hit, dict):
                continue

            value = (
                hit.get("totalseg_label_name")
                or hit.get("label_name")
                or hit.get("name")
            )

            value = clean_text(value)

            if value:
                recovered.append(value)

    return list(dict.fromkeys(recovered))


def get_lesion_type(ex):
    candidate_fields = [
        "oracle_lesion_type_name",
        "lesion_type_name",
        "lesion_type",
        "lesion_type_text",
        "lesion_type_merged",
        "predicted_lesion_type",
    ]

    for field in candidate_fields:
        value = ex.get(field)

        if value is None:
            continue

        value = normalize_lesion_type(value)

        if value and value not in {
            "none",
            "unknown",
            "lesion",
        }:
            return value

    # Last fallback: parse the merged anatomy-name list.
    names = get_rough_anatomy_names(ex)

    for i, value in enumerate(names):
        if value == "lesion type" and i + 1 < len(names):
            return normalize_lesion_type(names[i + 1])

    return ""


def bbox_features(ex):
    boxes = ex.get("bboxes", [])

    if not isinstance(boxes, list) or not boxes:
        return np.zeros(4, dtype=np.float32)

    valid = []

    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        try:
            valid.append([float(v) for v in box])
        except Exception:
            continue

    if not valid:
        return np.zeros(4, dtype=np.float32)

    return np.asarray(valid, dtype=np.float32).mean(axis=0)


def make_query_text(ex):
    """
    Build retrieval text from anatomy information only.

    The report text and lesion-type target are intentionally excluded.
    """
    rough = get_rough_anatomy_names(ex)
    coarse = get_totalseg_names(ex)
    detail = get_totalseg_detail_names(ex)

    terms = []

    # Rough anatomy is repeated to give it more influence.
    for value in rough:
        token = value.replace(" ", "_")
        terms.extend([
            f"rough_{token}",
            f"rough_{token}",
        ])

    for value in coarse:
        terms.append(
            f"coarse_{value.replace(' ', '_')}"
        )

    for value in detail:
        terms.append(
            f"detail_{value.replace(' ', '_')}"
        )

    if not terms:
        terms.append("anatomy_unknown")

    return " ".join(terms)


def pair_by_id(input_rows, target_rows):
    input_by_id = {
        str(x["id"]): x
        for x in input_rows
    }

    target_by_id = {
        str(x["id"]): x
        for x in target_rows
    }

    common = sorted(
        set(input_by_id) & set(target_by_id)
    )

    paired_inputs = []
    paired_targets = []
    missing_target = 0

    for sid in common:
        lesion_type = get_lesion_type(
            target_by_id[sid]
        )

        if not lesion_type:
            missing_target += 1
            continue

        paired_inputs.append(input_by_id[sid])
        paired_targets.append(lesion_type)

    return (
        paired_inputs,
        paired_targets,
        missing_target,
    )


class LesionTypeRAG:
    def __init__(
        self,
        bbox_weight=1.0,
        class_weight_power=0.5,
    ):
        self.bbox_weight = float(bbox_weight)
        self.class_weight_power = float(
            class_weight_power
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )

        self.nn = None
        self.train_rows = None
        self.train_targets = None
        self.train_ids = None
        self.class_weights = None

    def _combine(self, text_matrix, bbox_matrix):
        bbox_sparse = csr_matrix(
            np.asarray(
                bbox_matrix,
                dtype=np.float32,
            ) * self.bbox_weight
        )

        return hstack(
            [text_matrix, bbox_sparse],
            format="csr",
        )

    def fit(self, rows, targets):
        self.train_rows = rows
        self.train_targets = list(targets)
        self.train_ids = [
            str(x["id"])
            for x in rows
        ]

        counts = Counter(self.train_targets)
        total = len(self.train_targets)

        self.class_weights = {
            label: (
                total / count
            ) ** self.class_weight_power
            for label, count in counts.items()
        }

        texts = [
            make_query_text(x)
            for x in rows
        ]

        bboxes = np.stack([
            bbox_features(x)
            for x in rows
        ])

        text_matrix = (
            self.vectorizer.fit_transform(texts)
        )

        features = self._combine(
            text_matrix,
            bboxes,
        )

        self.nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        self.nn.fit(features)

        return self

    def _query_features(self, rows):
        texts = [
            make_query_text(x)
            for x in rows
        ]

        bboxes = np.stack([
            bbox_features(x)
            for x in rows
        ])

        text_matrix = (
            self.vectorizer.transform(texts)
        )

        return self._combine(
            text_matrix,
            bboxes,
        )

    def predict(self, rows, k):
        query = self._query_features(rows)

        k = min(k, len(self.train_rows))

        distances, indices = self.nn.kneighbors(
            query,
            n_neighbors=k,
            return_distance=True,
        )

        outputs = []

        for row, row_dist, row_idx in zip(
            rows,
            distances,
            indices,
        ):
            votes = defaultdict(float)
            retrieved = []

            for distance, idx in zip(
                row_dist,
                row_idx,
            ):
                similarity = max(
                    0.0,
                    1.0 - float(distance),
                )

                label = self.train_targets[idx]

                class_weight = (
                    self.class_weights.get(label, 1.0)
                )

                vote = max(
                    similarity,
                    1e-6,
                ) * class_weight

                votes[label] += vote

                retrieved.append({
                    "id": self.train_ids[idx],
                    "similarity": similarity,
                    "lesion_type": label,
                    "weighted_vote": vote,
                })

            ranked_votes = sorted(
                votes.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )

            predicted = ranked_votes[0][0]
            top_vote = ranked_votes[0][1]
            second_vote = (
                ranked_votes[1][1]
                if len(ranked_votes) >= 2
                else 0.0
            )

            total_vote = sum(votes.values())

            vote_share = (
                top_vote / total_vote
                if total_vote > 0
                else 0.0
            )

            vote_margin = (
                (top_vote - second_vote) / total_vote
                if total_vote > 0
                else 0.0
            )

            similarities = sorted(
                [
                    float(x["similarity"])
                    for x in retrieved
                ],
                reverse=True,
            )

            top_similarity = (
                similarities[0]
                if similarities
                else 0.0
            )

            combined_confidence = (
                0.60 * vote_share
                + 0.25 * top_similarity
                + 0.15 * vote_margin
            )

            combined_confidence = max(
                0.0,
                min(1.0, combined_confidence),
            )

            outputs.append({
                "id": str(row["id"]),
                "predicted": predicted,
                "vote_share": vote_share,
                "vote_margin": vote_margin,
                "top_similarity":
                    top_similarity,
                "confidence":
                    combined_confidence,
                "query_text":
                    make_query_text(row),
                "retrieved": retrieved,
            })

        return outputs


def evaluate_predictions(
    targets,
    predictions,
    labels,
):
    predicted_labels = [
        x["predicted"]
        for x in predictions
    ]

    return {
        "accuracy": accuracy_score(
            targets,
            predicted_labels,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                targets,
                predicted_labels,
            ),
        "macro_f1": f1_score(
            targets,
            predicted_labels,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            targets,
            predicted_labels,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "classification_report":
            classification_report(
                targets,
                predicted_labels,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
    }


def evaluate_threshold(
    targets,
    predictions,
    threshold,
):
    selected_indices = [
        i
        for i, pred in enumerate(predictions)
        if pred["confidence"] >= threshold
    ]

    n = len(targets)
    selected_n = len(selected_indices)

    if selected_n == 0:
        return {
            "threshold": threshold,
            "coverage": 0.0,
            "selected_n": 0,
            "selected_accuracy": 0.0,
            "selective_hmean": 0.0,
        }

    correct = sum(
        predictions[i]["predicted"]
        == targets[i]
        for i in selected_indices
    )

    coverage = selected_n / n
    selected_accuracy = correct / selected_n

    selective_hmean = (
        2 * coverage * selected_accuracy
        / (coverage + selected_accuracy)
        if coverage + selected_accuracy > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "coverage": coverage,
        "selected_n": selected_n,
        "selected_accuracy":
            selected_accuracy,
        "selective_hmean":
            selective_hmean,
    }


def remove_existing_lesion_type(names):
    """
    Remove an existing:
      ["lesion type", "nodule"]

    or:
      ["lesion", "type", "nodule"]

    suffix before appending a new prediction.
    """
    names = list(names)
    output = []
    skip_next = False

    for i, value in enumerate(names):
        if skip_next:
            skip_next = False
            continue

        normalized = clean_text(value)

        if normalized == "lesion type":
            skip_next = True
            continue

        output.append(normalized)

    # Handle tokenized form:
    # [..., "lesion", "type", "nodule"]
    if len(output) >= 3:
        if (
            output[-3] == "lesion"
            and output[-2] == "type"
        ):
            output = output[:-3]

    return [
        x for x in output
        if x
    ]


def apply_to_test(
    test_rows,
    predictions,
    confidence_threshold,
):
    pred_by_id = {
        x["id"]: x
        for x in predictions
    }

    output = []
    source_counts = Counter()
    predicted_distribution = Counter()

    for source_row in test_rows:
        x = copy.deepcopy(source_row)
        sid = str(x["id"])
        pred = pred_by_id[sid]

        anatomy_names = get_rough_anatomy_names(x)
        anatomy_names = remove_existing_lesion_type(
            anatomy_names
        )

        if "lesion" not in anatomy_names:
            anatomy_names = (
                ["lesion"] + anatomy_names
            )

        use_prediction = (
            pred["confidence"]
            >= confidence_threshold
        )

        final_names = list(anatomy_names)

        if use_prediction:
            final_names.extend([
                "lesion type",
                pred["predicted"],
            ])

            conditioning_source = (
                "predbbox_anatomy_plus_rag_lesion"
            )

            source_counts["rag_lesion_used"] += 1
            predicted_distribution[
                pred["predicted"]
            ] += 1
        else:
            conditioning_source = (
                "predbbox_anatomy_only_"
                "low_rag_confidence"
            )

            source_counts[
                "anatomy_only_fallback"
            ] += 1

        # Deduplicate complete entries while preserving order.
        cleaned_names = []
        seen = set()

        for value in final_names:
            value = clean_text(value)

            if not value or value in seen:
                continue

            cleaned_names.append(value)
            seen.add(value)

        anatomy_text = " ".join(cleaned_names)

        # Active R2Gen fields.
        x["rough_anatomy_names"] = cleaned_names
        x["rough_anatomy_name"] = anatomy_text
        x["anatomy_text"] = anatomy_text

        # RAG metadata.
        x["rag_lesion_type"] = pred["predicted"]
        x["rag_lesion_confidence"] = (
            pred["confidence"]
        )
        x["rag_lesion_vote_share"] = (
            pred["vote_share"]
        )
        x["rag_lesion_vote_margin"] = (
            pred["vote_margin"]
        )
        x["rag_lesion_top_similarity"] = (
            pred["top_similarity"]
        )
        x["rag_lesion_threshold"] = (
            confidence_threshold
        )
        x["rag_lesion_used"] = use_prediction
        x["rag_lesion_query_text"] = (
            pred["query_text"]
        )

        x["rag_lesion_retrieved_ids"] = [
            r["id"]
            for r in pred["retrieved"]
        ]

        x["rag_lesion_retrieved_types"] = [
            r["lesion_type"]
            for r in pred["retrieved"]
        ]

        x["rag_lesion_retrieved_similarities"] = [
            r["similarity"]
            for r in pred["retrieved"]
        ]

        x["conditioning_source"] = (
            conditioning_source
        )

        output.append(x)

    source_counts["total"] = len(output)

    return (
        output,
        dict(source_counts),
        dict(predicted_distribution),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--anatomy_only_json",
        required=True,
    )

    parser.add_argument(
        "--anatomy_plus_lesion_json",
        required=True,
    )

    parser.add_argument(
        "--predbbox_test_json",
        required=True,
    )

    parser.add_argument(
        "--out_json",
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        required=True,
    )

    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 10, 15],
    )

    parser.add_argument(
        "--bbox_weights",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0],
    )

    parser.add_argument(
        "--class_weight_powers",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5],
    )

    parser.add_argument(
        "--confidence_thresholds",
        nargs="+",
        type=float,
        default=[
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
        ],
    )

    args = parser.parse_args()

    anatomy_only = json.load(
        open(args.anatomy_only_json)
    )

    anatomy_plus_lesion = json.load(
        open(args.anatomy_plus_lesion_json)
    )

    predbbox = json.load(
        open(args.predbbox_test_json)
    )

    (
        train_rows,
        train_targets,
        train_missing,
    ) = pair_by_id(
        anatomy_only["train"],
        anatomy_plus_lesion["train"],
    )

    (
        val_rows,
        val_targets,
        val_missing,
    ) = pair_by_id(
        anatomy_only["val"],
        anatomy_plus_lesion["val"],
    )

    labels = sorted(
        set(train_targets)
        | set(val_targets)
        | set(DEFAULT_CLASSES)
    )

    print("training pairs:", len(train_rows))
    print("training missing target:", train_missing)
    print("validation pairs:", len(val_rows))
    print("validation missing target:", val_missing)
    print("predicted-box test:", len(predbbox["test"]))
    print("classes:", labels)

    print("\nTraining target distribution:")
    for label, count in Counter(
        train_targets
    ).most_common():
        print(f"{label:20s} {count}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid = []
    best_model_config = None

    for bbox_weight in args.bbox_weights:
        for class_weight_power in (
            args.class_weight_powers
        ):
            rag = LesionTypeRAG(
                bbox_weight=bbox_weight,
                class_weight_power=
                    class_weight_power,
            ).fit(
                train_rows,
                train_targets,
            )

            for k in args.k_values:
                predictions = rag.predict(
                    val_rows,
                    k=k,
                )

                metrics = evaluate_predictions(
                    val_targets,
                    predictions,
                    labels,
                )

                row = {
                    "bbox_weight": bbox_weight,
                    "class_weight_power":
                        class_weight_power,
                    "k": k,
                    "accuracy":
                        metrics["accuracy"],
                    "balanced_accuracy":
                        metrics[
                            "balanced_accuracy"
                        ],
                    "macro_f1":
                        metrics["macro_f1"],
                    "weighted_f1":
                        metrics["weighted_f1"],
                }

                grid.append(row)

                print(
                    f"bbox={bbox_weight:.2f} "
                    f"class_w={class_weight_power:.2f} "
                    f"k={k:2d} "
                    f"acc={metrics['accuracy']:.4f} "
                    f"bal={metrics['balanced_accuracy']:.4f} "
                    f"macro_f1={metrics['macro_f1']:.4f}"
                )

                score = (
                    metrics["macro_f1"],
                    metrics["balanced_accuracy"],
                    metrics["accuracy"],
                )

                if (
                    best_model_config is None
                    or score
                    > best_model_config["score"]
                ):
                    best_model_config = {
                        "score": score,
                        "bbox_weight":
                            bbox_weight,
                        "class_weight_power":
                            class_weight_power,
                        "k": k,
                        "metrics": metrics,
                        "predictions":
                            predictions,
                    }

    print("\n" + "=" * 90)
    print("BEST RETRIEVAL CONFIGURATION")
    print(
        "bbox_weight:",
        best_model_config["bbox_weight"],
    )
    print(
        "class_weight_power:",
        best_model_config[
            "class_weight_power"
        ],
    )
    print("k:", best_model_config["k"])

    best_metrics = best_model_config["metrics"]

    print("accuracy:", best_metrics["accuracy"])
    print(
        "balanced_accuracy:",
        best_metrics["balanced_accuracy"],
    )
    print("macro_f1:", best_metrics["macro_f1"])
    print(
        "weighted_f1:",
        best_metrics["weighted_f1"],
    )

    threshold_results = []

    for threshold in args.confidence_thresholds:
        result = evaluate_threshold(
            val_targets,
            best_model_config["predictions"],
            threshold,
        )

        threshold_results.append(result)

        print(
            f"threshold={threshold:.2f} "
            f"coverage={result['coverage']:.4f} "
            f"selected_acc="
            f"{result['selected_accuracy']:.4f} "
            f"hmean={result['selective_hmean']:.4f}"
        )

    # Select threshold using a balanced accuracy/coverage objective.
    # Require at least 15% validation coverage.
    valid_thresholds = [
        x for x in threshold_results
        if x["coverage"] >= 0.15
    ]

    if not valid_thresholds:
        valid_thresholds = threshold_results

    best_threshold = max(
        valid_thresholds,
        key=lambda x: (
            x["selective_hmean"],
            x["selected_accuracy"],
            x["coverage"],
        ),
    )

    print("\nBEST CONFIDENCE THRESHOLD")
    print(
        "threshold:",
        best_threshold["threshold"],
    )
    print(
        "coverage:",
        best_threshold["coverage"],
    )
    print(
        "selected accuracy:",
        best_threshold[
            "selected_accuracy"
        ],
    )

    with open(
        out_dir / "retrieval_grid.json",
        "w",
    ) as f:
        json.dump(grid, f, indent=2)

    config_to_save = {
        "bbox_weight":
            best_model_config["bbox_weight"],
        "class_weight_power":
            best_model_config[
                "class_weight_power"
            ],
        "k":
            best_model_config["k"],
        "validation_metrics": {
            key: value
            for key, value in best_metrics.items()
            if key != "classification_report"
        },
        "confidence_threshold":
            best_threshold["threshold"],
        "threshold_metrics":
            best_threshold,
    }

    with open(
        out_dir / "best_config.json",
        "w",
    ) as f:
        json.dump(
            config_to_save,
            f,
            indent=2,
        )

    with open(
        out_dir
        / "validation_classification_report.json",
        "w",
    ) as f:
        json.dump(
            best_metrics[
                "classification_report"
            ],
            f,
            indent=2,
        )

    with open(
        out_dir / "threshold_results.json",
        "w",
    ) as f:
        json.dump(
            threshold_results,
            f,
            indent=2,
        )

    # Train-only final retriever.
    final_rag = LesionTypeRAG(
        bbox_weight=
            best_model_config["bbox_weight"],
        class_weight_power=
            best_model_config[
                "class_weight_power"
            ],
    ).fit(
        train_rows,
        train_targets,
    )

    test_predictions = final_rag.predict(
        predbbox["test"],
        k=best_model_config["k"],
    )

    (
        final_test,
        source_counts,
        predicted_distribution,
    ) = apply_to_test(
        predbbox["test"],
        test_predictions,
        confidence_threshold=
            best_threshold["threshold"],
    )

    # The report checkpoint trained with lesion tokens should
    # receive the original lesion-added train and val splits.
    output_json = {
        "train": copy.deepcopy(
            anatomy_plus_lesion["train"]
        ),
        "val": copy.deepcopy(
            anatomy_plus_lesion["val"]
        ),
        "test": final_test,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out_json, "w") as f:
        json.dump(
            output_json,
            f,
            indent=2,
        )

    with open(
        out_dir / "test_predictions.json",
        "w",
    ) as f:
        json.dump(
            test_predictions,
            f,
            indent=2,
        )

    with open(
        out_dir / "test_source_counts.json",
        "w",
    ) as f:
        json.dump(
            source_counts,
            f,
            indent=2,
        )

    with open(
        out_dir
        / "test_predicted_distribution.json",
        "w",
    ) as f:
        json.dump(
            predicted_distribution,
            f,
            indent=2,
        )

    print("\nSaved R2Gen JSON:", out_json)
    print("Source counts:", source_counts)
    print(
        "Predicted type distribution:",
        predicted_distribution,
    )

    print("\nFirst test example:")
    first = final_test[0]

    for field in [
        "id",
        "bbox_source",
        "totalseg_anatomy_names",
        "rough_anatomy_names",
        "rag_lesion_type",
        "rag_lesion_confidence",
        "rag_lesion_used",
        "conditioning_source",
        "anatomy_text",
    ]:
        print(field, ":", first.get(field))


if __name__ == "__main__":
    main()
