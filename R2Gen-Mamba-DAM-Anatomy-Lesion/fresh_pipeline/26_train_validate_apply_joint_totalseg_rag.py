#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion
from tqdm import tqdm


# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------

def normalize_space(value):
    return " ".join(
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
        .split()
    )


def normalize_query(value):
    value = normalize_space(value)

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value,
    )

    return " ".join(value.split())


def split_report_tokens(report):
    """
    The DeepLesion short reports are comma-separated concept phrases.

    Example:
        chest, lung, nodule, left lung, reticular

    Returns a normalized, duplicate-free list while preserving order.
    """
    report = str(report or "").strip()

    if not report:
        return []

    raw_tokens = report.split(",")

    output = []
    seen = set()

    for token in raw_tokens:
        token = normalize_space(token)

        if not token:
            continue

        if token in seen:
            continue

        output.append(token)
        seen.add(token)

    return output


def join_report_tokens(tokens):
    return ", ".join(tokens)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def token_set_metrics(reference, prediction):
    reference_tokens = set(
        split_report_tokens(reference)
    )

    prediction_tokens = set(
        split_report_tokens(prediction)
    )

    true_positive = len(
        reference_tokens & prediction_tokens
    )

    false_positive = len(
        prediction_tokens - reference_tokens
    )

    false_negative = len(
        reference_tokens - prediction_tokens
    )

    precision = (
        true_positive /
        (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    union = reference_tokens | prediction_tokens

    jaccard = (
        len(reference_tokens & prediction_tokens)
        / len(union)
        if union
        else 0.0
    )

    exact = int(
        split_report_tokens(reference)
        == split_report_tokens(prediction)
    )

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "exact": exact,
    }


def evaluate_predictions(rows):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    sample_f1 = []
    sample_jaccard = []
    exact_values = []

    for row in rows:
        metrics = token_set_metrics(
            row["reference"],
            row["prediction"],
        )

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        sample_f1.append(metrics["f1"])
        sample_jaccard.append(
            metrics["jaccard"]
        )
        exact_values.append(metrics["exact"])

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    micro_f1 = (
        2.0 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    return {
        "samples": len(rows),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": (
            float(np.mean(sample_f1))
            if sample_f1
            else 0.0
        ),
        "mean_jaccard": (
            float(np.mean(sample_jaccard))
            if sample_jaccard
            else 0.0
        ),
        "exact_match": (
            float(np.mean(exact_values))
            if exact_values
            else 0.0
        ),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


# ---------------------------------------------------------------------
# Retrieval model
# ---------------------------------------------------------------------

def build_vectorizer():
    """
    Word features capture exact anatomy / lesion type concepts.
    Character features help when spacing or minor formatting differs.
    """
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                lowercase=True,
                sublinear_tf=True,
                norm="l2",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                lowercase=True,
                sublinear_tf=True,
                norm="l2",
                min_df=1,
            ),
        ),
    ])


def similarity_from_distance(distance):
    similarity = 1.0 - float(distance)

    return max(0.0, min(1.0, similarity))


def retrieve_neighbors(
    query_matrix,
    nn_model,
    train_rows,
    query_rows,
    max_k,
    exclude_self=False,
):
    """
    Returns a list of neighbor lists.

    Each neighbor:
        {
            "index": train-row index,
            "id": train sample ID,
            "query": training query,
            "report": training report,
            "similarity": cosine similarity,
            "rank": 1-based rank
        }
    """
    extra = 1 if exclude_self else 0

    requested_neighbors = min(
        len(train_rows),
        max_k + extra + 5,
    )

    distances, indices = (
        nn_model.kneighbors(
            query_matrix,
            n_neighbors=requested_neighbors,
            return_distance=True,
        )
    )

    all_neighbors = []

    for query_row, row_distances, row_indices in zip(
        query_rows,
        distances,
        indices,
    ):
        query_id = str(query_row["id"])

        neighbors = []

        for distance, index in zip(
            row_distances,
            row_indices,
        ):
            train_row = train_rows[int(index)]
            train_id = str(train_row["id"])

            if (
                exclude_self
                and train_id == query_id
            ):
                continue

            neighbor = {
                "index": int(index),
                "id": train_id,
                "query": train_row["query"],
                "report": train_row["report"],
                "similarity":
                    similarity_from_distance(
                        distance
                    ),
                "rank": len(neighbors) + 1,
            }

            neighbors.append(neighbor)

            if len(neighbors) >= max_k:
                break

        if not neighbors:
            raise RuntimeError(
                f"No neighbors found for query ID "
                f"{query_id}"
            )

        all_neighbors.append(neighbors)

    return all_neighbors


# ---------------------------------------------------------------------
# Report construction from retrieved neighbors
# ---------------------------------------------------------------------

def build_retrieved_report(
    neighbors,
    k,
    vote_threshold,
    similarity_power=2.0,
    rank_decay=0.10,
):
    """
    Build a report by weighted token voting over top-k reports.

    Weight for a neighbor:
        similarity ** similarity_power
        divided by rank-decay factor

    A token is retained when:
        token_weight / total_neighbor_weight
        >= vote_threshold

    Tokens are ordered by their weighted mean position in retrieved
    reports.
    """
    selected_neighbors = neighbors[:k]

    # k=1 should reproduce the top neighbor exactly.
    if k == 1:
        return (
            selected_neighbors[0]["report"],
            selected_neighbors,
            {
                "total_weight": 1.0,
                "selected_tokens":
                    split_report_tokens(
                        selected_neighbors[0][
                            "report"
                        ]
                    ),
                "token_scores": {},
            },
        )

    token_weight = defaultdict(float)
    token_position_weight = defaultdict(float)
    token_position_denominator = defaultdict(float)

    neighbor_weights = []

    for rank_index, neighbor in enumerate(
        selected_neighbors
    ):
        similarity = max(
            neighbor["similarity"],
            1e-8,
        )

        weight = (
            similarity ** similarity_power
        ) / (
            1.0 + rank_decay * rank_index
        )

        neighbor_weights.append(weight)

        tokens = split_report_tokens(
            neighbor["report"]
        )

        for position, token in enumerate(tokens):
            token_weight[token] += weight

            token_position_weight[token] += (
                weight * position
            )

            token_position_denominator[token] += (
                weight
            )

    total_weight = sum(neighbor_weights)

    if total_weight <= 0:
        total_weight = float(
            len(selected_neighbors)
        )

    token_scores = {
        token: weight / total_weight
        for token, weight in token_weight.items()
    }

    selected_tokens = [
        token
        for token, score in token_scores.items()
        if score >= vote_threshold
    ]

    # Guarantee a non-empty report.
    if not selected_tokens:
        selected_tokens = split_report_tokens(
            selected_neighbors[0]["report"]
        )

    def token_order_key(token):
        denominator = (
            token_position_denominator[token]
        )

        mean_position = (
            token_position_weight[token]
            / denominator
            if denominator > 0
            else 1e9
        )

        return (
            mean_position,
            -token_scores.get(token, 0.0),
            token,
        )

    selected_tokens = sorted(
        selected_tokens,
        key=token_order_key,
    )

    prediction = join_report_tokens(
        selected_tokens
    )

    debug = {
        "total_weight": total_weight,
        "selected_tokens": selected_tokens,
        "token_scores": {
            key: token_scores[key]
            for key in sorted(
                token_scores,
                key=lambda x: (
                    -token_scores[x],
                    x,
                ),
            )
        },
    }

    return prediction, selected_neighbors, debug


# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------

def prepare_split(
    raw_rows,
    split,
    query_field,
    report_field,
):
    prepared = []

    missing_query = []
    missing_report = []
    duplicate_ids = []

    seen_ids = set()

    for index, example in enumerate(raw_rows):
        sample_id = str(
            example.get("id", "")
        ).strip()

        if not sample_id:
            sample_id = f"{split}_{index:06d}"

        if sample_id in seen_ids:
            duplicate_ids.append(sample_id)

        seen_ids.add(sample_id)

        query = normalize_query(
            example.get(query_field)
        )

        report = str(
            example.get(report_field, "")
            or ""
        ).strip()

        if not query:
            missing_query.append(sample_id)

        if not report:
            missing_report.append(sample_id)

        prepared.append({
            "id": sample_id,
            "query": query,
            "report": report,
            "original_index": index,
        })

    if duplicate_ids:
        raise RuntimeError(
            f"{split}: duplicate IDs detected. "
            f"Examples: {duplicate_ids[:20]}"
        )

    if missing_query:
        raise RuntimeError(
            f"{split}: {len(missing_query)} "
            f"samples have missing query text. "
            f"Examples: {missing_query[:20]}"
        )

    if missing_report:
        raise RuntimeError(
            f"{split}: {len(missing_report)} "
            f"samples have missing report text. "
            f"Examples: {missing_report[:20]}"
        )

    return prepared


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def predict_from_neighbors(
    query_rows,
    all_neighbors,
    k,
    vote_threshold,
):
    predictions = []

    for query_row, neighbors in zip(
        query_rows,
        all_neighbors,
    ):
        (
            predicted_report,
            selected_neighbors,
            debug,
        ) = build_retrieved_report(
            neighbors=neighbors,
            k=k,
            vote_threshold=vote_threshold,
        )

        row = {
            "id": query_row["id"],
            "query": query_row["query"],
            "reference": query_row["report"],
            "prediction": predicted_report,
            "neighbor_ids": [
                x["id"]
                for x in selected_neighbors
            ],
            "neighbor_similarities": [
                x["similarity"]
                for x in selected_neighbors
            ],
            "neighbor_queries": [
                x["query"]
                for x in selected_neighbors
            ],
            "neighbor_reports": [
                x["report"]
                for x in selected_neighbors
            ],
            "token_scores":
                debug["token_scores"],
        }

        metrics = token_set_metrics(
            query_row["report"],
            predicted_report,
        )

        row.update({
            "sample_precision":
                metrics["precision"],
            "sample_recall":
                metrics["recall"],
            "sample_f1":
                metrics["f1"],
            "sample_jaccard":
                metrics["jaccard"],
            "sample_exact":
                metrics["exact"],
        })

        predictions.append(row)

    return predictions


def write_prediction_csv(path, rows):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "id",
        "query",
        "reference",
        "prediction",
        "sample_precision",
        "sample_recall",
        "sample_f1",
        "sample_jaccard",
        "sample_exact",
        "neighbor_ids",
        "neighbor_similarities",
        "neighbor_queries",
        "neighbor_reports",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:
            output = dict(row)

            for key in [
                "neighbor_ids",
                "neighbor_similarities",
                "neighbor_queries",
                "neighbor_reports",
            ]:
                output[key] = json.dumps(
                    output[key],
                    ensure_ascii=False,
                )

            writer.writerow({
                key: output.get(key)
                for key in columns
            })


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train, validate, and apply joint "
            "anatomy-plus-lesion report RAG."
        )
    )

    parser.add_argument(
        "--input_json",
        required=True,
        help=(
            "JSON containing train/val/test and "
            "rag_query_text."
        ),
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
        "--query_field",
        default="rag_query_text",
    )

    parser.add_argument(
        "--report_field",
        default="report",
    )

    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 9, 15],
    )

    parser.add_argument(
        "--vote_thresholds",
        type=float,
        nargs="+",
        default=[
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
        ],
    )

    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.out_json)
    output_dir = Path(args.out_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(input_path, "r") as f:
        raw_data = json.load(f)

    for split in ["train", "val", "test"]:
        if split not in raw_data:
            raise KeyError(
                f"Missing split '{split}' in "
                f"{input_path}"
            )

    train_rows = prepare_split(
        raw_data["train"],
        "train",
        args.query_field,
        args.report_field,
    )

    val_rows = prepare_split(
        raw_data["val"],
        "val",
        args.query_field,
        args.report_field,
    )

    test_rows = prepare_split(
        raw_data["test"],
        "test",
        args.query_field,
        args.report_field,
    )

    print("train rows:", len(train_rows))
    print("val rows:", len(val_rows))
    print("test rows:", len(test_rows))

    print("\nFitting train-only TF-IDF index...")

    vectorizer = build_vectorizer()

    train_queries = [
        row["query"]
        for row in train_rows
    ]

    train_matrix = vectorizer.fit_transform(
        train_queries
    )

    val_matrix = vectorizer.transform([
        row["query"]
        for row in val_rows
    ])

    test_matrix = vectorizer.transform([
        row["query"]
        for row in test_rows
    ])

    print(
        "train matrix shape:",
        train_matrix.shape,
    )

    nn_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nn_model.fit(train_matrix)

    max_k = max(args.k_values)

    print("\nRetrieving validation neighbors...")

    val_neighbors = retrieve_neighbors(
        query_matrix=val_matrix,
        nn_model=nn_model,
        train_rows=train_rows,
        query_rows=val_rows,
        max_k=max_k,
        exclude_self=False,
    )

    # -------------------------------------------------------------
    # Validation tuning
    # -------------------------------------------------------------

    tuning_rows = []

    best_config = None
    best_metrics = None

    print("\nValidation parameter search")

    for k in args.k_values:
        if k < 1:
            continue

        if k > len(train_rows):
            continue

        thresholds = (
            [0.0]
            if k == 1
            else args.vote_thresholds
        )

        for threshold in thresholds:
            predictions = predict_from_neighbors(
                query_rows=val_rows,
                all_neighbors=val_neighbors,
                k=k,
                vote_threshold=threshold,
            )

            metrics = evaluate_predictions(
                predictions
            )

            result = {
                "k": k,
                "vote_threshold": threshold,
                **metrics,
            }

            tuning_rows.append(result)

            print(
                f"k={k:2d} "
                f"threshold={threshold:.2f} "
                f"micro_F1="
                f"{metrics['micro_f1']:.6f} "
                f"macro_F1="
                f"{metrics['macro_f1']:.6f} "
                f"exact="
                f"{metrics['exact_match']:.6f}"
            )

            score = (
                metrics["micro_f1"],
                metrics["macro_f1"],
                metrics["mean_jaccard"],
                metrics["exact_match"],
                -k,
                -threshold,
            )

            if best_config is None:
                best_config = {
                    "k": k,
                    "vote_threshold":
                        threshold,
                }
                best_metrics = metrics
                best_score = score

            elif score > best_score:
                best_config = {
                    "k": k,
                    "vote_threshold":
                        threshold,
                }
                best_metrics = metrics
                best_score = score

    if best_config is None:
        raise RuntimeError(
            "No valid RAG parameter configuration."
        )

    print("\nBest validation configuration")
    print(json.dumps(
        {
            **best_config,
            **best_metrics,
        },
        indent=2,
    ))

    tuning_csv = (
        output_dir /
        "validation_parameter_search.csv"
    )

    with open(
        tuning_csv,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                tuning_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(tuning_rows)

    with open(
        output_dir / "best_rag_config.json",
        "w",
    ) as f:
        json.dump(
            {
                "input_json":
                    str(input_path),
                "query_field":
                    args.query_field,
                "report_field":
                    args.report_field,
                "training_database_size":
                    len(train_rows),
                "best_config":
                    best_config,
                "validation_metrics":
                    best_metrics,
            },
            f,
            indent=2,
        )

    best_k = int(best_config["k"])
    best_threshold = float(
        best_config["vote_threshold"]
    )

    # -------------------------------------------------------------
    # Retrieve train neighbors using leave-one-out
    # -------------------------------------------------------------

    print(
        "\nRetrieving train neighbors "
        "with self-match exclusion..."
    )

    train_neighbors = retrieve_neighbors(
        query_matrix=train_matrix,
        nn_model=nn_model,
        train_rows=train_rows,
        query_rows=train_rows,
        max_k=best_k,
        exclude_self=True,
    )

    print("\nRetrieving test neighbors...")

    test_neighbors = retrieve_neighbors(
        query_matrix=test_matrix,
        nn_model=nn_model,
        train_rows=train_rows,
        query_rows=test_rows,
        max_k=best_k,
        exclude_self=False,
    )

    # -------------------------------------------------------------
    # Final predictions
    # -------------------------------------------------------------

    print("\nGenerating train predictions...")

    train_predictions = predict_from_neighbors(
        query_rows=train_rows,
        all_neighbors=train_neighbors,
        k=best_k,
        vote_threshold=best_threshold,
    )

    print("Generating validation predictions...")

    val_predictions = predict_from_neighbors(
        query_rows=val_rows,
        all_neighbors=val_neighbors,
        k=best_k,
        vote_threshold=best_threshold,
    )

    print("Generating test predictions...")

    test_predictions = predict_from_neighbors(
        query_rows=test_rows,
        all_neighbors=test_neighbors,
        k=best_k,
        vote_threshold=best_threshold,
    )

    metrics_by_split = {
        "train_leave_one_out":
            evaluate_predictions(
                train_predictions
            ),
        "val":
            evaluate_predictions(
                val_predictions
            ),
        "test":
            evaluate_predictions(
                test_predictions
            ),
    }

    print("\nFinal retrieval metrics")

    for split, metrics in (
        metrics_by_split.items()
    ):
        print("\n", split)
        print(json.dumps(
            metrics,
            indent=2,
        ))

    with open(
        output_dir / "retrieval_metrics.json",
        "w",
    ) as f:
        json.dump(
            {
                "best_config":
                    best_config,
                "metrics":
                    metrics_by_split,
            },
            f,
            indent=2,
        )

    write_prediction_csv(
        output_dir /
        "train_leave_one_out_predictions.csv",
        train_predictions,
    )

    write_prediction_csv(
        output_dir /
        "val_predictions.csv",
        val_predictions,
    )

    write_prediction_csv(
        output_dir /
        "test_predictions.csv",
        test_predictions,
    )

    # -------------------------------------------------------------
    # Write retrieval fields into combined JSON
    # -------------------------------------------------------------

    output_data = copy.deepcopy(raw_data)

    prediction_lookup = {
        "train": {
            row["id"]: row
            for row in train_predictions
        },
        "val": {
            row["id"]: row
            for row in val_predictions
        },
        "test": {
            row["id"]: row
            for row in test_predictions
        },
    }

    for split in ["train", "val", "test"]:
        for example in output_data[split]:
            sample_id = str(
                example.get("id")
            )

            prediction = (
                prediction_lookup[split][
                    sample_id
                ]
            )

            example[
                "rag_retrieved_report"
            ] = prediction["prediction"]

            # Compatibility alias.
            example[
                "retrieved_report"
            ] = prediction["prediction"]

            example[
                "rag_neighbor_ids"
            ] = prediction["neighbor_ids"]

            example[
                "rag_neighbor_similarities"
            ] = prediction[
                "neighbor_similarities"
            ]

            example[
                "rag_neighbor_queries"
            ] = prediction[
                "neighbor_queries"
            ]

            example[
                "rag_neighbor_reports"
            ] = prediction[
                "neighbor_reports"
            ]

            example["rag_top_k"] = best_k

            example[
                "rag_vote_threshold"
            ] = best_threshold

            example[
                "rag_index_source"
            ] = "train_only"

            example[
                "rag_train_self_match_excluded"
            ] = (
                split == "train"
            )

            example[
                "rag_retrieval_sample_f1"
            ] = prediction["sample_f1"]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            output_data,
            f,
            indent=2,
        )

    print("\nSaved output JSON:", output_path)
    print("Saved results directory:", output_dir)


if __name__ == "__main__":
    main()
