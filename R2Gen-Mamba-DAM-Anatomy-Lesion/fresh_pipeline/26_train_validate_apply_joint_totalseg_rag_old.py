#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def clean(value):
    return " ".join(
        str(value or "")
        .replace("_", " ")
        .strip()
        .lower()
        .split()
    )


def clean_list(values):
    """
    Clean set-like labels and remove duplicates.

    Use this for TotalSegmentator anatomy labels, where repeated
    labels such as two identical lobe names do not add information.
    """
    if isinstance(values, str):
        values = [values]

    output = []
    seen = set()

    for value in values or []:
        value = clean(value)

        if not value or value in seen:
            continue

        output.append(value)
        seen.add(value)

    return output


def clean_sequence(values):
    """
    Clean an ordered token sequence without removing duplicates.

    This preserves targets such as:
      lesion lung left upper lesion type nodule
    """
    if isinstance(values, str):
        values = [values]

    output = []

    for value in values or []:
        value = clean(value)

        if value:
            output.append(value)

    return output


def get_totalseg_coarse(ex):
    names = ex.get("totalseg_anatomy_names", [])

    if not names:
        one = ex.get("totalseg_anatomy_name")
        names = [one] if one else []

    return clean_list(names)


def get_totalseg_details(ex):
    names = ex.get("totalseg_detail_names", [])

    if names:
        return clean_list(names)

    recovered = []

    for hit in ex.get("totalseg_detail_hits", []) or []:
        if not isinstance(hit, dict):
            continue

        value = (
            hit.get("totalseg_label_name")
            or hit.get("label_name")
            or hit.get("name")
        )

        if value:
            recovered.append(value)

    return clean_list(recovered)


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


def pixel_features(ex):
    counts = ex.get("totalseg_anatomy_pixel_counts", {})

    if not isinstance(counts, dict) or not counts:
        return []

    valid_counts = {}

    for name, value in counts.items():
        try:
            value = float(value)
        except Exception:
            continue

        if value > 0:
            valid_counts[name] = value

    total = sum(valid_counts.values())

    if total <= 0:
        return []

    terms = []

    for name, value in sorted(valid_counts.items()):
        ratio = value / total

        if ratio >= 0.50:
            bucket = "dominant"
        elif ratio >= 0.20:
            bucket = "medium"
        else:
            bucket = "minor"

        token = clean(name).replace(" ", "_")
        terms.append(f"pixel_{token}_{bucket}")

    return terms


def make_query_text(ex):
    """
    Build the RAG query using only raw TotalSegmentator information.

    Oracle anatomy fields, oracle lesion fields, and reports are ignored.
    """
    coarse = get_totalseg_coarse(ex)
    details = get_totalseg_details(ex)

    terms = []

    for value in coarse:
        token = value.replace(" ", "_")

        # Repeat coarse terms to increase their TF-IDF influence.
        terms.extend([
            f"coarse_{token}",
            f"coarse_{token}",
        ])

    for value in details:
        token = value.replace(" ", "_")
        terms.append(f"detail_{token}")

    terms.extend(pixel_features(ex))

    if not terms:
        terms.append("totalseg_unknown")

    return " ".join(terms)


def get_joint_target(ex):
    """
    Return the exact ordered anatomy-plus-lesion target used by R2Gen.

    The target must come from rough_anatomy_names because anatomy_text
    may omit the lesion-type suffix.
    """
    names = ex.get("rough_anatomy_names", [])

    if isinstance(names, str):
        names = [names]

    names = clean_sequence(names)

    if not names:
        text = clean(ex.get("anatomy_text"))
        names = text.split() if text else []

    if "lesion" not in names:
        names = ["lesion"] + names

    return clean_sequence(names)


def contains_lesion_type(names):
    names = clean_sequence(names)

    for i, value in enumerate(names):
        if value == "lesion type" and i + 1 < len(names):
            return True

        if (
            value == "lesion"
            and i + 2 < len(names)
            and names[i + 1] == "type"
        ):
            return True

    return False


def pair_by_id(query_rows, target_rows):
    query_by_id = {
        str(x["id"]): x
        for x in query_rows
    }

    target_by_id = {
        str(x["id"]): x
        for x in target_rows
    }

    common_ids = sorted(
        set(query_by_id) & set(target_by_id)
    )

    paired_queries = []
    paired_targets = []
    missing_lesion_type = 0

    for sid in common_ids:
        target = get_joint_target(
            target_by_id[sid]
        )

        if not contains_lesion_type(target):
            missing_lesion_type += 1

        paired_queries.append(
            query_by_id[sid]
        )

        paired_targets.append(
            tuple(target)
        )

    return (
        paired_queries,
        paired_targets,
        missing_lesion_type,
    )


def token_f1(true_names, pred_names):
    true_tokens = set()

    for value in true_names:
        true_tokens.update(clean(value).split())

    pred_tokens = set()

    for value in pred_names:
        pred_tokens.update(clean(value).split())

    if not true_tokens and not pred_tokens:
        return 1.0

    if not true_tokens or not pred_tokens:
        return 0.0

    overlap = len(true_tokens & pred_tokens)

    precision = overlap / len(pred_tokens)
    recall = overlap / len(true_tokens)

    if precision + recall == 0:
        return 0.0

    return (
        2.0 * precision * recall
        / (precision + recall)
    )


class JointTotalSegRAG:
    def __init__(self, bbox_weight=1.0):
        self.bbox_weight = float(bbox_weight)

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
        self.train_rows = list(rows)
        self.train_targets = list(targets)

        self.train_ids = [
            str(x["id"])
            for x in self.train_rows
        ]

        texts = [
            make_query_text(x)
            for x in self.train_rows
        ]

        bboxes = np.stack([
            bbox_features(x)
            for x in self.train_rows
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
        rows = list(rows)

        query_features = self._query_features(rows)

        k = min(
            int(k),
            len(self.train_rows),
        )

        distances, indices = self.nn.kneighbors(
            query_features,
            n_neighbors=k,
            return_distance=True,
        )

        outputs = []

        for row, row_distances, row_indices in zip(
            rows,
            distances,
            indices,
        ):
            votes = defaultdict(float)
            retrieved = []

            for distance, idx in zip(
                row_distances,
                row_indices,
            ):
                similarity = max(
                    0.0,
                    1.0 - float(distance),
                )

                weight = max(
                    similarity,
                    1e-6,
                )

                target = self.train_targets[idx]
                votes[target] += weight

                retrieved.append({
                    "id": self.train_ids[idx],
                    "similarity": similarity,
                    "target": list(target),
                })

            ranked = sorted(
                votes.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )

            best_target, best_vote = ranked[0]

            second_vote = (
                ranked[1][1]
                if len(ranked) >= 2
                else 0.0
            )

            total_vote = sum(votes.values())

            vote_share = (
                best_vote / total_vote
                if total_vote > 0
                else 0.0
            )

            vote_margin = (
                (best_vote - second_vote)
                / total_vote
                if total_vote > 0
                else 0.0
            )

            top_similarity = max(
                [
                    item["similarity"]
                    for item in retrieved
                ],
                default=0.0,
            )

            confidence = (
                0.60 * vote_share
                + 0.25 * top_similarity
                + 0.15 * vote_margin
            )

            confidence = max(
                0.0,
                min(1.0, confidence),
            )

            outputs.append({
                "id": str(row["id"]),
                "predicted_names": list(best_target),
                "confidence": confidence,
                "vote_share": vote_share,
                "vote_margin": vote_margin,
                "top_similarity": top_similarity,
                "query_text": make_query_text(row),
                "retrieved": retrieved,
            })

        return outputs


def evaluate(targets, predictions):
    exact = 0
    f1_values = []
    lesion_type_present = 0
    mismatches = Counter()

    for true_target, pred in zip(
        targets,
        predictions,
    ):
        true_names = list(true_target)
        pred_names = pred["predicted_names"]

        if true_names == pred_names:
            exact += 1
        else:
            mismatches[(
                " ".join(true_names),
                " ".join(pred_names),
            )] += 1

        f1_values.append(
            token_f1(
                true_names,
                pred_names,
            )
        )

        lesion_type_present += int(
            contains_lesion_type(pred_names)
        )

    n = len(targets)

    return {
        "n": n,
        "exact": exact,
        "exact_accuracy":
            exact / n if n else 0.0,
        "mean_token_f1":
            float(np.mean(f1_values))
            if f1_values else 0.0,
        "predicted_lesion_type_rate":
            lesion_type_present / n
            if n else 0.0,
        "top_mismatches": [
            {
                "true": true,
                "pred": pred,
                "count": count,
            }
            for (
                true,
                pred,
            ), count in mismatches.most_common(30)
        ],
    }


def apply_to_test(test_rows, predictions):
    pred_by_id = {
        x["id"]: x
        for x in predictions
    }

    output = []
    target_distribution = Counter()

    for source in test_rows:
        x = copy.deepcopy(source)
        sid = str(x["id"])

        if sid not in pred_by_id:
            raise KeyError(
                f"Missing RAG prediction for test ID: {sid}"
            )

        pred = pred_by_id[sid]

        names = clean_sequence(
            pred["predicted_names"]
        )

        if "lesion" not in names:
            names = ["lesion"] + names

        anatomy_text = " ".join(names)

        # Fields used by the R2Gen loader.
        x["rough_anatomy_names"] = names
        x["rough_anatomy_name"] = anatomy_text
        x["anatomy_text"] = anatomy_text

        # RAG provenance.
        x["joint_rag_predicted_names"] = names
        x["joint_rag_confidence"] = pred["confidence"]
        x["joint_rag_vote_share"] = pred["vote_share"]
        x["joint_rag_vote_margin"] = pred["vote_margin"]
        x["joint_rag_top_similarity"] = (
            pred["top_similarity"]
        )
        x["joint_rag_query_text"] = pred["query_text"]

        x["joint_rag_retrieved_ids"] = [
            r["id"]
            for r in pred["retrieved"]
        ]

        x["joint_rag_retrieved_targets"] = [
            r["target"]
            for r in pred["retrieved"]
        ]

        x["joint_rag_retrieved_similarities"] = [
            r["similarity"]
            for r in pred["retrieved"]
        ]

        x["conditioning_source"] = (
            "raw_totalseg_joint_rag"
        )

        target_distribution[
            anatomy_text
        ] += 1

        output.append(x)

    return output, target_distribution


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gtbbox_totalseg_json",
        required=True,
    )

    parser.add_argument(
        "--oracle_anatomy_lesion_json",
        required=True,
    )

    parser.add_argument(
        "--predbbox_totalseg_json",
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
        default=[1, 3, 5, 7, 10, 15, 25],
    )

    parser.add_argument(
        "--bbox_weights",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 3.0],
    )

    args = parser.parse_args()

    print("Loading GT-box TotalSeg query JSON...", flush=True)

    query_data = json.load(
        open(args.gtbbox_totalseg_json)
    )

    print("Loading oracle anatomy-plus-lesion target JSON...", flush=True)

    target_data = json.load(
        open(args.oracle_anatomy_lesion_json)
    )

    print("Loading predicted-box TotalSeg test JSON...", flush=True)

    predbbox_data = json.load(
        open(args.predbbox_totalseg_json)
    )

    (
        train_queries,
        train_targets,
        train_missing_lesion,
    ) = pair_by_id(
        query_data["train"],
        target_data["train"],
    )

    (
        val_queries,
        val_targets,
        val_missing_lesion,
    ) = pair_by_id(
        query_data["val"],
        target_data["val"],
    )

    test_queries = predbbox_data["test"]

    print("Train pairs:", len(train_queries), flush=True)
    print("Validation pairs:", len(val_queries), flush=True)
    print(
        "Predicted-box test queries:",
        len(test_queries),
        flush=True,
    )
    print(
        "Train targets missing lesion type:",
        train_missing_lesion,
        flush=True,
    )
    print(
        "Validation targets missing lesion type:",
        val_missing_lesion,
        flush=True,
    )

    if train_missing_lesion != 0:
        raise RuntimeError(
            "Training targets are missing lesion-type tokens."
        )

    if val_missing_lesion != 0:
        raise RuntimeError(
            "Validation targets are missing lesion-type tokens."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid = []
    best = None

    for bbox_weight in args.bbox_weights:
        print(
            f"\nFitting RAG with bbox_weight={bbox_weight}...",
            flush=True,
        )

        rag = JointTotalSegRAG(
            bbox_weight=bbox_weight
        ).fit(
            train_queries,
            train_targets,
        )

        for k in args.k_values:
            predictions = rag.predict(
                val_queries,
                k=k,
            )

            metrics = evaluate(
                val_targets,
                predictions,
            )

            row = {
                "bbox_weight": bbox_weight,
                "k": k,
                "exact_accuracy":
                    metrics["exact_accuracy"],
                "mean_token_f1":
                    metrics["mean_token_f1"],
                "predicted_lesion_type_rate":
                    metrics[
                        "predicted_lesion_type_rate"
                    ],
            }

            grid.append(row)

            print(
                f"bbox={bbox_weight:.2f} "
                f"k={k:2d} "
                f"exact={metrics['exact_accuracy']:.4f} "
                f"token_f1={metrics['mean_token_f1']:.4f} "
                f"lesion_rate="
                f"{metrics['predicted_lesion_type_rate']:.4f}",
                flush=True,
            )

            score = (
                metrics["mean_token_f1"],
                metrics["exact_accuracy"],
            )

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "bbox_weight": bbox_weight,
                    "k": k,
                    "metrics": metrics,
                }

    if best is None:
        raise RuntimeError(
            "No RAG configuration was evaluated."
        )

    print("\nBEST CONFIGURATION", flush=True)
    print(
        "bbox_weight:",
        best["bbox_weight"],
        flush=True,
    )
    print(
        "k:",
        best["k"],
        flush=True,
    )
    print(
        "exact_accuracy:",
        best["metrics"]["exact_accuracy"],
        flush=True,
    )
    print(
        "mean_token_f1:",
        best["metrics"]["mean_token_f1"],
        flush=True,
    )

    with open(
        out_dir / "validation_grid.json",
        "w",
    ) as f:
        json.dump(grid, f, indent=2)

    with open(
        out_dir / "best_config.json",
        "w",
    ) as f:
        json.dump(best, f, indent=2)

    print(
        "\nFitting final RAG model and applying to predicted-box test set...",
        flush=True,
    )

    final_rag = JointTotalSegRAG(
        bbox_weight=best["bbox_weight"]
    ).fit(
        train_queries,
        train_targets,
    )

    test_predictions = final_rag.predict(
        test_queries,
        k=best["k"],
    )

    (
        final_test,
        target_distribution,
    ) = apply_to_test(
        test_queries,
        test_predictions,
    )

    output = {
        "train": copy.deepcopy(
            target_data["train"]
        ),
        "val": copy.deepcopy(
            target_data["val"]
        ),
        "test": final_test,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

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
        out_dir / "test_target_distribution.json",
        "w",
    ) as f:
        json.dump(
            dict(target_distribution),
            f,
            indent=2,
        )

    print(
        "\nSaved R2Gen JSON:",
        out_path,
        flush=True,
    )

    print(
        "Saved predictions:",
        out_dir / "test_predictions.json",
        flush=True,
    )

    first = final_test[0]

    print(
        "\nFirst predicted test example:",
        flush=True,
    )

    for key in [
        "id",
        "bbox_source",
        "totalseg_anatomy_names",
        "totalseg_detail_names",
        "joint_rag_predicted_names",
        "joint_rag_confidence",
        "rough_anatomy_names",
        "anatomy_text",
    ]:
        print(
            key,
            ":",
            first.get(key),
            flush=True,
        )


if __name__ == "__main__":
    main()
