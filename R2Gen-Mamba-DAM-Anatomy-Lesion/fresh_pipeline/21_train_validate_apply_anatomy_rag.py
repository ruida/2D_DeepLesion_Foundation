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


def clean_name(value):
    if value is None:
        return ""

    return " ".join(
        str(value)
        .replace("_", " ")
        .strip()
        .lower()
        .split()
    )


def clean_name_list(values):
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    cleaned = []
    seen = set()

    for value in values:
        name = clean_name(value)

        if not name or name in {"background"}:
            continue

        if name not in seen:
            cleaned.append(name)
            seen.add(name)

    return cleaned


def get_totalseg_names(ex):
    names = clean_name_list(ex.get("totalseg_anatomy_names"))

    if not names:
        one = clean_name(ex.get("totalseg_anatomy_name"))
        if one:
            names = [one]

    return names


def get_detail_names(ex):
    names = clean_name_list(ex.get("totalseg_detail_names"))

    if not names:
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

                value = clean_name(value)
                if value:
                    recovered.append(value)

        names = clean_name_list(recovered)

    return names


def get_target_names(ex):
    names = clean_name_list(ex.get("rough_anatomy_names"))

    if not names:
        one = clean_name(ex.get("rough_anatomy_name"))
        if one:
            # A singular field may contain an entire phrase.
            names = one.split()

    if not names:
        names = ["lesion", "unknown"]

    if "lesion" not in names:
        names = ["lesion"] + names

    return clean_name_list(names)


def bbox_features(ex):
    """
    Return mean normalized cx, cy, w, h across available boxes.
    """
    boxes = ex.get("bboxes", [])

    if not isinstance(boxes, list) or not boxes:
        return np.zeros(4, dtype=np.float32)

    valid = []

    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        try:
            values = [float(v) for v in box]
        except Exception:
            continue

        valid.append(values)

    if not valid:
        return np.zeros(4, dtype=np.float32)

    return np.asarray(valid, dtype=np.float32).mean(axis=0)


def make_retrieval_text(ex):
    coarse = get_totalseg_names(ex)
    detail = get_detail_names(ex)

    parts = []

    # Repeat coarse terms to give them slightly more retrieval weight.
    if coarse:
        parts.extend(["coarse_" + x.replace(" ", "_") for x in coarse])
        parts.extend(["coarse_" + x.replace(" ", "_") for x in coarse])

    if detail:
        parts.extend(["detail_" + x.replace(" ", "_") for x in detail])

    if not parts:
        parts.append("coarse_unknown")

    return " ".join(parts)


def target_key(names):
    return tuple(names)


def token_f1(true_names, pred_names):
    true_set = set(true_names) - {"lesion"}
    pred_set = set(pred_names) - {"lesion"}

    if not true_set and not pred_set:
        return 1.0

    if not true_set or not pred_set:
        return 0.0

    overlap = len(true_set & pred_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(true_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


class AnatomyRAG:
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
        self.train_samples = None
        self.train_targets = None
        self.train_ids = None

    def _combine_features(self, text_matrix, bbox_matrix):
        bbox_sparse = csr_matrix(
            np.asarray(bbox_matrix, dtype=np.float32) * self.bbox_weight
        )
        return hstack([text_matrix, bbox_sparse], format="csr")

    def fit(self, samples):
        self.train_samples = samples
        self.train_ids = [str(x.get("id")) for x in samples]
        self.train_targets = [
            target_key(get_target_names(x))
            for x in samples
        ]

        texts = [make_retrieval_text(x) for x in samples]
        bboxes = np.stack([bbox_features(x) for x in samples])

        text_matrix = self.vectorizer.fit_transform(texts)
        features = self._combine_features(text_matrix, bboxes)

        self.nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        self.nn.fit(features)
        return self

    def query_features(self, samples):
        texts = [make_retrieval_text(x) for x in samples]
        bboxes = np.stack([bbox_features(x) for x in samples])

        text_matrix = self.vectorizer.transform(texts)
        return self._combine_features(text_matrix, bboxes)

    def predict(self, samples, k):
        query = self.query_features(samples)

        k = min(int(k), len(self.train_samples))

        distances, indices = self.nn.kneighbors(
            query,
            n_neighbors=k,
            return_distance=True,
        )

        outputs = []

        for sample, row_distances, row_indices in zip(
            samples, distances, indices
        ):
            votes = defaultdict(float)
            retrieved = []

            for distance, idx in zip(row_distances, row_indices):
                similarity = max(0.0, 1.0 - float(distance))

                # Avoid zero total weight for very distant neighbors.
                weight = max(similarity, 1e-6)

                target = self.train_targets[idx]
                votes[target] += weight

                retrieved.append({
                    "id": self.train_ids[idx],
                    "similarity": similarity,
                    "target_names": list(target),
                    "retrieval_text":
                        make_retrieval_text(self.train_samples[idx]),
                })

            best_target, best_vote = max(
                votes.items(),
                key=lambda pair: pair[1],
            )

            total_vote = sum(votes.values())
            confidence = (
                best_vote / total_vote
                if total_vote > 0
                else 0.0
            )

            outputs.append({
                "id": str(sample.get("id")),
                "predicted_names": list(best_target),
                "confidence": confidence,
                "retrieved": retrieved,
                "query_text": make_retrieval_text(sample),
            })

        return outputs


def evaluate(samples, predictions):
    exact = 0
    f1_values = []
    no_overlap = 0
    mismatches = Counter()

    for ex, pred in zip(samples, predictions):
        true_names = get_target_names(ex)
        pred_names = pred["predicted_names"]

        if true_names == pred_names:
            exact += 1
        else:
            mismatches[
                (" ".join(true_names), " ".join(pred_names))
            ] += 1

        f1 = token_f1(true_names, pred_names)
        f1_values.append(f1)

        true_set = set(true_names) - {"lesion"}
        pred_set = set(pred_names) - {"lesion"}

        if true_set and pred_set and not (true_set & pred_set):
            no_overlap += 1

    n = len(samples)

    return {
        "n": n,
        "exact": exact,
        "exact_accuracy": exact / n if n else 0.0,
        "mean_token_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "no_overlap": no_overlap,
        "no_overlap_rate": no_overlap / n if n else 0.0,
        "top_mismatches": [
            {
                "true": true,
                "pred": pred,
                "count": count,
            }
            for (true, pred), count in mismatches.most_common(30)
        ],
    }


def apply_predictions(test_samples, predictions):
    pred_by_id = {
        x["id"]: x
        for x in predictions
    }

    output = []

    for src in test_samples:
        x = copy.deepcopy(src)
        sid = str(x.get("id"))
        pred = pred_by_id[sid]

        names = clean_name_list(pred["predicted_names"])

        if not names:
            names = ["lesion", "unknown"]

        if "lesion" not in names:
            names = ["lesion"] + names

        names = clean_name_list(names)
        anatomy_text = " ".join(names)

        # Fields consumed by modules/datasets.py in text mode.
        x["rough_anatomy_names"] = names
        x["rough_anatomy_name"] = anatomy_text
        x["anatomy_text"] = anatomy_text

        # Provenance/debug fields.
        x["anatomy_source"] = "train_only_rag"
        x["conditioning_source"] = "predbbox_totalseg_rag"
        x["rag_query_text"] = pred["query_text"]
        x["rag_predicted_anatomy_names"] = names
        x["rag_confidence"] = pred["confidence"]
        x["rag_retrieved_ids"] = [
            r["id"] for r in pred["retrieved"]
        ]
        x["rag_retrieved_similarities"] = [
            r["similarity"] for r in pred["retrieved"]
        ]
        x["rag_retrieved_targets"] = [
            r["target_names"] for r in pred["retrieved"]
        ]

        output.append(x)

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--merged_json",
        required=True,
        help="Train/val merged TotalSeg + oracle anatomy JSON.",
    )

    parser.add_argument(
        "--predbbox_json",
        required=True,
        help="Predicted-box TotalSeg JSON.",
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

    args = parser.parse_args()

    merged = json.load(open(args.merged_json))
    predbbox = json.load(open(args.predbbox_json))

    train_samples = merged["train"]
    val_samples = merged["val"]
    test_samples = predbbox["test"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("train index:", len(train_samples))
    print("validation:", len(val_samples))
    print("predicted-box test:", len(test_samples))

    all_results = []
    best = None

    for bbox_weight in args.bbox_weights:
        print("\n" + "=" * 90)
        print("bbox_weight:", bbox_weight)

        rag = AnatomyRAG(
            bbox_weight=bbox_weight
        ).fit(train_samples)

        for k in args.k_values:
            predictions = rag.predict(val_samples, k=k)
            metrics = evaluate(val_samples, predictions)

            row = {
                "bbox_weight": bbox_weight,
                "k": k,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "top_mismatches"
                },
            }

            all_results.append(row)

            print(
                f"k={k:2d} "
                f"exact={metrics['exact_accuracy']:.4f} "
                f"token_f1={metrics['mean_token_f1']:.4f} "
                f"no_overlap={metrics['no_overlap_rate']:.4f}"
            )

            score = (
                metrics["mean_token_f1"],
                metrics["exact_accuracy"],
                -metrics["no_overlap_rate"],
            )

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "bbox_weight": bbox_weight,
                    "k": k,
                    "metrics": metrics,
                }

    print("\n" + "=" * 90)
    print("BEST VALIDATION CONFIGURATION")
    print("bbox_weight:", best["bbox_weight"])
    print("k:", best["k"])
    print(
        "exact_accuracy:",
        best["metrics"]["exact_accuracy"],
    )
    print(
        "mean_token_f1:",
        best["metrics"]["mean_token_f1"],
    )
    print(
        "no_overlap_rate:",
        best["metrics"]["no_overlap_rate"],
    )

    with open(out_dir / "validation_grid_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(out_dir / "best_validation_config.json", "w") as f:
        json.dump(best, f, indent=2)

    # Refit using train only, with the selected hyperparameters.
    # Validation was used only for selecting k and bbox weight.
    rag = AnatomyRAG(
        bbox_weight=best["bbox_weight"]
    ).fit(train_samples)

    test_predictions = rag.predict(
        test_samples,
        k=best["k"],
    )

    final_test = apply_predictions(
        test_samples,
        test_predictions,
    )

    output_json = {
        # Preserve the original train/val paired target splits.
        "train": copy.deepcopy(merged["train"]),
        "val": copy.deepcopy(merged["val"]),
        "test": final_test,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(output_json, f, indent=2)

    with open(out_dir / "test_rag_predictions.json", "w") as f:
        json.dump(test_predictions, f, indent=2)

    print("\nSaved R2Gen JSON:", out_json)
    print(
        "Saved predictions:",
        out_dir / "test_rag_predictions.json",
    )

    print("\nFirst predicted test sample:")
    first = output_json["test"][0]

    for key in [
        "id",
        "bbox_source",
        "totalseg_anatomy_names",
        "totalseg_detail_names",
        "rag_query_text",
        "rag_predicted_anatomy_names",
        "rag_confidence",
        "rough_anatomy_names",
        "anatomy_text",
    ]:
        print(key, ":", first.get(key))


if __name__ == "__main__":
    main()
