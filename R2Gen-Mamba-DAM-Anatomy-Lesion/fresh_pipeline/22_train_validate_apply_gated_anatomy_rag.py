#!/usr/bin/env python3

import argparse
import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_base_rag_module(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Base RAG script not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "base_anatomy_rag",
        str(path),
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def clean_names(base, names):
    names = base.clean_name_list(names)

    if "lesion" not in names:
        names = ["lesion"] + names

    names = base.clean_name_list(names)

    if len(names) == 1:
        names.append("unknown")

    return names


def direct_totalseg_fallback(base, ex):
    """
    Validation fallback that does not use oracle anatomy.

    It builds conditioning only from TotalSeg coarse and detail fields.
    """
    coarse = base.get_totalseg_names(ex)
    detail = base.get_detail_names(ex)

    names = ["lesion"]

    for value in coarse + detail:
        value = base.clean_name(value)

        if not value or value in {
            "background",
            "unknown",
            "lesion",
        }:
            continue

        # Convert detailed TotalSeg labels into readable tokens.
        parts = value.replace("_", " ").split()

        for part in parts:
            if part not in names:
                names.append(part)

    if len(names) == 1:
        names.append("unknown")

    return clean_names(base, names)


def safe_json_fallback(base, ex):
    """
    Test-time safe fallback.

    Prefer the existing safe-mapped rough_anatomy_names.
    """
    names = ex.get("rough_anatomy_names", [])

    if isinstance(names, str):
        names = [names]

    if not names:
        name = ex.get("rough_anatomy_name")
        if name:
            names = str(name).split()

    if not names:
        names = direct_totalseg_fallback(base, ex)

    return clean_names(base, names)


def token_f1(base, true_names, pred_names):
    return base.token_f1(
        clean_names(base, true_names),
        clean_names(base, pred_names),
    )


def calculate_rag_confidence(prediction):
    """
    Combine:
      1. weighted target vote share
      2. closest-neighbor similarity
      3. gap between first and second neighbor

    This is still heuristic, so the threshold is selected on validation.
    """
    vote_share = float(prediction.get("confidence", 0.0))

    similarities = [
        float(x.get("similarity", 0.0))
        for x in prediction.get("retrieved", [])
    ]

    similarities = sorted(similarities, reverse=True)

    top_similarity = similarities[0] if similarities else 0.0
    second_similarity = (
        similarities[1] if len(similarities) >= 2 else 0.0
    )

    similarity_margin = max(
        0.0,
        top_similarity - second_similarity,
    )

    combined = (
        0.60 * vote_share
        + 0.30 * top_similarity
        + 0.10 * similarity_margin
    )

    combined = max(0.0, min(1.0, combined))

    return {
        "combined": combined,
        "vote_share": vote_share,
        "top_similarity": top_similarity,
        "similarity_margin": similarity_margin,
    }


def evaluate_gated_validation(
    base,
    val_samples,
    rag_predictions,
    confidence_threshold,
):
    exact = 0
    f1_values = []
    no_overlap = 0

    rag_used = 0
    fallback_used = 0

    rag_correct = 0
    fallback_correct = 0

    mismatches = Counter()

    rows = []

    for ex, rag_pred in zip(
        val_samples,
        rag_predictions,
    ):
        true_names = base.get_target_names(ex)
        rag_names = clean_names(
            base,
            rag_pred["predicted_names"],
        )

        fallback_names = direct_totalseg_fallback(base, ex)

        confidence = calculate_rag_confidence(rag_pred)
        use_rag = (
            confidence["combined"]
            >= confidence_threshold
        )

        if use_rag:
            final_names = rag_names
            source = "rag"
            rag_used += 1
        else:
            final_names = fallback_names
            source = "direct_totalseg_fallback"
            fallback_used += 1

        same = true_names == final_names
        exact += int(same)

        if source == "rag":
            rag_correct += int(same)
        else:
            fallback_correct += int(same)

        f1 = token_f1(
            base,
            true_names,
            final_names,
        )
        f1_values.append(f1)

        true_set = set(true_names) - {"lesion"}
        pred_set = set(final_names) - {"lesion"}

        if (
            true_set
            and pred_set
            and not (true_set & pred_set)
        ):
            no_overlap += 1

        if not same:
            mismatches[(
                " ".join(true_names),
                " ".join(final_names),
                source,
            )] += 1

        rows.append({
            "id": str(ex.get("id")),
            "true_names": true_names,
            "rag_names": rag_names,
            "fallback_names": fallback_names,
            "final_names": final_names,
            "source": source,
            "combined_confidence":
                confidence["combined"],
            "vote_share":
                confidence["vote_share"],
            "top_similarity":
                confidence["top_similarity"],
            "similarity_margin":
                confidence["similarity_margin"],
            "exact": int(same),
            "token_f1": f1,
        })

    n = len(val_samples)

    return {
        "n": n,
        "confidence_threshold":
            confidence_threshold,
        "exact": exact,
        "exact_accuracy":
            exact / n if n else 0.0,
        "mean_token_f1":
            float(np.mean(f1_values))
            if f1_values else 0.0,
        "no_overlap": no_overlap,
        "no_overlap_rate":
            no_overlap / n if n else 0.0,
        "rag_used": rag_used,
        "rag_coverage":
            rag_used / n if n else 0.0,
        "fallback_used": fallback_used,
        "rag_exact_accuracy":
            rag_correct / rag_used
            if rag_used else 0.0,
        "fallback_exact_accuracy":
            fallback_correct / fallback_used
            if fallback_used else 0.0,
        "top_mismatches": [
            {
                "true": true,
                "pred": pred,
                "source": source,
                "count": count,
            }
            for (
                true,
                pred,
                source,
            ), count in mismatches.most_common(30)
        ],
        "rows": rows,
    }


def apply_gated_test_predictions(
    base,
    raw_test_samples,
    safe_test_by_id,
    rag_predictions,
    confidence_threshold,
):
    output = []

    counts = Counter()

    for raw_ex, rag_pred in zip(
        raw_test_samples,
        rag_predictions,
    ):
        sid = str(raw_ex.get("id"))

        # Start with predicted-box sample so bboxes and TotalSeg
        # provenance remain realistic.
        x = copy.deepcopy(raw_ex)

        rag_names = clean_names(
            base,
            rag_pred["predicted_names"],
        )

        safe_ex = safe_test_by_id.get(sid)

        if safe_ex is not None:
            fallback_names = safe_json_fallback(
                base,
                safe_ex,
            )
            fallback_source = "safe_mapped_fallback"
        else:
            fallback_names = direct_totalseg_fallback(
                base,
                raw_ex,
            )
            fallback_source = "direct_totalseg_fallback"

        confidence = calculate_rag_confidence(rag_pred)

        use_rag = (
            confidence["combined"]
            >= confidence_threshold
        )

        if use_rag:
            final_names = rag_names
            final_source = "rag"
        else:
            final_names = fallback_names
            final_source = fallback_source

        final_names = clean_names(
            base,
            final_names,
        )

        anatomy_text = " ".join(final_names)

        # These are consumed by modules/datasets.py.
        x["rough_anatomy_names"] = final_names
        x["rough_anatomy_name"] = anatomy_text
        x["anatomy_text"] = anatomy_text

        # Provenance/debug fields.
        x["anatomy_source"] = "gated_train_only_rag"
        x["conditioning_source"] = final_source

        x["rag_predicted_anatomy_names"] = rag_names
        x["rag_fallback_anatomy_names"] = fallback_names
        x["rag_final_anatomy_names"] = final_names

        x["rag_vote_share"] = confidence["vote_share"]
        x["rag_top_similarity"] = (
            confidence["top_similarity"]
        )
        x["rag_similarity_margin"] = (
            confidence["similarity_margin"]
        )
        x["rag_combined_confidence"] = (
            confidence["combined"]
        )
        x["rag_confidence_threshold"] = (
            confidence_threshold
        )
        x["rag_used"] = use_rag

        x["rag_retrieved_ids"] = [
            r["id"]
            for r in rag_pred["retrieved"]
        ]
        x["rag_retrieved_similarities"] = [
            r["similarity"]
            for r in rag_pred["retrieved"]
        ]
        x["rag_retrieved_targets"] = [
            r["target_names"]
            for r in rag_pred["retrieved"]
        ]

        counts[final_source] += 1
        counts["total"] += 1

        output.append(x)

    return output, dict(counts)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_rag_script",
        default=(
            "fresh_pipeline/"
            "21_train_validate_apply_anatomy_rag.py"
        ),
    )

    parser.add_argument(
        "--merged_json",
        required=True,
    )

    parser.add_argument(
        "--predbbox_json",
        required=True,
    )

    parser.add_argument(
        "--safe_json",
        required=True,
        help=(
            "R2Gen-ready safe-mapped JSON used only "
            "as low-confidence test fallback."
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
        "--k_values",
        nargs="+",
        type=int,
        default=[3, 5, 7, 10, 15],
    )

    parser.add_argument(
        "--bbox_weights",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0],
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

    base = load_base_rag_module(
        args.base_rag_script
    )

    merged = json.load(open(args.merged_json))
    predbbox = json.load(open(args.predbbox_json))
    safe = json.load(open(args.safe_json))

    train_samples = merged["train"]
    val_samples = merged["val"]
    raw_test_samples = predbbox["test"]

    safe_test_by_id = {
        str(x.get("id")): x
        for x in safe["test"]
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("train index:", len(train_samples))
    print("validation:", len(val_samples))
    print("predicted-box test:", len(raw_test_samples))
    print("safe test fallback:", len(safe_test_by_id))

    grid_results = []
    best = None

    for bbox_weight in args.bbox_weights:
        print("\n" + "=" * 100)
        print("bbox_weight:", bbox_weight)

        rag = base.AnatomyRAG(
            bbox_weight=bbox_weight
        ).fit(train_samples)

        for k in args.k_values:
            rag_val_predictions = rag.predict(
                val_samples,
                k=k,
            )

            for confidence_threshold in (
                args.confidence_thresholds
            ):
                result = evaluate_gated_validation(
                    base=base,
                    val_samples=val_samples,
                    rag_predictions=rag_val_predictions,
                    confidence_threshold=
                        confidence_threshold,
                )

                row = {
                    key: value
                    for key, value in result.items()
                    if key not in {
                        "top_mismatches",
                        "rows",
                    }
                }

                row["bbox_weight"] = bbox_weight
                row["k"] = k

                grid_results.append(row)

                print(
                    f"k={k:2d} "
                    f"thr={confidence_threshold:.2f} "
                    f"rag_cov={result['rag_coverage']:.4f} "
                    f"exact={result['exact_accuracy']:.4f} "
                    f"token_f1={result['mean_token_f1']:.4f} "
                    f"no_overlap={result['no_overlap_rate']:.4f}"
                )

                # Primary objective: token F1.
                # Tie breaks: exact accuracy, lower no-overlap,
                # then higher RAG coverage.
                score = (
                    result["mean_token_f1"],
                    result["exact_accuracy"],
                    -result["no_overlap_rate"],
                    result["rag_coverage"],
                )

                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "bbox_weight": bbox_weight,
                        "k": k,
                        "confidence_threshold":
                            confidence_threshold,
                        "metrics": {
                            key: value
                            for key, value in result.items()
                            if key != "rows"
                        },
                        "validation_rows":
                            result["rows"],
                    }

    print("\n" + "=" * 100)
    print("BEST GATED VALIDATION CONFIGURATION")
    print("bbox_weight:", best["bbox_weight"])
    print("k:", best["k"])
    print(
        "confidence_threshold:",
        best["confidence_threshold"],
    )
    print(
        "rag_coverage:",
        best["metrics"]["rag_coverage"],
    )
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

    with open(
        out_dir / "gated_validation_grid_results.json",
        "w",
    ) as f:
        json.dump(grid_results, f, indent=2)

    best_to_save = {
        key: value
        for key, value in best.items()
        if key != "validation_rows"
    }

    with open(
        out_dir / "best_gated_validation_config.json",
        "w",
    ) as f:
        json.dump(best_to_save, f, indent=2)

    with open(
        out_dir / "best_validation_per_sample.json",
        "w",
    ) as f:
        json.dump(
            best["validation_rows"],
            f,
            indent=2,
        )

    # Fit final train-only index.
    rag = base.AnatomyRAG(
        bbox_weight=best["bbox_weight"]
    ).fit(train_samples)

    rag_test_predictions = rag.predict(
        raw_test_samples,
        k=best["k"],
    )

    final_test, source_counts = (
        apply_gated_test_predictions(
            base=base,
            raw_test_samples=raw_test_samples,
            safe_test_by_id=safe_test_by_id,
            rag_predictions=rag_test_predictions,
            confidence_threshold=
                best["confidence_threshold"],
        )
    )

    output_json = {
        "train": copy.deepcopy(merged["train"]),
        "val": copy.deepcopy(merged["val"]),
        "test": final_test,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out_json, "w") as f:
        json.dump(output_json, f, indent=2)

    with open(
        out_dir / "test_gated_rag_predictions.json",
        "w",
    ) as f:
        json.dump(
            rag_test_predictions,
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

    print("\nSaved:", out_json)
    print("Test source counts:", source_counts)

    print("\nFirst test example:")
    first = final_test[0]

    for key in [
        "id",
        "bbox_source",
        "totalseg_anatomy_names",
        "totalseg_detail_names",
        "rag_predicted_anatomy_names",
        "rag_fallback_anatomy_names",
        "rag_combined_confidence",
        "rag_confidence_threshold",
        "rag_used",
        "conditioning_source",
        "rough_anatomy_names",
        "anatomy_text",
    ]:
        print(key, ":", first.get(key))


if __name__ == "__main__":
    main()
