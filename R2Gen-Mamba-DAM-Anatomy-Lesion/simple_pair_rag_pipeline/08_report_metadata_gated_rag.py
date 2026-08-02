#!/usr/bin/env python3

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------
# Report concept lexicons
# ---------------------------------------------------------------------

ANATOMY_KEYWORDS = {
    "lung": [
        "lung", "pulmonary", "upper lobe", "middle lobe",
        "lower lobe", "pleural",
    ],
    "liver": [
        "liver", "hepatic",
    ],
    "kidney": [
        "kidney", "renal",
    ],
    "adrenal": [
        "adrenal",
    ],
    "abdomen": [
        "abdomen", "abdominal", "retroperitoneal",
        "retroperitoneum", "pancreas", "pancreatic",
        "spleen", "splenic", "stomach", "gastric",
        "bowel", "colon", "intestinal", "mesenteric",
    ],
    "pelvis": [
        "pelvis", "pelvic", "uterus", "uterine",
        "ovary", "ovarian", "prostate", "bladder",
        "iliac", "inguinal",
    ],
    "chest": [
        "chest", "thoracic", "mediastinal",
        "mediastinum", "hilar", "axillary",
    ],
    "brain_head_neck": [
        "brain", "intracranial", "head", "neck",
        "cervical", "thyroid", "parotid",
    ],
    "spine": [
        "spine", "spinal", "vertebral", "vertebra",
    ],
    "bone": [
        "bone", "osseous", "rib", "femur",
        "humerus", "sacrum",
    ],
}

PRIMARY_KEYWORDS = {
    "lymph_node": [
        "lymph node", "lymphadenopathy", "adenopathy",
        "nodal",
    ],
    "cystic": [
        "cyst", "cystic",
    ],
    "nodule": [
        "nodule", "nodular",
    ],
    "mass": [
        "mass", "tumor", "tumour",
    ],
    "consolidation": [
        "consolidation", "consolidative",
    ],
    "opacity": [
        "opacity", "opacification", "ground glass",
        "ground-glass",
    ],
    "other_lesion": [
        "lesion", "focus", "abnormality",
    ],
}

ATTRIBUTE_KEYWORDS = {
    "low_density": [
        "low density", "low-density", "hypodense",
        "hypoattenuating", "hypoattenuation",
    ],
    "enhancing": [
        "enhancing", "enhancement",
    ],
    "hyperdense": [
        "hyperdense", "hyperattenuating",
    ],
    "calcified": [
        "calcified", "calcification",
    ],
}


def clean(value):
    return " ".join(
        str(value or "")
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def canonical(value):
    return clean(value).replace(" ", "_")


def extract_concepts(report, lexicon):
    text = clean(report)
    found = []

    for label, keywords in lexicon.items():
        for keyword in keywords:
            if clean(keyword) in text:
                found.append(label)
                break

    return sorted(set(found))


def report_metadata(sample):
    report = sample.get("report", "")

    anatomy = extract_concepts(report, ANATOMY_KEYWORDS)
    primary = extract_concepts(report, PRIMARY_KEYWORDS)
    attributes = extract_concepts(report, ATTRIBUTE_KEYWORDS)

    oracle_a = canonical(
        sample.get("oracle_simple_anatomy", "unknown")
    )
    oracle_p = canonical(
        sample.get(
            "oracle_primary_lesion_type",
            "other_lesion",
        )
    )
    oracle_t = canonical(
        sample.get("oracle_lesion_attribute", "none")
    )

    anatomy_supported = oracle_a in anatomy
    primary_supported = oracle_p in primary

    if oracle_t == "none":
        attribute_supported = len(attributes) == 0
    else:
        attribute_supported = oracle_t in attributes

    # Anatomy and lesion type receive most of the reliability weight.
    consistency = (
        0.45 * float(anatomy_supported)
        + 0.45 * float(primary_supported)
        + 0.10 * float(attribute_supported)
    )

    return {
        "report_anatomy_concepts": anatomy,
        "report_primary_concepts": primary,
        "report_attribute_concepts": attributes,
        "report_supports_oracle_anatomy": anatomy_supported,
        "report_supports_oracle_primary": primary_supported,
        "report_supports_oracle_attribute": attribute_supported,
        "report_target_consistency": consistency,
    }


def make_row(sample):
    metadata = report_metadata(sample)

    details = sorted(
        {
            clean(x)
            for x in sample.get("raw_totalseg_details", [])
            if clean(x)
        }
    )

    return {
        "id": str(sample.get("id")),
        "raw_a": canonical(
            sample.get("raw_simple_anatomy", "unknown")
        ),
        "raw_p": canonical(
            sample.get(
                "raw_primary_lesion_type",
                "other_lesion",
            )
        ),
        "raw_t": canonical(
            sample.get("raw_lesion_attribute", "none")
        ),
        "details": tuple(details),
        "oracle_a": canonical(
            sample.get("oracle_simple_anatomy", "unknown")
        ),
        "oracle_p": canonical(
            sample.get(
                "oracle_primary_lesion_type",
                "other_lesion",
            )
        ),
        "oracle_t": canonical(
            sample.get("oracle_lesion_attribute", "none")
        ),
        "report_consistency": metadata[
            "report_target_consistency"
        ],
        "metadata": metadata,
    }


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def raw_similarity(query, reference):
    return (
        0.30 * float(query["raw_a"] == reference["raw_a"])
        + 0.25 * float(query["raw_p"] == reference["raw_p"])
        + 0.10 * float(query["raw_t"] == reference["raw_t"])
        + 0.35 * jaccard(
            query["details"],
            reference["details"],
        )
    )


def precompute_neighbors(
    queries,
    references,
    max_k,
):
    cache = []

    for i, query in enumerate(queries, 1):
        scored = []

        for reference in references:
            if query["id"] == reference["id"]:
                continue

            sim = raw_similarity(query, reference)
            scored.append((reference, sim))

        scored.sort(
            key=lambda x: (-x[1], x[0]["id"])
        )

        cache.append(scored[:max_k])

        if i == 1 or i % 100 == 0 or i == len(queries):
            print(
                f"[cache] {i}/{len(queries)}",
                flush=True,
            )

    return cache


def weighted_joint_vote(
    neighbors,
    k,
    report_strength,
):
    votes = Counter()
    used = []

    for reference, similarity in neighbors[:k]:
        consistency = reference["report_consistency"]

        reliability = (
            (1.0 - report_strength)
            + report_strength * consistency
        )

        weight = max(similarity, 1e-6) * reliability

        target = (
            reference["oracle_a"],
            reference["oracle_p"],
        )

        votes[target] += weight
        used.append(
            {
                "id": reference["id"],
                "similarity": similarity,
                "report_consistency": consistency,
                "vote_weight": weight,
                "target_anatomy": reference["oracle_a"],
                "target_primary": reference["oracle_p"],
            }
        )

    if not votes:
        return None, 0.0, 0.0, used

    ranked = votes.most_common()
    target, top_weight = ranked[0]

    second_weight = (
        ranked[1][1] if len(ranked) > 1 else 0.0
    )

    total_weight = sum(votes.values())

    purity = top_weight / total_weight
    margin = (top_weight - second_weight) / total_weight

    return target, purity, margin, used


def predict(
    query,
    neighbors,
    config,
):
    target, purity, margin, used = weighted_joint_vote(
        neighbors,
        config["k"],
        config["report_strength"],
    )

    top_similarity = (
        neighbors[0][1] if neighbors else 0.0
    )

    accepted = (
        target is not None
        and top_similarity >= config["min_similarity"]
        and purity >= config["min_purity"]
        and margin >= config["min_margin"]
    )

    if accepted:
        pred_a, pred_p = target
        method = "report_metadata_joint_rag"
    else:
        pred_a = query["raw_a"]
        pred_p = query["raw_p"]
        method = "raw_fallback"

    # Keep the raw dual-head attribute in this experiment.
    pred_t = query["raw_t"]

    return {
        "pred_a": pred_a,
        "pred_p": pred_p,
        "pred_t": pred_t,
        "method": method,
        "purity": purity,
        "margin": margin,
        "top_similarity": top_similarity,
        "neighbors": used,
    }


def metrics(rows):
    n = len(rows)

    anatomy = sum(
        x["pred_a"] == x["oracle_a"]
        for x in rows
    ) / n

    primary = sum(
        x["pred_p"] == x["oracle_p"]
        for x in rows
    ) / n

    pair = sum(
        x["pred_a"] == x["oracle_a"]
        and x["pred_p"] == x["oracle_p"]
        for x in rows
    ) / n

    full = sum(
        x["pred_a"] == x["oracle_a"]
        and x["pred_p"] == x["oracle_p"]
        and x["pred_t"] == x["oracle_t"]
        for x in rows
    ) / n

    return {
        "n": n,
        "anatomy_accuracy": anatomy,
        "primary_accuracy": primary,
        "anatomy_primary_joint": pair,
        "full_joint_with_raw_attribute": full,
    }


def objective(result):
    return (
        0.55 * result["anatomy_primary_joint"]
        + 0.25 * result["anatomy_accuracy"]
        + 0.20 * result["primary_accuracy"]
    )


def evaluate(
    queries,
    cache,
    config,
):
    rows = []

    for query, neighbors in zip(queries, cache):
        prediction = predict(
            query,
            neighbors,
            config,
        )

        rows.append(
            {
                **prediction,
                "oracle_a": query["oracle_a"],
                "oracle_p": query["oracle_p"],
                "oracle_t": query["oracle_t"],
            }
        )

    return metrics(rows)


def tokens(anatomy, primary, attribute):
    output = []

    output.extend(clean(anatomy).split())
    output.extend(clean(primary).split())

    if canonical(attribute) != "none":
        output.extend(clean(attribute).split())

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
    )
    parser.add_argument(
        "--out_json",
        required=True,
    )
    parser.add_argument(
        "--summary_json",
        required=True,
    )

    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    train = [make_row(x) for x in data["train"]]
    val = [make_row(x) for x in data["val"]]
    test = [make_row(x) for x in data["test"]]

    print(
        f"train={len(train)} "
        f"val={len(val)} "
        f"test={len(test)}",
        flush=True,
    )

    # -------------------------------------------------------------
    # Stage 1: tune using train -> validation
    # -------------------------------------------------------------

    print(
        "Precomputing train -> validation neighbors...",
        flush=True,
    )

    val_cache = precompute_neighbors(
        val,
        train,
        max_k=9,
    )

    configs = []

    for k in [3, 5, 7, 9]:
        for report_strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for min_similarity in [0.30, 0.40, 0.50]:
                for min_purity in [0.45, 0.55, 0.65, 0.75]:
                    for min_margin in [0.0, 0.10, 0.20]:
                        configs.append(
                            {
                                "k": k,
                                "report_strength": report_strength,
                                "min_similarity": min_similarity,
                                "min_purity": min_purity,
                                "min_margin": min_margin,
                            }
                        )

    print(
        f"Evaluating {len(configs)} configurations...",
        flush=True,
    )

    grid = []

    for i, config in enumerate(configs, 1):
        result = evaluate(
            val,
            val_cache,
            config,
        )

        grid.append(
            {
                "config": config,
                "metrics": result,
                "objective": objective(result),
            }
        )

        if i == 1 or i % 100 == 0 or i == len(configs):
            print(
                f"[grid] {i}/{len(configs)}",
                flush=True,
            )

    grid.sort(
        key=lambda x: (
            -x["objective"],
            -x["metrics"]["anatomy_primary_joint"],
            -x["metrics"]["anatomy_accuracy"],
        )
    )

    best = grid[0]

    print("Best configuration:")
    print(
        json.dumps(best["config"], indent=2),
        flush=True,
    )

    print("Best validation metrics:")
    print(
        json.dumps(best["metrics"], indent=2),
        flush=True,
    )

    # -------------------------------------------------------------
    # Stage 2: train+validation -> test
    # -------------------------------------------------------------

    references = train + val

    print(
        "Precomputing train+validation -> test neighbors...",
        flush=True,
    )

    test_cache = precompute_neighbors(
        test,
        references,
        max_k=best["config"]["k"],
    )

    output = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Preserve data and append report metadata to train/validation.
    for split_name, original_rows, converted_rows in [
        ("train", data["train"], train),
        ("val", data["val"], val),
    ]:
        for original, converted in zip(
            original_rows,
            converted_rows,
        ):
            row = dict(original)
            row.update(converted["metadata"])
            output[split_name].append(row)

    test_eval_rows = []
    methods = Counter()

    for i, (original, query, neighbors) in enumerate(
        zip(data["test"], test, test_cache),
        1,
    ):
        prediction = predict(
            query,
            neighbors,
            best["config"],
        )

        row = dict(original)

        row[
            "report_metadata_rag_mapped_anatomy"
        ] = prediction["pred_a"]

        row[
            "report_metadata_rag_mapped_primary_lesion_type"
        ] = prediction["pred_p"]

        row[
            "report_metadata_rag_lesion_attribute"
        ] = prediction["pred_t"]

        row[
            "report_metadata_rag_method"
        ] = prediction["method"]

        row[
            "report_metadata_rag_purity"
        ] = prediction["purity"]

        row[
            "report_metadata_rag_margin"
        ] = prediction["margin"]

        row[
            "report_metadata_rag_top_similarity"
        ] = prediction["top_similarity"]

        row[
            "report_metadata_rag_neighbors"
        ] = prediction["neighbors"]

        conditioning_tokens = tokens(
            prediction["pred_a"],
            prediction["pred_p"],
            prediction["pred_t"],
        )

        row["rough_anatomy_names"] = conditioning_tokens
        row["rough_anatomy_name"] = " ".join(
            conditioning_tokens
        )
        row["anatomy_text"] = " ".join(
            conditioning_tokens
        )
        row[
            "anatomy_source"
        ] = "report_metadata_gated_rag"

        output["test"].append(row)

        methods[prediction["method"]] += 1

        test_eval_rows.append(
            {
                **prediction,
                "oracle_a": query["oracle_a"],
                "oracle_p": query["oracle_p"],
                "oracle_t": query["oracle_t"],
            }
        )

        if i == 1 or i % 100 == 0 or i == len(test):
            print(
                f"[test] {i}/{len(test)}",
                flush=True,
            )

    test_results = metrics(test_eval_rows)

    reference_consistency = {
        "train_mean": sum(
            x["report_consistency"] for x in train
        ) / len(train),
        "val_mean": sum(
            x["report_consistency"] for x in val
        ) / len(val),
    }

    summary = {
        "protocol": {
            "tuning_reference": "train",
            "tuning_queries": "validation",
            "final_reference": "train+validation",
            "final_queries": "test",
            "test_report_used_for_prediction": False,
            "report_usage": (
                "train/validation reference reliability metadata only"
            ),
        },
        "source_features": [
            "raw_simple_anatomy",
            "raw_primary_lesion_type",
            "raw_lesion_attribute",
            "raw_totalseg_details",
        ],
        "target_fields": [
            "oracle_simple_anatomy",
            "oracle_primary_lesion_type",
        ],
        "attribute_policy": (
            "preserve raw dual-head lesion attribute"
        ),
        "best_config": best["config"],
        "best_validation_metrics": best["metrics"],
        "test_metrics": test_results,
        "test_methods": dict(methods),
        "reference_report_consistency": reference_consistency,
        "reference_size": len(references),
        "test_size": len(test),
    }

    out_path = Path(args.out_json)
    summary_path = Path(args.summary_json)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(summary, indent=2))
    print("Saved:", out_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
