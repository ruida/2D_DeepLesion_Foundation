#!/usr/bin/env python3

import argparse
import json
import math
from collections import Counter
from pathlib import Path


ANATOMY_KEYWORDS = {
    "lung": [
        "lung", "pulmonary", "upper lobe",
        "middle lobe", "lower lobe", "pleural",
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
        "abdomen", "abdominal",
        "retroperitoneal", "retroperitoneum",
        "pancreas", "pancreatic",
        "spleen", "splenic",
        "stomach", "gastric",
        "bowel", "colon",
        "intestinal", "mesenteric",
    ],
    "pelvis": [
        "pelvis", "pelvic",
        "uterus", "uterine",
        "ovary", "ovarian",
        "prostate", "bladder",
        "iliac", "inguinal",
    ],
    "chest": [
        "chest", "thoracic",
        "mediastinal", "mediastinum",
        "hilar", "axillary",
    ],
    "brain_head_neck": [
        "brain", "intracranial",
        "head", "neck",
        "thyroid", "parotid",
    ],
    "spine": [
        "spine", "spinal",
        "vertebral", "vertebra",
    ],
    "bone": [
        "bone", "osseous",
        "rib", "femur",
        "humerus", "sacrum",
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


def extract_report_anatomy(report):
    text = clean(report)
    found = set()

    for label, keywords in ANATOMY_KEYWORDS.items():
        if any(clean(keyword) in text for keyword in keywords):
            found.add(label)

    return found


def make_row(sample):
    oracle_anatomy = canonical(
        sample.get("oracle_simple_anatomy", "unknown")
    )

    report_anatomy = extract_report_anatomy(
        sample.get("report", "")
    )

    details = tuple(
        sorted(
            {
                clean(value)
                for value in sample.get(
                    "raw_totalseg_details",
                    [],
                )
                if clean(value)
            }
        )
    )

    report_support = (
        oracle_anatomy in report_anatomy
    )

    return {
        "id": str(sample.get("id")),

        "raw_anatomy": canonical(
            sample.get(
                "raw_simple_anatomy",
                "unknown",
            )
        ),

        "raw_primary": canonical(
            sample.get(
                "raw_primary_lesion_type",
                "other_lesion",
            )
        ),

        "raw_attribute": canonical(
            sample.get(
                "raw_lesion_attribute",
                "none",
            )
        ),

        "details": details,

        "oracle_anatomy": oracle_anatomy,

        "oracle_primary": canonical(
            sample.get(
                "oracle_primary_lesion_type",
                "other_lesion",
            )
        ),

        "oracle_attribute": canonical(
            sample.get(
                "oracle_lesion_attribute",
                "none",
            )
        ),

        "report_anatomy_concepts": sorted(
            report_anatomy
        ),

        "report_supports_oracle_anatomy": (
            report_support
        ),

        # Report-supported references receive higher weight.
        "report_reliability": (
            1.0 if report_support else 0.40
        ),
    }


def build_detail_idf(rows):
    frequencies = Counter()

    for row in rows:
        for detail in set(row["details"]):
            frequencies[detail] += 1

    n = len(rows)

    return {
        detail: math.log(
            (n + 1) / (count + 1)
        ) + 1.0
        for detail, count in frequencies.items()
    }


def weighted_jaccard(left, right, idf):
    left = set(left)
    right = set(right)

    if not left and not right:
        return 1.0

    if not left or not right:
        return 0.0

    union = left | right
    intersection = left & right

    union_weight = sum(
        idf.get(item, 1.0)
        for item in union
    )

    intersection_weight = sum(
        idf.get(item, 1.0)
        for item in intersection
    )

    if union_weight == 0:
        return 0.0

    return intersection_weight / union_weight


def similarity(query, reference, idf):
    return (
        0.30 * float(
            query["raw_anatomy"]
            == reference["raw_anatomy"]
        )
        + 0.20 * float(
            query["raw_primary"]
            == reference["raw_primary"]
        )
        + 0.10 * float(
            query["raw_attribute"]
            == reference["raw_attribute"]
        )
        + 0.40 * weighted_jaccard(
            query["details"],
            reference["details"],
            idf,
        )
    )


def precompute_neighbors(
    queries,
    references,
    idf,
    max_k,
):
    cache = []

    for index, query in enumerate(
        queries,
        1,
    ):
        scored = []

        for reference in references:
            if query["id"] == reference["id"]:
                continue

            score = similarity(
                query,
                reference,
                idf,
            )

            scored.append(
                (reference, score)
            )

        scored.sort(
            key=lambda item: (
                -item[1],
                item[0]["id"],
            )
        )

        cache.append(
            scored[:max_k]
        )

        if (
            index == 1
            or index % 100 == 0
            or index == len(queries)
        ):
            print(
                f"[cache] {index}/{len(queries)}",
                flush=True,
            )

    return cache


def weighted_vote(
    neighbors,
    k,
    class_frequencies,
    report_strength,
    class_balance_power,
):
    votes = Counter()
    evidence = []

    for reference, similarity_score in neighbors[:k]:
        target = reference["oracle_anatomy"]

        frequency = max(
            class_frequencies[target],
            1,
        )

        class_balance = (
            frequency ** (-class_balance_power)
        )

        reliability = (
            (1.0 - report_strength)
            + report_strength
            * reference["report_reliability"]
        )

        weight = (
            max(similarity_score, 1e-6)
            * reliability
            * class_balance
        )

        votes[target] += weight

        evidence.append({
            "id": reference["id"],
            "similarity": round(
                similarity_score,
                6,
            ),
            "report_reliability": (
                reference["report_reliability"]
            ),
            "vote_weight": weight,
            "target_anatomy": target,
        })

    if not votes:
        return (
            None,
            0.0,
            0.0,
            evidence,
        )

    ranked = votes.most_common()

    prediction, top_weight = ranked[0]

    second_weight = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    total_weight = sum(votes.values())

    purity = top_weight / total_weight

    margin = (
        top_weight - second_weight
    ) / total_weight

    return (
        prediction,
        purity,
        margin,
        evidence,
    )


def predict_one(
    query,
    neighbors,
    config,
    class_frequencies,
):
    (
        candidate,
        purity,
        margin,
        evidence,
    ) = weighted_vote(
        neighbors=neighbors,
        k=config["k"],
        class_frequencies=class_frequencies,
        report_strength=config[
            "report_strength"
        ],
        class_balance_power=config[
            "class_balance_power"
        ],
    )

    top_similarity = (
        neighbors[0][1]
        if neighbors
        else 0.0
    )

    accepted = (
        candidate is not None
        and top_similarity
        >= config["min_similarity"]
        and purity
        >= config["min_purity"]
        and margin
        >= config["min_margin"]
    )

    if accepted:
        mapped_anatomy = candidate
        method = "anatomy_agent_accept"
    else:
        mapped_anatomy = query[
            "raw_anatomy"
        ]
        method = "anatomy_agent_reject"

    return {
        "mapped_anatomy": mapped_anatomy,

        # Preserve raw dual-head outputs.
        "mapped_primary": query[
            "raw_primary"
        ],
        "mapped_attribute": query[
            "raw_attribute"
        ],

        "method": method,
        "purity": purity,
        "margin": margin,
        "top_similarity": top_similarity,
        "neighbors": evidence,
    }


def calculate_metrics(
    predictions,
    queries,
):
    n = len(queries)

    return {
        "n": n,

        "anatomy_accuracy": sum(
            prediction["mapped_anatomy"]
            == query["oracle_anatomy"]
            for prediction, query
            in zip(predictions, queries)
        ) / n,

        "primary_accuracy": sum(
            prediction["mapped_primary"]
            == query["oracle_primary"]
            for prediction, query
            in zip(predictions, queries)
        ) / n,

        "attribute_accuracy": sum(
            prediction["mapped_attribute"]
            == query["oracle_attribute"]
            for prediction, query
            in zip(predictions, queries)
        ) / n,

        "anatomy_primary_joint": sum(
            prediction["mapped_anatomy"]
            == query["oracle_anatomy"]
            and prediction["mapped_primary"]
            == query["oracle_primary"]
            for prediction, query
            in zip(predictions, queries)
        ) / n,

        "full_joint": sum(
            prediction["mapped_anatomy"]
            == query["oracle_anatomy"]
            and prediction["mapped_primary"]
            == query["oracle_primary"]
            and prediction["mapped_attribute"]
            == query["oracle_attribute"]
            for prediction, query
            in zip(predictions, queries)
        ) / n,
    }


def objective(metrics):
    return (
        0.65
        * metrics["anatomy_primary_joint"]
        + 0.35
        * metrics["anatomy_accuracy"]
    )


def evaluate_config(
    queries,
    neighbor_cache,
    config,
    class_frequencies,
):
    predictions = [
        predict_one(
            query=query,
            neighbors=neighbors,
            config=config,
            class_frequencies=class_frequencies,
        )
        for query, neighbors
        in zip(queries, neighbor_cache)
    ]

    return calculate_metrics(
        predictions,
        queries,
    )


def conditioning_tokens(
    anatomy,
    primary,
    attribute,
):
    output = []

    output.extend(
        clean(anatomy).split()
    )

    output.extend(
        clean(primary).split()
    )

    if canonical(attribute) != "none":
        output.extend(
            clean(attribute).split()
        )

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

    with open(args.input_json) as file:
        data = json.load(file)

    train = [
        make_row(sample)
        for sample in data["train"]
    ]

    validation = [
        make_row(sample)
        for sample in data["val"]
    ]

    test = [
        make_row(sample)
        for sample in data["test"]
    ]

    print(
        f"train={len(train)} "
        f"val={len(validation)} "
        f"test={len(test)}",
        flush=True,
    )

    # -------------------------------------------------------------
    # Stage 1: train -> validation tuning
    # -------------------------------------------------------------

    train_idf = build_detail_idf(
        train
    )

    train_class_frequencies = Counter(
        row["oracle_anatomy"]
        for row in train
    )

    print(
        "Precomputing train -> validation neighbors...",
        flush=True,
    )

    validation_cache = precompute_neighbors(
        queries=validation,
        references=train,
        idf=train_idf,
        max_k=11,
    )

    configurations = []

    for k in [3, 5, 7, 9, 11]:
        for report_strength in [
            0.0,
            0.25,
            0.50,
            0.75,
        ]:
            for min_purity in [
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
            ]:
                for min_margin in [
                    0.0,
                    0.05,
                    0.10,
                    0.15,
                    0.20,
                ]:
                    for balance_power in [
                        0.0,
                        0.10,
                        0.25,
                    ]:
                        configurations.append({
                            "k": k,
                            "report_strength": (
                                report_strength
                            ),
                            "min_similarity": 0.30,
                            "min_purity": min_purity,
                            "min_margin": min_margin,
                            "class_balance_power": (
                                balance_power
                            ),
                        })

    print(
        f"Evaluating "
        f"{len(configurations)} configurations...",
        flush=True,
    )

    grid = []

    for index, config in enumerate(
        configurations,
        1,
    ):
        result = evaluate_config(
            queries=validation,
            neighbor_cache=validation_cache,
            config=config,
            class_frequencies=(
                train_class_frequencies
            ),
        )

        grid.append({
            "config": config,
            "metrics": result,
            "objective": objective(result),
        })

        if (
            index == 1
            or index % 100 == 0
            or index == len(configurations)
        ):
            print(
                f"[grid] {index}/"
                f"{len(configurations)}",
                flush=True,
            )

    grid.sort(
        key=lambda item: (
            -item["objective"],
            -item["metrics"][
                "anatomy_primary_joint"
            ],
            -item["metrics"][
                "anatomy_accuracy"
            ],
        )
    )

    best = grid[0]

    print("Best configuration:")

    print(
        json.dumps(
            best["config"],
            indent=2,
        ),
        flush=True,
    )

    print("Best validation metrics:")

    print(
        json.dumps(
            best["metrics"],
            indent=2,
        ),
        flush=True,
    )

    # -------------------------------------------------------------
    # Stage 2: train+validation -> test
    # -------------------------------------------------------------

    references = train + validation

    reference_idf = build_detail_idf(
        references
    )

    reference_class_frequencies = Counter(
        row["oracle_anatomy"]
        for row in references
    )

    print(
        "Precomputing train+validation -> test neighbors...",
        flush=True,
    )

    test_cache = precompute_neighbors(
        queries=test,
        references=references,
        idf=reference_idf,
        max_k=best["config"]["k"],
    )

    test_predictions = [
        predict_one(
            query=query,
            neighbors=neighbors,
            config=best["config"],
            class_frequencies=(
                reference_class_frequencies
            ),
        )
        for query, neighbors
        in zip(test, test_cache)
    ]

    test_metrics = calculate_metrics(
        test_predictions,
        test,
    )

    output = {
        "train": data["train"],
        "val": data["val"],
        "test": [],
    }

    method_counts = Counter()

    for original, prediction in zip(
        data["test"],
        test_predictions,
    ):
        row = dict(original)

        row[
            "anatomy_agent_mapped_anatomy"
        ] = prediction["mapped_anatomy"]

        row[
            "anatomy_agent_primary_lesion_type"
        ] = prediction["mapped_primary"]

        row[
            "anatomy_agent_lesion_attribute"
        ] = prediction["mapped_attribute"]

        row[
            "anatomy_agent_method"
        ] = prediction["method"]

        row[
            "anatomy_agent_purity"
        ] = prediction["purity"]

        row[
            "anatomy_agent_margin"
        ] = prediction["margin"]

        row[
            "anatomy_agent_top_similarity"
        ] = prediction["top_similarity"]

        row[
            "anatomy_agent_neighbors"
        ] = prediction["neighbors"]

        tokens = conditioning_tokens(
            prediction["mapped_anatomy"],
            prediction["mapped_primary"],
            prediction["mapped_attribute"],
        )

        row[
            "rough_anatomy_names"
        ] = tokens

        row[
            "rough_anatomy_name"
        ] = " ".join(tokens)

        row[
            "anatomy_text"
        ] = " ".join(tokens)

        row[
            "anatomy_source"
        ] = "anatomy_only_agent_rag"

        output["test"].append(row)

        method_counts[
            prediction["method"]
        ] += 1

    summary = {
        "protocol": {
            "tuning_reference": "train",
            "tuning_queries": "validation",
            "final_reference": "train+validation",
            "final_queries": "test",
            "test_report_used_for_prediction": False,
        },

        "source_features": [
            "raw_simple_anatomy",
            "raw_primary_lesion_type",
            "raw_lesion_attribute",
            "raw_totalseg_details",
        ],

        "mapped_target": (
            "oracle_simple_anatomy"
        ),

        "preserved_fields": [
            "raw_primary_lesion_type",
            "raw_lesion_attribute",
        ],

        "best_config": (
            best["config"]
        ),

        "best_validation_metrics": (
            best["metrics"]
        ),

        "test_metrics": (
            test_metrics
        ),

        "method_counts": dict(
            method_counts
        ),

        "reference_size": len(
            references
        ),

        "test_size": len(test),
    }

    output_path = Path(
        args.out_json
    )

    summary_path = Path(
        args.summary_json
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(summary_path, "w") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "Saved:",
        output_path,
    )

    print(
        "Saved:",
        summary_path,
    )


if __name__ == "__main__":
    main()
