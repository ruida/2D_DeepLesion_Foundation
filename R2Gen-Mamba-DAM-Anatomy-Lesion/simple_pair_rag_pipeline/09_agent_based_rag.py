#!/usr/bin/env python3

import argparse
import json
import math
from collections import Counter
from pathlib import Path


ANATOMY_KEYWORDS = {
    "lung": ["lung", "pulmonary", "lobe", "pleural"],
    "liver": ["liver", "hepatic"],
    "kidney": ["kidney", "renal"],
    "adrenal": ["adrenal"],
    "abdomen": [
        "abdomen", "abdominal", "retroperitone",
        "pancrea", "spleen", "stomach",
        "bowel", "colon", "mesenter",
    ],
    "pelvis": [
        "pelvis", "pelvic", "uter", "ovar",
        "prostate", "bladder", "iliac", "inguinal",
    ],
    "chest": [
        "chest", "thoracic", "mediast",
        "hilar", "axillary",
    ],
    "brain_head_neck": [
        "brain", "intracranial", "head",
        "neck", "thyroid", "parotid",
    ],
    "spine": [
        "spine", "spinal", "vertebr",
    ],
    "bone": [
        "bone", "osseous", "rib",
        "femur", "humerus", "sacrum",
    ],
}

PRIMARY_KEYWORDS = {
    "lymph_node": [
        "lymph node", "lymphadenopathy",
        "adenopathy", "nodal",
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
        "opacity", "opacification",
        "ground glass", "ground-glass",
    ],
    "other_lesion": [
        "lesion", "focus", "abnormality",
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

    return {
        label
        for label, keywords in lexicon.items()
        if any(clean(keyword) in text for keyword in keywords)
    }


def convert_sample(sample):
    oracle_anatomy = canonical(
        sample.get("oracle_simple_anatomy", "unknown")
    )

    oracle_primary = canonical(
        sample.get(
            "oracle_primary_lesion_type",
            "other_lesion",
        )
    )

    report_anatomy = extract_concepts(
        sample.get("report", ""),
        ANATOMY_KEYWORDS,
    )

    report_primary = extract_concepts(
        sample.get("report", ""),
        PRIMARY_KEYWORDS,
    )

    details = tuple(
        sorted(
            {
                clean(value)
                for value in sample.get(
                    "raw_totalseg_details", []
                )
                if clean(value)
            }
        )
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
        "oracle_primary": oracle_primary,
        "oracle_attribute": canonical(
            sample.get(
                "oracle_lesion_attribute",
                "none",
            )
        ),

        # Report-derived reference reliability.
        "anatomy_reliability": (
            0.85
            if oracle_anatomy in report_anatomy
            else 0.35
        ),
        "primary_reliability": (
            0.85
            if oracle_primary in report_primary
            else 0.35
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

    union_weight = sum(
        idf.get(item, 1.0)
        for item in left | right
    )

    intersection_weight = sum(
        idf.get(item, 1.0)
        for item in left & right
    )

    if union_weight == 0:
        return 0.0

    return intersection_weight / union_weight


def anatomy_similarity(query, reference, idf):
    return (
        0.25 * (
            query["raw_anatomy"]
            == reference["raw_anatomy"]
        )
        + 0.20 * (
            query["raw_primary"]
            == reference["raw_primary"]
        )
        + 0.10 * (
            query["raw_attribute"]
            == reference["raw_attribute"]
        )
        + 0.45 * weighted_jaccard(
            query["details"],
            reference["details"],
            idf,
        )
    )


def primary_similarity(
    query,
    reference,
    mapped_anatomy,
    idf,
):
    return (
        0.40 * (
            query["raw_primary"]
            == reference["raw_primary"]
        )
        + 0.20 * (
            query["raw_attribute"]
            == reference["raw_attribute"]
        )
        + 0.20 * (
            mapped_anatomy
            == reference["oracle_anatomy"]
        )
        + 0.20 * weighted_jaccard(
            query["details"],
            reference["details"],
            idf,
        )
    )


def retrieve(
    query,
    references,
    score_function,
    max_k=9,
):
    scored = []

    for reference in references:
        if reference["id"] == query["id"]:
            continue

        score = score_function(
            query,
            reference,
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

    return scored[:max_k]


def adaptive_k(neighbors, target_field):
    if not neighbors:
        return 0

    if neighbors[0][1] >= 0.95:
        return 1

    if len(neighbors) >= 3:
        labels = [
            reference[target_field]
            for reference, _ in neighbors[:3]
        ]

        majority_count = Counter(
            labels
        ).most_common(1)[0][1]

        if majority_count >= 2:
            return 3

    return min(7, len(neighbors))


def weighted_vote(
    neighbors,
    target_field,
    reliability_field,
    class_frequencies,
    class_balance_power=0.25,
):
    votes = Counter()
    evidence = []

    for reference, similarity in neighbors:
        target = reference[target_field]

        frequency = max(
            class_frequencies[target],
            1,
        )

        class_balance = (
            frequency ** (-class_balance_power)
        )

        reliability = reference[
            reliability_field
        ]

        vote_weight = (
            max(similarity, 1e-6)
            * reliability
            * class_balance
        )

        votes[target] += vote_weight

        evidence.append({
            "id": reference["id"],
            "similarity": round(
                similarity,
                6,
            ),
            "report_reliability": reliability,
            "class_balance": class_balance,
            "vote_weight": vote_weight,
            "target": target,
        })

    if not votes:
        return None, 0.0, 0.0, evidence

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


def anatomy_judge(
    raw_value,
    candidate,
    purity,
    margin,
    top_similarity,
):
    accepted = (
        candidate is not None
        and top_similarity >= 0.30
        and purity >= 0.55
        and margin >= 0.10
    )

    if accepted:
        return (
            candidate,
            "anatomy_agent_accept",
        )

    return (
        raw_value,
        "anatomy_agent_reject",
    )


def primary_judge(
    raw_value,
    candidate,
    purity,
    margin,
    top_similarity,
):
    # Stricter gate because previous RAG only
    # slightly improved primary lesion accuracy.
    accepted = (
        candidate is not None
        and top_similarity >= 0.30
        and purity >= 0.75
        and margin >= 0.25
    )

    if accepted:
        return (
            candidate,
            "primary_agent_accept",
        )

    return (
        raw_value,
        "primary_agent_reject",
    )


def predict_queries(
    queries,
    references,
    idf,
):
    anatomy_frequency = Counter(
        row["oracle_anatomy"]
        for row in references
    )

    primary_frequency = Counter(
        row["oracle_primary"]
        for row in references
    )

    predictions = []

    for index, query in enumerate(
        queries,
        1,
    ):
        # ---------------------------------------------------------
        # Agent 1: anatomy
        # ---------------------------------------------------------

        anatomy_neighbors = retrieve(
            query,
            references,
            lambda q, r: anatomy_similarity(
                q,
                r,
                idf,
            ),
        )

        anatomy_k = adaptive_k(
            anatomy_neighbors,
            "oracle_anatomy",
        )

        (
            anatomy_candidate,
            anatomy_purity,
            anatomy_margin,
            anatomy_evidence,
        ) = weighted_vote(
            anatomy_neighbors[:anatomy_k],
            "oracle_anatomy",
            "anatomy_reliability",
            anatomy_frequency,
        )

        anatomy_top_similarity = (
            anatomy_neighbors[0][1]
            if anatomy_neighbors
            else 0.0
        )

        mapped_anatomy, anatomy_method = (
            anatomy_judge(
                query["raw_anatomy"],
                anatomy_candidate,
                anatomy_purity,
                anatomy_margin,
                anatomy_top_similarity,
            )
        )

        # ---------------------------------------------------------
        # Agent 2: primary lesion type
        # Uses the output of the anatomy agent.
        # ---------------------------------------------------------

        primary_neighbors = retrieve(
            query,
            references,
            lambda q, r: primary_similarity(
                q,
                r,
                mapped_anatomy,
                idf,
            ),
        )

        primary_k = adaptive_k(
            primary_neighbors,
            "oracle_primary",
        )

        (
            primary_candidate,
            primary_purity,
            primary_margin,
            primary_evidence,
        ) = weighted_vote(
            primary_neighbors[:primary_k],
            "oracle_primary",
            "primary_reliability",
            primary_frequency,
        )

        primary_top_similarity = (
            primary_neighbors[0][1]
            if primary_neighbors
            else 0.0
        )

        mapped_primary, primary_method = (
            primary_judge(
                query["raw_primary"],
                primary_candidate,
                primary_purity,
                primary_margin,
                primary_top_similarity,
            )
        )

        # Preserve raw dual-head attribute.
        mapped_attribute = query[
            "raw_attribute"
        ]

        predictions.append({
            "mapped_anatomy": mapped_anatomy,
            "mapped_primary": mapped_primary,
            "mapped_attribute": mapped_attribute,

            "anatomy_method": anatomy_method,
            "primary_method": primary_method,

            "anatomy_k": anatomy_k,
            "primary_k": primary_k,

            "anatomy_purity": anatomy_purity,
            "anatomy_margin": anatomy_margin,
            "primary_purity": primary_purity,
            "primary_margin": primary_margin,

            "anatomy_top_similarity": (
                anatomy_top_similarity
            ),
            "primary_top_similarity": (
                primary_top_similarity
            ),

            "anatomy_neighbors": (
                anatomy_evidence
            ),
            "primary_neighbors": (
                primary_evidence
            ),
        })

        if (
            index == 1
            or index % 100 == 0
            or index == len(queries)
        ):
            print(
                f"[agent] {index}/{len(queries)}",
                flush=True,
            )

    return predictions


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

        "anatomy_primary_joint": sum(
            prediction["mapped_anatomy"]
            == query["oracle_anatomy"]
            and prediction["mapped_primary"]
            == query["oracle_primary"]
            for prediction, query
            in zip(predictions, queries)
        ) / n,

        "full_joint_with_raw_attribute": sum(
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


def make_tokens(
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
        convert_sample(sample)
        for sample in data["train"]
    ]

    validation = [
        convert_sample(sample)
        for sample in data["val"]
    ]

    test = [
        convert_sample(sample)
        for sample in data["test"]
    ]

    print(
        f"train={len(train)} "
        f"val={len(validation)} "
        f"test={len(test)}",
        flush=True,
    )

    # Validation: train -> validation.
    validation_predictions = predict_queries(
        validation,
        train,
        build_detail_idf(train),
    )

    validation_metrics = calculate_metrics(
        validation_predictions,
        validation,
    )

    print(
        "Validation metrics:"
    )

    print(
        json.dumps(
            validation_metrics,
            indent=2,
        ),
        flush=True,
    )

    # Final prediction: train+validation -> test.
    references = train + validation

    test_predictions = predict_queries(
        test,
        references,
        build_detail_idf(references),
    )

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
        sample = dict(original)

        sample[
            "agent_rag_mapped_anatomy"
        ] = prediction["mapped_anatomy"]

        sample[
            "agent_rag_mapped_primary_lesion_type"
        ] = prediction["mapped_primary"]

        sample[
            "agent_rag_lesion_attribute"
        ] = prediction["mapped_attribute"]

        sample[
            "agent_rag_anatomy_method"
        ] = prediction["anatomy_method"]

        sample[
            "agent_rag_primary_method"
        ] = prediction["primary_method"]

        sample[
            "agent_rag_anatomy_k"
        ] = prediction["anatomy_k"]

        sample[
            "agent_rag_primary_k"
        ] = prediction["primary_k"]

        sample[
            "agent_rag_anatomy_purity"
        ] = prediction["anatomy_purity"]

        sample[
            "agent_rag_anatomy_margin"
        ] = prediction["anatomy_margin"]

        sample[
            "agent_rag_primary_purity"
        ] = prediction["primary_purity"]

        sample[
            "agent_rag_primary_margin"
        ] = prediction["primary_margin"]

        sample[
            "agent_rag_anatomy_neighbors"
        ] = prediction["anatomy_neighbors"]

        sample[
            "agent_rag_primary_neighbors"
        ] = prediction["primary_neighbors"]

        conditioning_tokens = make_tokens(
            prediction["mapped_anatomy"],
            prediction["mapped_primary"],
            prediction["mapped_attribute"],
        )

        sample[
            "rough_anatomy_names"
        ] = conditioning_tokens

        sample[
            "rough_anatomy_name"
        ] = " ".join(conditioning_tokens)

        sample[
            "anatomy_text"
        ] = " ".join(conditioning_tokens)

        sample[
            "anatomy_source"
        ] = "agent_based_rag"

        output["test"].append(sample)

        method_counts[
            prediction["anatomy_method"]
        ] += 1

        method_counts[
            prediction["primary_method"]
        ] += 1

    summary = {
        "protocol": {
            "validation_reference": "train",
            "validation_queries": "validation",
            "final_reference": "train+validation",
            "final_queries": "test",
            "test_report_used_for_prediction": False,
        },

        "agent_configuration": {
            "anatomy_gate": {
                "min_similarity": 0.30,
                "min_purity": 0.55,
                "min_margin": 0.10,
            },

            "primary_gate": {
                "min_similarity": 0.30,
                "min_purity": 0.75,
                "min_margin": 0.25,
            },

            "adaptive_k": [
                1,
                3,
                7,
            ],

            "class_balance_power": 0.25,

            "attribute_policy": (
                "preserve raw dual-head attribute"
            ),
        },

        "validation_metrics": (
            validation_metrics
        ),

        "test_metrics": test_metrics,

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
