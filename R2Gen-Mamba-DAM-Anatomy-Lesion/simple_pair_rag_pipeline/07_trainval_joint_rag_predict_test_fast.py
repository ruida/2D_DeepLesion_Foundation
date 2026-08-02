#!/usr/bin/env python3
"""Train/validation-tuned structured RAG, then train+val retrieval for test prediction.

Input JSON rows must contain:
  raw_simple_anatomy
  raw_primary_lesion_type
  raw_lesion_attribute
  raw_totalseg_details
  oracle_simple_anatomy
  oracle_primary_lesion_type
  oracle_lesion_attribute

The oracle fields from TEST are never used for retrieval or prediction. They are read
only after prediction to report held-out evaluation metrics.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def clean(value):
    return " ".join(str(value or "").lower().replace("_", " ").split())


def make_row(x):
    return {
        "id": str(x["id"]),
        "a": clean(x.get("raw_simple_anatomy", "unknown")),
        "p": clean(x.get("raw_primary_lesion_type", "other lesion")),
        "t": clean(x.get("raw_lesion_attribute", "none")),
        "d": tuple(sorted({clean(v) for v in x.get("raw_totalseg_details", []) if clean(v)})),
        "oa": clean(x.get("oracle_simple_anatomy", "unknown")),
        "op": clean(x.get("oracle_primary_lesion_type", "other lesion")),
        "ot": clean(x.get("oracle_lesion_attribute", "none")),
    }


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(q, r, weights):
    return (
        weights["anatomy"] * (q["a"] == r["a"])
        + weights["primary"] * (q["p"] == r["p"])
        + weights["attribute"] * (q["t"] == r["t"])
        + weights["details"] * jaccard(q["d"], r["d"])
    )


def query_key(q):
    return q["a"], q["p"], q["t"], q["d"]


def exact_table(reference):
    table = defaultdict(Counter)
    for r in reference:
        table[query_key(r)][(r["oa"], r["op"], r["ot"])] += 1
    return table


def retrieve(q, reference, k, weights, exclude_id=None):
    scored = []
    for r in reference:
        if exclude_id is not None and r["id"] == exclude_id:
            continue
        scored.append((r, similarity(q, r, weights)))
    scored.sort(key=lambda z: (-z[1], z[0]["id"]))
    return scored[:k]


def precompute_neighbors(queries, reference, max_k, weights):
    cache = []
    for i, q in enumerate(queries, 1):
        scored = []
        for r in reference:
            if r["id"] == q["id"]:
                continue
            scored.append((r, similarity(q, r, weights)))
        scored.sort(key=lambda z: (-z[1], z[0]["id"]))
        cache.append(scored[:max_k])

        if i == 1 or i % 100 == 0 or i == len(queries):
            print(f"[cache] {i}/{len(queries)} queries", flush=True)
    return cache


def predict_one_cached(q, neighbors, exact, cfg):
    key = query_key(q)
    if key in exact:
        counts = exact[key]
        target, count = counts.most_common(1)[0]
        support = sum(counts.values())
        purity = count / support
        if support >= cfg["exact_support"] and purity >= cfg["exact_purity"]:
            return target, "exact", purity, support, []

    neighbors = neighbors[: cfg["k"]]
    top_similarity = neighbors[0][1] if neighbors else 0.0

    joint_votes = Counter()
    for r, sim in neighbors:
        joint_votes[(r["oa"], r["op"], r["ot"])] += max(sim, 1e-6)

    if joint_votes:
        ranked = joint_votes.most_common()
        target, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(joint_votes.values())
        purity = top / total
        margin = (top - second) / total
        if (
            top_similarity >= cfg["min_similarity"]
            and purity >= cfg["joint_purity"]
            and margin >= cfg["joint_margin"]
        ):
            return target, "joint_rag", purity, len(neighbors), neighbors

    anatomy, pa, ma = weighted_vote(neighbors, "oa")
    primary, pp, mp = weighted_vote(neighbors, "op")
    attribute, pt, mt = weighted_vote(neighbors, "ot")

    accept_a = pa >= cfg["partial_purity"] and ma >= cfg["partial_margin"]
    accept_p = pp >= cfg["partial_purity"] and mp >= cfg["partial_margin"]
    accept_t = pt >= cfg["partial_purity"] and mt >= cfg["partial_margin"]

    target = (
        anatomy if accept_a else q["a"],
        primary if accept_p else q["p"],
        attribute if accept_t else q["t"],
    )
    accepted = sum((accept_a, accept_p, accept_t))
    method = {
        3: "partial_all",
        2: "partial_two",
        1: "partial_one",
        0: "raw_fallback",
    }[accepted]
    confidence = max(
        pa if accept_a else 0.0,
        pp if accept_p else 0.0,
        pt if accept_t else 0.0,
    )
    return target, method, confidence, len(neighbors), neighbors


def weighted_vote(neighbors, field):
    votes = Counter()
    for r, sim in neighbors:
        votes[r[field]] += max(sim, 1e-6)
    if not votes:
        return "unknown", 0.0, 0.0
    ranked = votes.most_common()
    label, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(votes.values())
    return label, top / total, (top - second) / total


def predict_one(q, reference, exact, cfg, weights, exclude_id=None):
    key = query_key(q)
    if key in exact:
        counts = exact[key]
        target, count = counts.most_common(1)[0]
        support = sum(counts.values())
        purity = count / support
        if support >= cfg["exact_support"] and purity >= cfg["exact_purity"]:
            return target, "exact", purity, support, []

    neighbors = retrieve(q, reference, cfg["k"], weights, exclude_id=exclude_id)
    top_similarity = neighbors[0][1] if neighbors else 0.0

    joint_votes = Counter()
    for r, sim in neighbors:
        joint_votes[(r["oa"], r["op"], r["ot"])] += max(sim, 1e-6)

    if joint_votes:
        ranked = joint_votes.most_common()
        target, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(joint_votes.values())
        purity = top / total
        margin = (top - second) / total
        if (
            top_similarity >= cfg["min_similarity"]
            and purity >= cfg["joint_purity"]
            and margin >= cfg["joint_margin"]
        ):
            return target, "joint_rag", purity, len(neighbors), neighbors

    anatomy, pa, ma = weighted_vote(neighbors, "oa")
    primary, pp, mp = weighted_vote(neighbors, "op")
    attribute, pt, mt = weighted_vote(neighbors, "ot")

    accept_a = pa >= cfg["partial_purity"] and ma >= cfg["partial_margin"]
    accept_p = pp >= cfg["partial_purity"] and mp >= cfg["partial_margin"]
    accept_t = pt >= cfg["partial_purity"] and mt >= cfg["partial_margin"]

    target = (
        anatomy if accept_a else q["a"],
        primary if accept_p else q["p"],
        attribute if accept_t else q["t"],
    )
    accepted = sum((accept_a, accept_p, accept_t))
    method = {
        3: "partial_all",
        2: "partial_two",
        1: "partial_one",
        0: "raw_fallback",
    }[accepted]
    confidence = max(pa if accept_a else 0.0, pp if accept_p else 0.0, pt if accept_t else 0.0)
    return target, method, confidence, len(neighbors), neighbors


def metrics(records):
    n = len(records)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "anatomy_accuracy": sum(x["pred_a"] == x["oa"] for x in records) / n,
        "primary_accuracy": sum(x["pred_p"] == x["op"] for x in records) / n,
        "attribute_accuracy": sum(x["pred_t"] == x["ot"] for x in records) / n,
        "anatomy_primary_joint": sum(
            (x["pred_a"], x["pred_p"]) == (x["oa"], x["op"]) for x in records
        ) / n,
        "full_joint": sum(
            (x["pred_a"], x["pred_p"], x["pred_t"])
            == (x["oa"], x["op"], x["ot"])
            for x in records
        ) / n,
    }


def objective(m):
    return (
        0.45 * m["anatomy_primary_joint"]
        + 0.25 * m["anatomy_accuracy"]
        + 0.20 * m["primary_accuracy"]
        + 0.10 * m["attribute_accuracy"]
    )


def evaluate_queries_cached(queries, neighbor_cache, exact, cfg):
    eval_rows = []
    for q, neighbors in zip(queries, neighbor_cache):
        target, _, _, _, _ = predict_one_cached(q, neighbors, exact, cfg)
        eval_rows.append(
            {
                "pred_a": target[0],
                "pred_p": target[1],
                "pred_t": target[2],
                "oa": q["oa"],
                "op": q["op"],
                "ot": q["ot"],
            }
        )
    return metrics(eval_rows)


def tokenize(a, p, t):
    tokens = a.replace("_", " ").split() + p.replace("_", " ").split()
    if t not in {"", "none"}:
        tokens += t.replace("_", " ").split()
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    data = json.load(open(args.input_json))
    train = [make_row(x) for x in data["train"]]
    val = [make_row(x) for x in data["val"]]
    test = [make_row(x) for x in data["test"]]

    weights = {"anatomy": 0.40, "primary": 0.30, "attribute": 0.15, "details": 0.15}

    print(
        f"train={len(train)} val={len(val)} test={len(test)}",
        flush=True,
    )

    # Stage 1: train -> validation hyperparameter tuning.
    # Compute nearest neighbors once, then reuse them for every configuration.
    print("Precomputing train -> validation neighbors...", flush=True)
    train_exact = exact_table(train)
    val_neighbor_cache = precompute_neighbors(
        val, train, max_k=7, weights=weights
    )

    grid = []
    config_index = 0
    for k in (1, 3, 5, 7):
        for exact_support in (2, 3, 5):
            for exact_purity in (0.55, 0.65):
                for joint_purity in (0.45, 0.55, 0.65):
                    for min_similarity in (0.30, 0.40, 0.50):
                        cfg = {
                            "k": k,
                            "exact_support": exact_support,
                            "exact_purity": exact_purity,
                            "joint_purity": joint_purity,
                            "joint_margin": 0.0,
                            "min_similarity": min_similarity,
                            "partial_purity": 0.65,
                            "partial_margin": 0.10,
                        }
                        config_index += 1
                        m = evaluate_queries_cached(
                            val, val_neighbor_cache, train_exact, cfg
                        )
                        grid.append(
                            {
                                "config": cfg,
                                "metrics": m,
                                "objective": objective(m),
                            }
                        )
                        if config_index == 1 or config_index % 25 == 0:
                            print(
                                f"[grid] {config_index}/216",
                                flush=True,
                            )

    print("[grid] 216/216", flush=True)
    grid.sort(key=lambda x: (-x["objective"], -x["metrics"]["full_joint"]))
    best = grid[0]["config"]

    print("Best configuration:", json.dumps(best, indent=2), flush=True)
    print(
        "Best validation metrics:",
        json.dumps(grid[0]["metrics"], indent=2),
        flush=True,
    )

    # Stage 2: freeze best settings; use train+val as final retrieval database.
    reference = train + val
    exact = exact_table(reference)

    output = {"train": data["train"], "val": data["val"], "test": []}
    methods = Counter()
    test_eval = []

    print("Predicting test from train+validation...", flush=True)
    for i, (original, q) in enumerate(zip(data["test"], test), 1):
        target, method, confidence, support, neighbors = predict_one(
            q, reference, exact, best, weights
        )
        y = dict(original)
        y["trainval_rag_mapped_anatomy"] = target[0]
        y["trainval_rag_mapped_primary_lesion_type"] = target[1]
        y["trainval_rag_mapped_lesion_attribute"] = target[2]
        y["trainval_rag_method"] = method
        y["trainval_rag_confidence"] = confidence
        y["trainval_rag_support"] = support
        y["trainval_rag_neighbor_ids"] = [r["id"] for r, _ in neighbors]
        y["trainval_rag_neighbor_similarities"] = [round(s, 6) for _, s in neighbors]

        tokens = tokenize(*target)
        y["rough_anatomy_names"] = tokens
        y["rough_anatomy_name"] = " ".join(tokens)
        y["anatomy_text"] = " ".join(tokens)
        y["anatomy_source"] = "trainval_joint_rag"
        output["test"].append(y)
        methods[method] += 1

        if i == 1 or i % 100 == 0 or i == len(test):
            print(f"[test] {i}/{len(test)}", flush=True)

        # Evaluation only. These test oracle labels are not used in prediction.
        test_eval.append(
            {
                "pred_a": target[0],
                "pred_p": target[1],
                "pred_t": target[2],
                "oa": q["oa"],
                "op": q["op"],
                "ot": q["ot"],
            }
        )

    summary = {
        "protocol": {
            "tuning_reference": "train",
            "tuning_queries": "validation",
            "final_reference": "train+validation",
            "final_queries": "test",
            "test_oracle_used_for_prediction": False,
        },
        "weights": weights,
        "best_config": best,
        "best_validation_metrics": grid[0]["metrics"],
        "test_metrics": metrics(test_eval),
        "test_methods": dict(methods),
        "reference_size": len(reference),
        "test_size": len(test),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.out_json, "w"), indent=2, ensure_ascii=False)
    json.dump(summary, open(args.summary_json, "w"), indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    print("Saved:", args.out_json)
    print("Saved:", args.summary_json)


if __name__ == "__main__":
    main()
