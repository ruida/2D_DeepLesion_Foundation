import json
from pathlib import Path
from collections import Counter, defaultdict

ORACLE_JSON = Path(
    "data/generated_annotations/original_report_anat_lesion_token_improved_v2/"
    "deeplesion_original_report_anat_lesion_token_all_improved_v2.json"
)

TOTALSEG_JSON = Path(
    "data/generated_annotations/regenerated_totalseg_gtbbox/"
    "deeplesion_trainvaltest_gtbbox_totalseg_anatomy_ids_top2_fixed_adrenal.json"
)

OUT_DIR = Path("data/generated_annotations/totalseg_mapped_to_improved_oracle_text")
OUT_JSON = OUT_DIR / "deeplesion_totalseg_top2_fixed_adrenal_mapped_to_improved_oracle_text.json"
OUT_MAP = OUT_DIR / "totalseg_key_to_improved_oracle_train_map.json"
OUT_SUMMARY = OUT_DIR / "mapping_summary.json"

# This is for the regenerated TotalSeg rough ID space.
# 13 is the generic lesion token.
ID_TO_NAME = {
    0: "unknown",
    1: "lung",
    2: "heart",
    3: "vessels",
    4: "mediastinum",
    5: "abdomen",
    6: "liver",
    7: "kidney",
    8: "bone",
    9: "rib",
    10: "spine",
    11: "pelvis",
    12: "chest",
    13: "lesion",
    14: "retroperitoneum",
    15: "adrenal gland",
    16: "pancreas",
    17: "axilla",
    18: "chest wall",
}

VALID = {
    "unknown", "lung", "heart", "vessels", "mediastinum",
    "abdomen", "liver", "kidney", "bone", "rib", "spine",
    "pelvis", "chest", "retroperitoneum", "adrenal gland",
    "pancreas", "axilla", "chest wall"
}

def norm(x):
    if x is None:
        return "unknown"
    x = str(x).strip().lower().replace("_", " ")
    if x in ["", "none", "pad"]:
        return "unknown"
    return x

def load_trainvaltest(path):
    d = json.load(open(path))
    if isinstance(d, dict) and all(k in d for k in ["train", "val", "test"]):
        return d
    raise ValueError(f"Expected train/val/test dict: {path}")

def get_id(ex):
    return ex.get("id")

def oracle_anat(ex):
    a = norm(ex.get("anatomy_part", ex.get("rough_anatomy", "unknown")))
    return a if a in VALID else "unknown"

def totalseg_terms(ex):
    names = []

    if isinstance(ex.get("rough_anatomy_names"), list):
        names = [norm(x) for x in ex["rough_anatomy_names"]]
    elif isinstance(ex.get("rough_anatomy_ids"), list):
        for rid in ex["rough_anatomy_ids"]:
            try:
                rid = int(rid)
            except Exception:
                continue
            names.append(ID_TO_NAME.get(rid, f"anat_{rid}"))
    elif isinstance(ex.get("anatomy_ids"), list):
        # fallback only
        for rid in ex["anatomy_ids"]:
            try:
                rid = int(rid)
            except Exception:
                continue
            names.append(ID_TO_NAME.get(rid, f"anat_{rid}"))

    clean = []
    for n in names:
        if n in ["lesion", "pad", ""]:
            continue
        if n not in clean:
            clean.append(n)

    if not clean:
        clean = ["unknown"]

    return clean

def totalseg_key(ex):
    return "|".join(totalseg_terms(ex))

def make_text(anat):
    if anat == "unknown":
        return ["lesion", "unknown"], "lesion unknown"
    return ["lesion", anat], f"lesion {anat}"

def build_mapping(oracle, totalseg):
    oracle_by_id = {get_id(ex): ex for ex in oracle["train"]}
    counts = defaultdict(Counter)

    for ex in totalseg["train"]:
        sid = get_id(ex)
        if sid not in oracle_by_id:
            continue

        key = totalseg_key(ex)
        target = oracle_anat(oracle_by_id[sid])
        counts[key][target] += 1

    mapping = {}
    for key, c in counts.items():
        best, best_n = c.most_common(1)[0]
        mapping[key] = {
            "mapped_anatomy": best,
            "best_count": best_n,
            "distribution": dict(c.most_common())
        }

    return mapping

def apply_mapping(oracle, totalseg, mapping):
    out = {}
    summary = {}

    for split in ["train", "val", "test"]:
        oracle_by_id = {get_id(ex): ex for ex in oracle[split]}

        c_key = Counter()
        c_mapped = Counter()
        c_oracle_ref = Counter()
        same = 0
        n_ref = 0
        out[split] = []

        for ex0 in totalseg[split]:
            ex = dict(ex0)
            key = totalseg_key(ex)
            c_key[key] += 1

            if key in mapping:
                mapped = mapping[key]["mapped_anatomy"]
                source = "train_empirical_map"
            else:
                first = totalseg_terms(ex)[0]
                mapped = first if first in VALID else "unknown"
                source = "fallback_totalseg_first_term"

            if mapped not in VALID:
                mapped = "unknown"

            names, anatomy_text = make_text(mapped)

            # Save original fields for debugging.
            ex["totalseg_original_rough_anatomy_ids"] = ex.get("rough_anatomy_ids")
            ex["totalseg_original_rough_anatomy_names"] = ex.get("rough_anatomy_names")
            ex["totalseg_original_key"] = key

            # New mapped fields used by text-anatomy training.
            ex["mapped_anatomy_part"] = mapped
            ex["mapped_anatomy_source"] = source
            ex["rough_anatomy"] = mapped
            ex["fine_anatomy"] = mapped
            ex["rough_anatomy_name"] = mapped
            ex["anatomy_part"] = mapped
            ex["fine_anatomy_part"] = mapped
            ex["rough_anatomy_names"] = names
            ex["anatomy_text"] = anatomy_text

            # Keep report as target. Do not feed lesion_type_text unless intentionally doing oracle lesion-type input.
            if "lesion_type_text" not in ex:
                ex["lesion_type_text"] = ex.get("lesion_type_merged", "")

            c_mapped[mapped] += 1

            sid = get_id(ex)
            if sid in oracle_by_id:
                ref = oracle_anat(oracle_by_id[sid])
                c_oracle_ref[ref] += 1
                n_ref += 1
                if ref == mapped:
                    same += 1

            out[split].append(ex)

        summary[split] = {
            "n": len(out[split]),
            "top_totalseg_key": c_key.most_common(30),
            "top_mapped_anatomy": c_mapped.most_common(30),
            "top_oracle_reference": c_oracle_ref.most_common(30),
            "same_as_oracle_reference": same,
            "n_reference": n_ref,
            "same_rate_reference": same / max(n_ref, 1),
        }

    return out, summary

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Oracle :", ORACLE_JSON)
    print("TotalSeg:", TOTALSEG_JSON)

    oracle = load_trainvaltest(ORACLE_JSON)
    totalseg = load_trainvaltest(TOTALSEG_JSON)

    mapping = build_mapping(oracle, totalseg)
    out, summary = apply_mapping(oracle, totalseg, mapping)

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(OUT_MAP, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved:", OUT_JSON)
    print("Saved:", OUT_MAP)
    print("Saved:", OUT_SUMMARY)

    for split in ["train", "val", "test"]:
        s = summary[split]
        print(f"\n{split}: n={s['n']} same_as_oracle_reference={s['same_rate_reference']:.2%}")
        print("top mapped anatomy:", s["top_mapped_anatomy"][:10])

if __name__ == "__main__":
    main()
