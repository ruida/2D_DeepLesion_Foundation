#!/usr/bin/env python3
import argparse
import re
import pandas as pd

from modules.metrics import compute_scores

def clean_text(s):
    s = str(s).lower()
    # remove standalone token "unknown"
    s = re.sub(r"\bunknown\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_csv", required=True)
    ap.add_argument("--gts_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    # header=None is important for your res.csv/gts.csv format
    res = pd.read_csv(args.res_csv, header=None)
    gts = pd.read_csv(args.gts_csv, header=None)

    assert len(res) == len(gts), (len(res), len(gts))

    pred = [clean_text(x) for x in res.iloc[:, 0].tolist()]
    gt = [clean_text(x) for x in gts.iloc[:, 0].tolist()]

    res_clean = pd.DataFrame(pred)
    gts_clean = pd.DataFrame(gt)

    res_clean.to_csv(f"{args.out_dir}/res_no_unknown.csv", index=False, header=False)
    gts_clean.to_csv(f"{args.out_dir}/gts_no_unknown.csv", index=False, header=False)

    # R2Gen-Mamba metrics expect dict: id -> [sentence]
    gts_dict = {i: [gt[i]] for i in range(len(gt))}
    res_dict = {i: [pred[i]] for i in range(len(pred))}

    scores = compute_scores(gts_dict, res_dict)

    print("Evaluation after removing token 'unknown' from BOTH prediction and GT:")
    for k, v in scores.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
