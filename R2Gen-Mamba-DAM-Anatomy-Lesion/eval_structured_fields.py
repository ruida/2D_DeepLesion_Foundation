import argparse
import pandas as pd
from sklearn.metrics import classification_report


def parse_structured(text):
    out = {
        "rough_anatomy": "",
        "fine_anatomy": "",
        "lesion_type": "",
    }

    text = str(text).strip().lower()

    for part in text.split(";"):
        part = part.strip()
        if ":" not in part:
            continue

        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key in out:
            out[key] = value

    return out


def read_prediction_csv(path):
    """
    gts.csv and res.csv have no header.
    Read as one-column CSV.
    """
    df = pd.read_csv(path, header=None)

    if df.shape[1] == 1:
        df.columns = ["text"]
    else:
        # fallback: use last column if more than one column exists
        df = df.rename(columns={df.columns[-1]: "text"})

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gts_csv", required=True)
    parser.add_argument("--res_csv", required=True)
    parser.add_argument(
        "--out_csv",
        default="results/deeplesion_structured_target_test/structured_field_eval.csv",
    )
    args = parser.parse_args()

    gts = read_prediction_csv(args.gts_csv)
    res = read_prediction_csv(args.res_csv)

    print("GT rows:", len(gts))
    print("Pred rows:", len(res))

    if len(gts) != len(res):
        raise ValueError(f"Length mismatch: gts={len(gts)}, res={len(res)}")

    rows = []

    for i in range(len(gts)):
        gt_text = gts.iloc[i]["text"]
        pred_text = res.iloc[i]["text"]

        gt = parse_structured(gt_text)
        pred = parse_structured(pred_text)

        row = {
            "idx": i,
            "gt_text": gt_text,
            "pred_text": pred_text,

            "gt_rough_anatomy": gt["rough_anatomy"],
            "pred_rough_anatomy": pred["rough_anatomy"],

            "gt_fine_anatomy": gt["fine_anatomy"],
            "pred_fine_anatomy": pred["fine_anatomy"],

            "gt_lesion_type": gt["lesion_type"],
            "pred_lesion_type": pred["lesion_type"],
        }

        row["rough_correct"] = int(row["gt_rough_anatomy"] == row["pred_rough_anatomy"])
        row["fine_correct"] = int(row["gt_fine_anatomy"] == row["pred_fine_anatomy"])
        row["lesion_correct"] = int(row["gt_lesion_type"] == row["pred_lesion_type"])
        row["all_correct"] = int(
            row["rough_correct"] == 1
            and row["fine_correct"] == 1
            and row["lesion_correct"] == 1
        )

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    print("\nSaved:", args.out_csv)
    print("Num samples:", len(out))

    print("\nField accuracy:")
    print("rough_anatomy_acc:", out["rough_correct"].mean())
    print("fine_anatomy_acc :", out["fine_correct"].mean())
    print("lesion_type_acc  :", out["lesion_correct"].mean())
    print("all_fields_acc   :", out["all_correct"].mean())

    print("\nLesion type classification report:")
    print(classification_report(
        out["gt_lesion_type"],
        out["pred_lesion_type"],
        digits=4,
        zero_division=0,
    ))

    print("\nRough anatomy classification report:")
    print(classification_report(
        out["gt_rough_anatomy"],
        out["pred_rough_anatomy"],
        digits=4,
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
