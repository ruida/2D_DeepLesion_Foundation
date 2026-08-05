#!/usr/bin/env python3
import argparse, json
from pathlib import Path

DEFAULT_PROMPT = "Please describe the lesion findings in this medical image concisely."

def make_record(rec, image_root: Path, image_mode: str, include_meta: bool):
    paths = rec["image_path"]
    if isinstance(paths, str):
        paths = [paths]
    if image_mode == "first":
        images = [paths[0]]
        img_tokens = "<image>"
    elif image_mode == "second":
        images = [paths[min(1, len(paths)-1)]]
        img_tokens = "<image>"
    else:
        images = paths
        img_tokens = " ".join(["<image>"] * len(images))

    prompt = DEFAULT_PROMPT
    if include_meta:
        prompt += f" Anatomy IDs: {rec.get('anatomy_ids', [])}. Bounding boxes: {rec.get('bboxes', [])}."

    prompt = f"{img_tokens}\n{prompt}"

    return {
        "id": rec.get("id", ""),
        "image": [str((image_root / p).resolve()) for p in images],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": rec.get("report", "").strip()},
        ],
    }

def convert_split(items, image_root: Path, image_mode: str, include_meta: bool):
    return [make_record(x, image_root, image_mode, include_meta) for x in items]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_path", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_mode", choices=["first", "second", "all"], default="first")
    ap.add_argument("--include_meta_in_prompt", action="store_true")
    args = ap.parse_args()

    ann = json.load(open(args.ann_path))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)
    for split in ("train", "val", "test"):
        data = convert_split(ann.get(split, []), image_root, args.image_mode, args.include_meta_in_prompt)
        with open(out / f"deeplesion_{split}.json", "w") as f:
            json.dump(data, f, indent=2)
        print(split, len(data), '->', out / f"deeplesion_{split}.json")

if __name__ == "__main__":
    main()
