#!/usr/bin/env python3
"""Convert R2Gen-DAM DeepLesion JSON into Qwen3-VL multi-image SFT JSON.

Visual modes:
  four_image: full CT, full mask, focal CT crop, focal crop mask
  two_overlay: full CT mask overlay, focal CT mask overlay

The generated JSON uses absolute image paths and is compatible with the existing
Qwen3-VL fine-tuning and evaluation code in this repository.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance

DEFAULT_PROMPT_FOUR = (
    "The four images are: (1) full CT key slice, (2) full lesion mask, "
    "(3) focal lesion crop with surrounding context, and (4) focal crop mask. "
    "Generate a concise lesion-focused DeepLesion report using short comma-separated medical tags."
)

DEFAULT_PROMPT_OVERLAY = (
    "The two images are: (1) full CT key slice with the lesion mask highlighted, "
    "and (2) focal lesion crop with the mask highlighted. "
    "Generate a concise lesion-focused DeepLesion report using short comma-separated medical tags."
)


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def first_image_path(item: Dict[str, Any]) -> str:
    paths = item["image_path"]
    if isinstance(paths, str):
        return paths
    if not paths:
        raise ValueError(f"Sample {item.get('id')} has empty image_path")
    return paths[0]


def first_bbox(item: Dict[str, Any]) -> List[float]:
    boxes = item.get("bboxes") or item.get("bbox")
    if boxes is None:
        raise ValueError(f"Sample {item.get('id')} has no bbox")
    if len(boxes) == 4 and not isinstance(boxes[0], (list, tuple)):
        return list(boxes)
    if not boxes:
        raise ValueError(f"Sample {item.get('id')} has empty bboxes")
    return list(boxes[0])


def to_xyxy(box: Sequence[float], fmt: str, w: int, h: int) -> Tuple[int, int, int, int]:
    fmt = (fmt or "xyxy").lower()
    vals = [float(x) for x in box]
    if fmt == "xyxy":
        x1, y1, x2, y2 = vals
    elif fmt in {"xywh", "coco"}:
        x, y, bw, bh = vals
        x1, y1, x2, y2 = x, y, x + bw, y + bh
    elif fmt in {"yolo", "cxcywh", "normalized_cxcywh"}:
        cx, cy, bw, bh = vals
        # Treat values <= 1.5 as normalized.
        if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
            cx, bw = cx * w, bw * w
            cy, bh = cy * h, bh * h
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    else:
        raise ValueError(f"Unsupported bbox format: {fmt}")

    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), w))
    y2 = max(y1 + 1, min(int(round(y2)), h))
    return x1, y1, x2, y2


def expanded_crop(box: Tuple[int, int, int, int], w: int, h: int, scale: float, min_size: int):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = x2 - x1, y2 - y1
    size_w = max(float(min_size), bw * scale)
    size_h = max(float(min_size), bh * scale)
    nx1 = int(round(cx - size_w / 2))
    ny1 = int(round(cy - size_h / 2))
    nx2 = int(round(cx + size_w / 2))
    ny2 = int(round(cy + size_h / 2))
    nx1 = max(0, nx1); ny1 = max(0, ny1)
    nx2 = min(w, nx2); ny2 = min(h, ny2)
    if nx2 <= nx1: nx2 = min(w, nx1 + 1)
    if ny2 <= ny1: ny2 = min(h, ny1 + 1)
    return nx1, ny1, nx2, ny2


def load_mask(path: Path, size: Tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def mask_rgb(mask: Image.Image) -> Image.Image:
    return mask.convert("RGB")


def overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.35) -> Image.Image:
    base = image.convert("RGB")
    m = np.asarray(mask.convert("L")) > 0
    arr = np.asarray(base).copy()
    red = np.zeros_like(arr)
    red[..., 0] = 255
    arr[m] = ((1.0 - alpha) * arr[m] + alpha * red[m]).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_record(item: Dict[str, Any], image_root: Path, derived_root: Path,
                visual_mode: str, crop_scale: float, min_crop_size: int) -> Dict[str, Any]:
    sid = str(item.get("id", ""))
    img_path = resolve_path(image_root, first_image_path(item))
    mask_path = resolve_path(image_root, item["mask_path"])
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    image = Image.open(img_path).convert("RGB")
    mask = load_mask(mask_path, image.size)
    w, h = image.size
    box = to_xyxy(first_bbox(item), item.get("bbox_format", "xyxy"), w, h)
    crop_box = expanded_crop(box, w, h, crop_scale, min_crop_size)
    crop_img = image.crop(crop_box)
    crop_mask = mask.crop(crop_box)

    split = str(item.get("_split", "unknown"))
    out_dir = derived_root / split / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    full_img_out = out_dir / "full_image.png"
    full_mask_out = out_dir / "full_mask.png"
    crop_img_out = out_dir / "focal_image.png"
    crop_mask_out = out_dir / "focal_mask.png"
    full_overlay_out = out_dir / "full_overlay.png"
    crop_overlay_out = out_dir / "focal_overlay.png"

    # Re-save all visual inputs to make the converted dataset self-consistent.
    image.save(full_img_out)
    mask_rgb(mask).save(full_mask_out)
    crop_img.save(crop_img_out)
    mask_rgb(crop_mask).save(crop_mask_out)

    if visual_mode == "four_image":
        paths = [full_img_out, full_mask_out, crop_img_out, crop_mask_out]
        prompt = DEFAULT_PROMPT_FOUR
    else:
        overlay(image, mask).save(full_overlay_out)
        overlay(crop_img, crop_mask).save(crop_overlay_out)
        paths = [full_overlay_out, crop_overlay_out]
        prompt = DEFAULT_PROMPT_OVERLAY

    tokens = "\n".join(["<image>"] * len(paths))
    return {
        "id": sid,
        "image": [str(p.resolve()) for p in paths],
        "conversations": [
            {"from": "human", "value": f"{tokens}\n{prompt}"},
            {"from": "gpt", "value": str(item.get("report", "")).strip()},
        ],
        "dam_metadata": {
            "source_image": str(img_path),
            "source_mask": str(mask_path),
            "bbox_xyxy": list(box),
            "crop_xyxy": list(crop_box),
            "bbox_source": item.get("bbox_source", "unspecified"),
            "mask_source": item.get("mask_source", "unspecified"),
            "visual_mode": visual_mode,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_path", required=True, help="R2Gen-DAM JSON with train/val/test")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--output_dir", required=True, help="Directory for Qwen JSON files")
    ap.add_argument("--derived_image_dir", required=True, help="Directory for generated DAM visual images")
    ap.add_argument("--visual_mode", choices=["four_image", "two_overlay"], default="four_image")
    ap.add_argument("--crop_scale", type=float, default=3.0)
    ap.add_argument("--min_crop_size", type=int, default=48)
    ap.add_argument("--max_samples_per_split", type=int, default=None)
    args = ap.parse_args()

    data = json.load(open(args.ann_path))
    image_root = Path(args.image_root)
    out_dir = Path(args.output_dir)
    derived = Path(args.derived_image_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train", "val", "test"):
        records = data.get(split, [])
        if args.max_samples_per_split is not None:
            records = records[:args.max_samples_per_split]
        converted = []
        for i, item in enumerate(records):
            row = dict(item)
            row["_split"] = split
            converted.append(make_record(row, image_root, derived, args.visual_mode,
                                         args.crop_scale, args.min_crop_size))
            if (i + 1) % 500 == 0:
                print(f"{split}: {i+1}/{len(records)}", flush=True)
        out_path = out_dir / f"deeplesion_{split}.json"
        with open(out_path, "w") as f:
            json.dump(converted, f, indent=2)
        summary[split] = len(converted)
        print(f"{split}: {len(converted)} -> {out_path}")

    with open(out_dir / "conversion_summary.json", "w") as f:
        json.dump({
            "visual_mode": args.visual_mode,
            "crop_scale": args.crop_scale,
            "min_crop_size": args.min_crop_size,
            "counts": summary,
        }, f, indent=2)


if __name__ == "__main__":
    main()
