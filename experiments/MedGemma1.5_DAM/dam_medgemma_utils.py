#!/usr/bin/env python3
"""Utilities for DAM-style DeepLesion multi-image prompting with MedGemma.

This is a practical integration that keeps MedGemma's pretrained vision tower intact.
Each sample is represented by four spatially aligned visual inputs:
  1. full CT image
  2. full binary lesion mask
  3. expanded focal CT crop
  4. focal binary mask crop

It is DAM-style focal prompting, not an exact insertion of DAM gated cross-attention
inside MedGemma's vision transformer.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


def resolve_path(root: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(root, value)


def open_rgb(path: str) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def open_binary_mask(path: str, expected_size: Tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != expected_size:
        mask = mask.resize(expected_size, resample=Image.Resampling.NEAREST)
    array = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return Image.fromarray(array, mode="L")


def first_bbox_xyxy(item: Dict[str, Any]) -> List[int]:
    boxes = item.get("bboxes") or item.get("bbox") or item.get("box")
    if boxes is None:
        raise KeyError(f"Sample {item.get('id')} has no bbox field")

    if isinstance(boxes, Sequence) and len(boxes) == 4 and not isinstance(boxes[0], Sequence):
        box = boxes
    else:
        if not boxes:
            raise ValueError(f"Sample {item.get('id')} has an empty bbox field")
        box = boxes[0]

    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid xyxy bbox for {item.get('id')}: {box}")
    return [x1, y1, x2, y2]


def expand_xyxy(
    bbox: Sequence[int],
    image_size: Tuple[int, int],
    scale: float = 3.0,
    min_size: int = 48,
) -> List[int]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    box_w = max(x2 - x1, float(min_size)) * scale
    box_h = max(y2 - y1, float(min_size)) * scale

    nx1 = max(0, int(round(cx - box_w / 2.0)))
    ny1 = max(0, int(round(cy - box_h / 2.0)))
    nx2 = min(width, int(round(cx + box_w / 2.0)))
    ny2 = min(height, int(round(cy + box_h / 2.0)))

    if nx2 <= nx1 or ny2 <= ny1:
        raise ValueError(f"Expanded bbox became invalid: {[nx1, ny1, nx2, ny2]}")
    return [nx1, ny1, nx2, ny2]


def mask_to_rgb(mask: Image.Image) -> Image.Image:
    return mask.convert("RGB")


def make_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.35) -> Image.Image:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask.convert("L")) > 0
    overlay = image_array.copy()
    # Bright neutral highlight avoids relying on a color-specific semantic.
    overlay[mask_array] = (1.0 - alpha) * overlay[mask_array] + alpha * 255.0
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def build_dam_visuals(
    item: Dict[str, Any],
    image_root: str,
    crop_scale: float,
    min_crop_size: int,
    visual_mode: str = "four_image",
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    image_rel = item["image_path"][0] if isinstance(item["image_path"], list) else item["image_path"]
    image_path = resolve_path(image_root, image_rel)
    mask_path = resolve_path(image_root, item["mask_path"])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    image = open_rgb(image_path)
    mask = open_binary_mask(mask_path, image.size)
    bbox = first_bbox_xyxy(item)
    crop_box = expand_xyxy(bbox, image.size, crop_scale, min_crop_size)

    focal_image = image.crop(tuple(crop_box))
    focal_mask = mask.crop(tuple(crop_box))

    if visual_mode == "four_image":
        visuals = [image, mask_to_rgb(mask), focal_image, mask_to_rgb(focal_mask)]
    elif visual_mode == "two_overlay":
        visuals = [make_overlay(image, mask), make_overlay(focal_image, focal_mask)]
    else:
        raise ValueError(f"Unsupported visual_mode: {visual_mode}")

    meta = {
        "image_path": image_rel,
        "mask_path": item["mask_path"],
        "bbox": bbox,
        "focal_crop_xyxy": crop_box,
        "visual_mode": visual_mode,
    }
    return visuals, meta


def visual_user_content(user_prompt: str, visual_mode: str) -> List[Dict[str, str]]:
    if visual_mode == "four_image":
        return [
            {"type": "text", "text": user_prompt + "\nImage 1: full CT slice."},
            {"type": "image"},
            {"type": "text", "text": "Image 2: full-resolution binary lesion mask aligned with Image 1."},
            {"type": "image"},
            {"type": "text", "text": "Image 3: expanded focal crop centered on the detected lesion."},
            {"type": "image"},
            {"type": "text", "text": "Image 4: binary lesion mask aligned with Image 3. Generate only the short report."},
            {"type": "image"},
        ]
    return [
        {"type": "text", "text": user_prompt + "\nImage 1 is the full masked CT view. Image 2 is the focal masked crop."},
        {"type": "image"},
        {"type": "image"},
    ]
