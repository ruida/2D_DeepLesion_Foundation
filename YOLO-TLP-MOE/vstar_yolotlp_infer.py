#!/usr/bin/env python3
"""
V*-style hierarchical crop-search wrapper for YOLO-TLP-MOE DeepLesion detection.

No LLM is added.
YOLO-TLP-MOE model code is not changed.

Pipeline:
1) Run YOLO on the full CT slice.
2) If detection is weak/missing, run hierarchical crop search.
3) Run YOLO on selected crops.
4) Map crop boxes back to original image coordinates.
5) Merge full-image and crop predictions by NMS.
6) Save YOLO-format txt predictions:
   class cx cy w h conf
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "Could not import ultralytics. Run this script inside the YOLO-TLP-MOE repo/environment."
    ) from e


Box = Tuple[float, float, float, float]


def list_images(source: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if source.is_file():
        return [source]
    return sorted([p for p in source.rglob("*") if p.suffix.lower() in exts])


def load_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path)

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        bg.alpha_composite(img.convert("RGBA"))
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

    return img


def to_gray_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L")).astype(np.float32)


def xyxy_clip(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box

    x1 = max(0.0, min(float(w), float(x1)))
    y1 = max(0.0, min(float(h), float(y1)))
    x2 = max(0.0, min(float(w), float(x2)))
    y2 = max(0.0, min(float(h), float(y2)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter

    if union <= 0:
        return 0.0
    return inter / union


def nms(preds: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    if not preds:
        return []

    kept = []
    classes = sorted(set(int(p["cls"]) for p in preds))

    for cls in classes:
        cls_preds = [p for p in preds if int(p["cls"]) == cls]
        cls_preds = sorted(cls_preds, key=lambda x: float(x["conf"]), reverse=True)

        while cls_preds:
            best = cls_preds.pop(0)
            kept.append(best)

            cls_preds = [
                p for p in cls_preds
                if iou_xyxy(best["box"], p["box"]) < iou_thr
            ]

    return sorted(kept, key=lambda x: float(x["conf"]), reverse=True)


def run_yolo_on_pil(
    model: YOLO,
    img: Image.Image,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    max_det: int,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    arr = np.asarray(img.convert("RGB"))

    results = model.predict(
        source=arr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
        verbose=verbose,
    )

    out = []

    if not results:
        return out

    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return out

    boxes = r.boxes.xyxy.detach().cpu().numpy()
    confs = r.boxes.conf.detach().cpu().numpy()
    clss = r.boxes.cls.detach().cpu().numpy()

    for box, score, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        out.append(
            {
                "cls": int(cls_id),
                "conf": float(score),
                "box": (x1, y1, x2, y2),
                "source": "full_or_crop",
            }
        )

    return out


def foreground_priority(gray_crop: np.ndarray) -> float:
    """
    CT foreground cue.

    Higher score for:
    - non-black body/tissue area
    - local contrast
    - reasonable foreground intensity

    This replaces the LLM cue from original V*.
    """
    if gray_crop.size == 0:
        return 0.0

    g = gray_crop.astype(np.float32)

    p1 = float(np.percentile(g, 1))
    p99 = float(np.percentile(g, 99))
    dynamic = max(1.0, p99 - p1)

    fg_thr = max(5.0, p1 + 0.03 * dynamic)
    fg = g > fg_thr

    fg_ratio = float(fg.mean())
    std_score = min(1.0, float(g.std()) / 64.0)

    if fg.any():
        mean_fg = float(g[fg].mean()) / 255.0
    else:
        mean_fg = 0.0

    score = 0.55 * fg_ratio + 0.30 * std_score + 0.15 * mean_fg
    return float(score)


def center_prior(box: Box, W: int, H: int) -> float:
    x1, y1, x2, y2 = box

    cx = 0.5 * (x1 + x2) / max(1, W)
    cy = 0.5 * (y1 + y2) / max(1, H)

    dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    return float(max(0.0, 1.0 - dist / 0.7071))


def lowconf_box_boost(child: Box, seed_boxes: List[Dict[str, Any]]) -> float:
    """
    If full-image YOLO gives weak boxes, use them as search hints.
    """
    if not seed_boxes:
        return 0.0

    x1, y1, x2, y2 = child
    best = 0.0

    for p in seed_boxes:
        bx1, by1, bx2, by2 = p["box"]
        bcx = 0.5 * (bx1 + bx2)
        bcy = 0.5 * (by1 + by2)

        inside = (x1 <= bcx <= x2) and (y1 <= bcy <= y2)
        overlap = iou_xyxy(child, p["box"])

        if inside or overlap > 0:
            best = max(best, float(p["conf"]))

    return best


def priority_for_box(
    gray_full: np.ndarray,
    box: Box,
    W: int,
    H: int,
    seed_boxes: List[Dict[str, Any]],
) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))

    crop = gray_full[y1:y2, x1:x2]

    tissue = foreground_priority(crop)
    center = center_prior(box, W, H)
    boost = lowconf_box_boost(box, seed_boxes)

    score = 0.60 * tissue + 0.20 * center + 0.20 * min(1.0, boost / 0.50)
    return float(score)


def split_box_vstar(box: Box, W: int, H: int, overlap: float) -> List[Box]:
    """
    V*-style recursive split.

    Landscape-like box: split vertically.
    Portrait-like box: split horizontally.
    Otherwise: split into 4 quadrants.

    overlap reduces missing lesions near crop borders.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    children: List[Box] = []

    def add(a, b, c, d):
        cb = xyxy_clip((a, b, c, d), W, H)
        if box_area(cb) > 16:
            children.append(cb)

    if w > 2.0 * h:
        step = w / 4.0
        pad = overlap * step

        for i in range(4):
            a = x1 + i * step - pad
            c = x1 + (i + 1) * step + pad
            add(a, y1, c, y2)

    elif h > 2.0 * w:
        step = h / 4.0
        pad = overlap * step

        for i in range(4):
            b = y1 + i * step - pad
            d = y1 + (i + 1) * step + pad
            add(x1, b, x2, d)

    else:
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        pad_x = overlap * w * 0.25
        pad_y = overlap * h * 0.25

        add(x1, y1, mx + pad_x, my + pad_y)
        add(mx - pad_x, y1, x2, my + pad_y)
        add(x1, my - pad_y, mx + pad_x, y2)
        add(mx - pad_x, my - pad_y, x2, y2)

    unique = []
    for c in children:
        if not any(iou_xyxy(c, u) > 0.98 for u in unique):
            unique.append(c)

    return unique


def crop_from_box(img: Image.Image, box: Box) -> Image.Image:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    return img.crop((x1, y1, x2, y2))


def map_crop_preds_to_global(
    crop_preds: List[Dict[str, Any]],
    crop_box: Box,
    W: int,
    H: int,
) -> List[Dict[str, Any]]:
    x0, y0, _, _ = crop_box
    out = []

    for p in crop_preds:
        bx1, by1, bx2, by2 = p["box"]
        global_box = xyxy_clip(
            (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
            W,
            H,
        )

        q = dict(p)
        q["box"] = global_box
        q["source"] = "vstar_crop"
        out.append(q)

    return out


def should_trigger_search(
    full_preds: List[Dict[str, Any]],
    final_conf: float,
    trigger_conf: float,
    always_search: bool,
) -> bool:
    if always_search:
        return True

    if not full_preds:
        return True

    max_conf = max(float(p["conf"]) for p in full_preds)

    strong = [p for p in full_preds if float(p["conf"]) >= final_conf]

    if not strong:
        return True

    if max_conf < trigger_conf:
        return True

    return False


def hierarchical_vstar_search(
    model: YOLO,
    img: Image.Image,
    gray_full: np.ndarray,
    full_lowconf_preds: List[Dict[str, Any]],
    args,
) -> Tuple[List[Dict[str, Any]], int]:
    W, H = img.size

    root: Box = (0.0, 0.0, float(W), float(H))

    queue = [
        {
            "box": root,
            "score": priority_for_box(gray_full, root, W, H, full_lowconf_preds),
            "depth": 0,
        }
    ]

    collected: List[Dict[str, Any]] = []
    searched_count = 0
    visited: List[Box] = []

    while queue and searched_count < args.max_patches:
        queue = sorted(queue, key=lambda z: z["score"], reverse=True)
        node = queue.pop(0)

        box = node["box"]
        depth = int(node["depth"])

        if any(iou_xyxy(box, v) > 0.995 for v in visited):
            continue

        visited.append(box)

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        if bw < args.min_patch or bh < args.min_patch:
            continue

        crop = crop_from_box(img, box)

        crop_preds = run_yolo_on_pil(
            model=model,
            img=crop,
            imgsz=args.imgsz,
            conf=args.crop_conf,
            iou=args.pred_iou,
            device=args.device,
            max_det=args.max_det_per_crop,
            verbose=False,
        )

        mapped = map_crop_preds_to_global(crop_preds, box, W, H)
        collected.extend(mapped)
        searched_count += 1

        if depth >= args.max_depth:
            continue

        children = split_box_vstar(box, W, H, args.overlap)

        ranked_children = []
        for c in children:
            s = priority_for_box(gray_full, c, W, H, full_lowconf_preds)
            ranked_children.append(
                {
                    "box": c,
                    "score": s,
                    "depth": depth + 1,
                }
            )

        ranked_children = sorted(ranked_children, key=lambda z: z["score"], reverse=True)
        queue.extend(ranked_children[: args.topk_children])

    return collected, searched_count


def save_yolo_txt(path: Path, preds: List[Dict[str, Any]], W: int, H: int):
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for p in preds:
        x1, y1, x2, y2 = p["box"]
        cls_id = int(p["cls"])
        conf = float(p["conf"])

        cx = ((x1 + x2) / 2.0) / W
        cy = ((y1 + y2) / 2.0) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H

        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        bw = min(1.0, max(0.0, bw))
        bh = min(1.0, max(0.0, bh))

        if bw <= 0 or bh <= 0:
            continue

        lines.append(
            f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}"
        )

    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def save_visualization(path: Path, img: Image.Image, preds: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    vis = img.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)

    for p in preds:
        x1, y1, x2, y2 = p["box"]
        conf = float(p["conf"])
        cls_id = int(p["cls"])
        src = str(p.get("source", ""))

        if src == "vstar_crop":
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text(
            (x1, max(0, y1 - 12)),
            f"{cls_id}:{conf:.2f}:{src}",
            fill=color,
        )

    vis.save(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--full-conf", type=float, default=0.01)
    parser.add_argument("--crop-conf", type=float, default=0.01)
    parser.add_argument("--final-conf", type=float, default=0.05)
    parser.add_argument("--trigger-conf", type=float, default=0.20)

    parser.add_argument("--pred-iou", type=float, default=0.70)
    parser.add_argument("--merge-iou", type=float, default=0.50)

    parser.add_argument("--max-det-full", type=int, default=20)
    parser.add_argument("--max-det-per-crop", type=int, default=10)
    parser.add_argument("--max-final-det", type=int, default=3)

    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-patches", type=int, default=24)
    parser.add_argument("--topk-children", type=int, default=3)
    parser.add_argument("--min-patch", type=int, default=96)
    parser.add_argument("--overlap", type=float, default=0.20)

    parser.add_argument("--always-search", action="store_true")
    parser.add_argument("--save-vis", action="store_true")

    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)

    labels_dir = out_dir / "labels"
    vis_dir = out_dir / "vis"
    summary_csv = out_dir / "vstar_search_summary.csv"

    labels_dir.mkdir(parents=True, exist_ok=True)

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(source)

    if not image_paths:
        raise RuntimeError(f"No images found in {source}")

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    rows = []

    for idx, img_path in enumerate(image_paths, start=1):
        img = load_image_rgb(img_path)
        W, H = img.size
        gray = to_gray_np(img)

        full_preds = run_yolo_on_pil(
            model=model,
            img=img,
            imgsz=args.imgsz,
            conf=args.full_conf,
            iou=args.pred_iou,
            device=args.device,
            max_det=args.max_det_full,
            verbose=False,
        )

        for p in full_preds:
            p["source"] = "full"

        search_needed = should_trigger_search(
            full_preds=full_preds,
            final_conf=args.final_conf,
            trigger_conf=args.trigger_conf,
            always_search=args.always_search,
        )

        crop_preds = []
        searched_patches = 0

        if search_needed:
            crop_preds, searched_patches = hierarchical_vstar_search(
                model=model,
                img=img,
                gray_full=gray,
                full_lowconf_preds=full_preds,
                args=args,
            )

        all_preds = full_preds + crop_preds

        all_preds = [p for p in all_preds if float(p["conf"]) >= args.final_conf]

        merged = nms(all_preds, iou_thr=args.merge_iou)
        merged = merged[: args.max_final_det]

        label_path = labels_dir / f"{img_path.stem}.txt"
        save_yolo_txt(label_path, merged, W, H)

        if args.save_vis:
            vis_path = vis_dir / f"{img_path.stem}.jpg"
            save_visualization(vis_path, img, merged)

        max_full_conf = max([float(p["conf"]) for p in full_preds], default=0.0)
        max_final_conf = max([float(p["conf"]) for p in merged], default=0.0)

        rows.append(
            {
                "image": str(img_path),
                "width": W,
                "height": H,
                "full_preds_raw": len(full_preds),
                "search_triggered": int(search_needed),
                "searched_patches": searched_patches,
                "crop_preds_raw": len(crop_preds),
                "final_preds": len(merged),
                "max_full_conf": f"{max_full_conf:.6f}",
                "max_final_conf": f"{max_final_conf:.6f}",
                "label_file": str(label_path),
            }
        )

        if idx % 50 == 0 or idx == len(image_paths):
            print(
                f"[{idx}/{len(image_paths)}] "
                f"{img_path.name} | "
                f"search={search_needed} | "
                f"patches={searched_patches} | "
                f"full_raw={len(full_preds)} | "
                f"crop_raw={len(crop_preds)} | "
                f"final={len(merged)}"
            )

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "width",
                "height",
                "full_preds_raw",
                "search_triggered",
                "searched_patches",
                "crop_preds_raw",
                "final_preds",
                "max_full_conf",
                "max_final_conf",
                "label_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Labels saved to: {labels_dir}")
    print(f"Summary saved to: {summary_csv}")

    if args.save_vis:
        print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
