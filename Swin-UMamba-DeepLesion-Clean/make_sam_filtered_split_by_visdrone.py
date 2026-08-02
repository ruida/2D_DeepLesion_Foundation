#!/usr/bin/env python3
"""
Split SAM filtered DeepLesion images/masks into train/val/test-dev
by matching file names to VisDroneDeepLesion_clear split folders.

Input SAM folders:
    images_filtered/
    labels_filtered/ or masks_filtered/

Reference split folders:
    VisDroneDeepLesion_clear/
        VisDrone2019-DET-train/images
        VisDrone2019-DET-val/images
        VisDrone2019-DET-test-dev/images

Normalization:
    000001_01_01_109.png
    000001_01_01_109_000.png

both become:
    000001_01_01_109

Output:
    data/sam_filtered_by_visdrone_split/
        train/images/
        train/masks/
        train/manifest.csv
        train/missing_from_sam.csv

        val/images/
        val/masks/
        val/manifest.csv
        val/missing_from_sam.csv

        test-dev/images/
        test-dev/masks/
        test-dev/manifest.csv
        test-dev/missing_from_sam.csv

The script uses symlinks by default.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
from collections import defaultdict


SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
}


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def canonical_stem(path_or_name) -> str:
    stem = Path(path_or_name).stem
    if stem.endswith("_000"):
        stem = stem[:-4]
    return stem


def collect_files(folder: Path, exts):
    groups = defaultdict(list)
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            groups[canonical_stem(p)].append(p)
    return groups


def choose_preferred(files):
    """
    Prefer non-_000 file if both exist.
    """
    files = sorted(files, key=lambda x: x.name)
    non_000 = [p for p in files if not p.stem.endswith("_000")]
    if non_000:
        return non_000[0]
    return files[0]


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(mode)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def process_split(
    split_name: str,
    visdrone_split_dir: Path,
    sam_image_groups,
    sam_mask_groups,
    out_root: Path,
    mode: str,
):
    ref_images_dir = visdrone_split_dir / "images"
    if not ref_images_dir.is_dir():
        raise FileNotFoundError(f"Missing VisDrone image dir: {ref_images_dir}")

    out_split = out_root / split_name
    out_images = out_split / "images"
    out_masks = out_split / "masks"

    rows = []
    missing_rows = []
    duplicate_rows = []

    ref_groups = collect_files(ref_images_dir, IMAGE_EXTS)
    ref_stems = sorted(ref_groups.keys())

    matched = 0
    matched_mask = 0

    for stem in ref_stems:
        ref_file = choose_preferred(ref_groups[stem])

        sam_img = None
        sam_mask = None

        if stem in sam_image_groups:
            sam_img = choose_preferred(sam_image_groups[stem])

        if stem in sam_mask_groups:
            sam_mask = choose_preferred(sam_mask_groups[stem])

        if sam_img is None:
            missing_rows.append({
                "canonical_stem": stem,
                "reason": "missing_sam_image",
                "reference_image": str(ref_file),
            })
            continue

        matched += 1

        if sam_mask is None:
            missing_rows.append({
                "canonical_stem": stem,
                "reason": "missing_sam_mask",
                "reference_image": str(ref_file),
            })
            continue

        matched_mask += 1

        out_img = out_images / f"{stem}{sam_img.suffix}"
        out_mask = out_masks / f"{stem}{sam_mask.suffix}"

        link_or_copy(sam_img, out_img, mode)
        link_or_copy(sam_mask, out_mask, mode)

        if len(sam_image_groups.get(stem, [])) > 1 or len(sam_mask_groups.get(stem, [])) > 1:
            duplicate_rows.append({
                "canonical_stem": stem,
                "sam_image_candidates": "|".join(str(p) for p in sorted(sam_image_groups.get(stem, []))),
                "sam_mask_candidates": "|".join(str(p) for p in sorted(sam_mask_groups.get(stem, []))),
                "chosen_sam_image": str(sam_img),
                "chosen_sam_mask": str(sam_mask),
            })

        rows.append({
            "split": split_name,
            "canonical_stem": stem,
            "reference_image": str(ref_file),
            "sam_image": str(sam_img),
            "sam_mask": str(sam_mask),
            "output_image": str(out_img),
            "output_mask": str(out_mask),
        })

    write_csv(
        out_split / "manifest.csv",
        rows,
        [
            "split",
            "canonical_stem",
            "reference_image",
            "sam_image",
            "sam_mask",
            "output_image",
            "output_mask",
        ],
    )

    write_csv(
        out_split / "missing_from_sam.csv",
        missing_rows,
        [
            "canonical_stem",
            "reason",
            "reference_image",
        ],
    )

    write_csv(
        out_split / "duplicates_in_sam.csv",
        duplicate_rows,
        [
            "canonical_stem",
            "sam_image_candidates",
            "sam_mask_candidates",
            "chosen_sam_image",
            "chosen_sam_mask",
        ],
    )

    print(f"[{split_name}]")
    print(f"  VisDrone reference images: {len(ref_stems)}")
    print(f"  matched SAM images:        {matched}")
    print(f"  matched SAM masks:         {matched_mask}")
    print(f"  missing rows:              {len(missing_rows)}")
    print(f"  duplicate SAM stems:       {len(duplicate_rows)}")
    print(f"  output images:             {out_images}")
    print(f"  output masks:              {out_masks}")
    print(f"  manifest:                  {out_split / 'manifest.csv'}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_images_dir", required=True)
    parser.add_argument("--sam_masks_dir", required=True)
    parser.add_argument("--visdrone_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    sam_images_dir = Path(args.sam_images_dir)
    sam_masks_dir = Path(args.sam_masks_dir)
    visdrone_root = Path(args.visdrone_root)
    out_root = Path(args.out_root)

    print("============================================================")
    print("Make SAM filtered split by VisDroneDeepLesion_clear")
    print(f"SAM images:     {sam_images_dir}")
    print(f"SAM masks:      {sam_masks_dir}")
    print(f"VisDrone root:  {visdrone_root}")
    print(f"Output root:    {out_root}")
    print(f"Mode:           {args.mode}")
    print("============================================================")
    print()

    sam_image_groups = collect_files(sam_images_dir, IMAGE_EXTS)
    sam_mask_groups = collect_files(sam_masks_dir, MASK_EXTS)

    print(f"SAM unique image stems: {len(sam_image_groups)}")
    print(f"SAM unique mask stems:  {len(sam_mask_groups)}")
    print()

    for split_name, split_dirname in SPLITS.items():
        process_split(
            split_name=split_name,
            visdrone_split_dir=visdrone_root / split_dirname,
            sam_image_groups=sam_image_groups,
            sam_mask_groups=sam_mask_groups,
            out_root=out_root,
            mode=args.mode,
        )

    print("[DONE]")


if __name__ == "__main__":
    main()
