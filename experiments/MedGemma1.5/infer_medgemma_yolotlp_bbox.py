#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFile
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True


SYSTEM_PROMPT = (
    "You are a radiology assistant. "
    "Generate a concise lesion-focused report from this cropped CT lesion image. "
    "Use short medical tag-style phrasing consistent with the target examples. "
    "Do not mention bounding boxes, crops, annotations, or training artifacts."
)

USER_PROMPT = (
    "Generate a short lesion-focused report for this DeepLesion CT lesion crop."
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--annotations_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--output_jsonl", required=True)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument(
        "--bbox_key",
        default="bboxes",
        help="JSON key containing predicted bounding boxes.",
    )
    parser.add_argument(
        "--bbox_format",
        default="xyxy",
        choices=["xyxy", "xywh", "cxcywh"],
    )
    parser.add_argument(
        "--normalized_bbox",
        action="store_true",
        help="Use when bbox coordinates are normalized to 0-1.",
    )
    parser.add_argument(
        "--bbox_index",
        type=int,
        default=0,
        help="Which predicted box to use; 0 means top prediction.",
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        default=2.0,
        help="Expansion around the predicted bounding box.",
    )
    parser.add_argument(
        "--min_crop_size",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--skip_missing_bbox",
        action="store_true",
        help="Skip samples without predicted boxes instead of using full image.",
    )

    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_open_image(path: str) -> Image.Image:
    image = Image.open(path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def resolve_image_path(item: Dict[str, Any], image_root: str) -> Tuple[str, str]:
    image_value = item.get("image_path")

    if isinstance(image_value, list):
        if not image_value:
            raise ValueError(
                f"Empty image_path for sample {item.get('id')}"
            )
        image_rel = str(image_value[0])
    elif isinstance(image_value, str):
        image_rel = image_value
    else:
        raise TypeError(
            f"Invalid image_path for sample {item.get('id')}: "
            f"{image_value!r}"
        )

    image_path = (
        image_rel
        if os.path.isabs(image_rel)
        else os.path.join(image_root, image_rel)
    )

    return image_rel, image_path


def extract_box(
    item: Dict[str, Any],
    bbox_key: str,
    bbox_index: int,
) -> Optional[List[float]]:
    boxes = item.get(bbox_key)

    if boxes is None:
        return None

    # One flat box: [x1, y1, x2, y2]
    if (
        isinstance(boxes, list)
        and len(boxes) >= 4
        and all(isinstance(x, (int, float)) for x in boxes[:4])
    ):
        return [float(x) for x in boxes[:4]]

    # List of boxes: [[x1, y1, x2, y2], ...]
    if isinstance(boxes, list) and boxes:
        if bbox_index >= len(boxes):
            return None

        selected = boxes[bbox_index]

        if isinstance(selected, dict):
            for key in ["bbox", "xyxy", "box"]:
                if key in selected:
                    selected = selected[key]
                    break

        if (
            isinstance(selected, list)
            and len(selected) >= 4
            and all(isinstance(x, (int, float)) for x in selected[:4])
        ):
            return [float(x) for x in selected[:4]]

    return None


def convert_box_to_xyxy(
    box: List[float],
    image_width: int,
    image_height: int,
    bbox_format: str,
    normalized: bool,
) -> Tuple[float, float, float, float]:
    a, b, c, d = box[:4]

    if normalized:
        if bbox_format == "xyxy":
            a *= image_width
            c *= image_width
            b *= image_height
            d *= image_height
        elif bbox_format in {"xywh", "cxcywh"}:
            a *= image_width
            c *= image_width
            b *= image_height
            d *= image_height

    if bbox_format == "xyxy":
        x1, y1, x2, y2 = a, b, c, d

    elif bbox_format == "xywh":
        x1, y1 = a, b
        x2, y2 = a + c, b + d

    elif bbox_format == "cxcywh":
        x1 = a - c / 2.0
        y1 = b - d / 2.0
        x2 = a + c / 2.0
        y2 = b + d / 2.0

    else:
        raise ValueError(f"Unsupported bbox format: {bbox_format}")

    return x1, y1, x2, y2


def expand_and_clip_box(
    box: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    crop_scale: float,
    min_crop_size: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)

    crop_width = max(box_width * crop_scale, float(min_crop_size))
    crop_height = max(box_height * crop_scale, float(min_crop_size))

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    new_x1 = center_x - crop_width / 2.0
    new_y1 = center_y - crop_height / 2.0
    new_x2 = center_x + crop_width / 2.0
    new_y2 = center_y + crop_height / 2.0

    # Shift crop back inside image while preserving size where possible.
    if new_x1 < 0:
        new_x2 -= new_x1
        new_x1 = 0

    if new_y1 < 0:
        new_y2 -= new_y1
        new_y1 = 0

    if new_x2 > image_width:
        shift = new_x2 - image_width
        new_x1 -= shift
        new_x2 = image_width

    if new_y2 > image_height:
        shift = new_y2 - image_height
        new_y1 -= shift
        new_y2 = image_height

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    ix1 = int(round(new_x1))
    iy1 = int(round(new_y1))
    ix2 = int(round(new_x2))
    iy2 = int(round(new_y2))

    if ix2 <= ix1:
        ix2 = min(image_width, ix1 + 1)

    if iy2 <= iy1:
        iy2 = min(image_height, iy1 + 1)

    return ix1, iy1, ix2, iy2


def build_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
    )

    model_kwargs = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if args.bf16 else torch.float16
            ),
        )
    else:
        model_kwargs["dtype"] = (
            torch.bfloat16 if args.bf16 else torch.float16
        )

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        local_files_only=args.local_files_only,
    )

    model.eval()

    return model, processor


def main():
    args = parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data = load_json(args.annotations_json)
    records = data[args.split]

    if args.max_samples is not None:
        records = records[:args.max_samples]

    model, processor = build_model_and_processor(args)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped_missing_bbox = 0
    used_full_image = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for item_index, item in enumerate(records):
            sample_id = item.get("id", str(item_index))

            image_rel, image_path = resolve_image_path(
                item,
                args.image_root,
            )

            if not os.path.isfile(image_path):
                print(
                    f"WARNING: missing image for {sample_id}: {image_path}",
                    flush=True,
                )
                continue

            image = safe_open_image(image_path)
            width, height = image.size

            raw_box = extract_box(
                item,
                args.bbox_key,
                args.bbox_index,
            )

            if raw_box is None:
                if args.skip_missing_bbox:
                    skipped_missing_bbox += 1
                    continue

                model_image = image
                crop_box = None
                used_full_image += 1

            else:
                xyxy_box = convert_box_to_xyxy(
                    raw_box,
                    width,
                    height,
                    args.bbox_format,
                    args.normalized_bbox,
                )

                crop_box = expand_and_clip_box(
                    xyxy_box,
                    width,
                    height,
                    args.crop_scale,
                    args.min_crop_size,
                )

                model_image = image.crop(crop_box)

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": USER_PROMPT,
                        },
                        {
                            "type": "image",
                        },
                    ],
                },
            ]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[prompt],
                images=[[model_image]],
                return_tensors="pt",
                padding=True,
            )

            model_device = next(model.parameters()).device

            inputs = {
                key: value.to(model_device)
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[:, input_length:]

            prediction = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

            reference = str(item.get("report", "")).strip()

            output_row = {
                "id": sample_id,
                "image_path": image_rel,
                "reference_report": reference,
                "reference": reference,
                "prediction": prediction,
                "split": args.split,
                "raw_bbox": raw_box,
                "crop_bbox_xyxy": list(crop_box)
                if crop_box is not None
                else None,
                "used_full_image": raw_box is None,
            }

            fout.write(
                json.dumps(output_row, ensure_ascii=False) + "\n"
            )
            fout.flush()

            processed += 1

            if processed % 100 == 0:
                print(
                    f"processed={processed}, "
                    f"skipped_missing_bbox={skipped_missing_bbox}, "
                    f"full_image_fallback={used_full_image}",
                    flush=True,
                )

    print(f"Saved predictions to: {output_path}")
    print(f"Processed: {processed}")
    print(f"Skipped missing bbox: {skipped_missing_bbox}")
    print(f"Full-image fallbacks: {used_full_image}")


if __name__ == "__main__":
    main()
