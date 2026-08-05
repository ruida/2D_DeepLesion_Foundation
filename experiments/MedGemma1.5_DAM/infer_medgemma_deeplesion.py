#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Any, Dict

import torch
from PIL import Image, ImageFile

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

SYSTEM_PROMPT = (
    "You are a radiology assistant. "
    "Generate a concise lesion-focused report from this CT key slice. "
    "Use short medical tag-style phrasing consistent with the target examples. "
    "Do not mention training artifacts or image annotations."
)

USER_PROMPT = (
    "Generate a short lesion-focused report for this DeepLesion CT key slice."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--annotations_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


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
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16

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

    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(records):
            image_rel = item["image_path"][0]
            image_path = image_rel if os.path.isabs(image_rel) else os.path.join(args.image_root, image_rel)
            image = safe_open_image(image_path)

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image"},
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
                images=[[image]],
                return_tensors="pt",
                padding=True,
            )

            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[:, input_len:]
            pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            out = {
                "id": item["id"],
                "image_path": image_rel,
                "reference_report": item["report"],
                "prediction": pred,
                "split": args.split,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            if (idx + 1) % 100 == 0:
                print(f"processed {idx + 1}/{len(records)}")

    print(f"Saved predictions to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
