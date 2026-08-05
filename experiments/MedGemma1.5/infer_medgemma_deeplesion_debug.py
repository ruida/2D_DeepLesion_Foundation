#!/usr/bin/env python3
import os
import json
import argparse
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are a radiology assistant. "
    "Generate a concise lesion-focused report from this CT key slice."
)

USER_PROMPT = "Generate a short lesion-focused report for this DeepLesion CT key slice."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--annotations_json", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data[args.split][: args.max_samples]

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        use_fast=False,
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
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16

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

    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pad_token_id = processor.tokenizer.pad_token_id
    eos_token_id = processor.tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for i, item in enumerate(records, 1):
            rel_path = item["image_path"][0] if isinstance(item["image_path"], list) else item["image_path"]
            image_path = os.path.join(args.image_root, rel_path)
            image = Image.open(image_path).convert("RGB")

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

            model_inputs = processor(
                text=[prompt],
                images=[[image]],
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {
                k: v.to(next(model.parameters()).device) for k, v in model_inputs.items()
            }

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    do_sample=False,
                )

            input_len = model_inputs["input_ids"].shape[1]
            new_ids = generated_ids[:, input_len:]
            pred = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

            ref = item["report"]

            out = {
                "id": item["id"],
                "image_path": item["image_path"],
                "reference_report": ref,
                "prediction": pred,
                "split": args.split,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"{i}/{len(records)} {item['id']}")
            print("  reference :", repr(ref))
            print("  prediction:", repr(pred))
            print("-" * 80)

    print(f"Saved predictions to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
