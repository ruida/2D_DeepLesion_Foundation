#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from dam_medgemma_utils import build_dam_visuals, visual_user_content

SYSTEM_PROMPT = (
    "You are a radiology assistant. Generate a concise lesion-focused report from the supplied CT views and spatial masks. "
    "The full view provides anatomy and the focal view provides lesion detail. Use short medical tag-style phrasing. "
    "Do not describe the mask, crop, bounding box, or annotations."
)
USER_PROMPT = "Generate a short lesion-focused DeepLesion report using the full view, focal view, and aligned masks."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--annotations_json", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=9223)
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--max_train_samples", type=int)
    p.add_argument("--max_eval_samples", type=int)
    p.add_argument("--crop_scale", type=float, default=3.0)
    p.add_argument("--min_crop_size", type=int, default=48)
    p.add_argument("--visual_mode", choices=["four_image", "two_overlay"], default="four_image")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    return p.parse_args()


class DAMDataset(Dataset):
    def __init__(self, records, image_root, crop_scale, min_crop_size, visual_mode):
        self.records = records
        self.image_root = image_root
        self.crop_scale = crop_scale
        self.min_crop_size = min_crop_size
        self.visual_mode = visual_mode

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        images, meta = build_dam_visuals(item, self.image_root, self.crop_scale, self.min_crop_size, self.visual_mode)
        report = str(item["report"]).strip()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": visual_user_content(USER_PROMPT, self.visual_mode)},
            {"role": "assistant", "content": [{"type": "text", "text": report}]},
        ]
        return {"id": item["id"], "images": images, "messages": messages, "report": report, "meta": meta}


@dataclass
class DAMCollator:
    processor: Any
    max_length: int

    def __call__(self, features):
        texts = [self.processor.apply_chat_template(f["messages"], tokenize=False, add_generation_prompt=False) for f in features]
        images = [f["images"] for f in features]
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        pad = self.processor.tokenizer.pad_token_id
        if pad is not None: labels[labels == pad] = -100
        image_id = getattr(self.processor.tokenizer, "image_token_id", None)
        if image_id is not None: labels[labels == image_id] = -100
        batch["labels"] = labels
        return batch


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if torch.cuda.is_available() and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    data = json.load(open(args.annotations_json))
    train = data["train"][:args.max_train_samples] if args.max_train_samples else data["train"]
    val = data["val"][:args.max_eval_samples] if args.max_eval_samples else data["val"]
    print(f"train={len(train)} val={len(val)} visual_mode={args.visual_mode}")

    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    kwargs = {"trust_remote_code": True, "local_files_only": args.local_files_only, "device_map": "auto"}
    if args.use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    else:
        kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(args.model_name, **kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"): model.config.use_cache = False
    if args.use_4bit: model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.output_dir, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, weight_decay=args.weight_decay, num_train_epochs=args.num_train_epochs, warmup_steps=args.warmup_steps, logging_steps=args.logging_steps, save_steps=args.save_steps, eval_steps=args.eval_steps, eval_strategy="steps", save_strategy="steps", save_total_limit=args.save_total_limit, bf16=args.bf16, fp16=not args.bf16, dataloader_num_workers=args.dataloader_num_workers, remove_unused_columns=False, report_to="tensorboard", lr_scheduler_type="cosine", optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch", max_grad_norm=1.0),
        train_dataset=DAMDataset(train, args.image_root, args.crop_scale, args.min_crop_size, args.visual_mode),
        eval_dataset=DAMDataset(val, args.image_root, args.crop_scale, args.min_crop_size, args.visual_mode),
        data_collator=DAMCollator(processor, args.max_length),
        processing_class=processor,
    )
    trainer.train(); trainer.save_model(args.output_dir); processor.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "dam_config.json"), "w") as f:
        json.dump({"crop_scale": args.crop_scale, "min_crop_size": args.min_crop_size, "visual_mode": args.visual_mode}, f, indent=2)

if __name__ == "__main__": main()
