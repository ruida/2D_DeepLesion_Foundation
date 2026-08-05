#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--annotations_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_split(data: Dict[str, Any], split_name: str) -> List[Dict[str, Any]]:
    if split_name not in data:
        raise KeyError(f"Split '{split_name}' not found. Available: {list(data.keys())}")
    split = data[split_name]
    if not isinstance(split, list):
        raise ValueError(f"Split '{split_name}' must be a list.")
    return split


def maybe_limit(records: List[Dict[str, Any]], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    if max_samples is None:
        return records
    return records[:max_samples]


def print_split_stats(train_records: List[Dict[str, Any]], val_records: List[Dict[str, Any]]) -> None:
    def avg_len(recs: List[Dict[str, Any]]) -> float:
        if not recs:
            return 0.0
        return sum(len(str(x["report"]).split()) for x in recs) / len(recs)

    print(f"train samples: {len(train_records)}")
    print(f"val samples:   {len(val_records)}")
    print(f"avg train report words: {avg_len(train_records):.1f}")
    print(f"avg val report words:   {avg_len(val_records):.1f}")


class DeepLesionShortDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], image_root: str) -> None:
        self.records = records
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]

        image_rel = item["image_path"][0]
        image_path = image_rel if os.path.isabs(image_rel) else os.path.join(self.image_root, image_rel)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = safe_open_image(image_path)
        report = str(item["report"]).strip()

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
            {
                "role": "assistant",
                "content": [{"type": "text", "text": report}],
            },
        ]

        return {
            "id": item["id"],
            "image": image,
            "messages": messages,
            "report": report,
        }


@dataclass
class MedGemmaShortCollator:
    processor: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(
                f["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for f in features
        ]

        images = [[f["image"]] for f in features]

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        image_token_id = getattr(self.processor.tokenizer, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch


def build_model_and_processor(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }

    if args.use_4bit:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"
        model_kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if torch.cuda.is_available() and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data = load_json(args.annotations_json)

    train_records = maybe_limit(get_split(data, "train"), args.max_train_samples)
    val_records = maybe_limit(get_split(data, "val"), args.max_eval_samples)

    print_split_stats(train_records, val_records)

    train_dataset = DeepLesionShortDataset(train_records, args.image_root)
    val_dataset = DeepLesionShortDataset(val_records, args.image_root)

    model, processor = build_model_and_processor(args)
    collator = MedGemmaShortCollator(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"Saved model and processor to: {args.output_dir}")


if __name__ == "__main__":
    main()
