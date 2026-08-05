#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    "Do not mention training artifacts, token IDs, or image annotations."
)


BASE_USER_PROMPT = (
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

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
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

    # New anatomy/lesion conditioning options.
    parser.add_argument(
        "--condition_style",
        type=str,
        default="names",
        choices=["names", "names_ids", "ids_only"],
        help="How to add anatomy and lesion-type context to the prompt.",
    )
    parser.add_argument(
        "--use_structured_context",
        action="store_true",
        help="Enable anatomy and lesion-type conditioning in the prompt.",
    )
    parser.add_argument(
        "--drop_context_prob",
        type=float,
        default=0.0,
        help="Randomly drop structured context during training. Use 0.0 for always-on context.",
    )

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


def extract_anat_lesion_context(item: Dict[str, Any]) -> Tuple[int, str, int, str]:
    """
    Expected final v3 JSON format:
      rough_anatomy_ids:   [anatomy_id, lesion_type_id]
      rough_anatomy_names: [anatomy_name, lesion_type:<name>]

    Example:
      [11, 105]
      ["pancreas", "lesion_type:mass_or_nodule"]
    """
    ids = item.get("rough_anatomy_ids", [])
    names = item.get("rough_anatomy_names", [])

    anatomy_id = -1
    anatomy_name = "unknown"
    lesion_type_id = -1
    lesion_type_name = "unknown"

    if isinstance(ids, list) and len(ids) >= 1:
        try:
            anatomy_id = int(ids[0])
        except Exception:
            anatomy_id = -1

    if isinstance(names, list) and len(names) >= 1:
        anatomy_name = str(names[0]).strip()
        if not anatomy_name:
            anatomy_name = "unknown"

    if isinstance(ids, list) and len(ids) >= 2:
        try:
            lesion_type_id = int(ids[1])
        except Exception:
            lesion_type_id = -1

    if isinstance(names, list) and len(names) >= 2:
        lesion_type_name = str(names[1]).strip()
        lesion_type_name = lesion_type_name.replace("lesion_type:", "").strip()
        if not lesion_type_name:
            lesion_type_name = "unknown"

    return anatomy_id, anatomy_name, lesion_type_id, lesion_type_name


def build_condition_prompt(
    item: Dict[str, Any],
    use_structured_context: bool,
    condition_style: str,
    drop_context_prob: float,
    is_train: bool,
) -> str:
    """
    Build user prompt.

    If use_structured_context=False:
      original image-only prompt.

    If use_structured_context=True:
      add anatomy and lesion-type context from JSON.
    """
    if not use_structured_context:
        return BASE_USER_PROMPT

    if is_train and drop_context_prob > 0.0:
        if random.random() < drop_context_prob:
            return BASE_USER_PROMPT

    anatomy_id, anatomy_name, lesion_type_id, lesion_type_name = extract_anat_lesion_context(item)

    if condition_style == "names":
        context = (
            "Use the following structured lesion context if it is consistent with the image:\n"
            f"- anatomy: {anatomy_name}\n"
            f"- lesion type: {lesion_type_name}"
        )
    elif condition_style == "names_ids":
        context = (
            "Use the following structured lesion context if it is consistent with the image:\n"
            f"- anatomy ID: {anatomy_id}\n"
            f"- anatomy: {anatomy_name}\n"
            f"- lesion type ID: {lesion_type_id}\n"
            f"- lesion type: {lesion_type_name}"
        )
    elif condition_style == "ids_only":
        context = (
            "Use the following structured lesion context if it is consistent with the image:\n"
            f"- anatomy ID: {anatomy_id}\n"
            f"- lesion type ID: {lesion_type_id}"
        )
    else:
        raise ValueError(f"Unknown condition_style: {condition_style}")

    prompt = (
        f"{BASE_USER_PROMPT}\n\n"
        f"{context}"
    )

    return prompt


def print_split_stats(train_records: List[Dict[str, Any]], val_records: List[Dict[str, Any]]) -> None:
    def avg_len(recs: List[Dict[str, Any]]) -> float:
        if not recs:
            return 0.0
        return sum(len(str(x["report"]).split()) for x in recs) / len(recs)

    def context_ok(recs: List[Dict[str, Any]]) -> int:
        ok = 0
        for x in recs:
            ids = x.get("rough_anatomy_ids", [])
            names = x.get("rough_anatomy_names", [])
            if isinstance(ids, list) and isinstance(names, list) and len(ids) >= 2 and len(names) >= 2:
                ok += 1
        return ok

    print(f"train samples: {len(train_records)}")
    print(f"val samples:   {len(val_records)}")
    print(f"avg train report words: {avg_len(train_records):.1f}")
    print(f"avg val report words:   {avg_len(val_records):.1f}")
    print(f"train samples with anatomy+lesion context: {context_ok(train_records)}")
    print(f"val samples with anatomy+lesion context:   {context_ok(val_records)}")


class DeepLesionAnatLesionDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        image_root: str,
        use_structured_context: bool,
        condition_style: str,
        drop_context_prob: float = 0.0,
        is_train: bool = False,
    ) -> None:
        self.records = records
        self.image_root = image_root
        self.use_structured_context = use_structured_context
        self.condition_style = condition_style
        self.drop_context_prob = drop_context_prob
        self.is_train = is_train

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

        user_prompt = build_condition_prompt(
            item=item,
            use_structured_context=self.use_structured_context,
            condition_style=self.condition_style,
            drop_context_prob=self.drop_context_prob,
            is_train=self.is_train,
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": report}],
            },
        ]

        anatomy_id, anatomy_name, lesion_type_id, lesion_type_name = extract_anat_lesion_context(item)

        return {
            "id": item["id"],
            "image": image,
            "messages": messages,
            "report": report,
            "anatomy_id": anatomy_id,
            "anatomy_name": anatomy_name,
            "lesion_type_id": lesion_type_id,
            "lesion_type_name": lesion_type_name,
        }


@dataclass
class MedGemmaAnatLesionCollator:
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
        # Some transformers versions use torch_dtype, newer ones use dtype.
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16

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


def save_run_config(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved run config to: {path}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    save_run_config(args)

    if torch.cuda.is_available() and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data = load_json(args.annotations_json)

    train_records = maybe_limit(get_split(data, "train"), args.max_train_samples)
    val_records = maybe_limit(get_split(data, "val"), args.max_eval_samples)

    print_split_stats(train_records, val_records)

    print("Structured context:", args.use_structured_context)
    print("Condition style:", args.condition_style)
    print("Drop context probability:", args.drop_context_prob)

    train_dataset = DeepLesionAnatLesionDataset(
        records=train_records,
        image_root=args.image_root,
        use_structured_context=args.use_structured_context,
        condition_style=args.condition_style,
        drop_context_prob=args.drop_context_prob,
        is_train=True,
    )

    val_dataset = DeepLesionAnatLesionDataset(
        records=val_records,
        image_root=args.image_root,
        use_structured_context=args.use_structured_context,
        condition_style=args.condition_style,
        drop_context_prob=0.0,
        is_train=False,
    )

    # Print one prompt for sanity check.
    example = train_dataset[0]
    print("\n========== Example user prompt ==========")
    for msg in example["messages"]:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content["type"] == "text":
                    print(content["text"])
    print("========== Example target report ==========")
    print(example["report"])
    print("==========================================\n")

    model, processor = build_model_and_processor(args)
    collator = MedGemmaAnatLesionCollator(processor=processor, max_length=args.max_length)

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
