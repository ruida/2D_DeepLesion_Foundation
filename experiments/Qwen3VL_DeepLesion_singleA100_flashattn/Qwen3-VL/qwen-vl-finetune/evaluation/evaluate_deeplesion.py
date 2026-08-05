#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def normalize_tags(text):
    parts = [p.strip().lower() for p in text.split(',')]
    return [p for p in parts if p]


def tag_score(pred, ref):
    p, r = set(normalize_tags(pred)), set(normalize_tags(ref))
    inter = len(p & r)
    prec = inter / len(p) if p else 0.0
    rec = inter / len(r) if r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    exact = 1.0 if p == r else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "exact": exact}


def text_score(pred, ref):
    pred_tokens = normalize_tags(pred)
    ref_tokens = normalize_tags(ref)

    smooth = SmoothingFunction().method1

    bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth) if pred_tokens and ref_tokens else 0.0
    bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth) if pred_tokens and ref_tokens else 0.0
    bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth) if pred_tokens and ref_tokens else 0.0
    bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) if pred_tokens and ref_tokens else 0.0

    meteor = meteor_score([ref_tokens], pred_tokens) if pred_tokens and ref_tokens else 0.0

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_l = scorer.score(", ".join(ref_tokens), ", ".join(pred_tokens))['rougeL'].fmeasure if pred_tokens and ref_tokens else 0.0

    return {
        "BLEU_1": bleu1,
        "BLEU_2": bleu2,
        "BLEU_3": bleu3,
        "BLEU_4": bleu4,
        "METEOR": meteor,
        "ROUGE_L": rouge_l,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--test_json', required=True)
    ap.add_argument('--output_json', required=True)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation='sdpa'
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()

    data = json.load(open(args.test_json))
    print(f"loaded {len(data)} test samples from {args.test_json}", flush=True)

    results = []
    agg = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "exact": 0.0,
        "BLEU_1": 0.0,
        "BLEU_2": 0.0,
        "BLEU_3": 0.0,
        "BLEU_4": 0.0,
        "METEOR": 0.0,
        "ROUGE_L": 0.0,
    }

    for idx, item in enumerate(data):
        image_list = item.get('image', [])
        prompt = item['conversations'][0]['value']
        ref = item['conversations'][1]['value']

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in image_list] +
                       [{"type": "text", "text": prompt.replace('<image>', '').strip()}]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image_list], padding=True, return_tensors='pt')
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs['input_ids'], gen)]
        pred = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        sc1 = tag_score(pred, ref)
        sc2 = text_score(pred, ref)
        sc = {**sc1, **sc2}

        for k in agg:
            agg[k] += sc[k]

        results.append({
            "id": item.get('id', idx),
            "prediction": pred,
            "reference": ref,
            **sc
        })

        if (idx + 1) == 1 or (idx + 1) % 10 == 0:
            print(
                f"processed {idx + 1}/{len(data)} | "
                f"id={item.get('id', idx)} | "
                f"pred={pred[:120]}",
                flush=True,
            )

    n = max(len(results), 1)
    summary = {k: v / n for k, v in agg.items()}
    payload = {"summary": summary, "results": results}

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, 'w'), indent=2)

    print(f"saved predictions to {out}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
