#!/usr/bin/env python3
import argparse, json, os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from dam_medgemma_utils import build_dam_visuals, visual_user_content

SYSTEM_PROMPT = "You are a radiology assistant. Generate a concise lesion-focused report from the supplied CT views and spatial masks. Use short medical tag-style phrasing. Do not describe masks, crops, boxes, or annotations."
USER_PROMPT = "Generate a short lesion-focused DeepLesion report using the full view, focal view, and aligned masks."

def args_parse():
    p=argparse.ArgumentParser(); p.add_argument("--base_model",required=True); p.add_argument("--adapter_dir",required=True); p.add_argument("--annotations_json",required=True); p.add_argument("--image_root",required=True); p.add_argument("--split",default="test",choices=["train","val","test"]); p.add_argument("--output_jsonl",required=True); p.add_argument("--crop_scale",type=float,default=3.0); p.add_argument("--min_crop_size",type=int,default=48); p.add_argument("--visual_mode",choices=["four_image","two_overlay"],default="four_image"); p.add_argument("--bf16",action="store_true"); p.add_argument("--use_4bit",action="store_true"); p.add_argument("--local_files_only",action="store_true"); p.add_argument("--max_new_tokens",type=int,default=64); p.add_argument("--max_samples",type=int); return p.parse_args()

def main():
    a=args_parse(); data=json.load(open(a.annotations_json)); rows=data[a.split][:a.max_samples] if a.max_samples else data[a.split]
    proc=AutoProcessor.from_pretrained(a.base_model,local_files_only=a.local_files_only)
    kw={"local_files_only":a.local_files_only,"trust_remote_code":True,"device_map":"auto"}
    if a.use_4bit: kw["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16 if a.bf16 else torch.float16)
    else: kw["dtype"]=torch.bfloat16 if a.bf16 else torch.float16
    model=PeftModel.from_pretrained(AutoModelForImageTextToText.from_pretrained(a.base_model,**kw),a.adapter_dir,local_files_only=a.local_files_only); model.eval()
    os.makedirs(os.path.dirname(a.output_jsonl) or ".",exist_ok=True)
    with open(a.output_jsonl,"w") as out:
        for i,item in enumerate(rows):
            images,meta=build_dam_visuals(item,a.image_root,a.crop_scale,a.min_crop_size,a.visual_mode)
            msg=[{"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},{"role":"user","content":visual_user_content(USER_PROMPT,a.visual_mode)}]
            prompt=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
            inputs=proc(text=[prompt],images=[images],return_tensors="pt",padding=True)
            dev=next(model.parameters()).device; inputs={k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inputs.items()}
            with torch.no_grad(): output=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False)
            pred=proc.batch_decode(output[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip()
            out.write(json.dumps({"id":item["id"],"reference_report":item["report"],"prediction":pred,"split":a.split,**meta},ensure_ascii=False)+"\n")
            if (i+1)%100==0: print(f"processed {i+1}/{len(rows)}")
    print("Saved",a.output_jsonl)
if __name__=="__main__": main()
