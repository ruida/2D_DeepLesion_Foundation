#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from importlib.util import spec_from_file_location, module_from_spec

HERE=Path(__file__).resolve().parent
spec=spec_from_file_location("trainmod",HERE/"01_train_dualhead_classifier.py")
m=module_from_spec(spec); spec.loader.exec_module(m)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_json",required=True); ap.add_argument("--image_root",required=True)
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--out_json",required=True)
    ap.add_argument("--batch_size",type=int,default=64); ap.add_argument("--num_workers",type=int,default=8)
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(args.checkpoint,map_location="cpu")
    model=m.DualHeadResNet(ckpt["args"].get("backbone","resnet18"))
    model.load_state_dict(ckpt["model"]); model.to(device).eval()
    data=json.load(open(args.dataset_json)); out={}

    with torch.no_grad():
        for split in ["train","val","test"]:
            ds=m.CropDataset(data[split],args.image_root,False,
                             ckpt["args"].get("crop_size",224),
                             ckpt["args"].get("expand_ratio",.25))
            dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,
                          num_workers=args.num_workers,pin_memory=True)
            pi={}
            for b in tqdm(dl,desc=split):
                lp,la=model(b["image"].to(device))
                pp=torch.softmax(lp,1); pa=torch.softmax(la,1)
                pidx=pp.argmax(1).cpu(); aidx=pa.argmax(1).cpu()
                pc=pp.max(1).values.cpu(); ac=pa.max(1).values.cpu()
                for sid,ip,ia,cp,ca in zip(b["id"],pidx,aidx,pc,ac):
                    pi[str(sid)]={
                        "predicted_primary_lesion_type":ckpt["primary_classes"][int(ip)],
                        "predicted_primary_confidence":float(cp),
                        "predicted_lesion_attribute":ckpt["attribute_classes"][int(ia)],
                        "predicted_attribute_confidence":float(ca),
                    }
            out[split]=[]
            for r in data[split]:
                y=dict(r); y.update(pi[str(r["id"])]); out[split].append(y)

    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    json.dump(out,open(args.out_json,"w"),indent=2,ensure_ascii=False)
    print("Saved:",args.out_json)
if __name__=="__main__": main()
