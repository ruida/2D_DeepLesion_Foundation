#!/usr/bin/env python3
import argparse, json, math, random
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

PRIMARY_CLASSES = ["lymph_node","mass","nodule","opacity","consolidation","cystic","other_lesion"]
ATTRIBUTE_CLASSES = ["low_density","enhancing","hyperdense","calcified","none"]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resolve(root, p):
    p = Path(p)
    return p if p.is_absolute() else Path(root) / p

def xyxy(box, fmt, w, h):
    b = [float(v) for v in box]

    if fmt == "xyxy":
        x1, y1, x2, y2 = b

    elif fmt == "xywh":
        x, y, bw, bh = b
        x1, y1 = x, y
        x2, y2 = x + bw, y + bh

    elif fmt == "cxcywh":
        cx, cy, bw, bh = b
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

    elif fmt == "xyxy_norm":
        x1, y1, x2, y2 = b
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h

    elif fmt == "xywh_norm":
        x, y, bw, bh = b
        x1 = x * w
        y1 = y * h
        x2 = (x + bw) * w
        y2 = (y + bh) * h

    elif fmt == "cxcywh_norm":
        cx, cy, bw, bh = b
        cx *= w
        cy *= h
        bw *= w
        bh *= h

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

    else:
        raise ValueError(f"Unsupported bbox format: {fmt}")

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    return x1, y1, x2, y2

def expand(box, w, h, r):
    x1,y1,x2,y2 = box; bw=x2-x1; bh=y2-y1
    return (max(0,int(x1-bw*r)), max(0,int(y1-bh*r)),
            min(w,int(x2+bw*r)), min(h,int(y2+bh*r)))

class CropDataset(Dataset):
    def __init__(self, rows, image_root, train, crop_size=224, expand_ratio=0.25):
        self.rows=rows; self.image_root=Path(image_root); self.expand_ratio=expand_ratio
        aug = [transforms.Resize((crop_size,crop_size))]
        if train:
            aug += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(7),
                    transforms.ColorJitter(brightness=.12, contrast=.12)]
        aug += [transforms.ToTensor(),
                transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.tf = transforms.Compose(aug)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        img=Image.open(resolve(self.image_root, r["image_path"][0])).convert("RGB")
        w,h=img.size
        box=xyxy(r["bboxes"][0], r.get("bbox_format","xyxy"), w, h)
        box=expand(box,w,h,self.expand_ratio)
        crop=img.crop(box)
        return {
            "image": self.tf(crop),
            "primary": torch.tensor(PRIMARY_CLASSES.index(r["primary_label"])),
            "attribute": torch.tensor(ATTRIBUTE_CLASSES.index(r["attribute_label"])),
            "id": r["id"],
        }

class DualHeadResNet(nn.Module):
    def __init__(self, backbone="resnet18", dropout=.2):
        super().__init__()
        if backbone=="resnet18":
            net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            net=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        d=net.fc.in_features; net.fc=nn.Identity()
        self.backbone=net; self.drop=nn.Dropout(dropout)
        self.primary_head=nn.Linear(d,len(PRIMARY_CLASSES))
        self.attribute_head=nn.Linear(d,len(ATTRIBUTE_CLASSES))
    def forward(self,x):
        z=self.drop(self.backbone(x))
        return self.primary_head(z), self.attribute_head(z)

def weights(rows,key,classes):
    c=Counter(r[key] for r in rows)
    v=torch.tensor([0 if c[k]==0 else 1/math.sqrt(c[k]) for k in classes],dtype=torch.float32)
    nz=v[v>0]
    return v/(nz.mean() if len(nz) else 1)

@torch.no_grad()
def eval_model(model,loader,device):
    model.eval(); n=po=ao=jo=0
    for b in loader:
        x=b["image"].to(device); yp=b["primary"].to(device); ya=b["attribute"].to(device)
        lp,la=model(x); pp=lp.argmax(1); pa=la.argmax(1)
        n+=len(x); po+=(pp==yp).sum().item(); ao+=(pa==ya).sum().item()
        jo+=((pp==yp)&(pa==ya)).sum().item()
    return {"primary_acc":po/n,"attribute_acc":ao/n,"joint_acc":jo/n}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_json",required=True); ap.add_argument("--image_root",required=True)
    ap.add_argument("--save_dir",required=True); ap.add_argument("--backbone",default="resnet18")
    ap.add_argument("--batch_size",type=int,default=64); ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--weight_decay",type=float,default=1e-4)
    ap.add_argument("--num_workers",type=int,default=8); ap.add_argument("--crop_size",type=int,default=224)
    ap.add_argument("--expand_ratio",type=float,default=.25)
    ap.add_argument("--attribute_loss_weight",type=float,default=.5)
    ap.add_argument("--patience",type=int,default=12); ap.add_argument("--seed",type=int,default=9223)
    args=ap.parse_args()

    seed_all(args.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=json.load(open(args.dataset_json)); tr=data["train"]; va=data["val"]
    trl=DataLoader(CropDataset(tr,args.image_root,True,args.crop_size,args.expand_ratio),
                   batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True)
    val=DataLoader(CropDataset(va,args.image_root,False,args.crop_size,args.expand_ratio),
                    batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)

    model=DualHeadResNet(args.backbone).to(device)
    cp=nn.CrossEntropyLoss(weight=weights(tr,"primary_label",PRIMARY_CLASSES).to(device))
    ca=nn.CrossEntropyLoss(weight=weights(tr,"attribute_label",ATTRIBUTE_CLASSES).to(device))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=3,factor=.5)
    sd=Path(args.save_dir); sd.mkdir(parents=True,exist_ok=True)
    best=-1; stale=0

    for epoch in range(1,args.epochs+1):
        model.train(); total=loss_sum=0
        for b in trl:
            x=b["image"].to(device); yp=b["primary"].to(device); ya=b["attribute"].to(device)
            lp,la=model(x); loss=cp(lp,yp)+args.attribute_loss_weight*ca(la,ya)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total+=len(x); loss_sum+=loss.item()*len(x)
        m=eval_model(model,val,device)
        score=.6*m["primary_acc"]+.2*m["attribute_acc"]+.2*m["joint_acc"]
        sched.step(score)
        ckpt={"epoch":epoch,"model":model.state_dict(),"primary_classes":PRIMARY_CLASSES,
              "attribute_classes":ATTRIBUTE_CLASSES,"args":vars(args),"val":m,"score":score}
        torch.save(ckpt,sd/"current_checkpoint.pth")
        print({"epoch":epoch,"train_loss":loss_sum/total,"val":m,"score":score},flush=True)
        if score>best:
            best=score; stale=0; torch.save(ckpt,sd/"model_best.pth")
            print("Saved best:",sd/"model_best.pth",flush=True)
        else:
            stale+=1
            if stale>=args.patience:
                print("Early stopping.",flush=True); break
if __name__=="__main__": main()
