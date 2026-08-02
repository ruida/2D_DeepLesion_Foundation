#!/usr/bin/env python3
import argparse, json
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pred_json",required=True); a=ap.parse_args()
    d=json.load(open(a.pred_json))
    for split in ["val","test"]:
        r=d[split]; n=len(r)
        p=sum(x["predicted_primary_lesion_type"]==x["primary_label"] for x in r)/n
        t=sum(x["predicted_lesion_attribute"]==x["attribute_label"] for x in r)/n
        j=sum((x["predicted_primary_lesion_type"],x["predicted_lesion_attribute"])==
              (x["primary_label"],x["attribute_label"]) for x in r)/n
        print(split,{"n":n,"primary_accuracy":p,"attribute_accuracy":t,"joint_accuracy":j})
if __name__=="__main__": main()
