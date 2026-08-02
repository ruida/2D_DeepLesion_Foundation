#!/usr/bin/env python3
import json
from pathlib import Path
P=Path("/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text")
pred=json.load(open(P/"data/simple_dualhead_classifier/dualhead_predictions.json"))
base=json.load(open(P/"data/oracle_pair_rag_annotations/02_totalseg_plus_predicted_lesion_complete.json"))
out={}
MAP={"lung":"lung","liver":"liver","kidney":"kidney","adrenal":"adrenal","abdomen":"abdomen",
     "pelvis":"pelvis","chest":"chest","brain head neck":"brain_head_neck","brain_head_neck":"brain_head_neck",
     "spine":"spine","bone":"bone","bowel":"abdomen","pancreas":"abdomen","stomach":"abdomen",
     "vessel":"abdomen","unknown":"unknown"}
def clean(s): return " ".join(str(s or "").lower().replace("_"," ").split())
for split in ["train","val","test"]:
    idx={str(x["id"]):x for x in pred[split]}; rows=[]
    for x in base[split]:
        p=idx[str(x["id"])]; y=dict(x)
        y["raw_simple_anatomy"]=MAP.get(clean(x.get("totalseg_anatomy_name","unknown")),"unknown")
        y["raw_primary_lesion_type"]=p["predicted_primary_lesion_type"]
        y["raw_lesion_attribute"]=p["predicted_lesion_attribute"]
        y["raw_totalseg_details"]=[clean(v) for v in x.get("totalseg_detail_names",[]) if clean(v)]
        y["predicted_primary_confidence"]=p["predicted_primary_confidence"]
        y["predicted_attribute_confidence"]=p["predicted_attribute_confidence"]
        rows.append(y)
    out[split]=rows
target=P/"data/simple_pair_rag_annotations/01b_raw_simple_inputs_dualhead.json"
target.parent.mkdir(parents=True,exist_ok=True)
json.dump(out,open(target,"w"),indent=2,ensure_ascii=False)
print("Saved:",target)
