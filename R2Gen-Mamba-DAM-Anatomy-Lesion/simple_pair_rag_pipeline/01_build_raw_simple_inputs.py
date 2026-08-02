#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter
PROJECT=Path('/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text')
INPUT=PROJECT/'data/oracle_pair_rag_annotations/02_totalseg_plus_predicted_lesion_complete.json'
OUTPUT=PROJECT/'data/simple_pair_rag_annotations/01_raw_simple_inputs.json'
SUMMARY=PROJECT/'data/simple_pair_rag_annotations/01_raw_simple_inputs_summary.json'
AM={'lung':'lung','liver':'liver','kidney':'kidney','adrenal':'adrenal','abdomen':'abdomen','pelvis':'pelvis','chest':'chest','brain head neck':'brain_head_neck','brain_head_neck':'brain_head_neck','spine':'spine','bone':'bone','bowel':'abdomen','pancreas':'abdomen','stomach':'abdomen','vessel':'abdomen','unknown':'unknown'}
PM={'lymph node':'lymph_node','mass or nodule':'mass','mass_or_nodule':'mass','low density or cystic':'cystic','low_density_or_cystic':'cystic','enhancing or hyperdense':'other_lesion','enhancing_or_hyperdense':'other_lesion','opacity or consolidation':'opacity','opacity_or_consolidation':'opacity','calcified lesion':'other_lesion','metastatic lesion':'other_lesion','other lesion':'other_lesion'}
TM={'low density or cystic':'low_density','low_density_or_cystic':'low_density','enhancing or hyperdense':'enhancing','enhancing_or_hyperdense':'enhancing','calcified lesion':'calcified'}
def clean(s): return ' '.join(str(s or '').lower().replace('_',' ').split())
def uniq(vs):
    o=[]; seen=set()
    for v in vs or []:
        z=clean(v)
        if z and z not in seen: seen.add(z); o.append(z)
    return o
data=json.load(open(INPUT)); out={}; summary={}
for split in ['train','val','test']:
    rows=[]; ca=Counter(); cp=Counter(); ct=Counter()
    for x in data[split]:
        ra=clean(x.get('totalseg_anatomy_name','unknown')); rc=str(x.get('predicted_lesion_type_name','other lesion')).strip().lower(); cc=clean(rc)
        a=AM.get(ra,'unknown'); p=PM.get(rc,PM.get(cc,'other_lesion')); t=TM.get(rc,TM.get(cc,'none'))
        y=dict(x); y['raw_simple_anatomy']=a; y['raw_primary_lesion_type']=p; y['raw_lesion_attribute']=t; y['raw_totalseg_details']=uniq(x.get('totalseg_detail_names',[])); y['classifier_confidence']=float(x.get('predicted_lesion_type_confidence',0.0) or 0.0)
        rows.append(y); ca[a]+=1; cp[p]+=1; ct[t]+=1
    out[split]=rows; summary[split]={'n':len(rows),'anatomy':ca.most_common(),'primary':cp.most_common(),'attribute':ct.most_common()}
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
json.dump(out,open(OUTPUT,'w'),indent=2,ensure_ascii=False); json.dump(summary,open(SUMMARY,'w'),indent=2,ensure_ascii=False)
print('Saved:',OUTPUT)
