#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter
PROJECT=Path('/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text')
INPUT=PROJECT/'data/generated_annotations/merged_anatomy_plus_lesion_r2gen/deeplesion_merged_anatomy_plus_oracle_lesion_r2gen.json'
OUTPUT=PROJECT/'data/simple_pair_rag_annotations/00_oracle_simple_labels.json'
SUMMARY=PROJECT/'data/simple_pair_rag_annotations/00_oracle_simple_labels_summary.json'
ANATOMY_RULES=[('adrenal',['adrenal gland','left adrenal','right adrenal','adrenal']),('kidney',['left kidney','right kidney','kidney','renal']),('liver',['liver','hepatic']),('lung',['left lung','right lung','upper lung','lower lung','lung','pulmonary']),('pelvis',['pelvis','pelvic','iliac']),('chest',['chest wall','mediastinum','mediastinal','hilum','hilar','axilla','axillary','chest']),('brain_head_neck',['brain','skull','head','neck']),('spine',['vertebra','vertebral','spine','lumbar','thoracic vertebra','cervical vertebra']),('bone',['rib','femur','humerus','scapula','bone','osseous']),('abdomen',['retroperitoneum','pancreas','pancreatic','bowel','colon','stomach','abdomen','abdominal'])]
PRIMARY_RULES = [
    ("lymph_node", [
        "lymph node",
        "lymphadenopathy",
        "nodal",
    ]),
    ("cystic", [
        "cystic",
        "cyst",
    ]),
    ("nodule", [
        "nodule",
        "nodular",
    ]),
    ("mass", [
        "mass",
        "tumor",
    ]),
    ("consolidation", [
        "consolidation",
        "consolidative",
    ]),
    ("opacity", [
        "ground glass",
        "ground-glass",
        "opacity",
        "reticular",
    ]),
]

ATTRIBUTE_RULES=[('calcified',['calcified','calcification']),('enhancing',['enhancing','enhancement']),('hyperdense',['hyperdense','hyperattenuating','high density','high attenuation']),('low_density',['hypoattenuating','hypoattenuation','hypodense','low density','low attenuation'])]
def norm(s): return ' '.join(str(s or '').lower().replace('_',' ').split())
def first(text,rules,default):
    for label,kws in rules:
        for kw in kws:
            if kw in text: return label
    return default
data=json.load(open(INPUT)); out={}; summary={}
for split in ['train','val','test']:
    rows=[]; ca=Counter(); cp=Counter(); ct=Counter()
    for x in data[split]:
        report=norm(x.get('report',''))
        a=first(report,ANATOMY_RULES,'unknown'); p=first(report,PRIMARY_RULES,'other_lesion'); t=first(report,ATTRIBUTE_RULES,'none')
        y=dict(x); y['oracle_simple_anatomy']=a; y['oracle_primary_lesion_type']=p; y['oracle_lesion_attribute']=t
        y['oracle_simple_tokens']=a.replace('_',' ').split()+p.replace('_',' ').split()+([] if t=='none' else t.replace('_',' ').split())
        rows.append(y); ca[a]+=1; cp[p]+=1; ct[t]+=1
    out[split]=rows; summary[split]={'n':len(rows),'anatomy':ca.most_common(),'primary':cp.most_common(),'attribute':ct.most_common()}
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
json.dump(out,open(OUTPUT,'w'),indent=2,ensure_ascii=False); json.dump(summary,open(SUMMARY,'w'),indent=2,ensure_ascii=False)
print('Saved:',OUTPUT); print('Summary:',SUMMARY)
