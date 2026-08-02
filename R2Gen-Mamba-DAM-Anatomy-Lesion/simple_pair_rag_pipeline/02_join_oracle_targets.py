#!/usr/bin/env python3
import json
from pathlib import Path
P=Path('/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text')
RAW=P/'data/simple_pair_rag_annotations/01_raw_simple_inputs.json'; ORACLE=P/'data/simple_pair_rag_annotations/00_oracle_simple_labels.json'; OUT=P/'data/simple_pair_rag_annotations/02_simple_rag_canonical.json'
r=json.load(open(RAW)); o=json.load(open(ORACLE)); out={}
for split in ['train','val','test']:
    idx={str(x['id']):x for x in o[split]}; rows=[]
    for x in r[split]:
        z=idx[str(x['id'])]; y=dict(x)
        for k in ['oracle_simple_anatomy','oracle_primary_lesion_type','oracle_lesion_attribute','oracle_simple_tokens']: y[k]=z[k]
        rows.append(y)
    out[split]=rows; print(split,len(rows))
OUT.parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(OUT,'w'),indent=2,ensure_ascii=False); print('Saved:',OUT)
