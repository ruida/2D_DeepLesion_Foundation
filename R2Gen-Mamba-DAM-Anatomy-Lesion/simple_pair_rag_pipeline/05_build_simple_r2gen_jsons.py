#!/usr/bin/env python3
import json
from pathlib import Path
P=Path('/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text'); I=P/'data/simple_pair_rag_annotations/03_simple_pair_rag_applied.json'; OO=P/'data/simple_pair_rag_annotations/04_oracle_simple_r2gen.json'; RO=P/'data/simple_pair_rag_annotations/05_simple_pair_rag_r2gen_ready.json'; WO=P/'data/simple_pair_rag_annotations/06_raw_simple_r2gen_ready.json'; d=json.load(open(I))
def toks(a,p,t): return str(a).replace('_',' ').split()+str(p).replace('_',' ').split()+([] if t=='none' else str(t).replace('_',' ').split())
objs=[{}, {}, {}]
for split in ['train','val','test']:
    outs=[[],[],[]]
    for x in d[split]:
        vals=[(x['oracle_simple_anatomy'],x['oracle_primary_lesion_type'],x['oracle_lesion_attribute'],'oracle_simple'),(x['rag_mapped_anatomy'],x['rag_mapped_primary_lesion_type'],x['rag_mapped_lesion_attribute'],'simple_pair_rag'),(x['raw_simple_anatomy'],x['raw_primary_lesion_type'],x['raw_lesion_attribute'],'raw_simple')]
        for i,(a,p,t,src) in enumerate(vals):
            y=dict(x); ts=toks(a,p,t); y['rough_anatomy_names']=ts; y['rough_anatomy_name']=' '.join(ts); y['anatomy_text']=' '.join(ts); y['anatomy_source']=src; outs[i].append(y)
    for i in range(3): objs[i][split]=outs[i]
for path,obj in [(OO,objs[0]),(RO,objs[1]),(WO,objs[2])]: path.parent.mkdir(parents=True,exist_ok=True); json.dump(obj,open(path,'w'),indent=2,ensure_ascii=False); print('Saved:',path)
