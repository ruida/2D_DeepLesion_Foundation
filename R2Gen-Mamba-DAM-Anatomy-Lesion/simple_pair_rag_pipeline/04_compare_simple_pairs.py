#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter
P=Path('/vf/users/ruida/LLM/R2Gen-Mamba-Merged_text'); I=P/'data/simple_pair_rag_annotations/03_simple_pair_rag_applied.json'; O=P/'results/simple_pair_rag/comparison.json'; d=json.load(open(I)); s={}
for split in ['val','test']:
    r=d[split]; n=len(r)
    def acc(a,b): return sum(str(x[a])==str(x[b]) for x in r)/n
    s[split]={'raw':{'anatomy':acc('raw_simple_anatomy','oracle_simple_anatomy'),'primary':acc('raw_primary_lesion_type','oracle_primary_lesion_type'),'attribute':acc('raw_lesion_attribute','oracle_lesion_attribute'),'anatomy_primary_joint':sum((x['raw_simple_anatomy'],x['raw_primary_lesion_type'])==(x['oracle_simple_anatomy'],x['oracle_primary_lesion_type']) for x in r)/n,'full_joint':sum((x['raw_simple_anatomy'],x['raw_primary_lesion_type'],x['raw_lesion_attribute'])==(x['oracle_simple_anatomy'],x['oracle_primary_lesion_type'],x['oracle_lesion_attribute']) for x in r)/n},'rag':{'anatomy':acc('rag_mapped_anatomy','oracle_simple_anatomy'),'primary':acc('rag_mapped_primary_lesion_type','oracle_primary_lesion_type'),'attribute':acc('rag_mapped_lesion_attribute','oracle_lesion_attribute'),'anatomy_primary_joint':sum((x['rag_mapped_anatomy'],x['rag_mapped_primary_lesion_type'])==(x['oracle_simple_anatomy'],x['oracle_primary_lesion_type']) for x in r)/n,'full_joint':sum((x['rag_mapped_anatomy'],x['rag_mapped_primary_lesion_type'],x['rag_mapped_lesion_attribute'])==(x['oracle_simple_anatomy'],x['oracle_primary_lesion_type'],x['oracle_lesion_attribute']) for x in r)/n},'methods':Counter(x.get('rag_method','unknown') for x in r)}
O.parent.mkdir(parents=True,exist_ok=True); json.dump(s,open(O,'w'),indent=2,ensure_ascii=False); print(json.dumps(s,indent=2)); print('Saved:',O)
