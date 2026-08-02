#!/usr/bin/env python3
import argparse, copy, json, re
from pathlib import Path

ANAT_CANDS = [
    'rag_query_anatomy','predicted_oracle_anatomy_name','mapped_anatomy_name',
    'totalseg_anatomy','totalseg_anatomy_name','raw_totalseg_anatomy',
    'rough_anatomy_name','anatomy_text'
]
LESION_CANDS = [
    'rag_query_lesion_type','predicted_lesion_type','predicted_lesion_type_name',
    'lesion_type_pred','lesion_type_text'
]
DETAIL_CANDS = [
    'totalseg_detail_names','totalseg_anatomy_names','raw_totalseg_anatomy_names',
    'rough_anatomy_names','anatomy_names'
]
ORACLE_ANAT_CANDS = [
    'oracle_anatomy_name','oracle_anatomy','oracle_anatomy_text',
    'anatomy_text','rough_anatomy_name'
]
ORACLE_LESION_CANDS = [
    'oracle_lesion_type','oracle_lesion_type_name','oracle_lesion_type_text',
    'lesion_type_text'
]
CONF_CANDS = ['lesion_type_confidence','predicted_lesion_confidence','classifier_confidence','confidence']


def norm(x):
    if x is None: return ''
    if isinstance(x, list): x = ' '.join(map(str,x))
    x = str(x).strip().lower().replace('_',' ')
    x = re.sub(r'\blesion\s+type\b','',x)
    x = re.sub(r'\blesion\b','',x)
    x = re.sub(r'[^a-z0-9]+',' ',x)
    return re.sub(r'\s+',' ',x).strip()


def pick(ex, explicit, cands, allow_list=False):
    keys = [explicit] if explicit else cands
    for k in keys:
        if not k or k not in ex: continue
        v = ex[k]
        if allow_list and isinstance(v,list) and v:
            return [norm(z) for z in v if norm(z)]
        if v is not None and str(v).strip() not in ('','[]','None'):
            return norm(v)
    return [] if allow_list else ''


def parse_args():
    p=argparse.ArgumentParser(description='Canonicalize rough inputs and oracle pair targets for pair-RAG.')
    p.add_argument('--input_json',required=True)
    p.add_argument('--out_json',required=True)
    p.add_argument('--raw_anatomy_field')
    p.add_argument('--raw_lesion_field')
    p.add_argument('--detail_field')
    p.add_argument('--oracle_anatomy_field')
    p.add_argument('--oracle_lesion_field')
    p.add_argument('--confidence_field')
    p.add_argument('--allow_test_oracle_for_evaluation',action='store_true')
    return p.parse_args()


def main():
    a=parse_args(); data=json.load(open(a.input_json))
    if not isinstance(data,dict) or not all(s in data for s in ('train','val','test')):
        raise ValueError('Input must be a dict with train/val/test lists.')
    out={}; stats={}
    for split, rows in data.items():
        out[split]=[]; miss={'raw_anatomy':0,'raw_lesion':0,'oracle_anatomy':0,'oracle_lesion':0}
        for ex in rows:
            z=copy.deepcopy(ex)
            raw_a=pick(ex,a.raw_anatomy_field,ANAT_CANDS)
            raw_l=pick(ex,a.raw_lesion_field,LESION_CANDS)
            details=pick(ex,a.detail_field,DETAIL_CANDS,True)
            # Remove generic tokens and prevent target leakage from detail list.
            details=[d for d in details if d not in ('lesion','type','unknown') and d!=raw_l]
            oa=pick(ex,a.oracle_anatomy_field,ORACLE_ANAT_CANDS)
            ol=pick(ex,a.oracle_lesion_field,ORACLE_LESION_CANDS)
            if split=='test' and not a.allow_test_oracle_for_evaluation:
                # Keep test labels only when explicitly requested for offline evaluation.
                oa_eval,ol_eval=oa,ol
            else:
                oa_eval,ol_eval=oa,ol
            conf=pick(ex,a.confidence_field,CONF_CANDS)
            try: conf=float(conf) if conf else None
            except Exception: conf=None
            z.update({
                'pair_rag_raw_anatomy': raw_a or 'unknown',
                'pair_rag_raw_lesion_type': raw_l or 'other lesion',
                'pair_rag_raw_details': sorted(set(details)),
                'pair_rag_oracle_anatomy': oa_eval or 'unknown',
                'pair_rag_oracle_lesion_type': ol_eval or 'other lesion',
                'pair_rag_classifier_confidence': conf,
                'pair_rag_query_text': f"anatomy {raw_a or 'unknown'} lesion type {raw_l or 'other lesion'} details {' '.join(sorted(set(details)))}".strip()
            })
            for k,v in [('raw_anatomy',raw_a),('raw_lesion',raw_l),('oracle_anatomy',oa),('oracle_lesion',ol)]:
                if not v: miss[k]+=1
            out[split].append(z)
        stats[split]={'count':len(rows),'missing':miss}
    Path(a.out_json).parent.mkdir(parents=True,exist_ok=True)
    json.dump(out,open(a.out_json,'w'),indent=2)
    print(json.dumps(stats,indent=2)); print('Saved:',a.out_json)
    print('\nFirst train canonical fields:')
    for k in [x for x in out['train'][0] if x.startswith('pair_rag_')]:
        print(f'  {k}: {out["train"][0][k]}')

if __name__=='__main__': main()
