#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
def clean(s): return ' '.join(str(s or '').lower().replace('_',' ').split())
def row(x): return {'id':str(x['id']),'a':clean(x['raw_simple_anatomy']),'p':clean(x['raw_primary_lesion_type']),'t':clean(x['raw_lesion_attribute']),'d':tuple(sorted(set(clean(v) for v in x.get('raw_totalseg_details',[]) if clean(v)))),'oa':clean(x['oracle_simple_anatomy']),'op':clean(x['oracle_primary_lesion_type']),'ot':clean(x['oracle_lesion_attribute'])}
def jac(a,b):
    a,b=set(a),set(b)
    if not a and not b:return 1.0
    if not a or not b:return 0.0
    return len(a&b)/len(a|b)
def sim(q,r): return .4*(q['a']==r['a'])+.3*(q['p']==r['p'])+.15*(q['t']==r['t'])+.15*jac(q['d'],r['d'])
def key(q): return (q['a'],q['p'],q['t'],q['d'])
def cache(queries,train,maxk):
    uq={}
    for q in queries: uq.setdefault(key(q),q)
    c={}; print('Unique queries:',len(uq),flush=True)
    for i,(k,q) in enumerate(uq.items(),1):
        v=[(r,sim(q,r)) for r in train]; v.sort(key=lambda z:(-z[1],z[0]['id'])); c[k]=v[:maxk+1]
        if i%25==0 or i==len(uq): print('cached',i,'/',len(uq),flush=True)
    return c
def neigh(q,c,k,exclude=None):
    v=c[key(q)]
    if exclude is not None: v=[(r,s) for r,s in v if r['id']!=exclude]
    return v[:k]
def vote(ns,f):
    c=Counter()
    for r,s in ns: c[r[f]]+=max(s,1e-6)
    if not c:return 'unknown',0,0
    vals=c.most_common(); lab,w=vals[0]; tot=sum(c.values()); sec=vals[1][1] if len(vals)>1 else 0
    return lab,w/tot,(w-sec)/tot
def exact_table(train):
    d=defaultdict(Counter)
    for r in train:d[key(r)][(r['oa'],r['op'],r['ot'])]+=1
    return d
def decide(q,ns,ex,cfg):
    if key(q) in ex:
        cc=ex[key(q)]; pair,n=cc.most_common(1)[0]; total=sum(cc.values()); pur=n/total
        if total>=cfg['exact_support'] and pur>=cfg['exact_purity']: return pair,'exact',pur,total
    top=ns[0][1] if ns else 0
    va,pa,ma=vote(ns,'oa'); vp,pp,mp=vote(ns,'op'); vt,pt,mt=vote(ns,'ot')
    jc=Counter()
    for r,s in ns: jc[(r['oa'],r['op'],r['ot'])]+=max(s,1e-6)
    if jc:
        vals=jc.most_common(); pair,w=vals[0]; total=sum(jc.values()); sec=vals[1][1] if len(vals)>1 else 0; pur=w/total; mar=(w-sec)/total
        if top>=cfg['min_similarity'] and pur>=cfg['joint_purity'] and mar>=cfg['joint_margin']: return pair,'joint_rag',pur,len(ns)
    ao=pa>=cfg['partial_purity'] and ma>=cfg['partial_margin']; po=pp>=cfg['partial_purity'] and mp>=cfg['partial_margin']; to=pt>=cfg['partial_purity'] and mt>=cfg['partial_margin']
    pair=(va if ao else q['a'],vp if po else q['p'],vt if to else q['t'])
    method='raw_fallback'
    if ao and po and to: method='partial_all'
    elif ao and po: method='partial_anatomy_primary'
    elif ao: method='partial_anatomy'
    elif po: method='partial_primary'
    elif to: method='partial_attribute'
    return pair,method,max(pa if ao else 0,pp if po else 0,pt if to else 0),len(ns)
def metrics(ev,p):
    n=len(ev); return {'n':n,'anatomy_accuracy':sum(x[p+'a']==x['oa'] for x in ev)/n,'primary_accuracy':sum(x[p+'p']==x['op'] for x in ev)/n,'attribute_accuracy':sum(x[p+'t']==x['ot'] for x in ev)/n,'anatomy_primary_joint':sum((x[p+'a'],x[p+'p'])==(x['oa'],x['op']) for x in ev)/n,'full_joint':sum((x[p+'a'],x[p+'p'],x[p+'t'])==(x['oa'],x['op'],x['ot']) for x in ev)/n}
def objective(m): return .45*m['anatomy_primary_joint']+.25*m['anatomy_accuracy']+.2*m['primary_accuracy']+.1*m['attribute_accuracy']
def apply(split,orig,qs,c,ex,cfg):
    out=[]; ev=[]; methods=Counter()
    for x,q in zip(orig,qs):
        ns=neigh(q,c,cfg['k'],q['id'] if split=='train' else None); pair,m,conf,supp=decide(q,ns,ex,cfg); y=dict(x)
        y['rag_mapped_anatomy'],y['rag_mapped_primary_lesion_type'],y['rag_mapped_lesion_attribute']=pair; y['rag_method']=m; y['rag_confidence']=conf; y['rag_support']=supp; y['rag_neighbor_ids']=[r['id'] for r,_ in ns]; y['rag_neighbor_similarities']=[round(s,6) for _,s in ns]; out.append(y); methods[m]+=1
        ev.append({'rawa':q['a'],'rawp':q['p'],'rawt':q['t'],'raga':pair[0],'ragp':pair[1],'ragt':pair[2],'oa':q['oa'],'op':q['op'],'ot':q['ot']})
    return out,ev,dict(methods)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input_json',required=True); ap.add_argument('--out_json',required=True); ap.add_argument('--out_dir',required=True); ap.add_argument('--max_k',type=int,default=7); a=ap.parse_args()
    d=json.load(open(a.input_json)); tr=[row(x) for x in d['train']]; va=[row(x) for x in d['val']]; te=[row(x) for x in d['test']]; c=cache(tr+va+te,tr,a.max_k); ex=exact_table(tr); grid=[]
    for k in [1,3,5,7]:
      if k>a.max_k: continue
      for es in [2,3,5]:
       for ep in [.55,.65]:
        for jp in [.45,.55,.65]:
         for ms in [.3,.4,.5]:
          cfg={'k':k,'exact_support':es,'exact_purity':ep,'joint_purity':jp,'joint_margin':0.0,'min_similarity':ms,'partial_purity':.65,'partial_margin':.1}
          _,ev,_=apply('val',d['val'],va,c,ex,cfg); m=metrics(ev,'rag'); grid.append({'config':cfg,'metrics':m,'objective':objective(m)})
    grid.sort(key=lambda z:(-z['objective'],-z['metrics']['full_joint'])); best=grid[0]['config']; print('Best config:',json.dumps(best,indent=2)); out={}; summary={'best_config':best,'splits':{}}
    for s,orig,qs in [('train',d['train'],tr),('val',d['val'],va),('test',d['test'],te)]:
        rows,ev,methods=apply(s,orig,qs,c,ex,best); out[s]=rows; summary['splits'][s]={'raw':metrics(ev,'raw'),'rag':metrics(ev,'rag'),'methods':methods}
    od=Path(a.out_dir); od.mkdir(parents=True,exist_ok=True); Path(a.out_json).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out_json,'w'),indent=2,ensure_ascii=False); json.dump(summary,open(od/'summary.json','w'),indent=2,ensure_ascii=False); json.dump(grid[:50],open(od/'validation_grid.json','w'),indent=2,ensure_ascii=False); print(json.dumps(summary,indent=2)); print('Saved:',a.out_json)
if __name__=='__main__': main()
