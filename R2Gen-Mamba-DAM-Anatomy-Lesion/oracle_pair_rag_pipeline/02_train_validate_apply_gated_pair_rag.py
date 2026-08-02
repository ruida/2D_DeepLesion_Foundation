#!/usr/bin/env python3
import argparse, copy, json, math
from collections import Counter, defaultdict
from pathlib import Path


def jac(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else 0.0

def sim(q,x):
    s=0.0
    s += 0.45*(q['a']==x['a'])
    s += 0.30*(q['l']==x['l'])
    s += 0.25*jac(q['d'],x['d'])
    return s

def row(ex):
    return {'a':ex['pair_rag_raw_anatomy'],'l':ex['pair_rag_raw_lesion_type'],
            'd':ex.get('pair_rag_raw_details',[]),'oa':ex['pair_rag_oracle_anatomy'],
            'ol':ex['pair_rag_oracle_lesion_type'],'id':str(ex.get('id',''))}

def weighted_vote(items, attr):
    c=defaultdict(float)
    for r,s in items: c[r[attr]] += max(s,1e-6)
    ranked=sorted(c.items(),key=lambda z:(-z[1],z[0]))
    total=sum(c.values()) or 1.0
    best=ranked[0]
    second=ranked[1][1] if len(ranked)>1 else 0.0
    return best[0],best[1]/total,(best[1]-second)/total

def joint_vote(items):
    c=defaultdict(float)
    for r,s in items: c[(r['oa'],r['ol'])] += max(s,1e-6)
    ranked=sorted(c.items(),key=lambda z:(-z[1],z[0]))
    total=sum(c.values()) or 1.0; best=ranked[0]; second=ranked[1][1] if len(ranked)>1 else 0.0
    return best[0],best[1]/total,(best[1]-second)/total

def build_exact(train):
    tab=defaultdict(Counter)
    for r in train: tab[(r['a'],r['l'],tuple(r['d']))][(r['oa'],r['ol'])]+=1
    return tab

def query_key(q):
    return (
        q["a"],
        q["l"],
        tuple(sorted(set(q.get("d", [])))),
    )


def build_neighbor_cache(queries, train, max_k):
    cache = {}

    unique_queries = {}
    for q in queries:
        unique_queries.setdefault(query_key(q), q)

    print(
        "Unique retrieval queries:",
        len(unique_queries),
        "from",
        len(queries),
        "samples",
        flush=True,
    )

    for index, (key, q) in enumerate(unique_queries.items(), 1):
        vals = [(r, sim(q, r)) for r in train]
        vals.sort(key=lambda z: (-z[1], z[0]["id"]))

        # Keep one extra neighbor because train leave-one-out may remove self.
        cache[key] = vals[: max_k + 1]

        if index % 25 == 0 or index == len(unique_queries):
            print(
                "Cached queries:",
                index,
                "/",
                len(unique_queries),
                flush=True,
            )

    return cache


def retrieve_cached(q, cache, k, exclude_id=None):
    vals = cache[query_key(q)]

    if exclude_id is not None:
        vals = [
            (r, score)
            for r, score in vals
            if r["id"] != exclude_id
        ]

    return vals[:k]

def decide(q,neigh,exact_tab,cfg):
    key=(q['a'],q['l'],tuple(q['d']))
    cnt=exact_tab.get(key,Counter())
    if cnt:
        pair,n=cnt.most_common(1)[0]; support=sum(cnt.values()); purity=n/support
        if support>=cfg['exact_support'] and purity>=cfg['exact_purity']:
            return pair,'exact',purity,support,neigh[0][1] if neigh else 0.0
    pair,purity,margin=joint_vote(neigh)
    top_sim=neigh[0][1] if neigh else 0.0
    if purity>=cfg['joint_purity'] and margin>=cfg['joint_margin'] and top_sim>=cfg['min_similarity']:
        return pair,'joint_rag',purity,len(neigh),top_sim
    aa,ap,am=weighted_vote([(r,s) for r,s in neigh],'oa')
    ll,lp,lm=weighted_vote([(r,s) for r,s in neigh],'ol')
    use_a=ap>=cfg['partial_purity'] and am>=cfg['partial_margin']
    use_l=lp>=cfg['partial_purity'] and lm>=cfg['partial_margin']
    if use_a or use_l:
        return (aa if use_a else q['a'], ll if use_l else q['l']), ('partial_both' if use_a and use_l else 'partial_anatomy' if use_a else 'partial_lesion'), min(ap if use_a else 1,lp if use_l else 1),len(neigh),top_sim
    return (q['a'],q['l']),'raw_fallback',0.0,0,top_sim

def metrics(pred,gold):
    n=len(gold); ac=sum(p[0]==g[0] for p,g in zip(pred,gold)); lc=sum(p[1]==g[1] for p,g in zip(pred,gold)); jc=sum(p==g for p,g in zip(pred,gold))
    return {'n':n,'anatomy_accuracy':ac/n,'lesion_accuracy':lc/n,'joint_accuracy':jc/n}

def parse():
    p=argparse.ArgumentParser(description='Train-only structured oracle-pair RAG with validation-selected gate and rule agent.')
    p.add_argument('--input_json',required=True); p.add_argument('--out_json',required=True); p.add_argument('--out_dir',required=True)
    p.add_argument('--max_k',type=int,default=25)
    return p.parse_args()

def main():
    a=parse(); data=json.load(open(a.input_json)); train=[row(x) for x in data['train']]; val=[row(x) for x in data['val']]
    exact=build_exact(train)
    # Calculate retrieval only once for each unique structured query.
    all_queries = (
        [row(x) for x in data["train"]]
        + [row(x) for x in data["val"]]
        + [row(x) for x in data["test"]]
    )
    neighbor_cache = build_neighbor_cache(
        all_queries,
        train,
        a.max_k,
    )

    val_neigh = [
        retrieve_cached(q, neighbor_cache, a.max_k)
        for q in val
    ]
    grid=[]
    for k in (3,5,9,15,25):
      if k>a.max_k: continue
      for ep in (.55,.65,.75):
       for jp in (.45,.55,.65,.75):
        for jm in (0.0,.10,.20):
         for ms in (.30,.50,.70):
          cfg={'k':k,'exact_support':3,'exact_purity':ep,'joint_purity':jp,'joint_margin':jm,'min_similarity':ms,'partial_purity':.65,'partial_margin':.10}
          preds=[]
          for q,nn in zip(val,val_neigh): preds.append(decide(q,nn[:k],exact,cfg)[0])
          m=metrics(preds,[(q['oa'],q['ol']) for q in val]); grid.append((m['joint_accuracy'],(m['anatomy_accuracy']+m['lesion_accuracy'])/2,cfg,m))
    grid.sort(key=lambda z:(-z[0],-z[1],z[2]['k'])); best=grid[0]; cfg=best[2]
    out={}; summaries={}
    for split,examples in data.items():
        rr=[row(x) for x in examples]; preds=[]; methods=Counter(); new=[]
        for ex,q in zip(examples,rr):
            nn = retrieve_cached(
                q,
                neighbor_cache,
                cfg["k"],
                exclude_id=q["id"] if split == "train" else None,
            )
            pair,method,conf,support,top_sim=decide(q,nn,exact,cfg)
            preds.append(pair); methods[method]+=1
            z=copy.deepcopy(ex)
            z.update({
              'pair_rag_mapped_anatomy':pair[0], 'pair_rag_mapped_lesion_type':pair[1],
              'pair_rag_method':method,'pair_rag_confidence':conf,'pair_rag_support':support,'pair_rag_top_similarity':top_sim,
              'pair_rag_neighbor_ids':[r['id'] for r,s in nn],
              'pair_rag_neighbor_similarities':[round(s,6) for r,s in nn],
              'rough_anatomy_names':[pair[0],pair[1]],
              'anatomy_text':f"lesion {pair[0]}",
              'lesion_type_text':pair[1]
            })
            new.append(z)
        out[split]=new
        gold=[(q['oa'],q['ol']) for q in rr]
        summaries[split]={'raw':metrics([(q['a'],q['l']) for q in rr],gold),'after_rag':metrics(preds,gold),'methods':dict(methods)}
    Path(a.out_dir).mkdir(parents=True,exist_ok=True); Path(a.out_json).parent.mkdir(parents=True,exist_ok=True)
    json.dump(out,open(a.out_json,'w'),indent=2)
    json.dump({'best_config':cfg,'validation':best[3],'all_splits':summaries},open(Path(a.out_dir)/'summary.json','w'),indent=2)
    json.dump({'best_config':cfg,'top10':[{'joint':x[0],'mean_single':x[1],'config':x[2],'metrics':x[3]} for x in grid[:10]]},open(Path(a.out_dir)/'validation_grid.json','w'),indent=2)
    print('Best config:',json.dumps(cfg,indent=2)); print(json.dumps(summaries,indent=2)); print('Saved:',a.out_json)

if __name__=='__main__': main()
