"""Secondary paired uncertainty calculations, no model requests or tuning."""
from pathlib import Path
import hashlib
import json
import numpy as np
OUT=Path(__file__).resolve().parent
a=json.loads((OUT/'baseline30-v1.json').read_text())['rows']
b=json.loads((OUT/'candidate30-v1.json').read_text())['rows']
assert len(a)==len(b)==30
assert [(r['query_id'],r['gold_label']) for r in a]==[(r['query_id'],r['gold_label']) for r in b]
def counts(rows):
    result=[]
    for r in rows:
        predicted=None if r.get('failure') else r['predicted_label']
        expected=set(r['gold_documents']); actual=set(r['cited_documents']) if predicted in ('SUPPORT','CONTRADICT') else set()
        result.append((len(expected & actual) if predicted==r['gold_label'] else 0,len(actual),len(expected),int(predicted==r['gold_label'])))
    return np.array(result)
x,y=counts(a),counts(b)
rng=np.random.default_rng(20261107); idx=rng.integers(0,30,(10000,30))
def metrics(t):
    tp,pred,gold=t[...,0],t[...,1],t[...,2]
    def div(n,d):return np.divide(n,d,out=np.zeros_like(n,dtype=float),where=d!=0)
    return {'citation_precision':div(tp,pred),'citation_recall':div(tp,gold),'citation_f1':div(2*tp,pred+gold)}
mx,my=metrics(x.sum(0)),metrics(y.sum(0));rx,ry=metrics(x[idx].sum(1)),metrics(y[idx].sum(1))
result={'scope':'secondary paired query uncertainty; failure-inclusive custom document+label metrics; no model requests, no config or sample changes; not official sentence rationale scoring', 'resamples':10000,'seed':20261107,'method':'paired query percentile bootstrap, non-stratified citation samples; shared documents may induce dependence, one generation per config', 'metrics':{k:{'baseline':float(mx[k]),'candidate':float(my[k]),'difference':float(my[k]-mx[k]),'difference_bootstrap_95':np.quantile(ry[k]-rx[k],[.025,.975]).tolist()} for k in mx}}
nei=np.array([i for i,r in enumerate(a) if r['gold_label']=='NOT_ENOUGH_INFO']);idxn=rng.integers(0,len(nei),(10000,len(nei)))
delta=y[nei,3]-x[nei,3]; result['nei_accuracy']={'n':len(nei),'baseline_correct':int(x[nei,3].sum()),'candidate_correct':int(y[nei,3].sum()),'difference':float(delta.mean()),'difference_bootstrap_95':np.quantile(delta[idxn].mean(1),[.025,.975]).tolist(),'improved':int(((x[nei,3]==0)&(y[nei,3]==1)).sum()),'regressed':int(((x[nei,3]==1)&(y[nei,3]==0)).sum())}
result['original_report_sha256']={n:json.loads((OUT/(n+'.json')).read_text()).get('original_report_sha256',hashlib.sha256((OUT/(n+'.json')).read_bytes()).hexdigest()) for n in ('baseline30-v1','candidate30-v1')}
print(json.dumps(result))
