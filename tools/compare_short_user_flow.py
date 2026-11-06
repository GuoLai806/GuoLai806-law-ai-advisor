"""Paired, failure-inclusive analysis of the two predeclared 30-query runs."""
import argparse
from collections import Counter
from math import comb, sqrt
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.e2e import file_hash, save_json
from benchmarks.verification_metrics import aggregate


def wilson(successes,total):
    z=1.959963984540054; p=successes/total; denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    radius=z*sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [center-radius,center+radius]


def correct(row): return not row.get('failure') and row.get('predicted_label')==row['gold_label']


def latency(rows,field):
    result={}
    for status in ('all','completed','incomplete'):
        selected=[r for r in rows if status=='all' or bool(r.get('user_flow_completed'))==(status=='completed')]
        values=[r[field] for r in selected if r.get(field) is not None and r.get('wall_clock_valid')]
        result[status]={'n':len(values),'p50':float(np.quantile(values,.5)) if values else None,
                        'p95':float(np.quantile(values,.95)) if values else None}
    return result


def summary(report,raw_dir,name):
    rows=report['rows']; n=len(rows); quality=aggregate(rows)
    raw_metadata=[]
    for row in rows:
        raw_path=Path(raw_dir)/(name+'-'+row['query_id']+'.json')
        if file_hash(raw_path)!=row['raw_sha256']: raise ValueError('Raw event identity changed')
        raw=json.loads(raw_path.read_text())
        validations=[e['data'] for e in raw['events'] if e['event']=='grounding_validation']
        raw_metadata.append({'query_id':row['query_id'],'grounding_validation':validations})
    valid=[item for record in raw_metadata for item in record['grounding_validation']
           if item.get('status')=='verbatim_quotes_checked']
    count=sum(correct(r) for r in rows)
    return {'queries':n,'correct':count,'accuracy':count/n,'accuracy_wilson_95':wilson(count,n),
            'verification_metrics':quality,'retrieval_metrics':report['retrieval_metrics'],
            'by_label':{label:{'n':sum(r['gold_label']==label for r in rows),
                               'correct':sum(correct(r) and r['gold_label']==label for r in rows)}
                        for label in ('SUPPORT','CONTRADICT','NOT_ENOUGH_INFO')},
            'flow_completed':sum(bool(r.get('user_flow_completed')) for r in rows),
            'failure_counts':dict(Counter((r.get('failure') or {}).get('type','none') for r in rows)),
            'failure_provenance':dict(Counter((r.get('failure') or {}).get('error_type','none') for r in rows)),
            'latency_wall_ms':{f:latency(rows,f) for f in ('first_answer_wall_ms','stream_total_wall_ms','user_total_wall_ms')},
            'timing_health':report['timing_health'],'run_wall_ms':report['run_wall_ms'],
            'usage_totals':report['usage_totals'],'usage_complete_queries':report['usage_complete_queries'],
            'retried_queries':report['retried_queries'],'generation_attempt_count':report['generation_attempt_count'],
            'grounded_validation_responses':len(valid),
            'responses_with_quotes':sum(bool(i.get('quotes')) for i in valid),
            'responses_without_quotes':sum(not i.get('quotes') for i in valid),
            'quotes_checked_spans':sum(len(i.get('quotes',[])) for i in valid),
            'quote_check_scope':'verbatim occurrence and source identifiers only; semantic support is model assessed, not independently verified',
            'raw_grounding_metadata':raw_metadata}


def compare(baseline,candidate,raw_dir,output):
    bp,cp=Path(baseline),Path(candidate)
    a,b=json.loads(bp.read_text()),json.loads(cp.read_text())
    if a['split']!='test' or b['split']!='test': raise ValueError('Formal runs required')
    for r in (a,b):
        if r['queries_expected']!=30 or r['queries_recorded']!=30 or len(r['rows'])!=30:
            raise ValueError('All 30 rows, including failures, required')
        if not r['source_unchanged'] or not r['runtime_identity_unchanged']: raise ValueError('Runtime changed')
    if a['source_sha256']!=b['source_sha256'] or a['manifest_sha256']!=b['manifest_sha256']:
        raise ValueError('Same source and frozen sample required')
    ar,br=a['rows'],b['rows']
    if [(r['query_id'],r['gold_label'],r['gold_documents']) for r in ar]!=[(r['query_id'],r['gold_label'],r['gold_documents']) for r in br]:
        raise ValueError('Paired gold rows differ')
    if len({r['query_id'] for r in ar})!=30: raise ValueError('Duplicate paired question')
    x,y=np.array([correct(r) for r in ar],dtype=float),np.array([correct(r) for r in br],dtype=float)
    improved=int(sum((x==0)&(y==1)));regressed=int(sum((x==1)&(y==0)));discordant=improved+regressed
    rng=np.random.default_rng(20261107); indices=rng.integers(0,30,(10000,30))
    deltas=(y-x)[indices].mean(axis=1)
    result={'baseline_sha256':file_hash(bp),'candidate_sha256':file_hash(cp),'manifest_sha256':a['manifest_sha256'],
            'scope':'public, previously seen, class-balanced 30-query paired experiment; joint answer-format and generation-gate change; not independent hidden test, whole-RAG factual accuracy, or frontier proof',
            'baseline':summary(a,raw_dir,bp.stem),'candidate':summary(b,raw_dir,cp.stem),
            'paired':{'accuracy_difference':float((y-x).mean()),'difference_bootstrap_95':[float(np.quantile(deltas,.025)),float(np.quantile(deltas,.975))],
                      'bootstrap':'paired query percentile; 10000 resamples; seed20261107; shared documents may induce dependence',
                      'improved':improved,'regressed':regressed,'both_correct':int(sum((x==1)&(y==1))),'both_incorrect':int(sum((x==0)&(y==0))),
                      'mcnemar_exact_two_sided_p':min(1.,2*sum(comb(discordant,i) for i in range(min(improved,regressed)+1))/2**discordant) if discordant else 1.},
            'per_query':[{'query_id':left['query_id'],'gold_label':left['gold_label'],
                          'baseline_correct':bool(correct(left)),'candidate_correct':bool(correct(right)),
                          'baseline_failure':left.get('failure'),'candidate_failure':right.get('failure'),
                          'retrieval_metrics_unchanged':left['retrieval_metrics']==right['retrieval_metrics']}
                         for left,right in zip(ar,br,strict=True)],
            'latency_limits':'serial local MPS, warm reused index, network and cache may differ; valid observed wall clocks, failures retained and separately summarized; not production SLA or randomized order speed causation',
            'usage_scope':'reported provider token counts including failed attempts; unavailable usage is unknown, not zero'}
    save_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k in ('paired','baseline_sha256','candidate_sha256')}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('baseline','candidate','raw-dir','output'):p.add_argument('--'+field,required=True)
    args=p.parse_args(); compare(args.baseline,args.candidate,args.raw_dir,args.output)
