"""Paired, failure-inclusive analysis of the two predeclared 30-query runs."""
import argparse
from collections import Counter
from copy import deepcopy
from math import comb, sqrt
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.e2e import file_hash, save_json
from benchmarks.verification_metrics import aggregate
from tools.audit_user_flow import exploration_checks


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


def distribution(values):
    values=[value for value in values if value is not None]
    return {'n':len(values),'p50':float(np.quantile(values,.5)) if values else None,
            'p95':float(np.quantile(values,.95)) if values else None,
            'total':sum(values) if values else None}


def comparison_strategy(baseline,candidate):
    paths=(baseline.get('user_flow_strategy','direct'),candidate.get('user_flow_strategy','direct'))
    ar,br=baseline['runtime_identity'],candidate['runtime_identity']
    if paths==('direct','explore_then_answer'):
        if ar!=br or any(r.get('answer_format')!='grounded_json' or r.get('generation_gate')!='grounded_evidence' for r in (ar,br)):
            raise ValueError('Exploration comparison requires identical grounded service/runtime')
        if not baseline.get('freeze_pair_sha256') or baseline['freeze_pair_sha256']!=candidate.get('freeze_pair_sha256'):
            raise ValueError('Exploration comparison requires the same frozen pair')
        return 'exploration'
    if paths!=('direct','direct'):
        raise ValueError('Unsupported or reversed user path comparison')
    if (ar.get('answer_format')!='text' or ar.get('generation_gate')!='reranker_threshold'
            or br.get('answer_format')!='grounded_json' or br.get('generation_gate')!='grounded_evidence'):
        raise ValueError('Grounding comparison requires historical text and grounded candidate modes')
    copies=[]
    for runtime in (ar,br):
        normalized=deepcopy(runtime)
        for key in ('answer_format','generation_gate','system_prompt_sha256'):
            normalized.pop(key,None)
        copies.append(normalized)
    if copies[0]!=copies[1] or str(ar.get('rerank_threshold')) not in ('0','0.0'):
        raise ValueError('Grounding runtimes differ outside the declared joint strategy')
    return 'grounding'


def summary(report,raw_dir,name):
    rows=report['rows']; n=len(rows); quality=aggregate(rows)
    raw_metadata=[]; observations=[]; evidence_bytes=[]
    strategy=report.get('user_flow_strategy','direct')
    for row in rows:
        raw_path=Path(raw_dir)/(name+'-'+row['query_id']+'.json')
        if file_hash(raw_path)!=row['raw_sha256']: raise ValueError('Raw event identity changed')
        raw=json.loads(raw_path.read_text())
        if not all(exploration_checks(row,raw,strategy).values()):
            raise ValueError('Exploration observation is inconsistent with preserved raw artifacts')
        record=raw.get('exploration') or {}
        preview_ms=(record.get('preview_received') or {}).get('monotonic_ms')
        original_ms=(record.get('original_received') or {}).get('monotonic_ms')
        encoded_bytes=sum(len(json.dumps(e['data'],ensure_ascii=False,separators=(',',':')).encode('utf-8'))
                          for e in raw['events'] if e['event']=='evidence_ready')
        evidence_bytes.append(encoded_bytes if any(e['event']=='evidence_ready' for e in raw['events']) else None)
        observations.append({'query_id':row['query_id'],'attempted':raw.get('exploration') is not None,
                             'preview_ms':preview_ms,'original_ms':original_ms,
                             'original_read_ms':original_ms-preview_ms if original_ms is not None and preview_ms is not None else None,
                             'response_bytes':record.get('response_bytes'),'original_bytes':record.get('original_bytes'),
                             'preview_http_status':record.get('preview_http_status'),'original_http_status':record.get('original_http_status'),
                             'original_verified':bool(record.get('original_verified')),
                             'answer_evidence_ready_payload_bytes':evidence_bytes[-1],
                             'failed':bool(row.get('failure'))})
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
            'user_flow_strategy':strategy,
            'exploration_observations':{
                'attempted_queries':sum(item['attempted'] for item in observations),
                'verified_original_queries':sum(item['original_verified'] for item in observations),
                'preview_http_status_counts':dict(Counter(str(item['preview_http_status']) for item in observations if item['preview_http_status'] is not None)),
                'original_http_status_counts':dict(Counter(str(item['original_http_status']) for item in observations if item['original_http_status'] is not None)),
                'metrics':{field:distribution([item[field] for item in observations]) for field in
                           ('preview_ms','original_ms','original_read_ms','response_bytes','original_bytes','answer_evidence_ready_payload_bytes')},
                'byte_scope':'Preview and first-original response bytes are preserved HTTP UTF-8 JSON body bytes; evidence_ready payload size is compact JSON re-encoding of parsed SSE data, excludes SSE framing/other events/HTTP headers and is not observed wire size. Preview adds requests and does not replace answer evidence.',
                'timing_scope':'preview_ms and original_ms start at the user action; original_read_ms is their difference. Observed failed HTTP responses are included; only preserved and checked observations are summarized.',
                'per_query':observations},
            'raw_grounding_metadata':raw_metadata}


def compare(baseline,candidate,raw_dir,output):
    bp,cp=Path(baseline),Path(candidate)
    a,b=json.loads(bp.read_text()),json.loads(cp.read_text())
    if a['split']!='test' or b['split']!='test': raise ValueError('Formal runs required')
    for r in (a,b):
        if r['queries_expected']!=30 or r['queries_recorded']!=30 or len(r['rows'])!=30:
            raise ValueError('All 30 rows, including failures, required')
        if not r['source_unchanged'] or not r['runtime_identity_unchanged']: raise ValueError('Runtime changed')
    strategy=comparison_strategy(a,b)
    for path,report in ((bp,a),(cp,b)):
        header=json.loads(Path(str(path)+'.journal.jsonl').read_text().splitlines()[0])
        if (header.get('type')!='header' or header.get('user_flow_strategy','direct')!=report.get('user_flow_strategy','direct')
                or header.get('config_sha256')!=report['config_sha256']):
            raise ValueError('Report/journal strategy or configuration changed')
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
            'comparison_strategy':strategy,
            'scope':('public, previously seen, class-balanced 30-query paired experiment; identical grounded answer service/runtime, only direct versus explore-then-answer user path differs; '
                     if strategy=='exploration' else 'public, previously seen, class-balanced 30-query paired experiment; joint answer-format and generation-gate change; ')
                    +'not independent hidden test, whole-RAG factual accuracy, or frontier proof',
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
