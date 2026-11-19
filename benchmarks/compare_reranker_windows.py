"""Compare the reranker input-window modes via paired requests to real RAG instances."""

import argparse
import gzip
import hashlib
import json
import math
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


def metrics(values):
    ordered = sorted(values)
    return {
        'count': len(ordered),
        'p50': statistics.median(ordered),
        'p95': ordered[math.ceil(len(ordered)*.95)-1],
    } if ordered else {'count': 0, 'p50': None, 'p95': None}


def retrieval_result(bundle):
    trace = bundle.get('retrieval_trace') or {}
    graph = bundle.get('graph') or {}
    return {
        'candidate_order': [row['unit_id'] for row in trace.get('candidates', [])],
        'evidence': [(row['unit_id'], row.get('part_no'), row['content_sha256'])
                     for row in bundle['evidence']],
        'scores': {row['unit_id']: row['reranker_score'] for row in trace.get('candidates', [])},
        'graph': [(item['edge_id'],item['unit_id'],item['char_start'],item['char_end'])
                  for item in graph.get('paths', [])],
        'status': bundle['status'],
        'graph_state': [graph.get('traversal_status'), graph.get('evidence_status')],
    }


def compare(pair):
    if any(row['error'] for row in pair.values()):
        return {'comparable': False}
    legacy, batch = (retrieval_result(pair[key]['bundle']) for key in ('legacy','guarded_batch'))
    delta = [abs(score - batch['scores'][uid]) for uid, score in legacy['scores'].items()
             if score is not None and batch['scores'].get(uid) is not None]
    return {
        'comparable': True,
        'candidate_order_same': legacy['candidate_order'] == batch['candidate_order'],
        'evidence_same': legacy['evidence'] == batch['evidence'],
        'graph_same': legacy['graph'] == batch['graph'],
        'outcomes_same': (legacy['status'],legacy['graph_state']) == (batch['status'],batch['graph_state']),
        'scores_same': legacy['scores'] == batch['scores'],
        'max_shared_score_delta': max(delta, default=None),
    }


def call(client, job, session_id):
    body = {'query':job['question'], 'session_id':session_id, 'include_retrieval_trace':True}
    if job['mode'] == 'graph':
        body.update(retrieval_mode='graph_dependency', event_date='2026-11-18')
    start = time.perf_counter()
    row = {'http_status': None, 'error': None, 'bundle': None}
    try:
        response = client.post('/api/v2/retrieve', json=body)
        row['http_status'] = response.status_code
        row['bundle'] = response.json()
        response.raise_for_status()
        bundle = row['bundle']
        trace = bundle['retrieval_trace']
        if (trace['dense_backend'] != 'openviking' or not trace['dense_hits'] or
                bundle['degraded'] or not trace['candidates'] or
                any(item['reranker_score'] is None for item in trace['candidates'])):
            row['error'] = 'incomplete_real_retrieval'
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        row['error'] = type(exc).__name__
    row['http_ms'] = round((time.perf_counter()-start)*1000,3)
    return row


def summarize(rows):
    summary = {}
    for key in ('legacy','guarded_batch'):
        results = [entry['response'][key] for entry in rows]
        valid = [row for row in results if not row['error']]
        summary[key] = {
            'failures': len(results)-len(valid),
            'http_ms': metrics([row['http_ms'] for row in valid]),
            'rerank_stage_ms': metrics([row['bundle']['retrieval_trace']['timings_ms']['rank_and_rerank_ms']
                                        for row in valid]),
        }
    for key in ('comparable','candidate_order_same','evidence_same','graph_same','outcomes_same','scores_same'):
        summary[key] = sum(row['comparison'].get(key, False) for row in rows)
    summary['max_shared_score_delta'] = max(
        (row['comparison']['max_shared_score_delta'] for row in rows
         if row['comparison'].get('max_shared_score_delta') is not None), default=None)
    return summary


def main(args):
    dataset = args.questions.read_bytes()
    questions = [json.loads(line) for line in dataset.splitlines()]
    jobs = [dict(q, mode='text') for q in questions]
    jobs.extend(dict(q, mode='graph') for q in questions if q['category']=='multi_document')
    started = datetime.now(timezone.utc).isoformat()
    rng = random.Random(191126)
    with (httpx.Client(base_url=args.legacy, timeout=30,trust_env=False) as legacy,
          httpx.Client(base_url=args.batch, timeout=30,trust_env=False) as batch):
        servers = {'legacy':legacy,'guarded_batch':batch}
        models = {mode:client.get('/api/v2/models').raise_for_status().json()
                  for mode, client in servers.items()}
        assert all(models[mode]['reranker_window_check']==mode for mode in servers)
        assert models['legacy']['runtime_identity']==models['guarded_batch']['runtime_identity']
        warmup = {mode:[call(client, job, 'rerank-warmup') for job in jobs[:3]]
                  for mode, client in servers.items()}
        entire = []
        for round_number in range(args.rounds):
            ordering = list(jobs)
            rng.shuffle(ordering)
            for index, job in enumerate(ordering):
                modes = list(servers)
                if (round_number+index)%2:
                    modes.reverse()
                responses = {mode:call(servers[mode],job,'rerank-ab') for mode in modes}
                entire.append({
                    'round':round_number+1, 'first':modes[0], 'question_id':job['id'],
                    'mode':job['mode'], 'response':responses, 'comparison':compare(responses),
                })
    summary = {mode:summarize([row for row in entire if row['mode']==mode])
               for mode in ('text','graph')}
    summary['all'] = summarize(entire)
    result = {
        'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat(),
        'host':platform.platform(), 'questions_sha256':hashlib.sha256(dataset).hexdigest(),
        'seed':191126, 'rounds':args.rounds, 'runtime':models, 'warmup':warmup,
        'scope':'Single concurrency HTTP retrieval, local real OpenViking, generation excluded; fixed development set.',
        'summary':summary, 'rows':entire,
    }
    with gzip.open(args.output,'xt',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy',required=True)
    parser.add_argument('--batch',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=3,choices=range(1,11))
    main(parser.parse_args())
