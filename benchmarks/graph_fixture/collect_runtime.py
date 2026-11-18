"""Collect real HTTP responses on the fixed development fixture, with scoped metrics."""
import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx

ROOT = Path(__file__).parent


def has_path(paths, start, target):
    outgoing = defaultdict(list)
    for row in paths:
        outgoing[row['src_id']].append(row['dst_id'])
    seen, queue = {start}, deque([start])
    while queue:
        current = queue.popleft()
        for node in outgoing[current]:
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                queue.append(node)
    return False


def jobs(questions, rounds):
    for question in questions:
        yield 'quality', 'text', 0, question
        if question['category'] == 'multi_document':
            yield 'quality', 'graph_seeded', 0, question
            yield 'quality', 'graph_auto', 0, question
    multihop = [q for q in questions if q['category']=='multi_document']
    for trial in range(1, rounds+1):
        schedule = [(mode, q) for q in multihop for mode in ('text','graph_seeded')]
        random.Random(181126+trial).shuffle(schedule)
        for mode, q in schedule:
            yield 'timing', mode, trial, q


async def collect(args):
    raw_questions = (ROOT/'questions.jsonl').read_bytes()
    questions = [json.loads(line) for line in raw_questions.splitlines()]
    schedule = list(jobs(questions,args.rounds))
    rows = []
    async with httpx.AsyncClient(base_url=args.url, timeout=30, trust_env=False) as client:
        caps = (await client.get('/api/v2/models')).raise_for_status().json()
        assert caps['dense_configured'] and caps['graph_available']
        assert bool(caps['reranker_loaded']) == (not args.no_reranker)
        gate = asyncio.Semaphore(args.concurrency)

        async def ask(phase, mode, trial, q):
            body = {'query':q['question'], 'session_id':f'graph-dev-{args.concurrency}',
                    'event_date':'2026-11-18', 'include_retrieval_trace':True}
            gold = q.get('expected_answer')
            if mode != 'text':
                body['retrieval_mode'] = 'graph_dependency'
                if mode == 'graph_seeded':
                    body['graph_seed'] = gold[0]
            start = time.perf_counter()
            status, error, result = None, None, {}
            async with gate:
                start = time.perf_counter()
                try:
                    response = await client.post('/api/v2/retrieve',json=body)
                    status = response.status_code
                    result = response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    error = type(exc).__name__
            elapsed = round((time.perf_counter()-start)*1000,3)
            graph = result.get('graph') or {}
            trace = result.get('retrieval_trace') or {}
            evidence = [i.get('document_id') for i in result.get('evidence',[])]
            required = q.get('required_evidence') or []
            return {'phase':phase,'mode':mode,'trial':trial,'id':q['id'],'category':q['category'],
                    'status':status,'error':error,'ms':elapsed,
                    'real_dense_hits':len(trace.get('dense_hits',[])),
                    'complete_evidence':set(required)<=set(evidence) if required else None,
                    'gold_reachable':has_path(graph.get('paths',[]),gold[0],gold[-1])
                       if mode!='text' and q['category']=='multi_document' else None,
                    'response':result}

        for batch in range(0,len(schedule),args.concurrency):
            rows.extend(await asyncio.gather(*(ask(*job) for job in schedule[batch:batch+args.concurrency])))
    aggregates = {}
    for mode in ('text','graph_seeded','graph_auto'):
        quality=[r for r in rows if r['phase']=='quality' and r['mode']==mode and r['category']=='multi_document']
        timing=[r for r in rows if r['phase']=='timing' and r['mode']==mode]
        vals=sorted(r['ms'] for r in timing if r['status']==200)
        aggregates[mode]={'multihop_count':len(quality),
                          'complete_displayed_evidence':sum(r['complete_evidence'] is True for r in quality),
                          'verified_gold_path_reached':sum(r['gold_reachable'] is True for r in quality) if mode!='text' else None,
                          'timing_count':len(timing),'http_errors':sum(r['status']!=200 for r in timing),
                          'p50_ms':statistics.median(vals) if vals else None,
                          'p95_ms':vals[math.ceil(len(vals)*0.95)-1] if vals else None}
    return {'dataset_sha256':hashlib.sha256(raw_questions).hexdigest(),
            'runtime':caps,'concurrency':args.concurrency,
            'scope':'retrieval-only development experiment; curated dependency subset and expert-provided seeds, not a blinded closure/end-to-end benchmark',
            'aggregates':aggregates,'runs':rows}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=9)
    p.add_argument('--concurrency', type=int, choices=[1,4], default=1)
    p.add_argument('--no-reranker', action='store_true')
    args=p.parse_args()
    payload=asyncio.run(collect(args))
    with args.output.open('x') as f:
        json.dump(payload,f,ensure_ascii=False)
        f.write('\n')
    print(json.dumps(payload['aggregates'],indent=2))
