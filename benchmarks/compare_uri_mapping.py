"""Replay real dense results against one read-only Catalog; exclude retrieval and scoring."""
import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from evidence_rag.catalog import Catalog


def point_lookup(db, revision, hits):
    values = []
    for hit in hits:
        row = db.execute('SELECT unit_id FROM units WHERE revision=? AND uri=?',
                         (revision, hit['uri'])).fetchone()
        if row is not None:
            values.append((row[0], hit['score']))
    return values


def batch_lookup(catalog, revision, hits):
    lookup = catalog.unit_ids_for_uris(revision, (hit['uri'] for hit in hits))
    return [(lookup[hit['uri']], hit['score']) for hit in hits if hit['uri'] in lookup]


def measure(args):
    cases = []
    with gzip.open(args.trials, 'rt', encoding='utf-8') as fp:
        for line in fp:
            row = json.loads(line)
            if row['arm'] != 'baseline':
                continue
            assert row['http_status'] == 200 and not row['error']
            body = json.loads(row['response_raw'])
            trace = body['retrieval_trace']
            assert trace['dense_backend'] == 'openviking' and not body['degraded']
            cases.append((row['query_id'], body['corpus_revision'], trace['dense_hits']))
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    order = [i % 2 for i in range(len(cases))]
    rng.shuffle(order)
    catalog = Catalog(args.db, read_only=True)
    rows = []
    try:
        for index, (qid, revision, hits) in enumerate(cases):
            methods = [('baseline', lambda: point_lookup(catalog.db, revision, hits)),
                       ('candidate', lambda: batch_lookup(catalog, revision, hits))]
            if order[index]:
                methods.reverse()
            durations = {}
            values = {}
            for label, query in methods:
                start = time.perf_counter_ns()
                values[label] = query()
                durations[label] = (time.perf_counter_ns()-start)/1_000_000
            same = values['candidate'] == values['baseline']
            rows.append({'qid':qid,'hits':len(hits),'mapped':len(values['baseline']),
                         'order':[m[0] for m in methods],'ms':durations,'identical':same,
                         'mapping_sha256':hashlib.sha256(json.dumps(values['baseline']).encode()).hexdigest()})
    finally:
        catalog.db.close()
    metrics = {}
    for label in ('baseline','candidate'):
        ordered = sorted(record['ms'][label] for record in rows)
        metrics[label] = {'p50_ms':statistics.median(ordered),
                          'p95_ms':ordered[math.ceil(.95*len(ordered))-1]}
    return {'scope':'SQLite mapping microbenchmark on recorded dense hits. No HTTP merge qualification.',
            'seed':args.seed,'input_sha256':hashlib.sha256(args.trials.read_bytes()).hexdigest(),
            'metrics':metrics,'queries':len(rows),'identical':sum(r['identical'] for r in rows),
            'mapped':sum(r['mapped'] for r in rows),'rows':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('db','trials','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261121)
    args=parser.parse_args()
    result=measure(args)
    with args.output.open('x') as file:
        json.dump(result,file,indent=2)
        file.write('\n')
    print({k:v for k,v in result.items() if k!='rows'})
