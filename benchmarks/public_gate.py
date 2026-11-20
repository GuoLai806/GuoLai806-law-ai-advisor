"""Paired BEIR retrieval gate. Run with python -m benchmarks.public_gate --help."""
import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal

from .prepare import filehash, jsonl, qrels
from .metrics import evaluate


def loopback_url(value):
    u = urllib.parse.urlsplit(value)
    try:
        host_ok = u.hostname == 'localhost' or ipaddress.ip_address(u.hostname).is_loopback
    except ValueError:
        host_ok = False
    if u.scheme != 'http' or not host_ok or u.username or u.password or u.query or u.fragment or u.path != '/api/v2/retrieve' or not u.port:
        raise ValueError('endpoint must be an explicit loopback http://host:port/api/v2/retrieve URL')
    return value


def load_data(root, manifest_path):
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    paths = {'corpus': root / 'corpus.jsonl', 'queries': root / 'queries.jsonl', 'qrels': root / 'qrels/test.tsv'}
    hashes = {k: filehash(p) for k, p in paths.items()}
    for key, expected in [('corpus', manifest['corpus_sha256']), ('queries', manifest['queries_sha256']), ('qrels', manifest['qrels']['test']['sha256'])]:
        if hashes[key] != expected:
            raise ValueError(f'{key} hash mismatch')
    corpus = jsonl(paths['corpus']); queries = jsonl(paths['queries']); rel = qrels(paths['qrels'])
    if len(corpus) != manifest['corpus_count'] or len(queries) != manifest['query_count_total'] or len(rel) != manifest['qrels']['test']['qids']:
        raise ValueError('manifest count mismatch')
    docs = set(corpus); lookup = {qid: value['text'] for qid, value in queries.items()}
    if not rel or set(rel) - set(lookup) or any(set(d) - docs or not any(v > 0 for v in d.values()) for d in rel.values()):
        raise ValueError('invalid qrels references or no positive relevance')
    return lookup, rel, hashes


def trace_error(body, revision, rerank_limit):
    if not isinstance(body, dict) or body.get('corpus_revision') != revision:
        return 'missing or unexpected corpus revision'
    if body.get('degraded') is not False or body.get('degraded_reasons'):
        return 'degraded or missing degradation status'
    if not isinstance(body.get('evidence'), list):
        return 'missing evidence'
    if any(not isinstance(x, dict) or not isinstance(x.get('document_id'), str) or not x['document_id'] for x in body['evidence']):
        return 'invalid evidence document_id'
    trace = body.get('retrieval_trace')
    if not isinstance(trace, dict) or not isinstance(trace.get('timings_ms'), dict):
        return 'missing retrieval trace'
    candidates = trace.get('candidates')
    if trace.get('dense_backend') != 'openviking' or 'dense_ms' not in (trace.get('timings_ms') or {}) or not isinstance(trace.get('dense_hits'), list) or not trace['dense_hits']:
        return 'dense execution not established'
    if not isinstance(candidates, list) or not candidates or 'rank_and_rerank_ms' not in trace['timings_ms']:
        return 'reranking execution not established'
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            return 'invalid candidate'
        score = candidate.get('reranker_score')
        if score is None and i >= min(rerank_limit, len(candidates)):
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            return 'reranking execution not established'
    return None


def p95(values):
    return sorted(values)[math.ceil(len(values) * .95) - 1]


def decision(old_ms, new_ms, old_quality, new_quality):
    old_ms, new_ms, old_quality, new_quality = map(lambda x: Decimal(str(x)), (old_ms, new_ms, old_quality, new_quality))
    if old_ms <= 0 or old_quality <= 0:
        return False
    return ((new_ms < Decimal('.9') * old_ms and new_quality >= old_quality) or
            (new_quality > Decimal('1.1') * old_quality and new_ms <= old_ms))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, payload, timeout, token):
    data = json.dumps(payload, ensure_ascii=False).encode()
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    start = time.perf_counter_ns()
    status = None; raw = ''; error = None
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=timeout) as response:
            status = response.status
            raw = response.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read().decode('utf-8', errors='replace')
        error = f'HTTP {status}'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    body = None
    if error is None:
        try:
            body = json.loads(raw)
        except (ValueError, TypeError):
            error = 'invalid JSON response'
    return elapsed_ms, status, raw, body, error


def run(args):
    urls = {'baseline': loopback_url(args.baseline), 'candidate': loopback_url(args.candidate)}
    if urls['baseline'] == urls['candidate']:
        raise ValueError('endpoints must differ')
    if args.timeout <= 0 or not math.isfinite(args.timeout) or args.max_queries is not None and args.max_queries < 1:
        raise ValueError('invalid timeout or max-queries')
    limits = {'baseline': args.baseline_rerank_limit, 'candidate': args.candidate_rerank_limit}
    if any(limit < 1 for limit in limits.values()):
        raise ValueError('rerank limits must be positive')
    lookup, rel, hashes = load_data(args.data_dir, args.manifest)
    warmup_query = 'Explain how ocean currents transport heat.'
    if warmup_query in lookup.values():
        raise ValueError('synthetic warmup overlaps dataset queries')
    qids = sorted(rel)
    rng = random.Random(args.seed)
    rng.shuffle(qids)
    if args.max_queries is not None:
        qids = qids[:args.max_queries]
    orders = [0] * (len(qids) // 2) + [1] * (len(qids) // 2)
    if len(qids) % 2:
        orders.append(rng.randrange(2))
    rng.shuffle(orders)
    config = {'version': 2, 'endpoints': urls, 'revision': args.revision, 'dataset_hashes': hashes,
              'manifest_sha256': filehash(args.manifest), 'full_qrels_query_count': len(rel), 'selected_query_count': len(qids),
              'development_slice': args.max_queries is not None, 'max_queries': args.max_queries, 'seed': args.seed,
              'timeout_seconds': args.timeout, 'max_evidence_items': 10, 'rerank_limits': limits,
              'warmup_query': warmup_query, 'warmups_per_arm': 1, 'order': 'balanced randomized AB/BA; odd count differs by one', 'metric': 'TREC ndcg_cut_10 linear gain, BEIR self-id filtering',
              'latency': 'HTTP request through complete body read, nearest-rank p95, includes errors',
              'code_sha256': filehash(Path(__file__)), 'authorization': 'environment token present' if os.environ.get('PUBLIC_GATE_TOKEN') else 'none'}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps({**config, 'fingerprint': fingerprint}, indent=2) + '\n')
    rows = []
    warmup_failures = 0
    warmup_arms = list(urls)
    rng.shuffle(warmup_arms)
    with (output / 'warmups.jsonl').open('x', encoding='utf-8') as warmups:
        for arm in warmup_arms:
            payload = {'query': warmup_query, 'session_id': f'public_warmup_{arm}', 'profile': 'general', 'corpus_revision': args.revision,
                       'max_evidence_items': 10, 'include_retrieval_trace': True}
            ms, status, raw, body, error = request(urls[arm], payload, args.timeout, os.environ.get('PUBLIC_GATE_TOKEN'))
            if error is None and (not isinstance(body, dict) or body.get('corpus_revision') != args.revision or body.get('degraded') is not False):
                error = 'warmup response invalid, wrong revision, or degraded'
            warmup_failures += error is not None
            warmups.write(json.dumps({'arm': arm, 'elapsed_ms': ms, 'http_status': status, 'response_raw': raw, 'error': error}, ensure_ascii=False) + '\n')
            warmups.flush()
    with (output / 'trials.jsonl').open('x', encoding='utf-8') as stream:
        for index, qid in enumerate(qids):
            order = ['baseline', 'candidate'] if orders[index] == 0 else ['candidate', 'baseline']
            for arm in order:
                payload = {'query': lookup[qid], 'session_id': f'public_gate_{index}_{arm}', 'profile': 'general', 'corpus_revision': args.revision,
                           'max_evidence_items': 10, 'include_retrieval_trace': True}
                ms, status, raw, body, error = request(urls[arm], payload, args.timeout, os.environ.get('PUBLIC_GATE_TOKEN'))
                if error is None:
                    error = trace_error(body, args.revision, limits[arm])
                ranking = list(dict.fromkeys(e['document_id'] for e in body['evidence'])) if error is None else []
                scores = evaluate(qid, rel[qid], ranking) if error is None else {'ndcg_at_10': 0.0, 'mrr_at_10': 0.0}
                row = {'query_id': qid, 'arm': arm, 'pair_index': index, 'order': order, 'elapsed_ms': ms,
                       'http_status': status, 'response_raw': raw, 'error': error, 'ndcg_at_10': scores['ndcg_at_10'], 'mrr_at_10': scores['mrr_at_10']}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
                rows.append({k: row[k] for k in ('arm', 'elapsed_ms', 'ndcg_at_10', 'mrr_at_10', 'error')})
    metrics = {}
    for arm in urls:
        selected = [r for r in rows if r['arm'] == arm]
        metrics[arm] = {'p50_ms': statistics.median(r['elapsed_ms'] for r in selected), 'p95_ms': p95([r['elapsed_ms'] for r in selected]), 'mean_ndcg_at_10': statistics.mean(r['ndcg_at_10'] for r in selected), 'mean_mrr_at_10': statistics.mean(r['mrr_at_10'] for r in selected), 'failures': sum(r['error'] is not None for r in selected)}
    eligible = not config['development_slice'] and warmup_failures == 0 and all(m['failures'] == 0 for m in metrics.values()) and metrics['baseline']['p95_ms'] > 0 and metrics['baseline']['mean_ndcg_at_10'] > 0
    passed = eligible and decision(metrics['baseline']['p95_ms'], metrics['candidate']['p95_ms'], metrics['baseline']['mean_ndcg_at_10'], metrics['candidate']['mean_ndcg_at_10'])
    summary = {'fingerprint': fingerprint, 'metrics': metrics, 'warmup_failures': warmup_failures, 'eligible': eligible, 'passed': passed,
               'identity_note': 'Input files match manifest; response revision matches requested revision. Serving index content identity requires separate index provenance verification.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('data-dir', 'manifest', 'baseline', 'candidate', 'revision', 'output-dir'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--baseline-rerank-limit', type=int, default=60)
    parser.add_argument('--candidate-rerank-limit', type=int, default=60)
    parser.add_argument('--max-queries', type=int, help='development only; any use makes run ineligible')
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
