"""Paired real-OpenViking find comparison; never a mock latency benchmark.

Run against one pinned OpenViking server/scope. Variant A uses the production
adapter's per-operation lifecycle; B uses its shared client. Both issue the same
find request. No model generation or reranking is measured here.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from urllib.parse import urlsplit

import httpx
from evidence_rag.retrieval import OpenViking
from .e2e import Journal, file_hash, save_json, source_hashes
from .prepare import jsonl, qrels

VARIANTS = ('per_operation', 'shared')


def order_for(index):
    """Counterbalance which lifecycle sees a query first; never parallelize."""
    return VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))


def percentile(values, fraction):
    """Linear interpolation (Hyndman-Fan type 7), null for no observations."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def utc_ns(value):
    seconds, nanos = divmod(value, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + f'.{nanos:09d}Z'


def load_queries(data_dir, manifest_path, split='test'):
    """Use frozen IDs, verify actual dataset bytes, and do not select by results."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('dataset') != 'scifact' or manifest.get('short_evaluation_protocol') != 'scifact-stratified-30-v1':
        raise ValueError('Expected the frozen SciFact short evaluation manifest')
    data = Path(data_dir)
    for filename, field in [('corpus.jsonl', 'corpus_sha256'), ('queries.jsonl', 'queries_sha256')]:
        if file_hash(data / filename) != manifest[field]:
            raise ValueError('Dataset byte identity mismatch')
    queries = jsonl(data / 'queries.jsonl')
    corpus = jsonl(data / 'corpus.jsonl')
    if len(corpus) != manifest['corpus_count'] or len(queries) != manifest['query_count_total']:
        raise ValueError('Dataset coverage mismatch')
    block = manifest['validation' if split == 'validation' else 'short_evaluation']
    source = 'train' if split == 'validation' else 'test'
    expected_count = 6 if split == 'validation' else 30
    identifiers = block['query_ids']
    if block.get('source_split') != source or block.get('count') != expected_count or len(identifiers) != expected_count or len(set(identifiers)) != expected_count:
        raise ValueError('Frozen sample size/source mismatch')
    if set(manifest['validation']['query_ids']) & set(manifest['short_evaluation']['query_ids']):
        raise ValueError('Development and formal sample overlap')
    if file_hash(data / 'qrels' / (source + '.tsv')) != manifest['qrels'][source]['sha256']:
        raise ValueError('Qrels byte identity mismatch')
    references = qrels(data / 'qrels' / (source + '.tsv'))
    if len(references) != manifest['qrels'][source]['qids'] or set(identifiers) - set(references) or set(identifiers) - set(queries):
        raise ValueError('Frozen query membership mismatch')
    if any(not isinstance(queries[qid].get('text'), str) or not queries[qid]['text'] for qid in identifiers):
        raise ValueError('Invalid query text')
    return manifest, [(qid, queries[qid]['text']) for qid in identifiers]


def summarize(pairs):
    """Failures stay in denominators; latency uses explicitly counted observations."""
    result = {'query_count': len(pairs), 'requests_expected': 2 * len(pairs),
              'all_successful_pairs': 0, 'exact_result_equal_pairs': 0,
              'result_mismatch_pairs': 0, 'failed_pairs': 0, 'variants': {}}
    differences = []
    for variant in VARIANTS:
        requests = [pair['requests'][variant] for pair in pairs]
        successes = [request for request in requests if request['success']]
        observed = [request['elapsed_ms'] for request in requests]
        successful_times = [request['elapsed_ms'] for request in successes]
        result['variants'][variant] = {
            'requests': len(requests), 'successes': len(successes),
            'failures': len(requests) - len(successes),
            'observed_latency_ms': {'n': len(observed), 'median': statistics.median(observed) if observed else None, 'p95': percentile(observed, .95)},
            'successful_latency_ms': {'n': len(successful_times), 'median': statistics.median(successful_times) if successful_times else None, 'p95': percentile(successful_times, .95)},
        }
    for pair in pairs:
        a, b = (pair['requests'][variant] for variant in VARIANTS)
        if a['success'] and b['success']:
            result['all_successful_pairs'] += 1
            if a['rows_sha256'] == b['rows_sha256']:
                result['exact_result_equal_pairs'] += 1
            else:
                result['result_mismatch_pairs'] += 1
            differences.append(b['elapsed_ms'] - a['elapsed_ms'])
        else:
            result['failed_pairs'] += 1
    result['paired_shared_minus_per_operation_ms'] = {
        'n': len(differences), 'median': statistics.median(differences) if differences else None,
        'mean': statistics.mean(differences) if differences else None,
        'p95': percentile(differences, .95),
        'shared_faster_pairs': sum(value < 0 for value in differences),
        'definition': 'B elapsed minus A elapsed; successful pairs only, negative is faster shared; no failed pair imputation',
    }
    return result


async def health_identity(url, headers):
    """Record safe identity fields; never echo backend body, URL or exception text."""
    started = time.time_ns()
    try:
        async with httpx.AsyncClient(timeout=5, trust_env=False, follow_redirects=False, headers=headers) as client:
            response = await client.get(url.rstrip('/') + '/health')
            response.raise_for_status()
            data = response.json()
            safe = {key: data[key] for key in ('status', 'version', 'openviking_version')
                    if isinstance(data, dict) and isinstance(data.get(key), (str, int, float, bool))}
            return {'success': True, 'http_status': response.status_code,
                    'public_fields': safe, 'body_sha256': hashlib.sha256(response.content).hexdigest(),
                    'observed_utc': utc_ns(started), 'observed_wall_ns': started}
    except Exception as exc:
        return {'success': False, 'error_type': type(exc).__name__,
                'observed_utc': utc_ns(started), 'observed_wall_ns': started}


async def observe(adapter, query, qid, variant, phase):
    started_monotonic = time.monotonic_ns()
    started_wall = time.time_ns()
    record = {'query_id': qid, 'query_sha256': hashlib.sha256(query.encode()).hexdigest(),
              'variant': variant, 'phase': phase, 'started_utc': utc_ns(started_wall),
              'started_wall_ns': started_wall, 'success': False}
    try:
        rows = await adapter.find(query)
        record.update(success=True, rows=rows, rows_sha256=digest_json(rows), row_count=len(rows))
    except Exception as exc:
        # A URL or credential can appear in an exception message: only store type.
        record['error_type'] = type(exc).__name__
    finally:
        ended_monotonic = time.monotonic_ns()
        ended_wall = time.time_ns()
        elapsed = (ended_monotonic - started_monotonic) / 1_000_000
        wall = (ended_wall - started_wall) / 1_000_000
        record.update(elapsed_ms=elapsed, observed_wall_ms=wall,
                      wall_minus_monotonic_ms=wall - elapsed,
                      wall_clock_reversed=ended_wall < started_wall,
                      wall_monotonic_divergence=abs(wall - elapsed) > 1000,
                      finished_utc=utc_ns(ended_wall), finished_wall_ns=ended_wall)
    return record


async def run_comparison(queries, adapters, journal, warmup=True):
    warmups = []
    if warmup and queries:
        qid, query = queries[0]
        for variant in VARIANTS:
            record = await observe(adapters[variant], query, qid, variant, 'warmup')
            warmups.append(record)
            journal.record(record)
    pairs = []
    for index, (qid, query) in enumerate(queries):
        pair = {'query_id': qid, 'index': index, 'order': list(order_for(index)), 'requests': {}}
        for variant in order_for(index):
            record = await observe(adapters[variant], query, qid, variant, 'measured')
            pair['requests'][variant] = record
            journal.record(record)
        a, b = (pair['requests'][variant] for variant in VARIANTS)
        pair['exact_results_equal'] = a['rows_sha256'] == b['rows_sha256'] if a['success'] and b['success'] else None
        pair['shared_minus_per_operation_ms'] = b['elapsed_ms'] - a['elapsed_ms'] if a['success'] and b['success'] else None
        pairs.append(pair)
    return warmups, pairs


async def run(args, headers):
    manifest, queries = load_queries(args.data_dir, args.manifest, args.split)
    output = Path(args.output)
    journal_path = Path(str(output) + '.journal.jsonl')
    if output.exists() or journal_path.exists():
        raise FileExistsError('Output or journal exists; preserve previous measurements')
    started = time.time_ns()
    initial_source = source_hashes()
    adapters = {
        'per_operation': OpenViking(args.url, args.scope, headers=headers, timeout=args.timeout, reuse_client=False),
        'shared': OpenViking(args.url, args.scope, headers=headers, timeout=args.timeout, reuse_client=True),
    }
    journal = Journal(journal_path)
    journal.record({'phase': 'run_started', 'started_wall_ns': started, 'started_utc': utc_ns(started),
                    'manifest_sha256': file_hash(args.manifest), 'source_sha256': initial_source,
                    'backend_url_sha256': hashlib.sha256(args.url.encode()).hexdigest(),
                    'dense_scope': args.scope.rstrip('/') + '/', 'timeout_seconds': args.timeout,
                    'query_ids': [qid for qid, _ in queries], 'expected_measured_requests': 2 * len(queries)})
    try:
        health = await health_identity(args.url, headers)
        warmups, pairs = await run_comparison(queries, adapters, journal, not args.no_warmup)
    finally:
        for adapter in adapters.values():
            await adapter.aclose()
        journal.close()
    ended = time.time_ns()
    final_source = source_hashes()
    versions = {}
    for package in ('httpx', 'httpcore'):
        versions[package] = importlib.metadata.version(package)
    report = {
        'protocol': 'openviking-production-client-lifecycle-paired-v1',
        'scope': 'Real find HTTP stage only; excludes lexical retrieval, reranking, generation and user flow. No industry or answer-latency claim.',
        'configuration': {'dense_scope': args.scope.rstrip('/') + '/',
                          'backend_url_sha256': hashlib.sha256(args.url.encode()).hexdigest(),
                          'timeout_seconds': args.timeout, 'trust_env': False, 'follow_redirects': False,
                          'find_parameters': {'context_type': 'resource', 'level': 2, 'read_content': False, 'limit': 40},
                          'warmup': 'one request per variant on first frozen query' if not args.no_warmup else 'none',
                          'order': 'alternating AB/BA per frozen manifest order; serial; no retries'},
        'dataset': {'name': manifest['dataset'], 'manifest_sha256': file_hash(args.manifest),
                    'corpus_sha256': manifest['corpus_sha256'], 'queries_sha256': manifest['queries_sha256'],
                    'corpus_count': manifest['corpus_count'], 'split': args.split,
                    'query_ids': [qid for qid, _ in queries]},
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                    'packages': versions, 'backend_health': health},
        'source_sha256': initial_source, 'source_sha256_at_end': final_source,
        'source_unchanged': initial_source == final_source,
        'started_utc': utc_ns(started), 'started_wall_ns': started,
        'finished_utc': utc_ns(ended), 'finished_wall_ns': ended,
        'warmups': warmups, 'pairs': pairs, 'summary': summarize(pairs),
        'journal_sha256': file_hash(journal_path),
        'limitations': ['Local transport only when run on loopback; no connection-count measurement.',
                        'Server/vector/model caches and serial-order effects remain; order counterbalanced, not eliminated.',
                        'Failure latencies retained separately; paired differences only include successful pairs.',
                        'Backend health may omit version; operator must preserve actual backend version/configuration evidence separately.',
                        'Frozen public small sample; not unseen quality evaluation or confidence of industry-leading performance.'],
    }
    save_json(output, report)
    print(json.dumps({'report': output.name, 'summary': report['summary'],
                      'source_unchanged': report['source_unchanged']}, ensure_ascii=False))
    return 0 if not report['summary']['failed_pairs'] and not report['summary']['result_mismatch_pairs'] and report['source_unchanged'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('url', 'scope', 'manifest', 'data-dir', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--no-warmup', action='store_true')
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Expected a clean HTTP endpoint without credentials, query or fragment')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('Timeout must be finite and positive')
    private = json.load(sys.stdin)
    if not isinstance(private, dict) or set(private) - {'headers'}:
        raise ValueError('stdin must be an object containing only optional headers')
    headers = private.get('headers', {})
    if not isinstance(headers, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()):
        raise ValueError('headers must be a string mapping')
    raise SystemExit(asyncio.run(run(args, headers)))


if __name__ == '__main__':
    main()
