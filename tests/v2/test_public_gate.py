import json
from types import SimpleNamespace

import pytest

from benchmarks import public_gate as gate
from benchmarks.prepare import filehash


def fixture_data(tmp_path):
    root = tmp_path / 'data'; (root / 'qrels').mkdir(parents=True)
    (root / 'corpus.jsonl').write_text(''.join(json.dumps({'_id': x, 'title': '', 'text': x}) + '\n' for x in ['a', 'b']))
    (root / 'queries.jsonl').write_text(''.join(json.dumps({'_id': x, 'text': x}) + '\n' for x in ['q1', 'q2']))
    (root / 'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\nq1\ta\t2\nq1\tb\t1\nq2\tb\t1\n')
    manifest = {'corpus_sha256': filehash(root / 'corpus.jsonl'), 'queries_sha256': filehash(root / 'queries.jsonl'), 'corpus_count': 2, 'query_count_total': 2, 'qrels': {'test': {'sha256': filehash(root / 'qrels/test.tsv'), 'qids': 2}}}
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest))
    return root, path


def response():
    return {'corpus_revision': 'r1', 'degraded': False, 'degraded_reasons': [],
            'evidence': [{'document_id': 'a'}, {'document_id': 'a'}, {'document_id': 'b'}],
            'retrieval_trace': {'dense_backend': 'openviking', 'dense_hits': [{'uri': 'viking://doc'}], 'timings_ms': {'dense_ms': 2, 'rank_and_rerank_ms': 3}, 'candidates': [{'reranker_score': .5}]}}


def args(tmp_path, max_queries=None):
    root, manifest = fixture_data(tmp_path)
    return SimpleNamespace(data_dir=root, manifest=manifest, baseline='http://127.0.0.1:9000/api/v2/retrieve', candidate='http://127.0.0.1:9001/api/v2/retrieve', revision='r1', output_dir=tmp_path / 'run', timeout=5, seed=1, max_queries=max_queries, baseline_rerank_limit=60, candidate_rerank_limit=60)


def test_strict_decimal_boundaries():
    assert not gate.decision(100, 90, .5, .5)
    assert gate.decision(100, 89.999, .5, .5)
    assert not gate.decision(100, 100, .5, .55)
    assert gate.decision(100, 100, .5, .55001)
    assert not gate.decision(0, 0, .5, .9)
    assert not gate.decision(100, 50, 0, 1)


def test_paired_run_and_dedupe(tmp_path, monkeypatch):
    a = args(tmp_path); seen = []
    def fake(url, payload, timeout, token):
        seen.append((url, payload))
        return (100 if ':9000/' in url else 80), 200, json.dumps(response()), response(), None
    monkeypatch.setattr(gate, 'request', fake)
    result = gate.run(a)
    rows = [json.loads(x) for x in (a.output_dir / 'trials.jsonl').read_text().splitlines()]
    assert len(rows) == 4 and len(seen) == 6 and all(x[1]['max_evidence_items'] == 10 for x in seen)
    assert len((a.output_dir / 'warmups.jsonl').read_text().splitlines()) == 2
    assert {tuple(rows[i]['order']) for i in (0, 2)} == {('baseline', 'candidate'), ('candidate', 'baseline')}
    assert result['metrics']['baseline']['p50_ms'] == 100
    assert result['metrics']['baseline']['mean_mrr_at_10'] > 0
    assert rows[0]['query_id'] == rows[1]['query_id'] and rows[2]['query_id'] == rows[3]['query_id']
    assert result['eligible'] and result['passed']
    assert next(x for x in rows if x['query_id'] == 'q1')['ndcg_at_10'] == 1
    assert (a.output_dir / 'config.json').exists()


def test_development_and_failures_cannot_qualify(tmp_path, monkeypatch):
    a = args(tmp_path, 1)
    monkeypatch.setattr(gate, 'request', lambda *a: (22, 503, 'bad', None, 'HTTP 503'))
    result = gate.run(a)
    assert not result['eligible'] and not result['passed']
    assert result['metrics']['baseline']['p95_ms'] == 22
    assert len((a.output_dir / 'trials.jsonl').read_text().splitlines()) == 2


def test_manifest_and_trace_checks(tmp_path):
    root, manifest = fixture_data(tmp_path)
    gate.load_data(root, manifest)
    (root / 'corpus.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        gate.load_data(root, manifest)
    good = response(); assert gate.trace_error(good, 'r1', 60) is None
    good['retrieval_trace']['candidates'][0]['reranker_score'] = None
    assert gate.trace_error(good, 'r1', 60)
    good = response(); good['degraded'] = True
    assert gate.trace_error(good, 'r1', 60)
    good = response(); del good['retrieval_trace']['timings_ms']['dense_ms']
    assert gate.trace_error(good, 'r1', 60)
    with pytest.raises(ValueError): gate.loopback_url('http://example.com:123/api/v2/retrieve')


def test_window_and_dense_validation():
    good = response()
    good['retrieval_trace']['candidates'] = [{'reranker_score': .8}, {'reranker_score': .7}, {'reranker_score': None}]
    assert gate.trace_error(good, 'r1', 2) is None
    assert gate.trace_error(good, 'r1', 3)
    good['retrieval_trace']['candidates'][0]['reranker_score'] = None
    assert gate.trace_error(good, 'r1', 2)
    good = response(); good['retrieval_trace']['dense_hits'] = []
    assert gate.trace_error(good, 'r1', 60)


def test_warmup_failure_invalidates_formal_run(tmp_path, monkeypatch):
    a = args(tmp_path)
    def fake(url, payload, timeout, token):
        if payload['session_id'].startswith('public_warmup_'):
            return 4, 503, 'failed', None, 'HTTP 503'
        return 20, 200, json.dumps(response()), response(), None
    monkeypatch.setattr(gate, 'request', fake)
    result = gate.run(a)
    assert not result['eligible'] and result['warmup_failures'] == 2
    assert result['metrics']['baseline']['failures'] == 0
