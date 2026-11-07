import json
from pathlib import Path

import pytest

from benchmarks.run_openviking_connections import (
    VARIANTS, load_queries, order_for, percentile, run_comparison, summarize,
)
from benchmarks.e2e import file_hash


def test_counterbalanced_order_is_serial_and_equal():
    assert order_for(0) == ('per_operation', 'shared')
    assert order_for(1) == ('shared', 'per_operation')
    assert [order_for(i)[0] for i in range(30)].count('shared') == 15


def request(elapsed, success=True, fingerprint='same'):
    result = {'elapsed_ms': elapsed, 'success': success}
    if success:
        result['rows_sha256'] = fingerprint
    return result


def test_failures_stay_in_pair_and_request_denominators():
    pairs = [
        {'requests': {'per_operation': request(10), 'shared': request(6)}},
        {'requests': {'per_operation': request(20, False), 'shared': request(12)}},
        {'requests': {'per_operation': request(8), 'shared': request(9, fingerprint='changed')}},
    ]
    summary = summarize(pairs)
    assert summary['query_count'] == 3
    assert summary['requests_expected'] == 6
    assert summary['failed_pairs'] == 1
    assert summary['all_successful_pairs'] == 2
    assert summary['exact_result_equal_pairs'] == 1
    assert summary['result_mismatch_pairs'] == 1
    assert summary['variants']['per_operation']['failures'] == 1
    assert summary['variants']['per_operation']['observed_latency_ms']['n'] == 3
    assert summary['variants']['per_operation']['successful_latency_ms']['n'] == 2
    assert summary['paired_shared_minus_per_operation_ms']['n'] == 2
    assert summary['paired_shared_minus_per_operation_ms']['mean'] == -1.5


def test_percentile_definition_is_explicit():
    assert percentile([], .95) is None
    assert percentile([7], .95) == 7
    assert percentile([0, 10], .95) == 9.5
    assert percentile([1, 2, 3], .5) == 2


class JournalRecorder:
    def __init__(self):
        self.records = []

    def record(self, record):
        self.records.append(record)


class FakeAdapter:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    async def find(self, query):
        self.calls.append((self.name, query))
        if self.name == 'shared' and query == 'second':
            raise RuntimeError('Do not serialize credential http://secret:password@private.example')
        return [{'uri': 'viking://resources/rag/scifact/r1/abc'}]


@pytest.mark.asyncio
async def test_warmups_are_separate_failures_continue_and_exception_text_is_not_recorded():
    calls = []
    adapters = {name: FakeAdapter(name, calls) for name in VARIANTS}
    journal = JournalRecorder()
    warmups, pairs = await run_comparison([('q1', 'first'), ('q2', 'second')], adapters, journal)
    assert calls == [('per_operation', 'first'), ('shared', 'first'),
                     ('per_operation', 'first'), ('shared', 'first'),
                     ('shared', 'second'), ('per_operation', 'second')]
    assert len(warmups) == 2 and len(pairs) == 2
    assert len(journal.records) == 6
    assert pairs[1]['requests']['shared']['error_type'] == 'RuntimeError'
    assert pairs[1]['exact_results_equal'] is None
    assert summarize(pairs)['failed_pairs'] == 1
    serialized = json.dumps(journal.records)
    assert 'private.example' not in serialized and 'password' not in serialized
    assert 'query_sha256' in serialized and '"query"' not in serialized
    for entry in journal.records:
        assert entry['finished_wall_ns'] >= entry['started_wall_ns']
        assert entry['elapsed_ms'] >= 0


def fixture_dataset(tmp_path):
    data = tmp_path / 'scifact'
    data.mkdir()
    (data / 'qrels').mkdir()
    ids = [str(i) for i in range(36)]
    (data / 'corpus.jsonl').write_text(json.dumps({'_id': 'd', 'title': 'T', 'text': 'B'}) + '\n')
    (data / 'queries.jsonl').write_text(''.join(json.dumps({'_id': qid, 'text': 'query ' + qid}) + '\n' for qid in ids))
    for name, subset in [('train', ids[:6]), ('test', ids[6:])]:
        (data / 'qrels' / (name + '.tsv')).write_text('query-id\tcorpus-id\tscore\n' + ''.join(qid + '\td\t1\n' for qid in subset))
    manifest = {'dataset': 'scifact', 'short_evaluation_protocol': 'scifact-stratified-30-v1',
                'corpus_count': 1, 'query_count_total': 36,
                'corpus_sha256': file_hash(data / 'corpus.jsonl'),
                'queries_sha256': file_hash(data / 'queries.jsonl'),
                'qrels': {name: {'sha256': file_hash(data / 'qrels' / (name + '.tsv')), 'qids': count}
                          for name, count in [('train', 6), ('test', 30)]},
                'validation': {'query_ids': ids[:6], 'count': 6, 'source_split': 'train'},
                'short_evaluation': {'query_ids': ids[6:], 'count': 30, 'source_split': 'test'}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return data, path


def test_frozen_manifest_membership_and_dataset_bytes_are_checked(tmp_path):
    data, manifest = fixture_dataset(tmp_path)
    _, queries = load_queries(data, manifest)
    assert len(queries) == 30 and queries[0] == ('6', 'query 6')
    (data / 'queries.jsonl').write_text((data / 'queries.jsonl').read_text() + '\n')
    with pytest.raises(ValueError, match='byte identity'):
        load_queries(data, manifest)


def test_duplicate_or_missing_frozen_ids_are_rejected(tmp_path):
    data, manifest = fixture_dataset(tmp_path)
    value = json.loads(manifest.read_text())
    value['short_evaluation']['query_ids'][0] = value['short_evaluation']['query_ids'][1]
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='size/source'):
        load_queries(data, manifest)
