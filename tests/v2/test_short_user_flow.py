import io
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from benchmarks import run_user_flow
from benchmarks.e2e import file_hash, source_hashes
from tools.prepare_user_flow_subset import build_manifest, stratified_ids


@pytest.fixture
def full_scifact_fixture(tmp_path):
    data = tmp_path / 'data'
    (data / 'qrels').mkdir(parents=True)
    (data / 'corpus.jsonl').write_text(json.dumps({'_id': 'd1', 'text': 'Fixture source'}) + '\n')
    queries = []
    paths = {}
    ids = {}
    for split, each in [('train', 14), ('test', 100)]:
        rows = []
        ids[split] = []
        for label in ['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO']:
            for index in range(each):
                identifier = split + '-' + label + '-' + str(index)
                ids[split].append(identifier)
                claim = 'Fixture claim ' + identifier
                queries.append({'_id': identifier, 'text': claim})
                evidence = {} if label == 'NOT_ENOUGH_INFO' else {'d1': [{'label': label}]}
                rows.append({'id': identifier, 'claim': claim, 'evidence': evidence})
        paths[split] = tmp_path / (split + '-claims.jsonl')
        paths[split].write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (data / 'qrels' / (split + '.tsv')).write_text('query-id\tcorpus-id\tscore\n' + ''.join(identifier + '\td1\t1\n' for identifier in ids[split]))
    (data / 'queries.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in queries))
    manifest = {'version': 1, 'dataset': 'scifact', 'corpus_count': 1,
                'query_count_total': len(queries), 'corpus_sha256': file_hash(data / 'corpus.jsonl'),
                'queries_sha256': file_hash(data / 'queries.jsonl'), 'official_test_count': 300,
                'qrels': {split: {'qids': len(ids[split]), 'sha256': file_hash(data / 'qrels' / (split + '.tsv'))} for split in paths},
                'validation': {'source_split': 'train', 'query_ids': ids['train'][:6], 'count': 6}}
    original = tmp_path / 'original.json'
    original.write_text(json.dumps(manifest))
    short = tmp_path / 'short.json'
    build_manifest(original, data, paths['train'], paths['test'], short)
    return SimpleNamespace(data=data, claims=paths, ids=ids, original=original, short=short, tmp=tmp_path)


def test_subset_is_fixed_balanced_and_preserves_original(full_scifact_fixture):
    f = full_scifact_fixture
    original = f.original.read_bytes()
    second = f.tmp / 'second.json'
    result = build_manifest(f.original, f.data, f.claims['train'], f.claims['test'], second)
    assert result == json.loads(f.short.read_text())
    assert f.original.read_bytes() == original
    assert result['official_test_count'] == 300
    assert result['validation']['count'] == 6
    assert result['short_evaluation']['count'] == 30
    assert result['validation']['label_counts'] == {'SUPPORT': 2, 'CONTRADICT': 2, 'NOT_ENOUGH_INFO': 2}
    assert result['short_evaluation']['label_counts'] == {'SUPPORT': 10, 'CONTRADICT': 10, 'NOT_ENOUGH_INFO': 10}
    assert len(set(result['short_evaluation']['query_ids'])) == 30
    assert set(result['validation']['query_ids']).isdisjoint(result['short_evaluation']['query_ids'])
    assert result['short_evaluation']['claims_sha256'] == file_hash(f.claims['test'])
    with pytest.raises(FileExistsError):
        build_manifest(f.original, f.data, f.claims['train'], f.claims['test'], second)


def test_subset_refuses_changed_data_and_missing_class(full_scifact_fixture):
    f = full_scifact_fixture
    with pytest.raises(ValueError, match='Insufficient'):
        stratified_ids({'x': {'d1': 1}}, {'x': {'id': 'x', 'claim': 'Fixture', 'evidence': {}}}, 10)
    with (f.data / 'queries.jsonl').open('a') as stream:
        stream.write(json.dumps({'_id': 'extra', 'text': 'Changed'}) + '\n')
    with pytest.raises(ValueError, match='identity'):
        build_manifest(f.original, f.data, f.claims['train'], f.claims['test'], f.tmp / 'changed.json')


async def run_fixture_protocol(f, monkeypatch, manifest_path, *, selected_manifest_hash=None, fail_stream=False, split='test'):
    identity = {'dense_scope': 'viking://resources/fixture/r1/','answer_format':'text',
                'generation_gate':'reranker_threshold','rerank_threshold':'0','system_prompt_sha256':'fixture-text'}
    config = f.tmp / ('config-' + manifest_path.stem + '.json')
    config.write_text(json.dumps({'source_sha256': source_hashes(), 'claims_sha256': {'test': file_hash(f.claims['test']), 'validation': file_hash(f.claims['train'])},
                                 'runtime_identity': identity, 'service_configuration': {'store_conversations': '0',
                                  'answer_format':'text','generation_gate':'reranker_threshold','rerank_threshold':0}, 'max_evidence_items': 1}))
    selection = f.tmp / ('selection-' + manifest_path.stem + '.json')
    selection.write_text(json.dumps({'config_sha256': file_hash(config), 'manifest_sha256': selected_manifest_hash or file_hash(manifest_path)}))
    calls = []

    async def handler(request):
        if request.url.path == '/api/v2/models':
            return httpx.Response(200, json={'runtime_identity': identity})
        if request.url.path == '/health/ready':
            return httpx.Response(200, json={'ready': True})
        if request.url.path == '/api/v2/chat/stream':
            calls.append(json.loads(request.content)['query'])
            if fail_stream:
                return httpx.Response(503, json={'error': 'Fixture failure'})
            bundle = {'corpus_revision': 'r1', 'budget': {}, 'evidence': [],
                      'retrieval_trace': {'dense_backend': 'openviking', **identity, 'candidates': [{'document_id': 'd1'}]}}
            events = [('evidence_ready', {'bundle': bundle}), ('answer_delta', {'delta': 'NOT_ENOUGH_INFO [E1]'}),
                      ('verification', {'status': 'structurally_valid'}), ('completed', {'status': 'ready'})]
            return httpx.Response(200, text=''.join('event: ' + kind + '\ndata: ' + json.dumps(payload) + '\n\n' for kind, payload in events))
        raise AssertionError(str(request.url))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', partial(real_client, transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(run_user_flow.sys, 'stdin', io.StringIO(json.dumps({'token': 'fixture-token'})))
    output = f.tmp / ('report-' + manifest_path.stem + '.json')
    if split=='test' and 'short_evaluation' in json.loads(manifest_path.read_text()):
        from copy import deepcopy
        from tools.freeze_short_run import freeze
        from benchmarks.verification_metrics import gold_label
        baseline=json.loads(config.read_text());candidate=deepcopy(baseline)
        candidate['service_configuration'].update(answer_format='grounded_json',generation_gate='grounded_evidence')
        candidate['runtime_identity'].update(answer_format='grounded_json',generation_gate='grounded_evidence',system_prompt_sha256='fixture-grounded')
        dev_config=f.tmp/'fixture-dev.config.json';dev_config.write_text(json.dumps(candidate))
        manifest=json.loads(manifest_path.read_text()); dev_report=f.tmp/'fixture-dev.json'
        claims={str(r['id']):r for r in map(json.loads,f.claims['train'].read_text().splitlines())}
        rows=[{'query_id':qid,'gold_label':gold_label(claims[qid]),'failure':None,'user_flow_completed':True} for qid in manifest['validation']['query_ids']]
        report={'split':'validation','queries_expected':6,'queries_recorded':6,'rows':rows,'failures':0,
                'source_identity_checked':True,'source_unchanged':True,'runtime_identity_unchanged':True,
                'source_sha256':source_hashes(),'manifest_sha256':file_hash(manifest_path),
                'config_sha256':file_hash(dev_config),'runtime_identity':candidate['runtime_identity']}
        dev_report.write_text(json.dumps(report))
        entries=[{'type':'header','query_ids':manifest['validation']['query_ids'],**{k:report[k] for k in ('source_sha256','config_sha256','manifest_sha256')}},
                 *[{'type':'query','completed':i,'row':r} for i,r in enumerate(rows,1)],
                 {'type':'complete','queries':6,'failures':0,'report_sha256':file_hash(dev_report)}]
        Path(str(dev_report)+'.journal.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in entries))
        selection.unlink()  # Only this synthetic fixture's placeholder is replaced.
        freeze(SimpleNamespace(manifest=str(manifest_path),development_report=str(dev_report),development_config=str(dev_config),
                               baseline_config=str(config),candidate_config=str(dev_config),baseline_test_output=str(output),
                               candidate_test_output=str(f.tmp/'fixture-candidate.json'),baseline_selection=str(selection),
                               candidate_selection=str(f.tmp/'fixture-candidate-selection.json')))
        if selected_manifest_hash:
            stored=json.loads(selection.read_text());stored['manifest_sha256']=selected_manifest_hash;selection.write_text(json.dumps(stored))
    args = SimpleNamespace(output=str(output), manifest=str(manifest_path), config=str(config), data_dir=str(f.data),
                           claims=str(f.claims['test' if split == 'test' else 'train']), split=split, selection=str(selection), raw_dir=str(f.tmp / 'raw'), url='http://fixture')
    result = await run_user_flow.execute(args)
    return result, json.loads(output.read_text()), calls


@pytest.mark.asyncio
async def test_short_runner_sends_exactly_thirty_frozen_queries(full_scifact_fixture, monkeypatch):
    f = full_scifact_fixture
    _, report, calls = await run_fixture_protocol(f, monkeypatch, f.short)
    assert len(calls) == report['queries_recorded'] == report['queries_expected'] == 30
    assert [row['query_id'] for row in report['rows']] == json.loads(f.short.read_text())['short_evaluation']['query_ids']
    assert report['evaluation_protocol']['scope'].startswith('Public stratified small')


@pytest.mark.asyncio
async def test_original_runner_still_uses_all_three_hundred(full_scifact_fixture, monkeypatch):
    _, report, calls = await run_fixture_protocol(full_scifact_fixture, monkeypatch, full_scifact_fixture.original)
    assert len(calls) == report['queries_recorded'] == report['queries_expected'] == 300
    assert report['evaluation_protocol']['kind'] == 'full_test'


@pytest.mark.asyncio
async def test_short_validation_sends_exactly_six_train_queries(full_scifact_fixture, monkeypatch):
    f = full_scifact_fixture
    _, report, calls = await run_fixture_protocol(f, monkeypatch, f.short, split='validation')
    assert len(calls) == report['queries_recorded'] == report['queries_expected'] == 6
    assert [row['query_id'] for row in report['rows']] == json.loads(f.short.read_text())['validation']['query_ids']
    assert report['evaluation_protocol']['source_split'] == 'train'
    assert report['evaluation_protocol']['label_counts'] == {'SUPPORT': 2, 'CONTRADICT': 2, 'NOT_ENOUGH_INFO': 2}


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['duplicate', 'unknown', 'wrong_count', 'wrong_class', 'different_seed', 'different_member',
                                    'order', 'claims_identity', 'source_count', 'source_split', 'selection_rule'])
async def test_short_runner_refuses_invalid_subset_before_model_calls(full_scifact_fixture, monkeypatch, mutation):
    f = full_scifact_fixture
    changed = json.loads(f.short.read_text())
    block = changed['short_evaluation']
    if mutation == 'duplicate':
        block['query_ids'][1] = block['query_ids'][0]
    elif mutation == 'unknown':
        block['query_ids'][0] = 'not-in-source'
    elif mutation == 'wrong_count':
        block['count'] = 29
    elif mutation == 'wrong_class':
        block['label_counts']['SUPPORT'] = 11
    elif mutation == 'different_seed':
        block['seed'] = 'changed-after-freeze'
    elif mutation == 'different_member':
        current = block['query_ids'][0]
        label = current.split('-')[1]
        replacement = next(identifier for identifier in f.ids['test'] if identifier.split('-')[1] == label and identifier not in block['query_ids'])
        block['query_ids'][0] = replacement
    elif mutation == 'order':
        block['query_ids'].reverse()
    elif mutation == 'claims_identity':
        block['claims_sha256'] = 'changed'
    elif mutation == 'source_count':
        block['full_source_query_count'] = 299
    elif mutation == 'source_split':
        block['source_split'] = 'train'
    else:
        block['selection'] = 'take the easiest claims'
    altered = f.tmp / 'altered.json'
    altered.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        await run_fixture_protocol(f, monkeypatch, altered)
    assert not (f.tmp / 'report-altered.json.journal.jsonl').exists()


@pytest.mark.asyncio
async def test_short_runner_requires_manifest_bound_selection(full_scifact_fixture, monkeypatch):
    f = full_scifact_fixture
    with pytest.raises(ValueError, match='manifest'):
        await run_fixture_protocol(f, monkeypatch, f.short, selected_manifest_hash='wrong')
