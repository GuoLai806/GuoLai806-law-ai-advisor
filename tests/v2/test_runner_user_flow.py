import hashlib
import io
import json
from functools import partial
from types import SimpleNamespace

import httpx
import pytest

from benchmarks import run_user_flow
from benchmarks.e2e import file_hash, source_hashes


@pytest.mark.asyncio
@pytest.mark.parametrize('session_status', [200, 500])
async def test_invalid_classification_still_checks_real_session_path(tmp_path, monkeypatch, session_status):
    data = tmp_path / 'data'
    (data / 'qrels').mkdir(parents=True)
    (data / 'queries.jsonl').write_text(json.dumps({'_id': 'q1', 'text': 'Synthetic question'}) + '\n')
    (data / 'corpus.jsonl').write_text(json.dumps({'_id': 'd1', 'text': 'Synthetic source'}) + '\n')
    relevance = data / 'qrels/train.tsv'
    relevance.write_text('query-id\tcorpus-id\tscore\nq1\td1\t1\n')
    claims = tmp_path / 'claims.jsonl'
    claims.write_text(json.dumps({'id': 'q1', 'claim': 'Synthetic question', 'evidence': {'d1': [{'label': 'SUPPORT'}]}}) + '\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'validation': {'source_split': 'train', 'query_ids': ['q1']},
                                    'qrels': {'train': {'sha256': file_hash(relevance)}},
                                    'corpus_sha256': file_hash(data / 'corpus.jsonl'),
                                    'queries_sha256': file_hash(data / 'queries.jsonl')}))
    identity = {'dense_scope': 'viking://resources/fixture/r1/'}
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'source_sha256': source_hashes(), 'claims_sha256': {'validation': file_hash(claims)},
                                 'max_evidence_items': 1, 'runtime_identity': identity,
                                 'service_configuration': {'store_conversations': '1'}}))
    answer = 'Evidence appears relevant [E1].'  # No explicit verdict; never infer SUPPORT.
    evidence = {'unit_id': 'u1', 'document_id': 'd1', 'citation_id': 'E1',
                'content_sha256': hashlib.sha256(b'Synthetic source').hexdigest()}
    bundle = {'corpus_revision': 'r1', 'budget': {}, 'evidence': [evidence],
              'retrieval_trace': {'dense_backend': 'openviking', **identity,
                                  'candidates': [{'document_id': 'd1'}]}}
    session_reads = []
    request_body = {}

    async def handler(request):
        if request.url.path == '/api/v2/models':
            return httpx.Response(200, json={'runtime_identity': identity})
        if request.url.path == '/health/ready':
            return httpx.Response(200, json={'ready': True})
        if request.url.path == '/api/v2/chat/stream':
            request_body.update(json.loads(request.content))
            events = [('evidence_ready', {'bundle': bundle}), ('answer_delta', {'delta': answer}),
                      ('verification', {'status': 'structurally_valid'}), ('completed', {'status': 'ready'})]
            return httpx.Response(200, text=''.join('event: '+kind+'\ndata: '+json.dumps(payload)+'\n\n' for kind, payload in events))
        if request.url.path == '/api/v2/units/u1':
            return httpx.Response(200, json={'text': 'Synthetic source'})
        if request.url.path.endswith('/turns'):
            session_reads.append(request.url.path)
            return httpx.Response(session_status, json={'turns': [{'query': request_body['query'], 'answer': answer}]})
        raise AssertionError(str(request.url))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', partial(real_client, transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(run_user_flow.sys, 'stdin', io.StringIO(json.dumps({'token': 'synthetic-local-token'})))
    output = tmp_path / 'report.json'
    args = SimpleNamespace(output=str(output), manifest=str(manifest), config=str(config), data_dir=str(data),
                           claims=str(claims), split='validation', raw_dir=str(tmp_path / 'raw'), url='http://fixture')
    assert await run_user_flow.execute(args) == 1
    report = json.loads(output.read_text())
    row = report['rows'][0]
    assert len(session_reads) == 1
    assert row['failure']['type'] == 'invalid_answer_format'
    assert row['predicted_label'] is None
    assert report['verification_metrics']['claim_label_accuracy'] == 0
    assert row['user_flow_completed'] is (session_status == 200)
    assert report['user_flow_metrics']['completed'] == int(session_status == 200)
    if session_status == 500:
        assert row['flow_failure']['type'] == 'HTTPStatusError'
