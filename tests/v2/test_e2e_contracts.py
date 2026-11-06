import json
import httpx
import pytest

from evidence_rag.answer import OpenAIStream
from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import OpenViking, RetrievalService


@pytest.mark.asyncio
@pytest.mark.parametrize('thinking', ['disabled', 'enabled'])
async def test_provider_receives_explicit_thinking_configuration(thinking):
    async def handler(request):
        body = json.loads(request.content)
        assert body['thinking'] == {'type': thinking}
        assert body['max_tokens'] == 4096
        assert body['reasoning_effort'] == 'high'
        return httpx.Response(200, content='data: {"choices":[{"delta":{"content":"SUPPORT [E1]"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    provider = OpenAIStream('http://fixture', 'fixture', 'unused', transport=httpx.MockTransport(handler),
                            max_output_tokens=4096, thinking=thinking, reasoning_effort='high')
    evidence = [{'citation_id': 'E1', 'title': 'Fixture', 'text': 'Synthetic evidence'}]
    assert ''.join([delta async for delta in provider.stream('question', evidence)]) == 'SUPPORT [E1]'


@pytest.mark.asyncio
async def test_truncated_generation_preserves_usage_before_failure():
    body = 'data: '+json.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}],
                                'usage': {'prompt_tokens': 20, 'completion_tokens': 1024, 'total_tokens': 1044,
                                          'completion_tokens_details': {'reasoning_tokens': 1024}}})+'\n\ndata: [DONE]\n\n'
    provider = OpenAIStream('http://fixture', 'fixture', 'unused',
                            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))
    events = []
    with pytest.raises(RuntimeError):
        async for event in provider.stream_events('question', [{'citation_id': 'E1', 'title': 'Fixture', 'text': 'Synthetic'}]):
            events.append(event)
    assert events == [{'type': 'usage', 'usage': {'prompt_tokens': 20, 'completion_tokens': 1024,
                                                 'total_tokens': 1044, 'reasoning_tokens': 1024}}]


@pytest.mark.asyncio
async def test_truncated_stream_retains_separate_terminal_usage_packet():
    body = ('data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":1024,"total_tokens":1044}}\n\n'
            'data: [DONE]\n\n')
    provider = OpenAIStream('http://fixture', 'fixture', 'unused',
                            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))
    events = []
    with pytest.raises(RuntimeError):
        async for event in provider.stream_events('question', [{'citation_id':'E1','title':'Fixture','text':'Synthetic'}]):
            events.append(event)
    assert events == [{'type':'usage','usage':{'prompt_tokens':20,'completion_tokens':1024,'total_tokens':1044}}]


@pytest.mark.asyncio
async def test_trace_supports_legal_units_without_generic_document_metadata(tmp_path):
    catalog = Catalog(tmp_path / 'catalog.sqlite3')
    try:
        catalog.stage(['第一条 合同成立。'], {'doc_id': 'fixture', 'version_id': 'v1', 'title': '测试法',
                      'verification_state': 'approved', 'source_kind': 'synthetic'}, 'r1')
        catalog.activate('r1', None)
        result = await RetrievalService(catalog, profile='legal').retrieve('第一条 原文', return_retrieval_trace=True)
        assert result['evidence'][0]['article_label'] == '第一条'
        assert result['retrieval_trace']['candidates'][0]['document_id'] is None
    finally:
        catalog.db.close()


@pytest.mark.asyncio
async def test_prepared_unit_is_not_split_again_by_openviking():
    uid = 'a' * 32
    scope = 'viking://resources/rag/fixture/r1/'
    async def handler(request):
        if request.url.path.endswith('temp_upload'):
            return httpx.Response(200, json={'status': 'ok', 'result': {'temp_file_id': 'fixture'}})
        body = json.loads(request.content)
        assert body['args'] == {'parse_mode': 'no_split'}
        assert body['to'] == scope + uid
        assert body['processing_mode'] == 'vectors_only'
        return httpx.Response(200, json={'status': 'ok', 'result': {'root_uri': scope + uid}})
    receipt = await OpenViking('http://fixture', scope, transport=httpx.MockTransport(handler)).ingest(uid, 'Prepared complete paragraph')
    assert receipt['expected_l2_uri'] == scope + uid + '/' + uid + '.md'


@pytest.mark.asyncio
async def test_background_ranking_has_materialized_sources_before_executor(tmp_path):
    catalog = Catalog(tmp_path / 'catalog.sqlite3')
    catalog.stage_documents([{'_id': 'd1', 'title': 'Synthetic', 'text': 'alpha evidence'}],
                            {'corpus_id': 'fixture', 'version_id': 'v1', 'source_kind': 'synthetic',
                             'verification_state': 'provided'}, 'r1')
    catalog.activate('r1', None)
    class Ranker:
        def rank(self, query, candidates):
            return [({**item, 'rerank_part_no': item['parts'][0]['part_no']}, 1.0) for item in candidates]
    class Executor:
        async def run(self, operation, timeout):
            # Simulate a cancelled request closing its request-scoped connection
            # while the background operation still exists. Only model work may remain.
            catalog.db.close()
            return operation()
    result = await RetrievalService(catalog, reranker=Ranker(), profile='general',
                                     ranking_executor=Executor()).retrieve('alpha')
    assert 'reranker_failed' not in result['degraded_reasons']
    assert result['evidence'][0]['document_id'] == 'd1'
