import asyncio
import json

import httpx
import pytest

from evidence_rag.answer import OpenAIStream, ModelStreamIncomplete

EVIDENCE=[{'citation_id':'E1','title':'Synthetic','text':'Synthetic source','profile':'general'}]

def successful_response():
    return httpx.Response(200,content=(
        'data: {"choices":[{"delta":{"content":"SUPPORT [E1]"},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
        'data: [DONE]\n\n'))

def provider(handler):
    return OpenAIStream('http://fixture','fixture','private-fixture',timeout=60,
                        transport=httpx.MockTransport(handler),include_usage=True,
                        max_output_tokens=8192,thinking='enabled',reasoning_effort='low',
                        empty_retry_thinking='disabled',retry_timeout=45,retry_max_output_tokens=2048)

@pytest.mark.asyncio
async def test_empty_timeout_retries_once_with_explicit_fallback_configuration():
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests)==1:raise httpx.ReadTimeout('synthetic timeout')
        return successful_response()
    events=[event async for event in provider(handler).stream_events('question',EVIDENCE)]
    assert len(requests)==2
    assert requests[0]['thinking']=={'type':'enabled'} and requests[0]['max_tokens']==8192
    assert requests[1]['thinking']=={'type':'disabled'} and requests[1]['max_tokens']==2048
    assert requests[0]['messages']==requests[1]['messages']
    assert [event['attempt'] for event in events if event['type']=='generation_attempt']==[1,2]
    retry=next(event for event in events if event['type']=='generation_retry')
    assert retry['reason']=='timeout' and retry['discard_pending_answer']
    assert ''.join(event['delta'] for event in events if event['type']=='delta')=='SUPPORT [E1]'

@pytest.mark.asyncio
async def test_empty_length_retains_first_attempt_usage():
    calls=0
    def handler(request):
        nonlocal calls
        calls+=1
        if calls==1:
            return httpx.Response(200,content=(
                'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":8192,"total_tokens":8202}}\n\n'
                'data: [DONE]\n\n'))
        return successful_response()
    events=[event async for event in provider(handler).stream_events('question',EVIDENCE)]
    usages=[event for event in events if event['type']=='usage']
    assert [(event['attempt'],event['usage']['completion_tokens']) for event in usages]==[(1,8192),(2,5)]
    assert next(event for event in events if event['type']=='generation_retry')['reason']=='length'

@pytest.mark.asyncio
@pytest.mark.parametrize('reason',['length','content_filter','tool_calls'])
async def test_partial_answers_and_restricted_finishes_never_retry(reason):
    calls=0
    def handler(request):
        nonlocal calls
        calls+=1
        delta={'content':'Partial [E1]'} if reason=='length' else {}
        return httpx.Response(200,content='data: '+json.dumps({'choices':[{'delta':delta,'finish_reason':reason}]})+'\n\ndata: [DONE]\n\n')
    with pytest.raises(ModelStreamIncomplete):
        _=[event async for event in provider(handler).stream_events('question',EVIDENCE)]
    assert calls==1

@pytest.mark.asyncio
async def test_second_timeout_has_no_third_attempt():
    calls=0
    def handler(request):
        nonlocal calls
        calls+=1
        raise httpx.ReadTimeout('synthetic timeout')
    with pytest.raises(httpx.ReadTimeout):
        _=[event async for event in provider(handler).stream_events('question',EVIDENCE)]
    assert calls==2

@pytest.mark.asyncio
async def test_cancellation_does_not_retry():
    calls=0
    def handler(request):
        nonlocal calls
        calls+=1
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        _=[event async for event in provider(handler).stream_events('question',EVIDENCE)]
    assert calls==1

@pytest.mark.parametrize('first_mode',['length','timeout'])
def test_api_retry_commits_one_turn_and_reports_known_usage(tmp_path,monkeypatch,first_mode):
    from fastapi.testclient import TestClient
    import evidence_rag.api as api
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'))
    monkeypatch.setenv('EVIDENCE_RAG_AUTH','{"fixture":"user"}')
    monkeypatch.setenv('EVIDENCE_RAG_STORE_CONVERSATIONS','1')
    async def bundle(*args,**kwargs):
        return {'trace_id':'fixture','status':'ready','corpus_revision':'r1',
                'query_plan':{'mode':'source_lookup','profile':'general','applicability_review':False},
                'evidence':EVIDENCE,'budget':{'state':'measured','omitted':[]},'degraded':False}
    calls=0
    def handler(request):
        nonlocal calls
        calls+=1
        if calls==1:
            if first_mode=='timeout':raise httpx.ReadTimeout('synthetic timeout')
            return httpx.Response(200,content=(
                'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":8190,"total_tokens":8200}}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":8192,"total_tokens":8202}}\n\n'
                'data: [DONE]\n\n'))
        return successful_response()
    monkeypatch.setattr(api,'get_bundle',bundle)
    monkeypatch.setattr(api,'generation_allowed',lambda *args:True)
    with TestClient(api.app) as client:
        cat=api.app.state.catalog
        cat.stage_documents([{'_id':'d1','title':'Synthetic','text':'Synthetic source'}],
                            {'corpus_id':'fixture','version_id':'v1','source_kind':'synthetic','verification_state':'provided'},'r1')
        cat.activate('r1',None)
        api.app.state.generator=provider(handler)
        response=client.post('/api/v2/chat/stream',json={'query':'question','session_id':'fixture'},headers={'Authorization':'Bearer fixture'})
        events=[]
        for block in response.text.strip().split('\n\n'):
            lines=block.splitlines()
            events.append({'event':lines[0][7:],'data':json.loads(lines[1][6:])})
        final=next(event['data'] for event in events if event['event']=='completed')
        assert final['status']=='ready' and len(final['generation_attempts'])==2
        assert len([event for event in events if event['event']=='generation_retry'])==1
        assert cat.db.execute('SELECT count(*) FROM turns').fetchone()[0]==1
        expected_completion=8197 if first_mode=='length' else 5
        assert final['usage']['completion_tokens']==expected_completion
        assert final['usage_complete'] is (first_mode=='length')
        from benchmarks.run_user_flow import generation_metadata
        metadata=generation_metadata(events)
        assert metadata['usage']==final['usage']
        assert metadata['usage_complete']==final['usage_complete']
        assert len(metadata['generation_attempts'])==2
        assert all(event['data']['elapsed_ms']>=0 for event in events if event['event'] in ('generation_attempt','generation_retry'))

@pytest.mark.asyncio
async def test_timeout_after_partial_text_never_retries():
    calls=0
    class PartialStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Partial [E1]"},"finish_reason":null}]}\n\n'
            raise httpx.ReadTimeout('synthetic timeout')
    def handler(request):
        nonlocal calls
        calls+=1
        return httpx.Response(200,stream=PartialStream())
    with pytest.raises(httpx.ReadTimeout):
        _=[event async for event in provider(handler).stream_events('question',EVIDENCE)]
    assert calls==1

@pytest.mark.parametrize('changed', [
    {'thinking':'disabled'}, {'max_output_tokens':None},
    {'retry_timeout':0}, {'retry_max_output_tokens':8193},
])
def test_invalid_retry_configuration_fails_before_any_request(changed):
    options={'max_output_tokens':8192,'thinking':'enabled','empty_retry_thinking':'disabled',
             'retry_timeout':45,'retry_max_output_tokens':2048}
    with pytest.raises(ValueError):
        OpenAIStream('http://fixture','fixture','fixture',**{**options,**changed})
