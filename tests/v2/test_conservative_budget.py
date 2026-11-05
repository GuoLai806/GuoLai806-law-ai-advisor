import json
import pytest
from evidence_rag.answer import OpenAIStream, messages_for
from evidence_rag.budget import ConservativeByteBudget, assemble


def item(uid,text):
    return {'citation_id':'E'+uid,'unit_id':uid,'document_id':'doc-'+uid,'title':'资料','text':text,'profile':'general'}

def test_conservative_bytes_whole_units_and_reserves():
    e=[item('1','完整段落，不得截断。'),item('2','另一段内容。')]
    b=ConservativeByteBudget(10000,300,700)
    exact=b.measure_content(messages_for('问题',e[:1]))
    selected,info=assemble(e,b,exact,'问题')
    assert selected==e[:1] and info['omitted']==['2']
    assert info['state']=='conservative_bytes' and info['exact_tokens'] is False and info['used_tokens'] is None
    assert info['content_bytes']==exact and info['content_bytes']+info['framing_reserve']+info['output_reserve']<=info['model_context_window']
    assert 'not formally bounded' in info['guarantee']
    with pytest.raises(ValueError):ConservativeByteBudget(1000,0,1)
    with pytest.raises(ValueError):ConservativeByteBudget(1000,600,400)

@pytest.mark.asyncio
async def test_usage_packet_with_choices_is_preserved_and_not_secret():
    import httpx
    async def handler(request):
        body=json.loads(request.content);assert body['stream_options']=={'include_usage':True}
        return httpx.Response(200,content='data: {"choices":[{"delta":{"content":"Fact [E1]"}}],"usage":{"prompt_tokens":17,"completion_tokens":5,"total_tokens":22}}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    provider=OpenAIStream('http://test/v1','fixture','fake-key',transport=httpx.MockTransport(handler),include_usage=True)
    events=[e async for e in provider.stream_events('question',[item('1','evidence')])]
    assert events==[{'type':'delta','delta':'Fact [E1]'},{'type':'usage','usage':{'prompt_tokens':17,'completion_tokens':5,'total_tokens':22}}]
    assert 'fake-key' not in json.dumps(events)

def test_api_conservative_opt_in_usage_and_legal_guard(tmp_path,monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from evidence_rag.api import app
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('EVIDENCE_RAG_AUTH',json.dumps({'local-token':'u'}))
    monkeypatch.delenv('EVIDENCE_RAG_ALLOW_CONSERVATIVE_BYTES',raising=False)
    class Dense:
        scope='viking://resources/rag/probe/r1/'
        async def find(self,q):return []
    class Rerank:
        def rank(self,q,items):return [(x,0.9) for x in items]
    async def handler(request):
        return httpx.Response(200,content='data: {"choices":[{"delta":{"content":"Supported [E1]"}}],"usage":{"prompt_tokens":21,"completion_tokens":7,"total_tokens":28}}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    with TestClient(app) as client:
        cat=app.state.catalog
        cat.stage_documents([{'_id':'doc','title':'Synthetic','text':'A synthetic source paragraph.'}],{'corpus_id':'probe','version_id':'v1','source_kind':'synthetic','verification_state':'provided'},'r1');cat.activate('r1',None)
        uid=cat.exact_documents('r1','',doc_id='doc')[0];cat.db.execute('UPDATE units SET uri=? WHERE revision=? AND unit_id=?',('viking://resources/rag/probe/r1/'+uid+'/'+uid+'.md','r1',uid));cat.db.execute("INSERT INTO settings VALUES('dense_validated:r1','yes')")
        app.state.service.dense=Dense();app.state.service.reranker=Rerank();app.state.generator=OpenAIStream('http://test/v1','model','fake-key',transport=httpx.MockTransport(handler),max_output_tokens=512)
        body={'query':'show source','document_id':'doc','session_id':'s','budget_mode':'conservative_bytes'};headers={'Authorization':'Bearer local-token'}
        assert client.post('/api/v2/chat',json=body,headers=headers).status_code==503
        monkeypatch.setenv('EVIDENCE_RAG_ALLOW_CONSERVATIVE_BYTES','1');monkeypatch.setenv('EVIDENCE_RAG_MODEL_CONTEXT_WINDOW','16000');monkeypatch.setenv('EVIDENCE_RAG_FRAMING_RESERVE','1024');monkeypatch.setenv('EVIDENCE_RAG_OUTPUT_RESERVE','2048')
        response=client.post('/api/v2/chat',json=body,headers=headers);data=response.json()
        assert response.status_code==200 and data['answer']=='Supported [E1]' and data['usage']['total_tokens']==28
        assert data['budget']['state']=='conservative_bytes' and data['budget']['exact_tokens'] is False and data['verification']['claim_support']=='not_verified'
        stream=client.post('/api/v2/chat/stream',json=body,headers=headers)
        assert 'event: answer_delta' in stream.text and 'event: usage' in stream.text and '"total_tokens": 28' in stream.text
        assert client.post('/api/v2/chat',json={**body,'profile':'legal'},headers=headers).status_code==422
        strict=client.post('/api/v2/chat',json={k:v for k,v in body.items() if k!='budget_mode'},headers=headers).json()
        assert strict['verification']['status']=='not_generated' and strict['budget']['state']=='tokenizer_unavailable'
        ready=client.get('/health/ready').json();assert not ready['ready'] and ready['conservative_bytes_available']

def test_oversized_single_unit_is_omitted_as_whole():
    b=ConservativeByteBudget(2000,400,400)
    e=[item('1','Whole paragraph: '+'文'*1000)]
    selected,info=assemble(e,b,300,'query')
    assert not selected and info['omitted']==['1'] and info['used_tokens'] is None
    assert info['state']=='conservative_bytes' and info['content_bytes']<=info['max_content_bytes']

def test_insufficient_framing_is_visible():
    b=ConservativeByteBudget(2000,900,900)
    selected,info=assemble([item('1','text')],b,1,'query')
    assert not selected and info['fits'] is False and info['omitted']==['1']

def test_byte_mode_requires_bounded_output_and_keeps_threshold(monkeypatch):
    from types import SimpleNamespace
    from evidence_rag.api import Query,generation_allowed
    monkeypatch.setenv('EVIDENCE_RAG_ALLOW_CONSERVATIVE_BYTES','1');monkeypatch.setenv('EVIDENCE_RAG_MODEL_CONTEXT_WINDOW','16000');monkeypatch.setenv('EVIDENCE_RAG_FRAMING_RESERVE','1024');monkeypatch.setenv('EVIDENCE_RAG_OUTPUT_RESERVE','2048');monkeypatch.delenv('EVIDENCE_RAG_RERANK_THRESHOLD',raising=False);monkeypatch.delenv('LEGAL_RAG_RERANK_THRESHOLD',raising=False)
    body=Query(query='show',session_id='s',budget_mode='conservative_bytes')
    bundle={'status':'ready','query_plan':{'mode':'source_lookup','profile':'general','applicability_review':False},'evidence':[{'reranker_score':.9}],'degraded':False,'budget':{'state':'conservative_bytes','omitted':[]}}
    app=SimpleNamespace(state=SimpleNamespace(generator=SimpleNamespace(max_output_tokens=None)))
    assert not generation_allowed(bundle,body,app)
    app.state.generator.max_output_tokens=2049;assert not generation_allowed(bundle,body,app)
    app.state.generator.max_output_tokens=1024;assert generation_allowed(bundle,body,app)
    bundle['query_plan']['mode']='general_retrieval';assert not generation_allowed(bundle,body,app)

def test_source_document_expands_complete_paragraphs_before_budget():
    import hashlib
    text1='First complete paragraph.';text2='Second complete paragraph that does not fit.'
    full=item('1',text1+'\n\n'+text2);full['content_sha256']=hashlib.sha256(full['text'].encode()).hexdigest()
    def record(text,start,end,n):return {'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest(),'locator':{'char_start':start,'char_end':end},'part_no':n}
    p1=record(text1+'\n\n',0,len(text1)+2,1);p2=record(text2,len(text1)+2,len(full['text']),2)
    byte=ConservativeByteBudget(10000,100,100,paragraph_loader=lambda uid:{'parts':[p1,p2]})
    e1={**full,'text':p1['text'],'content_sha256':p1['sha256'],'source_locator':p1['locator'],'document_sha256':full['content_sha256'],'part_no':1,'has_more':True,'citation_id':'E1'}
    limit=byte.measure_content(messages_for('question',[e1]))
    selected,info=assemble([full],byte,limit,'question')
    assert len(selected)==1 and selected[0]['text']==p1['text'] and selected[0]['content_sha256']==p1['sha256']
    assert selected[0]['document_sha256']==full['content_sha256'] and info['omitted']==['1#part-2']

def test_legal_app_does_not_advertise_byte_mode(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from evidence_rag.api import legal_app
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('EVIDENCE_RAG_ALLOW_CONSERVATIVE_BYTES','1');monkeypatch.setenv('EVIDENCE_RAG_MODEL_CONTEXT_WINDOW','16000');monkeypatch.setenv('EVIDENCE_RAG_FRAMING_RESERVE','1024');monkeypatch.setenv('EVIDENCE_RAG_OUTPUT_RESERVE','2048')
    with TestClient(legal_app) as client:
        assert client.get('/health/ready').json()['conservative_bytes_available'] is False
