import json
import pytest
from legal_rag.catalog import Catalog, parse_pages
from legal_rag.retrieval import RetrievalService, rrf

MANIFEST={'doc_id':'fixture','version_id':'v1','title':'测试法','verification_state':'approved','source_kind':'synthetic','source_url':None}
PAGES=['第一条 本条跨页，除非','另有规定。\n第二条 合同成立。\n第三条 合同无效。']

def test_parser_cross_page_and_locator():
    out=parse_pages(PAGES)
    assert len(out)==3 and out[0]['locator']['pages']==[1,2]
    assert '另有规定' in out[0]['text']

def test_staging_cas_idempotency_and_isolation(tmp_path):
    c=Catalog(tmp_path/'db'); job=c.stage(PAGES,MANIFEST,'r1','same'); assert c.stage(PAGES,MANIFEST,'r2','same')['revision']=='r1'
    with pytest.raises(ValueError): c.activate('r1','different')
    c.activate('r1',None); assert c.active()=='r1'
    c.stage(['第一条 新版本。'],MANIFEST,'r2','new'); assert c.active()=='r1'
    assert c.get('r2',c.exact('r1','第一条')[0])['text']=='第一条 新版本。'
    c.activate('r2','r1'); assert c.active()=='r2'

def test_pending_never_activates(tmp_path):
    c=Catalog(tmp_path/'db'); c.stage(PAGES,{**MANIFEST,'verification_state':'pending'},'pending')
    with pytest.raises(ValueError): c.activate('pending',None)

def test_rrf_not_raw_scores():
    order,sources=rrf({'a':['x','y'],'b':['y','x']}); assert set(order)=={'x','y'} and len(sources['y'])==2

@pytest.mark.asyncio
async def test_retrieve_degrades_and_cites_complete_article(tmp_path):
    c=Catalog(tmp_path/'db'); c.stage(PAGES,MANIFEST,'r1'); c.activate('r1',None)
    b=await RetrievalService(c).retrieve('第一条',event_date='2024-01-01')
    assert b['degraded'] and b['evidence'][0]['article_label']=='第一条'
    assert '另有规定' in b['evidence'][0]['text'] and b['evidence'][0]['content_sha256']

def test_api_auth_and_sessions(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from legal_rag.api import app
    monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'api.db')); monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'secret-a':'a','secret-b':'b'}))
    with TestClient(app) as client:
        c=app.state.catalog; c.stage(PAGES,MANIFEST,'r1'); c.activate('r1',None)
        body={'query':'第一条','session_id':'session1'}
        assert client.post('/api/v2/retrieve',json=body).status_code==401
        assert client.post('/api/v2/retrieve',json=body,headers={'Authorization':'Bearer secret-a'}).status_code==200
        assert client.post('/api/v2/retrieve',json=body,headers={'Authorization':'Bearer secret-b'}).status_code==403
        assert client.get('/health/ready').status_code==503

@pytest.mark.asyncio
async def test_genuine_model_delta_and_citation_validation():
    import httpx
    from legal_rag.answer import OpenAIStream, validate_citations
    async def handler(request):
        assert json.loads(request.content)['stream'] is True
        return httpx.Response(200,content='data: {"choices":[{"delta":{"content":"依据"}}]}\n\ndata: {"choices":[{"delta":{"content":"[E1]"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    provider=OpenAIStream('http://test/v1','test','unused',transport=httpx.MockTransport(handler))
    e=[{'citation_id':'E1','title':'法','article_label':'第一条','text':'原文'}]
    deltas=[d async for d in provider.stream('问题',e)]
    assert deltas==['依据','[E1]'] and validate_citations(''.join(deltas),e)['claim_support']=='not_verified'
    assert validate_citations('[E2]',e)['status']=='invalid'

@pytest.mark.asyncio
async def test_viking_scope_and_contract():
    import httpx
    from legal_rag.retrieval import OpenViking
    async def handler(request):
        body=json.loads(request.content)
        assert request.url.path=='/api/v1/search/find'
        assert body['target_uri']=='viking://resources/law-cn/r1/' and body['level']==2 and body['read_content'] is False
        return httpx.Response(200,json={'status':'ok','result':{'resources':[{'uri':'viking://resources/law-cn/r1/a.md'},{'uri':'viking://resources/law-cn/r2/a.md'}]}})
    v=OpenViking('http://test','viking://resources/law-cn/r1',transport=httpx.MockTransport(handler))
    assert len(await v.find('合同'))==1

@pytest.mark.asyncio
async def test_viking_upload_then_ingest_not_retried():
    import httpx
    from legal_rag.retrieval import OpenViking
    calls=[]
    async def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith('temp_upload'): return httpx.Response(200,json={'status':'ok','result':{'temp_file_id':'tmp'}})
        body=json.loads(request.content); assert body['wait'] and body['strict'] and body['processing_mode']=='vectors_only' and body['to'].startswith('viking://resources/law-cn/r1/')
        return httpx.Response(200,json={'status':'ok','result':{'uri':body['to']}})
    v=OpenViking('http://test','viking://resources/law-cn/r1',transport=httpx.MockTransport(handler))
    result=await v.ingest('a'*32,'正文')
    assert result['state']=='needs_reconciliation' and calls==['/api/v1/resources/temp_upload','/api/v1/resources']

def test_paragraph_item_and_section_boundary():
    out=parse_pages(['第一条 第一款。\n\u3000\u3000第二款。\n\u3000\u3000（一）第一项。\n\u3000\u3000第二章 总则\n第二条 新条。'])
    assert len(out[0]['paragraphs'])==3 and out[0]['paragraphs'][2]['kind']=='item'
    assert '第二章' not in out[0]['text']

def test_token_budget_keeps_complete_units():
    from legal_rag.budget import assemble
    class T:
        def count_chat(self, messages): return sum(len(x['content'].encode('utf-8')) for x in messages)
    e=[{'unit_id':'a','citation_id':'E1','title':'法','article_label':'第一条','text':'除外情形完整'}, {'unit_id':'b','citation_id':'E2','title':'法','article_label':'第二条','text':'另一条'}]
    single=T().count_chat(__import__('legal_rag.answer',fromlist=['messages_for']).messages_for('问题',e[:1]))
    selected, info=assemble(e,T(),single,'问题')
    assert selected==e[:1] and info['omitted']==['b'] and info['used_tokens']<=single
    assert assemble(e,None,80)[1]['state']=='tokenizer_unavailable'

def test_idempotency_conflict(tmp_path):
    c=Catalog(tmp_path/'db'); c.stage(PAGES,MANIFEST,'r1','same-key')
    with pytest.raises(ValueError): c.stage(PAGES,{**MANIFEST,'title':'different'},'r2','same-key')

def test_ingest_admin_boundary(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from legal_rag.api import app
    monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'api.db')); monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'a':'admin','b':'other'})); monkeypatch.setenv('LEGAL_RAG_ADMINS','admin'); monkeypatch.setenv('LEGAL_RAG_INGEST_DIR',str(tmp_path))
    with TestClient(app) as client:
        body={'filename':'sample.pdf','manifest':MANIFEST,'idempotency_key':'valid-key'}
        assert client.post('/api/v2/ingest-jobs',json=body,headers={'Authorization':'Bearer b'}).status_code==403
        assert client.post('/api/v2/ingest-jobs',json=body,headers={'Authorization':'Bearer a'}).status_code==404

@pytest.mark.asyncio
async def test_viking_journal_prevents_ambiguous_replay(tmp_path):
    import httpx
    from legal_rag.retrieval import OpenViking
    from legal_rag.viking_index import index_one,reconcile_one
    c=Catalog(tmp_path/'db'); c.stage(PAGES,MANIFEST,'r1'); uid=c.exact('r1','第一条')[0]; unit=c.get('r1',uid); calls=[]
    async def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith('temp_upload'): return httpx.Response(200,json={'status':'ok','result':{'temp_file_id':'tmp'}})
        if request.url.path.endswith('resources'): raise httpx.ReadTimeout('unknown')
        return httpx.Response(200,json={'status':'ok','result':unit['text']})
    adapter=OpenViking('http://test','viking://resources/law-cn/r1',transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ReadTimeout): await index_one(c,adapter,'r1',uid)
    assert (await index_one(c,adapter,'r1',uid))['state']=='delivery_unknown' and len(calls)==2
    assert (await reconcile_one(c,adapter,'r1',uid))['state']=='content_verified'
    assert c.get('r1',uid)['uri'].endswith(uid+'.md')

def test_stream_deltas_are_real_upstream_chunks(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from legal_rag.api import app
    monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'api.db')); monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'x':'user'}))
    class Dense:
        scope='viking://resources/law-cn/r1/'
        async def find(self,q): return []
    class Rank:
        def rank(self,q,items): return [(item,1.0) for item in items]
    class Tokenizer:
        def count_chat(self,messages): return sum(len(m['content']) for m in messages)
    monkeypatch.setenv("LEGAL_RAG_MODEL_CONTEXT_WINDOW","16000");monkeypatch.setenv("LEGAL_RAG_FRAMING_RESERVE","1024");monkeypatch.setenv("LEGAL_RAG_OUTPUT_RESERVE","2048")
    class Generator:
        max_output_tokens=1024
        async def stream(self,q,evidence):
            yield '依据'; yield '[E1]'
    with TestClient(app) as client:
        c=app.state.catalog;c.stage(PAGES,MANIFEST,'r1');c.activate('r1',None);uid=c.exact('r1','第一条')[0];c.db.execute('UPDATE units SET uri=? WHERE revision=? AND unit_id=?',('viking://resources/law-cn/r1/'+uid+'.md','r1',uid))
        c.db.execute("INSERT INTO settings VALUES('dense_validated:r1','yes')");app.state.service.dense=Dense();app.state.service.reranker=Rank();app.state.service.tokenizer=Tokenizer();app.state.generator=Generator()
        r=client.post('/api/v2/chat/stream',json={'query':'查询第一条原文','session_id':'s'},headers={'Authorization':'Bearer x'})
        assert r.status_code==200 and r.text.count('event: answer_delta')==2 and '"claim_support": "not_verified"' in r.text

@pytest.mark.asyncio
async def test_viking_index_validation_requires_all_mappings(tmp_path):
    from legal_rag.viking_index import validate_index
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1')
    class Dense:
        scope='viking://resources/law-cn/r1/'
        async def find(self,q): return []
    result=await validate_index(c,Dense(),'r1')
    assert result['index_visibility_validated'] is False and len(result['units'])==3
    assert c.db.execute("SELECT value FROM settings WHERE key='dense_validated:r1'").fetchone() is None

def test_legacy_adapter_is_explicit_and_keeps_user_isolation(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from legal_rag.api import app
    monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'api.db')); monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'a':'alice','b':'bob'}))
    with TestClient(app) as client:
        app.state.catalog.stage(PAGES,MANIFEST,'r1');app.state.catalog.activate('r1',None)
        headers={'Authorization':'Bearer a'}; body={'message':'第一条'}
        assert client.post('/api/chat',json=body,headers=headers).status_code==404
        monkeypatch.setenv('LEGAL_RAG_LEGACY_ADAPTER','1')
        r=client.post('/api/chat',json=body,headers=headers); assert r.status_code==200 and r.json()['backend']=='v2' and r.json()['intent_type']=='consultation'
        assert client.post('/chat',json={**body,'session_id':'shared'},headers=headers).status_code==200
        assert client.post('/chat',json={**body,'session_id':'shared'},headers={'Authorization':'Bearer b'}).status_code==403
        s=client.post('/api/chat/stream',json=body,headers=headers); assert 'data: [DONE]' in s.text

def test_durable_session_history_is_opt_in_and_isolated(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from legal_rag.api import app
    monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'api.db')); monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'a':'alice','b':'bob'})); monkeypatch.setenv('LEGAL_RAG_STORE_CONVERSATIONS','1')
    with TestClient(app) as client:
        app.state.catalog.stage(PAGES,MANIFEST,'r1');app.state.catalog.activate('r1',None)
        body={'query':'第一条','session_id':'persist'}; headers={'Authorization':'Bearer a'}
        assert client.post('/api/v2/chat',json=body,headers=headers).status_code==200
    with TestClient(app) as client:
        r=client.get('/api/v2/sessions/persist/turns',headers=headers)
        assert r.status_code==200 and r.json()['turns'][0]['query']=='第一条'
        assert client.get('/api/v2/sessions/persist/turns',headers={'Authorization':'Bearer b'}).status_code==403

def test_activation_rejects_corrupt_index(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1');c.db.execute("DELETE FROM unit_fts WHERE revision='r1' AND unit_id=(SELECT unit_id FROM units WHERE revision='r1' LIMIT 1)")
    with pytest.raises(ValueError,match='index count'):c.activate('r1',None)
    assert c.active() is None

def test_real_source_activation_requires_dense_index(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,{**MANIFEST,'verification_state':'pending','source_kind':'law','source_url':'https://example.org/source','source_sha256':'f'*64},'r1')
    c.approve_source('r1','reviewer','Reviewed official source and compared pages.')
    with pytest.raises(ValueError,match='dense index not validated'): c.activate('r1',None)
    assert c.active() is None

def test_repeated_ingest_after_source_review_is_idempotent(tmp_path):
    c=Catalog(tmp_path/'db');m={**MANIFEST,'source_kind':'law','verification_state':'pending','source_url':'https://example.org/doc','source_sha256':'f'*64}
    c.stage(PAGES,m,'r1','same-key');c.approve_source('r1','reviewer','Compared the source against official publication.')
    assert c.stage(PAGES,m,'r2','same-key')['revision']=='r1'

def test_real_file_cannot_use_synthetic_activation_bypass(tmp_path):
    c=Catalog(tmp_path/'db')
    with pytest.raises(ValueError,match='synthetic fixture'): c.stage(PAGES,{**MANIFEST,'source_sha256':'f'*64},'r1')

@pytest.mark.asyncio
async def test_reranker_cannot_invent_units(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1');c.activate('r1',None)
    class BadRanker:
        def rank(self,q,items): return [({'unit_id':'invented'},1.0)]
    b=await RetrievalService(c,reranker=BadRanker()).retrieve('第一条')
    assert 'reranker_failed' in b['degraded_reasons'] and b['evidence'][0]['article_label']=='第一条'

def test_admin_actual_pdf_ingest_is_pending_and_queryable(tmp_path,monkeypatch):
    import shutil
    from pathlib import Path
    from fastapi.testclient import TestClient
    from legal_rag.api import app
    pdf=next(Path('main-system').glob('*.pdf')); shutil.copyfile(pdf,tmp_path/'source.pdf')
    monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'api.db'));monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'a':'admin'}));monkeypatch.setenv('LEGAL_RAG_ADMINS','admin');monkeypatch.setenv('LEGAL_RAG_INGEST_DIR',str(tmp_path))
    with TestClient(app) as client:
        body={'filename':'source.pdf','manifest':{'doc_id':'fixture-pdf','version_id':'v1','title':'PDF','source_kind':'repository-source-unverified','verification_state':'approved'},'idempotency_key':'pdf-job-1'};headers={'Authorization':'Bearer a'}
        r=client.post('/api/v2/ingest-jobs',json=body,headers=headers);assert r.status_code==202
        job=r.json(); assert client.get('/api/v2/ingest-jobs/'+job['job_id'],headers=headers).json()['state']=='indexed'
        assert json.loads(app.state.catalog.revision(job['revision'])['manifest'])['verification_state']=='pending'
        assert client.post('/api/v2/ingest-jobs',json=body,headers=headers).json()['job_id']==job['job_id']

def test_reranker_checks_complete_paragraph_windows():
    from legal_rag.retrieval import CrossEncoderReranker
    class Tokenizer:
        def __call__(self,q,d,**kwargs): return {'input_ids':list(range(len(d)))}
    class Model:
        max_length=15; tokenizer=Tokenizer();default_prompt_name=None;prompts={}
        def preprocess(self,pairs,**kwargs):
            import numpy as np
            return {'attention_mask':np.ones((1,len(pairs[0][1])),dtype=int)}
        def predict(self,pairs,**kwargs): return [float(len(d)) for q,d in pairs]
    reranker=CrossEncoderReranker.__new__(CrossEncoderReranker);reranker.model=Model()
    c={'unit_id':'1','title':'法','label':'第一条','text':'短。另一款。','parts':[{'text':'短。'},{'text':'另一款。'}]}
    assert reranker.rank('问',[c])[0][0]['unit_id']=='1'
    c['parts']=[{'text':'超过长度'*20}]
    with pytest.raises(ValueError,match='exceeds'): reranker.rank('问',[c])

def test_article_number_normalization(tmp_path):
    from legal_rag.catalog import canonical_label
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1')
    assert c.exact('r1','第１条')==c.exact('r1','第一条')
    assert canonical_label('第一千二百六十条')==(1260,0)
    assert canonical_label('第１２条之一')==(12,1)

def test_activation_rejects_same_count_wrong_fts_mapping(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1')
    uid=c.exact('r1','第一条')[0]
    c.db.execute('UPDATE unit_fts SET unit_id=? WHERE revision=? AND unit_id=?',('not-a-unit','r1',uid))
    with pytest.raises(ValueError,match='mapping mismatch'):c.activate('r1',None)

@pytest.mark.asyncio
async def test_query_modes_and_temporal_uncertainty(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1');c.activate('r1',None);service=RetrievalService(c)
    source=await service.retrieve('请查询第一条原文')
    assert source['status']=='ready' and source['query_plan']['mode']=='source_lookup' and not source['missing_facts']
    general=await service.retrieve('合同是什么')
    assert general['status']=='ready' and general['query_plan']['mode']=='general_explanation'
    case=await service.retrieve('我的合同纠纷怎么办')
    assert case['status']=='needs_clarification' and case['missing_facts']
    dated=await service.retrieve('我的合同纠纷怎么办',event_date='2019-01-01')
    assert dated['query_plan']['applicability_review'] and dated['missing_facts']

def test_general_generation_needs_a_threshold(monkeypatch):
    from types import SimpleNamespace
    from legal_rag.api import generation_allowed
    app=SimpleNamespace(state=SimpleNamespace(generator=SimpleNamespace(max_output_tokens=1024)))
    monkeypatch.setenv('LEGAL_RAG_MODEL_CONTEXT_WINDOW','16000');monkeypatch.setenv('LEGAL_RAG_FRAMING_RESERVE','1024');monkeypatch.setenv('LEGAL_RAG_OUTPUT_RESERVE','2048')
    bundle={'status':'ready','query_plan':{'mode':'general_explanation','applicability_review':False},'evidence':[{'reranker_score':0.8}],'degraded':False,'budget':{'state':'measured','used_tokens':2000,'omitted':[]}}
    monkeypatch.delenv('LEGAL_RAG_RERANK_THRESHOLD',raising=False)
    assert not generation_allowed(bundle,None,app)
    monkeypatch.setenv('LEGAL_RAG_RERANK_THRESHOLD','0.9');assert not generation_allowed(bundle,None,app)
    monkeypatch.setenv('LEGAL_RAG_RERANK_THRESHOLD','0.7');assert generation_allowed(bundle,None,app)

@pytest.mark.asyncio
async def test_exact_lookup_only_returns_requested_article(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1');c.activate('r1',None)
    b=await RetrievalService(c).retrieve('查询第一条原文')
    assert [e['article_label'] for e in b['evidence']]==['第一条']

def test_evaluation_empty_proxy_reports_false_positive(tmp_path):
    import subprocess,sys
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1');c.db.close()
    data={'kind':'synthetic','queries':[{'query':'第一条','expected_labels':['第一条']},{'query':'合同','expected_labels':[]}]}
    dataset=tmp_path/'data.json';out=tmp_path/'out.json';dataset.write_text(json.dumps(data,ensure_ascii=False))
    subprocess.run([sys.executable,'-m','legal_rag.cli','evaluate','--db',str(tmp_path/'db'),'--revision','r1','--dataset',str(dataset),'--output',str(out),'--variant','bm25'],check=True,capture_output=True)
    result=json.loads(out.read_text());assert result['empty_retrieval_proxy']['fn']==1 and result['results'][1]['failure']=='unexpected_results'

def test_article_relations_are_candidates_with_local_offsets(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(['第一条 根据第二条办理。\n第二条 其他规定。'],MANIFEST,'r1')
    first=c.get('r1',c.exact('r1','第一条')[0]); relation=first['relations'][0]
    assert relation['verified']==0 and first['text'][relation['char_start']:relation['char_end']]=='第二条'
    assert relation['to_id']==c.exact('r1','第二条')[0]

@pytest.mark.asyncio
async def test_truncated_model_stream_is_not_complete():
    import httpx
    from legal_rag.answer import OpenAIStream
    async def handler(request):return httpx.Response(200,content='data: {"choices":[{"delta":{"content":"[E1]"}}]}\n\n')
    p=OpenAIStream('http://test/v1','test','unused',transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError,match='completion marker'):
        _=[x async for x in p.stream('问',[{'citation_id':'E1','title':'法','article_label':'第一条','text':'原文'}])]

def test_stream_session_forbidden_before_headers(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from legal_rag.api import app
    monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'a':'a','b':'b'}))
    with TestClient(app) as client:
        app.state.catalog.stage(PAGES,MANIFEST,'r1');app.state.catalog.activate('r1',None)
        body={'query':'第一条','session_id':'same'}
        assert client.post('/api/v2/retrieve',json=body,headers={'Authorization':'Bearer a'}).status_code==200
        assert client.post('/api/v2/chat/stream',json=body,headers={'Authorization':'Bearer b'}).status_code==403

@pytest.mark.asyncio
async def test_request_revision_pin_survives_active_swap(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from legal_rag.api import Query,get_bundle,check_access
    c=Catalog(tmp_path/'db');c.db.execute('CREATE TABLE sessions(session_id TEXT PRIMARY KEY,owner TEXT NOT NULL)')
    c.stage(PAGES,MANIFEST,'r1');c.activate('r1',None)
    request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(catalog=c,service=RetrievalService(c))))
    body=Query(query='查询第一条原文',session_id='pinned')
    pinned=check_access(body,request,'user');c.stage(['第一条 新版本。'],MANIFEST,'r2');c.activate('r2','r1')
    result=await get_bundle(body,request,'user',pinned_revision=pinned)
    assert result['corpus_revision']=='r1' and '跨页' in result['evidence'][0]['text']

@pytest.mark.asyncio
async def test_explicit_other_law_never_returns_same_number(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,{**MANIFEST,'title':'中华人民共和国民法典'},'r1');c.activate('r1',None)
    for q in ['刑法第一条原文','《中华人民共和国刑法》第一条原文','劳动合同法第一条原文','知识产权法第一条原文','个人信息保护法第一条原文']:
        b=await RetrievalService(c).retrieve(q)
        assert b['status']=='insufficient_evidence' and not b['evidence'] and b['query_plan']['mode']=='out_of_scope_law'
    assert (await RetrievalService(c).retrieve('民法典第一条原文'))['status']=='ready'

def test_retained_revision_citation_can_be_read(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from legal_rag.api import app
    monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'a':'a'}))
    with TestClient(app) as client:
        c=app.state.catalog;c.stage(PAGES,MANIFEST,'r1');c.activate('r1',None);uid=c.exact('r1','第一条')[0]
        c.stage(['第一条 新版。'],MANIFEST,'r2');c.activate('r2','r1');headers={'Authorization':'Bearer a'}
        old=client.get('/api/v2/units/'+uid,params={'revision':'r1'},headers=headers)
        assert old.status_code==200 and '跨页' in old.json()['text']
        assert client.get('/api/v2/units/'+uid,params={'revision':'unknown'},headers=headers).status_code==404

def test_model_experiment_fuses_candidates_without_downloading(tmp_path,monkeypatch):
    import sys,types,numpy as np
    from legal_rag import model_experiment
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1');c.db.close()
    dataset=tmp_path/'data.json';dataset.write_text(json.dumps({'kind':'synthetic','queries':[{'query':'第一条','expected_labels':['第一条']}]},ensure_ascii=False));out=tmp_path/'out.json'
    class FakeEmbedding:
        max_seq_length=512;device='cpu'
        tokenizer=staticmethod(lambda text,**kwargs:{'input_ids':[1,2]})
        def __init__(self,*args,**kwargs): assert kwargs['local_files_only']
        def encode(self,items,**kwargs):return np.ones((len(items),2))
        def parameters(self):return iter([types.SimpleNamespace(dtype='torch.float32')])
    class FakeReranker:
        revision='fake';model=types.SimpleNamespace(max_length=512,device='cpu',parameters=lambda:iter([types.SimpleNamespace(dtype='torch.float32')]))
        def __init__(self,*args,**kwargs): pass
        def rank(self,q,items):return [(x,0.5) for x in items]
    monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(SentenceTransformer=FakeEmbedding))
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(float32='torch.float32'))
    monkeypatch.setattr(model_experiment,'CrossEncoderReranker',FakeReranker)
    monkeypatch.setattr(model_experiment.importlib.metadata,'version',lambda _: 'fake')
    monkeypatch.setattr(sys,'argv',['model_experiment','--db',str(tmp_path/'db'),'--revision','r1','--dataset',str(dataset),'--output',str(out),'--embedding','bge','--reranker','bge'])
    model_experiment.main();result=json.loads(out.read_text())
    assert result['results'][0]['ranking'][0]['label']=='第一条' and result['results'][0]['ranking'][0]['retrieval_sources']

def test_activation_rejects_wrong_fts_content(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1')
    uid=c.exact('r1','第一条')[0];c.db.execute('UPDATE unit_fts SET text=? WHERE revision=? AND unit_id=?',('错误 内容','r1',uid))
    with pytest.raises(ValueError,match='content mismatch'):c.activate('r1',None)

@pytest.mark.asyncio
async def test_openviking_protocol_over_real_loopback_http(tmp_path):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from threading import Thread
    from urllib.parse import parse_qs,urlparse
    from legal_rag.retrieval import OpenViking
    from legal_rag.viking_index import index_one,reconcile_one
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1');uid=c.exact('r1','第一条')[0];unit=c.get('r1',uid);calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,data):
            body=json.dumps(data,ensure_ascii=False).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            raw=self.rfile.read(int(self.headers['Content-Length']));calls.append(self.path)
            if self.path.endswith('temp_upload'):
                assert b'multipart/form-data' in self.headers['Content-Type'].encode() and unit['text'].encode() in raw
                self.respond({'status':'ok','result':{'temp_file_id':'fake-upload'}})
            elif self.path.endswith('/resources'):
                body=json.loads(raw);assert body['wait'] and body['to'].endswith('/'+uid)
                self.respond({'status':'ok','result':{'uri':body['to']}})
            elif self.path.endswith('/find'):
                body=json.loads(raw);assert body['level']==2 and body['read_content'] is False
                self.respond({'status':'ok','result':{'resources':[{'uri':'viking://resources/law-cn/r1/'+uid+'/'+uid+'.md'}]}})
        def do_GET(self):
            calls.append(urlparse(self.path).path);assert parse_qs(urlparse(self.path).query)['uri'][0].endswith('/'+uid+'/'+uid+'.md')
            self.respond({'status':'ok','result':unit['text']})
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        v=OpenViking('http://127.0.0.1:'+str(server.server_port),'viking://resources/law-cn/r1')
        assert (await index_one(c,v,'r1',uid))['state']=='delivery_unknown'
        assert (await reconcile_one(c,v,'r1',uid))['state']=='content_verified'
        assert (await v.find('第一条'))[0]['uri'].endswith(uid+'.md')
        assert calls==['/api/v1/resources/temp_upload','/api/v1/resources','/api/v1/content/read','/api/v1/search/find']
    finally:
        server.shutdown();server.server_close();thread.join()

@pytest.mark.asyncio
async def test_lexical_raw_score_keeps_source_specific_meaning(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1');c.activate('r1',None)
    b=await RetrievalService(c).retrieve('合同')
    bm25=[x for e in b['evidence'] for x in e['retrieval_sources'] if x['source']=='bm25']
    assert bm25 and bm25[0]['score_kind']=='sqlite_bm25_lower_is_better' and isinstance(bm25[0]['raw_score'],float)

@pytest.mark.asyncio
async def test_missing_exact_article_never_substitutes_similar_text(tmp_path):
    c=Catalog(tmp_path/'db');c.stage(PAGES,MANIFEST,'r1');c.activate('r1',None)
    b=await RetrievalService(c).retrieve('查询第九十九条原文')
    assert b['status']=='insufficient_evidence' and not b['evidence'] and 'article_not_found' in b['degraded_reasons']
    mixed=await RetrievalService(c).retrieve('查询第一条和第九十九条原文')
    assert mixed['status']=='insufficient_evidence' and not mixed['evidence']

def test_supplementary_heading_not_in_prior_article():
    rows=parse_pages(['第一条 原文。\n\u3000\u3000附则\n第二条 新文。'])
    assert rows[0]['text']=='第一条 原文。' and '附则' not in rows[0]['text']

def test_reranker_budget_counts_model_prompt_and_chat_template():
    import numpy as np
    from legal_rag.retrieval import CrossEncoderReranker
    class Model:
        max_length=10;default_prompt_name='query';prompts={'query':'real instruction'}
        def preprocess(self,pairs,prompt,processing_kwargs):
            assert prompt=='real instruction' and processing_kwargs['chat_template']['truncation'] is False and processing_kwargs['text']['truncation'] is False
            return {'attention_mask':np.ones((1,11),dtype=int)}
        def predict(self,pairs,**kwargs):raise AssertionError('over-budget input must not run')
    reranker=CrossEncoderReranker.__new__(CrossEncoderReranker);reranker.model=Model()
    with pytest.raises(ValueError,match='exceeds'):reranker.rank('短问',[{'unit_id':'1','title':'法','label':'第一条','text':'短文'}])

def test_reranker_propagates_device_and_dtype(monkeypatch):
    import sys,types
    from legal_rag.retrieval import CrossEncoderReranker
    seen={}
    def create(model,**kwargs):seen.update(kwargs);return object()
    monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(CrossEncoder=create))
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(float16='half'))
    CrossEncoderReranker('bge',device='mps',dtype='float16')
    assert seen['device']=='mps' and seen['model_kwargs']['dtype']=='half' and seen['local_files_only']
    with pytest.raises(ValueError):CrossEncoderReranker('bge',dtype='int8')
