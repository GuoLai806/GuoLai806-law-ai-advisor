"""Fail-closed generation and resource checks, all providers are synthetic."""
import json
from types import SimpleNamespace
import httpx
import pytest
from evidence_rag.answer import OpenAIStream

EVIDENCE=[{'citation_id':'E1','title':'Synthetic','text':'Synthetic evidence.'}]

@pytest.mark.parametrize('kind',['http','transport','timeout'])
def test_failure_diagnostic_excludes_upstream_credentials_and_body(kind):
    from evidence_rag.api import safe_pipeline_error
    secret='synthetic-private-key'
    request=httpx.Request('POST','https://fixture/'+secret,headers={'Authorization':'Bearer '+secret})
    if kind=='http':
        exc=httpx.HTTPStatusError(secret,request=request,response=httpx.Response(503,text=secret,request=request))
    elif kind=='transport':exc=httpx.ConnectError(secret,request=request)
    else:exc=TimeoutError(secret)
    result=safe_pipeline_error(exc,'generation')
    assert secret not in json.dumps(result)
    assert result['error_type']==type(exc).__name__
    assert result['failure_phase']=='generation'
    assert result.get('http_status')==(503 if kind=='http' else None)

@pytest.mark.asyncio
@pytest.mark.parametrize('reason',['length','content_filter','tool_calls',None])
async def test_done_does_not_override_incomplete_finish(reason):
    body='data: '+json.dumps({'choices':[{'delta':{'content':'Partial [E1]'},'finish_reason':reason}]})+'\n\ndata: [DONE]\n\n'
    p=OpenAIStream('http://fake/v1','fake','synthetic',transport=httpx.MockTransport(lambda r:httpx.Response(200,content=body)))
    with pytest.raises(RuntimeError):
        _=[x async for x in p.stream('question',EVIDENCE)]

def test_measured_prompt_requires_provider_window_and_bounded_output(monkeypatch):
    from evidence_rag.api import generation_allowed
    bundle={'status':'ready','query_plan':{'mode':'source_lookup','applicability_review':False},'evidence':[{}],'degraded':False,'budget':{'state':'measured','used_tokens':5000,'omitted':[]}}
    app=SimpleNamespace(state=SimpleNamespace(generator=SimpleNamespace(max_output_tokens=None)))
    for prefix in ('EVIDENCE_RAG_','LEGAL_RAG_'):
        for name in ('MODEL_CONTEXT_WINDOW','FRAMING_RESERVE','OUTPUT_RESERVE'):
            monkeypatch.delenv(prefix+name,raising=False)
    assert not generation_allowed(bundle,None,app)
    monkeypatch.setenv('EVIDENCE_RAG_MODEL_CONTEXT_WINDOW','4096');monkeypatch.setenv('EVIDENCE_RAG_FRAMING_RESERVE','256');monkeypatch.setenv('EVIDENCE_RAG_OUTPUT_RESERVE','1024')
    app.state.generator.max_output_tokens=512
    assert not generation_allowed(bundle,None,app)
    bundle['budget']['used_tokens']=2000
    assert generation_allowed(bundle,None,app)
    app.state.generator.max_output_tokens=2048
    assert not generation_allowed(bundle,None,app)

def test_stream_rejects_invalid_citation_without_committing_turn(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    import evidence_rag.api as api
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('EVIDENCE_RAG_AUTH','{"fixture":"user"}');monkeypatch.setenv('EVIDENCE_RAG_STORE_CONVERSATIONS','1')
    async def bundle(*a,**kw):
        return {'trace_id':'fake','status':'ready','corpus_revision':'r1','query_plan':{'mode':'source_lookup','profile':'general','applicability_review':False},'evidence':EVIDENCE,'budget':{'state':'measured','omitted':[]},'degraded':False}
    class Generator:
        async def stream(self,*a,**kw):yield 'Unsupported [E999]'
    monkeypatch.setattr(api,'get_bundle',bundle);monkeypatch.setattr(api,'generation_allowed',lambda *a:True)
    with TestClient(api.app) as client:
        cat=api.app.state.catalog
        cat.stage_documents([{'_id':'d','title':'Synthetic','text':'Synthetic evidence.'}],{'corpus_id':'fixture','version_id':'v1','source_kind':'synthetic','verification_state':'provided'},'r1');cat.activate('r1',None)
        api.app.state.generator=Generator()
        body={'query':'question','session_id':'session'};headers={'Authorization':'Bearer fixture'}
        assert client.post('/api/v2/chat',json=body,headers=headers).json()['status']=='insufficient_evidence'
        response=client.post('/api/v2/chat/stream',json=body,headers=headers)
        assert 'event: answer_rejected' in response.text
        assert '"status": "insufficient_evidence"' in response.text
        assert '"status": "ready"' not in response.text.split('event: completed')[-1]
        assert cat.db.execute('SELECT count(*) FROM turns').fetchone()[0]==0

def test_oversized_admin_source_is_rejected_before_any_full_read(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from pathlib import Path
    from evidence_rag.api import app
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('EVIDENCE_RAG_AUTH','{"fixture":"admin"}');monkeypatch.setenv('EVIDENCE_RAG_ADMINS','admin');monkeypatch.setenv('EVIDENCE_RAG_INGEST_DIR',str(tmp_path))
    path=tmp_path/'oversize.txt'
    with path.open('wb') as f:f.truncate(60_000_000)
    with TestClient(app) as client:
        monkeypatch.setattr(Path,'read_bytes',lambda *a:(_ for _ in ()).throw(AssertionError('full read forbidden')))
        response=client.post('/api/v2/ingest-jobs',headers={'Authorization':'Bearer fixture'},json={'filename':'oversize.txt','manifest':{},'idempotency_key':'oversize-fixture'})
        assert response.status_code==413

def test_cli_oversized_source_rejected_before_read(tmp_path,monkeypatch):
    from pathlib import Path
    from evidence_rag.ingest import load_documents
    path=tmp_path/'oversize.txt'
    with path.open('wb') as f:f.truncate(110_000_000)
    monkeypatch.setattr(Path,'read_bytes',lambda *a:(_ for _ in ()).throw(AssertionError('full read forbidden')))
    with pytest.raises(ValueError,match='100 MB'):load_documents(path)

def generic_catalog(tmp_path):
    from evidence_rag.catalog import Catalog
    cat=Catalog(tmp_path/'catalog.sqlite3')
    cat.stage_documents([{'_id':'d1','title':'manual','text':'alpha first paragraph irrelevant.\n\nSecond paragraph contains the desired evidence.'}],{'corpus_id':'fixture','version_id':'v1','source_kind':'synthetic','verification_state':'provided'},'r1')
    cat.activate('r1',None)
    uid=cat.exact_documents('r1','',doc_id='d1')[0]
    cat.db.execute('UPDATE units SET uri=? WHERE revision=? AND unit_id=?',('viking://resources/rag/fixture/r1/'+uid+'/'+uid+'.md','r1',uid))
    cat.db.execute("INSERT INTO settings VALUES('dense_validated:r1','yes')")
    return cat,uid

@pytest.mark.parametrize('corruption',['unit','part','document','locator'])
def test_catalog_read_revalidates_hashes_and_generic_locators(tmp_path,corruption):
    cat,uid=generic_catalog(tmp_path)
    try:
        if corruption=='unit':cat.db.execute("UPDATE units SET text='corrupted' WHERE revision='r1'")
        elif corruption=='part':cat.db.execute("UPDATE parts SET text='corrupted' WHERE revision='r1' AND part_no=2")
        elif corruption=='document':cat.db.execute("UPDATE documents SET content_sha256='corrupted' WHERE revision='r1'")
        else:cat.db.execute("UPDATE parts SET locator=? WHERE revision='r1' AND part_no=2",(json.dumps({'char_start':0,'char_end':1}),))
        with pytest.raises(ValueError,match='mismatch'):cat.get('r1',uid)
        with pytest.raises(ValueError,match='mismatch'):cat.best_part('r1',uid,'alpha')
    finally:cat.db.close()

@pytest.mark.asyncio
@pytest.mark.parametrize('part_identity',[True,False])
async def test_evidence_uses_scored_part_or_removes_unattributed_score(tmp_path,part_identity):
    from evidence_rag.retrieval import CrossEncoderReranker,RetrievalService
    import hashlib
    cat,uid=generic_catalog(tmp_path)
    class Mask:
        def __getitem__(self,key):return self
        def sum(self):return 2
    ranker=object.__new__(CrossEncoderReranker)
    ranker.model=SimpleNamespace(max_length=4096,prompts={},default_prompt_name=None,preprocess=lambda *a,**kw:{'attention_mask':Mask()},predict=lambda *a,**kw:[-10.0,10.0])
    if not part_identity:
        ranker=SimpleNamespace(rank=lambda q,items:[(x,10.0) for x in items])
    class Dense:
        scope='viking://resources/rag/fixture/r1/'
        async def find(self,q):return []
    try:
        result=await RetrievalService(cat,Dense(),ranker,profile='general').retrieve('alpha')
        item=result['evidence'][0]
        if part_identity:
            assert item['part_no']==2 and item['reranker_score']==10.0
            assert item['text'].startswith('Second paragraph') and not result['degraded']
            assert hashlib.sha256(item['text'].encode()).hexdigest()==item['content_sha256']
        else:
            assert item['reranker_score'] is None and result['degraded']
            assert 'reranker_part_unavailable' in result['degraded_reasons']
    finally:cat.db.close()
