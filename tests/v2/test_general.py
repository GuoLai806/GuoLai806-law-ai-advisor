"""Public behavior for the general evidence profile."""
import json,subprocess,sys
import pytest
from evidence_rag.catalog import Catalog
from evidence_rag.ingest import load_documents
from evidence_rag.retrieval import RetrievalService,OpenViking

DOCS=[{'_id':'doc-a','title':'Ocean currents','text':'Ocean currents move heat.\n\nWind drives surface currents.','source_url':'https://example.org/ocean'}, {'_id':'doc-b','title':'Mountain weather','text':'High mountains can be cold. Snow reflects sunlight.'}]
META={'corpus_id':'demo','version_id':'v1','source_kind':'user-provided','verification_state':'provided'}

def test_generic_jsonl_stage_version_and_source(tmp_path):
    path=tmp_path/'docs.jsonl';path.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in DOCS),encoding='utf8')
    docs,provenance=load_documents(path);c=Catalog(tmp_path/'db');job=c.stage_documents(docs,{**META,**provenance},'r1');c.activate('r1',None)
    uid=c.exact_documents('r1','not the title','doc-a')[0];unit=c.get('r1',uid)
    assert unit['document']['doc_id']=='doc-a' and unit['document']['source_url']=='https://example.org/ocean'
    assert len(unit['parts'])==2 and ''.join(x['text'] for x in unit['parts'])==DOCS[0]['text']
    assert c.stage_documents(docs,{**META,**provenance},'r2')['revision']==job['revision']
    with pytest.raises(ValueError):c.stage_documents([DOCS[0],DOCS[0]],META,'r3')

@pytest.mark.asyncio
async def test_general_retrieval_has_no_legal_date_or_law_filter(tmp_path):
    c=Catalog(tmp_path/'db');c.stage_documents(DOCS,META,'r1');c.activate('r1',None);service=RetrievalService(c)
    b=await service.retrieve('What drives ocean currents?')
    assert b['status']=='ready' and b['query_plan']['profile']=='general' and b['missing_facts']==[]
    assert b['evidence'][0]['document_id']=='doc-a' and b['evidence'][0]['text'] in DOCS[0]['text'] and b['evidence'][0]['has_more']
    assert 'article_label' not in b['evidence'][0] and b['evidence'][0]['source_locator']['kind']=='text'
    exact=await service.retrieve('any text',document_id='doc-b')
    assert [x['document_id'] for x in exact['evidence']]==['doc-b']
    missing=await service.retrieve('ocean',document_id='missing');assert missing['status']=='insufficient_evidence' and not missing['evidence']
    with pytest.raises(LookupError):await RetrievalService(c,profile='legal').retrieve('第一条')

def test_general_scope_and_legacy_scope_are_distinct():
    OpenViking('http://localhost','viking://resources/rag/demo/r1')
    OpenViking('http://localhost','viking://resources/law-cn/r1')
    for bad in ('viking://resources/rag/demo/../r1','viking://user/private','viking://resources/rag/demo/r1evil/path'):
        with pytest.raises(ValueError):OpenViking('http://localhost',bad)

def test_general_api_and_legacy_profile_mismatch(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from evidence_rag.api import app,legal_app
    monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'test-token':'u'}))
    with TestClient(app) as client:
        app.state.catalog.stage_documents(DOCS,META,'r1');app.state.catalog.activate('r1',None)
        body={'query':'What drives ocean currents?','session_id':'s'};headers={'Authorization':'Bearer test-token'}
        r=client.post('/api/v2/retrieve',json=body,headers=headers)
        assert r.status_code==200 and r.json()['query_plan']['profile']=='general' and not r.json()['missing_facts']
        assert client.post('/api/v2/retrieve',json={**body,'profile':'legal'},headers=headers).status_code==404
        assert client.post('/api/v2/retrieve',json=body).status_code==401
    with TestClient(legal_app) as client:
        assert client.post('/api/v2/retrieve',json=body,headers=headers).status_code==404

def test_generic_cli_jsonl_roundtrip(tmp_path):
    path=tmp_path/'docs.jsonl';path.write_text('\n'.join(json.dumps(x) for x in DOCS));db=tmp_path/'db'
    run=subprocess.run([sys.executable,'-m','evidence_rag.cli','ingest','--db',str(db),'--input',str(path),'--corpus-id','demo','--revision','r1'],capture_output=True,text=True,check=True)
    assert json.loads(run.stdout)['state']=='indexed'
    subprocess.run([sys.executable,'-m','evidence_rag.cli','activate','--db',str(db),'--revision','r1'],capture_output=True,check=True)
    assert Catalog(db).active()=='r1'

def test_general_text_and_pdf_page_locators(tmp_path):
    import pymupdf
    text=tmp_path/'note.txt';text.write_text('First paragraph.\n\nSecond paragraph.')
    docs,meta=load_documents(text);assert docs[0]['text']=='First paragraph.\n\nSecond paragraph.' and meta['input_format']=='text'
    pdf=pymupdf.open()
    pdf.new_page().insert_text((72,72),'Ocean currents move heat.')
    pdf.new_page().insert_text((72,72),'Mountain snow reflects sunlight.')
    path=tmp_path/'example.pdf';pdf.save(path);pdf.close()
    docs,meta=load_documents(path,doc_id='pdf-1',source_url='https://example.org/file.pdf')
    c=Catalog(tmp_path/'db');c.stage_documents(docs,{**META,**meta},'r1');unit=c.get('r1',c.exact_documents('r1','',doc_id='pdf-1')[0])
    assert unit['locator']['pages']==[1,2] and unit['document']['source_url']=='https://example.org/file.pdf'
    assert unit['parts'][0]['locator']['pages']==[1] and unit['parts'][-1]['locator']['pages']==[2]

def test_general_admin_ingest_jsonl_is_staged(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from evidence_rag.api import app
    (tmp_path/'docs.jsonl').write_text('\n'.join(json.dumps(x) for x in DOCS))
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('EVIDENCE_RAG_AUTH',json.dumps({'a':'admin'}));monkeypatch.setenv('EVIDENCE_RAG_ADMINS','admin');monkeypatch.setenv('EVIDENCE_RAG_INGEST_DIR',str(tmp_path))
    with TestClient(app) as client:
        headers={'Authorization':'Bearer a'};body={'filename':'docs.jsonl','manifest':META,'idempotency_key':'generic-job'}
        r=client.post('/api/v2/ingest-jobs',json=body,headers=headers)
        assert r.status_code==202 and r.json()['state']=='indexed' and app.state.catalog.active() is None
        assert client.get('/api/v2/ingest-jobs/'+r.json()['job_id'],headers=headers).status_code==200

@pytest.mark.asyncio
async def test_index_write_rejects_wrong_corpus_scope_before_dispatch(tmp_path):
    from evidence_rag.viking_index import index_one
    c=Catalog(tmp_path/'db');c.stage_documents(DOCS,META,'r1');uid=c.exact_documents('r1','',doc_id='doc-a')[0]
    class Unsafe:
        scope='viking://resources/rag/other/r1/'
        async def ingest(self,*args):raise AssertionError('must not dispatch')
    with pytest.raises(ValueError,match='scope does not match'):await index_one(c,Unsafe(),'r1',uid)
    assert not c.db.execute("SELECT name FROM sqlite_master WHERE name='viking_journal'").fetchone()

def test_generic_ingest_unicode_filename_and_traversal(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from evidence_rag.api import app
    (tmp_path/'资料.txt').write_text('Ocean currents move heat.')
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('EVIDENCE_RAG_AUTH',json.dumps({'a':'admin'}));monkeypatch.setenv('EVIDENCE_RAG_ADMINS','admin');monkeypatch.setenv('EVIDENCE_RAG_INGEST_DIR',str(tmp_path))
    with TestClient(app) as client:
        h={'Authorization':'Bearer a'};b={'filename':'资料.txt','manifest':META,'idempotency_key':'unicode-job'}
        assert client.post('/api/v2/ingest-jobs',json=b,headers=h).status_code==202
        assert client.post('/api/v2/ingest-jobs',json={**b,'filename':'../资料.txt','idempotency_key':'path-escape'},headers=h).status_code==422

def test_general_evaluation_reports_original_document_ids(tmp_path):
    c=Catalog(tmp_path/'db');c.stage_documents(DOCS,META,'r1');c.db.close()
    data={'kind':'synthetic','queries':[{'query':'ocean currents','expected_ids':['doc-a']}]};dataset=tmp_path/'queries.json';dataset.write_text(json.dumps(data));output=tmp_path/'result.json'
    subprocess.run([sys.executable,'-m','evidence_rag.cli','evaluate','--db',str(tmp_path/'db'),'--revision','r1','--dataset',str(dataset),'--output',str(output),'--variant','bm25'],check=True,capture_output=True)
    report=json.loads(output.read_text());assert report['results'][0]['ranked_document_ids'][0]=='doc-a' and report['mean_recall_at_20']==1.0

def test_ready_rejects_same_revision_other_corpus_scope(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from evidence_rag.api import app
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'));monkeypatch.setenv('EVIDENCE_RAG_AUTH',json.dumps({'a':'u'}))
    with TestClient(app) as client:
        c=app.state.catalog;c.stage_documents(DOCS,META,'r1');c.activate('r1',None)
        app.state.service.dense=OpenViking('http://127.0.0.1:1','viking://resources/rag/other/r1')
        r=client.get('/health/ready');assert r.status_code==503 and 'openviking' in r.json()['missing']

@pytest.mark.asyncio
async def test_general_evidence_returns_complete_relevant_paragraph_and_hash(tmp_path):
    import hashlib
    c=Catalog(tmp_path/'db');c.stage_documents(DOCS,META,'r1');c.activate('r1',None)
    result=await RetrievalService(c).retrieve('Wind drives surface currents')
    item=result['evidence'][0]
    assert item['document_id']=='doc-a' and item['part_no']==2 and item['text']=='Wind drives surface currents.'
    assert item['source_locator']['char_start']==DOCS[0]['text'].index('Wind') and item['content_sha256']==hashlib.sha256(item['text'].encode()).hexdigest()
    assert item['document_sha256']==hashlib.sha256(DOCS[0]['text'].encode()).hexdigest()

def test_generic_activation_rejects_corrupt_paragraph_index(tmp_path):
    c=Catalog(tmp_path/'db');c.stage_documents(DOCS,META,'r1');c.db.execute("UPDATE part_fts SET text='corrupt' WHERE revision='r1'")
    with pytest.raises(ValueError,match='paragraph index mismatch'):c.activate('r1',None)

def test_shared_ranker_limits_sources_and_preserves_ids(tmp_path):
    from evidence_rag.retrieval import rank_candidates,document_embedding_text,reranker_document_text
    c=Catalog(tmp_path/'db');c.stage_documents(DOCS,META,'r1');a=c.exact_documents('r1','',doc_id='doc-a')[0];b=c.exact_documents('r1','',doc_id='doc-b')[0]
    result=rank_candidates(c,'r1','ocean',iter([(a,-1.2)]),iter([(b,.8)]))
    assert set(x['document']['doc_id'] for x in result['candidates'])=={'doc-a','doc-b'} and result['sources'][a][0]['raw_score']==-1.2
    unit=c.get('r1',a);assert document_embedding_text(unit)=='Ocean currents\n'+DOCS[0]['text'] and reranker_document_text(unit,unit['parts'][0]).startswith('Ocean currents\n')
    assert 'doc-a' not in document_embedding_text(unit) and 'doc-a' not in reranker_document_text(unit,unit['parts'][0])

def test_empty_title_never_inserts_document_id_into_model_input(tmp_path):
    from evidence_rag.retrieval import document_embedding_text,reranker_document_text
    c=Catalog(tmp_path/'db');c.stage_documents([{'_id':'secret-id','title':'','text':'neutral content'}],META,'r1');uid=c.exact_documents('r1','',doc_id='secret-id')[0];unit=c.get('r1',uid)
    assert document_embedding_text(unit)=='\nneutral content' and 'secret-id' not in reranker_document_text(unit,unit['parts'][0])

def test_shared_rank_limits_top40_each_and_at_most_60(tmp_path):
    from evidence_rag.retrieval import rank_candidates
    docs=[{'_id':f'id-{i:03d}','title':f'Document {i}','text':f'content {i}'} for i in range(85)]
    c=Catalog(tmp_path/'db');c.stage_documents(docs,META,'r1')
    ids=[c.exact_documents('r1','',doc_id=d['_id'])[0] for d in docs]
    lexical=[(uid,-float(i)) for i,uid in enumerate(ids[:45])];dense=[(uid,1.0-i/100) for i,uid in enumerate(ids[40:85])]
    result=rank_candidates(c,'r1','question',lexical,dense,exact=[ids[84]])
    assert len(result['candidates'])==60 and result['candidates'][0]['document']['doc_id']=='id-084'
    assert ids[84] not in {x for x,_ in dense[:40]} and 'exact' in {x['source'] for x in result['sources'][ids[84]]}
    assert ids[39] in result['sources'] and ids[80] not in result['sources']

def test_empty_title_does_not_leak_identifier_into_lexical_index(tmp_path):
    c=Catalog(tmp_path/'db');c.stage_documents([{'_id':'secret-id','title':'','text':'neutral content'}],META,'r1')
    uid=c.exact_documents('r1','',doc_id='secret-id')[0]
    assert c.get('r1',uid)['label']=='' and c.lexical('r1','secret-id')==[]
    assert c.lexical('r1','neutral')==[uid]
