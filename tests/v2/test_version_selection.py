import pytest
from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import RetrievalService

META={'doc_id':'civil','version_id':'v1','title':'中华人民共和国民法典','source_kind':'law','source_url':'https://example.test/law','source_sha256':'a'*64}
PAGES=['第一条 旧规则。','第一条 新规则。']
def prepare(tmp_path, *, activate_old=True):
 c=Catalog(tmp_path/'db')
 c.stage(PAGES,{**META,'version_id':'old','effective_from':'2010-01-01','effective_to':'2020-12-31','verification_state':'approved'},'old')
 c.approve_source('old','fixture','Verified historical effective interval.')
 c.db.execute("INSERT INTO settings VALUES('dense_validated:old','yes')")
 if activate_old:c.activate('old',None)
 c.stage(PAGES,{**META,'version_id':'new','effective_from':'2021-01-01','effective_to':None,'verification_state':'approved'},'new')
 c.approve_source('new','fixture','Verified new effective interval.')
 c.db.execute("INSERT INTO settings VALUES('dense_validated:new','yes')")
 return c

@pytest.mark.asyncio
async def test_service_auto_selects_revision_by_event_date(tmp_path):
 c=prepare(tmp_path)
 a=await RetrievalService(c,profile='legal').retrieve('第一条是什么',event_date='2015-01-01')
 b=await RetrievalService(c,profile='legal').retrieve('第一条是什么',event_date='2022-01-01')
 assert a['corpus_revision']=='old' and b['corpus_revision']=='new'
 assert a['query_plan']['revision_selection']['selected']=='old'
 assert b['query_plan']['revision_selection']['selected']=='new'

@pytest.mark.asyncio
async def test_no_verified_interval_never_falls_back_to_pending(tmp_path):
 c=Catalog(tmp_path/'db');c.stage(['第一条 未核验。'],{**META,'verification_state':'pending'},'pending')
 c.approve_source('pending','fixture','Interval is not approved for this fixture.')
 # approve_source turns the revision approved but leaves dates empty
 b=await RetrievalService(c,profile='legal').retrieve('第一条',revision='pending',event_date='2022-01-01')
 assert b['status']=='insufficient_evidence' and b['evidence']==[]


@pytest.mark.asyncio
async def test_auto_selection_can_use_known_version_when_another_is_active(tmp_path):
 c=prepare(tmp_path)
 b=await RetrievalService(c,profile='legal').retrieve('第一条是什么',event_date='2022-01-01')
 assert b['corpus_revision']=='new' and b['evidence']


@pytest.mark.asyncio
async def test_no_known_interval_leaves_active_fixture_available(tmp_path):
 c=Catalog(tmp_path/'db');m={k:v for k,v in META.items() if k!='source_sha256'};c.stage(['第一条 内容。'],{**m,'source_kind':'synthetic','verification_state':'approved'},'r1');c.activate('r1',None)
 b=await RetrievalService(c,profile='legal').retrieve('第一条')
 assert b['corpus_revision']=='r1' and b['evidence']


@pytest.mark.asyncio
async def test_api_selects_retained_version_when_active_is_another_revision(tmp_path,monkeypatch):
 from fastapi.testclient import TestClient
 from legal_rag.api import app
 import json
 monkeypatch.setenv('LEGAL_RAG_DB',str(tmp_path/'api.db'));monkeypatch.setenv('LEGAL_RAG_AUTH',json.dumps({'x':'user'}))
 with TestClient(app) as client:
  c=app.state.catalog
  c.stage(PAGES,{**META,'version_id':'old','effective_from':'2010-01-01','effective_to':'2020-12-31','verification_state':'approved'},'old')
  c.approve_source('old','fixture','Verified old interval.');c.db.execute("INSERT INTO settings VALUES('dense_validated:old','yes')");c.activate('old',None)
  c.stage(PAGES,{**META,'version_id':'new','effective_from':'2021-01-01','effective_to':None,'verification_state':'approved'},'new')
  c.approve_source('new','fixture','Verified new interval.');c.db.execute("INSERT INTO settings VALUES('dense_validated:new','yes')");c.activate('new','old')
  r=client.post('/api/v2/retrieve',json={'query':'第一条是什么','session_id':'s','event_date':'2015-01-01'},headers={'Authorization':'Bearer x'})
  assert r.status_code==200 and r.json()['corpus_revision']=='old'
