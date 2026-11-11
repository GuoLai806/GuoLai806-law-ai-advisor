import json
from pathlib import Path
import pytest
from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import RetrievalService

META={'doc_id':'civil','version_id':'v1','title':'中华人民共和国民法典','verification_state':'approved','source_kind':'law','source_url':'https://example.test/law','source_sha256':'a'*64,'effective_from':'2021-01-01','effective_to':None}
PAGES=['第一条 合同的一般规定。\n第二条 合同无效。\n第三条 另一规定。']

def catalog(tmp_path):
 c=Catalog(tmp_path/'db');c.stage(PAGES,META,'r1');c.approve_source('r1','fixture','Approved fixture intervals for temporal test.');c.db.execute("INSERT INTO settings VALUES('dense_validated:r1','yes')");c.activate('r1',None);return c

@pytest.mark.asyncio
async def test_legal_query_finds_explicit_event_date_and_keeps_valid_date_metadata(tmp_path):
 c=catalog(tmp_path)
 b=await RetrievalService(c,profile='legal').retrieve('我在2021年6月1日签订的合同如何处理',return_retrieval_trace=True)
 assert b['query_plan']['event_date']=='2021-06-01' and b['query_plan']['query_time']
 assert b['status']=='ready' and b['evidence']
 assert all(e['temporal']['validity']=='valid' for e in b['evidence'])
 assert b['retrieval_trace']['temporal']

@pytest.mark.asyncio
async def test_event_before_effect_never_generates_from_future_article(tmp_path):
 c=catalog(tmp_path)
 b=await RetrievalService(c,profile='legal').retrieve('合同的效力',event_date='2020-06-01',return_retrieval_trace=True)
 assert b['status']=='insufficient_evidence' and b['evidence']==[]
 assert b['query_plan']['mode']=='temporal_validity_unknown'  # test fixture has approved public source, but effective interval should filter it
 assert b['degraded_reasons']
 # Actually validity should be not_yet_effective, so no candidate is eligible; inspect trace for reason.
 assert any(x['validity']=='not_yet_effective' for x in b['retrieval_trace']['temporal'].values())

@pytest.mark.asyncio
async def test_unverified_source_never_becomes_valid(tmp_path):
 c=Catalog(tmp_path/'db');c.stage(PAGES,{**META,'verification_state':'pending'},'r1')
 b=await RetrievalService(c,profile='legal').retrieve('合同的效力',revision='r1',event_date='2023-01-01')
 assert b['status']=='insufficient_evidence' and b['evidence']==[]
 assert 'temporal_validity_unknown' in b['degraded_reasons']

def test_api_query_trace_carries_temporal_dates_when_enabled(tmp_path):
 # API integration is covered separately; this guards the request model shape without network.
 from evidence_rag.api import Query
 q=Query(query='合同',session_id='s',event_date='2021-01-01',include_retrieval_trace=True)
 assert q.event_date.isoformat()=='2021-01-01' and q.include_retrieval_trace


def test_configured_service_uses_the_configured_default_profile():
    from pathlib import Path
    source=(Path(__file__).parents[2]/'evidence_rag/api.py').read_text()
    assert "profile=setting('DEFAULT_PROFILE','general')" in source
    assert "RetrievalService(db,dense,reranker,tokenizer,profile=profile" in source


def test_user_flow_launcher_uses_the_legal_compatibility_app():
    from pathlib import Path
    source=(Path(__file__).parents[2]/'benchmarks/serve_user_flow.py').read_text()
    assert "application='evidence_rag.api:legal_app' if config.get('profile')=='legal'" in source


def test_legal_app_lifespan_keeps_the_legal_service_profile():
    from fastapi.testclient import TestClient
    from evidence_rag.api import legal_app
    import os
    keys=(
        'EVIDENCE_RAG_DB','EVIDENCE_RAG_AUTH','EVIDENCE_RAG_DEFAULT_PROFILE',
        'EVIDENCE_RAG_OV_URL','EVIDENCE_RAG_OV_SCOPE',
    )
    old={key:os.environ.get(key) for key in keys}
    try:
        descriptor,path=__import__('tempfile').mkstemp(suffix='.sqlite3')
        os.close(descriptor)
        os.environ.update({
            'EVIDENCE_RAG_DB':path,
            'EVIDENCE_RAG_AUTH':'{"fixture-token":"fixture-user"}',
            'EVIDENCE_RAG_DEFAULT_PROFILE':'legal',
            'EVIDENCE_RAG_OV_URL':'http://127.0.0.1:1',
            'EVIDENCE_RAG_OV_SCOPE':'viking://resources/law-cn/r1',
        })
        with TestClient(legal_app) as client:
            assert client.app.state.default_profile=='legal'
            assert client.app.state.service.profile=='legal'
    finally:
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
