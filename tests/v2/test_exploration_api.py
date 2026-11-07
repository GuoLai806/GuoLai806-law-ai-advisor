"""Public exploration boundaries; synthetic corpus, no quality claims."""
import hashlib
import json
import pytest
from fastapi.testclient import TestClient
from evidence_rag.api import app

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'))
    monkeypatch.setenv('EVIDENCE_RAG_AUTH',json.dumps({'a':'alice','b':'bob'}))
    monkeypatch.setenv('EVIDENCE_RAG_EXPLORE_ENABLED','1')
    with TestClient(app) as client:
        c=app.state.catalog
        c.stage_documents([{'_id':'long','title':'Ocean source','text':'Ocean '+('complete original paragraph. '*140)}],
                          {'corpus_id':'demo','source_kind':'user-provided','verification_state':'provided','version_id':'v1'},'r1')
        c.activate('r1',None)
        yield client

BODY={'query':'Ocean','session_id':'inspect','max_evidence_tokens':1}
HEAD={'Authorization':'Bearer a'}

def test_explore_auth_owner_and_disable(client,monkeypatch):
    assert client.post('/api/v2/explore',json=BODY).status_code==401
    assert client.post('/api/v2/explore',json=BODY,headers=HEAD).status_code==200
    assert client.post('/api/v2/explore',json=BODY,headers={'Authorization':'Bearer b'}).status_code==403
    monkeypatch.setenv('EVIDENCE_RAG_EXPLORE_ENABLED','0')
    assert client.post('/api/v2/explore',json=BODY,headers=HEAD).status_code==404

def test_preview_bypasses_generation_budget_but_answer_does_not(client):
    class Tokenizer:
        def count_chat(self,messages):return sum(len(x['content']) for x in messages)
    app.state.service.tokenizer=Tokenizer()
    explored=client.post('/api/v2/explore',json=BODY,headers=HEAD).json()
    assert explored['status']=='exploration_only' and 'evidence' not in explored and 'budget' not in explored
    item=explored['items'][0]
    assert item['preview_truncated'] and len(item['preview'])==320
    source=client.get(item['read_pointer'],headers=HEAD).json()
    assert hashlib.sha256(source['text'].encode()).hexdigest()==item['document_sha256']
    retrieved=client.post('/api/v2/retrieve',json=BODY,headers=HEAD).json()
    assert not retrieved['evidence'] and retrieved['budget']['omitted']
    answer=client.post('/api/v2/chat',json=BODY,headers=HEAD).json()
    assert answer['status']=='evidence_only'

def test_same_version_read_survives_activation_answer_rejects_stale(client):
    explored=client.post('/api/v2/explore',json=BODY,headers=HEAD).json()
    c=app.state.catalog
    c.stage_documents([{'_id':'long','title':'New Ocean','text':'Changed Ocean source.'}],
                      {'corpus_id':'demo','source_kind':'user-provided','verification_state':'provided','version_id':'v2'},'r2')
    c.activate('r2','r1')
    original=client.get(explored['items'][0]['read_pointer'],headers=HEAD)
    assert original.status_code==200 and original.json()['revision']=='r1'
    assert client.post('/api/v2/explore',json={**BODY,'corpus_revision':'r1'},headers=HEAD).status_code==404
    # Direct chat never silently moves an explored query to a new revision.
    assert client.post('/api/v2/chat',json={**BODY,'corpus_revision':'r1'},headers=HEAD).status_code==404
