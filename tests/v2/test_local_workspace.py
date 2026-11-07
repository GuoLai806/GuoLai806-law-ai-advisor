"""Explicit local demo access; no model invocation or quality claims."""
import json
import pytest
from fastapi.testclient import TestClient
from evidence_rag.api import app


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv('EVIDENCE_RAG_DB', str(tmp_path / 'catalog.db'))
    monkeypatch.setenv('EVIDENCE_RAG_AUTH', json.dumps({'demo-token': 'demo', 'private-token': 'owner'}))
    monkeypatch.setenv('EVIDENCE_RAG_LOCAL_DEMO_USER', 'demo')
    monkeypatch.delenv('EVIDENCE_RAG_ADMINS', raising=False)
    return monkeypatch


def test_local_access_is_opt_in_and_existing_bearer_still_works(configured):
    configured.delenv('EVIDENCE_RAG_LOCAL_DEMO_USER')
    with TestClient(app, base_url='http://localhost', client=('127.0.0.1', 5000)) as client:
        assert client.get('/api/v2/models').status_code == 401
        assert client.get('/api/v2/models', headers={'Authorization': 'Bearer private-token'}).status_code == 200


def test_loopback_page_assets_and_identity_without_token(configured):
    with TestClient(app, base_url='http://localhost', client=('127.0.0.1', 5000)) as client:
        assert client.get('/api/v2/models').status_code == 200
        html = client.get('/').text
        assert 'type="password"' not in html and 'id="access"' not in html
        script = client.get('/assets/workspace.mjs')
        assert script.status_code == 200 and 'javascript' in script.headers['content-type']
        assert 'Authorization' not in script.text and 'demo-token' not in html + script.text
        c = app.state.catalog
        c.stage_documents([{'_id': 'ocean', 'title': 'Ocean', 'text': 'Ocean source evidence.'}],
                          {'corpus_id': 'test', 'source_kind': 'user-provided', 'verification_state': 'provided', 'version_id': '1'}, 'r1')
        c.activate('r1', None)
        body = {'query': 'Ocean', 'session_id': 'local-session'}
        result = client.post('/api/v2/explore', json=body, headers={'Origin': 'http://localhost', 'Sec-Fetch-Site': 'same-origin'})
        assert result.status_code == 200
        assert c.db.execute('SELECT owner FROM sessions WHERE session_id=?', ('local-session',)).fetchone()[0] == 'demo'
        assert client.post('/api/v2/explore', json=body, headers={'Authorization': 'Bearer private-token'}).status_code == 403


@pytest.mark.parametrize('headers', [
    {'Authorization': 'Bearer invalid'}, {'Authorization': 'Basic invalid'},
    {'Authorization': 'Bearer '}, {'Origin': 'https://other.example'},
    {'Origin': 'null'}, {'Sec-Fetch-Site': 'cross-site'}, {'Host': 'other.example'},
])
def test_invalid_credentials_and_cross_origin_cannot_fall_back(configured, headers):
    with TestClient(app, base_url='http://localhost', client=('127.0.0.1', 5000)) as client:
        assert client.get('/api/v2/models', headers=headers).status_code == 401


def test_forwarded_headers_do_not_grant_remote_local_access(configured):
    with TestClient(app, base_url='http://localhost', client=('203.0.113.1', 5000)) as client:
        assert client.get('/api/v2/models', headers={'X-Forwarded-For': '127.0.0.1', 'Origin': 'http://localhost'}).status_code == 401
        assert client.get('/api/v2/models', headers={'Authorization': 'Bearer private-token'}).status_code == 200


@pytest.mark.parametrize('user,admins', [('unknown', ''), ('demo', 'demo')])
def test_local_identity_must_exist_and_not_be_admin(configured, user, admins):
    configured.setenv('EVIDENCE_RAG_LOCAL_DEMO_USER', user)
    configured.setenv('EVIDENCE_RAG_ADMINS', admins)
    with pytest.raises(ValueError, match='non-administrator'):
        with TestClient(app):
            pass
