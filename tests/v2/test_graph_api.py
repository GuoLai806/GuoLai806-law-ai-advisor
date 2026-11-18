import json

import pytest
from fastapi.testclient import TestClient

from evidence_rag.api import app
from evidence_rag.catalog import Catalog
from evidence_rag.graph_index import GraphStore


@pytest.fixture
def service(tmp_path, monkeypatch):
    path = tmp_path / 'test.sqlite3'
    catalog = Catalog(path)
    docs = [
        {'_id': 'a', 'title': 'Alpha', 'text': 'Alpha requires Beta.\n\nUnrelated appendix.'},
        {'_id': 'b', 'title': 'Beta', 'text': 'Beta requires Gamma.'},
        {'_id': 'c', 'title': 'Gamma', 'text': 'Gamma.'},
    ]
    manifest = {'corpus_id': 'demo', 'version_id': '1',
                'source_kind': 'fixture', 'verification_state': 'approved'}
    catalog.stage_documents(docs, manifest, 'r1')
    catalog.activate('r1', None)
    units = {doc: catalog.get('r1', uid) for doc, uid in catalog.db.execute(
        'SELECT doc_id,unit_id FROM documents WHERE revision=?', ('r1',))}
    entities = []
    for doc in docs:
        unit = units[doc['_id']]
        entities.append({
            'entity_id': doc['_id'], 'name': doc['title'], 'type': 'module',
            'aliases': ['package'] if doc['_id'] != 'c' else [],
            'mentions': [{'unit_id': unit['unit_id'], 'char_start': 0,
                          'char_end': len(doc['title'])}],
        })
    edges = []
    for src, dst, start, end in [('a','b','2020-01-01',None), ('b','c','2021-01-01','2024-01-01')]:
        unit = units[src]
        part = unit['parts'][0]
        edges.append({'src_id':src, 'dst_id':dst, 'predicate':'depends_on', 'supports':[{
            'unit_id':unit['unit_id'], 'part_no':part['part_no'], 'char_start':0,
            'char_end':len(part['text']), 'content_sha256':unit['sha256'], 'quote':part['text'],
            'verification_state':'approved', 'temporal_status':'interval',
            'valid_from':start, 'valid_to':end,
        }]})
    store = GraphStore(catalog)
    store.migrate()
    build = store.stage('r1', {'revision':'r1', 'coverage_complete':True, 'extraction_identity':'fixture',
                               'entities':entities, 'edges':edges}, 'fixture')
    store.activate('r1', build['graph_build_id'], None)
    catalog.db.close()
    for key, value in {'DB':str(path), 'DEFAULT_PROFILE':'general', 'AUTH':json.dumps({'example':'viewer','another':'other'}),
                       'GRAPH_ENABLED':'1'}.items():
        monkeypatch.setenv('EVIDENCE_RAG_'+key,value)
    with TestClient(app) as client:
        yield client, {'Authorization':'Bearer example'}


def request(seed='Alpha', **options):
    return {'query':'What does Alpha depend on?', 'session_id':'fixture-session',
            'event_date':'2022-02-01', 'retrieval_mode':'graph_dependency',
            'graph_seed':seed, **options}


def test_graph_retrieve_returns_exact_parts_and_original_units(service):
    client, auth = service
    r = client.post('/api/v2/retrieve', json=request(), headers=auth)
    assert r.status_code == 200, r.text
    bundle = r.json()
    assert 'retrieval_trace' not in bundle
    graph = bundle['graph']
    assert graph['traversal_status'] == 'exhausted'
    assert graph['evidence_status'] == 'sufficient'
    assert 'evidence_unit_ids' not in graph
    assert [(p['src_name'], p['dst_name'], p['depth'], p['citation_id']) for p in graph['paths']] == [
        ('Alpha','Beta',1,'E1'), ('Beta','Gamma',2,'E2')]
    assert [e['text'] for e in bundle['evidence']] == ['Alpha requires Beta.\n\n', 'Beta requires Gamma.']
    last = graph['paths'][-1]
    source = client.get('/api/v2/units/'+last['unit_id'], params={'revision':'r1'}, headers=auth).json()
    assert source['sha256'] == last['content_sha256']
    assert source['text'][last['char_start']:last['char_end']] == 'Beta requires Gamma.'
    automatic = client.post('/api/v2/retrieve', json=request(seed=None), headers=auth).json()['graph']
    assert automatic['seed_id'] == 'a' and automatic['max_depth_reached'] == 2


def test_graph_dates_fallback_and_session_boundary(service, monkeypatch):
    client, auth = service
    uri = '/api/v2/retrieve'
    assert client.post(uri,json=request()).status_code == 401
    old = client.post(uri,json=request(event_date='2024-01-01'),headers=auth).json()['graph']
    assert old['edge_count'] == 1 and old['paths'][0]['dst_name'] == 'Beta'
    assert client.post(uri,json=request(),headers={'Authorization':'Bearer another'}).status_code == 403
    uncertain = client.post(uri,json=request(event_date=None,query='2023年 Alpha 的依赖'),headers=auth)
    assert uncertain.json()['graph']['stop_reason'] == 'ambiguous_date'
    ambiguous = client.post(uri,json=request(seed='package'),headers=auth).json()['graph']
    assert ambiguous['evidence_status'] == 'ambiguous_seed' and not ambiguous['paths']
    plain = client.post(uri,json=request(retrieval_mode='text'),headers=auth).json()
    assert 'graph' not in plain
    monkeypatch.setenv('EVIDENCE_RAG_GRAPH_ENABLED','0')
    fallback = client.post(uri,json=request(),headers=auth).json()
    assert fallback['graph']['stop_reason'] == 'graph_disabled'
    assert fallback['evidence'] == plain['evidence']


def test_graph_stream_returns_graph_without_claiming_model_answer(service):
    client, auth = service
    r = client.post('/api/v2/chat/stream',json=request(),headers=auth)
    assert r.status_code == 200
    assert 'event: evidence_ready' in r.text and '"max_depth_reached": 2' in r.text
    assert 'event: completed' in r.text and '"status": "evidence_only"' in r.text
    assert 'event: answer_delta' not in r.text


def test_graph_paging_pins_original_build_and_requires_identity(service):
    client, auth = service
    endpoint = '/api/v2/retrieve'
    initial = client.post(endpoint, json=request(), headers=auth).json()
    pin = initial['graph']['graph_build_id']
    assert client.post(endpoint, json=request(graph_path_offset=1), headers=auth).status_code == 422
    options = {'graph_path_offset':1, 'corpus_revision':'r1', 'graph_expected_build_id':pin}
    next_page = client.post(endpoint, json=request(**options), headers=auth).json()['graph']
    assert next_page['traversal_status'] == 'exhausted'
    assert next_page['edge_count'] == 2
    assert [p['dst_name'] for p in next_page['paths']] == ['Gamma']
    options['graph_expected_build_id'] = 'graph_' + '0'*32
    changed = client.post(endpoint, json=request(**options), headers=auth).json()['graph']
    assert changed['stop_reason'] == 'graph_changed' and not changed['paths']


def test_in_flight_source_revocation_is_rechecked_before_return(service, monkeypatch):
    from evidence_rag import graph_flow

    client, auth = service
    walk = graph_flow.walk_dependencies

    def revoke_after_traversal(catalog, *args, **kwargs):
        result = walk(catalog, *args, **kwargs)
        other = Catalog(catalog.path)
        try:
            manifest = json.loads(other.revision('r1')['manifest'])
            manifest['verification_state'] = 'pending'
            other.db.execute("UPDATE revisions SET manifest=? WHERE revision='r1'", (json.dumps(manifest),))
        finally:
            other.db.close()
        return result

    monkeypatch.setattr(graph_flow, 'walk_dependencies', revoke_after_traversal)
    bundle = client.post('/api/v2/retrieve', json=request(), headers=auth).json()
    assert bundle['graph']['stop_reason'] == 'source_revoked'
    assert not bundle['graph']['paths'] and not bundle['evidence']


def test_disabled_graph_uses_existing_normal_generation(service, monkeypatch):
    from evidence_rag.retrieval import RetrievalService

    client, auth = service
    evidence = [{'citation_id':'E1','unit_id':'u','title':'Source', 'text':'Alpha requires Beta.'}]

    async def ordinary(*args, **kwargs):
        return {'status':'ready','degraded':False,'query_plan':{'profile':'general','mode':'general_retrieval',
                'applicability_review':False}, 'corpus_revision':'r1', 'retrieval_ms':1,
                'budget':{'state':'measured','used_tokens':150,'omitted':[]}, 'evidence':evidence}

    class Generator:
        answer_format='grounded_json'
        max_output_tokens=64

        async def stream_events(self, query, sources):
            assert query == 'What does Alpha depend on?'
            assert sources == evidence
            yield {'type':'delta','delta':'Alpha requires Beta [E1].'}

    for key, value in {'GRAPH_ENABLED':'0', 'GENERATION_GATE':'grounded_evidence',
                       'MODEL_CONTEXT_WINDOW':'2048', 'FRAMING_RESERVE':'48', 'OUTPUT_RESERVE':'128'}.items():
        monkeypatch.setenv('EVIDENCE_RAG_'+key,value)
    monkeypatch.setattr(RetrievalService,'retrieve',ordinary)
    monkeypatch.setattr(app.state,'generator',Generator())
    response = client.post('/api/v2/chat', json=request(), headers=auth).json()
    assert response['graph']['uses_text_fallback'] is True
    assert response['answer'] == 'Alpha requires Beta [E1].'


def test_source_revoked_after_normal_retrieval_is_not_a_fallback(service, monkeypatch):
    from evidence_rag.retrieval import RetrievalService

    client, auth = service
    retrieve = RetrievalService.retrieve

    async def changed(self, *args, **kwargs):
        initial = await retrieve(self, *args, **kwargs)
        assert initial['evidence']
        manifest = json.loads(self.catalog.revision('r1')['manifest'])
        manifest['verification_state'] = 'pending'
        self.catalog.db.execute("UPDATE revisions SET manifest=? WHERE revision='r1'", (json.dumps(manifest),))
        return initial

    monkeypatch.setattr(RetrievalService,'retrieve',changed)
    bundle = client.post('/api/v2/retrieve', json=request(include_retrieval_trace=True), headers=auth).json()
    assert bundle['graph']['stop_reason'] == 'source_revoked'
    assert not bundle['graph']['uses_text_fallback']
    assert not bundle['graph']['paths'] and not bundle['evidence']
    assert 'retrieval_trace' not in bundle
