import datetime as dt
import json
import subprocess
import sys

import pytest
from evidence_rag.catalog import Catalog
from evidence_rag.graph_index import GraphStore, compile_candidates


@pytest.fixture
def graph(tmp_path):
    c=Catalog(tmp_path/'catalog.db')
    c.stage_documents([{'_id':'x','title':'Doc','text':'A needs B.\n\nB needs C.'}],
                      {'corpus_id':'demo','version_id':'1','source_kind':'docs','verification_state':'approved'},'r1')
    c.activate('r1',None)
    u=c.db.execute('SELECT unit_id FROM units WHERE revision=?',('r1',)).fetchone()[0]
    unit=c.get('r1',u)
    payload={'revision':'r1','extraction_identity':'test','coverage_complete':True,
             'entities':[{'entity_id':v,'name':v,'type':'module','aliases':[], 'mentions':[{'unit_id':u,'char_start':pos,'char_end':pos+1}]} for v,pos in [('A',0),('B',8),('C',20)]],
             'edges':[{'src_id':'A','dst_id':'B','predicate':'depends_on','supports':[{'unit_id':u,'part_no':1,'char_start':0,'char_end':10,'content_sha256':unit['sha256'],'quote':'A needs B.','verification_state':'approved','temporal_status':'interval','valid_from':'2020-01-01','valid_to':'2022-01-01'}]}]}
    store=GraphStore(c)
    yield c,store,payload,u
    c.db.close()


def test_lifecycle_and_half_open(graph):
    c,s,p,u=graph
    assert s.active_build('r1') is None
    assert not c.db.execute("SELECT 1 FROM sqlite_master WHERE name='graph_builds'").fetchone()
    s.migrate(); b=s.stage('r1',p,'one'); bid=b['graph_build_id']
    assert s.stage('r1',p,'one')['graph_build_id']==bid
    assert s.validate('r1',bid)['valid']
    s.activate('r1',bid,None)
    assert s.active_build('r1')['coverage_complete']
    assert s.resolve_seeds('r1',bid,'A')==[{'entity_id':'A','name':'A'}]
    assert len(s.resolve_seeds('r1',bid,'',[u]))==3
    assert len(list(s.outgoing('r1',bid,['A'],dt.date(2020,1,1))))==1
    assert not list(s.outgoing('r1',bid,['A'],dt.date(2022,1,1)))
    with pytest.raises(ValueError,match='active graph changed'): s.activate('r1',bid,None)
    assert s.active_build('r1')['graph_build_id']==bid


def test_rejects_cross_revision_bad_hash_and_partial_coverage(graph):
    c,s,p,u=graph; s.migrate()
    p['edges'][0]['supports'][0]['content_sha256']='wrong'
    with pytest.raises(ValueError,match='hash'): s.stage('r1',p,'bad')
    p['edges'][0]['supports'][0]['verification_state']='pending'
    with pytest.raises(ValueError,match='incomplete'): s.stage('r1',p,'pending')
    p['coverage_complete']=False
    bid=s.stage('r1',p,'pending')['graph_build_id']; s.activate('r1',bid,None)
    assert not list(s.outgoing('r1',bid,['A'],dt.date.today()))
    p['revision']='other'
    with pytest.raises(ValueError,match='revision mismatch'): s.stage('r1',p,'wrong')


def test_revoked_source_and_invalidated_manifest_disable_read(graph):
    c,s,p,u=graph; s.migrate(); bid=s.stage('r1',p,'one')['graph_build_id']; s.activate('r1',bid,None)
    row=c.revision('r1'); manifest=json.loads(row['manifest']); manifest['verification_state']='pending'
    c.db.execute('UPDATE revisions SET manifest=? WHERE revision=?',(json.dumps(manifest),'r1'))
    assert s.active_build('r1') is None
    with pytest.raises(ValueError,match='unavailable'): list(s.outgoing('r1',bid,['A'],dt.date.today()))


def test_compile_candidates_preserves_uri_as_pending(tmp_path):
    pytest.importorskip('yaml')
    (tmp_path/'entities').mkdir()
    for name in ['A','B']:
        (tmp_path/'entities'/f'{name}.md').write_text(f'---\ntype: entity\nid: {name}\ntitle: {name}\nentity_type: module\naliases: []\nsources: [viking://resources/doc.md]\n---\n')
    (tmp_path/'relations.jsonl').write_text(json.dumps({'from':'A','to':'B','relation':'depends_on','evidence':['viking://resources/doc.md']})+'\n')
    result=compile_candidates(tmp_path,'r1')
    assert result['coverage_complete'] is False
    assert result['edges'][0]['supports']==[{'verification_state':'pending','source_reference':'viking://resources/doc.md'}]
    run=subprocess.run([sys.executable,'-m','evidence_rag.graph_index','compile-candidates','--directory',str(tmp_path),'--revision','r1','--output',str(tmp_path/'out.json')],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads((tmp_path/'out.json').read_text())==result


def test_tampered_build_fails_activation_atomically(graph):
    c,s,p,u=graph; s.migrate(); bid=s.stage('r1',p,'one')['graph_build_id']
    c.db.execute('DELETE FROM graph_supports WHERE graph_build_id=?',(bid,))
    with pytest.raises(ValueError,match='derived data changed'): s.activate('r1',bid,None)
    assert s.active_build('r1') is None
