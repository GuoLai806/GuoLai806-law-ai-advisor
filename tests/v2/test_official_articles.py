from evidence_rag.catalog import Catalog

def test_official_tree_staging_preserves_hierarchy_not_label_guesses(tmp_path):
 c=Catalog(tmp_path/'db');arts=[{'label':'第一条','text':'第一条 内容。','structure':['第一编 别称测试','第一章 别称测试'],'locator':{'page':1},'external_id':'official-1'},{'label':'第二条','text':'第二条 内容。','structure':['附则'],'locator':{'page':2},'external_id':'official-2'}]
 c.stage_official_articles(arts,{'doc_id':'official','version_id':'v1','title':'测试法','source_kind':'law','verification_state':'approved','published_at':'2020-01-01','effective_from':'2021-01-01'},'r1')
 assert c.get('r1',c.exact('r1','第一条')[0])['structure']==['第一编 别称测试','第一章 别称测试']
 assert c.get('r1',c.exact('r1','第二条')[0])['structure']==['附则']
 assert c.temporal_version('r1')['valid_from']=='2021-01-01'
