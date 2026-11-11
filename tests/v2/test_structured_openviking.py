from evidence_rag.retrieval import document_embedding_text
from evidence_rag.catalog import Catalog

def test_legal_embedding_text_includes_path_and_full_article_without_source_scope(tmp_path):
 c=Catalog(tmp_path/'db');c.stage(['第一编 总则\n第一章 基本规定\n第一条 为了保护民事主体。\n第二条 跨界条文。'],{'doc_id':'civil','version_id':'v1','title':'中华人民共和国民法典','source_kind':'law','verification_state':'approved','effective_from':'2021-01-01','source_url':'https://example.test/law','source_sha256':'a'*64},'r1')
 uid=c.exact('r1','第一条')[0];u=c.get('r1',uid)
 text=document_embedding_text(u)
 assert '中华人民共和国民法典' in text and '第一编 总则' in text and '第一章 基本规定' in text and '第一条' in text and '为了保护民事主体' in text
 assert '第一条' in text and '第二条' not in text
