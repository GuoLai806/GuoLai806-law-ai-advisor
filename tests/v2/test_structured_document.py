from evidence_rag.catalog import Catalog
from evidence_rag.viking_index import structured_document

def test_openviking_document_keeps_full_article_and_explicit_structure(tmp_path):
 c=Catalog(tmp_path/'db');c.stage(['第一编 总则\n第一章 基本规定\n第一条 为了保护主体。\n第二条 其他。'],{'doc_id':'civil','version_id':'v1','title':'中华人民共和国民法典','source_kind':'synthetic','verification_state':'approved'},'r1')
 u=c.get('r1',c.exact('r1','第一条')[0]);text=structured_document(u)
 assert '第一编 总则' in text and '第一章 基本规定' in text and '第一条' in text and '为了保护主体。' in text
 assert 'unit_id' in text and 'content_sha256' in text
