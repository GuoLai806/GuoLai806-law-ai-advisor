from types import SimpleNamespace

import pytest

from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import CrossEncoderReranker, RetrievalService


@pytest.mark.asyncio
@pytest.mark.parametrize('doc_id,trace,builtin,expected', [
    ('2',False,True,1), ('2',True,True,3), (None,False,True,3), ('2',False,False,3),
])
async def test_narrow_scoring_only_for_addressed_source_without_trace(tmp_path, doc_id, trace, builtin, expected):
    cat=Catalog(tmp_path/'catalog.db')
    try:
        docs=[{'_id':str(i),'title':f'Title {i}','text':f'shared words {i}\n\nsource text {i}'} for i in range(3)]
        cat.stage_documents(docs,{'corpus_id':'sample','version_id':'v1','source_kind':'benchmark','verification_state':'provided'},'r1')
        cat.activate('r1',None)
        scored=[]
        def rank(q,items):
            scored.extend(c['document']['doc_id'] for c in items)
            return [(dict(c,rerank_part_no=2),.75) for c in items]
        model=object.__new__(CrossEncoderReranker) if builtin else SimpleNamespace()
        model.rank=rank;model.revision=CrossEncoderReranker.MODELS['bge'][1]
        service=RetrievalService(cat,reranker=model,profile='general')
        reply=await service.retrieve('shared words',document_id=doc_id,return_retrieval_trace=trace)
        assert len(scored)==expected
        if doc_id is not None:
            assert scored[0]==doc_id and len(reply['evidence'])==1
            item=reply['evidence'][0]
            assert item['document_id']==doc_id and item['text']==docs[2]['text'] and item['reranker_score']==.75
        if trace:assert len(reply['retrieval_trace']['candidates'])==3
        else:assert 'retrieval_trace' not in reply
    finally:cat.db.close()
