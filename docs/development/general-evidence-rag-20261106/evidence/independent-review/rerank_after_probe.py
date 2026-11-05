import asyncio,json,tempfile
from pathlib import Path
from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import RetrievalService
class Dense:
    scope='viking://resources/rag/demo/r1/'
    async def find(self,q):return []
class Rank:
    def __init__(self,provide_part):self.provide_part=provide_part
    def rank(self,q,items):
        return [(({**item,'rerank_part_no':2} if self.provide_part else item),10.0) for item in items]
class Tokenizer:
    def count_chat(self,messages):return sum(len(m['content']) for m in messages)
async def main():
    with tempfile.TemporaryDirectory(prefix='rag-part-retest-') as tmp:
        c=Catalog(Path(tmp)/'db');c.stage_documents([{'_id':'d1','title':'manual','text':'alpha first paragraph irrelevant.\n\nSECOND_PARAGRAPH_RELEVANT contains the actual evidence.'}],{'corpus_id':'demo','version_id':'v1','source_kind':'user-provided','verification_state':'provided'},'r1');c.activate('r1',None)
        uid=c.exact_documents('r1','',doc_id='d1')[0];c.db.execute('UPDATE units SET uri=? WHERE revision=? AND unit_id=?',('viking://resources/rag/demo/r1/'+uid+'/'+uid+'.md','r1',uid));c.db.execute("INSERT INTO settings VALUES('dense_validated:r1','yes')")
        result={}
        for provided in (True,False):
            bundle=await RetrievalService(c,Dense(),Rank(provided),Tokenizer()).retrieve('alpha')
            result['part_identity' if provided else 'no_identity']={'evidence':bundle['evidence'],'status':bundle['status'],'degraded':bundle['degraded'],'degraded_reasons':bundle['degraded_reasons']}
        print(json.dumps(result,ensure_ascii=False,indent=2));c.db.close()
asyncio.run(main())
