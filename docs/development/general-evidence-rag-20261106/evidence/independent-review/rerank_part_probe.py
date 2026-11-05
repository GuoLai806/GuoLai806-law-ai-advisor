import asyncio,json,tempfile
from pathlib import Path
from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import RetrievalService
class Dense:
    scope='viking://resources/rag/demo/r1/'
    async def find(self,q): return []
class Rank:
    def rank(self,query,items):
        output=[]
        for item in items:
            winner=next((p for p in item['parts'] if 'SECOND_PARAGRAPH_RELEVANT' in p['text']),None)
            output.append((item,10.0 if winner else 0.0))
        return output
class Tokenizer:
    def count_chat(self,messages): return sum(len(m['content']) for m in messages)
async def main():
    with tempfile.TemporaryDirectory(prefix='rag-parts-') as temp:
        c=Catalog(Path(temp)/'db')
        c.stage_documents([{'_id':'d1','title':'manual','text':'alpha first paragraph irrelevant.\n\nSECOND_PARAGRAPH_RELEVANT contains the actual evidence.'}],{'corpus_id':'demo','version_id':'v1','source_kind':'user-provided','verification_state':'provided'},'r1')
        c.activate('r1',None);uid=c.exact_documents('r1','',doc_id='d1')[0]
        c.db.execute('UPDATE units SET uri=? WHERE revision=? AND unit_id=?',('viking://resources/rag/demo/r1/'+uid+'/'+uid+'.md','r1',uid))
        c.db.execute("INSERT INTO settings VALUES('dense_validated:r1','yes')")
        result=await RetrievalService(c,dense=Dense(),reranker=Rank(),tokenizer=Tokenizer()).retrieve('alpha')
        print(json.dumps({'query':'alpha','ranker_best_part':2,'ranker_score':10.0,'actual_evidence':result['evidence'],'status':result['status'],'degraded':result['degraded']},ensure_ascii=False,indent=2))
        c.db.close()
asyncio.run(main())
