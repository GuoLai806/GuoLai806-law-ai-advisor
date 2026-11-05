import asyncio,json,tempfile
from pathlib import Path
from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import RetrievalService
async def main():
 with tempfile.TemporaryDirectory(prefix='rag-hash-retest-') as tmp:
    c=Catalog(Path(tmp)/'db');c.stage_documents([{'_id':'source','title':'Sample','text':'The original immutable text.'}],{'corpus_id':'demo','version_id':'v1','source_kind':'user-provided','verification_state':'provided'},'r1');c.activate('r1',None)
    uid=c.exact_documents('r1','',doc_id='source')[0];c.db.execute('UPDATE units SET text=? WHERE revision=? AND unit_id=?',('Altered after activation.','r1',uid))
    try:
        result=await RetrievalService(c).retrieve('query',document_id='source')
        out={'rejected':False,'result':result}
    except Exception as exc:out={'rejected':True,'error_type':type(exc).__name__,'error':str(exc)}
    print(json.dumps(out,indent=2));c.db.close()
asyncio.run(main())
