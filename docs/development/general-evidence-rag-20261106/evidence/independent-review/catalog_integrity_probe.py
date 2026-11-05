import asyncio,hashlib,json,tempfile
from pathlib import Path
from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import RetrievalService
async def main():
    with tempfile.TemporaryDirectory(prefix='rag-hash-probe-') as tmp:
        c=Catalog(Path(tmp)/'db')
        c.stage_documents([{'_id':'source','title':'Sample','text':'The original immutable text.'}],{'corpus_id':'demo','version_id':'v1','source_kind':'user-provided','verification_state':'provided'},'r1');c.activate('r1',None)
        uid=c.exact_documents('r1','',doc_id='source')[0]
        c.db.execute('UPDATE units SET text=? WHERE revision=? AND unit_id=?',('Altered after activation.','r1',uid))
        result=await RetrievalService(c).retrieve('query',document_id='source')
        entry=result['evidence'][0]
        print(json.dumps({'status':result['status'],'degraded':result['degraded'],'returned_text':entry['text'],'reported_hash':entry['content_sha256'],'actual_hash':hashlib.sha256(entry['text'].encode()).hexdigest(),'hash_matches':entry['content_sha256']==hashlib.sha256(entry['text'].encode()).hexdigest(),'method':'isolated post-activation DB corruption simulation'},indent=2))
        c.db.close()
asyncio.run(main())
