"""Index an already-staged official legal revision into OpenViking.

This command intentionally does not parse or activate the source. Callers must
stage and approve the legal revision first, preserving the single source-of-
truth Catalog. Unknown writes remain journaled and require reconciliation.
"""
import argparse,asyncio,json,time
from pathlib import Path
import httpx
from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import OpenViking
from evidence_rag.viking_index import index_one,reconcile_one,validate_index
from .e2e import Journal,now,save_json


async def execute(args):
    output=Path(args.output);journal_path=Path(str(output)+'.jsonl')
    if output.exists() or journal_path.exists():raise FileExistsError('preserve earlier index reports')
    cat=Catalog(args.db);adapter=None;journal=None;workers=[]
    try:
        row=cat.revision(args.revision)
        if not row:raise ValueError('unknown revision')
        meta=json.loads(row['manifest'])
        scope='viking://resources/law-cn/'+args.revision
        if args.scope!=scope:raise ValueError('scope must match revision')
        units=[dict(x) for x in cat.db.execute('SELECT unit_id FROM units WHERE revision=? ORDER BY label',(args.revision,))]
        if not units:raise ValueError('empty legal revision')
        adapter=OpenViking(args.url,args.scope,timeout=300);journal=Journal(journal_path)
        header={'started_utc':now(),'operation':args.operation,'revision':args.revision,'scope':args.scope,'units':len(units),'source_sha256':meta.get('source_sha256'),'article_count':len(units)}
        journal.record({'type':'header',**header});started=time.monotonic();outcomes=[]
        async with httpx.AsyncClient(timeout=20,trust_env=False) as client:
            health=await client.get(args.url.rstrip('/')+'/health');health.raise_for_status();header['openviking_health']=health.json()
        if args.operation=='index':
            gate=asyncio.Semaphore(args.concurrency);stopped=False
            async def process(item):
                nonlocal stopped
                async with gate:
                    if stopped:return
                    uid=item['unit_id'];tick=time.monotonic();result={'unit_id':uid}
                    try:
                        result['send']=await index_one(cat,adapter,args.revision,uid)
                        result.update(await reconcile_one(cat,adapter,args.revision,uid))
                    except Exception as exc:
                        stopped=True;result['failure']={'type':type(exc).__name__,'message':str(exc)[:500]}
                    result['elapsed_ms']=round((time.monotonic()-tick)*1000,2);outcomes.append(result);journal.record({'type':'unit','completed':len(outcomes),'row':result})
                    if len(outcomes)%50==0 or result.get('failure'):print(json.dumps({'completed':len(outcomes),'total':len(units),'failures':sum(bool(x.get('failure')) for x in outcomes)}),flush=True)
            workers=[asyncio.create_task(process(item)) for item in units];await asyncio.gather(*workers);complete=len(outcomes)==len(units) and not any(x.get('failure') for x in outcomes)
        else:
            def observer(row):journal.record({'type':'validation','row':row})
            validated=await validate_index(cat,adapter,args.revision,args.concurrency,observer);outcomes=validated['units'];complete=validated['index_visibility_validated']
            if complete and cat.active()!=args.revision:cat.activate(args.revision,cat.active())
        result={**header,'finished_utc':now(),'complete':complete,'units':outcomes,'elapsed_seconds':time.monotonic()-started,'article_count':len(units)}
        if complete and len(units)==1260:result['complete_article_tree']=True
        save_json(output,result);journal.record({'type':'complete','complete':complete});return 0 if complete else 1
    finally:
        for worker in workers:
            if not worker.done():worker.cancel()
        if workers:await asyncio.gather(*workers,return_exceptions=True)
        if adapter is not None:await adapter.aclose()
        if journal is not None:journal.close()
        cat.db.close()

def main():
    p=argparse.ArgumentParser()
    for name in ('operation',):p.add_argument(name,choices=('index','validate'))
    for name in ('db','revision','url','scope','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--concurrency',type=int,default=4);a=p.parse_args()
    if not 1<=a.concurrency<=16:p.error('concurrency must be 1..16')
    raise SystemExit(asyncio.run(execute(a)))
if __name__=='__main__':main()
