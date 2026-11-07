"""Journaled OpenViking indexing: unknown writes are never replayed automatically."""
import argparse, asyncio, json, os
import httpx
from .catalog import Catalog, digest
from .retrieval import OpenViking, document_embedding_text

def validate_scope(cat,adapter,revision):
    row=cat.revision(revision)
    if not row: raise ValueError('unknown revision')
    manifest=json.loads(row['manifest'])
    expected=('viking://resources/rag/'+manifest['corpus_id']+'/'+revision) if manifest.get('profile')=='general' else ('viking://resources/law-cn/'+revision)
    if adapter.scope.rstrip('/')!=expected: raise ValueError('scope does not match corpus revision')
    if manifest.get('profile')=='general' and (manifest.get('embedding_serialization')!='title-newline-text-v1' or manifest.get('parser_version')!='generic-documents-v2'): raise ValueError('generic index serialization identity unavailable; reindex new revision')

async def index_one(cat, adapter, revision, uid):
    validate_scope(cat,adapter,revision)
    unit=cat.get(revision,uid)
    if not unit: raise ValueError('unknown unit')
    cat.db.execute('CREATE TABLE IF NOT EXISTS viking_journal(revision TEXT, unit_id TEXT, uri TEXT, content_hash TEXT, state TEXT, PRIMARY KEY(revision,unit_id))')
    row=cat.db.execute('SELECT * FROM viking_journal WHERE revision=? AND unit_id=?',(revision,uid)).fetchone()
    if row: return {'state':row['state'],'uri':row['uri']}
    uri=adapter.scope+uid+'/'+uid+'.md'
    # Save before dispatch, so a timeout or process crash is an unknown outcome, not retry permission.
    content=document_embedding_text(unit)
    cat.db.execute('INSERT INTO viking_journal VALUES(?,?,?,?,?)',(revision,uid,uri,digest(content),'delivery_unknown'))
    receipt=await adapter.ingest(uid,content)
    response=receipt['response']
    return {'state':'delivery_unknown','uri':uri,'receipt':{key:response[key] for key in ('root_uri','context_count','queue_status','usage') if key in response}}

async def reconcile_one(cat,adapter,revision,uid):
    validate_scope(cat,adapter,revision)
    row=cat.db.execute('SELECT * FROM viking_journal WHERE revision=? AND unit_id=?',(revision,uid)).fetchone()
    if not row or not row['uri'].startswith(adapter.scope): raise ValueError('no scoped journal entry')
    async with httpx.AsyncClient(timeout=adapter.timeout,headers=adapter.headers,transport=adapter.transport,trust_env=False) as client:
        r=await client.get(adapter.base+'/api/v1/content/read',params={'uri':row['uri'],'raw':'true'})
        r.raise_for_status(); obj=r.json()
    if obj.get('status')!='ok' or not isinstance(obj.get('result'),str) or digest(obj['result'])!=row['content_hash']: raise ValueError('remote content mismatch or unknown')
    cat.db.execute('BEGIN IMMEDIATE')
    try:
        cat.db.execute('UPDATE units SET uri=? WHERE revision=? AND unit_id=?',(row['uri'],revision,uid))
        cat.db.execute("UPDATE viking_journal SET state='content_verified' WHERE revision=? AND unit_id=?",(revision,uid)); cat.db.execute('COMMIT')
    except Exception: cat.db.execute('ROLLBACK'); raise
    return {'state':'content_verified','uri':row['uri'],'note':'vector visibility still requires separate validation'}

async def validate_index(cat,adapter,revision,concurrency=4,on_result=None):
    validate_scope(cat,adapter,revision)
    if not isinstance(concurrency,int) or isinstance(concurrency,bool) or not 1<=concurrency<=16: raise ValueError('validation concurrency must be 1..16')
    rows=cat.db.execute('SELECT unit_id,uri,text FROM units WHERE revision=? ORDER BY unit_id',(revision,)).fetchall()
    if not rows: raise ValueError('empty revision')
    gate=asyncio.Semaphore(concurrency)
    async def inspect(row):
        async with gate:
            if not row['uri'] or not row['uri'].startswith(adapter.scope):
                outcome={'unit_id':row['unit_id'],'result':'missing_mapping'}
            else:
                try:
                    content=document_embedding_text(cat.get(revision,row['unit_id']))
                    hits=await adapter.find(content)
                    matched=next((rank for rank,hit in enumerate(hits,1) if hit['uri']==row['uri']),None)
                    outcome={'unit_id':row['unit_id'],'result':'visible' if matched else 'not_visible','matching_rank':matched,'query_sha256':digest(content)}
                except Exception:
                    outcome={'unit_id':row['unit_id'],'result':'backend_error'}
            if on_result:on_result(outcome)
            return outcome
    workers = [asyncio.create_task(inspect(row)) for row in rows]
    try:
        outcomes = await asyncio.gather(*workers)
    finally:
        # A failing observer must not strand readers that still use this catalog.
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    passed=all(x['result']=='visible' for x in outcomes)
    if passed:
        cat.db.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",('dense_validated:'+revision,'yes'))
        cat.db.execute("UPDATE jobs SET state='validated' WHERE revision=?",(revision,))
    else:
        cat.db.execute('DELETE FROM settings WHERE key=?',('dense_validated:'+revision,))
    return {'revision':revision,'index_visibility_validated':passed,'validation_query':'exact indexed text serialization; integrity check, not query-quality benchmark','units':outcomes}

async def execute(args):
    """Keep the adapter's use and teardown on the same asyncio event loop."""
    if not args.scope.rstrip('/').endswith('/' + args.revision):
        raise ValueError('scope must end with revision')
    if args.operation != 'validate' and not args.unit_id:
        raise ValueError('--unit-id is required')
    token = os.environ.get('OPENVIKING_API_KEY')
    headers = {'X-API-Key': token} if token else {}
    cat = Catalog(args.db)
    adapter = None
    try:
        adapter = OpenViking(args.url, args.scope, headers=headers, timeout=60)
        if args.operation == 'validate':
            return await validate_index(cat, adapter, args.revision)
        action = index_one if args.operation == 'send' else reconcile_one
        return await action(cat, adapter, args.revision, args.unit_id)
    finally:
        try:
            if adapter is not None:
                await adapter.aclose()
        finally:
            cat.db.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('operation',choices=['send','reconcile','validate']);p.add_argument('--db',required=True);p.add_argument('--revision',required=True);p.add_argument('--unit-id');p.add_argument('--url',required=True);p.add_argument('--scope',required=True);a=p.parse_args()
    print(json.dumps(asyncio.run(execute(a))))
if __name__=='__main__': main()
