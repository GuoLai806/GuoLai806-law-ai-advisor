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
    await adapter.ingest(uid,content)
    return {'state':'delivery_unknown','uri':uri}

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

async def validate_index(cat,adapter,revision):
    validate_scope(cat,adapter,revision)
    rows=cat.db.execute('SELECT unit_id,uri,text FROM units WHERE revision=? ORDER BY unit_id',(revision,)).fetchall()
    if not rows: raise ValueError('empty revision')
    outcomes=[]
    for row in rows:
        if not row['uri'] or not row['uri'].startswith(adapter.scope):
            outcomes.append({'unit_id':row['unit_id'],'result':'missing_mapping'}); continue
        try:
            hits=await adapter.find(''.join(row['text'].split())[:80])
            outcomes.append({'unit_id':row['unit_id'],'result':'visible' if any(x['uri']==row['uri'] for x in hits) else 'not_visible'})
        except Exception:
            outcomes.append({'unit_id':row['unit_id'],'result':'backend_error'})
    passed=all(x['result']=='visible' for x in outcomes)
    if passed:
        cat.db.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",('dense_validated:'+revision,'yes'))
        cat.db.execute("UPDATE jobs SET state='validated' WHERE revision=?",(revision,))
    else:
        cat.db.execute('DELETE FROM settings WHERE key=?',('dense_validated:'+revision,))
    return {'revision':revision,'index_visibility_validated':passed,'units':outcomes}

def main():
    p=argparse.ArgumentParser(); p.add_argument('operation',choices=['send','reconcile','validate']);p.add_argument('--db',required=True);p.add_argument('--revision',required=True);p.add_argument('--unit-id');p.add_argument('--url',required=True);p.add_argument('--scope',required=True);a=p.parse_args()
    if not a.scope.rstrip('/').endswith('/'+a.revision): raise ValueError('scope must end with revision')
    token=os.environ.get('OPENVIKING_API_KEY'); headers={'X-API-Key':token} if token else {}
    cat=Catalog(a.db); adapter=OpenViking(a.url,a.scope,headers=headers,timeout=60)
    if a.operation=='validate': result=asyncio.run(validate_index(cat,adapter,a.revision))
    else:
        if not a.unit_id: raise ValueError('--unit-id is required')
        action=index_one if a.operation=='send' else reconcile_one
        result=asyncio.run(action(cat,adapter,a.revision,a.unit_id))
    print(json.dumps(result))
if __name__=='__main__': main()
