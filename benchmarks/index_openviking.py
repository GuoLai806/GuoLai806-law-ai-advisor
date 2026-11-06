import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx
from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import OpenViking
from evidence_rag.viking_index import index_one, reconcile_one, validate_index
from .e2e import Journal, file_hash, now, save_json, source_hashes
from .prepare import jsonl

async def execute(args):
    output = Path(args.output)
    if output.exists() or Path(str(output) + '.journal.jsonl').exists():
        raise FileExistsError('Preserve earlier index reports')
    manifest = json.loads(Path(args.manifest).read_text())
    if file_hash(args.corpus) != manifest['corpus_sha256']:
        raise ValueError('Corpus identity mismatch')
    corpus = jsonl(args.corpus)
    if len(corpus) != manifest['corpus_count']:
        raise ValueError('Corpus size mismatch')
    catalog = Catalog(args.db)
    catalog.stage_documents([{'_id':key,'title':row.get('title') or '', 'text':row['text']}
                             for key, row in sorted(corpus.items())],
        {'corpus_id':args.corpus_id,'version_id':manifest['archive_sha256'],
         'source_kind':'benchmark','verification_state':'provided',
         'source_url':manifest['url'],'source_sha256':manifest['corpus_sha256'],'openviking_parse_mode':'no_split','openviking_ingest_layout':'unit-directory-no-split-v1'},args.revision)
    adapter = OpenViking(args.url, 'viking://resources/rag/' + args.corpus_id + '/' + args.revision, timeout=300)
    rows = catalog.db.execute('SELECT doc_id,unit_id FROM documents WHERE revision=? ORDER BY doc_id', (args.revision,)).fetchall()
    journal = Journal(str(output) + '.journal.jsonl')
    index_sources = {'evidence_rag/catalog.py','evidence_rag/retrieval.py','evidence_rag/viking_index.py','benchmarks/index_openviking.py','benchmarks/e2e.py','benchmarks/prepare.py'}
    before = {key:value for key,value in source_hashes().items() if key in index_sources}
    header = {'started_utc':now(),'operation':args.operation,'corpus_count':len(rows),
              'corpus_sha256':manifest['corpus_sha256'],'revision':args.revision,'scope':adapter.scope,
              'source_sha256':before,'concurrency':args.concurrency}
    journal.record({'type':'header',**header})
    outcomes = []
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=20,trust_env=False) as client:
            health = await client.get(args.url.rstrip('/')+'/health')
            health.raise_for_status()
            header['openviking_health'] = health.json()
        if args.operation == 'index':
            gate = asyncio.Semaphore(args.concurrency)
            stopped = False
            async def process(row):
                nonlocal stopped
                async with gate:
                    if stopped:
                        return
                    tick = time.monotonic()
                    result = {'document_id':row['doc_id'],'unit_id':row['unit_id']}
                    try:
                        dispatched=await index_one(catalog,adapter,args.revision,row['unit_id'])
                        result['dispatch_receipt']=dispatched.get('receipt')
                        reconciled = await reconcile_one(catalog,adapter,args.revision,row['unit_id'])
                        result.update(reconciled)
                    except Exception as exc:
                        stopped = True
                        result['failure'] = {'type':type(exc).__name__,'message':str(exc)}
                    result['elapsed_ms'] = round((time.monotonic()-tick)*1000,2)
                    outcomes.append(result)
                    journal.record({'type':'unit','completed':len(outcomes),'row':result})
                    if len(outcomes)%50 == 0 or result.get('failure'):
                        print(json.dumps({'completed':len(outcomes),'total':len(rows),'failures':sum(bool(item.get('failure')) for item in outcomes)}),flush=True)
            await asyncio.gather(*(process(row) for row in rows))
            complete = len(outcomes)==len(rows) and not any(row.get('failure') for row in outcomes)
        else:
            checked=[]
            def observe(row):
                checked.append(row)
                journal.record({'type':'validation','completed':len(checked),'row':row})
                if len(checked)%100==0:
                    print(json.dumps({'validated':len(checked),'total':len(rows),'failures':sum(item['result']!='visible' for item in checked)}),flush=True)
            validated = await validate_index(catalog,adapter,args.revision,args.concurrency,observe)
            outcomes = validated['units']
            complete = validated['index_visibility_validated']
            if complete and catalog.active() != args.revision:
                catalog.activate(args.revision,catalog.active())
        result = {**header,'finished_utc':now(),'complete':complete,'units':outcomes,
                  'elapsed_seconds':time.monotonic()-started,'source_unchanged':before=={key:value for key,value in source_hashes().items() if key in index_sources}}
        save_json(output,result)
        journal.record({'type':'complete','complete':complete,'report_sha256':file_hash(output)})
        print(json.dumps({'complete':complete,'units':len(outcomes),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
        return 0 if complete else 1
    finally:
        journal.close()
        catalog.db.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('index','validate'))
    for name in ('corpus','manifest','db','revision','corpus-id','url','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--concurrency',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.concurrency<=16:
        parser.error('concurrency must be 1..16')
    raise SystemExit(asyncio.run(execute(args)))

if __name__=='__main__':
    main()
