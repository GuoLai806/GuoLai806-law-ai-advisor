import argparse, asyncio, hashlib, json, pathlib, time, math
from .catalog import Catalog, PARSER_VERSION
from .retrieval import RetrievalService

def main(default_profile='general'):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('ingest'); s.add_argument('--db',required=True); s.add_argument('--pdf'); s.add_argument('--input'); s.add_argument('--format',choices=['auto','jsonl','text','pdf'],default='auto'); s.add_argument('--manifest'); s.add_argument('--corpus-id'); s.add_argument('--revision'); s.add_argument('--profile',choices=['general','legal'])
    v=sub.add_parser('approve-source'); v.add_argument('--db',required=True); v.add_argument('--revision',required=True); v.add_argument('--reviewer',required=True); v.add_argument('--evidence',required=True)
    a=sub.add_parser('activate'); a.add_argument('--db',required=True); a.add_argument('--revision',required=True); a.add_argument('--expected')
    g=sub.add_parser('generate-synthetic'); g.add_argument('--db',required=True); g.add_argument('--revision',required=True); g.add_argument('--output',required=True); g.add_argument('--count',type=int,default=300)
    e=sub.add_parser('evaluate'); e.add_argument('--db',required=True); e.add_argument('--revision',required=True); e.add_argument('--dataset',required=True); e.add_argument('--output',required=True); e.add_argument('--variant',choices=['exact','bm25','rrf'],default='rrf')
    args=p.parse_args(); c=Catalog(args.db)
    if args.cmd=='ingest':
        profile=args.profile or default_profile
        path=args.input or args.pdf
        if not path: raise ValueError('--input or --pdf is required')
        m=json.loads(pathlib.Path(args.manifest).read_text()) if args.manifest else {}
        if profile=='legal':
            import pymupdf
            if not args.manifest: raise ValueError('legal profile requires a source manifest')
            raw=pathlib.Path(path).read_bytes();m.update(source_sha256=hashlib.sha256(raw).hexdigest(),parser_version=PARSER_VERSION,profile='legal')
            with pymupdf.open(stream=raw,filetype='pdf') as pdf: pages=[page.get_text() for page in pdf]
            result=c.stage(pages,m,args.revision)
        else:
            from .ingest import load_documents
            documents,provenance=load_documents(path,args.format)
            m={'corpus_id':args.corpus_id or m.get('corpus_id') or 'default','version_id':m.get('version_id') or args.revision or 'v1','source_kind':m.get('source_kind') or 'user-provided','verification_state':m.get('verification_state') or 'provided',**m,**provenance}
            result=c.stage_documents(documents,m,args.revision)
        print(json.dumps(result,ensure_ascii=False))
    elif args.cmd=='approve-source': c.approve_source(args.revision,args.reviewer,args.evidence); print(json.dumps({'revision':args.revision,'state':'verified'}))
    elif args.cmd=='activate': c.activate(args.revision,args.expected); print(json.dumps({'active':c.active()}))
    elif args.cmd=='generate-synthetic':
        rows=c.db.execute('SELECT unit_id,label,text FROM units WHERE revision=? ORDER BY rowid LIMIT ?',(args.revision,args.count)).fetchall()
        profile=json.loads(c.revision(args.revision)['manifest']).get('profile','legal')
        queries=[]
        for i,row in enumerate(rows):
            body=row['text'].replace(row['label'],'',1).strip(); excerpt=''.join(body.split())[:24]
            if not excerpt: continue
            target=c.get(args.revision,row['unit_id']);field='expected_ids' if profile=='general' else 'expected_labels';identity=target['document']['doc_id'] if profile=='general' else row['label']
            queries.append({'query':excerpt,field:[identity],'generation':'source-excerpt','source_index':i})
        pathlib.Path(args.output).write_text(json.dumps({'kind':'synthetic-source-derived-not-human-verified','queries':queries},ensure_ascii=False,indent=2)+'\n')
    else:
        data=json.loads(pathlib.Path(args.dataset).read_text()); results=[]
        profile=json.loads(c.revision(args.revision)['manifest']).get('profile','legal')
        exact=(lambda q:c.exact_documents(args.revision,q)) if profile=='general' else (lambda q:c.exact(args.revision,q))
        for item in data['queries']:
            start=time.perf_counter()
            if args.variant=='exact': ids=exact(item['query'])
            elif args.variant=='bm25': ids=c.lexical(args.revision,item['query'])
            else:
                from .retrieval import rrf
                ids=rrf({'exact':exact(item['query']),'bm25':c.lexical(args.revision,item['query'])})[0]
            items=[c.get(args.revision,x) for x in ids[:20]];labels=[x['document']['doc_id'] if profile=='general' else x['label'] for x in items];expected=item.get('expected_ids',item.get('expected_labels',[]));hits=len(set(labels)&set(expected))
            dcg=sum((1/math.log2(i+2)) for i,label in enumerate(labels[:10]) if label in expected); ideal=sum(1/math.log2(i+2) for i in range(min(len(expected),10))); results.append({'query':item['query'],'generation':item.get('generation'),'expected_labels':expected if profile=='legal' else None,'expected_ids':expected if profile=='general' else None,'ranked_labels':labels if profile=='legal' else None,'ranked_document_ids':labels if profile=='general' else None,'recall_at_20':hits/len(expected) if expected else None,'ndcg_at_10':dcg/ideal if ideal else None,'latency_ms':round((time.perf_counter()-start)*1000,3),'empty_retrieval':not bool(labels),'expect_empty':not bool(expected),'failure':('unexpected_results' if labels else None) if not expected else (None if hits==len(expected) else 'miss')})
        corpus=c.revision(args.revision); out={'dataset_kind':data.get('kind'),'variant':args.variant,'revision':args.revision,'corpus_manifest':json.loads(corpus['manifest']),'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'dataset_sha256':hashlib.sha256(pathlib.Path(args.dataset).read_bytes()).hexdigest(),'results':results,'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path('evidence_rag').glob('*.py'))},'mean_recall_at_20':None,'mean_ndcg_at_10':None}
        def mean(items,key):
            values=[x[key] for x in items if x[key] is not None]
            return sum(values)/len(values) if values else None
        out['mean_recall_at_20']=mean(results,'recall_at_20');out['mean_ndcg_at_10']=mean(results,'ndcg_at_10')
        tp=sum(x['expect_empty'] and x['empty_retrieval'] for x in results); fp=sum(not x['expect_empty'] and x['empty_retrieval'] for x in results); fn=sum(x['expect_empty'] and not x['empty_retrieval'] for x in results)
        out['empty_retrieval_proxy']={'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'notice':'Only empty-retrieval proxy, not answer-level abstention or legal correctness'}
        kinds=sorted({x['generation'] for x in results if x['generation']})
        out['slices']={kind:{'count':len(group),'mean_recall_at_20':mean(group,'recall_at_20'),'mean_ndcg_at_10':mean(group,'ndcg_at_10')} for kind in kinds for group in [[x for x in results if x['generation']==kind]]}
        pathlib.Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__': main()
