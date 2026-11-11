import argparse, asyncio, hashlib, json, pathlib, time, math
from .catalog import Catalog, PARSER_VERSION
from .retrieval import RetrievalService

def main(default_profile='general'):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('ingest'); s.add_argument('--db',required=True); s.add_argument('--pdf'); s.add_argument('--input'); s.add_argument('--format',choices=['auto','jsonl','text','pdf'],default='auto'); s.add_argument('--manifest'); s.add_argument('--corpus-id'); s.add_argument('--revision'); s.add_argument('--profile',choices=['general','legal'])
    t=sub.add_parser('set-temporal-version'); t.add_argument('--db',required=True); t.add_argument('--revision',required=True); t.add_argument('--valid-from'); t.add_argument('--valid-to'); t.add_argument('--published-at'); t.add_argument('--revision-note')
    v=sub.add_parser('approve-source'); v.add_argument('--db',required=True); v.add_argument('--revision',required=True); v.add_argument('--reviewer',required=True); v.add_argument('--evidence',required=True)
    a=sub.add_parser('activate'); a.add_argument('--db',required=True); a.add_argument('--revision',required=True); a.add_argument('--expected')
    g=sub.add_parser('generate-synthetic'); g.add_argument('--db',required=True); g.add_argument('--revision',required=True); g.add_argument('--output',required=True); g.add_argument('--count',type=int,default=300)
    s=sub.add_parser('search-official'); s.add_argument('--title',required=True); s.add_argument('--output',required=True)
    o=sub.add_parser('ingest-official'); o.add_argument('--db',required=True); o.add_argument('--title',required=True); o.add_argument('--revision',required=True); o.add_argument('--evidence',required=True); o.add_argument('--verified-tree',action='store_true',help='official 1260-article tree itself is source-verified')
    f=sub.add_parser('attach-legal-text'); f.add_argument('--db',required=True); f.add_argument('--revision',required=True); f.add_argument('--pdf',required=True); f.add_argument('--ocr',help='optional full OCR result JSON'); f.add_argument('--evidence',required=True)
    e=sub.add_parser('evaluate'); e.add_argument('--db',required=True); e.add_argument('--revision',required=True); e.add_argument('--dataset',required=True); e.add_argument('--output',required=True); e.add_argument('--variant',choices=['exact','bm25','rrf'],default='rrf')
    args=p.parse_args()
    if args.cmd=='search-official':
        from .legal_source import search_npc_document
        acquired=asyncio.run(search_npc_document(args.title));meta=acquired['metadata'].copy();meta.pop('content_root',None)
        pathlib.Path(args.output).write_text(json.dumps({'verification_state':acquired['verification_state'],'metadata':meta,'download_url_host':acquired.get('download_url_host'),'download_failure':acquired.get('download_failure'),'body_sha256':acquired.get('body_sha256')},ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'title':meta.get('title'),'published_at':meta.get('published_at'),'effective_from':meta.get('effective_from'),'verification_state':acquired['verification_state'],'download_url_host':acquired.get('download_url_host'),'body_sha256':acquired.get('body_sha256')},ensure_ascii=False));return
    c=Catalog(args.db)
    from .catalog import tokens
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
    elif args.cmd=='ingest-official':
        from .legal_source import search_npc_document,official_tree_articles,official_snapshot_manifest
        import uuid
        acquired=asyncio.run(search_npc_document(args.title));tree=official_tree_articles(acquired['metadata']['content_root']);manifest=official_snapshot_manifest(acquired)
        article_rows=[]
        for item in tree:
            label=item['label'];text=label
            loc={'kind':'official-tree','npc_index':item['index'],'external_id':item['external_id']}
            article_rows.append({'label':label,'text':text,'structure':item['structure'],'locator':loc,'external_id':item['external_id']})
        if args.verified_tree:
            manifest['verification_state']='approved';manifest['revision_note']='official database metadata and complete 1260-article tree verified; body text remains from separately parsed official source'
        result=c.stage_official_articles(article_rows,manifest,args.revision,idem='official:'+manifest['version_id']+':'+manifest['verification_state'])
        pathlib.Path(args.evidence).write_text(json.dumps({'metadata':{k:v for k,v in acquired['metadata'].items() if k!='content_root'},'source_state':acquired['verification_state'],'body_sha256':acquired.get('body_sha256'),'job':result,'article_count':len(article_rows),'article_tree':[{k:x[k] for k in ('label','structure','external_id','index')} for x in tree]},ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'revision':result['revision'],'articles':len(article_rows),'source_state':acquired['verification_state']},ensure_ascii=False))
    elif args.cmd=='attach-legal-text':
        import pymupdf
        raw=pathlib.Path(args.pdf).read_bytes();pages=None
        with pymupdf.open(stream=raw,filetype='pdf') as pdf:pages=[p.get_text() for p in pdf]
        from .catalog import parse_pages
        native_list=parse_pages(pages);parsed={x['label']:x for x in native_list}
        row=c.revision(args.revision)
        if not row:raise ValueError('unknown revision')
        meta=json.loads(row['manifest'])
        ocr_summary=None
        if args.ocr:
            import evidence_rag.ocr as ocrlib
            from .catalog import parse_pages
            native_articles=[{**a,'next_label':native_list[i+1]['label'] if i+1<len(native_list) else ''} for i,a in enumerate(native_list)]
            ocr_doc=json.loads(pathlib.Path(args.ocr).read_text());ocr_text='\n\n'.join(page['text'] for page in ocr_doc['pages'])
            merged,ocr_summary=ocrlib.validate_ocr_articles(ocr_text,native_articles)
            parsed={x['label']:x for x in merged};meta['ocr']=ocr_summary
        updated=[]
        for uid,label in c.db.execute('SELECT unit_id,label FROM units WHERE revision=?',(args.revision,)):
            item=parsed.get(label)
            if not item:continue
            text=item['text'];loc=item['locator']
            c.db.execute('UPDATE units SET text=?,sha256=?,locator=?,structure=?,raw_text=? WHERE revision=? AND unit_id=?',(text,hashlib.sha256(text.encode()).hexdigest(),json.dumps(loc,ensure_ascii=False),json.dumps(item.get('structure',[]),ensure_ascii=False),item.get('raw_text'),args.revision,uid))
            c.db.execute('DELETE FROM unit_fts WHERE revision=? AND unit_id=?',(args.revision,uid));c.db.execute('INSERT INTO unit_fts VALUES(?,?,?)',(args.revision,uid,tokens(label+' '+text)))
            updated.append(label)
        meta['source_body_sha256']=hashlib.sha256(raw).hexdigest();meta['source_sha256']=meta['source_body_sha256'];meta['revision_note']='official tree verified; body text imported from separately identified PDF and requires evidence review';meta['text_verification_state']='ocr_reconciled_review_required' if ocr_summary and ocr_summary.get('review_required') else ('ocr_reconciled' if ocr_summary else 'pending')
        c.db.execute('UPDATE revisions SET manifest=? WHERE revision=?',(json.dumps(meta,ensure_ascii=False,sort_keys=True),args.revision))
        pathlib.Path(args.evidence).write_text(json.dumps({'revision':args.revision,'pdf_sha256':meta['source_body_sha256'],'ocr':ocr_summary,'matched_articles':len(updated),'parsed_articles':len(parsed),'review_state':'ocr-reconciled-review-required' if ocr_summary and ocr_summary.get('review_required') else ('ocr-reconciled' if ocr_summary else 'needs-text-verification')},ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'revision':args.revision,'matched_articles':len(updated)},ensure_ascii=False))
    elif args.cmd=='set-temporal-version': c.set_temporal_version(args.revision,valid_from=args.valid_from,valid_to=args.valid_to,published_at=args.published_at,revision_note=args.revision_note); print(json.dumps(c.temporal_version(args.revision),ensure_ascii=False))
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
