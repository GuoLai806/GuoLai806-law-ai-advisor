import asyncio, math, time, re, numbers
from collections import defaultdict
import httpx

class OpenViking:
    def __init__(self, base_url, scope, headers=None, timeout=5, transport=None):
        if not re.fullmatch(r'viking://resources/(?:law-cn/[A-Za-z0-9_-]+|rag/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+)/?',scope): raise ValueError('invalid resource scope')
        self.base, self.scope, self.headers, self.timeout, self.transport = base_url.rstrip('/'), scope.rstrip('/')+'/', headers or {}, timeout, transport
    async def find(self, query):
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers, transport=self.transport,trust_env=False) as client:
            r = await client.post(self.base+'/api/v1/search/find',json={'query':query,'target_uri':self.scope,'context_type':'resource','level':2,'read_content':False,'limit':40})
            r.raise_for_status(); data=r.json()
        if data.get('status') != 'ok' or not isinstance(data.get('result'),dict): raise ValueError('invalid OpenViking envelope')
        rows=data['result'].get('resources',[])
        if not isinstance(rows,list): raise ValueError('invalid resources')
        return [x for x in rows if isinstance(x,dict) and isinstance(x.get('uri'),str) and x['uri'].startswith(self.scope)]

    async def ingest(self, uid, content):
        # Never retry a write after ambiguous delivery; caller must journal and reconcile.
        if not __import__('re').fullmatch('[0-9a-f]{32}',uid): raise ValueError('invalid unit ID')
        target=self.scope+uid
        async with httpx.AsyncClient(timeout=self.timeout,headers=self.headers,transport=self.transport,trust_env=False) as client:
            upload=await client.post(self.base+'/api/v1/resources/temp_upload',files={'file':(uid+'.md',content.encode('utf-8'),'text/markdown')})
            upload.raise_for_status(); envelope=upload.json()
            if envelope.get('status')!='ok' or not isinstance(envelope.get('result'),dict) or not envelope['result'].get('temp_file_id'): raise ValueError('invalid upload envelope')
            response=await client.post(self.base+'/api/v1/resources',json={'temp_file_id':envelope['result']['temp_file_id'],'to':target,'wait':True,'strict':True,'create_parent':True,'processing_mode':'vectors_only'})
            response.raise_for_status(); result=response.json()
            if result.get('status')!='ok' or not isinstance(result.get('result'),dict): raise ValueError('invalid ingest envelope')
            root=result['result'].get('root_uri')
            if root is not None and root.rstrip('/') != target: raise ValueError('ingest root differs from requested target')
            return {'requested_uri':target,'expected_l2_uri':target+'/'+uid+'.md','response':result['result'],'state':'needs_reconciliation'}

def document_embedding_text(unit):
    """Exact text sent to the dense backend; no identifiers or relevance labels."""
    if unit.get('document') is not None:
        return (unit.get('title') or '')+'\n'+unit['text']
    return unit['text']

def reranker_document_text(unit, part=None):
    """Exact per-paragraph pair text; generic documents never expose IDs as text."""
    body=(part or {'text':unit['text']})['text']
    if unit.get('document') is not None:
        return (unit.get('title') or '')+'\n'+body
    return unit['title']+' '+unit['label']+' '+body

class CrossEncoderReranker:
    MODELS={'bge':('BAAI/bge-reranker-v2-m3','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'), 'qwen':('Qwen/Qwen3-Reranker-4B','22e683669bc0f0bd69640a1354a6d0aebcfeede5')}
    def __init__(self, name, device=None, dtype=None, batch_size=2):
        from sentence_transformers import CrossEncoder
        if dtype not in (None,'float16','float32'): raise ValueError('unsupported dtype')
        if batch_size<1: raise ValueError('batch size must be positive')
        self.batch_size=batch_size
        model, self.revision=self.MODELS[name]
        kwargs={'device':device,'revision':self.revision,'local_files_only':True,'max_length':4096}
        if dtype:
            import torch
            kwargs['model_kwargs']={'dtype':getattr(torch,dtype)}
        self.model=CrossEncoder(model, **kwargs)
        self.requested_dtype=dtype
    def rank(self, query, candidates):
        pairs=[]; owners=[]
        for i,c in enumerate(candidates):
            parts=c.get('parts') or [{'text':c['text']}]
            for part in parts:
                document=reranker_document_text(c,part)
                max_length=self.model.max_length
                if not max_length: raise ValueError('reranker token limit unknown')
                # Use the same model preprocessing and resolved prompt as predict(),
                # including the Qwen chat template, with truncation explicitly disabled.
                prompt=self.model.prompts.get(self.model.default_prompt_name) if self.model.default_prompt_name else None
                encoded=self.model.preprocess([(query,document)],prompt=prompt,processing_kwargs={'text':{'truncation':False},'chat_template':{'truncation':False}})
                length=int(encoded['attention_mask'][0].sum())
                if length>max_length: raise ValueError('whole paragraph exceeds reranker token limit')
                pairs.append((query,document)); owners.append(i)
        if not pairs: return []
        scores=self.model.predict(pairs,batch_size=getattr(self,'batch_size',2))
        if len(scores)!=len(pairs) or any(not math.isfinite(float(s)) for s in scores): raise ValueError('invalid reranker output')
        best=[-math.inf]*len(candidates)
        for owner,score in zip(owners,scores): best[owner]=max(best[owner],float(score))
        return sorted(zip(candidates,best),key=lambda p:p[1],reverse=True)

def rrf(lists, k=60):
    scores=defaultdict(float); sources=defaultdict(list)
    for source, items in lists.items():
        for rank, uid in enumerate(dict.fromkeys(items),1): scores[uid]+=1/(k+rank); sources[uid].append({'source':source,'rank':rank})
    return sorted(scores,key=lambda x:(-scores[x],x)),sources

def rank_candidates(catalog, revision, query, lexical, dense, exact=(), reranker=None, limit=60):
    """Shared pre-budget ranker. Inputs are ranked (unit_id, source-score) pairs."""
    if not 1<=limit<=60: raise ValueError('candidate limit must be 1..60')
    lexical=list(lexical)[:40];dense=list(dense)[:40]
    streams={'exact':list(dict.fromkeys(exact))[:10],'bm25':[uid for uid,_ in lexical],'dense':[uid for uid,_ in dense]}
    order,sources=rrf(streams);rrf_scores={uid:sum(1/(60+x['rank']) for x in entries) for uid,entries in sources.items()}
    raw={name:{} for name in ('bm25','dense')}
    for name,rows in (('bm25',lexical),('dense',dense)):
        for uid,score in rows:
            if score is not None:
                if not isinstance(score,numbers.Real) or not math.isfinite(score): raise ValueError('nonfinite source score')
                raw[name].setdefault(uid,float(score))
    for uid,entries in sources.items():
        for entry in entries:
            kind=entry['source']
            if uid in raw.get(kind,{}):entry.update(raw_score=raw[kind][uid],score_kind='sqlite_bm25_lower_is_better' if kind=='bm25' else 'dense_backend_score')
    order=list(dict.fromkeys(streams['exact']+order))[:limit]
    candidates=[catalog.get(revision,uid) for uid in order]
    if any(item is None for item in candidates): raise ValueError('candidate not in pinned revision')
    scores={}
    if reranker:
        ranked=reranker.rank(query,candidates)
        expected={x['unit_id'] for x in candidates};returned=[x['unit_id'] for x,score in ranked]
        if len(returned)!=len(expected) or set(returned)!=expected or any(not math.isfinite(float(score)) for _,score in ranked): raise ValueError('invalid reranker response')
        scores={x['unit_id']:float(score) for x,score in ranked};byid={x['unit_id']:x for x,score in ranked}
        order=list(dict.fromkeys(streams['exact']+[x['unit_id'] for x,score in ranked]))[:limit]
        candidates=[byid[uid] for uid in order]
    return {'candidates':candidates,'sources':dict(sources),'rrf_scores':rrf_scores,'reranker_scores':scores}

class RetrievalService:
    def __init__(self,catalog, dense=None, reranker=None, tokenizer=None, profile='general'): 
        if profile not in ('general','legal'): raise ValueError('invalid profile')
        self.catalog,self.dense,self.reranker,self.tokenizer,self.profile=catalog,dense,reranker,tokenizer,profile
    async def retrieve(self, query, revision=None, event_date=None, max_units=8, max_evidence_tokens=6000, document_id=None):
        start=time.monotonic(); rev=revision or self.catalog.active()
        if not rev or not self.catalog.revision(rev): raise LookupError('revision unavailable')
        manifest=__import__('json').loads(self.catalog.revision(rev)['manifest'])
        corpus_profile=manifest.get('profile','legal')
        if corpus_profile!=self.profile: raise LookupError('revision profile mismatch')
        if self.profile=='general' and manifest.get('parser_version')!='generic-documents-v2': raise LookupError('generic revision requires reindex')
        lexical=self.catalog.lexical_scored(rev,query); raw_scores={'bm25':dict(lexical),'dense':{}}
        lists={'exact':[], 'bm25':[uid for uid,score in lexical]}; degraded=[]
        wants_source=False
        if self.profile=='legal':
            title=manifest.get('title','')
            named=re.findall(r'《([^》]+)》',query)
            known=['民法典','刑法','劳动合同法','公司法','民事诉讼法','行政诉讼法','刑事诉讼法','婚姻法','继承法']
            named.extend(name for name in known if name in query)
            explicit=re.search(r'([^\s，。？?《》]{2,30}(?:法|条例|规定))第[一二三四五六七八九十百千万零〇两\d]+条',query)
            if explicit:
                law=re.sub(r'^(?:请问|请查询|请查阅|请查|查询|查阅|查看|我想查询|我想查|我要查)+','',explicit.group(1))
                if law: named.append(law)
            if named and any(name not in title and title not in name for name in named):
                return {'status':'insufficient_evidence','query_plan':{'mode':'out_of_scope_law','event_date':str(event_date) if event_date else None,'applicability_review':False},'corpus_revision':rev,'degraded':True,'degraded_reasons':['law_out_of_scope'],'missing_facts':[],'evidence':[],'budget':{'state':'not_applicable','used_tokens':0,'omitted':[]},'token_budget_status':'not_applicable','retrieval_ms':round((time.monotonic()-start)*1000,2)}
            lists['exact']=self.catalog.exact(rev,query)
            wants_source=bool(re.search(r'第[一二三四五六七八九十百千万零〇两\d]+条',query)) and bool(re.search(r'原文|全文|查询|查阅|条文内容',query)) and not re.search(r'适用|我的|我能|案件|纠纷|诉讼',query)
            if wants_source:
                from .catalog import canonical_label
                requested={canonical_label(label) for label in re.findall(r'第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?',query)}
                matched={canonical_label(self.catalog.get(rev,uid)['label']) for uid in lists['exact']}
            if wants_source and requested != matched:
                return {'status':'insufficient_evidence','query_plan':{'mode':'source_lookup','event_date':str(event_date) if event_date else None,'applicability_review':False},'corpus_revision':rev,'degraded':True,'degraded_reasons':['article_not_found'],'missing_facts':[],'evidence':[],'budget':{'state':'not_applicable','used_tokens':0,'omitted':[]},'token_budget_status':'not_applicable','retrieval_ms':round((time.monotonic()-start)*1000,2)}
        else:
            lists['exact']=self.catalog.exact_documents(rev,query,document_id)
            wants_source=document_id is not None
            if wants_source and not lists['exact']:
                return {'status':'insufficient_evidence','query_plan':{'mode':'source_lookup','profile':'general','applicability_review':False},'corpus_revision':rev,'degraded':True,'degraded_reasons':['document_not_found'],'missing_facts':[],'evidence':[],'budget':{'state':'not_applicable','used_tokens':0,'omitted':[]},'token_budget_status':'not_applicable','retrieval_ms':round((time.monotonic()-start)*1000,2)}
        if self.dense:
            try:
                if self.profile=='general' and manifest.get('embedding_serialization')!='title-newline-text-v1': raise ValueError('generic embedding serialization mismatch')
                expected_scope=('viking://resources/rag/'+manifest['corpus_id']+'/'+rev) if self.profile=='general' else ('viking://resources/law-cn/'+rev)
                if self.dense.scope.rstrip('/')!=expected_scope: raise ValueError('dense scope revision mismatch')
                if not self.catalog.db.execute('SELECT 1 FROM units WHERE revision=? AND uri IS NOT NULL LIMIT 1',(rev,)).fetchone(): raise ValueError('dense index not mapped')
                if not self.catalog.db.execute('SELECT 1 FROM settings WHERE key=? AND value=?',('dense_validated:'+rev,'yes')).fetchone(): raise ValueError('dense index not validated')
                rows=await self.dense.find(query); ids=[]
                for x in rows:
                    r=self.catalog.db.execute('SELECT unit_id FROM units WHERE revision=? AND uri=?',(rev,x['uri'])).fetchone()
                    if r:
                        ids.append(r[0])
                        score=x.get('score')
                        if isinstance(score,numbers.Real) and math.isfinite(score): raw_scores['dense'].setdefault(r[0],score)
                lists['dense']=ids
            except Exception: degraded.append('dense_unavailable')
        else: degraded.append('dense_not_configured')
        dense_pairs=[(uid,raw_scores['dense'].get(uid)) for uid in lists.get('dense',[])]
        try:
            ranking=await asyncio.wait_for(asyncio.to_thread(rank_candidates,self.catalog,rev,query,lexical,dense_pairs,lists['exact'],self.reranker),timeout=10)
            if not self.reranker:degraded.append('reranker_not_configured')
        except Exception:
            degraded.append('reranker_failed')
            ranking=rank_candidates(self.catalog,rev,query,lexical,dense_pairs,lists['exact'])
        candidates=ranking['candidates'];sources=ranking['sources'];rerank_scores=ranking['reranker_scores']
        byid={c['unit_id']:c for c in candidates};selected=[c['unit_id'] for c in candidates[:max_units]]
        evidence=[]
        if self.profile=='legal':
            source_lookup=bool(lists['exact']) and wants_source
            general=bool(re.search(r'是什么|一般规定|一般如何|介绍|概念',query)) and not re.search(r'我的|我能|本人|案件|纠纷|起诉|被告',query)
            mode='source_lookup' if source_lookup else ('general_explanation' if general else 'case_question')
            applicability_review=mode=='case_question' and bool(event_date)
            missing_facts=(['事件发生日期及具体事实'] if mode=='case_question' and not event_date else (['法律适用及过渡规定需核验'] if applicability_review else []))
        else:
            source_lookup=bool(lists['exact']) and wants_source
            mode='source_lookup' if source_lookup else 'general_retrieval'
            applicability_review=False;missing_facts=[]
        if source_lookup: selected=[uid for uid in selected if uid in lists['exact']]
        for i,uid in enumerate(selected,1):
            c=byid[uid];doc=c.get('document') or {}
            item={'citation_id':f'E{i}','unit_id':uid,'document_id':doc.get('doc_id'),'title':c['title'],'text':c['text'],'source_url':doc.get('source_url') or manifest.get('source_url'),'source_kind':doc.get('source_kind') or manifest.get('source_kind'),'source_locator':c['locator'],'content_sha256':c['sha256'],'retrieval_sources':sources[uid],'reranker_score':rerank_scores.get(uid),'verification_state':manifest.get('verification_state'),'profile':self.profile}
            if self.profile=='legal':item.update(article_label=c['label'],candidate_relations=c['relations'],effective_from=manifest.get('effective_from'),effective_to=manifest.get('effective_to'),applicability_review=applicability_review)
            elif not source_lookup and c['parts']:
                part=self.catalog.best_part(rev,uid,query)
                if part:item.update(text=part['text'],source_locator=part['locator'],content_sha256=part['sha256'],part_no=part['part_no'],document_sha256=c['sha256'],has_more=len(c['parts'])>1)
            evidence.append(item)
        from .budget import assemble
        budgeted,budget=assemble(evidence,self.tokenizer,max_evidence_tokens,query)
        # Retrieval results can be inspected without tokenizer, but are not generation-ready.
        if self.tokenizer: evidence=budgeted
        return {'query_plan':{'mode':mode,'profile':self.profile,'event_date':str(event_date) if event_date and self.profile=='legal' else None,'applicability_review':applicability_review},'budget':budget, 'status':('insufficient_evidence' if not evidence else ('needs_clarification' if mode=='case_question' and not event_date else 'ready')),'corpus_revision':rev,'degraded':bool(degraded),'degraded_reasons':degraded,'missing_facts':missing_facts,'token_budget_status':budget['state'],'evidence':evidence,'retrieval_ms':round((time.monotonic()-start)*1000,2)}
