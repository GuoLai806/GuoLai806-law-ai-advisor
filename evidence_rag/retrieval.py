import asyncio, math, time, re, numbers
from datetime import date
from collections import defaultdict
from contextlib import asynccontextmanager
import httpx
from .temporal import TemporalConfig, TimeContext, candidate_time_score, temporal_compatibility
from .legal_versions import select_version
from .time_query import extract_event_date

class OpenViking:
    """A scoped adapter whose requests share one explicitly owned HTTP client.

    Injected clients are borrowed unless ``owns_client=True`` is supplied. The
    adapter always drains its own in-flight operations before closing an owned
    client. A borrowed client must outlive the adapter. ``reuse_client=False``
    retains the original per-operation lifecycle for rollback and comparison.
    """
    def __init__(self, base_url, scope, headers=None, timeout=5, transport=None,
                 client=None, owns_client=None, reuse_client=True):
        if not re.fullmatch(r'viking://resources/(?:law-cn/[A-Za-z0-9_-]+|rag/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+)/?', scope):
            raise ValueError('invalid resource scope')
        if not isinstance(reuse_client, bool) or (owns_client is not None and not isinstance(owns_client, bool)):
            raise ValueError('invalid client ownership configuration')
        if client is not None:
            if not isinstance(client, httpx.AsyncClient):
                raise ValueError('client must be an httpx.AsyncClient')
            if transport is not None or not reuse_client:
                raise ValueError('injected client conflicts with transport or per-operation mode')
            if client.is_closed or client._trust_env:
                raise ValueError('injected client must be open with trust_env=False')
        elif owns_client is False:
            raise ValueError('self-created clients must be owned by the adapter')
        self.base = base_url.rstrip('/')
        self.scope = scope.rstrip('/') + '/'
        self.headers = dict(headers or {})
        self.timeout, self.transport = timeout, transport
        self.reuse_client = reuse_client
        self.owns_client = client is None if owns_client is None else owns_client
        self._client = client
        self._state = 'open'
        self._active = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self._close_task = None

    def _make_client(self):
        return httpx.AsyncClient(timeout=self.timeout, headers=self.headers,
                                 transport=self.transport, trust_env=False,
                                 follow_redirects=False)

    def _begin_request(self):
        # No await between the state check and registration: shutdown cannot
        # miss a request that has already been accepted on this event loop.
        if self._state != 'open':
            raise RuntimeError('OpenViking adapter is closing or closed')
        self._active += 1
        self._drained.clear()

    def _end_request(self):
        self._active -= 1
        if self._active == 0:
            self._drained.set()

    @staticmethod
    def _observe_completion(task):
        if not task.cancelled():
            task.exception()  # Observe errors even if every shielded waiter was cancelled.

    async def _close_transient(self, client):
        try:
            await client.aclose()
        finally:
            self._end_request()

    @asynccontextmanager
    async def _request_client(self):
        self._begin_request()
        transient = None
        try:
            if self.reuse_client:
                if self._client is None:
                    self._client = self._make_client()
                client = self._client
            else:
                transient = self._make_client()
                client = transient
            yield client
        finally:
            if transient is None:
                self._end_request()
            else:
                # A second cancellation must not leak a transient connection or
                # let shutdown finish before that connection has been closed.
                cleanup = asyncio.create_task(self._close_transient(transient))
                cleanup.add_done_callback(self._observe_completion)
                await asyncio.shield(cleanup)

    async def _finish_close(self):
        try:
            await self._drained.wait()
            if self.owns_client and self._client is not None:
                await self._client.aclose()
        finally:
            self._state = 'closed'

    async def aclose(self):
        if self._close_task is None:
            self._state = 'closing'
            self._close_task = asyncio.create_task(self._finish_close())
            self._close_task.add_done_callback(self._observe_completion)
        # Cancellation of a shutdown waiter does not cancel the shutdown itself.
        await asyncio.shield(self._close_task)

    async def __aenter__(self):
        if self._state != 'open':
            raise RuntimeError('OpenViking adapter is closing or closed')
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.aclose()

    async def _post(self, client, path, **kwargs):
        # Override injected-client defaults as well: neither redirects nor a
        # different deadline may silently relax this adapter's request contract.
        return await client.post(self.base + path, headers=self.headers,
                                 timeout=self.timeout, follow_redirects=False, **kwargs)

    async def find(self, query):
        async with self._request_client() as client:
            r = await self._post(client, '/api/v1/search/find', json={
                'query': query, 'target_uri': self.scope, 'context_type': 'resource',
                'level': 2, 'read_content': False, 'limit': 40})
            r.raise_for_status()
            data = r.json()
            if data.get('status') != 'ok' or not isinstance(data.get('result'), dict):
                raise ValueError('invalid OpenViking envelope')
            rows = data['result'].get('resources', [])
            if not isinstance(rows, list):
                raise ValueError('invalid resources')
            return [x for x in rows if isinstance(x, dict) and isinstance(x.get('uri'), str)
                    and x['uri'].startswith(self.scope)]

    async def ingest(self, uid, content, target_uri=None):
        # Never retry a write after ambiguous delivery; caller must journal and reconcile.
        if not re.fullmatch('[0-9a-f]{32}', uid):
            raise ValueError('invalid unit ID')
        target = target_uri or (self.scope + uid)
        if target.endswith('.md'):target=target.rsplit('/',1)[0]
        async with self._request_client() as client:
            upload = await self._post(client, '/api/v1/resources/temp_upload', files={
                'file': (uid + '.md', content.encode('utf-8'), 'text/markdown')})
            upload.raise_for_status()
            envelope = upload.json()
            if envelope.get('status') != 'ok' or not isinstance(envelope.get('result'), dict) or not envelope['result'].get('temp_file_id'):
                raise ValueError('invalid upload envelope')
            response = await self._post(client, '/api/v1/resources', json={
                'temp_file_id': envelope['result']['temp_file_id'], 'to': target,
                'wait': True, 'strict': True, 'create_parent': True,
                'processing_mode': 'vectors_only', 'args': {'parse_mode': 'no_split'}})
            response.raise_for_status()
            result = response.json()
            if result.get('status') != 'ok' or not isinstance(result.get('result'), dict):
                raise ValueError('invalid ingest envelope')
            root = result['result'].get('root_uri')
            if root is not None and root.rstrip('/') != target:
                raise ValueError('ingest root differs from requested target')
            # OpenViking assigns a generated leaf name within the requested
            # directory; reconcile reads the reported root and verifies content.
            expected_l2 = target + '/' + uid + '.md'
            return {'requested_uri': target, 'expected_l2_uri': expected_l2,
                    'response': result['result'], 'state': 'needs_reconciliation'}

def document_embedding_text(unit):
    """Exact text sent to the dense backend; no identifiers or relevance labels."""
    if unit.get('document') is not None:
        return (unit.get('title') or '')+'\n'+unit['text']
    structure=unit.get('structure') or []
    # Structural context helps recall while the full article remains the
    # authoritative evidence. This serialization is hashed and validated.
    prefix=' / '.join(structure)
    if prefix:
        return (unit.get('title') or '')+'\n'+prefix+'\n'+unit['label']+'\n'+unit['text']
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
        pairs=[]; owners=[]; pair_parts=[]
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
                pairs.append((query,document)); owners.append(i);pair_parts.append(part.get('part_no'))
        if not pairs: return []
        scores=self.model.predict(pairs,batch_size=getattr(self,'batch_size',2))
        if len(scores)!=len(pairs) or any(not math.isfinite(float(s)) for s in scores): raise ValueError('invalid reranker output')
        best=[-math.inf]*len(candidates);best_parts=[None]*len(candidates)
        for owner,score,part_no in zip(owners,scores,pair_parts):
            if float(score)>best[owner]:best[owner]=float(score);best_parts[owner]=part_no
        annotated=[{**c,'rerank_part_no':best_parts[i]} for i,c in enumerate(candidates)]
        return sorted(zip(annotated,best),key=lambda p:p[1],reverse=True)

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
    result={'candidates':candidates,'sources':dict(sources),'rrf_scores':rrf_scores,'reranker_scores':{}}
    return rerank_candidates(result,query,streams['exact'],reranker) if reranker else result

def rerank_candidates(ranking,query,exact,reranker):
    """Model work uses materialized evidence, never a request-scoped database."""
    candidates=ranking['candidates'];limit=len(candidates);scores={}
    if reranker:
        ranked=reranker.rank(query,candidates)
        expected={x['unit_id'] for x in candidates};returned=[x['unit_id'] for x,score in ranked]
        if len(returned)!=len(expected) or set(returned)!=expected or any(not math.isfinite(float(score)) for _,score in ranked): raise ValueError('invalid reranker response')
        scores={x['unit_id']:float(score) for x,score in ranked};byid={x['unit_id']:x for x,score in ranked}
        order=list(dict.fromkeys(list(exact)+[x['unit_id'] for x,score in ranked]))[:limit]
        candidates=[byid[uid] for uid in order]
    return {**ranking,'candidates':candidates,'reranker_scores':scores}

def temporal_score(candidate, context, config=None):
    """Return an auditable temporal score for a catalog candidate.

    The online path uses this as a guard/reranker after semantic retrieval. It
    does not derive legal validity from a PDF or ingestion timestamp.
    """
    cfg=config or TemporalConfig()
    values=temporal_compatibility(context,valid_from=candidate.get('valid_from'),valid_to=candidate.get('valid_to'),available_from=candidate.get('available_from'),eligible=candidate.get('verified',False),config=cfg)
    semantic=candidate.get('semantic_score',0.0)
    return values,candidate_time_score(semantic,values,config=cfg)


def _query_time(query, query_time=None):
    return extract_event_date(query,query_time=query_time)

def _query_event_date(query, query_time=None):
    return _query_time(query,query_time).value

class RankingExecutor:
    def __init__(self):
        self.gate = asyncio.Semaphore(1)
        self.pending = set()

    async def run(self, operation, timeout):
        started = time.monotonic()
        await asyncio.wait_for(self.gate.acquire(), timeout=timeout)
        task = asyncio.create_task(asyncio.to_thread(operation))
        self.pending.add(task)
        def finished(completed):
            self.pending.discard(completed)
            self.gate.release()
            if not completed.cancelled():
                completed.exception()
        task.add_done_callback(finished)
        remaining = max(0.001, timeout - (time.monotonic() - started))
        return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)

class RetrievalService:
    def __init__(self,catalog, dense=None, reranker=None, tokenizer=None, profile='general', rerank_timeout=120, ranking_executor=None, temporal_config=None):
        if profile not in ('general','legal'): raise ValueError('invalid profile')
        if not isinstance(rerank_timeout,(int,float)) or isinstance(rerank_timeout,bool) or not math.isfinite(rerank_timeout) or not 0<rerank_timeout<=600: raise ValueError('rerank timeout must be in (0,600] seconds')
        self.catalog,self.dense,self.reranker,self.tokenizer,self.profile=catalog,dense,reranker,tokenizer,profile
        self.rerank_timeout=rerank_timeout
        self.ranking_executor=ranking_executor or RankingExecutor()
        self.temporal_config=temporal_config or TemporalConfig()
        self.last_version_selection=None
    async def retrieve(self, query, revision=None, event_date=None, max_units=8, max_evidence_tokens=6000, document_id=None, return_retrieval_trace=False, query_time=None):
        start=time.monotonic()
        revision_choice=None
        self.last_version_selection=None
        if not revision and self.profile=='legal':
            records=self.catalog.verified_temporal_versions()
            if records:
                revision_choice=select_version(records,event_date=event_date,query_time=query_time)
                revision=revision_choice.get('selected')
                self.last_version_selection=revision_choice
        rev=revision or self.catalog.active()
        timings={};dense_rows=[]
        if not rev or not self.catalog.revision(rev):
            if self.profile=='legal' and revision_choice and revision_choice.get('selected') is None:
                return {'status':'insufficient_evidence','query_plan':{'mode':'no_temporally_valid_version','profile':'legal','event_date':str(event_date or _query_event_date(query,query_time)),'query_time':str(query_time or date.today()),'revision_selection':revision_choice,'applicability_review':bool(event_date or _query_event_date(query,query_time))},'corpus_revision':None,'degraded':True,'degraded_reasons':['no_temporally_valid_version'],'missing_facts':['有可核验生效日期的法律版本'],'evidence':[],'budget':{'state':'not_applicable','used_tokens':0,'omitted':[]},'token_budget_status':'not_applicable','retrieval_ms':round((time.monotonic()-start)*1000,2)}
            raise LookupError('revision unavailable')
        manifest=__import__('json').loads(self.catalog.revision(rev)['manifest'])
        temporal_manifest=self.catalog.temporal_version(rev) or {}
        corpus_profile=manifest.get('profile','legal')
        if corpus_profile!=self.profile: raise LookupError('revision profile mismatch')
        if self.profile=='general' and manifest.get('parser_version')!='generic-documents-v2': raise LookupError('generic revision requires reindex')
        lexical_started=time.monotonic()
        lexical=self.catalog.lexical_scored(rev,query); raw_scores={'bm25':dict(lexical),'dense':{}}
        timings['lexical_ms']=round((time.monotonic()-lexical_started)*1000,2)
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
            dense_started=time.monotonic()
            try:
                if self.profile=='general' and manifest.get('embedding_serialization')!='title-newline-text-v1': raise ValueError('generic embedding serialization mismatch')
                expected_scope=('viking://resources/rag/'+manifest['corpus_id']+'/'+rev) if self.profile=='general' else ('viking://resources/law-cn/'+rev)
                if self.dense.scope.rstrip('/')!=expected_scope: raise ValueError('dense scope revision mismatch')
                if not self.catalog.db.execute('SELECT 1 FROM units WHERE revision=? AND uri IS NOT NULL LIMIT 1',(rev,)).fetchone(): raise ValueError('dense index not mapped')
                if not self.catalog.db.execute('SELECT 1 FROM settings WHERE key=? AND value=?',('dense_validated:'+rev,'yes')).fetchone(): raise ValueError('dense index not validated')
                rows=await self.dense.find(query); ids=[]
                dense_rows=[{'uri':x['uri'],'score':x.get('score') if isinstance(x.get('score'),numbers.Real) and math.isfinite(x['score']) else None} for x in rows]
                for x in rows:
                    r=self.catalog.db.execute('SELECT unit_id FROM units WHERE revision=? AND uri=?',(rev,x['uri'])).fetchone()
                    if r:
                        ids.append(r[0])
                        score=x.get('score')
                        if isinstance(score,numbers.Real) and math.isfinite(score): raw_scores['dense'].setdefault(r[0],score)
                lists['dense']=ids
            except Exception: degraded.append('dense_unavailable')
            timings['dense_ms']=round((time.monotonic()-dense_started)*1000,2)
        else: degraded.append('dense_not_configured')
        dense_pairs=[(uid,raw_scores['dense'].get(uid)) for uid in lists.get('dense',[])]
        ranking_started=time.monotonic()
        prepared=rank_candidates(self.catalog,rev,query,lexical,dense_pairs,lists['exact'])
        try:
            operation=lambda:rerank_candidates(prepared,query,lists['exact'],self.reranker)
            ranking=await self.ranking_executor.run(operation,self.rerank_timeout)
            if not self.reranker:degraded.append('reranker_not_configured')
        except Exception:
            degraded.append('reranker_failed')
            ranking=prepared
        timings['rank_and_rerank_ms']=round((time.monotonic()-ranking_started)*1000,2)
        candidates=ranking['candidates'];sources=ranking['sources'];rerank_scores=ranking['reranker_scores']
        temporal_decisions={}
        if self.profile=='legal' and manifest.get('verification_state') in ('pending','approved') and manifest.get('source_kind')!='synthetic':
            explicit_event=event_date or _query_event_date(query,query_time)
            context=TimeContext(query_time or date.today(), explicit_event)
            ranked=[]
            for index,unit in enumerate(candidates):
                decision=temporal_compatibility(context,valid_from=temporal_manifest.get('valid_from') or manifest.get('effective_from'),valid_to=temporal_manifest.get('valid_to') or manifest.get('effective_to'),available_from=temporal_manifest.get('published_at') or manifest.get('published_at') or manifest.get('fetched_at'),eligible=True,config=self.temporal_config)
                if (temporal_manifest.get('verification_state') or manifest.get('verification_state'))!='approved':
                    decision={**decision,'eligible':False,'validity':'unknown'}
                temporal_decisions[unit['unit_id']]=decision
                if decision['eligible']:
                    # Normalize the textual rank across the current candidate
                    # pool, then combine validity/time. This is the online
                    # TimelyRAG-style second stage; gold labels are not used.
                    semantic=1.0-(index/max(1,len(candidates)-1))
                    combo=candidate_time_score(semantic,decision,config=self.temporal_config)
                    ranked.append((combo['final'],-index,unit))
            ranked=[unit for _,__,unit in sorted(ranked,key=lambda row:(row[0],row[1]),reverse=True)]
            candidates=ranked
            sources={uid:sources[uid] for uid in (x['unit_id'] for x in candidates) if uid in sources}
            if not candidates:
                degraded.append('temporal_validity_unknown')
        timings['temporal_filter_ms']=round((time.monotonic()-ranking_started)*1000,2) - timings['rank_and_rerank_ms']
        if self.profile=='legal' and not candidates and (event_date or _query_event_date(query,query_time) or 'temporal_validity_unknown' in degraded):
            result={'status':'insufficient_evidence','query_plan':{'mode':'temporal_validity_unknown' if 'temporal_validity_unknown' in degraded else 'temporal_filtered','profile':'legal','event_date':str(event_date or _query_event_date(query,query_time)),'query_time':str(query_time or date.today()),'applicability_review':bool(event_date or _query_event_date(query,query_time))},'corpus_revision':rev,'degraded':True,'degraded_reasons':degraded,'missing_facts':['有可核验生效日期的条文'],'evidence':[],'budget':{'state':'not_applicable','used_tokens':0,'omitted':[]},'token_budget_status':'not_applicable','retrieval_ms':round((time.monotonic()-start)*1000,2)}
            if return_retrieval_trace:result['retrieval_trace']={'temporal':temporal_decisions,'dense_backend':'openviking' if isinstance(self.dense,OpenViking) else type(self.dense).__name__,'timings_ms':timings}
            return result
        byid={c['unit_id']:c for c in candidates};selected=[c['unit_id'] for c in candidates[:max_units]]
        evidence=[]
        if self.profile=='legal':
            source_lookup=bool(lists['exact']) and wants_source
            general=bool(re.search(r'是什么|一般规定|一般如何|介绍|概念',query)) and not re.search(r'我的|我能|本人|案件|纠纷|起诉|被告',query)
            mode='source_lookup' if source_lookup else ('general_explanation' if general else 'case_question')
            applicability_review=mode=='case_question' and bool(event_date)
            effective_event=event_date or _query_event_date(query,query_time)
            missing_facts=(['事件发生日期及具体事实'] if mode=='case_question' and not effective_event else (['法律适用及过渡规定需核验'] if applicability_review else []))
        else:
            source_lookup=bool(lists['exact']) and wants_source
            mode='source_lookup' if source_lookup else 'general_retrieval'
            applicability_review=False;missing_facts=[]
        if source_lookup: selected=[uid for uid in selected if uid in lists['exact']]
        for i,uid in enumerate(selected,1):
            c=byid[uid];doc=c.get('document') or {}
            item={'citation_id':f'E{i}','unit_id':uid,'structure':c.get('structure') or [],'document_id':doc.get('doc_id'),'title':c['title'],'text':c['text'],'source_url':doc.get('source_url') or manifest.get('source_url'),'source_kind':doc.get('source_kind') or manifest.get('source_kind'),'source_locator':c['locator'],'content_sha256':c['sha256'],'retrieval_sources':sources[uid],'reranker_score':rerank_scores.get(uid),'verification_state':manifest.get('verification_state'),'profile':self.profile}
            if self.profile=='legal':item.update(article_label=c['label'],candidate_relations=c['relations'],effective_from=temporal_manifest.get('valid_from') or manifest.get('effective_from'),effective_to=temporal_manifest.get('valid_to') or manifest.get('effective_to'),applicability_review=applicability_review,temporal=temporal_decisions.get(uid))
            elif not source_lookup and c['parts']:
                winner=c.get('rerank_part_no')
                part=next((p for p in c['parts'] if p['part_no']==winner),None) if winner is not None else None
                if part is None:
                    part=self.catalog.best_part(rev,uid,query)
                    if rerank_scores.get(uid) is not None and len(c['parts'])>1:
                        # A custom document-only scorer cannot lend its score to
                        # a different lexical paragraph. Inspection may continue.
                        item['reranker_score']=None
                        degraded.append('reranker_part_unavailable')
                if part:item.update(text=part['text'],source_locator=part['locator'],content_sha256=part['sha256'],part_no=part['part_no'],document_sha256=c['sha256'],has_more=len(c['parts'])>1)
            evidence.append(item)
        from .budget import assemble
        budgeted,budget=assemble(evidence,self.tokenizer,max_evidence_tokens,query)
        # Retrieval results can be inspected without tokenizer, but are not generation-ready.
        if self.tokenizer: evidence=budgeted
        result={'query_plan':{'mode':mode,'profile':self.profile,'event_date':str(event_date or _query_event_date(query,query_time)) if self.profile=='legal' else None,'query_time':str(query_time or date.today()) if self.profile=='legal' else None,'revision_selection':revision_choice,'applicability_review':applicability_review},'budget':budget, 'status':('insufficient_evidence' if not evidence else ('needs_clarification' if mode=='case_question' and not (event_date or _query_event_date(query,query_time)) else 'ready')),'corpus_revision':rev,'degraded':bool(degraded),'degraded_reasons':degraded,'missing_facts':missing_facts,'token_budget_status':budget['state'],'evidence':evidence,'retrieval_ms':round((time.monotonic()-start)*1000,2)}
        if return_retrieval_trace:
            result['retrieval_trace']={'temporal':{uid:value for uid,value in temporal_decisions.items()},'revision_selection':revision_choice,'dense_backend':'openviking' if isinstance(self.dense,OpenViking) else type(self.dense).__name__,'dense_scope':getattr(self.dense,'scope',None),'dense_hits':dense_rows,'timings_ms':timings,'rerank_timeout_seconds':self.rerank_timeout,'candidates':[{'unit_id':item['unit_id'],'document_id':(item.get('document') or {}).get('doc_id'),'rrf_score':ranking['rrf_scores'].get(item['unit_id']),'reranker_score':rerank_scores.get(item['unit_id']),'sources':sources[item['unit_id']]} for item in candidates]}
        return result
