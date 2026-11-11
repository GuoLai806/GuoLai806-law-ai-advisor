"""Isolated v2 API; bearer identities are configured by the operator, never request bodies."""
import asyncio, hmac, json, os, re, time, uuid
import httpx
from datetime import date
from typing import Literal
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .catalog import Catalog
from .retrieval import RetrievalService, CrossEncoderReranker, OpenViking
from .temporal import TemporalConfig
from .budget import HuggingFaceTokenizer, ConservativeByteBudget
from .answer import OpenAIStream, ModelStreamIncomplete, validate_citations, provider_events, messages_for
from .grounding import GroundingRejected
from .exploration import ExploreBundle, explore_bundle

def setting(name, default=None):
    return os.environ.get('EVIDENCE_RAG_'+name,os.environ.get('LEGAL_RAG_'+name,default))

def default_evidence_items():
    value=int(setting('DEFAULT_EVIDENCE_ITEMS','8'))
    if not 1<=value<=60:
        raise ValueError('DEFAULT_EVIDENCE_ITEMS must be between 1 and 60')
    return value

class Query(BaseModel):
    query: str = Field(min_length=1,max_length=4000)
    session_id: str = Field(min_length=1,max_length=128,pattern=r'^[A-Za-z0-9_-]+$')
    event_date: date | None = None
    query_time: date | None = None
    profile: Literal['general','legal'] | None = None
    document_id: str | None = Field(default=None,max_length=512)
    budget_mode: Literal['tokenizer','conservative_bytes'] = 'tokenizer'
    corpus_revision: str | None = None
    max_evidence_tokens: int = Field(default=6000,ge=1,le=6000)
    max_evidence_items: int = Field(default_factory=default_evidence_items,ge=1,le=60)
    include_retrieval_trace: bool = False

@asynccontextmanager
async def lifespan(app):
    current_app=app
    default_evidence_items()  # Reject invalid operator configuration at startup.
    profile='legal' if getattr(app.state,'default_profile_explicit',False) else setting('DEFAULT_PROFILE','general')
    app.state.default_profile=profile
    demo_user=setting('LOCAL_DEMO_USER')
    if demo_user:
        try: accounts=json.loads(setting('AUTH','{}'))
        except ValueError: accounts={}
        if not isinstance(accounts,dict) or demo_user not in accounts.values() or demo_user in setting('ADMINS','').split(','):
            raise ValueError('local demo must use an existing non-administrator identity')
    answer_format=setting('ANSWER_FORMAT','text')
    generation_gate=setting('GENERATION_GATE','reranker_threshold')
    if answer_format not in ('text','grounded_json') or generation_gate not in ('reranker_threshold','grounded_evidence'):
        raise ValueError('invalid answer format or generation gate')
    if generation_gate=='grounded_evidence' and answer_format!='grounded_json':
        raise ValueError('grounded evidence gate requires grounded JSON answers')
    # Budget measurement and the provider must see the same optional prompt format.
    os.environ['EVIDENCE_RAG_ANSWER_FORMAT']=answer_format
    db=Catalog(setting('DB','evidence-rag.sqlite3'))
    db.db.execute('CREATE TABLE IF NOT EXISTS sessions(session_id TEXT PRIMARY KEY, owner TEXT NOT NULL)')
    db.db.execute('CREATE TABLE IF NOT EXISTS turns(turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, query TEXT NOT NULL, answer TEXT NOT NULL, revision TEXT NOT NULL, created REAL NOT NULL, FOREIGN KEY(session_id) REFERENCES sessions(session_id))')
    app.state.catalog=db
    reranker=CrossEncoderReranker(setting('RERANKER'),device=setting('RERANK_DEVICE'),dtype=setting('RERANK_DTYPE'),batch_size=int(setting('RERANK_BATCH_SIZE','2'))) if setting('RERANKER') else None
    tokenizer=HuggingFaceTokenizer(setting('TOKENIZER'),setting('TOKENIZER_REVISION')) if setting('TOKENIZER') and setting('TOKENIZER_REVISION') else None
    ov_key=os.environ.get('OPENVIKING_API_KEY')
    ov_headers={'X-API-Key':ov_key} if ov_key else {}
    dense=OpenViking(setting('OV_URL'),setting('OV_SCOPE'),headers=ov_headers,timeout=float(setting('OV_TIMEOUT','30')),reuse_client=setting('OV_REUSE_CLIENT','1')=='1') if setting('OV_URL') and setting('OV_SCOPE') else None
    temporal_config=TemporalConfig(retention_days=int(setting('TEMPORAL_RETENTION_DAYS','0')),decay_days=int(setting('TEMPORAL_DECAY_DAYS','3650')),decay_gamma=float(setting('TEMPORAL_DECAY_GAMMA','2.0')))
    current_app.state.default_profile=profile
    current_app.state.service=RetrievalService(db,dense,reranker,tokenizer,profile=profile,rerank_timeout=float(setting('RERANK_TIMEOUT','120')),temporal_config=temporal_config)
    app.state.generator=OpenAIStream(setting('MODEL_URL'),setting('MODEL_NAME'),setting('MODEL_KEY',''),timeout=float(setting('MODEL_TIMEOUT','90')),include_usage=setting('STREAM_INCLUDE_USAGE')=='1',max_output_tokens=int(setting('MAX_OUTPUT_TOKENS')) if setting('MAX_OUTPUT_TOKENS') else None,temperature=float(setting('MODEL_TEMPERATURE')) if setting('MODEL_TEMPERATURE') is not None else None,thinking=setting('MODEL_THINKING'),reasoning_effort=setting('MODEL_REASONING_EFFORT'),empty_retry_thinking=setting('MODEL_EMPTY_RETRY_THINKING'),retry_timeout=float(setting('MODEL_RETRY_TIMEOUT')) if setting('MODEL_RETRY_TIMEOUT') else None,retry_max_output_tokens=int(setting('MODEL_RETRY_MAX_OUTPUT_TOKENS')) if setting('MODEL_RETRY_MAX_OUTPUT_TOKENS') else None) if setting('MODEL_URL') and setting('MODEL_NAME') else None
    if current_app.state.generator:
        current_app.state.generator.answer_format=answer_format
    try:
        yield
    finally:
        try:
            if dense:
                await dense.aclose()
        finally:
            db.db.close()

app=FastAPI(title='Evidence RAG v2',lifespan=lifespan)
app.state.default_profile=setting('DEFAULT_PROFILE','general')
legal_app=FastAPI(title="Legal RAG compatibility",lifespan=lifespan)
legal_app.state.default_profile='legal'
legal_app.state.default_profile_explicit=True
app.state.default_profile_explicit=False

@app.get('/',include_in_schema=False)
def workspace():
    from pathlib import Path
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).parent/'static'/'index.html')

@app.get('/assets/workspace.mjs',include_in_schema=False)
def workspace_script():
    from pathlib import Path
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).parent/'static'/'workspace.mjs',media_type='text/javascript')

def local_demo_request(request):
    """Explicit local UI access; never trust forwarded client or origin headers."""
    if not request.client or request.client.host not in ('127.0.0.1','::1'):
        return False
    if request.url.hostname not in ('127.0.0.1','localhost','::1'):
        return False
    origin=request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        return False
    return request.headers.get('sec-fetch-site','same-origin') in ('same-origin','none')

def identity(request: Request,authorization: str | None=Header(default=None)):
    # Map opaque credentials to principals, populated from local environment by operator.
    try: accounts=json.loads(setting('AUTH','{}'))
    except ValueError: accounts={}
    if not isinstance(accounts,dict): accounts={}
    token=authorization[7:] if authorization and authorization.startswith('Bearer ') else ''
    if not token:
        demo_user=setting('LOCAL_DEMO_USER')
        if authorization is None and demo_user and isinstance(accounts,dict) and demo_user in accounts.values() and demo_user not in setting('ADMINS','').split(',') and local_demo_request(request):
            return demo_user
        raise HTTPException(401,'authentication required')
    for secret, principal in accounts.items():
        if isinstance(secret,str) and hmac.compare_digest(secret.encode('utf-8'),token.encode('utf-8')) and isinstance(principal,str) and principal: return principal
    raise HTTPException(401,'authentication required')

def bind(db, session, user):
    db.execute('INSERT OR IGNORE INTO sessions VALUES(?,?)',(session,user))
    row=db.execute('SELECT owner FROM sessions WHERE session_id=?',(session,)).fetchone()
    if row['owner']!=user: raise HTTPException(403,'session unavailable')

def byte_budget_config():
    if setting('ALLOW_CONSERVATIVE_BYTES')!='1': raise ValueError('conservative byte budgeting not enabled')
    try:
        return ConservativeByteBudget(int(setting('MODEL_CONTEXT_WINDOW')),int(setting('FRAMING_RESERVE')),int(setting('OUTPUT_RESERVE')))
    except (ValueError,TypeError) as exc: raise ValueError('explicit context window and reserves required') from exc

def generation_limits():
    """Explicit provider envelope; a measured prompt alone is insufficient."""
    try:
        window=int(setting('MODEL_CONTEXT_WINDOW'));framing=int(setting('FRAMING_RESERVE'));output=int(setting('OUTPUT_RESERVE'))
    except (ValueError,TypeError) as exc: raise ValueError('generation context window and reserves required') from exc
    if min(window,framing,output)<=0 or framing+output>=window: raise ValueError('invalid generation envelope')
    return window,framing,output

async def get_bundle(body, request, user, pinned_revision=None, *, for_exploration=False):
    cat=Catalog(request.app.state.catalog.path)
    try:
        bind(cat.db,body.session_id,user)
        requested_revision=pinned_revision
        initial_profile=body.profile or getattr(request.app.state,'default_profile','legal')
        active=requested_revision or cat.active()
        # A legal request with verified temporal records may select a retained
        # revision inside RetrievalService. Do not reject it for not being the
        # current active revision before selection runs.
        temporal_versions=[
            item for item in cat.verified_temporal_versions()
            if item.get('valid_from') or item.get('valid_to')
        ]
        require_temporal_selection=initial_profile=='legal' and not requested_revision and bool(temporal_versions)
        if not active and not require_temporal_selection: raise HTTPException(503,'no active verified corpus')
        if body.corpus_revision and body.corpus_revision!=active: raise HTTPException(404,'revision unavailable')
        if active and not cat.revision(active): raise HTTPException(404,'revision unavailable')
        configured=request.app.state.service
        profile=body.profile or getattr(request.app.state,'default_profile','legal')
        tokenizer=configured.tokenizer
        # A request without a pin lets RetrievalService select the verified
        # interval itself and retain its full revision-selection trace. A pin
        # is the only case where the API supplies a fixed revision.
        # Verify the requested interval before allocating the service or
        # touching generation resources. The service repeats this check and
        # retains the full revision-selection trace for a normal request.
        if require_temporal_selection:
            from .legal_versions import select_version
            selection=select_version(
                temporal_versions,
                event_date=body.event_date,
                query_time=body.query_time,
            )
            if not selection.get('selected'):
                return {
                    'status':'insufficient_evidence',
                    'query_plan':{
                        'mode':'no_temporally_valid_version',
                        'profile':'legal',
                        'event_date':str(body.event_date) if body.event_date else None,
                        'query_time':str(body.query_time or date.today()),
                        'revision_selection':selection,
                        'applicability_review':bool(body.event_date),
                    },
                    'corpus_revision':None,
                    'degraded':True,
                    'degraded_reasons':['no_temporally_valid_version'],
                    'missing_facts':['有可核验生效日期的法律版本'],
                    'evidence':[],
                    'budget':{'state':'not_applicable','used_tokens':0,'omitted':[]},
                    'token_budget_status':'not_applicable',
                    'retrieval_ms':0,
                }
        selected_revision=requested_revision or (
            None if require_temporal_selection else active
        )
        if body.budget_mode=='conservative_bytes':
            if profile!='general': raise HTTPException(422,'conservative byte mode is limited to general profile')
            try:
                tokenizer=byte_budget_config()
                tokenizer.paragraph_loader=lambda uid:cat.get(active,uid)
            except ValueError as exc: raise HTTPException(503,str(exc)) from exc
        if for_exploration:
            # Exploration never measures or supplies a generation prompt. Keep the
            # complete selected paragraphs, including those too large to generate.
            tokenizer=None
        service=RetrievalService(cat,configured.dense,configured.reranker,tokenizer,profile=profile,rerank_timeout=configured.rerank_timeout,ranking_executor=configured.ranking_executor,temporal_config=getattr(configured,'temporal_config',None))
        maximum=body.max_evidence_tokens
        try:
            window,framing,output=generation_limits()
            maximum=min(maximum,window-framing-output)
        except ValueError: pass
        try:
            result=await service.retrieve(body.query,selected_revision,body.event_date,max_evidence_tokens=maximum,max_units=body.max_evidence_items,document_id=body.document_id,return_retrieval_trace=body.include_retrieval_trace,query_time=body.query_time)
        except LookupError as exc: raise HTTPException(404,str(exc)) from exc
        except ValueError as exc: raise HTTPException(503,'catalog or retrieval integrity unavailable') from exc
        # The service may select another verified legal revision after the
        # API's initial check; its returned corpus_revision is authoritative.
        if pinned_revision and result.get('corpus_revision') not in (None,pinned_revision):
            raise HTTPException(409,'temporal revision changed during a pinned request')
        result['trace_id']='tr_'+uuid.uuid4().hex
        return result
    finally: cat.db.close()

@app.get('/health/live')
def live(): return {'status':'live'}
@app.get('/health/ready')
async def ready(request: Request):
    import httpx
    from fastapi.responses import JSONResponse
    cat=Catalog(request.app.state.catalog.path); missing=[]
    try:
        rev=cat.active(); service=request.app.state.service; generator=request.app.state.generator
        if not rev: missing.append('active_revision')
        else:
            n=cat.db.execute('SELECT count(*) FROM units WHERE revision=?',(rev,)).fetchone()[0]
            f=cat.db.execute('SELECT count(*) FROM unit_fts WHERE revision=?',(rev,)).fetchone()[0]
            if not n or n!=f: missing.append('fts_consistency')
            if not cat.db.execute('SELECT 1 FROM settings WHERE key=? AND value=?',('dense_validated:'+rev,'yes')).fetchone(): missing.append('dense_validation')
        meta=json.loads(cat.revision(rev)['manifest']) if rev and cat.revision(rev) else {}
        if meta.get('profile')=='general' and (meta.get('embedding_serialization')!='title-newline-text-v1' or meta.get('parser_version')!='generic-documents-v2'): missing.append('embedding_identity')
        expected_scope=(('viking://resources/rag/'+meta['corpus_id']+'/'+rev) if meta.get('profile')=='general' else ('viking://resources/law-cn/'+rev)) if rev else None
        if not service.dense or not rev or service.dense.scope.rstrip('/')!=expected_scope: missing.append('openviking')
        else:
            try:
                async with httpx.AsyncClient(timeout=3,headers=service.dense.headers,trust_env=False) as client:
                    response=await client.get(service.dense.base+'/health'); response.raise_for_status()
            except Exception: missing.append('openviking_health')
        if not service.reranker: missing.append('reranker')
        if not service.tokenizer: missing.append('tokenizer')
        try: generation_limits()
        except ValueError: missing.append('generation_budget_config')
        if not generator: missing.append('answer_model')
        elif not isinstance(generator.max_output_tokens,int): missing.append('generation_output_limit')
        if generator:
            try:
                async with httpx.AsyncClient(timeout=3,headers={'Authorization':'Bearer '+generator.api_key},trust_env=False) as client:
                    response=await client.get(generator.base_url+'/models'); response.raise_for_status(); data=response.json()
                    if not any(x.get('id')==generator.model for x in data.get('data',[])): missing.append('answer_model_identity')
            except Exception: missing.append('answer_model_health')
        try:
            byte_budget_config();byte_mode=getattr(request.app.state,'default_profile','general')=='general'
        except ValueError: byte_mode=False
        return JSONResponse(status_code=503 if missing else 200,content={'ready':not missing,'active_revision':rev,'missing':missing,'conservative_bytes_available':byte_mode,'budget_guarantee':'approximate only' if byte_mode else None})
    finally: cat.db.close()
@app.post('/api/v2/retrieve')
async def retrieve(body: Query,request: Request,user=Depends(identity)): return await get_bundle(body,request,user)
@app.post('/api/v2/explore',response_model=ExploreBundle)
async def explore(body: Query,request: Request,user=Depends(identity)):
    if setting('EXPLORE_ENABLED','1')!='1':
        raise HTTPException(404,'exploration unavailable')
    bundle=await get_bundle(body,request,user,for_exploration=True)
    try:
        return explore_bundle(bundle)
    except ValueError as exc:
        raise HTTPException(503,'catalog or preview integrity unavailable') from exc

@app.get('/api/v2/models')
def model_capabilities(request: Request,user=Depends(identity)):
    service=request.app.state.service
    generator=request.app.state.generator
    return {'dense_configured':bool(service.dense),'reranker_loaded':bool(service.reranker),'tokenizer_loaded':bool(service.tokenizer),'generator_configured':bool(generator),'provider_health_verified':False,
            'exploration_available':setting('EXPLORE_ENABLED','1')=='1',
            'runtime_identity':{'ov_reuse_client':getattr(service.dense,'reuse_client',None),'explore_enabled':setting('EXPLORE_ENABLED','1')=='1','dense_scope':getattr(service.dense,'scope',None),'reranker_revision':getattr(service.reranker,'revision',None),'reranker_batch_size':getattr(service.reranker,'batch_size',None),'rerank_timeout_seconds':service.rerank_timeout,'tokenizer_revision':getattr(service.tokenizer,'revision',None),'answer_model':getattr(generator,'model',None),'max_output_tokens':getattr(generator,'max_output_tokens',None),'thinking':getattr(generator,'thinking',None),'reasoning_effort':getattr(generator,'reasoning_effort',None),'default_evidence_items':default_evidence_items(),**generation_identity(request)}}

def generation_identity(request):
    """Public operational identity: hashes/configuration, never credentials or prompts."""
    import hashlib
    profile=getattr(request.app.state,'default_profile','general')
    system=messages_for('',[{'citation_id':'E1','title':'','text':'','profile':profile}],answer_format=getattr(request.app.state.generator,'answer_format','text'))[0]['content']
    try: window,framing,output=generation_limits()
    except ValueError: window=framing=output=None
    generator=request.app.state.generator
    return {'default_profile':profile,'system_prompt_sha256':hashlib.sha256(system.encode()).hexdigest(),
            'context_window':window,'framing_reserve':framing,'output_reserve':output,
            'provider_timeout_seconds':getattr(generator,'timeout',None),
            'temperature':getattr(generator,'temperature',None),
            'stream_include_usage':bool(getattr(generator,'include_usage',False)),
            'store_conversations':setting('STORE_CONVERSATIONS')=='1',
            'rerank_threshold':setting('RERANK_THRESHOLD'),
            'empty_retry_thinking':getattr(generator,'empty_retry_thinking',None),
            'retry_timeout_seconds':getattr(generator,'retry_timeout',None),
            'retry_max_output_tokens':getattr(generator,'retry_max_output_tokens',None),
            'answer_format':getattr(generator,'answer_format','text'),
            'generation_gate':setting('GENERATION_GATE','reranker_threshold')}

def generation_allowed(bundle,body,app):
    budget_state=bundle['budget']['state']
    allowed_budget=False
    if budget_state=='measured':
        try:
            window,framing,output=generation_limits();used=bundle['budget'].get('used_tokens');maximum=getattr(app.state.generator,'max_output_tokens',None)
            allowed_budget=isinstance(used,int) and not isinstance(used,bool) and used>=0 and used+framing+output<=window and isinstance(maximum,int) and not isinstance(maximum,bool) and 0<maximum<=output
        except ValueError: pass
    if budget_state=='conservative_bytes' and getattr(body,'budget_mode',None)=='conservative_bytes' and bundle['query_plan'].get('profile')=='general':
        try:
            config=byte_budget_config();maximum=getattr(app.state.generator,'max_output_tokens',None)
            allowed_budget=isinstance(maximum,int) and not isinstance(maximum,bool) and 0<maximum<=config.output_reserve
        except ValueError: allowed_budget=False
    if not (app.state.generator and bundle['status']=='ready' and not bundle['query_plan']['applicability_review'] and bundle['evidence'] and not bundle['degraded'] and allowed_budget and not bundle['budget']['omitted']): return False
    if bundle['query_plan']['mode']=='source_lookup': return True
    if bundle['query_plan'].get('profile')=='general' and setting('GENERATION_GATE','reranker_threshold')=='grounded_evidence':
        # Relevance logits do not establish entailment. The opt-in provider must
        # check its quoted evidence before yielding any deliverable answer text.
        return getattr(app.state.generator,'answer_format','text')=='grounded_json'
    # General legal explanation requires a separately calibrated reranker threshold.
    threshold=setting('RERANK_THRESHOLD')
    try:
        import math
        value=float(threshold); score=bundle['evidence'][0]['reranker_score']
        return math.isfinite(value) and score is not None and score>=value
    except (ValueError,TypeError): return False

def save_turn(request, user, session_id, query, answer, revision):
    if setting('STORE_CONVERSATIONS')!='1': return
    import time
    cat=Catalog(request.app.state.catalog.path)
    try:
        bind(cat.db,session_id,user)
        cat.db.execute('INSERT INTO turns VALUES(?,?,?,?,?,?)',('turn_'+uuid.uuid4().hex,session_id,query,answer,revision,time.time()))
    finally: cat.db.close()

@app.get('/api/v2/sessions/{session_id}/turns')
def session_turns(session_id: str,request: Request,user=Depends(identity)):
    if not re.fullmatch('[A-Za-z0-9_-]{1,128}',session_id): raise HTTPException(404,'session unavailable')
    cat=Catalog(request.app.state.catalog.path)
    try:
        owner=cat.db.execute('SELECT owner FROM sessions WHERE session_id=?',(session_id,)).fetchone()
        if not owner: raise HTTPException(404,'session unavailable')
        if owner['owner']!=user: raise HTTPException(403,'session unavailable')
        return {'session_id':session_id,'turns':[dict(x) for x in cat.db.execute('SELECT turn_id,query,answer,revision,created FROM turns WHERE session_id=? ORDER BY created,turn_id',(session_id,))]}
    finally: cat.db.close()

@app.post('/api/v2/chat')
async def chat(body: Query,request: Request,user=Depends(identity)):
    b=await get_bundle(body,request,user)
    usage_by_attempt={}; attempts=[]; retries=[]; models=[]; generated=False; grounding=None; phase='generation'
    def accounting():
        reported=[{'attempt':number,'usage':value} for number,value in sorted(usage_by_attempt.items())]
        usage=({key:sum(item['usage'].get(key,0) for item in reported)
                for key in set().union(*(item['usage'] for item in reported))} if reported else None)
        expected={item['attempt'] for item in attempts} if attempts else {1}
        complete=bool(usage) and set(usage_by_attempt)==expected
        source=('not_generated' if not generated else 'provider' if complete else
                'provider_known_attempts' if usage else 'unavailable')
        return {'usage':usage,'usage_attempts':reported,'usage_complete':complete,'usage_source':source,
                'generation_attempts':attempts,'generation_retries':retries,
                'observed_answer_models':models,'grounding_validation':grounding}
    if generation_allowed(b,body,request.app):
        generated=True
        try:
            parts=[]
            async for event in provider_events(request.app.state.generator,body.query,b['evidence']):
                if event.get('type')=='delta': parts.append(event['delta'])
                elif event.get('type')=='usage': usage_by_attempt[event.get('attempt',1)]=event['usage']
                elif event.get('type')=='generation_attempt': attempts.append({key:value for key,value in event.items() if key!='type'})
                elif event.get('type')=='generation_retry': retries.append({key:value for key,value in event.items() if key!='type'})
                elif event.get('type')=='model_identity': models.append({key:value for key,value in event.items() if key!='type'})
                elif event.get('type')=='grounding_validation': grounding={key:value for key,value in event.items() if key!='type'}
            answer=''.join(parts); verification=validate_citations(answer,b['evidence'])
            if verification['status']=='invalid':
                return {'trace_id':b['trace_id'],'status':'insufficient_evidence','answer':'生成结果的引用未通过检查，请查看原文并进一步核验。','claims':[],'citations':b['evidence'],'verification':verification,**accounting(),'budget':b['budget'],'degraded':True}
            phase='session_save'
            save_turn(request,user,body.session_id,body.query,answer,b['corpus_revision'])
            return {'trace_id':b['trace_id'],'status':b['status'],'answer':answer,'claims':[],'citations':b['evidence'],'verification':verification,**accounting(),'budget':b['budget'],'degraded':False}
        except asyncio.CancelledError: raise
        except Exception as exc:
            fallback='本次未取得经过引用检查的完整回答，请查看资料或重新提问。'
            return {'trace_id':b['trace_id'],'status':'evidence_only','answer':fallback,'claims':[],
                    'citations':b['evidence'],'verification':{'status':'not_generated'},**accounting(),
                    'budget':b['budget'],'degraded':True,'failure':safe_pipeline_error(exc,phase)}
    fallback=('目前只能展示检索到的资料，不能据此生成法律结论；请补充事实并由专业人士核验。' if b['query_plan'].get('profile','legal')=='legal' else '目前只能展示检索到的资料，尚不能生成经过核验的回答。')
    save_turn(request,user,body.session_id,body.query,fallback,b['corpus_revision'])
    return {'trace_id':b['trace_id'],'status':'evidence_only','answer':fallback,'claims':[],'citations':b['evidence'],'verification':{'status':'not_generated'},**accounting(),'budget':b['budget'],'degraded':True}
@app.get('/api/v2/units/{unit_id}')
def unit(unit_id: str,request: Request,revision: str | None=None,user=Depends(identity)):
    if not re.fullmatch('[0-9a-f]{32}',unit_id): raise HTTPException(404,'unit unavailable')
    cat=Catalog(request.app.state.catalog.path)
    try:
        rev=revision or cat.active(); meta=cat.revision(rev) if rev else None
        if not meta or meta['state'] not in ('active','retained') or json.loads(meta['manifest']).get('verification_state') not in ('approved','provided'): raise HTTPException(404,'unit unavailable')
        found=cat.get(rev,unit_id)
        if not found: raise HTTPException(404,'unit unavailable')
        return found
    finally: cat.db.close()

def check_access(body,request,user):
    cat=Catalog(request.app.state.catalog.path)
    try:
        bind(cat.db,body.session_id,user)
        active=cat.active()
        if not active: raise HTTPException(503,'no active verified corpus')
        if body.corpus_revision and body.corpus_revision!=active: raise HTTPException(404,'revision unavailable')
        return active
    finally: cat.db.close()

@app.post('/api/v2/chat/stream')
async def stream(body: Query,request: Request,user=Depends(identity)):
    pinned=check_access(body,request,user)
    # No synthetic typewriter effect: the service reports lack of an actual generator.
    trace='tr_'+uuid.uuid4().hex
    started=time.monotonic()
    async def events():
        seq=0
        phase='retrieval'
        def event(kind,payload):
            nonlocal seq
            seq+=1
            return 'event: '+kind+'\ndata: '+json.dumps({'trace_id':trace,'seq':seq,**payload},ensure_ascii=False)+'\n\n'
        yield event('retrieval_started',{})
        try:
            bundle=await get_bundle(body,request,user,pinned_revision=pinned); bundle['trace_id']=trace
            yield event('evidence_ready',{'bundle':bundle})
            if generation_allowed(bundle,body,request.app):
                phase='generation'
                parts=[]; first_token_ms=None;usage=None;usage_attempts=[];usage_by_attempt={};attempts=[]
                async for incoming in provider_events(request.app.state.generator,body.query,bundle['evidence']):
                    if await request.is_disconnected(): return
                    if incoming.get('type') in ('generation_attempt','generation_retry'):
                        if incoming['type']=='generation_attempt':attempts.append({key:value for key,value in incoming.items() if key!='type'})
                        yield event(incoming['type'],{**{key:value for key,value in incoming.items() if key!='type'},'elapsed_ms':round((time.monotonic()-started)*1000,2)});continue
                    if incoming.get('type')=='model_identity':
                        yield event('model_identity',{'model':incoming['model'],'system_fingerprint':incoming.get('system_fingerprint'),'attempt':incoming.get('attempt',1)});continue
                    if incoming.get('type')=='grounding_validation':
                        yield event('grounding_validation',{key:value for key,value in incoming.items() if key!='type'});continue
                    if incoming.get('type')=='usage':
                        usage_by_attempt[incoming.get('attempt',1)]=incoming['usage']
                        usage_attempts=[{'attempt':number,'usage':value} for number,value in sorted(usage_by_attempt.items())]
                        usage={key:sum(item['usage'].get(key,0) for item in usage_attempts) for key in set().union(*(item['usage'] for item in usage_attempts))}
                        yield event('usage',{'usage':incoming['usage'],'attempt':incoming.get('attempt',1),'usage_source':'provider'});continue
                    if incoming.get('type')!='delta': continue
                    delta=incoming['delta']
                    if first_token_ms is None: first_token_ms=round((time.monotonic()-started)*1000,2)
                    parts.append(delta); yield event('answer_delta',{'delta':delta,'verification':'pending'})
                answer=''.join(parts); verification=validate_citations(answer,bundle['evidence'])
                yield event('verification',verification)
                if verification['status']=='invalid':
                    yield event('answer_rejected',{'discard_pending_answer':True,'reason':'invalid_citations'})
                    status='insufficient_evidence'
                else:
                    phase='session_save'
                    save_turn(request,user,body.session_id,body.query,answer,bundle['corpus_revision'])
                    status=bundle['status']
                usage_complete=bool(usage) and {item['attempt'] for item in usage_attempts}==({item['attempt'] for item in attempts} if attempts else {1})
                yield event('completed',{'status':status,'first_answer_ms':first_token_ms,'total_ms':round((time.monotonic()-started)*1000,2),'usage':usage,'usage_source':'provider' if usage_complete else 'provider_known_attempts' if usage else 'unavailable','usage_complete':usage_complete,'usage_attempts':usage_attempts,'generation_attempts':attempts})
            else:
                yield event('verification',{'status':'not_generated'})
                yield event('completed',{'status':'evidence_only'})
        except HTTPException as e: yield event('error',{'status':e.status_code,'message':e.detail})
        except ModelStreamIncomplete as e: yield event('error',{'code':'model_incomplete','reason':e.reason,'discard_pending_answer':True})
        except asyncio.CancelledError: raise
        except Exception as exc: yield event('error',{'message':'provider or retrieval unavailable','discard_pending_answer':True,**safe_pipeline_error(exc,phase)})
    return StreamingResponse(events(),media_type='text/event-stream')

def safe_pipeline_error(exc,phase):
    """Preserve failure provenance without upstream text, URLs or credentials."""
    result={'error_type':type(exc).__name__,'failure_phase':phase}
    if isinstance(exc,httpx.HTTPStatusError): result['http_status']=exc.response.status_code
    if isinstance(exc,(GroundingRejected,ModelStreamIncomplete)): result['reason']=exc.reason
    return result

class IngestRequest(BaseModel):
    filename: str = Field(min_length=1,max_length=200)
    profile: Literal['general','legal'] | None = None
    manifest: dict
    idempotency_key: str = Field(min_length=8,max_length=128,pattern=r'^[A-Za-z0-9_-]+$')

def admin(user):
    if user not in setting('ADMINS','').split(','): raise HTTPException(403,'administrator required')

@app.post('/api/v2/ingest-jobs',status_code=202)
def ingest_job(body: IngestRequest,request: Request,user=Depends(identity)):
    import hashlib
    from pathlib import Path
    import pymupdf
    from .catalog import PARSER_VERSION
    admin(user)
    root=setting('INGEST_DIR')
    if not root: raise HTTPException(503,'ingest directory not configured')
    if '\x00' in body.filename or Path(body.filename).name!=body.filename or Path(body.filename).suffix.lower() not in ('.pdf','.jsonl','.txt','.md'): raise HTTPException(422,'invalid source filename')
    root=Path(root).resolve(); path=(root/body.filename).resolve()
    if path.parent!=root or not path.is_file(): raise HTTPException(404,'source unavailable')
    import os,stat
    limit=50_000_000
    if path.stat().st_size>limit: raise HTTPException(413,'source too large')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as source:
        meta=os.fstat(source.fileno())
        if not stat.S_ISREG(meta.st_mode): raise HTTPException(422,'source must be a regular file')
        if meta.st_size>limit: raise HTTPException(413,'source too large')
        raw=source.read(limit+1)
    if len(raw)>50_000_000: raise HTTPException(413,'source too large')
    profile=body.profile or getattr(request.app.state,'default_profile','general')
    manifest={**body.manifest,'source_sha256':hashlib.sha256(raw).hexdigest(),'verification_state':'pending' if profile=='legal' else 'provided'}
    try:
        if profile=='legal':
            if path.suffix.lower()!='.pdf': raise ValueError('legal profile requires PDF')
            manifest['parser_version']=PARSER_VERSION
            with pymupdf.open(stream=raw,filetype='pdf') as pdf: pages=[p.get_text() for p in pdf]
        else:
            from .ingest import load_documents
            documents, provenance=load_documents(path)
            manifest.update(provenance)
        cat=Catalog(request.app.state.catalog.path)
        try:
            result=cat.stage(pages,manifest,idem=body.idempotency_key) if profile=='legal' else cat.stage_documents(documents,manifest,idem=body.idempotency_key)
        finally: cat.db.close()
    except (ValueError,KeyError) as exc: raise HTTPException(422,str(exc)) from exc
    return {'job_id':result['job_id'],'revision':result['revision'],'state':result['state']}

@app.get('/api/v2/ingest-jobs/{job_id}')
def get_job(job_id: str,request: Request,user=Depends(identity)):
    admin(user)
    cat=Catalog(request.app.state.catalog.path)
    try: row=cat.db.execute('SELECT job_id,revision,state,error FROM jobs WHERE job_id=?',(job_id,)).fetchone()
    finally: cat.db.close()
    if not row: raise HTTPException(404,'job unavailable')
    return dict(row)

class LegacyQuery(BaseModel):
    message: str = Field(min_length=1,max_length=4000)
    model_type: str = Field(default='local',max_length=80)
    session_id: str | None = Field(default=None,max_length=128,pattern=r'^[A-Za-z0-9_-]+$')
    event_date: date | None = None

def legacy_query(body, user):
    import hashlib, logging
    if setting('LEGACY_ADAPTER')!='1': raise HTTPException(404,'legacy adapter disabled')
    logging.getLogger('legal_rag').info('legacy route backend=v2')
    sid=body.session_id or 'legacy_'+hashlib.sha256(user.encode()).hexdigest()[:32]
    return Query(query=body.message,session_id=sid,event_date=body.event_date,profile='legal')

@app.post('/api/chat')
@app.post('/chat')
async def legacy_chat(body: LegacyQuery,request: Request,user=Depends(identity)):
    result=await chat(legacy_query(body,user),request,user)
    return {'response':result['answer'],'model_type':body.model_type,'intent_type':'consultation','references':result['citations'],'backend':'v2','verification':result['verification']}

@app.post('/api/chat/stream')
async def legacy_stream(body: LegacyQuery,request: Request,user=Depends(identity)):
    query=legacy_query(body,user); bundle=await get_bundle(query,request,user)
    async def emit():
        try:
            if generation_allowed(bundle,query,request.app):
                parts=[]
                async for delta in request.app.state.generator.stream(query.query,bundle['evidence']):
                    if await request.is_disconnected(): return
                    parts.append(delta)
                    # SSE multiline data uses one data field per line.
                    yield ''.join('data: '+line+'\n' for line in delta.split('\n'))+'\n'
                if validate_citations(''.join(parts),bundle['evidence'])['status']=='invalid':
                    yield 'event: answer_rejected\ndata: {"discard_pending_answer":true,"reason":"invalid_citations"}\n\n'
                    return
            else: yield 'data: 目前只能展示检索资料，不能据此生成法律结论，请补充事实并由专业人士核验。\n\n'
            yield 'data: [DONE]\n\n'
        except asyncio.CancelledError: raise
        except Exception: yield 'data: 服务暂不可用，请稍后重试。\n\n'
    return StreamingResponse(emit(),media_type='text/event-stream')

legal_app.include_router(app.router)
