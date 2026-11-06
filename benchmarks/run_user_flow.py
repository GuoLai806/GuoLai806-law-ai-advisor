import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import httpx
from .e2e import Journal,file_hash,save_json,source_hashes
from .metrics import evaluate
from .prepare import jsonl,qrels
from .verification_metrics import LABELS,aggregate,gold_label,parse_verdict

SHORT_SEED = '20261107'
SHORT_RULE = 'Within each gold class take first N SHA256(seed + query_id UTF-8), lexical tie break; order selected IDs by the same key.'
WALL_DIVERGENCE_THRESHOLD_MS = 1000


def clock_sample():
    """Sample both clocks locally; a missing wall clock stays unknown."""
    return time.monotonic(), time.time_ns()


def utc_from_ns(value):
    if value is None:
        return None
    seconds, nanos = divmod(value, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + f'.{nanos:09d}Z'


class ClockTracker:
    def __init__(self, observer=None):
        self.started_monotonic, self.started_wall_ns = clock_sample()
        self.last_wall_ns = self.started_wall_ns
        self.anomalies = [] if self.started_wall_ns is not None else ['wall_clock_missing']
        self.observer = observer
        if self.observer:
            self.observer(self.started_monotonic, self.started_wall_ns)

    def sample(self):
        monotonic, wall_ns = clock_sample()
        if self.observer:
            self.observer(monotonic, wall_ns)
        return self.observe(monotonic, wall_ns)

    def observe(self, monotonic, wall_ns):
        """Observe an existing snapshot, including clock jumps between queries."""
        if wall_ns is None:
            anomaly = 'wall_clock_missing'
        elif self.last_wall_ns is not None and wall_ns < self.last_wall_ns:
            anomaly = 'wall_clock_reversed'
        else:
            anomaly = None
        if anomaly and anomaly not in self.anomalies:
            self.anomalies.append(anomaly)
        if wall_ns is not None:
            self.last_wall_ns = wall_ns
        elapsed_ms = (monotonic - self.started_monotonic) * 1000
        observed_wall_ms = (wall_ns - self.started_wall_ns) / 1_000_000 if wall_ns is not None and self.started_wall_ns is not None else None
        return {'monotonic_ms': elapsed_ms, 'wall_ms': observed_wall_ms if not self.anomalies else None,
                'observed_wall_ms': observed_wall_ms, 'utc': utc_from_ns(wall_ns), 'wall_ns': wall_ns}

    def fields(self, end):
        difference = end['observed_wall_ms'] - end['monotonic_ms'] if end['observed_wall_ms'] is not None else None
        return {'started_utc': utc_from_ns(self.started_wall_ns), 'finished_utc': end['utc'],
                'started_wall_ns': self.started_wall_ns, 'finished_wall_ns': end['wall_ns'],
                'observed_user_wall_ms': end['observed_wall_ms'], 'wall_minus_monotonic_ms': difference,
                'wall_clock_valid': not self.anomalies, 'clock_anomalies': list(self.anomalies),
                'wall_monotonic_divergence': abs(difference) > WALL_DIVERGENCE_THRESHOLD_MS if difference is not None else None}


def clock_identity():
    def details(name):
        info = time.get_clock_info(name)
        return {'implementation': info.implementation, 'resolution_seconds': info.resolution,
                'monotonic': info.monotonic, 'adjustable': info.adjustable}
    return {'monotonic': details('monotonic'), 'wall': {**details('time'), 'api': 'time.time_ns', 'utc_precision': 'nanoseconds'},
            'sampling': 'time.monotonic then time.time_ns; sequential local calls, not atomic',
            'divergence_threshold_ms': WALL_DIVERGENCE_THRESHOLD_MS,
            'scope': 'Observed clock differences only; no sleep attribution, no historical latency correction.'}


def read_claims(path):
    claims = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        claim = json.loads(line)
        identifier = claim.get('id')
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or not str(identifier):
            raise ValueError('Invalid claim identifier')
        identifier = str(identifier)
        if identifier in claims:
            raise ValueError('Duplicate claim identifier')
        gold_label(claim)
        claims[identifier] = claim
    return claims


def evaluation_ids(manifest, split, references, claims, claims_sha256):
    if 'short_evaluation' not in manifest:
        identifiers = manifest['validation']['query_ids'] if split == 'validation' else sorted(references)
        return identifiers, {'kind': 'validation' if split == 'validation' else 'full_test',
                             'scope': 'Original source manifest; full test qrels retained.'}
    if manifest.get('short_evaluation_protocol') != 'scifact-stratified-30-v1':
        raise ValueError('Unknown short evaluation protocol')
    for field, source, count in [('validation', 'train', 6), ('short_evaluation', 'test', 30)]:
        block = manifest[field]
        identifiers = block['query_ids']
        if block.get('source_split') != source or block.get('count') != count or len(identifiers) != count:
            raise ValueError('Short evaluation size/source mismatch')
        if any(not isinstance(identifier, str) for identifier in identifiers) or len(set(identifiers)) != count:
            raise ValueError('Short evaluation IDs must be unique strings')
        if block.get('seed') != SHORT_SEED or block.get('selection') != SHORT_RULE:
            raise ValueError('Short evaluation selection rule changed')
        if block.get('label_counts') != {label: count // 3 for label in LABELS}:
            raise ValueError('Short evaluation class counts changed')
        if not isinstance(block.get('scope'), str) or not block['scope'].startswith('Public stratified small'):
            raise ValueError('Short evaluation scope must identify the public small evaluation')
    if set(manifest['validation']['query_ids']) & set(manifest['short_evaluation']['query_ids']):
        raise ValueError('Short development and formal IDs overlap')
    block = manifest['validation' if split == 'validation' else 'short_evaluation']
    identifiers = block['query_ids']
    if block.get('claims_sha256') != claims_sha256:
        raise ValueError('Short evaluation claim source identity changed')
    if block.get('full_source_query_count') != len(references) or manifest['qrels'][block['source_split']].get('qids') != len(references):
        raise ValueError('Short evaluation full source count changed')
    if set(references) - set(claims) or set(identifiers) - set(references):
        raise ValueError('Short evaluation IDs are outside the source split')
    groups = {label: [] for label in LABELS}
    for identifier in references:
        groups[gold_label(claims[identifier])].append(identifier)
    per_label = block['count'] // 3
    if any(len(group) < per_label for group in groups.values()):
        raise ValueError('Insufficient queries in a short evaluation class')
    def key(identifier):
        return hashlib.sha256((SHORT_SEED + identifier).encode('utf-8')).hexdigest(), identifier
    expected = sorted([identifier for label in LABELS for identifier in sorted(groups[label], key=key)[:per_label]], key=key)
    if identifiers != expected or dict(Counter(gold_label(claims[key]) for key in identifiers)) != block['label_counts']:
        raise ValueError('Short evaluation deterministic membership changed')
    return identifiers, {'kind': 'public_stratified_small', 'protocol': manifest['short_evaluation_protocol'],
                         **{key: block[key] for key in ('source_split', 'seed', 'selection', 'count', 'label_counts', 'scope')}}

def percentile(values, quantile):
    ordered=sorted(values)
    return ordered[min(len(ordered)-1,max(0,int((len(ordered)-1)*quantile)))] if ordered else None

def generation_metadata(events):
    attempts=[{**event['data'],'received_ms':event.get('received_ms')} for event in events if event['event']=='generation_attempt']
    retries=[{**event['data'],'received_ms':event.get('received_ms')} for event in events if event['event']=='generation_retry']
    identities=[event['data'] for event in events if event['event']=='model_identity']
    by_attempt={event['data'].get('attempt',1):event['data']['usage'] for event in events if event['event']=='usage'}
    usage={key:sum(value.get(key,0) for value in by_attempt.values()) for key in set().union(*by_attempt.values())} if by_attempt else None
    expected={attempt['attempt'] for attempt in attempts} if attempts else {1}
    validations=[event['data'] for event in events if event['event']=='grounding_validation']
    return {'generation_attempts':attempts,'generation_retries':retries,'observed_answer_models':identities,
            'observed_answer_model':identities[0] if identities else None,'usage':usage,
            'usage_attempts':[{'attempt':number,'usage':value} for number,value in sorted(by_attempt.items())],
            'usage_complete':bool(usage) and set(by_attempt)==expected,
            'grounding_validation':validations[0] if len(validations)==1 else None}

async def execute(args):
    output=Path(args.output)
    if output.exists() or Path(str(output)+'.journal.jsonl').exists():
        raise FileExistsError('Preserve earlier benchmark outputs')
    manifest=json.loads(Path(args.manifest).read_text())
    config=json.loads(Path(args.config).read_text())
    if config['source_sha256']!=source_hashes():
        raise ValueError('Frozen source identity changed')
    queries=jsonl(Path(args.data_dir)/'queries.jsonl')
    claims=read_claims(args.claims)
    if file_hash(args.claims)!=config['claims_sha256'][args.split]:
        raise ValueError('Gold claim identity changed')
    split_name=manifest['validation']['source_split'] if args.split=='validation' else 'test'
    references=qrels(Path(args.data_dir)/'qrels'/f'{split_name}.tsv')
    if file_hash(Path(args.data_dir)/'qrels'/f'{split_name}.tsv')!=manifest['qrels'][split_name]['sha256']:
        raise ValueError('Relevance annotations changed')
    identifiers,protocol=evaluation_ids(manifest,args.split,references,claims,file_hash(args.claims))
    if set(identifiers)-set(claims) or set(identifiers)-set(queries) or any(claims[key]['claim']!=queries[key]['text'] for key in identifiers):
        raise ValueError('Claim/query mapping mismatch')
    if file_hash(Path(args.data_dir)/'corpus.jsonl')!=manifest['corpus_sha256'] or file_hash(Path(args.data_dir)/'queries.jsonl')!=manifest['queries_sha256']:
        raise ValueError('Benchmark data identity changed')
    if args.split=='test':
        if not args.selection:
            raise ValueError('Test requires frozen validated selection')
        selection=json.loads(Path(args.selection).read_text())
        if selection['config_sha256']!=file_hash(args.config):
            raise ValueError('Test does not match validation selection')
        if 'short_evaluation' in manifest and selection.get('manifest_sha256')!=file_hash(args.manifest):
            raise ValueError('Test selection does not match the frozen short manifest')
    frozen_identity={}
    if args.split=='test' and 'short_evaluation' in manifest:
        from .short_protocol import validate_selection
        frozen_identity=validate_selection(args.selection,args.config,args.manifest,args.output)
    incoming=json.load(sys.stdin)
    token=incoming['token']
    journal=Journal(str(output)+'.journal.jsonl')
    raw_root=Path(args.raw_dir)
    raw_root.mkdir(parents=True,exist_ok=True)
    run_clock=ClockTracker()
    report={'started_utc':utc_from_ns(run_clock.started_wall_ns),'split':args.split,'queries_expected':len(identifiers),
            'config_sha256':file_hash(args.config),'manifest_sha256':file_hash(args.manifest),
            'source_sha256':source_hashes(),'source_identity_checked':True,'evaluation_protocol':protocol,
            'clock_identity':clock_identity(),**frozen_identity,'rows':[]}
    journal.record({'type':'header',**{key:value for key,value in report.items() if key!='rows'},'query_ids':identifiers})
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300,connect=10),trust_env=False,headers={'Authorization':'Bearer '+token}) as client:
            capabilities=await client.get(args.url.rstrip('/')+'/api/v2/models')
            capabilities.raise_for_status()
            identity=capabilities.json()['runtime_identity']
            for key,expected in config['runtime_identity'].items():
                if identity.get(key)!=expected:
                    raise ValueError('Service identity mismatch: '+key)
            report['runtime_identity']=identity
            health=await client.get(args.url.rstrip('/')+'/health/ready')
            health.raise_for_status()
            report['ready']=health.json()
            for completed,query_id in enumerate(identifiers,1):
                query_clock=ClockTracker(observer=run_clock.observe)
                row={'query_id':query_id,'gold_label':gold_label(claims[query_id]),
                     'gold_documents':sorted(claims[query_id].get('evidence',{})),
                     'predicted_label':None,'cited_documents':[],'failure':None,'user_flow_completed':False}
                events=[]
                bundle=None
                answer=[]
                final=None
                event_kind=None
                first_answer=None
                first_answer_wall=None
                stream_end=None
                body=None
                try:
                    body={'query':queries[query_id]['text'],'session_id':f"benchmark-{args.split}-{output.stem}-{query_id}",
                          'max_evidence_items':config['max_evidence_items'],'max_evidence_tokens':6000,'include_retrieval_trace':True}
                    try:
                        async with client.stream('POST',args.url.rstrip('/')+'/api/v2/chat/stream',json=body) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if line.startswith('event: '):
                                    event_kind=line[7:]
                                elif line.startswith('data: '):
                                    payload=json.loads(line[6:])
                                    received=query_clock.sample()
                                    events.append({'event':event_kind,'data':payload,'received_ms':received['monotonic_ms'],
                                                   'received_wall_ms':received['wall_ms'],'observed_received_wall_ms':received['observed_wall_ms'],
                                                   'received_utc':received['utc'],'received_wall_ns':received['wall_ns']})
                                    if event_kind=='evidence_ready':
                                        bundle=payload['bundle']
                                    elif event_kind=='answer_delta':
                                        if first_answer is None:
                                            first_answer=received['monotonic_ms']
                                            first_answer_wall=received['wall_ms']
                                        answer.append(payload['delta'])
                                    elif event_kind=='completed':
                                        final=payload
                                    elif event_kind in ('error','answer_rejected'):
                                        row['failure']={'type':event_kind,'detail':payload.get('reason',payload.get('message','stream rejected')),
                                                        **{key:payload[key] for key in ('code','error_type','http_status','failure_phase') if key in payload}}
                    finally:
                        stream_end=query_clock.sample()
                        row['stream_total_ms']=stream_end['monotonic_ms']
                    if not bundle:
                        raise ValueError('Missing retrieval event')
                    trace=bundle.get('retrieval_trace') or {}
                    if trace.get('dense_backend')!='openviking' or trace.get('dense_scope')!=config['runtime_identity']['dense_scope']:
                        raise ValueError('Request did not use required OpenViking backend')
                    row['retrieval_trace']=trace
                    row['budget']=bundle['budget']
                    row['degraded_reasons']=bundle.get('degraded_reasons',[])
                    if bundle.get('degraded'):
                        row['failure']=row['failure'] or {'type':'retrieval_degraded','detail':row['degraded_reasons']}
                    candidates=[item['document_id'] for item in trace.get('candidates',[]) if item.get('document_id')]
                    row['retrieval_metrics']=evaluate(query_id,references[query_id],candidates)
                    evidence=bundle['evidence']
                    evidence_ids=list(dict.fromkeys(item['document_id'] for item in evidence))
                    relevant={key for key,value in references[query_id].items() if value>0}
                    row['final_evidence_recall']=len(set(evidence_ids)&relevant)/len(relevant)
                    row['evidence']=[{key:item.get(key) for key in ('citation_id','unit_id','document_id','content_sha256','document_sha256','source_locator','reranker_score')} for item in evidence]
                    for item in evidence:
                        read=await client.get(args.url.rstrip('/')+'/api/v2/units/'+item['unit_id'],params={'revision':bundle['corpus_revision']})
                        read.raise_for_status()
                        unit=read.json()
                        matching=next((part for part in unit.get('parts',[]) if part['part_no']==item.get('part_no')),unit)
                        if hashlib.sha256(matching['text'].encode()).hexdigest()!=item['content_sha256']:
                            raise ValueError('Source reread does not match evidence')
                    row['source_reread_verified']=True
                    text=''.join(answer)
                    row['answer']=text
                    verification=next((event['data'] for event in events if event['event']=='verification'),{})
                    row['verification']=verification
                    grounding_events=[e['data'] for e in events if e['event']=='grounding_validation']
                    row['grounding_validation']=grounding_events[0] if len(grounding_events)==1 else None
                    row['observed_answer_model']=next((event['data'] for event in events if event['event']=='model_identity'),None)
                    if not final or final.get('status')!='ready' or not text or verification.get('status')!='structurally_valid':
                        row['failure']=row['failure'] or {'type':'answer_not_completed','detail':final.get('status') if final else 'no completion event'}
                    flow_ready=not row['failure']
                    if not row['failure']:
                        row['predicted_label']=parse_verdict(text)
                        if config['runtime_identity'].get('answer_format')=='grounded_json':
                            grounding=row['grounding_validation']
                            if (not grounding or grounding.get('status')!='verbatim_quotes_checked'
                                    or grounding.get('decision') not in LABELS
                                    or grounding['decision']!=row['predicted_label']):
                                row['predicted_label']=None
                        if not row['predicted_label']:
                            row['failure']={'type':'invalid_answer_format'}
                        cited=set(re.findall(r'\[(E\d+)\]',text))
                        row['cited_documents']=sorted({item['document_id'] for item in evidence if item['citation_id'] in cited})
                    row['first_answer_ms']=first_answer
                    row['service_total_ms']=final.get('total_ms') if final else None
                    if flow_ready and config['service_configuration'].get('store_conversations')=='1':
                        saved=await client.get(args.url.rstrip('/')+'/api/v2/sessions/'+body['session_id']+'/turns')
                        saved.raise_for_status()
                        if not any(turn['answer']==text and turn['query']==body['query'] for turn in saved.json()['turns']):
                            raise ValueError('Final answer was not persisted for the user session')
                        row['session_reread_verified']=True
                    row['user_flow_completed']=flow_ready
                except Exception as exc:
                    exception={'type':type(exc).__name__,'detail':str(exc).replace(token,'<redacted>')}
                    row['flow_failure']=exception
                    row['failure']=row['failure'] or exception
                    row.setdefault('retrieval_metrics',evaluate(query_id,references[query_id],[]))
                    row.setdefault('final_evidence_recall',0.0)
                row.update(generation_metadata(events))
                row['first_answer_ms']=first_answer
                row['service_total_ms']=final.get('total_ms') if final else None
                if row['failure']:
                    row['predicted_label']=None
                    row['cited_documents']=[]
                raw_path=raw_root/(output.stem+'-'+query_id+'.json')
                # Preserve a stream record even if source/session verification fails later.
                raw={'query_id':query_id,'request':body,'events':events,'failure':row['failure'],
                     'started_utc':utc_from_ns(query_clock.started_wall_ns),'started_wall_ns':query_clock.started_wall_ns,
                     'stream_finished_utc':stream_end['utc'] if stream_end else None,
                     'stream_finished_wall_ns':stream_end['wall_ns'] if stream_end else None,
                     'stream_total_ms':stream_end['monotonic_ms'] if stream_end else None,
                     'stream_total_wall_ms':stream_end['wall_ms'] if stream_end else None,
                     'stream_observed_wall_ms':stream_end['observed_wall_ms'] if stream_end else None,
                     'clock_anomalies':list(query_clock.anomalies),'clock_identity':report['clock_identity'],
                     'clock_scope':'Raw stream end snapshot; final user-path end and anomalies are stored in the hashed report row.'}
                save_json(raw_path,raw)
                row['raw_sha256']=file_hash(raw_path)
                query_end=query_clock.sample()
                row.update(query_clock.fields(query_end))
                row['user_total_ms']=query_end['monotonic_ms']
                row['user_total_wall_ms']=query_end['wall_ms']
                row['stream_total_wall_ms']=stream_end['wall_ms'] if stream_end and not query_clock.anomalies else None
                row['first_answer_wall_ms']=first_answer_wall if not query_clock.anomalies else None
                report['rows'].append(row)
                journal.record({'type':'query','completed':completed,'row':row})
                print(json.dumps({'completed':completed,'total':len(identifiers),'failures':sum(bool(item['failure']) for item in report['rows']),'label_correct':sum(item.get('predicted_label')==item['gold_label'] for item in report['rows']),'last_ms':row['user_total_ms']}),flush=True)
            final_capabilities=await client.get(args.url.rstrip('/')+'/api/v2/models')
            final_capabilities.raise_for_status()
            report['runtime_identity_unchanged']=final_capabilities.json()['runtime_identity']==identity
        rows=report['rows']
        run_end=run_clock.sample()
        report['finished_utc']=run_end['utc']
        report['run_monotonic_ms']=run_end['monotonic_ms']
        report['run_wall_ms']=run_end['wall_ms']
        report['run_observed_wall_ms']=run_end['observed_wall_ms']
        report['run_clock_anomalies']=list(run_clock.anomalies)
        report['run_wall_minus_monotonic_ms']=run_end['observed_wall_ms']-run_end['monotonic_ms'] if run_end['observed_wall_ms'] is not None else None
        report['failures']=sum(bool(row['failure']) for row in rows)
        report['user_flow_metrics']={'completed':sum(row['user_flow_completed'] for row in rows),'expected':len(rows),
                                  'source_reread_verified':sum(bool(row.get('source_reread_verified')) for row in rows),
                                  'session_reread_verified':sum(bool(row.get('session_reread_verified')) for row in rows),
                                  'scope':'ready answer, structurally valid citations, non-degraded retrieval, source hash reread and configured session reread; separate from classification correctness'}
        report['verification_metrics']=aggregate(rows)
        report['retrieval_metrics']={key:sum(row['retrieval_metrics'][key] for row in rows)/len(rows) for key in ('ndcg_at_10','recall_at_20','mrr_at_10')}
        report['final_evidence_recall']=sum(row['final_evidence_recall'] for row in rows)/len(rows)
        report['latency_ms']={key:{'p50':percentile([row[key] for row in rows if row.get(key) is not None],.5),'p95':percentile([row[key] for row in rows if row.get(key) is not None],.95)} for key in ('stream_total_ms','user_total_ms','first_answer_ms','service_total_ms')}
        report['wall_latency_ms']={key:{'n':len(values),'p50':percentile(values,.5),'p95':percentile(values,.95)}
                                   for key in ('stream_total_wall_ms','user_total_wall_ms','first_answer_wall_ms')
                                   for values in [[row[key] for row in rows if row.get('wall_clock_valid') and row.get(key) is not None]]}
        report['timing_health']={'wall_monotonic_divergence_rows':sum(row['wall_monotonic_divergence'] is True for row in rows),
                                 'clock_anomaly_rows':sum(bool(row['clock_anomalies']) for row in rows),
                                 'all_wall_clocks_valid':not run_clock.anomalies and all(row['wall_clock_valid'] for row in rows),
                                 'scope':'Wall quantiles include valid observed clocks only; all questions remain in quality and flow metrics.'}
        report['usage_totals']={key:sum((row.get('usage') or {}).get(key,0) for row in rows) for key in ('prompt_tokens','completion_tokens','total_tokens','reasoning_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens')}
        report['usage_available_queries']=sum(bool(row.get('usage')) for row in rows)
        report['usage_complete_queries']=sum(bool(row.get('usage_complete')) for row in rows)
        report['retried_queries']=sum(bool(row.get('generation_retries')) for row in rows)
        report['generation_attempt_count']=sum(len(row.get('generation_attempts',[])) or int(bool(row.get('observed_answer_model') or row.get('usage'))) for row in rows)
        report['usage_totals_scope']='known provider usage only; incomplete attempts are not assumed to cost zero'
        report['source_unchanged']=config['source_sha256']==source_hashes()
        report['queries_recorded']=len(rows)
        save_json(output,report)
        journal.record({'type':'complete','queries':len(rows),'failures':report['failures'],'report_sha256':file_hash(output)})
        print(json.dumps({key:report[key] for key in ('queries_recorded','failures','verification_metrics','retrieval_metrics','latency_ms','usage_totals')}),flush=True)
        return 0 if report['failures']==0 and report['source_unchanged'] and report['runtime_identity_unchanged'] else 1
    finally:
        journal.close()

def main():
    parser=argparse.ArgumentParser()
    for name in ('url','data-dir','manifest','claims','config','output','raw-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--split',choices=('validation','test'),required=True)
    parser.add_argument('--selection')
    raise SystemExit(asyncio.run(execute(parser.parse_args())))

if __name__=='__main__':
    main()
