import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time

import httpx
from .e2e import Journal,file_hash,now,save_json,source_hashes
from .metrics import evaluate
from .prepare import jsonl,qrels
from .verification_metrics import aggregate,gold_label,parse_verdict

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
    return {'generation_attempts':attempts,'generation_retries':retries,'observed_answer_models':identities,
            'observed_answer_model':identities[0] if identities else None,'usage':usage,
            'usage_attempts':[{'attempt':number,'usage':value} for number,value in sorted(by_attempt.items())],
            'usage_complete':bool(usage) and set(by_attempt)==expected}

async def execute(args):
    output=Path(args.output)
    if output.exists() or Path(str(output)+'.journal.jsonl').exists():
        raise FileExistsError('Preserve earlier benchmark outputs')
    manifest=json.loads(Path(args.manifest).read_text())
    config=json.loads(Path(args.config).read_text())
    if config['source_sha256']!=source_hashes():
        raise ValueError('Frozen source identity changed')
    queries=jsonl(Path(args.data_dir)/'queries.jsonl')
    claims={str(row['id']):row for row in map(json.loads,Path(args.claims).read_text().splitlines())}
    if file_hash(args.claims)!=config['claims_sha256'][args.split]:
        raise ValueError('Gold claim identity changed')
    split_name=manifest['validation']['source_split'] if args.split=='validation' else 'test'
    references=qrels(Path(args.data_dir)/'qrels'/f'{split_name}.tsv')
    if file_hash(Path(args.data_dir)/'qrels'/f'{split_name}.tsv')!=manifest['qrels'][split_name]['sha256']:
        raise ValueError('Relevance annotations changed')
    identifiers=manifest['validation']['query_ids'] if args.split=='validation' else sorted(references)
    if set(identifiers)-set(claims) or any(claims[key]['claim']!=queries[key]['text'] for key in identifiers):
        raise ValueError('Claim/query mapping mismatch')
    if file_hash(Path(args.data_dir)/'corpus.jsonl')!=manifest['corpus_sha256'] or file_hash(Path(args.data_dir)/'queries.jsonl')!=manifest['queries_sha256']:
        raise ValueError('Benchmark data identity changed')
    if args.split=='test':
        if not args.selection:
            raise ValueError('Test requires frozen validated selection')
        selection=json.loads(Path(args.selection).read_text())
        if selection['config_sha256']!=file_hash(args.config):
            raise ValueError('Test does not match validation selection')
    incoming=json.load(sys.stdin)
    token=incoming['token']
    journal=Journal(str(output)+'.journal.jsonl')
    raw_root=Path(args.raw_dir)
    raw_root.mkdir(parents=True,exist_ok=True)
    report={'started_utc':now(),'split':args.split,'queries_expected':len(identifiers),
            'config_sha256':file_hash(args.config),'manifest_sha256':file_hash(args.manifest),
            'source_sha256':source_hashes(),'source_identity_checked':True,'rows':[]}
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
                started=time.monotonic()
                row={'query_id':query_id,'gold_label':gold_label(claims[query_id]),
                     'gold_documents':sorted(claims[query_id].get('evidence',{})),
                     'predicted_label':None,'cited_documents':[],'failure':None,'user_flow_completed':False}
                events=[]
                bundle=None
                answer=[]
                final=None
                event_kind=None
                first_answer=None
                try:
                    body={'query':queries[query_id]['text'],'session_id':f"benchmark-{args.split}-{output.stem}-{query_id}",
                          'max_evidence_items':config['max_evidence_items'],'max_evidence_tokens':6000,'include_retrieval_trace':True}
                    async with client.stream('POST',args.url.rstrip('/')+'/api/v2/chat/stream',json=body) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith('event: '):
                                event_kind=line[7:]
                            elif line.startswith('data: '):
                                payload=json.loads(line[6:])
                                events.append({'event':event_kind,'data':payload,'received_ms':(time.monotonic()-started)*1000})
                                if event_kind=='evidence_ready':
                                    bundle=payload['bundle']
                                elif event_kind=='answer_delta':
                                    if first_answer is None:
                                        first_answer=(time.monotonic()-started)*1000
                                    answer.append(payload['delta'])
                                elif event_kind=='completed':
                                    final=payload
                                elif event_kind in ('error','answer_rejected'):
                                    row['failure']={'type':event_kind,'detail':payload.get('reason',payload.get('message','stream rejected')),
                                                    **{key:payload[key] for key in ('code','error_type','http_status','failure_phase') if key in payload}}
                    raw={'query_id':query_id,'request':body,'events':events}
                    row['stream_total_ms']=(time.monotonic()-started)*1000
                    raw_path=raw_root/(output.stem+'-'+query_id+'.json')
                    save_json(raw_path,raw)
                    row['raw_sha256']=file_hash(raw_path)
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
                    row['observed_answer_model']=next((event['data'] for event in events if event['event']=='model_identity'),None)
                    if not final or final.get('status')!='ready' or not text or verification.get('status')!='structurally_valid':
                        row['failure']=row['failure'] or {'type':'answer_not_completed','detail':final.get('status') if final else 'no completion event'}
                    flow_ready=not row['failure']
                    if not row['failure']:
                        row['predicted_label']=parse_verdict(text)
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
                if not row.get('raw_sha256'):
                    raw_path=raw_root/(output.stem+'-'+query_id+'.json')
                    save_json(raw_path,{'query_id':query_id,'events':events,'failure':row['failure']})
                    row['raw_sha256']=file_hash(raw_path)
                row['user_total_ms']=(time.monotonic()-started)*1000
                report['rows'].append(row)
                journal.record({'type':'query','completed':completed,'row':row})
                print(json.dumps({'completed':completed,'total':len(identifiers),'failures':sum(bool(item['failure']) for item in report['rows']),'label_correct':sum(item.get('predicted_label')==item['gold_label'] for item in report['rows']),'last_ms':row['user_total_ms']}),flush=True)
            final_capabilities=await client.get(args.url.rstrip('/')+'/api/v2/models')
            final_capabilities.raise_for_status()
            report['runtime_identity_unchanged']=final_capabilities.json()['runtime_identity']==identity
        rows=report['rows']
        report['finished_utc']=now()
        report['failures']=sum(bool(row['failure']) for row in rows)
        report['user_flow_metrics']={'completed':sum(row['user_flow_completed'] for row in rows),'expected':len(rows),
                                  'source_reread_verified':sum(bool(row.get('source_reread_verified')) for row in rows),
                                  'session_reread_verified':sum(bool(row.get('session_reread_verified')) for row in rows),
                                  'scope':'ready answer, structurally valid citations, non-degraded retrieval, source hash reread and configured session reread; separate from classification correctness'}
        report['verification_metrics']=aggregate(rows)
        report['retrieval_metrics']={key:sum(row['retrieval_metrics'][key] for row in rows)/len(rows) for key in ('ndcg_at_10','recall_at_20','mrr_at_10')}
        report['final_evidence_recall']=sum(row['final_evidence_recall'] for row in rows)/len(rows)
        report['latency_ms']={key:{'p50':percentile([row[key] for row in rows if row.get(key) is not None],.5),'p95':percentile([row[key] for row in rows if row.get(key) is not None],.95)} for key in ('stream_total_ms','user_total_ms','first_answer_ms','service_total_ms')}
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
