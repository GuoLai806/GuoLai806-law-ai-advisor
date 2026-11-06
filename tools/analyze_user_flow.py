"""Retrospective analysis: never changes queries, answers or selected configuration."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import sys

import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from benchmarks.metrics import evaluate
from benchmarks.verification_metrics import aggregate

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

def analyze(report, qrels, trec, output):
    source = json.loads(Path(report).read_text())
    rows = source['rows']
    assert source['queries_expected'] == source['queries_recorded'] == len(rows)
    assert len({row['query_id'] for row in rows}) == len(rows)
    result = {'input_sha256': digest(report), 'queries': len(rows), 'source_unchanged': source['source_unchanged'],
              'failure_counts': dict(Counter((row['failure'] or {}).get('type', 'none') for row in rows)),
              'source_reread_verified': sum(bool(row.get('source_reread_verified')) for row in rows),
              'session_reread_verified': sum(bool(row.get('session_reread_verified')) for row in rows),
              'structurally_valid_citations': sum(row.get('verification', {}).get('status') == 'structurally_valid' for row in rows),
              'observed_model_identities': [json.loads(value) for value in sorted({json.dumps({k:v for k,v in (row.get('observed_answer_model') or {}).items() if k not in ('trace_id','seq')},sort_keys=True) for row in rows})],
              'interpretation': 'structural citation validity and original-text hashes do not establish semantic support'}
    rng = np.random.default_rng(20261106)
    indices = rng.integers(0, len(rows), (10000, len(rows)))
    vectors = {name: np.array([row['retrieval_metrics'][name] for row in rows]) for name in ('ndcg_at_10','recall_at_20','mrr_at_10')}
    vectors['claim_label_accuracy'] = np.array([not row.get('failure') and row.get('predicted_label') == row['gold_label'] for row in rows], dtype=float)
    result['bootstrap'] = {'method': 'query-level percentile bootstrap; 10,000 samples; seed 20261106; 95% interval',
                           'limits': 'within-dataset uncertainty; queries sharing evidence may be dependent; no external paired significance claim',
                           'metrics': {name: {'mean': float(vector.mean()), 'low': float(np.quantile(vector[indices].mean(axis=1), .025)),
                                             'high': float(np.quantile(vector[indices].mean(axis=1), .975))} for name,vector in vectors.items()}}
    measured=aggregate(rows)
    nonlinear={key:[] for key in ('claim_label_macro_f1','cited_document_label_f1')}
    for sample in indices:
        statistics=aggregate([rows[int(index)] for index in sample])
        for key in nonlinear:
            nonlinear[key].append(statistics[key])
    for key,values in nonlinear.items():
        result['bootstrap']['metrics'][key]={'mean':measured[key],'low':float(np.quantile(values,.025)),
                                            'high':float(np.quantile(values,.975))}
    result['flow_audit']={'queries_without_evaluation_failure':sum(not row.get('failure') for row in rows),
                          'actual_user_flow_completed':sum(bool(row.get('user_flow_completed')) for row in rows) if all('user_flow_completed' in row for row in rows) else None,
                          'user_flow_measurement_available':all('user_flow_completed' in row for row in rows),
                          'failed_queries':sum(bool(row.get('failure')) for row in rows),
                          'usage_available':sum(bool(row.get('usage')) for row in rows),
                          'usage_complete':sum(bool(row.get('usage_complete')) for row in rows),
                          'retried_queries':sum(bool(row.get('generation_retries')) for row in rows),
                          'generation_attempts':sum(len(row.get('generation_attempts',[])) for row in rows),
                          'queries_with_generation_attempts':sum(bool(row.get('generation_attempts')) for row in rows),
                          'queries_without_generation_attempts':sum(not row.get('generation_attempts') for row in rows) if source.get('runtime_identity',{}).get('empty_retry_thinking') else None,
                          'usage_scope':'sum of reported usage only; unknown attempts remain unknown',
                          'actual_openviking_queries':sum(row.get('retrieval_trace',{}).get('dense_backend')=='openviking' for row in rows),
                          'failed_predictions_are_empty':all(not row.get('predicted_label') and not row.get('cited_documents') for row in rows if row.get('failure')),
                          'all_raw_hashes_available':all(row.get('raw_sha256') for row in rows)}
    result['failure_provenance_counts']=dict(Counter(
        str((row.get('failure') or {}).get('error_type',(row.get('failure') or {}).get('type','none')))
        +':'+str((row.get('failure') or {}).get('http_status','')) for row in rows))
    usage_attempts=[attempt['usage'] for row in rows for attempt in row.get('usage_attempts',[])]
    rate_fields={'prompt_cache_hit_tokens','prompt_cache_miss_tokens','completion_tokens'}
    if source.get('runtime_identity',{}).get('answer_model')=='deepseek-v4-pro' and usage_attempts and all(rate_fields<=set(usage) for usage in usage_attempts):
        off_peak=sum(usage['prompt_cache_hit_tokens']*.022+usage['prompt_cache_miss_tokens']*.66+usage['completion_tokens']*1.98 for usage in usage_attempts)/1_000_000
        result['known_usage_price_reference_usd']={'off_peak':off_peak,'peak':2*off_peak,
            'source':'https://api-docs.deepseek.com/quick_start/pricing/',
            'snapshot_html_sha256':'af6679690eb9a7838c47727bfc5ffdc4ac91c4445a0c1219741ffe00c076ece5',
            'scope':'reported attempts only; reasoning is already in completion tokens; excludes unreported attempts, local hardware, tax and billing adjustments; not a bill or total cost bound'}
    result['answer_latency_ms']= {}
    for name in ('stream_total_ms','user_total_ms','first_answer_ms','service_total_ms'):
        values=[row[name] for row in rows if row.get(name) is not None]
        result['answer_latency_ms'][name]={'n':len(values),'p50':float(np.quantile(values,.5)),
                                          'p95':float(np.quantile(values,.95))} if values else {'n':0}
    result['percentile_policy']='analysis uses linearly interpolated NumPy quantiles; original runner order-statistic quantiles also retained'
    result['user_wait_by_flow_status_ms']={}
    if all('user_flow_completed' in row for row in rows):
        for completed,name in ((True,'completed'),(False,'failed')):
            values=[row['user_total_ms'] for row in rows if row['user_flow_completed'] is completed]
            result['user_wait_by_flow_status_ms'][name]={'n':len(values),'p50':float(np.quantile(values,.5)),
                'p95':float(np.quantile(values,.95))} if values else {'n':0}
    result['stage_latency_ms'] = {}
    for stage in ('lexical_ms','dense_ms','rank_and_rerank_ms'):
        values = [row.get('retrieval_trace',{}).get('timings_ms',{}).get(stage) for row in rows]
        values = [value for value in values if value is not None]
        result['stage_latency_ms'][stage] = {'n':len(values), 'p50':float(np.quantile(values,.5)), 'p95':float(np.quantile(values,.95))} if values else {'n':0}
    result['latency_definition'] = {'stream_total_ms':'POST start through SSE end; includes retrieval and answer',
                                   'user_total_ms':'POST start through all evidence source and session rereads',
                                   'first_answer_ms':'first user-visible answer content; reasoning is not visible',
                                   'service_total_ms':'server completion latency, available only on completed streams'}
    references = {}
    for line in Path(qrels).read_text().splitlines()[1:]:
        qid,did,value = line.split('\t')
        references.setdefault(qid,{})[did] = int(value)
    ablations = {name:[] for name in ('bm25','openviking_dense','rrf_before_reranking')}
    coverage_gaps = []
    for row in rows:
        candidates = row.get('retrieval_trace',{}).get('candidates',[])
        for stream,name in (('bm25','bm25'),('dense','openviking_dense')):
            ranked = {source['rank']:item['document_id'] for item in candidates for source in item['sources']
                      if source['source']==stream and item.get('document_id')}
            missing = sorted(set(range(1,21))-set(ranked))
            if missing:
                coverage_gaps.append({'query_id':row['query_id'],'stream':stream,'missing_top20':missing})
            else:
                ablations[name].append({'query_id':row['query_id'],'metrics':evaluate(row['query_id'],references[row['query_id']],
                                       [ranked[rank] for rank in sorted(ranked)])})
        fused = sorted(candidates,key=lambda item:(-item['rrf_score'],item['unit_id']))
        ablations['rrf_before_reranking'].append({'query_id':row['query_id'],'metrics':evaluate(row['query_id'],references[row['query_id']],
                                                 [item['document_id'] for item in fused if item.get('document_id')])})
    result['retrieval_stage_ablations'] = {
        'method':'reconstruct original source rank and pre-reranking RRF from actual trace; require every source rank 1..20; no additional model answer runs',
        'coverage_gaps':coverage_gaps,
        'scope':'retrieval-stage comparison on the same actual user queries; not full-answer A/B or independent system benchmarks',
        'results':{name:{'queries_recorded':len(values),'complete':len(values)==len(rows),
                        'metrics':{metric:float(np.mean([value['metrics'][metric] for value in values])) for metric in ('ndcg_at_10','recall_at_20','mrr_at_10')} if len(values)==len(rows) else None,
                        'per_query':values} for name,values in ablations.items()}}
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        selected = {row['query_id'] for row in rows}
        q = directory/'qrels'
        lines = Path(qrels).read_text().splitlines()
        converted = []
        for line in lines[1:]:
            qid,did,value = line.split('\t')
            if qid in selected: converted.append(f'{qid} 0 {did} {value}\n')
        q.write_text(''.join(converted))
        def run_file(limit):
            file = directory/f'run-{limit}'
            lines = []
            for row in rows:
                ranked = list(dict.fromkeys(item['document_id'] for item in row.get('retrieval_trace',{}).get('candidates',[]) if item.get('document_id') and item['document_id'] != row['query_id']))
                lines.extend(f"{row['query_id']} Q0 {did} {rank+1} {-rank} openviking-e2e\n" for rank,did in enumerate(ranked[:limit]))
            file.write_text(''.join(lines))
            return file
        values = {}
        for run, metrics in ((run_file(60), ['ndcg_cut.10','recall.20']), (run_file(10),['recip_rank'])):
            command = [str(trec),'-c','-q'] + [part for metric in metrics for part in ['-m',metric]] + [str(q),str(run)]
            text = subprocess.run(command, text=True, capture_output=True, check=True).stdout
            for line in text.splitlines():
                metric,qid,value = line.split()
                values.setdefault(qid,{})[metric] = float(value)
        assert set(values) == selected | {'all'}
        names = {'ndcg_at_10':'ndcg_cut_10','recall_at_20':'recall_20','mrr_at_10':'recip_rank'}
        differences = [{'query_id':row['query_id'], 'max_absolute_difference': max(abs(row['retrieval_metrics'][name]-values[row['query_id']][metric]) for name,metric in names.items())} for row in rows]
        aggregate_difference = {name:abs(source['retrieval_metrics'][name]-values['all'][metric]) for name,metric in names.items()}
        result['trec_eval'] = {'source':'https://github.com/usnistgov/trec_eval/tree/v9.0.8', 'binary_sha256':digest(trec),
                               'method':'-c -q, linear-gain nDCG@10, Recall@20; reciprocal rank on top10; BEIR same-ID filtering',
                               'reported':{name:values['all'][metric] for name,metric in names.items()},
                               'aggregate_absolute_difference':aggregate_difference,'per_query':differences,
                               'within_rounding':max([d['max_absolute_difference'] for d in differences]+list(aggregate_difference.values()))<=.000051}
        assert result['trec_eval']['within_rounding']
    write(output,result)
    print(json.dumps({key:result[key] for key in ('queries','failure_counts','source_reread_verified','session_reread_verified','stage_latency_ms','bootstrap')},ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('report','qrels','trec','output'): parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    analyze(args.report,args.qrels,args.trec,args.output)
