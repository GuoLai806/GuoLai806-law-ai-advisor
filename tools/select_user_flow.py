"""Freeze one selection from the predeclared final validation pair, before test."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.e2e import file_hash, source_hashes


def select(args):
    if Path(args.output).exists() or Path(args.test_output).exists() or Path(args.test_output+'.journal.jsonl').exists():
        raise FileExistsError('Selection must be frozen before any official test records exist')
    records=[]
    shared=[]
    datasets=[]
    for report_path,config_path in zip(args.reports,args.configs,strict=True):
        report=json.loads(Path(report_path).read_text())
        config=json.loads(Path(config_path).read_text())
        assert report['split']=='validation'
        assert report['queries_expected']==report['queries_recorded']==len(report['rows'])==32
        assert len({row['query_id'] for row in report['rows']})==32
        assert report['source_unchanged'] and report['runtime_identity_unchanged']
        assert report['source_sha256']==config['source_sha256']==source_hashes()
        assert report['config_sha256']==file_hash(config_path)
        assert all('user_flow_completed' in row for row in report['rows'])
        datasets.append((report['manifest_sha256'],[row['query_id'] for row in report['rows']]))
        normalized=deepcopy(config)
        del normalized['service_configuration']['reranker']
        del normalized['runtime_identity']['reranker_revision']
        shared.append(normalized)
        quality=report['verification_metrics']
        records.append({'reranker':config['service_configuration']['reranker'],
                        'report_path':str(Path(report_path).resolve()),'report_sha256':file_hash(report_path),
                        'config_path':str(Path(config_path).resolve()),'config_sha256':file_hash(config_path),
                        'macro_f1':quality['claim_label_macro_f1'],'cited_document_label_f1':quality['cited_document_label_f1'],
                        'ndcg_at_10':report['retrieval_metrics']['ndcg_at_10'],
                        'user_p50_ms':float(np.quantile([row['user_total_ms'] for row in report['rows']],.5)),
                        'evaluation_failures':report['failures'],
                        'user_flow_completed':report['user_flow_metrics']['completed']})
    assert shared[0]==shared[1], 'Candidates must differ only in reranker identity'
    assert datasets[0]==datasets[1], 'Candidates must use the identical frozen validation queries'
    assert {record['reranker'] for record in records}=={'bge','qwen'}
    chosen=max(records,key=lambda row:(row['macro_f1'],row['cited_document_label_f1'],row['ndcg_at_10'],-row['user_p50_ms']))
    result={'selected_utc':datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'config_path':chosen['config_path'],'config_sha256':chosen['config_sha256'],
            'chosen_reranker':chosen['reranker'],'validation_report_sha256':chosen['report_sha256'],
            'source_sha256':source_hashes(),'validation_candidates':records,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
            'rule':'predeclared final pair only; complete 32, unchanged source/runtime, no known implementation defect; all real failures penalized as INVALID; maximize macroF1, then cited-document+label F1, nDCG10, then minimize user median latency',
            'limits':'32-query development signal, not statistical superiority; do not choose the best historical run or tune on formal test',
            'candidate_run_names':args.reports}
    with Path(args.output).open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'chosen_reranker':chosen['reranker'],'config_sha256':chosen['config_sha256'],'candidates':records}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reports',nargs=2,required=True)
    parser.add_argument('--configs',nargs=2,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--test-output',required=True)
    select(parser.parse_args())
