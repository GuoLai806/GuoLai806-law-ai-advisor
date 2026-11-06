"""Paired descriptive comparison of rankings recorded in one actual run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(report_path, analysis_path, output_path):
    payload = Path(report_path).read_bytes()
    report = json.loads(payload)
    analysis = json.loads(Path(analysis_path).read_bytes())
    if analysis['input_sha256'] != hashlib.sha256(payload).hexdigest():
        raise ValueError('Analysis does not belong to this exact report')
    rows = report['rows']
    query_ids = [row['query_id'] for row in rows]
    if len(rows) != report['queries_expected'] or len(set(query_ids)) != len(rows):
        raise ValueError('A complete unique query set is required')
    results = analysis['retrieval_stage_ablations']['results']
    stages = ('bm25', 'openviking_dense', 'rrf_before_reranking')
    incomplete = [name for name in stages if not results[name]['complete']]
    if incomplete:
        raise ValueError('Incomplete source rankings: ' + ', '.join(incomplete))
    rng = np.random.default_rng(20261106)
    samples = rng.integers(0, len(rows), (10000, len(rows)))
    comparisons = {}
    for stage in stages:
        baseline_rows = results[stage]['per_query']
        baseline = {row['query_id']: row['metrics'] for row in baseline_rows}
        if len(baseline_rows) != len(rows) or set(baseline) != set(query_ids):
            raise ValueError('Ranking query IDs do not match the original report')
        metrics = {}
        for metric in ('ndcg_at_10', 'recall_at_20', 'mrr_at_10'):
            final = np.array([row['retrieval_metrics'][metric] for row in rows])
            initial = np.array([baseline[qid][metric] for qid in query_ids])
            delta = final - initial
            bootstrap = delta[samples].mean(axis=1)
            mean = float(initial.mean())
            metrics[metric] = {
                'stage_mean': mean, 'final_mean': float(final.mean()),
                'absolute_difference': float(delta.mean()),
                'relative_difference': float(delta.mean()) / mean if mean else None,
                'difference_95_interval': [float(np.quantile(bootstrap, .025)),
                                           float(np.quantile(bootstrap, .975))],
                'queries_improved': int((delta > 0).sum()),
                'queries_equal': int((delta == 0).sum()),
                'queries_worsened': int((delta < 0).sum()),
            }
        comparisons[stage] = metrics
    output = {
        'report_sha256': hashlib.sha256(payload).hexdigest(),
        'analysis_sha256': hashlib.sha256(Path(analysis_path).read_bytes()).hexdigest(),
        'queries': len(rows), 'comparisons_final_minus_stage': comparisons,
        'method': 'Retrospective paired query bootstrap, 10000 samples, seed20261106, '
            '95% percentile intervals; original rankings and failures unchanged.',
        'limits': 'Ranking-stage comparison from one observed user-flow run, not separate '
            'end-to-end answer A/B or an external-system significance test. Query dependence '
            'and shared-corpus effects remain; formal results did not tune the selected configuration.',
    }
    with Path(output_path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('report', 'analysis', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    compare(args.report, args.analysis, args.output)
