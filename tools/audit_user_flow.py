"""Verify preserved benchmark artifacts; never repair or rescore a failed row."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.e2e import file_hash, source_hashes
from benchmarks.verification_metrics import aggregate, gold_label


def verdict_from_raw_text(text):
    """Independently read the declared first-line grammar; never infer from prose."""
    if not text.strip():
        return None
    line = text.lstrip().splitlines()[0]
    line = re.sub(r'^#{1,6}\s+', '', line, count=1)
    if line.startswith('**'):
        line = line[2:]
    # Keep the same accepted presentation contract, but do not call the scorer.
    prefix = re.match(r'(?i:(?:final\s+)?(?:verdict|conclusion|label))|(?:最终)?(?:结论|判断|判定|标签)', line)
    if prefix:
        remainder = line[prefix.end():].lstrip()
        if remainder.startswith('**'):
            remainder = remainder[2:].lstrip()
        if not remainder.startswith((':', '：')):
            return None
        line = remainder[1:].lstrip()
    if line.startswith('**'):
        line = line[2:]
    verdict = re.match(r'(SUPPORT|CONTRADICT|NOT_ENOUGH_INFO)\b', line)
    return verdict.group(1) if verdict else None


def generation_from_raw(events):
    """Rebuild attempts and last cumulative usage directly from original SSE."""
    attempts, retries, identities, latest_usage = [], [], [], {}
    for event in events:
        kind, data = event['event'], event['data']
        if kind == 'generation_attempt':
            attempts.append(dict(data, received_ms=event.get('received_ms')))
        elif kind == 'generation_retry':
            retries.append(dict(data, received_ms=event.get('received_ms')))
        elif kind == 'model_identity':
            identities.append(data)
        elif kind == 'usage':
            latest_usage[data.get('attempt', 1)] = data['usage']
    totals = None
    if latest_usage:
        totals = {}
        for usage in latest_usage.values():
            for key, count in usage.items():
                totals[key] = totals.get(key, 0) + count
    attempted = {item['attempt'] for item in attempts} if attempts else {1}
    return {
        'generation_attempts': attempts,
        'generation_retries': retries,
        'observed_answer_models': identities,
        'observed_answer_model': identities[0] if identities else None,
        'usage': totals,
        'usage_attempts': [{'attempt': number, 'usage': latest_usage[number]} for number in sorted(latest_usage)],
        'usage_complete': bool(totals) and set(latest_usage) == attempted,
    }


def scoring_and_usage_checks(row, raw):
    events = raw['events']
    text = ''.join(event['data']['delta'] for event in events if event['event'] == 'answer_delta')
    bundle = next((event['data']['bundle'] for event in events if event['event'] == 'evidence_ready'), {})
    raw_label = verdict_from_raw_text(text)
    if row.get('failure'):
        expected_label, expected_documents = None, []
    else:
        expected_label = raw_label
        cited = set(re.findall(r'\[(E\d+)\]', text))
        expected_documents = sorted({item['document_id'] for item in bundle.get('evidence', [])
                                     if item['citation_id'] in cited})
    metadata = generation_from_raw(events)
    return {
        'prediction_matches_raw': row.get('predicted_label') == expected_label
            and (bool(row.get('failure')) or raw_label is not None),
        'cited_documents_match_raw': row.get('cited_documents', []) == expected_documents,
        'format_failure_consistent': (row.get('failure') or {}).get('type') != 'invalid_answer_format'
            or raw_label is None,
        'generation_metadata_matches_raw': all(row.get(key) == value for key, value in metadata.items()),
        'generation_field_mismatches': [key for key, value in metadata.items() if row.get(key) != value],
    }


def audit(args):
    path = Path(args.report)
    report = json.loads(path.read_text())
    config = json.loads(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    claims = {str(row['id']): row for row in map(json.loads, Path(args.claims).read_text().splitlines())}
    qrel_ids = {line.split('\t')[0] for line in Path(args.qrels).read_text().splitlines()[1:]}
    lines = [json.loads(line) for line in Path(str(path)+'.journal.jsonl').read_text().splitlines()]
    rows = report['rows']
    checks = {
        'record_count': report['queries_expected'] == report['queries_recorded'] == len(rows),
        'unique_queries': len({row['query_id'] for row in rows}) == len(rows),
        'config_hash': report['config_sha256'] == file_hash(args.config),
        'manifest_hash': report['manifest_sha256'] == file_hash(args.manifest),
        'source_hash': report['source_sha256'] == config['source_sha256'] == source_hashes(),
        'source_and_runtime_unchanged': report['source_unchanged'] and report['runtime_identity_unchanged'],
        'journal_rows_match': [line['row'] for line in lines if line['type'] == 'query'] == rows,
        'journal_final_hash': lines[-1]['type'] == 'complete' and lines[-1]['report_sha256'] == file_hash(path),
        'gold_file_hash': config['claims_sha256'][report['split']] == file_hash(args.claims),
        'gold_labels_unchanged': all(row['gold_label'] == gold_label(claims[row['query_id']]) and row['gold_documents'] == sorted(claims[row['query_id']].get('evidence', {})) for row in rows),
        'verification_metrics_match': aggregate(rows) == report['verification_metrics'],
        'failed_predictions_empty': all(row.get('predicted_label') is None and not row.get('cited_documents') for row in rows if row.get('failure')),
        'actual_openviking_scope': all(row.get('retrieval_trace', {}).get('dense_backend') == 'openviking' and row['retrieval_trace']['dense_scope'] == config['runtime_identity']['dense_scope'] for row in rows),
        'configured_runtime_matches': all(report['runtime_identity'].get(key) == value for key, value in config['runtime_identity'].items()),
    }
    if report['split'] == 'validation':
        checks['query_set'] = [row['query_id'] for row in rows] == manifest['validation']['query_ids']
        checks['qrels_hash'] = file_hash(args.qrels) == manifest['qrels'][manifest['validation']['source_split']]['sha256']
    else:
        checks['query_set'] = {row['query_id'] for row in rows} == qrel_ids and len(rows) == manifest['qrels']['test']['qids']
        checks['qrels_hash'] = file_hash(args.qrels) == manifest['qrels']['test']['sha256']
        if args.selection:
            selected = json.loads(Path(args.selection).read_text())
            checks['frozen_selection'] = selected['config_sha256'] == file_hash(args.config)
    raw_rows = []
    for row in rows:
        raw_path = Path(args.raw_dir) / (path.stem+'-'+row['query_id']+'.json')
        raw = json.loads(raw_path.read_text())
        deltas = ''.join(event['data']['delta'] for event in raw['events'] if event['event'] == 'answer_delta')
        bundle = next((event['data']['bundle'] for event in raw['events'] if event['event']=='evidence_ready'),None)
        final = next((event['data'] for event in raw['events'] if event['event']=='completed'),None)
        flow_consistent = not row.get('user_flow_completed') or (
            bool(bundle) and not bundle.get('degraded') and bool(final) and final.get('status')=='ready'
            and bool(deltas) and row.get('verification',{}).get('status')=='structurally_valid'
            and bool(row.get('source_reread_verified'))
            and (config['service_configuration'].get('store_conversations')!='1' or bool(row.get('session_reread_verified'))))
        raw_rows.append({'query_id': row['query_id'], 'hash_matches': file_hash(raw_path) == row['raw_sha256'],
                         'answer_matches_raw': row.get('answer', deltas) == deltas,
                         'trace_matches_raw': bool(bundle) and bundle.get('retrieval_trace')==row.get('retrieval_trace'),
                         'flow_flag_consistent':flow_consistent,
                         **scoring_and_usage_checks(row, raw)})
    checks['all_raw_hashes_match'] = all(row['hash_matches'] for row in raw_rows)
    checks['all_answers_match_raw'] = all(row['answer_matches_raw'] for row in raw_rows)
    checks['all_traces_match_raw'] = all(row['trace_matches_raw'] for row in raw_rows)
    checks['all_flow_flags_consistent'] = all(row['flow_flag_consistent'] for row in raw_rows)
    for field in ('prediction_matches_raw', 'cited_documents_match_raw', 'format_failure_consistent', 'generation_metadata_matches_raw'):
        checks['all_'+field] = all(row[field] for row in raw_rows)
    if args.model_before and args.model_after:
        before = json.loads(Path(args.model_before).read_text())['models']
        after = json.loads(Path(args.model_after).read_text())['models']
        checks['model_bytes_unchanged'] = before == after
    if args.backend_before and args.backend_after:
        before = json.loads(Path(args.backend_before).read_text())
        after = json.loads(Path(args.backend_after).read_text())
        checks['backend_identity_unchanged'] = before == after
    result = {'report_sha256': file_hash(path), 'checks': checks, 'passed': all(checks.values()),
              'raw_rows': raw_rows, 'user_flow_completed': sum(bool(row.get('user_flow_completed')) for row in rows),
              'evaluation_failures': sum(bool(row.get('failure')) for row in rows),
              'usage_complete': sum(bool(row.get('usage_complete')) for row in rows),
              'scope': 'artifact consistency audit, including independent first-line label, cited-document and per-attempt usage extraction from original SSE; a passed audit does not mean every answer is correct'}
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'raw_rows'}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for field in ('report', 'config', 'manifest', 'claims', 'qrels', 'raw-dir', 'output'):
        parser.add_argument('--'+field, required=True)
    for field in ('selection', 'model-before', 'model-after', 'backend-before', 'backend-after'):
        parser.add_argument('--'+field)
    raise SystemExit(audit(parser.parse_args()))
