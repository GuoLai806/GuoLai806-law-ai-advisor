"""Verify preserved benchmark artifacts; never repair or rescore a failed row."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import quote, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.e2e import file_hash, source_hashes
from benchmarks.verification_metrics import aggregate, gold_label
from benchmarks.run_user_flow import evaluation_ids, read_claims
from benchmarks.prepare import qrels
from benchmarks.short_protocol import validate_selection
from tools.audit_flow_timing import row_clock_checks,report_clock_checks,ns,same,elapsed


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


def scoring_and_usage_checks(row, raw, answer_format='text'):
    events = raw['events']
    text = ''.join(event['data']['delta'] for event in events if event['event'] == 'answer_delta')
    bundle = next((event['data']['bundle'] for event in events if event['event'] == 'evidence_ready'), {})
    raw_label = verdict_from_raw_text(text)
    validations=[e['data'] for e in events if e['event']=='grounding_validation']
    grounding=validations[0] if len(validations)==1 else None
    format_valid=raw_label is not None
    if answer_format=='grounded_json':
        format_valid=bool(grounding and grounding.get('status')=='verbatim_quotes_checked'
                          and grounding.get('decision') in ('SUPPORT','CONTRADICT','NOT_ENOUGH_INFO')
                          and grounding['decision']==raw_label)
        if not format_valid: raw_label=None
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
            or not format_valid,
        'grounding_metadata_matches_raw': answer_format!='grounded_json' or row.get('grounding_validation')==grounding,
        'generation_metadata_matches_raw': all(row.get(key) == value for key, value in metadata.items()),
        'generation_field_mismatches': [key for key, value in metadata.items() if row.get(key) != value],
    }


def exploration_checks(row, raw, strategy='direct'):
    """Independently audit previews and the first preserved full-source read.

    A provider/HTTP/integrity failure is valid evidence of a failed attempt. It
    must not be turned into a successful preview or a verified source in rows.
    Unopened previews retain supplied source hashes; only the preserved first
    original can be independently hash-verified by this artifact auditor.
    """
    checks = {'strategy_matches_raw': strategy in ('direct', 'explore_then_answer')
              and raw.get('user_flow_strategy', 'direct') == strategy}
    observation = raw.get('exploration')
    if strategy == 'direct':
        checks['direct_has_no_exploration'] = observation is None and row.get('exploration') is None
        return checks
    checks['exploration_attempt_preserved'] = isinstance(observation, dict) or bool(row.get('failure'))
    if not isinstance(observation, dict):
        checks['missing_exploration_not_successful'] = bool(row.get('failure')) and not row.get('user_flow_completed') and not row.get('exploration')
        return checks
    try:
        previous = 0.0
        for name in ('started', 'preview_received', 'original_received'):
            sample = observation.get(name)
            if sample is None:
                continue
            monotonic = sample['monotonic_ms']
            ordered = (isinstance(monotonic, (int, float)) and not isinstance(monotonic, bool)
                       and math.isfinite(monotonic) and monotonic >= previous
                       and (row.get('user_total_ms') is None or monotonic <= row['user_total_ms']))
            checks[name + '_clock_preserved'] = (ordered and ns(sample['utc']) == sample['wall_ns']
                                                  and same(sample['observed_wall_ms'], elapsed(raw.get('started_wall_ns'), sample['wall_ns'])))
            previous = monotonic
        for prefix, bytes_field in [('preview', 'response_bytes'), ('original', 'original_bytes')]:
            text = observation.get(prefix + '_response_text')
            present = text is not None
            status = observation.get(prefix + '_http_status')
            checks[prefix + '_response_preserved'] = (
                (not present and status is None and observation.get(bytes_field) is None
                 and observation.get(prefix + '_received') is None)
                or (isinstance(text, str) and isinstance(status, int) and not isinstance(status, bool)
                    and 100 <= status <= 599 and observation.get(prefix + '_received') is not None
                    and observation.get(bytes_field) == len(text.encode('utf-8'))))
        bundle = observation.get('bundle')
        contract_valid = False
        if bundle is not None:
            parsed = json.loads(observation['preview_response_text'])
            # ExploreBundle serialization supplies these nullable defaults.
            parsed.setdefault('trace_id', None)
            parsed.setdefault('retrieval_trace', None)
            checks['preview_matches_response'] = parsed == bundle
            checks['preview_contract_preserved'] = (bundle.get('kind') == 'exploration'
                and bundle.get('status') == 'exploration_only' and 'evidence' not in bundle
                and isinstance(bundle.get('items'), list) and isinstance(bundle.get('corpus_revision'), str))
            for item in bundle['items']:
                pointer = '/api/v2/units/' + quote(item['unit_id'], safe='') + '?' + urlencode({'revision': bundle['corpus_revision']})
                valid_item = (item['revision'] == bundle['corpus_revision'] and item['read_pointer'] == pointer
                              and re.fullmatch('[0-9a-f]{64}', item['content_sha256']) is not None
                              and re.fullmatch('[0-9a-f]{64}', item['document_sha256']) is not None
                              and isinstance(item['preview'], str) and len(item['preview']) <= 320
                              and type(item['preview_truncated']) is bool and type(item['has_more']) is bool
                              and (not item['preview_truncated'] or item['has_more']) and 'text' not in item)
                checks['preview_contract_preserved'] &= valid_item
            contract_valid = checks['preview_contract_preserved']
            checks['preview_contract_preserved'] = contract_valid or (bool(row.get('failure'))
                and not observation.get('original_verified') and not row.get('user_flow_completed'))
        unit = observation.get('original')
        if unit is not None:
            checks['original_matches_response'] = json.loads(observation['original_response_text']) == unit
        verified = False
        if bundle and bundle.get('items') and unit:
            item = bundle['items'][0]
            part = next((part for part in unit.get('parts', []) if part['part_no'] == item.get('part_no')), unit)
            verified = (contract_valid and unit['unit_id'] == item['unit_id'] and unit['revision'] == bundle['corpus_revision']
                        and hashlib.sha256(unit['text'].encode()).hexdigest() == item['document_sha256']
                        and hashlib.sha256(part['text'].encode()).hexdigest() == item['content_sha256']
                        and part['text'].startswith(item['preview'])
                        and item['preview_truncated'] == (len(part['text']) > len(item['preview']))
                        and part.get('locator') == item['locator'])
        checks['original_verified_flag_consistent'] = bool(observation.get('original_verified')) == verified
        expected = None
        if observation.get('original_verified'):
            checks['successful_exploration_has_successful_http'] = (200 <= observation['preview_http_status'] < 300
                and 200 <= observation['original_http_status'] < 300 and not bundle['degraded'] and bool(bundle['items']))
            expected = {'preview_ms': observation['preview_received']['monotonic_ms'],
                        'original_ms': observation['original_received']['monotonic_ms'],
                        'response_bytes': observation['response_bytes'], 'original_bytes': observation['original_bytes'],
                        'items': len(bundle['items']), 'truncated': sum(item['preview_truncated'] for item in bundle['items']),
                        'original_verified': True}
        checks['exploration_row_matches_observation'] = row.get('exploration') == expected
        checks['failed_exploration_not_successful'] = bool(observation.get('original_verified')) or (bool(row.get('failure')) and not row.get('user_flow_completed'))
        answer_bundle = next((e['data']['bundle'] for e in raw['events'] if e['event'] == 'evidence_ready'), None)
        if answer_bundle and bundle:
            checks['answer_uses_explored_revision'] = (answer_bundle['corpus_revision'] == bundle['corpus_revision']
                and raw['request']['corpus_revision'] == bundle['corpus_revision'])
    except (KeyError, TypeError, ValueError, StopIteration):
        checks['exploration_artifact_structure_valid'] = False
    return checks


def audit(args):
    path = Path(args.report)
    report = json.loads(path.read_text())
    config = json.loads(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    claims = read_claims(args.claims)
    qrel_ids = {line.split('\t')[0] for line in Path(args.qrels).read_text().splitlines()[1:]}
    lines = [json.loads(line) for line in Path(str(path)+'.journal.jsonl').read_text().splitlines()]
    rows = report['rows']
    strategy = report.get('user_flow_strategy', 'direct')
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
        'configured_runtime_matches': all(report['runtime_identity'].get(key) == value for key, value in config['runtime_identity'].items()),
        'user_flow_strategy_matches': strategy in ('direct', 'explore_then_answer')
            and strategy == config.get('user_flow_strategy', 'direct') == lines[0].get('user_flow_strategy', 'direct'),
    }
    if report['split'] == 'validation':
        checks['query_set'] = [row['query_id'] for row in rows] == manifest['validation']['query_ids']
        checks['qrels_hash'] = file_hash(args.qrels) == manifest['qrels'][manifest['validation']['source_split']]['sha256']
    else:
        if 'short_evaluation' in manifest:
            identifiers, _ = evaluation_ids(manifest,'test',qrels(Path(args.qrels)),claims,file_hash(args.claims))
            checks['query_set'] = [row['query_id'] for row in rows] == identifiers and len(rows)==30
        else:
            checks['query_set'] = {row['query_id'] for row in rows} == qrel_ids and len(rows) == manifest['qrels']['test']['qids']
        checks['qrels_hash'] = file_hash(args.qrels) == manifest['qrels']['test']['sha256']
        if args.selection:
            selected = json.loads(Path(args.selection).read_text())
            checks['frozen_selection'] = selected['config_sha256'] == file_hash(args.config)
            if 'short_evaluation' in manifest:
                checks['selection_manifest_hash'] = selected.get('manifest_sha256') == file_hash(args.manifest)
    if 'short_evaluation' in manifest and report['split']=='validation':
        identifiers, _ = evaluation_ids(manifest,'validation',qrels(Path(args.qrels)),claims,file_hash(args.claims))
        checks['query_set'] = [row['query_id'] for row in rows] == identifiers and len(rows)==6
    if 'short_evaluation' in manifest and report['split']=='test':
        if not args.selection: raise ValueError('Short formal audit requires frozen selection')
        identity=validate_selection(args.selection,args.config,args.manifest,args.report,audit=True)
        checks['complete_frozen_protocol']=all(report.get(k)==v for k,v in identity.items())
    if 'clock_identity' in report:
        checks.update(report_clock_checks(report))
    raw_rows = []
    for row in rows:
        raw_path = Path(args.raw_dir) / (path.stem+'-'+row['query_id']+'.json')
        raw = json.loads(raw_path.read_text())
        deltas = ''.join(event['data']['delta'] for event in raw['events'] if event['event'] == 'answer_delta')
        bundle = next((event['data']['bundle'] for event in raw['events'] if event['event']=='evidence_ready'),None)
        observed_trace=(bundle or {}).get('retrieval_trace') or ((raw.get('exploration') or {}).get('bundle') or {}).get('retrieval_trace') or {}
        backend_required=(observed_trace.get('dense_backend')=='openviking'
                          and observed_trace.get('dense_scope')==config['runtime_identity']['dense_scope'])
        final = next((event['data'] for event in raw['events'] if event['event']=='completed'),None)
        flow_consistent = not row.get('user_flow_completed') or (
            bool(bundle) and not bundle.get('degraded') and bool(final) and final.get('status')=='ready'
            and bool(deltas) and row.get('verification',{}).get('status')=='structurally_valid'
            and bool(row.get('source_reread_verified'))
            and (config['service_configuration'].get('store_conversations')!='1' or bool(row.get('session_reread_verified'))))
        raw_rows.append({'query_id': row['query_id'], 'hash_matches': file_hash(raw_path) == row['raw_sha256'],
                         'answer_matches_raw': row.get('answer', deltas) == deltas,
                         'trace_matches_raw': (bundle.get('retrieval_trace')==row.get('retrieval_trace') if bundle
                                               else bool(row.get('failure')) and not row.get('retrieval_trace')),
                         'required_backend_observed':backend_required,
                         'successful_backend_matches':backend_required or bool(row.get('failure')) and not row.get('user_flow_completed'),
                         'flow_flag_consistent':flow_consistent,
                         **scoring_and_usage_checks(row, raw,config['runtime_identity'].get('answer_format','text'))})
        exploration = exploration_checks(row, raw, strategy)
        raw_rows[-1]['exploration_checks'] = exploration
        raw_rows[-1]['exploration_records_consistent'] = all(exploration.values())
        if 'clock_identity' in report:
            timing=row_clock_checks(row,raw)
            raw_rows[-1]['clock_checks']=timing
            raw_rows[-1]['clock_records_consistent']=all(timing.values())
    checks['all_raw_hashes_match'] = all(row['hash_matches'] for row in raw_rows)
    checks['all_answers_match_raw'] = all(row['answer_matches_raw'] for row in raw_rows)
    checks['all_traces_match_raw'] = all(row['trace_matches_raw'] for row in raw_rows)
    checks['all_flow_flags_consistent'] = all(row['flow_flag_consistent'] for row in raw_rows)
    checks['all_exploration_records_consistent'] = all(row['exploration_records_consistent'] for row in raw_rows)
    checks['actual_openviking_scope'] = all(row['successful_backend_matches'] for row in raw_rows)
    for field in ('prediction_matches_raw', 'cited_documents_match_raw', 'format_failure_consistent', 'generation_metadata_matches_raw','grounding_metadata_matches_raw'):
        checks['all_'+field] = all(row[field] for row in raw_rows)
    if 'clock_identity' in report:
        checks['all_clock_records_consistent']=all(row['clock_records_consistent'] for row in raw_rows)
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
              'required_backend_observed_queries':sum(row['required_backend_observed'] for row in raw_rows),
              'required_backend_unobserved_or_mismatched_queries':sum(not row['required_backend_observed'] for row in raw_rows),
              'scope': 'artifact consistency audit, including independent first-line label, cited-document, per-attempt usage and preserved exploration/original extraction; failed requests may have no observed backend, explicitly counted above. A passed audit does not mean every answer is correct or unobserved backends were verified.'}
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
