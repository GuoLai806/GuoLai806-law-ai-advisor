"""Bind a development6 record and two predeclared configs before either formal30."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.e2e import file_hash, save_json, source_hashes
from benchmarks.verification_metrics import LABELS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_config(config):
    result = deepcopy(config)
    for key in ('answer_format', 'generation_gate', 'general_system_prompt'):
        result['service_configuration'].pop(key, None)
    for key in ('answer_format', 'generation_gate', 'system_prompt_sha256'):
        result['runtime_identity'].pop(key, None)
    return result


def freeze(args):
    input_paths = [Path(getattr(args, name)).resolve() for name in
                   ('manifest', 'development_report', 'development_config', 'baseline_config', 'candidate_config')]
    outputs = [Path(getattr(args, name)).resolve() for name in
               ('baseline_test_output', 'candidate_test_output', 'baseline_selection', 'candidate_selection')]
    require(len(set(outputs)) == len(outputs) and not set(outputs) & set(input_paths),
            'Formal outputs and selections must be distinct from each other and inputs')
    formal_outputs = outputs[:2]
    selection_paths = outputs[2:]

    def before_formal():
        for path in formal_outputs:
            if path.exists() or Path(str(path) + '.journal.jsonl').exists():
                raise FileExistsError('Freeze both configs before any formal report or journal exists')

    before_formal()
    if any(path.exists() for path in selection_paths):
        raise FileExistsError('Preserve earlier frozen selections')
    manifest = json.loads(input_paths[0].read_text(encoding='utf-8'))
    report = json.loads(input_paths[1].read_text(encoding='utf-8'))
    development = json.loads(input_paths[2].read_text(encoding='utf-8'))
    baseline = json.loads(input_paths[3].read_text(encoding='utf-8'))
    candidate = json.loads(input_paths[4].read_text(encoding='utf-8'))
    manifest_hash = file_hash(input_paths[0])
    source = source_hashes()
    require(manifest.get('short_evaluation_protocol') == 'scifact-stratified-30-v1',
            'The stratified30 manifest is required')
    for key, count, split in [('validation', 6, 'train'), ('short_evaluation', 30, 'test')]:
        block = manifest[key]
        require(block.get('count') == count and len(block['query_ids']) == len(set(block['query_ids'])) == count,
                'Frozen manifest count or uniqueness mismatch')
        require(block.get('source_split') == split and block.get('seed') == '20261107'
                and block.get('label_counts') == {label: count // 3 for label in LABELS},
                'Frozen manifest stratification mismatch')
    require(not set(manifest['validation']['query_ids']) & set(manifest['short_evaluation']['query_ids']),
            'Development and formal membership must be disjoint')
    require(report.get('split') == 'validation' and report.get('queries_expected') == report.get('queries_recorded') == len(report['rows']) == 6,
            'A complete development6 report is required')
    rows = report['rows']
    require([row['query_id'] for row in rows] == manifest['validation']['query_ids'],
            'Development report membership/order does not match manifest')
    require(dict(Counter(row['gold_label'] for row in rows)) == manifest['validation']['label_counts'],
            'Development report labels do not match stratification')
    require(all('failure' in row and 'user_flow_completed' in row for row in rows),
            'Development rows must preserve failures and user-flow completion')
    require(report.get('failures') == sum(bool(row['failure']) for row in rows),
            'Development failure count disagrees with retained rows')
    require(report.get('source_identity_checked') is True and report.get('source_unchanged') is True
            and report.get('runtime_identity_unchanged') is True, 'Development source/runtime identity was not stable')
    require(report.get('source_sha256') == development.get('source_sha256') == baseline.get('source_sha256') == candidate.get('source_sha256') == source,
            'Frozen source identity changed')
    require(report.get('manifest_sha256') == manifest_hash, 'Development manifest identity mismatch')
    require(report.get('config_sha256') == file_hash(input_paths[2]), 'Development config identity mismatch')
    require(all(report.get('runtime_identity', {}).get(key) == value for key, value in development['runtime_identity'].items()),
            'Development runtime does not match its configuration')
    require(development == candidate, 'Candidate formal config must exactly match the development6 config')
    require(normalized_config(baseline) == normalized_config(candidate),
            'Baseline/candidate differ outside the declared joint grounding strategy')
    for config in (baseline, candidate):
        require(config['claims_sha256'].get('validation') == manifest['validation']['claims_sha256']
                and config['claims_sha256'].get('test') == manifest['short_evaluation']['claims_sha256'],
                'Config claim identity does not match manifest')
    for config, answer_format, gate in [(baseline, 'text', 'reranker_threshold'),
                                        (candidate, 'grounded_json', 'grounded_evidence')]:
        service = config['service_configuration']
        runtime = config['runtime_identity']
        require(service.get('answer_format', 'text') == answer_format and service.get('generation_gate', 'reranker_threshold') == gate,
                'Config mode does not match the declared baseline/candidate strategy')
        require(runtime.get('answer_format', 'text') == answer_format and runtime.get('generation_gate', 'reranker_threshold') == gate,
                'Runtime mode does not match the declared strategy')
    require(str(baseline['service_configuration'].get('rerank_threshold')) in ('0', '0.0')
            and str(baseline['runtime_identity'].get('rerank_threshold')) in ('0', '0.0'),
            'Historical baseline must retain threshold0')

    journal_path = Path(str(input_paths[1]) + '.journal.jsonl')
    journal = [json.loads(line) for line in journal_path.read_text(encoding='utf-8').splitlines()]
    require(len(journal) == 8 and journal[0].get('type') == 'header' and journal[-1].get('type') == 'complete',
            'Development journal is incomplete')
    header = journal[0]
    require(header.get('query_ids') == manifest['validation']['query_ids']
            and header.get('config_sha256') == report['config_sha256']
            and header.get('manifest_sha256') == manifest_hash and header.get('source_sha256') == source,
            'Development journal header identity mismatch')
    require(all(entry.get('type') == 'query' and entry.get('completed') == index and entry.get('row') == row
                for index, (entry, row) in enumerate(zip(journal[1:-1], rows, strict=True), 1)),
            'Development journal rows do not match report')
    require(journal[-1].get('report_sha256') == file_hash(input_paths[1]) and journal[-1].get('queries') == 6
            and journal[-1].get('failures') == report['failures'], 'Development journal completion identity mismatch')
    development_record = {'report_path': str(input_paths[1]), 'report_sha256': file_hash(input_paths[1]),
                          'journal_path': str(journal_path), 'journal_sha256': file_hash(journal_path),
                          'config_path': str(input_paths[2]), 'config_sha256': file_hash(input_paths[2]),
                          'queries_recorded': 6, 'query_ids': [row['query_id'] for row in rows],
                          'evaluation_failures': report['failures'],
                          'user_flow_completed': sum(bool(row['user_flow_completed']) for row in rows),
                          'metrics': report.get('verification_metrics'),
                          'policy': 'All development failures are retained; real provider/retrieval/format failures are not automatically rejected or relabeled.'}
    selected_utc = datetime.now(timezone.utc).isoformat()
    shared = {'selected_utc': selected_utc, 'manifest_path': str(input_paths[0]), 'manifest_sha256': manifest_hash,
              'source_sha256': source, 'development6': development_record,
              'freeze_tool_sha256': file_hash(__file__),
              'rule': 'Predeclared two-config comparison, frozen before either formal report/journal exists; not a winner selection from development or formal scores.',
              'limits': 'Public stratified small evaluation; not hidden/unseen/full test. Joint format, prompt contract and evidence gate change; effects cannot be attributed to one component.'}
    results = []
    for kind, path, config_path, config, test_output in zip(
            ('baseline', 'candidate'), selection_paths, input_paths[3:], (baseline, candidate), formal_outputs, strict=True):
        results.append({**shared, 'kind': kind, 'config_path': str(config_path), 'config_sha256': file_hash(config_path),
                        'runtime_identity': config['runtime_identity'], 'formal_output': str(test_output),
                        'paired_selection_path': str(selection_paths[1] if kind == 'baseline' else selection_paths[0]),
                        'formal_query_ids': manifest['short_evaluation']['query_ids'],
                        'development_policy': 'Declared reuse of historical text+threshold0 settings; no separate baseline development6 or development tuning; candidate development6 is shared as integrity evidence only.' if kind == 'baseline'
                                              else 'Uses the exact candidate config of the complete development6; no formal30-driven tuning.'})
    pair_hash = hashlib.sha256(json.dumps(results, sort_keys=True).encode('utf-8')).hexdigest()
    # All reads and comparisons complete before either exclusive write; one caller owns this pair.
    before_formal()
    for path, result in zip(selection_paths, results, strict=True):
        save_json(path, {**result, 'freeze_pair_sha256': pair_hash})
    return results


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'development-report', 'development-config', 'baseline-config', 'candidate-config',
                 'baseline-test-output', 'candidate-test-output', 'baseline-selection', 'candidate-selection'):
        parser.add_argument('--' + name, required=True)
    results = freeze(parser.parse_args())
    print(json.dumps([{'kind': row['kind'], 'config_sha256': row['config_sha256'],
                       'manifest_sha256': row['manifest_sha256'],
                       'development_failures': row['development6']['evaluation_failures']} for row in results]))


if __name__ == '__main__':
    main()
