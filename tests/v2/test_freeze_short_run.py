from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.e2e import file_hash, source_hashes
from test_short_user_flow import full_scifact_fixture
from tools.freeze_short_run import freeze


@pytest.fixture
def freeze_inputs(full_scifact_fixture):
    f = full_scifact_fixture
    manifest = json.loads(f.short.read_text())
    candidate = {'source_sha256': source_hashes(),
                 'claims_sha256': {'validation': file_hash(f.claims['train']), 'test': file_hash(f.claims['test'])},
                 'max_evidence_items': 4,
                 'service_configuration': {'answer_format': 'grounded_json', 'generation_gate': 'grounded_evidence',
                                           'rerank_threshold': 0, 'general_system_prompt': 'Fixture shared prompt', 'model_temperature': 0},
                 'runtime_identity': {'answer_format': 'grounded_json', 'generation_gate': 'grounded_evidence',
                                      'rerank_threshold': '0', 'system_prompt_sha256': 'fixture-grounded-contract',
                                      'dense_scope': 'viking://resources/fixture/r1/'}}
    baseline = deepcopy(candidate)
    baseline['service_configuration'].update(answer_format='text', generation_gate='reranker_threshold')
    baseline['runtime_identity'].update(answer_format='text', generation_gate='reranker_threshold', system_prompt_sha256='fixture-text-contract')
    paths = {name: f.tmp / (name + '.json') for name in
             ('development_report', 'development_config', 'baseline_config', 'candidate_config',
              'baseline_test_output', 'candidate_test_output', 'baseline_selection', 'candidate_selection')}
    for key, value in [('development_config', candidate), ('candidate_config', candidate), ('baseline_config', baseline)]:
        paths[key].write_text(json.dumps(value))
    claims = {str(row['id']): row for row in map(json.loads, f.claims['train'].read_text().splitlines())}
    from benchmarks.verification_metrics import gold_label
    rows = [{'query_id': identifier, 'gold_label': gold_label(claims[identifier]),
             'failure': {'type': 'ProviderError', 'detail': 'Fixture failure'} if index == 0 else None,
             'user_flow_completed': index != 0} for index, identifier in enumerate(manifest['validation']['query_ids'])]
    report = {'split': 'validation', 'queries_expected': 6, 'queries_recorded': 6, 'rows': rows,
              'source_identity_checked': True, 'source_unchanged': True, 'runtime_identity_unchanged': True,
              'source_sha256': source_hashes(), 'manifest_sha256': file_hash(f.short),
              'config_sha256': file_hash(paths['development_config']), 'runtime_identity': candidate['runtime_identity'], 'failures': 1}
    paths['development_report'].write_text(json.dumps(report))
    args = SimpleNamespace(manifest=str(f.short), **{key: str(value) for key, value in paths.items()})
    write_journal(args)
    return args


def write_journal(args):
    from pathlib import Path
    path = Path(args.development_report)
    report = json.loads(path.read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    entries = [{'type': 'header', 'query_ids': manifest['validation']['query_ids'],
                **{key: report[key] for key in ('source_sha256', 'config_sha256', 'manifest_sha256')}},
               *[{'type': 'query', 'completed': index, 'row': row} for index, row in enumerate(report['rows'], 1)],
               {'type': 'complete', 'report_sha256': file_hash(path), 'queries': len(report['rows']), 'failures': report['failures']}]
    Path(str(path) + '.journal.jsonl').write_text(''.join(json.dumps(entry) + '\n' for entry in entries))


def replace_json(path, change):
    from pathlib import Path
    path = Path(path)
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_freeze_binds_both_configs_and_retains_real_development_failure(freeze_inputs):
    from pathlib import Path
    args = freeze_inputs
    results = freeze(args)
    assert [row['kind'] for row in results] == ['baseline', 'candidate']
    stored = [json.loads(Path(path).read_text()) for path in (args.baseline_selection, args.candidate_selection)]
    assert stored[0]['freeze_pair_sha256'] == stored[1]['freeze_pair_sha256']
    unhashed = [{key: value for key, value in row.items() if key != 'freeze_pair_sha256'} for row in stored]
    assert stored[0]['freeze_pair_sha256'] == hashlib.sha256(json.dumps(unhashed, sort_keys=True).encode('utf-8')).hexdigest()
    assert stored[0]['paired_selection_path'] == str(Path(args.candidate_selection).resolve())
    assert stored[1]['paired_selection_path'] == str(Path(args.baseline_selection).resolve())
    for result in stored:
        assert result['manifest_sha256'] == file_hash(args.manifest)
        assert result['source_sha256'] == source_hashes()
        assert result['development6']['queries_recorded'] == 6
        assert result['development6']['evaluation_failures'] == 1
        assert result['development6']['journal_sha256'] == file_hash(args.development_report + '.journal.jsonl')
        assert result['config_sha256'] == file_hash(result['config_path'])
    assert 'no separate baseline development6' in stored[0]['development_policy']
    assert 'exact candidate config' in stored[1]['development_policy']
    with pytest.raises(FileExistsError):
        freeze(args)


@pytest.mark.parametrize('path_key,suffix', [('baseline_test_output', ''), ('candidate_test_output', ''),
                                           ('baseline_test_output', '.journal.jsonl'), ('candidate_test_output', '.journal.jsonl')])
def test_freeze_refuses_either_formal_report_or_journal(freeze_inputs, path_key, suffix):
    from pathlib import Path
    args = freeze_inputs
    Path(getattr(args, path_key) + suffix).write_text('Preserved earlier formal record')
    with pytest.raises(FileExistsError):
        freeze(args)
    assert not Path(args.baseline_selection).exists()
    assert not Path(args.candidate_selection).exists()


@pytest.mark.parametrize('mutation', ['incomplete', 'membership', 'labels', 'source', 'source_flag', 'runtime_flag',
                                    'runtime_identity', 'manifest_identity', 'config_identity', 'failure_count',
                                    'candidate_changed', 'baseline_extra_change', 'baseline_mode', 'candidate_mode', 'threshold',
                                    'journal_incomplete', 'journal_changed', 'journal_missing'])
def test_freeze_refuses_inconsistent_development_or_configs(freeze_inputs, mutation):
    from pathlib import Path
    args = freeze_inputs
    changes = {
        'incomplete': lambda value: value.update(queries_recorded=5),
        'membership': lambda value: value['rows'][0].update(query_id='outside-manifest'),
        'labels': lambda value: value['rows'][0].update(gold_label='INVALID'),
        'source': lambda value: value.update(source_sha256={}),
        'source_flag': lambda value: value.update(source_unchanged=False),
        'runtime_flag': lambda value: value.update(runtime_identity_unchanged=False),
        'runtime_identity': lambda value: value['runtime_identity'].update(dense_scope='wrong-scope'),
        'manifest_identity': lambda value: value.update(manifest_sha256='wrong'),
        'config_identity': lambda value: value.update(config_sha256='wrong'),
        'failure_count': lambda value: value.update(failures=0),
    }
    if mutation in changes:
        replace_json(args.development_report, changes[mutation])
        write_journal(args)
    elif mutation == 'candidate_changed':
        replace_json(args.candidate_config, lambda value: value.update(max_evidence_items=8))
    elif mutation == 'baseline_extra_change':
        replace_json(args.baseline_config, lambda value: value['service_configuration'].update(model_temperature=1))
    elif mutation == 'baseline_mode':
        replace_json(args.baseline_config, lambda value: value['service_configuration'].update(answer_format='grounded_json'))
    elif mutation == 'candidate_mode':
        for path in (args.candidate_config, args.development_config):
            replace_json(path, lambda value: value['service_configuration'].update(generation_gate='unknown'))
        replace_json(args.development_report, lambda value: value.update(config_sha256=file_hash(args.development_config)))
        write_journal(args)
    elif mutation == 'threshold':
        for path in (args.baseline_config, args.candidate_config, args.development_config):
            replace_json(path, lambda value: value['service_configuration'].update(rerank_threshold=-1))
        replace_json(args.development_report, lambda value: value.update(config_sha256=file_hash(args.development_config)))
        write_journal(args)
    elif mutation == 'journal_incomplete':
        Path(args.development_report + '.journal.jsonl').write_text('{}\n')
    elif mutation == 'journal_changed':
        journal = Path(args.development_report + '.journal.jsonl')
        journal.write_text(journal.read_text().replace('ProviderError', 'ChangedFailure'))
    else:
        Path(args.development_report + '.journal.jsonl').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        freeze(args)
    assert not Path(args.baseline_selection).exists()
    assert not Path(args.candidate_selection).exists()


def test_freeze_rejects_output_aliasing_before_writes(freeze_inputs):
    args = freeze_inputs
    args.candidate_selection = args.baseline_selection
    with pytest.raises(ValueError, match='distinct'):
        freeze(args)
