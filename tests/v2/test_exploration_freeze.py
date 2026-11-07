"""An exploration comparison may change only its declared user path."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.e2e import file_hash
from benchmarks.short_protocol import validate_selection
from test_freeze_short_run import freeze_inputs, replace_json, write_journal
from test_short_user_flow import full_scifact_fixture
from tools.freeze_short_run import freeze, normalized_config


def write(path, value):
    Path(path).write_text(json.dumps(value))


def read(path):
    return json.loads(Path(path).read_text())


def update_development(args):
    config = read(args.candidate_config)
    write(args.development_config, config)
    report = read(args.development_report)
    report.update(config_sha256=file_hash(args.development_config),
                  runtime_identity=config['runtime_identity'], user_flow_strategy=config['user_flow_strategy'])
    write(args.development_report, report)
    write_journal(args)
    journal_path = Path(args.development_report + '.journal.jsonl')
    entries = [json.loads(line) for line in journal_path.read_text().splitlines()]
    entries[0]['user_flow_strategy'] = 'explore_then_answer'
    journal_path.write_text(''.join(json.dumps(entry) + '\n' for entry in entries))


@pytest.fixture
def exploration_inputs(freeze_inputs):
    args = freeze_inputs
    args.comparison_strategy = 'exploration'
    candidate = read(args.candidate_config)
    candidate['user_flow_strategy'] = 'explore_then_answer'
    baseline = deepcopy(candidate)
    baseline['user_flow_strategy'] = 'direct'
    write(args.candidate_config, candidate)
    write(args.baseline_config, baseline)
    update_development(args)
    return args


def validate(args, kind='baseline', **kw):
    return validate_selection(getattr(args, kind + '_selection'), getattr(args, kind + '_config'),
                              args.manifest, getattr(args, kind + '_test_output'), **kw)


def rehash_pair(args, change=lambda pair: None):
    paths = (args.baseline_selection, args.candidate_selection)
    pair = [read(path) for path in paths]
    change(pair)
    for item in pair:
        item.pop('freeze_pair_sha256', None)
    digest = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
    for path, item in zip(paths, pair, strict=True):
        write(path, {**item, 'freeze_pair_sha256': digest})


def test_exploration_freezes_same_grounded_service_with_two_explicit_paths(exploration_inputs):
    args = exploration_inputs
    freeze(args)
    for kind, strategy in [('baseline', 'direct'), ('candidate', 'explore_then_answer')]:
        selected = read(getattr(args, kind + '_selection'))
        assert selected['comparison_strategy'] == 'exploration'
        assert selected['user_flow_strategy'] == strategy
        assert selected['runtime_identity']['answer_format'] == 'grounded_json'
        assert selected['runtime_identity']['generation_gate'] == 'grounded_evidence'
        assert selected['development6']['evaluation_failures'] == 1
        assert 'Identical grounded answer service/runtime' in selected['limits']
        assert 'no formal30-driven tuning' in read(args.candidate_selection)['development_policy']
        assert validate(args, kind)['freeze_pair_sha256'] == selected['freeze_pair_sha256']


def test_normalization_removes_only_explicit_exploration_strategy(exploration_inputs):
    original = read(exploration_inputs.candidate_config)
    normalized = normalized_config(original, 'exploration')
    assert normalized == {key: value for key, value in original.items() if key != 'user_flow_strategy'}
    assert normalized['service_configuration']['general_system_prompt'] == 'Fixture shared prompt'
    assert normalized['runtime_identity']['system_prompt_sha256'] == 'fixture-grounded-contract'
    assert original['user_flow_strategy'] == 'explore_then_answer'


@pytest.mark.parametrize('mutation', ['prompt', 'prompt_hash', 'model', 'budget', 'runtime', 'path', 'mode',
                                     'dev_report_path', 'dev_journal_path', 'unknown_strategy', 'implicit_strategy'])
def test_freeze_refuses_extra_changes_or_wrong_declared_modes(exploration_inputs, mutation):
    args = exploration_inputs
    if mutation == 'prompt':
        replace_json(args.baseline_config, lambda c: c['service_configuration'].update(general_system_prompt='different'))
    elif mutation == 'prompt_hash':
        replace_json(args.baseline_config, lambda c: c['runtime_identity'].update(system_prompt_sha256='different'))
    elif mutation == 'model':
        replace_json(args.baseline_config, lambda c: c['service_configuration'].update(model_temperature=1))
    elif mutation == 'budget':
        replace_json(args.baseline_config, lambda c: c.update(max_evidence_items=8))
    elif mutation == 'runtime':
        replace_json(args.baseline_config, lambda c: c['runtime_identity'].update(dense_scope='different'))
    elif mutation == 'path':
        replace_json(args.baseline_config, lambda c: c.update(user_flow_strategy='explore_then_answer'))
    elif mutation == 'mode':
        for path in (args.baseline_config, args.candidate_config):
            replace_json(path, lambda c: c['service_configuration'].update(answer_format='text', generation_gate='reranker_threshold'))
            replace_json(path, lambda c: c['runtime_identity'].update(answer_format='text', generation_gate='reranker_threshold'))
        update_development(args)
    elif mutation == 'dev_report_path':
        replace_json(args.development_report, lambda r: r.update(user_flow_strategy='direct'))
        write_journal(args)
    elif mutation == 'dev_journal_path':
        path = Path(args.development_report + '.journal.jsonl')
        path.write_text(path.read_text().replace('explore_then_answer', 'direct'))
    elif mutation == 'unknown_strategy':
        args.comparison_strategy = 'unknown'
    else:
        del args.comparison_strategy
    with pytest.raises(ValueError):
        freeze(args)
    assert not Path(args.baseline_selection).exists() and not Path(args.candidate_selection).exists()


@pytest.mark.parametrize('mutation', ['prompt', 'mode', 'path', 'comparison_strategy'])
def test_validation_rejects_reforged_pair_with_semantically_invalid_change(exploration_inputs, mutation):
    args = exploration_inputs
    freeze(args)
    if mutation == 'comparison_strategy':
        rehash_pair(args, lambda pair: [item.update(comparison_strategy='grounding') for item in pair])
    else:
        config = read(args.baseline_config)
        if mutation == 'prompt':
            config['service_configuration']['general_system_prompt'] = 'changed'
        elif mutation == 'mode':
            config['service_configuration'].update(answer_format='text', generation_gate='reranker_threshold')
            config['runtime_identity'].update(answer_format='text', generation_gate='reranker_threshold')
        else:
            config['user_flow_strategy'] = 'explore_then_answer'
        write(args.baseline_config, config)
        rehash_pair(args, lambda pair: pair[0].update(config_sha256=file_hash(args.baseline_config),
                                                     runtime_identity=config['runtime_identity'],
                                                     user_flow_strategy=config['user_flow_strategy']))
    with pytest.raises(ValueError):
        validate(args)


def test_validation_rejects_disagreeing_explicit_pair_strategies(exploration_inputs):
    args = exploration_inputs
    freeze(args)
    rehash_pair(args, lambda pair: pair[1].update(comparison_strategy='grounding'))
    with pytest.raises(ValueError, match='comparison strategy'):
        validate(args)


def test_unknown_comparison_strategy_rejected_even_with_matching_pair_hash(exploration_inputs):
    args = exploration_inputs
    freeze(args)
    rehash_pair(args, lambda pair: [item.update(comparison_strategy='unknown') for item in pair])
    with pytest.raises(ValueError, match='Unknown comparison strategy'):
        validate(args)


def test_old_grounding_default_still_requires_historical_modes(freeze_inputs):
    args = freeze_inputs
    freeze(args)
    assert read(args.baseline_selection)['comparison_strategy'] == 'grounding'
    assert validate(args)['freeze_pair_sha256']
    # Removing the newly explicit metadata preserves the default for old pairs.
    rehash_pair(args, lambda pair: [item.pop('comparison_strategy') for item in pair])
    assert validate(args)['freeze_pair_sha256']


def test_exploration_cannot_replay_formal_record(exploration_inputs):
    args = exploration_inputs
    freeze(args)
    Path(args.baseline_test_output).write_text('{}')
    with pytest.raises(ValueError, match='Preserve earlier formal record'):
        validate(args)


def test_exploration_formal_journal_must_record_bound_path(exploration_inputs):
    args = exploration_inputs
    freeze(args)
    selection = read(args.baseline_selection)
    header = {'type': 'header', 'started_utc': selection['selected_utc'],
              'source_sha256': selection['source_sha256'], 'manifest_sha256': selection['manifest_sha256'],
              'config_sha256': selection['config_sha256'], 'query_ids': selection['formal_query_ids'],
              'freeze_pair_sha256': selection['freeze_pair_sha256'], 'user_flow_strategy': 'explore_then_answer'}
    Path(args.baseline_test_output + '.journal.jsonl').write_text(json.dumps(header) + '\n')
    with pytest.raises(ValueError, match='Formal journal exploration path changed'):
        validate(args, audit=True)
