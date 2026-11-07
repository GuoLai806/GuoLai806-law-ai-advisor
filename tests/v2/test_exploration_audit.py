"""Preserved previews, failed observations and paired claims must be auditable."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.e2e import file_hash
from evidence_rag.exploration import explore_bundle
from tools.audit_user_flow import audit, exploration_checks
from tools.compare_short_user_flow import compare, comparison_strategy, summary
from test_short_user_flow import full_scifact_fixture, run_fixture_protocol


START_NS = 1794009600_000000000


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sample(ms):
    return {'monotonic_ms': ms, 'wall_ms': ms, 'observed_wall_ms': ms,
            'wall_ns': START_NS + ms * 1_000_000,
            'utc': '2026-11-07T00:00:00.' + str(ms * 1_000_000).zfill(9) + 'Z'}


def complete_observation():
    paragraph = 'A complete source paragraph. ' * 30
    full = 'Background.\n\n' + paragraph
    locator = {'kind': 'text', 'char_start': len('Background.\n\n'), 'char_end': len(full)}
    item = {'unit_id': 'a' * 32, 'document_id': 'd1', 'title': 'Source', 'text': paragraph,
            'content_sha256': digest(paragraph), 'document_sha256': digest(full), 'part_no': 2,
            'has_more': True, 'source_locator': locator}
    preview = explore_bundle({'corpus_revision': 'r1', 'evidence': [item], 'degraded': False,
                              'degraded_reasons': [], 'retrieval_ms': 1})
    preview.setdefault('retrieval_trace', None)
    unit = {'unit_id': 'a' * 32, 'revision': 'r1', 'text': full, 'locator': {},
            'parts': [{'part_no': 2, 'text': paragraph, 'locator': locator}]}
    preview_response = json.dumps(preview, ensure_ascii=False)
    original_response = json.dumps(unit, ensure_ascii=False)
    observation = {'started': sample(0), 'preview_received': sample(10), 'original_received': sample(15),
                   'preview_http_status': 200, 'original_http_status': 200,
                   'response_bytes': len(preview_response.encode()), 'original_bytes': len(original_response.encode()),
                   'preview_response_text': preview_response, 'original_response_text': original_response,
                   'bundle': preview, 'original': unit, 'original_verified': True}
    row = {'query_id': 'q1', 'user_total_ms': 100, 'failure': None, 'user_flow_completed': True,
           'exploration': {'preview_ms': 10, 'original_ms': 15, 'response_bytes': observation['response_bytes'],
                           'original_bytes': observation['original_bytes'], 'items': 1, 'truncated': 1,
                           'original_verified': True}}
    raw = {'user_flow_strategy': 'explore_then_answer', 'exploration': observation,
           'started_wall_ns': START_NS, 'request': {'corpus_revision': 'r1'},
           'events': [{'event': 'evidence_ready', 'data': {'bundle': {'corpus_revision': 'r1', 'evidence': [item]}}}]}
    return row, raw


def test_preview_full_hash_version_and_row_fields_are_independently_checked():
    row, raw = complete_observation()
    assert all(exploration_checks(row, raw, 'explore_then_answer').values())


@pytest.mark.parametrize('mutation', ['raw_strategy', 'row_bytes', 'raw_bytes', 'preview_json', 'original_json',
                                     'full_hash', 'paragraph_hash', 'revision', 'unit_id', 'locator',
                                     'preview', 'truncated', 'pointer', 'timestamp', 'verified', 'answer_revision'])
def test_audit_rejects_forged_success_fields(mutation):
    row, raw = complete_observation()
    o = raw['exploration']
    if mutation == 'raw_strategy': raw['user_flow_strategy'] = 'direct'
    elif mutation == 'row_bytes': row['exploration']['response_bytes'] += 1
    elif mutation == 'raw_bytes': o['response_bytes'] += 1
    elif mutation == 'preview_json': o['preview_response_text'] = '{}'
    elif mutation == 'original_json': o['original_response_text'] = '{}'
    elif mutation == 'full_hash': o['bundle']['items'][0]['document_sha256'] = '0' * 64
    elif mutation == 'paragraph_hash': o['bundle']['items'][0]['content_sha256'] = '0' * 64
    elif mutation == 'revision': o['original']['revision'] = 'r2'
    elif mutation == 'unit_id': o['original']['unit_id'] = 'b' * 32
    elif mutation == 'locator': o['bundle']['items'][0]['locator'] = {}
    elif mutation == 'preview': o['bundle']['items'][0]['preview'] = 'Fake quote'
    elif mutation == 'truncated': o['bundle']['items'][0]['preview_truncated'] = False
    elif mutation == 'pointer': o['bundle']['items'][0]['read_pointer'] += '&revision=r2'
    elif mutation == 'timestamp': o['preview_received']['utc'] = sample(20)['utc']
    elif mutation == 'verified': o['original_verified'] = False
    else: raw['events'][0]['data']['bundle']['corpus_revision'] = 'r2'
    assert not all(exploration_checks(row, raw, 'explore_then_answer').values())


@pytest.mark.parametrize('stage', ['connection', 'preview503', 'original404', 'corrupt_original'])
def test_failed_attempts_keep_observed_responses_and_times_without_success(stage):
    row, raw = complete_observation()
    row.update(failure={'type': 'HTTPStatusError'}, user_flow_completed=False)
    row.pop('exploration')
    raw['events'] = []
    observation = raw['exploration']
    observation.pop('original_verified')
    if stage == 'connection':
        raw['exploration'] = {'started': sample(0)}
    elif stage == 'preview503':
        response = '{"error":"unavailable"}'
        raw['exploration'] = {'started': sample(0), 'preview_received': sample(10),
                              'preview_http_status': 503, 'preview_response_text': response,
                              'response_bytes': len(response.encode())}
    elif stage == 'original404':
        observation.pop('original')
        response = '{"detail":"source unavailable"}'
        observation.update(original_http_status=404, original_response_text=response,
                           original_bytes=len(response.encode()))
    else:
        observation['original']['text'] = 'Corrupt source'
        observation['original_response_text'] = json.dumps(observation['original'])
        observation['original_bytes'] = len(observation['original_response_text'].encode())
    assert all(exploration_checks(row, raw, 'explore_then_answer').values())
    row['user_flow_completed'] = True
    assert not all(exploration_checks(row, raw, 'explore_then_answer').values())


def test_old_direct_records_without_strategy_remain_compatible():
    assert all(exploration_checks({'failure': None}, {'events': []}).values())
    assert not all(exploration_checks({'failure': None}, {'events': [], 'exploration': {}}).values())


def grounded_runtime():
    return {'answer_format': 'grounded_json', 'generation_gate': 'grounded_evidence',
            'system_prompt_sha256': 'same', 'dense_scope': 'scope', 'rerank_threshold': '0'}


def pair_reports():
    a = {'user_flow_strategy': 'direct', 'runtime_identity': grounded_runtime(), 'freeze_pair_sha256': 'pair'}
    b = deepcopy(a)
    b['user_flow_strategy'] = 'explore_then_answer'
    return a, b


@pytest.mark.parametrize('mutation', ['prompt', 'mode', 'scope', 'pair', 'reversed', 'unknown'])
def test_compare_refuses_mixed_or_unfrozen_exploration_claims(mutation):
    a, b = pair_reports()
    if mutation == 'prompt': b['runtime_identity']['system_prompt_sha256'] = 'different'
    elif mutation == 'mode': a['runtime_identity'].update(answer_format='text', generation_gate='reranker_threshold')
    elif mutation == 'scope': b['runtime_identity']['dense_scope'] = 'different'
    elif mutation == 'pair': b['freeze_pair_sha256'] = 'different'
    elif mutation == 'reversed': a['user_flow_strategy'], b['user_flow_strategy'] = b['user_flow_strategy'], a['user_flow_strategy']
    else: b['user_flow_strategy'] = 'unknown'
    with pytest.raises(ValueError): comparison_strategy(a, b)


def test_compare_accepts_same_grounded_exploration_and_preserves_old_joint_scope():
    a, b = pair_reports()
    assert comparison_strategy(a, b) == 'exploration'
    a.pop('user_flow_strategy'); b.pop('user_flow_strategy')
    a['runtime_identity'].update(answer_format='text', generation_gate='reranker_threshold', system_prompt_sha256='text')
    assert comparison_strategy(a, b) == 'grounding'


def summary_report(tmp_path, name, strategy):
    row, raw = complete_observation()
    raw['user_flow_strategy'] = strategy
    if strategy == 'direct':
        raw['exploration'] = None
        row.pop('exploration')
    row.update(gold_label='SUPPORT', predicted_label='SUPPORT', gold_documents=['d1'], cited_documents=['d1'],
               retrieval_metrics={'ndcg_at_10': 1, 'recall_at_20': 1, 'mrr_at_10': 1},
               first_answer_wall_ms=50, stream_total_wall_ms=90, user_total_wall_ms=100, wall_clock_valid=True)
    path = tmp_path / (name + '-q1.json')
    path.write_text(json.dumps(raw))
    row['raw_sha256'] = file_hash(path)
    return {'user_flow_strategy': strategy, 'rows': [row], 'retrieval_metrics': row['retrieval_metrics'],
            'timing_health': {}, 'run_wall_ms': 100, 'usage_totals': None, 'usage_complete_queries': 0,
            'retried_queries': 0, 'generation_attempt_count': 0}


def test_summary_only_counts_checked_observed_preview_and_original_data(tmp_path):
    report = summary_report(tmp_path, 'candidate', 'explore_then_answer')
    result = summary(report, tmp_path, 'candidate')['exploration_observations']
    assert result['attempted_queries'] == result['verified_original_queries'] == 1
    assert result['metrics']['preview_ms'] == {'n': 1, 'p50': 10.0, 'p95': 10.0, 'total': 10}
    assert result['metrics']['original_read_ms']['p50'] == 5
    assert result['metrics']['response_bytes']['p50'] > 0
    assert result['metrics']['answer_evidence_ready_payload_bytes']['p50'] > 0
    assert 'not observed wire size' in result['byte_scope']
    report['rows'][0]['exploration']['preview_ms'] = -999
    with pytest.raises(ValueError, match='inconsistent'):
        summary(report, tmp_path, 'candidate')


def test_direct_summary_has_no_fake_zero_observations(tmp_path):
    report = summary_report(tmp_path, 'baseline', 'direct')
    result = summary(report, tmp_path, 'baseline')['exploration_observations']
    assert result['attempted_queries'] == 0
    assert result['metrics']['response_bytes'] == {'n': 0, 'p50': None, 'p95': None, 'total': None}


def test_summary_preserves_failed_http_observations_without_verified_original(tmp_path):
    report = summary_report(tmp_path, 'candidate', 'explore_then_answer')
    row = report['rows'][0]
    raw_path = tmp_path / 'candidate-q1.json'
    raw = json.loads(raw_path.read_text())
    raw['events'] = []
    response = '{"error":"unavailable"}'
    raw['exploration'] = {'started': sample(0), 'preview_received': sample(10),
                          'preview_http_status': 503, 'preview_response_text': response,
                          'response_bytes': len(response.encode())}
    raw_path.write_text(json.dumps(raw))
    row.update(raw_sha256=file_hash(raw_path), failure={'type': 'HTTPStatusError'}, user_flow_completed=False,
               predicted_label=None, cited_documents=[])
    row.pop('exploration')
    result = summary(report, tmp_path, 'candidate')['exploration_observations']
    assert result['verified_original_queries'] == 0
    assert result['preview_http_status_counts'] == {'503': 1}
    assert result['metrics']['preview_ms']['n'] == 1
    assert result['metrics']['response_bytes']['total'] == len(response.encode())
    assert result['metrics']['original_ms']['n'] == 0


def formal_reports(tmp_path):
    paths = []
    for name, strategy in [('baseline', 'direct'), ('candidate', 'explore_then_answer')]:
        report = summary_report(tmp_path, name, strategy)
        first_raw = json.loads((tmp_path / (name + '-q1.json')).read_text())
        initial = deepcopy(report['rows'][0])
        report['rows'] = []
        for i in range(30):
            row = deepcopy(initial)
            row['query_id'] = 'q' + str(i)
            raw_path = tmp_path / (name + '-' + row['query_id'] + '.json')
            raw_path.write_text(json.dumps(first_raw))
            row['raw_sha256'] = file_hash(raw_path)
            report['rows'].append(row)
        report.update(split='test', queries_expected=30, queries_recorded=30, source_unchanged=True,
                      runtime_identity_unchanged=True, source_sha256={'code': 'same'}, manifest_sha256='manifest',
                      runtime_identity=grounded_runtime(), freeze_pair_sha256='pair', config_sha256=name)
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(report))
        Path(str(path) + '.journal.jsonl').write_text(json.dumps({'type': 'header', 'user_flow_strategy': strategy,
                                                               'config_sha256': name}) + '\n')
        paths.append(path)
    return paths


def test_paired_output_describes_actual_exploration_path_without_joint_mode_claim(tmp_path):
    a, b = formal_reports(tmp_path)
    output = tmp_path / 'paired.json'
    compare(a, b, tmp_path, output)
    result = json.loads(output.read_text())
    assert result['comparison_strategy'] == 'exploration'
    assert 'only direct versus explore-then-answer user path differs' in result['scope']
    assert 'joint answer-format' not in result['scope']
    assert result['candidate']['exploration_observations']['verified_original_queries'] == 30
    assert result['paired']['accuracy_difference'] == 0


def test_report_header_strategy_disagreement_is_rejected(tmp_path):
    a, b = formal_reports(tmp_path)
    Path(str(b) + '.journal.jsonl').write_text(json.dumps({'type': 'header', 'user_flow_strategy': 'direct',
                                                        'config_sha256': 'candidate'}) + '\n')
    with pytest.raises(ValueError, match='Report/journal strategy'):
        compare(a, b, tmp_path, tmp_path / 'paired.json')


@pytest.mark.asyncio
@pytest.mark.parametrize('failed', [False, True])
async def test_whole_artifact_audit_retains_failed_requests_and_direct_compatibility(full_scifact_fixture, monkeypatch, failed):
    f = full_scifact_fixture
    await run_fixture_protocol(f, monkeypatch, f.short, split='validation', fail_stream=failed)
    args = SimpleNamespace(report=str(f.tmp / 'report-short.json'), config=str(f.tmp / 'config-short.json'),
                           manifest=str(f.short), claims=str(f.claims['train']), qrels=str(f.data / 'qrels/train.tsv'),
                           raw_dir=str(f.tmp / 'raw'), output=str(f.tmp / 'audit.json'), selection=None,
                           model_before=None, model_after=None, backend_before=None, backend_after=None)
    assert audit(args) == 0
    result = json.loads(Path(args.output).read_text())
    assert result['passed']
    assert result['evaluation_failures'] == (6 if failed else 0)
    assert result['required_backend_unobserved_or_mismatched_queries'] == (6 if failed else 0)
