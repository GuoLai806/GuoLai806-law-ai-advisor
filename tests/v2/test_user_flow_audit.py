"""The auditor must detect drift between original SSE and scored report fields."""
from copy import deepcopy

import pytest

from tools.audit_user_flow import generation_from_raw, scoring_and_usage_checks, verdict_from_raw_text


def event(kind, data, received_ms=1):
    return {'event': kind, 'data': data, 'received_ms': received_ms}


def raw_answer(text='SUPPORT\nA source supports the claim [E1].'):
    return {'events': [
        event('evidence_ready', {'bundle': {'evidence': [
            {'citation_id': 'E1', 'document_id': 'd1'}, {'citation_id': 'E2', 'document_id': 'd2'}]}}),
        event('generation_attempt', {'attempt': 1, 'thinking': 'enabled'}),
        event('model_identity', {'model': 'fixture', 'attempt': 1}),
        event('answer_delta', {'delta': text}),
        event('usage', {'attempt': 1, 'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14}}),
    ]}


def scored_row(raw, prediction='SUPPORT', cited_documents=None, failure=None):
    return {'predicted_label': prediction, 'cited_documents': ['d1'] if cited_documents is None else cited_documents,
            'failure': failure, **generation_from_raw(raw['events'])}


@pytest.mark.parametrize(('text', 'expected'), [
    ('SUPPORT\nExplanation [E1]', 'SUPPORT'),
    ('### 最终判定：**CONTRADICT** [E1]', 'CONTRADICT'),
    ('**Final verdict**: **NOT_ENOUGH_INFO** [E1]', 'NOT_ENOUGH_INFO'),
    ('  \n最终结论：SUPPORT [E1]', 'SUPPORT'),
    ('The explanation mentions SUPPORT [E1]', None),
    ('Answer\nSUPPORT [E1]', None),
    ('SUPPORTING [E1]', None),
    ('**Final verdict** SUPPORT [E1]', None),
    ('', None),
])
def test_independent_first_line_grammar(text, expected):
    assert verdict_from_raw_text(text) == expected


def test_correct_row_and_tampered_prediction_or_citation_are_distinguished():
    raw = raw_answer()
    row = scored_row(raw)
    checks = scoring_and_usage_checks(row, raw)
    assert all(value for key, value in checks.items() if key != 'generation_field_mismatches')
    tampered = dict(row, predicted_label='CONTRADICT', cited_documents=['d2'])
    checks = scoring_and_usage_checks(tampered, raw)
    assert not checks['prediction_matches_raw']
    assert not checks['cited_documents_match_raw']


def test_format_failure_can_have_completed_flow_without_a_scored_prediction():
    raw = raw_answer('Evidence appears relevant [E1].')
    row = scored_row(raw, prediction=None, cited_documents=[], failure={'type': 'invalid_answer_format'})
    row['user_flow_completed'] = True
    checks = scoring_and_usage_checks(row, raw)
    assert checks['prediction_matches_raw']
    assert checks['cited_documents_match_raw']
    assert checks['format_failure_consistent']
    raw['events'][3]['data']['delta'] = 'SUPPORT [E1]'
    assert not scoring_and_usage_checks(row, raw)['format_failure_consistent']


def test_evidence_only_has_no_generation_or_usage_and_remains_failed():
    raw = {'events': [event('evidence_ready', {'bundle': {'evidence': []}}),
                      event('completed', {'status': 'evidence_only'})]}
    row = scored_row(raw, prediction=None, cited_documents=[], failure={'type': 'answer_not_completed'})
    checks = scoring_and_usage_checks(row, raw)
    assert checks['generation_metadata_matches_raw']
    assert checks['prediction_matches_raw']
    assert row['usage'] is None and row['usage_complete'] is False
    assert row['generation_attempts'] == []


def test_retry_uses_last_cumulative_usage_per_attempt_and_preserves_unknown_attempts():
    events = [
        event('generation_attempt', {'attempt': 1}),
        event('usage', {'attempt': 1, 'usage': {'total_tokens': 10}}),
        event('usage', {'attempt': 1, 'usage': {'total_tokens': 15}}),
        event('generation_retry', {'attempt': 1, 'next_attempt': 2}),
        event('generation_attempt', {'attempt': 2}),
        event('usage', {'attempt': 2, 'usage': {'total_tokens': 20}}),
    ]
    rebuilt = generation_from_raw(events)
    assert rebuilt['usage'] == {'total_tokens': 35}
    assert rebuilt['usage_attempts'] == [{'attempt': 1, 'usage': {'total_tokens': 15}},
                                         {'attempt': 2, 'usage': {'total_tokens': 20}}]
    assert rebuilt['usage_complete']
    rebuilt = generation_from_raw(events[:-1])
    assert rebuilt['usage'] == {'total_tokens': 15}
    assert not rebuilt['usage_complete']
    assert len(rebuilt['generation_attempts']) == 2


def test_modified_usage_attempt_model_or_retry_metadata_is_detected():
    raw = raw_answer()
    row = scored_row(raw)
    tampered = deepcopy(row)
    tampered['usage']['total_tokens'] = 999
    tampered['observed_answer_models'] = []
    tampered['generation_retries'] = [{'attempt': 1, 'next_attempt': 2}]
    checks = scoring_and_usage_checks(tampered, raw)
    assert not checks['generation_metadata_matches_raw']
    assert set(checks['generation_field_mismatches']) == {'usage', 'observed_answer_models', 'generation_retries'}
