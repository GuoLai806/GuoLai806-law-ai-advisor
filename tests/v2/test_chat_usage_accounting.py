"""Non-streaming chat must retain the accounting contract of retry-enabled SSE.

These tests use synthetic provider events and directly invoke the route handler.
They do not start a service, load a model, or contact a provider.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

import evidence_rag.api as api


PRIMARY_USAGE = {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
RETRY_USAGE = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}
TOTAL_USAGE = {'prompt_tokens': 200, 'completion_tokens': 70, 'total_tokens': 270}


def provider_event(kind, **data):
    return {'type': kind, **data}


def retry_events(missing_usage=None, fail_retry=False, invalid_citation=False):
    events = [
        provider_event('generation_attempt', attempt=1, thinking='enabled', timeout_seconds=60, max_output_tokens=8192),
        provider_event('model_identity', attempt=1, model='synthetic-model', system_fingerprint='synthetic-primary'),
    ]
    if missing_usage not in (1, 'both'):
        events.append(provider_event('usage', attempt=1, usage=PRIMARY_USAGE))
    events.extend([
        provider_event('generation_retry', reason='timeout', discard_pending_answer=True, attempt=1, next_attempt=2),
        provider_event('generation_attempt', attempt=2, thinking='disabled', timeout_seconds=45, max_output_tokens=2048),
        provider_event('model_identity', attempt=2, model='synthetic-model', system_fingerprint='synthetic-retry'),
    ])
    if fail_retry:
        return events
    events.append(provider_event('delta', attempt=2, delta='SUPPORT [E99]' if invalid_citation else 'SUPPORT [E1]'))
    if missing_usage not in (2, 'both'):
        events.append(provider_event('usage', attempt=2, usage=RETRY_USAGE))
    return events


class SyntheticProvider:
    def __init__(self, events, fail_after_events=False):
        self.events = events
        self.fail_after_events = fail_after_events
        self.calls = 0

    async def stream_events(self, query, evidence):
        self.calls += 1
        for item in self.events:
            yield deepcopy(item)
        if self.fail_after_events:
            raise TimeoutError('synthetic retry timeout')


class LegacyTextProvider:
    def __init__(self):
        self.calls = 0

    async def stream(self, query, evidence):
        self.calls += 1
        yield 'SUPPORT [E1]'


@pytest.fixture
def call_chat(monkeypatch):
    async def invoke(events, fail_after_events=False, generation=True, legacy_text_only=False):
        provider = LegacyTextProvider() if legacy_text_only else SyntheticProvider(events, fail_after_events=fail_after_events)
        bundle = {'trace_id': 'synthetic-trace', 'status': 'ready', 'evidence': [{'citation_id': 'E1'}],
                  'budget': {}, 'corpus_revision': 'synthetic-revision', 'query_plan': {'profile': 'general'}}

        async def get_bundle(*args, **kwargs):
            return bundle

        monkeypatch.setattr(api, 'get_bundle', get_bundle)
        monkeypatch.setattr(api, 'generation_allowed', lambda *args: generation)
        monkeypatch.setattr(api, 'save_turn', lambda *args: None)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(generator=provider)))
        body = SimpleNamespace(query='Synthetic question', session_id='synthetic-session')
        return await api.chat(body, request, 'synthetic-owner'), provider

    return invoke


def assert_metadata(response, events):
    for kind, field in (('generation_attempt', 'generation_attempts'), ('generation_retry', 'generation_retries'),
                        ('model_identity', 'observed_answer_models')):
        expected = [{key: value for key, value in item.items() if key != 'type'}
                    for item in events if item['type'] == kind]
        actual = response[field]
        assert len(actual) == len(expected)
        for received, emitted in zip(actual, expected, strict=True):
            assert all(received.get(key) == value for key, value in emitted.items())


@pytest.mark.asyncio
async def test_chat_counts_both_reported_retry_attempts(call_chat):
    events = retry_events()
    response, provider = await call_chat(events)
    assert provider.calls == 1
    assert response['status'] == 'ready'
    assert response['usage'] == TOTAL_USAGE
    assert response['usage_attempts'] == [{'attempt': 1, 'usage': PRIMARY_USAGE}, {'attempt': 2, 'usage': RETRY_USAGE}]
    assert response['usage_complete'] is True
    assert response['usage_source'] == 'provider'
    assert_metadata(response, events)


@pytest.mark.asyncio
async def test_chat_counts_only_last_cumulative_usage_within_each_attempt(call_chat):
    events = retry_events()
    usage_index = next(index for index, item in enumerate(events) if item['type'] == 'usage')
    events.insert(usage_index, provider_event('usage', attempt=1,
                                             usage={'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}))
    response, _ = await call_chat(events)
    assert response['usage'] == TOTAL_USAGE
    assert response['usage_attempts'] == [{'attempt': 1, 'usage': PRIMARY_USAGE}, {'attempt': 2, 'usage': RETRY_USAGE}]
    assert response['usage_complete'] is True


@pytest.mark.asyncio
@pytest.mark.parametrize('missing_attempt', [1, 2])
async def test_chat_marks_missing_attempt_usage_as_unknown(call_chat, missing_attempt):
    events = retry_events(missing_usage=missing_attempt)
    response, _ = await call_chat(events)
    known_attempt = 2 if missing_attempt == 1 else 1
    known_usage = RETRY_USAGE if known_attempt == 2 else PRIMARY_USAGE
    assert response['usage'] == known_usage
    assert response['usage_attempts'] == [{'attempt': known_attempt, 'usage': known_usage}]
    assert response['usage_complete'] is False
    assert response['usage_source'] == 'provider_known_attempts'
    assert_metadata(response, events)


@pytest.mark.asyncio
async def test_chat_never_substitutes_zero_for_unreported_usage(call_chat):
    events = retry_events(missing_usage='both')
    response, _ = await call_chat(events)
    assert response['usage'] is None
    assert response['usage_attempts'] == []
    assert response['usage_complete'] is False
    assert response['usage_source'] == 'unavailable'
    assert_metadata(response, events)


@pytest.mark.asyncio
async def test_chat_retains_primary_usage_and_metadata_after_retry_failure(call_chat):
    events = retry_events(fail_retry=True)
    response, _ = await call_chat(events, fail_after_events=True)
    assert response['status'] == 'evidence_only'
    assert response['usage'] == PRIMARY_USAGE
    assert response['usage_attempts'] == [{'attempt': 1, 'usage': PRIMARY_USAGE}]
    assert response['usage_complete'] is False
    assert response['usage_source'] == 'provider_known_attempts'
    assert_metadata(response, events)


@pytest.mark.asyncio
async def test_chat_retains_accounting_when_citations_are_rejected(call_chat):
    events = retry_events(invalid_citation=True)
    response, _ = await call_chat(events)
    assert response['status'] == 'insufficient_evidence'
    assert response['verification']['status'] == 'invalid'
    assert response['usage'] == TOTAL_USAGE
    assert response['usage_attempts'] == [{'attempt': 1, 'usage': PRIMARY_USAGE}, {'attempt': 2, 'usage': RETRY_USAGE}]
    assert response['usage_complete'] is True
    assert response['usage_source'] == 'provider'
    assert_metadata(response, events)


@pytest.mark.asyncio
async def test_chat_explicitly_reports_no_generation_accounting(call_chat):
    response, provider = await call_chat([], generation=False)
    assert provider.calls == 0
    assert response['status'] == 'evidence_only'
    assert response['usage'] is None
    assert response['usage_attempts'] == []
    assert response['generation_attempts'] == []
    assert response['generation_retries'] == []
    assert response['observed_answer_models'] == []
    assert response['usage_complete'] is False
    assert response['usage_source'] == 'not_generated'


@pytest.mark.asyncio
async def test_chat_legacy_events_without_attempt_numbers_use_default_attempt_one(call_chat):
    events = [
        provider_event('usage', usage={'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}),
        provider_event('delta', delta='SUPPORT [E1]'),
        provider_event('usage', usage=PRIMARY_USAGE),
    ]
    response, provider = await call_chat(events)
    assert provider.calls == 1
    assert response['status'] == 'ready'
    assert response['usage'] == PRIMARY_USAGE
    assert response['usage_attempts'] == [{'attempt': 1, 'usage': PRIMARY_USAGE}]
    assert response['usage_complete'] is True
    assert response['usage_source'] == 'provider'
    assert_metadata(response, events)


@pytest.mark.asyncio
async def test_chat_legacy_text_stream_still_completes_with_unavailable_usage(call_chat):
    response, provider = await call_chat([], legacy_text_only=True)
    assert provider.calls == 1
    assert response['status'] == 'ready'
    assert response['answer'] == 'SUPPORT [E1]'
    assert response['usage'] is None
    assert response['usage_attempts'] == []
    assert response['generation_attempts'] == []
    assert response['generation_retries'] == []
    assert response['observed_answer_models'] == []
    assert response['usage_complete'] is False
    assert response['usage_source'] == 'unavailable'
