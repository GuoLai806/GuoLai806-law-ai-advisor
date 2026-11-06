"""Grounded-output contract tests; no real model, endpoint or downloaded weights."""
import asyncio
import hashlib
import json

import httpx
import pytest

from evidence_rag.answer import LEGAL_SYSTEM, ModelStreamIncomplete, OpenAIStream, messages_for
from evidence_rag.budget import assemble

SOURCE = 'The intervention reduced inflammation in adult mice but did not improve survival.'
QUOTE = 'The intervention reduced inflammation in adult mice'
EVIDENCE = [
    {'citation_id': 'E1', 'unit_id': 'u1', 'title': 'Synthetic study', 'text': SOURCE, 'profile': 'general'},
    {'citation_id': 'E2', 'unit_id': 'u2', 'title': 'Unrelated study', 'text': 'Another intervention was tested in a different population.', 'profile': 'general'},
]
USAGE = {'prompt_tokens': 100, 'completion_tokens': 40, 'total_tokens': 140}


def payload(decision='SUPPORT', answer='The reported intervention reduced inflammation in adult mice.', evidence=None):
    return {'decision': decision, 'answer': answer,
            'evidence': [{'id': 'E1', 'quote': QUOTE}] if evidence is None else evidence}


def parse(value, evidence=EVIDENCE):
    from evidence_rag.grounding import parse_grounded_answer
    return parse_grounded_answer(value if isinstance(value, str) else json.dumps(value), evidence)


def response(text, finish='stop', done=True, usage=USAGE):
    midpoint = len(text) // 2
    chunks = [text[:midpoint], text[midpoint:]] if text else []
    packets = [{'model': 'synthetic-model', 'choices': [{'delta': {'content': chunk}, 'finish_reason': None}]}
               for chunk in chunks]
    packets.append({'choices': [{'delta': {}, 'finish_reason': finish}]})
    if usage is not None:
        packets.append({'choices': [], 'usage': usage})
    body = ''.join('data: '+json.dumps(packet)+'\n\n' for packet in packets)
    if done:
        body += 'data: [DONE]\n\n'
    return httpx.Response(200, content=body)


def provider(handler, **options):
    return OpenAIStream('http://fixture', 'synthetic-model', 'fixture-key',
                        transport=httpx.MockTransport(handler), include_usage=True,
                        answer_format='grounded_json', **options)


def test_known_quote_renders_readable_claim_and_scoped_metadata():
    rendered, metadata = parse(payload())
    assert rendered.splitlines()[0] == 'SUPPORT'
    assert QUOTE in rendered and '[E1]' in rendered and '[E2]' not in rendered
    assert not rendered.startswith('{')
    assert metadata['decision'] == 'SUPPORT'
    assert metadata['status'] == 'verbatim_quotes_checked'
    assert metadata['semantic_support'] == 'model_assessed_not_independently_verified'
    assert metadata['evidence_ids'] == ['E1']
    assert metadata['quotes'][0] == {'id': 'E1', 'quote_sha256': hashlib.sha256(QUOTE.encode()).hexdigest(),
                                     'quote_chars': len(QUOTE)}
    assert SOURCE not in json.dumps(metadata) and QUOTE not in json.dumps(metadata)


def test_whitespace_only_normalization_accepts_original_span():
    evidence = [{**EVIDENCE[0], 'text': SOURCE.replace('reduced inflammation', 'reduced\n   inflammation')}]
    rendered, _ = parse(payload(), evidence)
    assert QUOTE in rendered


def test_ordinary_answer_does_not_display_internal_answer_token():
    rendered, metadata = parse(payload(decision='ANSWER'))
    assert not rendered.startswith('ANSWER')
    assert rendered.startswith('The reported') and '[E1]' in rendered
    assert metadata['decision'] == 'ANSWER'


def test_nei_without_quote_has_reference_only_citation_and_no_supported_ids():
    rendered, metadata = parse(payload(decision='NOT_ENOUGH_INFO', answer='The supplied study does not address that claim.', evidence=[]))
    assert rendered.splitlines()[0] == 'NOT_ENOUGH_INFO'
    assert '参考资料（尚不能支持结论）' in rendered and '[E1]' in rendered
    assert metadata['evidence_ids'] == [] and metadata['quotes'] == []
    assert metadata['semantic_support'] == 'model_assessed_not_independently_verified'


@pytest.mark.parametrize('decision', ['SUPPORT', 'CONTRADICT', 'ANSWER'])
def test_non_nei_requires_a_source_quote(decision):
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected):
        parse(payload(decision=decision, evidence=[]))


@pytest.mark.parametrize('evidence', [
    [{'id': 'E1', 'quote': QUOTE.replace('reduced', 'increased')}],
    [{'id': 'E1', 'quote': 'The intervention did not reduce inflammation in adult mice'}],
    [{'id': 'E99', 'quote': QUOTE}],
    [{'id': 'E1', 'quote': 'mice'}],
    [{'id': 'E1', 'quote': 'x' * 1201}],
])
def test_fabricated_changed_unknown_or_out_of_bounds_quote_is_rejected(evidence):
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected):
        parse(payload(evidence=evidence))


@pytest.mark.parametrize('answer', ['The study supports the claim [E2].', 'See invented source [E99].', 'See malformed source [Ebad].'])
def test_inline_citation_must_be_one_of_the_selected_quote_sources(answer):
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected):
        parse(payload(answer=answer))


def test_nei_no_quote_cannot_put_unvalidated_citation_inside_answer():
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected):
        parse(payload(decision='NOT_ENOUGH_INFO', answer='No support [E1].', evidence=[]))


@pytest.mark.parametrize('raw', [
    '```json\n'+json.dumps(payload())+'\n```',
    '[]',
    '{"decision":"SUPPORT","decision":"NOT_ENOUGH_INFO","answer":"Hidden duplicate","evidence":[]}',
    '{"decision":"NOT_ENOUGH_INFO","answer":"Nested duplicate","evidence":[{"id":"E1","id":"E2","quote":"Synthetic quote"}]}',
    json.dumps({**payload(), 'unknown': True}),
    json.dumps(payload(decision='MAYBE')),
    json.dumps(payload(answer='  ')),
    json.dumps({**payload(), 'evidence': 'E1'}),
    json.dumps(payload(evidence=[{'id': 'E1', 'quote': QUOTE, 'unknown': True}])),
])
def test_invalid_schema_or_json_is_rejected_without_exposing_raw(raw):
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected) as error:
        parse(raw)
    assert raw not in str(error.value)
    assert 'Hidden duplicate' not in str(error.value)


def test_duplicate_quote_sources_and_excess_quote_count_are_rejected():
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected):
        parse(payload(evidence=[{'id': 'E1', 'quote': QUOTE}] * 2))
    many = [{**EVIDENCE[0], 'citation_id': 'E'+str(number)} for number in range(1, 6)]
    with pytest.raises(GroundingRejected):
        parse(payload(evidence=[{'id': item['citation_id'], 'quote': QUOTE} for item in many]), many)


@pytest.mark.asyncio
async def test_grounded_stream_has_one_call_and_no_raw_json_delta():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(payload()))
    events = [event async for event in provider(handler).stream_events('Synthetic claim', EVIDENCE)]
    assert len(requests) == 1
    assert len([e for e in events if e['type'] == 'delta']) == 1
    assert next(e for e in events if e['type'] == 'delta')['delta'].startswith('SUPPORT\n')
    kinds = [e['type'] for e in events]
    assert kinds.index('usage') < kinds.index('grounding_validation') < kinds.index('delta')
    assert next(e for e in events if e['type'] == 'usage')['usage'] == USAGE
    assert any(e['type'] == 'model_identity' for e in events)


@pytest.mark.asyncio
async def test_invalid_quote_preserves_known_usage_and_has_no_deliverable_delta():
    from evidence_rag.grounding import GroundingRejected
    events = []
    instance = provider(lambda request: response(json.dumps(payload(evidence=[{'id': 'E1', 'quote': 'Forged conclusion not in supplied source'}]))))
    with pytest.raises(GroundingRejected):
        async for event in instance.stream_events('claim', EVIDENCE):
            events.append(event)
    assert next(e for e in events if e['type'] == 'usage')['usage'] == USAGE
    assert not any(e['type'] in ('delta', 'grounding_validation') for e in events)


@pytest.mark.asyncio
async def test_oversized_buffer_rejects_but_retains_terminal_usage():
    from evidence_rag.grounding import GroundingRejected
    events = []
    with pytest.raises(GroundingRejected) as error:
        async for event in provider(lambda request: response('x' * 65537)).stream_events('claim', EVIDENCE):
            events.append(event)
    assert error.value.reason == 'buffer_limit'
    assert next(e for e in events if e['type'] == 'usage')['usage'] == USAGE
    assert not any(e['type'] == 'delta' for e in events)


@pytest.mark.asyncio
async def test_partial_invalid_stream_keeps_usage_and_does_not_retry_or_emit_text():
    requests = []
    def handler(request):
        requests.append(request)
        return response('{"decision":', finish='length')
    options = {'thinking': 'enabled', 'max_output_tokens': 8192, 'empty_retry_thinking': 'disabled',
               'retry_timeout': 45, 'retry_max_output_tokens': 2048}
    events = []
    with pytest.raises(ModelStreamIncomplete):
        async for event in provider(handler, **options).stream_events('claim', EVIDENCE):
            events.append(event)
    assert len(requests) == 1
    assert any(e['type'] == 'usage' for e in events)
    assert not any(e['type'] in ('delta', 'generation_retry', 'grounding_validation') for e in events)


@pytest.mark.asyncio
async def test_empty_first_attempt_retries_with_same_grounded_messages_and_retains_both_usages():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return response('', finish='length') if len(requests) == 1 else response(json.dumps(payload()))
    options = {'thinking': 'enabled', 'max_output_tokens': 8192, 'empty_retry_thinking': 'disabled',
               'retry_timeout': 45, 'retry_max_output_tokens': 2048}
    events = [e async for e in provider(handler, **options).stream_events('claim', EVIDENCE)]
    assert len(requests) == 2 and requests[0]['messages'] == requests[1]['messages']
    assert 'Output override' in requests[1]['messages'][0]['content']
    assert [e['attempt'] for e in events if e['type'] == 'usage'] == [1, 2]
    assert len([e for e in events if e['type'] == 'generation_retry']) == 1
    assert len([e for e in events if e['type'] == 'delta']) == 1


@pytest.mark.asyncio
async def test_nonempty_invalid_json_is_not_retried():
    from evidence_rag.grounding import GroundingRejected
    requests = []
    def handler(request):
        requests.append(request)
        return response('An ordinary prose answer instead of required JSON.')
    options = {'thinking': 'enabled', 'max_output_tokens': 8192, 'empty_retry_thinking': 'disabled',
               'retry_timeout': 45, 'retry_max_output_tokens': 2048}
    with pytest.raises(GroundingRejected):
        _ = [e async for e in provider(handler, **options).stream_events('claim', EVIDENCE)]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_cancelled_grounded_request_has_no_retry():
    requests = []
    def handler(request):
        requests.append(request)
        raise asyncio.CancelledError()
    options = {'thinking': 'enabled', 'max_output_tokens': 8192, 'empty_retry_thinking': 'disabled',
               'retry_timeout': 45, 'retry_max_output_tokens': 2048}
    with pytest.raises(asyncio.CancelledError):
        _ = [e async for e in provider(handler, **options).stream_events('claim', EVIDENCE)]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_legal_profile_keeps_text_even_if_grounded_was_requested(monkeypatch):
    monkeypatch.setenv('EVIDENCE_RAG_ANSWER_FORMAT', 'grounded_json')
    monkeypatch.setenv('EVIDENCE_RAG_GENERAL_SYSTEM_PROMPT', 'General custom rule')
    evidence = [{**EVIDENCE[0], 'profile': 'legal'}]
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return response('原文说明 [E1]')
    events = [e async for e in provider(handler).stream_events('法律问题', evidence)]
    assert requests[0]['messages'][0]['content'] == LEGAL_SYSTEM
    assert ''.join(e['delta'] for e in events if e['type'] == 'delta') == '原文说明 [E1]'
    assert not any(e['type'] == 'grounding_validation' for e in events)


@pytest.mark.asyncio
async def test_text_provider_unchanged_under_explicit_grounding_environment(monkeypatch):
    monkeypatch.setenv('EVIDENCE_RAG_ANSWER_FORMAT', 'grounded_json')
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return response('SUPPORT [E1]')
    instance = OpenAIStream('http://fixture', 'synthetic', 'fixture', transport=httpx.MockTransport(handler))
    events = [e async for e in instance.stream_events('claim', EVIDENCE)]
    assert ''.join(e['delta'] for e in events if e['type'] == 'delta') == 'SUPPORT [E1]'
    assert 'Output override' not in requests[0]['messages'][0]['content']


@pytest.mark.asyncio
async def test_budget_and_generation_use_identical_grounded_prompt_messages(monkeypatch):
    monkeypatch.setenv('EVIDENCE_RAG_ANSWER_FORMAT', 'grounded_json')
    monkeypatch.setenv('EVIDENCE_RAG_GENERAL_SYSTEM_PROMPT', 'First line used to require SUPPORT. Keep factual restrictions.')
    measured = []
    class Tokenizer:
        def count_chat(self, messages):
            measured.append(messages)
            return 100
    selected, budget = assemble(EVIDENCE, Tokenizer(), 6000, 'claim')
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(payload()))
    _ = [e async for e in provider(handler).stream_events('claim', selected)]
    assert budget['state'] == 'measured'
    assert measured[-1] == requests[0]['messages'] == messages_for('claim', selected, answer_format='grounded_json')
    assert 'Output override' in measured[-1][0]['content']


def test_bad_format_is_rejected_before_transport_use():
    with pytest.raises(ValueError):
        OpenAIStream('http://fixture', 'synthetic', 'fixture', answer_format='invalid')


@pytest.mark.parametrize('quote', [QUOTE.lower(), QUOTE + '.', QUOTE.replace('adult mice', 'adult humans')])
def test_verbatim_quote_does_not_normalize_case_punctuation_or_population(quote):
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected):
        parse(payload(evidence=[{'id': 'E1', 'quote': quote}]))


@pytest.mark.parametrize('evidence', [
    [], EVIDENCE + [EVIDENCE[0]], [{**EVIDENCE[0], 'citation_id': 'Ebad'}],
    [{**EVIDENCE[0], 'citation_id': 'E１'}], [{**EVIDENCE[0], 'text': None}],
    [{**EVIDENCE[0], 'title': '\ud800'}],
])
def test_malformed_source_contract_is_rejected_safely(evidence):
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected):
        parse(payload(), evidence)


@pytest.mark.parametrize('constant', ['NaN', 'Infinity', '-Infinity'])
def test_nonfinite_json_is_rejected_without_raw_model_material(constant):
    from evidence_rag.grounding import GroundingRejected
    raw = '{"decision":"SUPPORT","answer":'+constant+',"evidence":[]}'
    with pytest.raises(GroundingRejected) as error:
        parse(raw)
    assert error.value.reason == 'invalid_json'
    assert raw not in str(error.value)


@pytest.mark.parametrize('body', ['See source [E1', 'See [E1 extra]', 'See [Ebad].', 'See [E1\nnext line', 'See [E99\rnext line'])
def test_incomplete_or_malformed_bracket_id_is_rejected(body):
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected):
        parse(payload(answer=body))


def test_quote_cannot_inject_unknown_inline_source_id():
    from evidence_rag.grounding import GroundingRejected
    quote = 'The intervention reduced inflammation [E99]'
    evidence = [{**EVIDENCE[0], 'text': quote}]
    with pytest.raises(GroundingRejected) as error:
        parse(payload(evidence=[{'id': 'E1', 'quote': quote}]), evidence)
    assert error.value.reason == 'invalid_citation'


@pytest.mark.parametrize('body', ['\ud800', '\udfff'])
def test_escaped_invalid_unicode_body_raises_only_safe_error(body):
    from evidence_rag.grounding import GroundingRejected
    with pytest.raises(GroundingRejected) as error:
        parse(payload(answer=body))
    assert error.value.reason == 'invalid_encoding'


def test_standalone_parser_caps_bytes_not_character_count():
    from evidence_rag.grounding import GroundingRejected
    raw = json.dumps(payload(answer='中' * 23000), ensure_ascii=False)
    assert len(raw) < 65536 < len(raw.encode('utf-8'))
    with pytest.raises(GroundingRejected) as error:
        parse(raw)
    assert error.value.reason == 'buffer_limit'


@pytest.mark.asyncio
async def test_exact_buffer_boundary_accepts_and_one_more_byte_rejects():
    from evidence_rag.grounding import GroundingRejected
    raw = json.dumps(payload())
    boundary = raw + ' ' * (65536 - len(raw.encode('utf-8')))
    events = [e async for e in provider(lambda request: response(boundary)).stream_events('claim', EVIDENCE)]
    assert len([e for e in events if e['type'] == 'delta']) == 1
    with pytest.raises(GroundingRejected) as error:
        _ = [e async for e in provider(lambda request: response(boundary + ' ')).stream_events('claim', EVIDENCE)]
    assert error.value.reason == 'buffer_limit'


@pytest.mark.parametrize('content,reason', [(['invalid-type'], 'invalid_schema'), ('\ud800', 'invalid_encoding')])
@pytest.mark.asyncio
async def test_invalid_delta_preserves_terminal_usage_without_exposing_body(content, reason):
    from evidence_rag.grounding import GroundingRejected
    class SyntheticProvider(OpenAIStream):
        async def _stream_once(self, query, evidence):
            yield {'type': 'delta', 'delta': content}
            yield {'type': 'usage', 'usage': USAGE}
    instance = SyntheticProvider('http://fixture', 'synthetic', 'fixture', answer_format='grounded_json')
    events = []
    with pytest.raises(GroundingRejected) as error:
        async for event in instance.stream_events('claim', EVIDENCE):
            events.append(event)
    assert error.value.reason == reason
    assert events == [{'type': 'usage', 'usage': USAGE}]


@pytest.mark.parametrize('finish,done', [('stop', False), (None, True), ('content_filter', True)])
@pytest.mark.asyncio
async def test_no_deliverable_answer_without_both_success_markers(finish, done):
    events = []
    with pytest.raises(ModelStreamIncomplete):
        async for event in provider(lambda request: response(json.dumps(payload()), finish=finish, done=done)).stream_events('claim', EVIDENCE):
            events.append(event)
    assert any(e['type'] == 'usage' for e in events)
    assert not any(e['type'] in ('delta', 'grounding_validation') for e in events)


def test_legacy_format_environment_fallback_and_explicit_override(monkeypatch):
    monkeypatch.delenv('EVIDENCE_RAG_ANSWER_FORMAT', raising=False)
    monkeypatch.setenv('LEGAL_RAG_ANSWER_FORMAT', 'grounded_json')
    assert 'Output override' in messages_for('claim', EVIDENCE)[0]['content']
    assert 'Output override' not in messages_for('claim', EVIDENCE, answer_format='text')[0]['content']
    monkeypatch.setenv('EVIDENCE_RAG_ANSWER_FORMAT', 'text')
    assert 'Output override' not in messages_for('claim', EVIDENCE)[0]['content']
