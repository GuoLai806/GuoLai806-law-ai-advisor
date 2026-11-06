"""Strict grounded-output parsing; verbatim presence is not semantic verification."""
import hashlib
import json
import re

MAX_BUFFER_BYTES = 64 * 1024
MAX_QUOTES = 4
MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 1200
DECISIONS = frozenset(('SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO', 'ANSWER'))
_SOURCE_ID = re.compile(r'E[0-9]+\Z')
_INLINE_SOURCE = re.compile(r'\[E[^\]\r\n]*(?:\]|(?=[\r\n]|\Z))')
_REASONS = frozenset(('invalid_json', 'invalid_schema', 'invalid_source',
                      'invalid_quote', 'invalid_citation', 'invalid_encoding',
                      'buffer_limit'))


class GroundingRejected(RuntimeError):
    """Public errors contain only a fixed reason, never output or source material."""
    def __init__(self, reason):
        self.reason = reason if reason in _REASONS else 'invalid_schema'
        super().__init__('grounded answer rejected: ' + self.reason)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GroundingRejected('invalid_json')
        result[key] = value
    return result


def _reject_constant(_value):
    raise GroundingRejected('invalid_json')


def _normalize(value):
    return ' '.join(value.split())


def _check_encoding(value):
    try:
        return value.encode('utf-8')
    except UnicodeError:
        raise GroundingRejected('invalid_encoding') from None


def _check_inline_sources(text, selected_ids):
    # Also reject malformed citation-looking tokens instead of ignoring them.
    for match in _INLINE_SOURCE.finditer(text):
        token = match.group()
        if not token.endswith(']') or token[1:-1] not in selected_ids:
            raise GroundingRejected('invalid_citation')


def parse_grounded_answer(payload, evidence):
    """Return readable text and safe, quote-hash-only validation metadata.

    The model assesses the claim's meaning. This gate proves that each selected
    quote occurs in its source and that all displayed evidence IDs are valid.
    """
    if not isinstance(payload, str):
        raise GroundingRejected('invalid_json')
    if len(_check_encoding(payload)) > MAX_BUFFER_BYTES:
        raise GroundingRejected('buffer_limit')
    try:
        answer = json.loads(payload, object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except GroundingRejected:
        raise
    except (ValueError, TypeError, RecursionError):
        raise GroundingRejected('invalid_json') from None
    if not isinstance(answer, dict) or set(answer) != {'decision', 'answer', 'evidence'}:
        raise GroundingRejected('invalid_schema')
    decision, body, quotes = answer['decision'], answer['answer'], answer['evidence']
    if not isinstance(decision, str) or decision not in DECISIONS:
        raise GroundingRejected('invalid_schema')
    if not isinstance(body, str) or not body.strip() or not isinstance(quotes, list):
        raise GroundingRejected('invalid_schema')
    _check_encoding(body)
    if not isinstance(evidence, list) or not evidence:
        raise GroundingRejected('invalid_source')
    sources = {}
    for source in evidence:
        if not isinstance(source, dict):
            raise GroundingRejected('invalid_source')
        source_id, title, text = source.get('citation_id'), source.get('title'), source.get('text')
        if (not isinstance(source_id, str) or not _SOURCE_ID.fullmatch(source_id)
                or source_id in sources or not isinstance(title, str) or not isinstance(text, str)):
            raise GroundingRejected('invalid_source')
        material = title + '\n' + text
        _check_encoding(material)
        sources[source_id] = _normalize(material)
    if len(quotes) > min(MAX_QUOTES, len(sources)) or (decision != 'NOT_ENOUGH_INFO' and not quotes):
        raise GroundingRejected('invalid_quote')
    selected_ids, validated_quotes, quote_metadata = [], [], []
    for entry in quotes:
        if not isinstance(entry, dict) or set(entry) != {'id', 'quote'}:
            raise GroundingRejected('invalid_schema')
        source_id, raw_quote = entry['id'], entry['quote']
        if not isinstance(source_id, str) or source_id not in sources or source_id in selected_ids:
            raise GroundingRejected('invalid_source')
        if not isinstance(raw_quote, str):
            raise GroundingRejected('invalid_quote')
        _check_encoding(raw_quote)
        quote = _normalize(raw_quote)
        if not MIN_QUOTE_CHARS <= len(quote) <= MAX_QUOTE_CHARS or quote not in sources[source_id]:
            raise GroundingRejected('invalid_quote')
        selected_ids.append(source_id)
        validated_quotes.append((source_id, quote))
        quote_metadata.append({'id': source_id, 'quote_sha256': hashlib.sha256(quote.encode('utf-8')).hexdigest(),
                               'quote_chars': len(quote)})
    _check_inline_sources(body, selected_ids)
    for _source_id, quote in validated_quotes:
        _check_inline_sources(quote, selected_ids)
    lines = [] if decision == 'ANSWER' else [decision]
    lines.append(body.strip())
    if validated_quotes:
        lines.append('\n证据原文：')
        lines.extend(f'[{source_id}] “{quote}”' for source_id, quote in validated_quotes)
    else:
        lines.append('\n参考资料（尚不能支持结论）：[' + next(iter(sources)) + ']')
    return '\n'.join(lines), {
        'decision': decision,
        'status': 'verbatim_quotes_checked',
        'semantic_support': 'model_assessed_not_independently_verified',
        'evidence_ids': selected_ids,
        'quotes': quote_metadata,
    }
