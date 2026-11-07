"""Discovery may shorten display text, never its source identity or evidence."""
import copy
import hashlib
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from pydantic import ValidationError

from evidence_rag.exploration import ExploreBundle, explore_bundle


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def bundle(text='The complete paragraph.\n\nWith another sentence.'):
    return {
        'status': 'ready', 'corpus_revision': 'r1', 'trace_id': 'tr-fixture',
        'degraded': False, 'degraded_reasons': [], 'retrieval_ms': 17.25,
        'budget': {'state': 'not_measured'},
        'evidence': [{
            'citation_id': 'E1', 'unit_id': 'a' * 32, 'document_id': 'doc-1',
            'title': 'Complete source', 'text': text,
            'content_sha256': digest(text),
            'source_locator': {'kind': 'text', 'char_start': 0, 'char_end': len(text)},
            'profile': 'general', 'reranker_score': 4.5,
        }],
    }


def test_preview_shortens_display_without_mutating_full_source():
    source = bundle('A complete source that must never disappear.')
    before = copy.deepcopy(source)
    result = explore_bundle(source, preview_chars=10)
    item = result['items'][0]
    assert item['preview'] == 'A complete'
    assert item['preview_truncated'] is True and item['has_more'] is True
    assert item['content_sha256'] == digest(before['evidence'][0]['text'])
    assert item['document_sha256'] == item['content_sha256']
    assert item['read_pointer'] == '/api/v2/units/' + 'a' * 32 + '?revision=r1'
    assert source == before
    assert result['kind'] == 'exploration' and result['status'] == 'exploration_only'
    assert 'evidence' not in result and 'budget' not in result
    assert 'text' not in item and 'citation_id' not in item


@pytest.mark.parametrize('text,limit', [('中文😀原文', 3), ('abc', 3), ('abc', 5), ('', 1)])
def test_unicode_and_exact_boundaries(text, limit):
    item = explore_bundle(bundle(text), limit)['items'][0]
    assert item['preview'] == text[:limit]
    assert len(item['preview']) <= limit
    assert item['preview_truncated'] == (len(text) > limit)
    assert item['content_sha256'] == digest(text)


def test_complete_paragraph_keeps_distinct_document_identity_and_more_flag():
    paragraph = 'Second paragraph.'
    whole = 'First paragraph.\n\n' + paragraph
    source = bundle(paragraph)
    source['evidence'][0].update(part_no=2, document_sha256=digest(whole), has_more=True)
    item = explore_bundle(source)['items'][0]
    assert item['content_sha256'] == digest(paragraph)
    assert item['document_sha256'] == digest(whole)
    assert item['content_sha256'] != item['document_sha256']
    assert item['part_no'] == 2 and item['has_more'] and not item['preview_truncated']


def test_empty_degraded_result_is_exploration_only():
    source = bundle()
    source.update(status='insufficient_evidence', evidence=[], degraded=True,
                  degraded_reasons=['dense_unavailable'])
    result = explore_bundle(source)
    assert result['items'] == [] and result['status'] == 'exploration_only'
    assert result['degraded_reasons'] == ['dense_unavailable']
    assert result['retrieval_ms'] == 17.25


def test_read_pointer_encodes_identity_without_switching_revision():
    source = bundle()
    source['corpus_revision'] = 'revision / ?+中文'
    source['evidence'][0]['unit_id'] = 'unit/with?query#and space'
    result = explore_bundle(source)
    item = result['items'][0]
    parsed = urlsplit(item['read_pointer'])
    assert unquote(parsed.path.removeprefix('/api/v2/units/')) == source['evidence'][0]['unit_id']
    assert parse_qs(parsed.query) == {'revision': [source['corpus_revision']]}
    assert parsed.fragment == '' and item['revision'] == result['corpus_revision']


@pytest.mark.parametrize('limit', [0, -1, 2001, 3.0, True, None, '3'])
def test_invalid_preview_limit_is_rejected(limit):
    with pytest.raises(ValueError, match='preview_chars'):
        explore_bundle(bundle(), limit)


def test_tampered_complete_source_hash_is_rejected_before_preview():
    source = bundle()
    source['evidence'][0]['text'] = 'altered'
    with pytest.raises(ValueError, match='source hash mismatch'):
        explore_bundle(source, 1)


def test_paragraph_cannot_masquerade_as_complete_document():
    source = bundle()
    source['evidence'][0].update(part_no=2)
    with pytest.raises(ValueError, match='document hash'):
        explore_bundle(source)


def test_legal_whole_article_uses_article_identity():
    source = bundle('第一条 完整法条。')
    source['evidence'][0].update(profile='legal', document_id=None,
                                 article_label='第一条', source_locator={'pages': [1]})
    result = explore_bundle(source)
    item = result['items'][0]
    assert item['document_sha256'] == item['content_sha256']
    assert item['locator'] == {'pages': [1]}
    assert item['part_no'] is None and item['document_id'] is None


def test_model_cannot_accept_ready_or_evidence_payload():
    result = explore_bundle(bundle())
    with pytest.raises(ValidationError):
        ExploreBundle.model_validate({**result, 'status': 'ready'})
    with pytest.raises(ValidationError):
        ExploreBundle.model_validate({**result, 'evidence': bundle()['evidence']})


def test_trace_passes_only_when_requested_without_full_source_fields():
    source = bundle()
    assert 'retrieval_trace' not in explore_bundle(source)
    source['retrieval_trace'] = {'dense_backend': 'openviking', 'timings_ms': {'dense_ms': 5}}
    assert explore_bundle(source)['retrieval_trace'] == source['retrieval_trace']


@pytest.mark.asyncio
async def test_catalog_paragraph_preview_keeps_full_pinned_source(tmp_path):
    from evidence_rag.catalog import Catalog
    from evidence_rag.retrieval import RetrievalService

    paragraph = 'Ocean currents move heat. ' * 80
    full = 'Background observation.\n\n' + paragraph
    catalog = Catalog(tmp_path / 'catalog.sqlite3')
    try:
        catalog.stage_documents(
            [{'_id': 'ocean', 'title': 'Ocean currents', 'text': full}],
            {'corpus_id': 'fixture', 'version_id': 'v1',
             'source_kind': 'synthetic', 'verification_state': 'provided'},
            revision='r1',
        )
        catalog.activate('r1', None)
        ranked = await RetrievalService(catalog).retrieve('Ocean currents move heat')
        item = explore_bundle(ranked, 40)['items'][0]
        original = catalog.get('r1', item['unit_id'])
        part = next(part for part in original['parts'] if part['part_no'] == item['part_no'])
        assert item['preview'] == part['text'][:40]
        assert item['preview_truncated'] and item['has_more']
        assert item['content_sha256'] == digest(part['text'])
        assert item['document_sha256'] == digest(original['text'])
        assert original['text'] == full
        assert item['revision'] == 'r1'
    finally:
        catalog.db.close()
