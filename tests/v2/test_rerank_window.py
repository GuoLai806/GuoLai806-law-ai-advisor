from types import SimpleNamespace

import pytest

from evidence_rag.retrieval import RetrievalService, rerank_candidates
from evidence_rag.api import general_rerank_top_n, generation_allowed, model_capabilities


class ReverseReranker:
    def __init__(self): self.seen = []
    def rank(self, query, candidates):
        self.seen = [c['unit_id'] for c in candidates]
        return [({**c, 'rerank_part_no': 2}, float(i)) for i, c in enumerate(reversed(candidates))]


def test_window_preserves_exact_head_tail_order_and_scores():
    rows = [{'unit_id': str(i)} for i in range(30)]
    ranker = ReverseReranker()
    result = rerank_candidates({'candidates': rows}, 'q', ['0', '1'], ranker, 24)
    assert ranker.seen == [str(i) for i in range(24)]
    assert [c['unit_id'] for c in result['candidates']] == ['0', '1'] + [str(i) for i in range(23, 1, -1)] + [str(i) for i in range(24, 30)]
    assert set(result['reranker_scores']) == {str(i) for i in range(24)}
    assert result['candidates'][2]['rerank_part_no'] == 2
    assert all('rerank_part_no' not in row for row in result['candidates'][24:])


def test_default_scores_all_candidates_and_service_legal_does_not_change_config():
    rows = [{'unit_id': str(i)} for i in range(30)]
    ranker = ReverseReranker()
    result = rerank_candidates({'candidates': rows}, 'q', [], ranker)
    assert len(result['reranker_scores']) == len(ranker.seen) == 30
    assert RetrievalService(None, profile='legal', general_rerank_top_n=24).general_rerank_top_n == 24


@pytest.mark.parametrize('value', ['0', '61', 'bad'])
def test_invalid_window_rejected(monkeypatch, value):
    monkeypatch.setenv('EVIDENCE_RAG_GENERAL_RERANK_TOP_N', value)
    with pytest.raises(ValueError): general_rerank_top_n()


def test_identity_and_tail_cannot_enable_generation(monkeypatch):
    monkeypatch.setenv('EVIDENCE_RAG_GENERAL_RERANK_TOP_N', '24')
    service = SimpleNamespace(dense=None, reranker=None, tokenizer=None, rerank_timeout=120, general_rerank_top_n=general_rerank_top_n())
    generator = SimpleNamespace(max_output_tokens=1, answer_format='grounded_json')
    app = SimpleNamespace(state=SimpleNamespace(service=service, generator=generator, default_profile='general'))
    identity = model_capabilities(SimpleNamespace(app=app), 'fixture')['runtime_identity']
    assert identity['general_rerank_top_n'] == 24 and identity['general_rerank_window_enabled'] is True
    monkeypatch.setenv('EVIDENCE_RAG_GENERATION_GATE', 'grounded_evidence')
    monkeypatch.setenv('EVIDENCE_RAG_MODEL_CONTEXT_WINDOW', '100')
    monkeypatch.setenv('EVIDENCE_RAG_FRAMING_RESERVE', '1')
    monkeypatch.setenv('EVIDENCE_RAG_OUTPUT_RESERVE', '2')
    bundle = {'status':'ready','query_plan':{'mode':'general_retrieval','profile':'general','applicability_review':False},'evidence':[{'reranker_score':1.0}, {'reranker_score':None}],'degraded':False,'budget':{'state':'measured','used_tokens':2,'omitted':[]}}
    assert not generation_allowed(bundle, SimpleNamespace(), app)
    bundle['query_plan']['mode'] = 'source_lookup'
    assert not generation_allowed(bundle, SimpleNamespace(), app)
    bundle['evidence'].pop()
    assert generation_allowed(bundle, SimpleNamespace(), app)


def test_exact_match_remains_first_even_if_window_is_smaller_than_exact_set():
    rows = [{'unit_id': str(i)} for i in range(5)]
    result = rerank_candidates({'candidates': rows}, 'q', ['0', '1', '2'], ReverseReranker(), 1)
    assert [row['unit_id'] for row in result['candidates']] == ['0', '1', '2', '3', '4']
    assert set(result['reranker_scores']) == {'0'}
