"""Public API boundaries for opt-in grounded answers; synthetic evidence only."""
from types import SimpleNamespace
import json
import pytest
import evidence_rag.api as api
from evidence_rag.grounding import GroundingRejected


def bundle(profile='general'):
    return {'trace_id':'fixture', 'status':'ready', 'evidence':[{'citation_id':'E1','reranker_score':-3}],
            'corpus_revision':'fixture', 'degraded':False,
            'budget':{'state':'measured','used_tokens':200,'omitted':[]},
            'query_plan':{'profile':profile,'applicability_review':False,'mode':'general'}}


def test_grounded_gate_preserves_budget_and_legal_boundaries(monkeypatch):
    monkeypatch.setenv('EVIDENCE_RAG_MODEL_CONTEXT_WINDOW','32768')
    monkeypatch.setenv('EVIDENCE_RAG_FRAMING_RESERVE','512')
    monkeypatch.setenv('EVIDENCE_RAG_OUTPUT_RESERVE','8192')
    monkeypatch.setenv('EVIDENCE_RAG_RERANK_THRESHOLD','0')
    monkeypatch.setenv('EVIDENCE_RAG_GENERATION_GATE','grounded_evidence')
    app=SimpleNamespace(state=SimpleNamespace(generator=SimpleNamespace(answer_format='grounded_json',max_output_tokens=8192)))
    assert api.generation_allowed(bundle(),SimpleNamespace(),app)
    assert not api.generation_allowed(bundle('legal'),SimpleNamespace(),app)
    for changed in [dict(degraded=True),dict(status='pending'),dict(evidence=[]),
                    dict(budget={'state':'measured','used_tokens':32768,'omitted':[]}),
                    dict(budget={'state':'measured','used_tokens':200,'omitted':['E2']})]:
        assert not api.generation_allowed({**bundle(),**changed},SimpleNamespace(),app)
    app.state.generator.answer_format='text'
    assert not api.generation_allowed(bundle(),SimpleNamespace(),app)


@pytest.mark.asyncio
async def test_json_failure_has_safe_type_known_usage_and_no_saved_failed_turn(monkeypatch):
    saved=[]
    class Generator:
        async def stream_events(self,*args):
            yield {'type':'generation_attempt','attempt':1}
            yield {'type':'usage','attempt':1,'usage':{'total_tokens':123}}
            raise GroundingRejected('invalid_quote')
    async def get_bundle(*args,**kwargs): return bundle()
    monkeypatch.setattr(api,'get_bundle',get_bundle)
    monkeypatch.setattr(api,'generation_allowed',lambda *args:True)
    monkeypatch.setattr(api,'save_turn',lambda *args:saved.append(args))
    request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(generator=Generator())))
    result=await api.chat(SimpleNamespace(query='fixture',session_id='fixture'),request,'fixture')
    assert result['status']=='evidence_only'
    assert result['failure']['error_type']=='GroundingRejected'
    assert result['failure']['reason']=='invalid_quote'
    assert result['usage']=={'total_tokens':123} and result['usage_complete']
    assert saved==[]


@pytest.mark.asyncio
async def test_sse_failure_carries_grounding_reason_without_answer_delta(monkeypatch):
    class Generator:
        async def stream_events(self,*args):
            yield {'type':'usage','usage':{'total_tokens':123}}
            raise GroundingRejected('invalid_quote')
    async def get_bundle(*args,**kwargs): return bundle()
    monkeypatch.setattr(api,'get_bundle',get_bundle)
    monkeypatch.setattr(api,'check_access',lambda *args:'fixture')
    monkeypatch.setattr(api,'generation_allowed',lambda *args:True)
    class Request:
        app=SimpleNamespace(state=SimpleNamespace(generator=Generator()))
        async def is_disconnected(self): return False
    result=await api.stream(SimpleNamespace(query='fixture',session_id='fixture'),Request(),'fixture')
    text=''.join([part async for part in result.body_iterator])
    assert 'event: answer_delta' not in text
    assert '"reason": "invalid_quote"' in text
    assert 'event: usage' in text and '"discard_pending_answer": true' in text
