import pytest

from evidence_rag.retrieval import CrossEncoderReranker, RerankWindowExceeded


class MaskRow:
    def __init__(self, length):
        self.length = length

    def sum(self):
        return self.length


class Model:
    max_length = 32
    prompts = {}
    default_prompt_name = None

    def __init__(self):
        self.forwarded = []
        self.prechecks = 0

    def length(self, pair):
        return sum(map(len, pair)) + 4

    def preprocess(self, pairs, **kwargs):
        self.prechecks += 1
        return {'attention_mask': [MaskRow(self.length(pair)) for pair in pairs]}

    def predict(self, pairs, batch_size):
        if any(self.length(pair) > self.max_length for pair in pairs):
            raise RerankWindowExceeded('whole paragraph exceeds reranker token limit')
        self.forwarded.extend(pairs)
        return [0.9 if 'KEY' in passage else 0.3 for _, passage in pairs]


def ranker(policy='split', mode='guarded_batch'):
    instance = object.__new__(CrossEncoderReranker)
    instance.long_input = policy
    instance.window_check = mode
    instance.model = Model()
    instance.batch_size = 8
    return instance


def candidate(text):
    return {'unit_id': 'u', 'document': {'doc_id': 'doc'}, 'title': 'T', 'text': text,
            'parts': [{'part_no': 7, 'text': text}]}


@pytest.mark.parametrize('mode', ['legacy', 'guarded_batch'])
def test_whole_paragraph_covered_and_winner_points_into_original(mode):
    worker = ranker(mode=mode)
    text = '甲乙丙丁。many scientific words and spaces.\n' * 4 + 'KEY'
    unit = candidate(text)
    [(result, score)] = worker.rank('Q', [unit])
    windows = [passage.removeprefix('T\n') for query,passage in worker.model.forwarded]
    assert len(windows)>1 and ''.join(windows)==text
    assert all(query=='Q' and passage.startswith('T\n') for query, passage in worker.model.forwarded)
    assert all(worker.model.length(pair)<=32 for pair in worker.model.forwarded)
    assert result['parts']==unit['parts'] and result['rerank_part_no']==7 and score==0.9
    span=result['rerank_window']
    assert span['part_no']==7 and span['window_count']==len(windows)
    assert 'KEY' in text[span['char_start']:span['char_end']]


def test_short_guarded_inputs_keep_original_predict_path():
    worker=ranker()
    [(result, score)] = worker.rank('Q', [candidate('fits')])
    assert 'rerank_window' not in result and score==0.3
    assert worker.model.prechecks==0 and worker.model.forwarded==[('Q','T\nfits')]


def test_long_nonwinning_part_does_not_claim_a_short_parts_score():
    worker=ranker()
    unit=candidate('x'*100)
    unit['parts'].append({'part_no':8,'text':'KEY'})
    [(result, score)] = worker.rank('Q',[unit])
    assert result['rerank_part_no']==8 and score==0.9 and 'rerank_window' not in result


def test_strict_policy_and_unfit_query_remain_failures():
    with pytest.raises(RerankWindowExceeded): ranker('reject').rank('Q',[candidate('x'*120)])
    worker=ranker()
    with pytest.raises(RerankWindowExceeded): worker.rank('Q'*50,[candidate('x'*100)])
    assert not worker.model.forwarded


def test_unrelated_errors_are_not_retried_as_window_overflow():
    worker=ranker()
    def fail(*args,**kwargs):
        raise RuntimeError('backend unavailable')
    worker.model.predict=fail
    with pytest.raises(RuntimeError,match='backend unavailable'):worker.rank('Q',[candidate('fits')])


@pytest.mark.parametrize('profile', ['general', 'legal'])
def test_full_source_http_response_locates_its_winning_window(tmp_path, monkeypatch, profile):
    import hashlib
    import json
    from fastapi.testclient import TestClient
    from evidence_rag.api import app, legal_app

    service_app = legal_app if profile=='legal' else app
    monkeypatch.setenv('EVIDENCE_RAG_DB',str(tmp_path/'db'))
    monkeypatch.setenv('EVIDENCE_RAG_AUTH',json.dumps({'local-test':'reader'}))
    monkeypatch.setenv('EVIDENCE_RAG_DEFAULT_PROFILE',profile)
    source = '甲乙丙丁。'*20 + 'KEY'
    with TestClient(service_app) as client:
        cat=service_app.state.catalog
        request={'query':'第一条' if profile=='legal' else 'find','profile':profile,'session_id':'source'}
        if profile=='general':
            cat.stage_documents([{'_id':'doc','title':'T','text':'short\n\n'+source}],
                                {'corpus_id':'demo','version_id':'v1','source_kind':'benchmark','verification_state':'provided'},'r')
            request['document_id']='doc'
        else:
            cat.stage(['第一条 '+source],{'doc_id':'law','version_id':'v1','title':'测试法','source_kind':'synthetic','verification_state':'approved'},'r')
        cat.activate('r',None)
        service_app.state.service.reranker=ranker()
        resp=client.post('/api/v2/retrieve',json=request,headers={'Authorization':'Bearer local-test'})
        assert resp.status_code==200
        result=resp.json()
        assert 'retrieval_trace' not in result
        item=result['evidence'][0]
        unit=cat.get('r',item['unit_id'])
        assert item['text']==unit['text'] and item['content_sha256']==unit['sha256']
        window=item['rerank_window']
        part=next(p for p in unit['parts'] if p['part_no']==window['part_no'])
        assert item['reranker_score']==0.9 and window['window_count']>1
        assert window['part_locator']==part['locator']
        assert window['part_sha256']==hashlib.sha256(part['text'].encode()).hexdigest()
        assert 'KEY' in part['text'][window['char_start']:window['char_end']]
        assert 'scored_spans' not in window


def test_legacy_unit_without_parts_uses_its_own_locator():
    worker=ranker()
    text='word '*20+'KEY'
    unit={**candidate(text),'parts':None,'locator':{'pages':[3]},'sha256':'unit-hash'}
    [(result,score)]=worker.rank('Q',[unit])
    assert score==0.9 and result['rerank_window']['part_no'] is None
    assert result['rerank_window']['part_locator']==unit['locator']
    assert result['rerank_window']['part_sha256']==unit['sha256']
