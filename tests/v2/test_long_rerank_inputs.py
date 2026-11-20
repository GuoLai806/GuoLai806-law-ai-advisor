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
