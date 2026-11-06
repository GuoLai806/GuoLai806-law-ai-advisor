import pytest
from benchmarks.verification_metrics import aggregate,gold_label,parse_verdict
from evidence_rag.budget import HuggingFaceTokenizer

def test_failures_and_invalid_answers_remain_in_denominator():
    rows=[{'gold_label':'SUPPORT','predicted_label':'SUPPORT','gold_documents':['one'],'cited_documents':['one']},
          {'gold_label':'CONTRADICT','predicted_label':None,'gold_documents':['two'],'cited_documents':[]},
          {'gold_label':'NOT_ENOUGH_INFO','predicted_label':'SUPPORT','gold_documents':[],'cited_documents':['three']}]
    result=aggregate(rows)
    assert result['claim_label_accuracy']==pytest.approx(1/3)
    assert result['cited_document_label_precision']==.5
    assert result['cited_document_label_recall']==.5
    assert result['confusion']['CONTRADICT']['INVALID']==1

def test_gold_label_does_not_treat_empty_evidence_as_support():
    assert gold_label({'evidence':{}})=='NOT_ENOUGH_INFO'
    assert gold_label({'evidence':{'doc':[{'label':'CONTRADICT'}]}})=='CONTRADICT'
    with pytest.raises(ValueError):
        gold_label({'evidence':{'doc':[{'label':'CONTRADICT'},{'label':'SUPPORT'}]}})

def test_verdict_parser_requires_an_explicit_first_line():
    assert parse_verdict('**SUPPORT**\nEvidence [E1]')=='SUPPORT'
    assert parse_verdict('No clear evidence. SUPPORT would be speculative.') is None

def test_chat_budget_counts_tokens_instead_of_mapping_keys():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs['return_dict'] is False
            return list(range(150))
    counter=HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
    counter.tokenizer=Tokenizer()
    assert counter.count_chat([{'role':'user','content':'A longer question'}])==150
