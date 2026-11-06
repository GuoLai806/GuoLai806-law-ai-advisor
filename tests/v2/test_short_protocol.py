import json
from pathlib import Path
import pytest
from test_freeze_short_run import freeze_inputs
from test_short_user_flow import full_scifact_fixture
from tools.freeze_short_run import freeze
from benchmarks.short_protocol import validate_selection
from tools.audit_user_flow import scoring_and_usage_checks


def validate(args,**kw):
    return validate_selection(args.baseline_selection,args.baseline_config,args.manifest,args.baseline_test_output,**kw)


def test_complete_pair_and_real_failure_are_accepted(freeze_inputs):
    freeze(freeze_inputs)
    value=validate(freeze_inputs)
    assert value['freeze_pair_sha256'] and value['development_report_sha256']


@pytest.mark.parametrize('mutation',['two_hashes','output','peer','source','dev_report','journal','candidate_config'])
def test_changed_selection_evidence_is_rejected_before_calls(freeze_inputs,mutation):
    args=freeze_inputs;freeze(args)
    path=Path(args.baseline_selection);value=json.loads(path.read_text())
    if mutation=='two_hashes':value={k:value[k] for k in ('config_sha256','manifest_sha256')}
    elif mutation=='output':value['formal_output']=str(Path(args.baseline_test_output).with_name('another.json'))
    elif mutation=='peer':Path(args.candidate_selection).write_text('{}')
    elif mutation=='source':value['source_sha256']={}
    elif mutation=='dev_report':Path(args.development_report).write_text('{}')
    elif mutation=='journal':Path(args.development_report+'.journal.jsonl').write_text('{}\n')
    elif mutation=='candidate_config':Path(args.candidate_config).write_text('{}')
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError,KeyError)):
        validate(args)


def test_existing_formal_record_cannot_be_replayed(freeze_inputs):
    freeze(freeze_inputs);Path(freeze_inputs.baseline_test_output).write_text('{}')
    with pytest.raises(ValueError):validate(freeze_inputs)


def test_grounded_answer_token_cannot_be_scored_from_prose():
    metadata={'status':'verbatim_quotes_checked','decision':'ANSWER'}
    raw={'events':[{'event':'answer_delta','data':{'delta':'SUPPORT is a word [E1].'}},
                   {'event':'evidence_ready','data':{'bundle':{'evidence':[{'citation_id':'E1','document_id':'d1'}]}}},
                   {'event':'grounding_validation','data':metadata}]}
    failed={'predicted_label':None,'cited_documents':[],'failure':{'type':'invalid_answer_format'},
            'grounding_validation':metadata,'generation_attempts':[],'generation_retries':[],'observed_answer_models':[],
            'observed_answer_model':None,'usage':None,'usage_attempts':[],'usage_complete':False}
    checks=scoring_and_usage_checks(failed,raw,'grounded_json')
    assert checks['prediction_matches_raw'] and checks['format_failure_consistent'] and checks['grounding_metadata_matches_raw']
    claimed={**failed,'predicted_label':'SUPPORT','cited_documents':['d1'],'failure':None}
    assert not scoring_and_usage_checks(claimed,raw,'grounded_json')['prediction_matches_raw']
