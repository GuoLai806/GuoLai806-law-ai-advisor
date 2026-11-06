"""Export text-free numeric evidence while preserving original hashes locally."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.e2e import file_hash,save_json

FORBIDDEN={'answer','request','query','text','events','auth','api_key','model_key','plain_text','quote'}


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in FORBIDDEN}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,str):return re.sub(r'/Users/[^\s\"\'\)]+','<local-path>',value)
    return value


def publish(input_dir,output_dir,names,auxiliary):
    src,dest=Path(input_dir),Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in names:
        path=src/(name+'.json'); original=json.loads(path.read_text()); redacted=clean(original)
        redacted['original_report_sha256']=file_hash(path)
        redacted['redaction']='Question, source, answer and raw SSE plaintext remain local; all numeric rows, failures, rankings, hashes and usage retained.'
        target=dest/path.name;save_json(target,redacted)
        configuration=src/(name+'.config.json');value=json.loads(configuration.read_text())
        if value!=clean(value):raise ValueError('Configuration contains private paths or plaintext fields')
        with (dest/configuration.name).open('xb') as stream:stream.write(configuration.read_bytes())
        journal=Path(str(path)+'.journal.jsonl');jtarget=dest/journal.name
        with jtarget.open('x') as stream:
            for line in journal.read_text().splitlines():stream.write(json.dumps(clean(json.loads(line)),ensure_ascii=False)+'\n')
        keys=('query_id','gold_label','predicted_label','gold_documents','cited_documents','failure','retrieval_metrics',
              'final_evidence_recall','user_total_ms','user_total_wall_ms','first_answer_wall_ms','raw_sha256',
              'source_reread_verified','session_reread_verified','usage','usage_complete','grounding_validation')
        if not all({k:clean(r.get(k)) for k in keys}=={k:p.get(k) for k in keys} for r,p in zip(original['rows'],redacted['rows'],strict=True)):
            raise ValueError('Published evidence changed numeric or failure fields')
        records.append({'name':name,'rows':len(original['rows']),'original_report_sha256':file_hash(path),
                        'public_report_sha256':file_hash(target),'original_journal_sha256':file_hash(journal),
                        'public_journal_sha256':file_hash(jtarget),'config_sha256':file_hash(configuration),
                        'numerical_and_failure_fields_unchanged':True})
    aux=[]
    for name in auxiliary:
        p=src/name;target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        save_json(target,clean(json.loads(p.read_text())))
        aux.append({'name':name,'original_sha256':file_hash(p),'public_sha256':file_hash(target)})
    audit={'records':records,'auxiliary':aux,'raw_events':'retained locally, not redistributed',
           'policy':'same failed rows and original gold; no credentials, raw source/answer text, private code, weights or user files',
           'selection_paths':'public selection paths are redacted; paired hash refers to original local pair, regenerated from new local paths when reproducing'}
    save_json(dest/'PUBLIC-REDACTION-AUDIT.json',audit)
    print(json.dumps({'reports':len(records),'rows':sum(x['rows'] for x in records),'auxiliary':len(aux)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--reports',nargs='+',required=True);p.add_argument('--aux',nargs='*',default=[])
    a=p.parse_args();publish(a.input_dir,a.output_dir,a.reports,a.aux)
