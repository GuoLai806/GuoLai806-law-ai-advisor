"""Projection and retrieval scoring for the AI-curated, human-unreviewed set.

This tool never invokes a model or assigns answer correctness/abstention labels.
All partitions are author-reviewed development material, not a final or blind test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError('Dataset hash mismatch; do not evaluate a changed set')
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format') != 'ai-curated-retrieval-evaluation-v1':
        raise ValueError('Unexpected dataset format')
    ids, questions, topics, labels = set(), set(), {}, {}
    for case in data['questions']:
        if case['id'] in ids or case['query'] in questions:
            raise ValueError('Duplicate question ID or query')
        ids.add(case['id']); questions.add(case['query'])
        if case['author'] != 'AI' or case['review_state'] != 'needs_human_review':
            raise ValueError('Unsupported label provenance')
        if case['split'] not in ('train', 'dev', 'test'):
            raise ValueError('Invalid split')
        if topics.setdefault(case['topic'],case['split']) != case['split']:
            raise ValueError('Topic leaked across partitions')
        groups = case['expected_labels']
        if not isinstance(groups.get('required'),list) or not isinstance(groups.get('alternatives'),list):
            raise ValueError('Invalid expected-label groups')
        if any(not isinstance(g,list) or not g for g in groups['alternatives']):
            raise ValueError('Alternative group must be nonempty')
        expected_labels=set(groups['required']) | {x for g in groups['alternatives'] for x in g}
        if expected_labels != {source['label'] for source in case['source_units']}:
            raise ValueError('Labels and source units disagree')
        for label in expected_labels:
            if labels.setdefault(label,case['split']) != case['split']:
                raise ValueError('Article leaked across partitions')
        if type(case.get('metric_eligible')) is not bool:
            raise ValueError('Missing metric eligibility')
        if case['should_abstain'] and (not case.get('abstain_reason') or case['metric_eligible']):
            raise ValueError('Invalid abstention case')
        if not case['should_abstain'] and not groups['required'] and not groups['alternatives']:
            raise ValueError('Positive question requires evidence')
    return data, actual


def longest_normalized_overlap(left, right):
    left = ''.join(c for c in unicodedata.normalize('NFKC', left) if c.isalnum())
    right = ''.join(c for c in unicodedata.normalize('NFKC', right) if c.isalnum())
    previous = [0] * (len(right) + 1)
    best = 0
    for a in left:
        current = [0] * (len(right) + 1)
        for j, b in enumerate(right, 1):
            if a == b:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def validate_catalog(data, path):
    catalog = Path(path).resolve()
    wal = Path(str(catalog) + '-wal')
    # A checkpointed snapshot can be opened immutably without creating SQLite
    # side files in a shared directory. Never ignore a nonempty WAL.
    before_main = digest(catalog)
    before_wal = (wal.stat().st_size, digest(wal)) if wal.exists() else (0, None)
    immutable = before_wal[0] == 0
    db = sqlite3.connect(f'file:{catalog}?mode=ro' + ('&immutable=1' if immutable else ''), uri=True)
    count = 0
    flags = []
    try:
        for case in data['questions']:
            for source in case['source_units']:
                row=db.execute('SELECT label,text,sha256,locator FROM units WHERE revision=? AND unit_id=?',
                               (source['revision'],source['unit_id'])).fetchone()
                if row is None or row[0] != source['label'] or row[2] != source['sha256'] or hashlib.sha256(row[1].encode()).hexdigest() != source['sha256'] or json.loads(row[3]) != source['locator']:
                    raise ValueError('Source unit, hash or locator mismatch: '+case['id'])
                overlap = longest_normalized_overlap(case['query'], row[1])
                if overlap >= 14:
                    flags.append({'id':case['id'], 'label':source['label'], 'characters':overlap})
                count += 1
    finally:
        db.close()
    after_wal = (wal.stat().st_size, digest(wal)) if wal.exists() else (0, None)
    if digest(catalog) != before_main or after_wal != before_wal:
        raise ValueError('Catalog changed during source validation')
    return count, flags


def project(data, source_hash, split, include_reviewed_test):
    if split == 'test' and not include_reviewed_test:
        raise ValueError('Author-reviewed development partition requires --include-reviewed-test')
    selected=[c for c in data['questions'] if c['split']==split]
    # Legacy model_experiment accepts a flat list of required labels only. Do not
    # silently flatten alternatives or invent relevant labels for abstention cases.
    positive=[c for c in selected if not c['should_abstain'] and c['metric_eligible']]
    if any(c['expected_labels']['alternatives'] for c in positive):
        raise ValueError('Legacy projection cannot represent alternatives')
    return {'kind':'ai-curated-human-review-pending', 'original_dataset_sha256':source_hash,
            'split':split,'purpose':'development_only', 'partition_status':'author_reviewed_not_blind',
            'excluded_abstention_ids':[c['id'] for c in selected if c['should_abstain']],
            'excluded_label_review_ids':[c['id'] for c in selected if not c['should_abstain'] and not c['metric_eligible']],
            'queries':[{'id':c['id'],'query':c['query'],'expected_labels':c['expected_labels']['required']} for c in positive]}


def score(data, original_hash, projection_path, report_path, include_reviewed_test):
    projection=json.loads(Path(projection_path).read_text(encoding='utf-8'))
    split=projection.get('split')
    expected=project(data,original_hash,split,include_reviewed_test)
    if projection != expected: raise ValueError('Projection is not the exact frozen split')
    report=json.loads(Path(report_path).read_text(encoding='utf-8'))
    if report.get('dataset_sha256') != digest(projection_path):raise ValueError('Model report does not identify this projection hash')
    cases={c['query']:c for c in data['questions'] if c['split']==split and not c['should_abstain'] and c['metric_eligible']}
    records=report.get('results');
    if not isinstance(records,list) or len(records)!=len(cases):raise ValueError('Missing or extra result rows')
    seen=set();out=[]
    for record in records:
        query=record.get('query')
        if query not in cases or query in seen:raise ValueError('Unmatched or repeated model query')
        seen.add(query);case=cases[query]
        ranked=record.get('ranking')
        if not isinstance(ranked,list): raise ValueError('Ranked result is missing')
        ranking=[]
        for hit in ranked:
            label=hit.get('label') if isinstance(hit,dict) else None
            if not isinstance(label,str):raise ValueError('Invalid ranked label')
            if label not in ranking:ranking.append(label)
        required=case['expected_labels']['required'];top20=ranking[:20];hits=[v for v in required if v in top20]
        recall=len(hits)/len(required)
        ideal=sum(1/math.log2(i+2) for i in range(min(len(required),10)))
        dcg=sum(1/math.log2(i+2) for i,v in enumerate(ranking[:10]) if v in required)
        out.append({'id':case['id'],'question_kind':case['question_kind'],'topic':case['topic'],'expected':required,'ranking':ranking,'recall_at_20':recall,'all_required_at_20':len(hits)==len(required),'ndcg_at_10':dcg/ideal,'missing_at_20':[v for v in required if v not in top20]})
    def metrics(rows):
        return {'n':len(rows),'mean_recall_at_20':sum(x['recall_at_20'] for x in rows)/len(rows) if rows else None,
                'all_required_count':sum(x['all_required_at_20'] for x in rows),'mean_ndcg_at_10':sum(x['ndcg_at_10'] for x in rows)/len(rows) if rows else None}
    groups=defaultdict(list)
    for r in out:groups[r['question_kind']].append(r)
    return {'dataset_sha256':original_hash,'projection_sha256':digest(projection_path),'model_report_sha256':digest(report_path),'split':split,
            'human_reviewed':False,'limitation':'AI labels; retrieval ranking only. Excluded abstention cases are not scored and this is not a professional benchmark.',
            'summary':metrics(out),'by_kind':{k:metrics(v) for k,v in groups.items()},'results':out}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--expected-sha256',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('validate');v.add_argument('--catalog',required=True);v.add_argument('--output',required=True)
    q=sub.add_parser('project');q.add_argument('--split',choices=['train','dev','test'],required=True);q.add_argument('--include-reviewed-test',action='store_true');q.add_argument('--output',required=True)
    r=sub.add_parser('score');r.add_argument('--projection',required=True);r.add_argument('--model-report',required=True);r.add_argument('--include-reviewed-test',action='store_true');r.add_argument('--output',required=True)
    a=p.parse_args();data,h=load(a.dataset,a.expected_sha256)
    if a.command=='validate':
        count,flags=validate_catalog(data,a.catalog)
        wal=Path(str(a.catalog)+'-wal')
        result={'dataset_sha256':h,'validator_sha256':digest(__file__),'catalog_main_sha256':digest(a.catalog),
                'catalog_wal_sha256':digest(wal) if wal.exists() else None,'catalog_wal_bytes':wal.stat().st_size if wal.exists() else 0,
                'question_count':len(data['questions']),'source_references_validated':count,
                'normalized_overlap_threshold':14,'overlap_flags':flags,'human_legal_review':False}
    elif a.command=='project':result=project(data,h,a.split,a.include_reviewed_test)
    else:result=score(data,h,a.projection,a.model_report,a.include_reviewed_test)
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'command':a.command,'output_sha256':digest(a.output)},ensure_ascii=False))
if __name__=='__main__':main()
