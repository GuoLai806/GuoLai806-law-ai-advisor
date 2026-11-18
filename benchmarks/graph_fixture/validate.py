#!/usr/bin/env python3
"""Verify fixture provenance, edge pointers, and the 30 frozen draft items offline."""
import collections,hashlib,json,pathlib,re
P=pathlib.Path(__file__).resolve().parent
def lines(n): return [json.loads(x) for x in (P/n).read_text().splitlines()]
docs=lines('docs.jsonl'); edges=lines('edges.jsonl'); questions=lines('questions.jsonl')
d={x['_id']:x for x in docs}; assert len(d)==len(docs)==27
E={(x['source'],x['target']):x for x in edges};assert len(E)==len(edges)==67
for doc in docs:
    m=doc['metadata']; b=(P/m['source_file']).read_bytes(); source=json.loads(b)
    assert hashlib.sha256(b).hexdigest()==m['snapshot_sha256']
    assert source['registry_response_sha256']==m['source_sha256']
    assert len(m['source_sha256'])==64 and source['license'] in ('MIT','ISC','BSD-3-Clause')
    assert doc['_id']==source['name']+'@'+source['version']
    assert doc['source_url']=='https://registry.npmjs.org/'+source['name']+'/'+source['version']
    assert json.dumps(source.get('dependencies',{}),ensure_ascii=False,sort_keys=True) in doc['text']
for e in edges:
    a=d[e['source']]; b=d[e['target']]; source=json.loads((P/a['metadata']['source_file']).read_bytes())
    name=b['metadata']['name']; spec=source['dependencies'][name]
    assert re.fullmatch(r'\d+\.\d+\.\d+',spec) and spec==b['metadata']['version']==e['spec']
    assert e['evidence_id']==e['source'] and e['locator']=='/dependencies/'+name.replace('~','~0').replace('/','~1')
assert len(questions)==30 and len({q['id'] for q in questions})==30
assert collections.Counter(q['category'] for q in questions)=={'multi_document':12,'historical':6,'single_document':6,'negative_or_ambiguous':6}
for q in questions:
    assert all(e in d for e in q['required_evidence'])
    if q['category']=='multi_document':
        p=q['gold_path']; assert len(p)>=2
        for i,e in enumerate(p):
            assert E[(e['source'],e['target'])]==e
            if i: assert p[i-1]['target']==e['source']
        assert q['required_evidence']==[e['source'] for e in p]
    if q['category']=='single_document':
        doc=json.loads((P/d[q['required_evidence'][0]]['metadata']['source_file']).read_bytes())
        assert doc['dependencies'][q['locator'].split('/')[-1]]==q['expected_answer']
historical=questions[12:18]
def dep(n,k): return json.loads((P/d[n]['metadata']['source_file']).read_bytes())['dependencies'][k]
assert historical[0]['expected_answer']==[dep('express@4.18.2','body-parser'),dep('express@4.19.2','body-parser')]
assert historical[1]['expected_answer']==[dep('express@4.18.2','cookie'),dep('express@4.19.2','cookie')]
assert historical[2]['expected_answer']==[dep('body-parser@1.20.1','raw-body'),dep('body-parser@1.20.2','raw-body')]
assert historical[3]['expected_answer']==['express@4.18.2','body-parser@'+dep('express@4.18.2','body-parser'),'raw-body@'+dep('body-parser@1.20.1','raw-body')]
assert historical[4]['expected_answer']==['express@4.19.2','body-parser@'+dep('express@4.19.2','body-parser'),'raw-body@'+dep('body-parser@1.20.2','raw-body')]
assert historical[5]['expected_answer']==[dep('body-parser@1.20.1','content-type'),dep('body-parser@1.20.2','content-type')]
assert 'raw-body' not in json.loads((P/d['express@4.19.2']['metadata']['source_file']).read_bytes())['dependencies']
assert dep('body-parser@1.20.2','content-type')=='~1.0.5'
print('PASS: 27 source snapshots and checksums, 67 exact dependency edges, 30 draft questions; path and single-document gold verified')
