#!/usr/bin/env python3
"""Build a bounded source-faithful corpus; --fetch refreshes registry snapshots."""
import hashlib, json, pathlib, re, sys, urllib.request
from collections import deque
P = pathlib.Path(__file__).resolve().parent
PACKAGES = '''express@4.18.2 express@4.19.2 body-parser@1.20.1 body-parser@1.20.2 raw-body@2.5.1 raw-body@2.5.2 send@0.18.0 serve-static@1.15.0 http-errors@2.0.0 debug@2.6.9 ms@2.0.0 ms@2.1.3 bytes@3.1.2 depd@2.0.0 destroy@1.2.0 unpipe@1.0.0 iconv-lite@0.4.24 on-finished@2.4.1 qs@6.11.0 statuses@2.0.1 inherits@2.0.4 toidentifier@1.0.1 setprototypeof@1.2.0 cookie@0.5.0 cookie@0.6.0 safe-buffer@5.2.1 mime@1.6.0'''.split()
def dump(path, rows):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True)+'\n' for x in rows))
def read(n):
    return json.loads((P/'sources'/f'{n}.json').read_bytes())
if '--fetch' in sys.argv:
    for n in PACKAGES:
        name, version = n.rsplit('@', 1)
        url=f'https://registry.npmjs.org/{name}/{version}'
        with urllib.request.urlopen(url,timeout=30) as response: raw=response.read()
        doc=json.loads(raw)
        assert (doc['name'],doc['version'])==(name,version)
        snapshot={k:doc[k] for k in ('name','version','license','dependencies','gitHead') if k in doc}
        snapshot['registry_response_sha256']=hashlib.sha256(raw).hexdigest()
        (P/'sources'/f'{n}.json').write_text(json.dumps(snapshot,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
        print(n, snapshot['registry_response_sha256'])
docs=[]; edges=[]
for n in PACKAGES:
    name,version=n.rsplit('@',1); raw=(P/'sources'/f'{n}.json').read_bytes(); m=json.loads(raw)
    deps=m.get('dependencies',{})
    digest=hashlib.sha256(raw).hexdigest()
    text=f"Package {name}, version {version}. Runtime dependencies (npm package manifest JSON object /dependencies): "+json.dumps(deps,ensure_ascii=False,sort_keys=True)+'.'
    docs.append({'_id':n,'title':f'{name} {version} npm package manifest','text':text,'source_url':f'https://registry.npmjs.org/{name}/{version}', 'metadata':{'name':name,'version':version,'license':m.get('license'),'source_sha256':m['registry_response_sha256'],'snapshot_sha256':digest,'source_file':f'sources/{n}.json','locator':'/dependencies','extraction':'Selected name, version, and dependencies from original npm registry JSON; not an installed dependency lockfile.'}})
    for dep,spec in sorted(deps.items()):
        dest=f'{dep}@{spec}'
        if re.fullmatch(r'\d+\.\d+\.\d+',spec) and dest in PACKAGES:
            edges.append({'source':n,'predicate':'depends_on','target':dest,'evidence_id':n,'locator':'/dependencies/'+dep.replace('~','~0').replace('/','~1'),'spec':spec})
dump(P/'docs.jsonl',docs);dump(P/'edges.jsonl',edges)
adj={n:[] for n in PACKAGES}
for e in edges: adj[e['source']].append(e)
def path(a,b):
    q=deque([(a,[])]);seen={a}
    while q:
        node,p=q.popleft()
        if node==b:return p
        for edge in adj[node]:
            if edge['target'] not in seen:seen.add(edge['target']);q.append((edge['target'],p+[edge]))
    raise ValueError((a,b))
questions=[]
def add(i,category,q,answer,evidence,**extra):
    questions.append({'id':f'q{i:02d}','category':category,'question':q,'expected_answer':answer,'required_evidence':evidence,**extra})
pairs=[('express@4.18.2','bytes@3.1.2'),('express@4.18.2','raw-body@2.5.1'),('express@4.19.2','raw-body@2.5.2'),('express@4.19.2','bytes@3.1.2'),('express@4.18.2','inherits@2.0.4'),('express@4.19.2','inherits@2.0.4'),('serve-static@1.15.0','mime@1.6.0'),('serve-static@1.15.0','ms@2.1.3'),('body-parser@1.20.1','ms@2.0.0'),('body-parser@1.20.2','ms@2.0.0'),('send@0.18.0','inherits@2.0.4'),('raw-body@2.5.1','inherits@2.0.4')]
# A path is a proof of reachability, not a claim that no other path exists.
for i,(a,b) in enumerate(pairs,1):
    p=path(a,b)
    assert len(p)>=2,(a,b,p)
    add(i,'multi_document',f'In the documented exact-version subset, give one evidence-backed dependency path from {a} to {b}.', [a]+[e['target'] for e in p], [e['evidence_id'] for e in p],gold_path=p, grading='One valid path with all source documents; other supported paths are also valid.')
for i,(q,answer,ev) in enumerate([
('Compare the direct body-parser dependency in Express 4.18.2 and 4.19.2.',['1.20.1','1.20.2'],['express@4.18.2','express@4.19.2']),
('Compare the direct cookie dependency in Express 4.18.2 and 4.19.2.',['0.5.0','0.6.0'],['express@4.18.2','express@4.19.2']),
('Compare the direct raw-body dependency in body-parser 1.20.1 and 1.20.2.',['2.5.1','2.5.2'],['body-parser@1.20.1','body-parser@1.20.2']),
('Trace Express 4.18.2 to its exact raw-body version.',['express@4.18.2','body-parser@1.20.1','raw-body@2.5.1'],['express@4.18.2','body-parser@1.20.1']),
('Trace Express 4.19.2 to its exact raw-body version.',['express@4.19.2','body-parser@1.20.2','raw-body@2.5.2'],['express@4.19.2','body-parser@1.20.2']),
('Compare the direct content-type dependency specifications in body-parser 1.20.1 and 1.20.2.',['~1.0.4','~1.0.5'],['body-parser@1.20.1','body-parser@1.20.2'])],13):add(i,'historical',q,answer,ev,grading='Version snapshots only; no inference of calendar effective dates.')
for i,(n,d) in enumerate([('express@4.18.2','body-parser'),('express@4.19.2','cookie'),('body-parser@1.20.1','raw-body'),('body-parser@1.20.2','bytes'),('serve-static@1.15.0','send'),('send@0.18.0','ms')],19):
    v=read(n)['dependencies'][d];add(i,'single_document',f'What direct {d} dependency specification does {n} declare?',v,[n],locator='/dependencies/'+d)
for i,(q,answer,ev) in enumerate([
('Which Express version is intended if a request only says “Express” and gives no version?', 'ambiguous: both 4.18.2 and 4.19.2 are represented', ['express@4.18.2','express@4.19.2']),
('Which exact installed version does the ~1.0.5 content-type range in body-parser 1.20.2 resolve to?', 'unknown: a range is not a lockfile', ['body-parser@1.20.2']),
('Was Express 4.19.2 in force on 2023-01-01?', 'unknown: release and effective-date evidence is not supplied in this corpus', []),
('List every production dependency of Express 4.19.2, including versions resolved from all ranges.', 'not supported: this fixture has only a bounded exact-version subset and no resolution lockfile', ['express@4.19.2']),
('Does express 4.19.2 directly depend on raw-body 2.5.2?', 'no: body-parser 1.20.2 declares raw-body 2.5.2; Express declares body-parser 1.20.2', ['express@4.19.2','body-parser@1.20.2']),
('Which packages outside this corpus depend on bytes 3.1.2?', 'unknown: corpus is bounded and has no global reverse-dependency inventory', [])],25):add(i,'negative_or_ambiguous',q,answer,ev,grading='Correctly qualify the knowledge boundary; do not invent facts.')
dump(P/'questions.jsonl',questions)
print(f'{len(docs)} documents, {len(edges)} explicit exact-version subset edges, {len(questions)} questions')
