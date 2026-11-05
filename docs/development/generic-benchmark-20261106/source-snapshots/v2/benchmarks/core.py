"""Validated generic JSONL corpus, qrels, FTS5 BM25, exact ranking and fusion."""
import hashlib,json,math,re,sqlite3
from pathlib import Path
from .prepare import filehash,jsonl,qrels

FORMAT='title + newline + text; no ids, qrels or answers'

def load_dataset(data_dir,manifest_file,split):
 m=json.loads(Path(manifest_file).read_text());root=Path(data_dir)/m['dataset']
 for rel,expected in [('corpus.jsonl',m['corpus_sha256']),('queries.jsonl',m['queries_sha256'])]:
  if filehash(root/rel)!=expected:raise ValueError('Dataset file hash mismatch')
 corpus=jsonl(root/'corpus.jsonl');queries=jsonl(root/'queries.jsonl')
 if len(corpus)!=m['corpus_count']:raise ValueError('Corpus coverage mismatch')
 source=m['validation']['source_split'] if split=='validation' else 'test'
 if filehash(root/'qrels'/f'{source}.tsv')!=m['qrels'][source]['sha256']:raise ValueError('Qrels hash mismatch')
 rel=qrels(root/'qrels'/f'{source}.tsv');ids=m['validation']['query_ids'] if split=='validation' else sorted(rel)
 if len(ids)!=(m['validation']['count'] if split=='validation' else m['official_test_count']):raise ValueError('Query count mismatch')
 if len(ids)!=len(set(ids)) or set(ids)-set(rel) or set(ids)-set(queries):raise ValueError('Missing or duplicate query ID')
 if any(set(rel[q])-set(corpus) for q in ids):raise ValueError('Qrels contain missing document IDs')
 for doc in corpus.values():
  if not isinstance(doc.get('title',''),str) or not isinstance(doc.get('text'),str):raise ValueError('Invalid corpus document')
 for q in ids:
  if not isinstance(queries[q].get('text'),str):raise ValueError('Invalid query')
 return m,corpus,queries,{q:rel[q] for q in ids},ids

def input_text(doc):return (doc.get('title') or '')+'\n'+doc['text']

class BM25:
 def __init__(self,corpus):
  self.db=sqlite3.connect(':memory:');self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(docid UNINDEXED, body, tokenize='unicode61 remove_diacritics 2')")
  self.db.executemany('INSERT INTO docs(docid,body) VALUES (?,?)',((did,input_text(doc)) for did,doc in corpus.items()));self.db.commit()
 def search(self,query,limit=100):
  words=list(dict.fromkeys(re.findall(r'[^\W_]+',query,flags=re.UNICODE)))
  if not words:return []
  match=' OR '.join('"'+w.replace('"','""')+'"' for w in words)
  rows=self.db.execute('SELECT docid,bm25(docs) AS score FROM docs WHERE docs MATCH ? ORDER BY score ASC, docid DESC LIMIT ?', (match,limit)).fetchall()
  return [(did,float(score)) for did,score in rows]
 def close(self):self.db.close()

def fuse(lists,k=60):
 values={};sources={}
 for name,ranked in lists.items():
  seen=set()
  for rank,(did,raw) in enumerate(ranked,1):
   if did in seen:continue
   if not math.isfinite(raw):raise ValueError('nonfinite retrieval score')
   seen.add(did);values[did]=values.get(did,0.0)+1/(k+rank);sources.setdefault(did,[]).append({'source':name,'rank':rank,'raw_score':raw,'score_direction':'lower' if name=='bm25' else 'higher'})
 return sorted(values,key=lambda x:(values[x],x),reverse=True),values,sources
