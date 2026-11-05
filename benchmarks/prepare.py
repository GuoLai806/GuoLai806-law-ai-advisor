"""Fetch pinned BEIR archives into a user-selected data directory, never the repo."""
import argparse,csv,hashlib,json,os,shutil,tempfile,urllib.request,zipfile
from pathlib import Path,PurePosixPath

SPECS={
 'scifact':{'md5':'5f7d1de60b170fc8027bb7898e2efca1','corpus':5183,'test_queries':300,'validation_split':'train','license':'CC-BY-SA-4.0','license_source':'https://huggingface.co/datasets/BeIR/scifact'},
 'nfcorpus':{'md5':'a89dba18a62ef92f7d323ec890a0d38d','corpus':3633,'test_queries':323,'validation_split':'dev','license':'Original NFCorpus terms specify academic use; other NutritionFacts use requires consulting terms and author','license_source':'https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/'},
}
BASE='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/'
README='https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/README.md'

def filehash(path,algo='sha256'):
 h=hashlib.new(algo)
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def extract_safe(archive,dest,name):
 with zipfile.ZipFile(archive) as z:
  for entry in z.infolist():
   p=PurePosixPath(entry.filename)
   if p.is_absolute() or '..' in p.parts or '\\' in entry.filename or not p.parts or p.parts[0]!=name or (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe zip member')
   if entry.file_size>100_000_000 or sum(x.file_size for x in z.infolist())>300_000_000:raise ValueError('Unbounded archive')
  z.extractall(dest)

def jsonl(path):
 result={}
 for line in Path(path).read_text(encoding='utf-8').splitlines():
  x=json.loads(line);key=x['_id']
  if not isinstance(key,str) or not key or key in result:raise ValueError('duplicate or invalid identifier')
  result[key]=x
 return result

def qrels(path):
 result={}
 with Path(path).open(encoding='utf-8',newline='') as f:
  reader=csv.DictReader(f,delimiter='\t')
  if reader.fieldnames!=['query-id','corpus-id','score']:raise ValueError('Unexpected qrels columns')
  for row in reader:
   q,d,score=row['query-id'],row['corpus-id'],int(row['score'])
   if d in result.setdefault(q,{}):raise ValueError('duplicate qrel')
   result[q][d]=score
 return result

def inspect(root,name,archive_sha,validation_count):
 spec=SPECS[name];corpus=jsonl(root/'corpus.jsonl');queries=jsonl(root/'queries.jsonl');splits={}
 if len(corpus)!=spec['corpus']:raise ValueError('Unexpected corpus count')
 for n in ('train','dev','test'):
  f=root/'qrels'/f'{n}.tsv'
  if not f.exists():continue
  rel=qrels(f)
  if set(rel)-set(queries) or {d for ds in rel.values() for d in ds}-set(corpus):raise ValueError('qrel references unknown ID')
  if any(not any(v>0 for v in ds.values()) for ds in rel.values()):raise ValueError('query has no positive qrels')
  splits[n]={'qids':len(rel),'qrels':sum(map(len,rel.values())),'sha256':filehash(f)}
 if splits['test']['qids']!=spec['test_queries']:raise ValueError('Unexpected test query count')
 val=qrels(root/'qrels'/f"{spec['validation_split']}.tsv")
 if len(val)<validation_count or not 24<=validation_count<=40:raise ValueError('Validation count must be 24-40 and fit split')
 selected=sorted(val,key=lambda q:(hashlib.sha256(q.encode()).hexdigest(),q))[:validation_count]
 test=set(qrels(root/'qrels/test.tsv'))
 if set(selected)&test:raise ValueError('Validation/test query IDs overlap')
 return {'version':1,'dataset':name,'url':BASE+name+'.zip','expected_md5':spec['md5'],'archive_sha256':archive_sha,
  'corpus_sha256':filehash(root/'corpus.jsonl'),'queries_sha256':filehash(root/'queries.jsonl'),'corpus_count':len(corpus),'query_count_total':len(queries),'qrels':splits,
  'validation':{'source_split':spec['validation_split'],'selection':'first N SHA-256(query_id UTF-8), lexical tie break','query_ids':selected,'count':len(selected)},
  'official_test_count':len(test),'official_test_policy':'no model run before explicit config selection','license':spec['license'],'license_source':spec['license_source'],'beir_readme':README}

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=SPECS,required=True);p.add_argument('--data-dir',required=True);p.add_argument('--manifest',required=True);p.add_argument('--validation-count',type=int,default=32);a=p.parse_args()
 target=Path(a.data_dir).resolve();repo=Path(__file__).resolve().parents[1]
 if target==repo or repo in target.parents:raise ValueError('Raw BEIR data must be outside repository')
 target.mkdir(parents=True,exist_ok=True);final=target/a.dataset;archive=target/(a.dataset+'.zip');out=Path(a.manifest)
 if out.exists():raise FileExistsError('Manifest exists; preserve prior freeze')
 if final.exists() or archive.exists():raise FileExistsError('Destination exists; do not overwrite without explicit cleanup')
 with tempfile.TemporaryDirectory(dir=target,prefix='.download-') as tmp:
  temp=Path(tmp);zip_path=temp/'data.zip';url=BASE+a.dataset+'.zip'
  with urllib.request.urlopen(url,timeout=60) as resp,zip_path.open('wb') as f:
   total=0
   while chunk:=resp.read(1024*1024):
    total+=len(chunk)
    if total>100_000_000:raise ValueError('Archive too large')
    f.write(chunk)
  if filehash(zip_path,'md5')!=SPECS[a.dataset]['md5']:raise ValueError('Official archive MD5 mismatch')
  extract_safe(zip_path,temp,a.dataset)
  manifest=inspect(temp/a.dataset,a.dataset,filehash(zip_path),a.validation_count)
  shutil.copy2(zip_path,archive);os.rename(temp/a.dataset,final)
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'dataset':a.dataset,'corpus_count':manifest['corpus_count'],'validation_count':manifest['validation']['count'],'test_count':manifest['official_test_count'],'archive_sha256':manifest['archive_sha256']}))
if __name__=='__main__':main()
