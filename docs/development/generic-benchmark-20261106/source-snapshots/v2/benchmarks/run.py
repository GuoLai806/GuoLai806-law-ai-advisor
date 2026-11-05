"""Frozen BEIR retrieval runner; no legal routes, no labels in model inputs."""
import argparse,gc,hashlib,importlib.metadata,json,math,os,platform,time,traceback
from pathlib import Path
from .core import BM25, FORMAT, fuse, input_text, load_dataset
from .metrics import evaluate
from .prepare import filehash

EMBED={'bge':('BAAI/bge-m3','5617a9f61b028005a4858fdac845db406aefb181'), 'qwen':('Qwen/Qwen3-Embedding-4B','5cf2132abc99cad020ac570b19d031efec650f2b')}
RERANK={'bge':('BAAI/bge-reranker-v2-m3','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'),'qwen':('Qwen/Qwen3-Reranker-4B','22e683669bc0f0bd69640a1354a6d0aebcfeede5')}
INSTRUCTION='Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery: '

def digest_bytes(x):return hashlib.sha256(x).hexdigest()
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError('Output exists; retain previous run')
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)

def dense_cache(cache_dir,metadata,model,documents,query_texts,ids,qids,batch):
 import fcntl
 signature=digest_bytes(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode());root=Path(cache_dir)/signature
 root.mkdir(parents=True,exist_ok=True)
 with (root/'.lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  try:return _dense_cache_unlocked(cache_dir,metadata,model,documents,query_texts,ids,qids,batch)
  finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def _dense_cache_unlocked(cache_dir,metadata,model,documents,query_texts,ids,qids,batch):
 import numpy as np
 signature=digest_bytes(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode());root=Path(cache_dir)/signature
 vec=root/'corpus.npy';qry=root/'queries.npy';meta=root/'manifest.json'
 if meta.exists():
  old=json.loads(meta.read_text())
  if old.get('key')!=metadata or any(filehash(root/name)!=old['sha256'][name] for name in ('corpus.npy','queries.npy')):raise ValueError('Cache metadata or hash mismatch')
  v=np.load(vec,allow_pickle=False);q=np.load(qry,allow_pickle=False)
  if v.shape!=(len(ids),metadata['dimension']) or q.shape!=(len(qids),metadata['dimension']) or not np.isfinite(v).all() or not np.isfinite(q).all():raise ValueError('Cache shape or values invalid')
  if not np.allclose(np.linalg.norm(v,axis=1),1,atol=.01) or not np.allclose(np.linalg.norm(q,axis=1),1,atol=.01):raise ValueError('Cache is not normalized')
  return v,q,{'cache_key':signature,'hit':True,'corpus_ms':0,'query_ms':0,'query_batches':[]}
 root.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter();v=model.encode(documents,normalize_embeddings=True,convert_to_numpy=True,batch_size=batch,prompt='');corpus_ms=(time.perf_counter()-start)*1000
 start=time.perf_counter();query_parts=[];query_batches=[]
 for offset in range(0,len(query_texts),batch):
  tick=time.perf_counter();part=model.encode(query_texts[offset:offset+batch],normalize_embeddings=True,convert_to_numpy=True,batch_size=batch,prompt='');query_parts.append(part);query_batches.append({'start':offset,'size':len(query_texts[offset:offset+batch]),'ms':round((time.perf_counter()-tick)*1000,2)})
 q=np.concatenate(query_parts,axis=0);query_ms=(time.perf_counter()-start)*1000
 v=np.asarray(v,dtype=np.float32);q=np.asarray(q,dtype=np.float32)
 if v.shape!=(len(ids),metadata['dimension']) or q.shape!=(len(qids),metadata['dimension']) or not np.isfinite(v).all() or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(v,axis=1),1,atol=.01) or not np.allclose(np.linalg.norm(q,axis=1),1,atol=.01):raise ValueError('Invalid generated embedding vectors')
 for file,arr in [(vec,v),(qry,q)]:
  tmp=file.with_suffix('.tmp')
  with tmp.open('xb') as f:np.save(f,arr,allow_pickle=False);f.flush();os.fsync(f.fileno())
  os.replace(tmp,file)
 info={'key':metadata,'sha256':{'corpus.npy':filehash(vec),'queries.npy':filehash(qry)}};tmp=meta.with_suffix('.tmp');tmp.write_text(json.dumps(info,sort_keys=True));os.replace(tmp,meta)
 return v,q,{'cache_key':signature,'hit':False,'corpus_ms':round(corpus_ms,2),'query_ms':round(query_ms,2),'query_batches':query_batches}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--manifest',required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--mode',choices=['bm25','hybrid'],required=True);p.add_argument('--embedding',choices=EMBED);p.add_argument('--reranker',choices=RERANK);p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--dtype',choices=['float16','float32'],default='float16');p.add_argument('--batch-size',type=int,default=2);p.add_argument('--cache-dir');p.add_argument('--selection');p.add_argument('--validation-report');p.add_argument('--output',required=True);a=p.parse_args()
 if a.batch_size!=2:raise ValueError('Frozen comparison requires batch size 2')
 if a.mode=='hybrid' and (not a.embedding or not a.reranker or not a.cache_dir):raise ValueError('Hybrid requires embedding, reranker and cache directory')
 if a.mode=='bm25' and (a.embedding or a.reranker):raise ValueError('BM25 does not use models')
 if a.cache_dir:
  cache=Path(a.cache_dir).resolve();repo=Path(__file__).resolve().parents[1]
  if cache==repo or repo in cache.parents:raise ValueError('Vector cache must be outside repository')
 manifest,corpus,queries,rels,qids=load_dataset(a.data_dir,a.manifest,a.split)
 config={'mode':a.mode,'embedding':a.embedding,'reranker':a.reranker,'device':a.device if a.mode=='hybrid' else None,'dtype':a.dtype if a.mode=='hybrid' else None,'batch_size':a.batch_size}
 if a.split=='test':
  if not a.selection or not a.validation_report:raise ValueError('Official test requires a frozen validation-based selection and matching report')
  selection=json.loads(Path(a.selection).read_text()); validation=json.loads(Path(a.validation_report).read_text())
  if selection.get('manifest_sha256')!=filehash(a.manifest) or selection.get('config')!=config or selection.get('code_sha256')!={str(p):filehash(p) for p in (Path('benchmarks/run.py'),Path('benchmarks/core.py'),Path('benchmarks/metrics.py'))} or selection.get('validation_failures')!=0 or selection.get('validation_report_sha256')!=filehash(a.validation_report) or validation.get('split')!='validation' or validation.get('config')!=config or validation.get('manifest_sha256')!=filehash(a.manifest) or validation.get('failures')!=0 or validation.get('eligibility')!='comparable' or len(validation.get('results',[]))!=manifest['validation']['count'] or {x.get('query_id') for x in validation.get('results',[])}!=set(manifest['validation']['query_ids']) or any(x.get('failure') is not None for x in validation.get('results',[])):raise ValueError('Selection does not match frozen configuration and validation report')
 run_start=time.perf_counter();docs=sorted(corpus);texts=[input_text(corpus[i]) for i in docs];qtexts=[queries[i]['text'] for i in qids]
 report={'version':1,'dataset':manifest['dataset'],'split':a.split,'manifest_sha256':filehash(a.manifest),'corpus_sha256':manifest['corpus_sha256'],'config':config,'corpus_count':len(docs),'query_count_expected':len(qids),'qids_sha256':digest_bytes(json.dumps(qids,ensure_ascii=False,separators=(',',':')).encode()),'created_utc':now(),'platform':platform.platform(),'python':platform.python_version(),'input_format':FORMAT,'code_sha256':{str(p):filehash(p) for p in (Path('benchmarks/run.py'),Path('benchmarks/core.py'),Path('benchmarks/metrics.py'))},'ranking_scope':'standalone model experiment; not application runtime or answer quality','metrics_implementation':'TREC ndcg_cut linear relevance; BEIR recall and MRR; identical query/document IDs ignored in results','warmups':[],'results':[]}
 start=time.perf_counter();bm25=BM25(corpus);report['bm25_index_ms']=round((time.perf_counter()-start)*1000,2)
 vectors=qvectors=reranker=model=None;runtime={};startup_error=None
 try:
  if a.mode=='hybrid':
   import numpy as np,torch
   from sentence_transformers import SentenceTransformer,CrossEncoder
   if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
   model_id,revision=EMBED[a.embedding];start=time.perf_counter()
   model=SentenceTransformer(model_id,revision=revision,device=a.device,model_kwargs={'dtype':getattr(torch,a.dtype)},local_files_only=True)
   dtype_set=sorted({str(x.dtype) for x in model.parameters() if x.is_floating_point()});actual_device=str(model.device)
   if not actual_device.startswith(a.device) or dtype_set!=['torch.'+a.dtype]:raise RuntimeError('Embedding device/dtype mismatch')
   maxlen=model.max_seq_length
   query_inputs=[INSTRUCTION+q if a.embedding=='qwen' else q for q in qtexts]
   for text in texts+query_inputs:
    if len(model.tokenizer(text,add_special_tokens=True,truncation=False)['input_ids'])>maxlen:raise ValueError('Embedding text exceeds model token limit; no silent truncation')
   dim=int(model.get_sentence_embedding_dimension())
   metadata={'corpus_sha256':manifest['corpus_sha256'],'corpus_ids_sha256':digest_bytes(json.dumps(docs,separators=(',',':')).encode()),'input_format':FORMAT,'model':model_id,'revision':revision,'dtype':a.dtype,'device':a.device,'normalized':True,'dimension':dim,'query_hash':digest_bytes(json.dumps(list(zip(qids,query_inputs)),ensure_ascii=False,separators=(',',':')).encode()),'batch_size':a.batch_size}
   warm=time.perf_counter();model.encode([texts[0]],normalize_embeddings=True,convert_to_numpy=True,batch_size=2,prompt='');report['warmups'].append({'stage':'embedding','ms':round((time.perf_counter()-warm)*1000,2),'count':1})
   vectors,qvectors,cache=dense_cache(a.cache_dir,metadata,model,texts,query_inputs,docs,qids,a.batch_size)
   runtime['embedding']={'id':model_id,'revision':revision,'device':actual_device,'dtypes':dtype_set,'dimension':dim,'max_seq_length':maxlen,'cache':cache,'load_and_encode_ms':round((time.perf_counter()-start)*1000,2)}
   del model;model=None;gc.collect()
   if a.device=='mps':torch.mps.empty_cache()
   rid,rrev=RERANK[a.reranker];start=time.perf_counter();reranker=CrossEncoder(rid,revision=rrev,device=a.device,model_kwargs={'dtype':getattr(torch,a.dtype)},local_files_only=True,max_length=4096)
   r_dtypes=sorted({str(x.dtype) for x in reranker.parameters() if x.is_floating_point()});rdev=str(reranker.model.device)
   if not rdev.startswith(a.device) or r_dtypes!=['torch.'+a.dtype]:raise RuntimeError('Reranker device/dtype mismatch')
   warm=time.perf_counter();warm_scores=reranker.predict([(qtexts[0],texts[0])],batch_size=2);report['warmups'].append({'stage':'reranker','ms':round((time.perf_counter()-warm)*1000,2),'count':1,'score_finite':bool(np.isfinite(warm_scores).all())})
   runtime['reranker']={'id':rid,'revision':rrev,'device':rdev,'dtypes':r_dtypes,'max_length':reranker.max_length,'load_and_warmup_ms':round((time.perf_counter()-start)*1000,2)}
   runtime['libraries']={x:importlib.metadata.version(x) for x in ['torch','transformers','sentence-transformers','numpy']};runtime['models_loaded_sequentially']=True
 except Exception as e:
  startup_error={'type':type(e).__name__,'message':str(e)}
  if model is not None:
   del model;model=None;gc.collect()
   if a.device=='mps':
    import torch;torch.mps.empty_cache()
 report['model_runtime']=runtime
 for index,qid in enumerate(qids):
  begin=time.perf_counter();row={'query_id':qid,'failure':None,'ranking':[],'source_candidates':{},'timings_ms':{}}
  try:
   if startup_error:raise RuntimeError('model setup failed: '+startup_error['type']+': '+startup_error['message'])
   tick=time.perf_counter();lexical=[(doc,score) for doc,score in bm25.search(qtexts[index],101) if doc!=qid][:100];row['timings_ms']['bm25']=round((time.perf_counter()-tick)*1000,2);row['source_candidates']['bm25']=[{'doc_id':doc,'score':score} for doc,score in lexical]
   if a.mode=='bm25': ranked=[{'doc_id':doc,'score':-score,'sources':[{'source':'bm25','rank':i+1,'raw_score':score}]} for i,(doc,score) in enumerate(lexical)]
   else:
    tick=time.perf_counter();scores=vectors@qvectors[index];order=sorted(range(len(docs)),key=lambda i:(float(scores[i]),docs[i]),reverse=True);dense=[(docs[i],float(scores[i])) for i in order if docs[i]!=qid][:100];row['timings_ms']['exact_cosine']=round((time.perf_counter()-tick)*1000,2);row['source_candidates']['dense']=[{'doc_id':doc,'score':score} for doc,score in dense];row['timings_ms']['query_embedding_cached']=cache['hit'];row['timings_ms']['query_embedding_batch']=next((b for b in cache['query_batches'] if b['start']<=index<b['start']+b['size']),None)
    fused,fs,sources=fuse({'bm25':lexical,'dense':dense});candidate=fused[:60]
    pairs=[(qtexts[index],input_text(corpus[doc])) for doc in candidate]
    prompt=reranker.prompts.get(reranker.default_prompt_name) if reranker.default_prompt_name else None
    for pair in pairs:
     encoded=reranker.preprocess([pair],prompt=prompt,processing_kwargs={'text':{'truncation':False},'chat_template':{'truncation':False}})
     if int(encoded['attention_mask'][0].sum())>reranker.max_length:raise ValueError('Reranker input exceeds token limit; no silent truncation')
    tick=time.perf_counter();rs=reranker.predict(pairs,batch_size=a.batch_size)
    if len(rs)!=len(candidate) or any(not math.isfinite(float(x)) for x in rs):raise ValueError('Invalid reranker scores')
    values={doc:float(score) for doc,score in zip(candidate,rs)};ranked=[{'doc_id':doc,'score':values[doc],'rrf_score':fs[doc],'sources':sources[doc]} for doc in sorted(candidate,key=lambda x:(values[x],x),reverse=True)];row['timings_ms']['rerank']=round((time.perf_counter()-tick)*1000,2)
   row['ranking']=ranked;row['metrics']=evaluate(qid,rels[qid],[r['doc_id'] for r in ranked])
  except Exception as e:
   row['failure']={'type':type(e).__name__,'message':str(e)};row['metrics']=evaluate(qid,rels[qid],[])
  row['timings_ms']['ranking_total']=round((time.perf_counter()-begin)*1000,2);report['results'].append(row)
 bm25.close();failures=sum(x['failure'] is not None for x in report['results']);report['failures']=failures;report['startup_error']=startup_error;report['query_count_recorded']=len(report['results']);report['coverage']=len(report['results'])/len(qids);report['metrics']={key:sum(x['metrics'][key] for x in report['results'])/len(qids) for key in ['ndcg_at_10','recall_at_20','mrr_at_10']};report['eligibility']='comparable' if failures==0 else 'failed_run_not_comparable';report['total_wall_ms']=round((time.perf_counter()-run_start)*1000,2)
 if reranker is not None:
  del reranker;gc.collect()
  if a.device=='mps':
   import torch;torch.mps.empty_cache()
 write_json(a.output,report);print(json.dumps({'dataset':manifest['dataset'],'split':a.split,'failures':failures,'queries':len(qids),'metrics':report['metrics'],'output_sha256':filehash(a.output)}));return 1 if failures else 0
if __name__=='__main__':raise SystemExit(main())
