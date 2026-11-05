"""BEIR using product catalogue, shared ranker and explicit exact-cosine reference dense backend."""
import argparse,gc,hashlib,importlib.metadata,json,math,time,platform
from pathlib import Path
from .core import load_dataset
from .metrics import evaluate
from .prepare import filehash
from .product import ProductIndex,REVISION
from .run import EMBED,RERANK,INSTRUCTION,dense_cache,write_json,digest_bytes,now
from evidence_rag.retrieval import rank_candidates,document_embedding_text,CrossEncoderReranker

FILES=('benchmarks/run_product.py','benchmarks/product.py','benchmarks/run.py','benchmarks/core.py','benchmarks/metrics.py','benchmarks/prepare.py','benchmarks/select.py','benchmarks/freeze_product.py','evidence_rag/catalog.py','evidence_rag/retrieval.py','evidence_rag/viking_index.py')
def source_hashes():return {p:filehash(p) for p in FILES}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--manifest',required=True);p.add_argument('--freeze',required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--mode',choices=['product-lexical','product-hybrid'],required=True);p.add_argument('--embedding',choices=EMBED);p.add_argument('--reranker',choices=RERANK);p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--dtype',choices=['float16','float32'],default='float16');p.add_argument('--batch-size',type=int,default=2);p.add_argument('--cache-dir');p.add_argument('--selection');p.add_argument('--validation-report');p.add_argument('--output',required=True);a=p.parse_args()
 if a.batch_size!=2:raise ValueError('Frozen experiment requires batch size 2')
 if a.mode=='product-hybrid' and (not a.embedding or not a.reranker or not a.cache_dir):raise ValueError('Hybrid requires embedding, reranker and cache directory')
 if a.mode=='product-lexical' and (a.embedding or a.reranker):raise ValueError('Lexical mode must not load models')
 if a.cache_dir:
  cache=Path(a.cache_dir).resolve();repo=Path(__file__).resolve().parents[1]
  if cache==repo or repo in cache.parents:raise ValueError('Vector cache must be outside repository')
 freeze=json.loads(Path(a.freeze).read_text());manifest,corpus,queries,rels,qids=load_dataset(a.data_dir,a.manifest,a.split)
 if (a.mode=='product-hybrid' and (a.device!=freeze.get('device') or a.dtype!=freeze.get('dtype'))) or freeze.get('source_sha256')!=source_hashes() or freeze.get('datasets',{}).get(manifest['dataset'])!=filehash(a.manifest) or freeze.get('candidate_top_per_source')!=40 or freeze.get('candidate_cap')!=60 or freeze.get('batch_size')!=2 or freeze.get('embedding_models')!={k:list(v) for k,v in EMBED.items()} or freeze.get('rerank_models')!={k:list(v) for k,v in RERANK.items()}:raise ValueError('Frozen validation configuration does not match current code or dataset')
 versions={x:importlib.metadata.version(x) for x in ('torch','transformers','sentence-transformers','numpy','jieba')}
 import jieba
 with jieba.get_dict_file() as f: dictionary_sha256=filehash(f.name)
 if freeze.get('jieba_dictionary_sha256')!=dictionary_sha256:raise ValueError('Frozen tokenizer dictionary changed')
 if freeze.get('library_versions')!=versions or freeze.get('platform')!={'system':platform.system(),'machine':platform.machine()}:raise ValueError('Frozen environment differs from current environment')
 config={'mode':a.mode,'embedding':a.embedding,'reranker':a.reranker,'device':a.device if a.mode=='product-hybrid' else None,'dtype':a.dtype if a.mode=='product-hybrid' else None,'batch_size':2}
 if a.split=='test':
  if not a.selection or not a.validation_report:raise ValueError('Official test requires frozen selection and validation report')
  sel=json.loads(Path(a.selection).read_text());v=json.loads(Path(a.validation_report).read_text())
  if sel.get('manifest_sha256')!=filehash(a.manifest) or sel.get('pipeline_config_sha256')!=filehash(a.freeze) or sel.get('config')!=config or sel.get('code_sha256')!=source_hashes() or sel.get('validation_report_sha256')!=filehash(a.validation_report) or sel.get('validation_failures')!=0 or v.get('split')!='validation' or v.get('pipeline_config_sha256')!=filehash(a.freeze) or v.get('config')!=config or v.get('failures')!=0 or v.get('eligibility')!='comparable' or len(v.get('results',[]))!=manifest['validation']['count'] or {x.get('query_id') for x in v['results']}!=set(manifest['validation']['query_ids']):raise ValueError('Selection or validation report does not match')
 start=time.perf_counter();docs=sorted(corpus);qtexts=[queries[q]['text'] for q in qids]
 report={'version':1,'dataset':manifest['dataset'],'split':a.split,'manifest_sha256':filehash(a.manifest),'pipeline_config_sha256':filehash(a.freeze),'code_sha256':source_hashes(),'config':config,'corpus_sha256':manifest['corpus_sha256'],'corpus_count':len(docs),'query_count_expected':len(qids),'qids_sha256':digest_bytes(json.dumps(qids,ensure_ascii=False,separators=(',',':')).encode()),'created_utc':now(),'platform':platform.platform(),'ranking_scope':'product shared pre-evidence-assembly candidate order; not end-to-end answer or final evidence metrics','dense_backend':'local exact cosine reference; not OpenViking index','input_format':'evidence_rag.retrieval.document_embedding_text: title + newline + body; no ID or qrel','results':[],'warmups':[]}
 index=embed=reranker=vectors=qvectors=None;runtime={};startup_error=None;cache=None
 try:
  index=ProductIndex(corpus,manifest);report['catalog_stage_ms']=index.stage_ms
  runtime['jieba_version']=importlib.metadata.version('jieba');runtime['jieba_dict_sha256']=dictionary_sha256
  units=[index.unit(doc_id) for doc_id in docs];texts=[document_embedding_text(unit) for unit in units]
  if len(units)!=manifest['corpus_count']:raise ValueError('Full product unit coverage mismatch')
  if a.mode=='product-hybrid':
   import numpy as np,torch
   from sentence_transformers import SentenceTransformer
   if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
   mid,rev=EMBED[a.embedding];tick=time.perf_counter();embed=SentenceTransformer(mid,revision=rev,device=a.device,model_kwargs={'dtype':getattr(torch,a.dtype)},local_files_only=True)
   dtypes=sorted({str(x.dtype) for x in embed.parameters() if x.is_floating_point()});dev=str(embed.device)
   if not dev.startswith(a.device) or dtypes!=['torch.'+a.dtype]:raise RuntimeError('Embedding device/dtype mismatch')
   limit=embed.max_seq_length;query_inputs=[INSTRUCTION+q if a.embedding=='qwen' else q for q in qtexts]
   for text in texts+query_inputs:
    if len(embed.tokenizer(text,add_special_tokens=True,truncation=False)['input_ids'])>limit:raise ValueError('Embedding input exceeds limit; no silent truncation')
   dim=int(embed.get_sentence_embedding_dimension());meta={'corpus_sha256':manifest['corpus_sha256'],'corpus_ids_sha256':digest_bytes(json.dumps(docs,separators=(',',':')).encode()),'input_format':report['input_format'],'input_serialization_sha256':source_hashes()['evidence_rag/retrieval.py'],'model':mid,'revision':rev,'dtype':a.dtype,'device':a.device,'normalized':True,'dimension':dim,'query_hash':digest_bytes(json.dumps(list(zip(qids,query_inputs)),ensure_ascii=False,separators=(',',':')).encode()),'batch_size':2}
   warm=time.perf_counter();embed.encode([texts[0]],normalize_embeddings=True,convert_to_numpy=True,batch_size=2,prompt='');report['warmups'].append({'stage':'embedding','ms':round((time.perf_counter()-warm)*1000,2),'count':1})
   vectors,qvectors,cache=dense_cache(a.cache_dir,meta,embed,texts,query_inputs,docs,qids,2)
   runtime['embedding']={'id':mid,'revision':rev,'device':dev,'dtypes':dtypes,'max_seq_length':limit,'dimension':dim,'corpus_embedded_count':int(vectors.shape[0]),'query_embedded_count':int(qvectors.shape[0]),'cache':cache,'load_and_encode_ms':round((time.perf_counter()-tick)*1000,2)}
   del embed;embed=None;gc.collect()
   if a.device=='mps':torch.mps.empty_cache()
   tick=time.perf_counter();reranker=CrossEncoderReranker(a.reranker,device=a.device,dtype=a.dtype,batch_size=2)
   types=sorted({str(x.dtype) for x in reranker.model.parameters() if x.is_floating_point()});rdev=str(reranker.model.model.device)
   if not rdev.startswith(a.device) or types!=['torch.'+a.dtype]:raise RuntimeError('Reranker device/dtype mismatch')
   warm=time.perf_counter();warm_result=reranker.rank(qtexts[0],[units[0]])
   if len(warm_result)!=1 or not math.isfinite(float(warm_result[0][1])):raise ValueError('Invalid warmup result')
   report['warmups'].append({'stage':'reranker','ms':round((time.perf_counter()-warm)*1000,2),'count':1})
   runtime['reranker']={'id':RERANK[a.reranker][0],'revision':reranker.revision,'device':rdev,'dtypes':types,'max_length':reranker.model.max_length,'load_and_warmup_ms':round((time.perf_counter()-tick)*1000,2)}
   runtime['libraries']={x:importlib.metadata.version(x) for x in ['torch','transformers','sentence-transformers','numpy']};runtime['models_loaded_sequentially']=True
 except Exception as e:
  startup_error={'type':type(e).__name__,'message':str(e).replace(str(Path.home()),'<HOME>')}
  if embed is not None:
   del embed;embed=None;gc.collect()
   if a.device=='mps':
    import torch;torch.mps.empty_cache()
 report['model_runtime']=runtime
 for idx,qid in enumerate(qids):
  tick=time.perf_counter();row={'query_id':qid,'failure':None,'ranking':[],'source_candidates':{},'timings_ms':{}}
  try:
   if startup_error:raise RuntimeError('setup failed: '+startup_error['type']+': '+startup_error['message'])
   t=time.perf_counter();lex=[(uid,float(score)) for uid,score in index.lexical(qtexts[idx])];exact=index.exact(qtexts[idx]);row['timings_ms']['product_lexical']=round((time.perf_counter()-t)*1000,2)
   dense=[]
   if a.mode=='product-hybrid':
    t=time.perf_counter();scores=vectors@qvectors[idx];indices=sorted(range(len(docs)),key=lambda i:(float(scores[i]),docs[i]),reverse=True)
    dense=[(index.doc_to_unit[docs[i]],float(scores[i])) for i in indices[:40]];row['timings_ms']['exact_cosine']=round((time.perf_counter()-t)*1000,2);row['timings_ms']['query_embedding_cached']=cache['hit'];row['timings_ms']['query_embedding_batch']=next((b for b in cache['query_batches'] if b['start']<=idx<b['start']+b['size']),None)
   row['source_candidates']={'lexical':[{'doc_id':index.unit_to_doc[u],'score':s} for u,s in lex],'dense':[{'doc_id':index.unit_to_doc[u],'score':s} for u,s in dense],'exact_title':[index.unit_to_doc[u] for u in exact]}
   t=time.perf_counter();ranked=rank_candidates(index.cat,REVISION,qtexts[idx],lex,dense,exact=exact,reranker=reranker,limit=60);row['timings_ms']['shared_rank_and_rerank']=round((time.perf_counter()-t)*1000,2)
   row['ranking']=[{'doc_id':unit['document']['doc_id'],'unit_id':unit['unit_id'],'rrf_score':ranked['rrf_scores'].get(unit['unit_id']),'reranker_score':ranked['reranker_scores'].get(unit['unit_id']),'sources':ranked['sources'].get(unit['unit_id'],[])} for unit in ranked['candidates']]
   row['metrics']=evaluate(qid,rels[qid],[x['doc_id'] for x in row['ranking']])
  except Exception as e:
   row['failure']={'type':type(e).__name__,'message':str(e).replace(str(Path.home()),'<HOME>')};row['metrics']=evaluate(qid,rels[qid],[])
  row['timings_ms']['ranking_total']=round((time.perf_counter()-tick)*1000,2);report['results'].append(row)
 if index:index.close()
 if reranker is not None:
  del reranker;gc.collect()
  if a.device=='mps':
   import torch;torch.mps.empty_cache()
 failures=sum(r['failure'] is not None for r in report['results']);report.update(failures=failures,startup_error=startup_error,query_count_recorded=len(report['results']),coverage=len(report['results'])/len(qids),eligibility='comparable' if failures==0 else 'failed_run_not_comparable',total_wall_ms=round((time.perf_counter()-start)*1000,2),metrics={k:sum(r['metrics'][k] for r in report['results'])/len(qids) for k in ('ndcg_at_10','recall_at_20','mrr_at_10')})
 write_json(a.output,report);print(json.dumps({'dataset':manifest['dataset'],'split':a.split,'failures':failures,'queries':len(qids),'metrics':report['metrics'],'output_sha256':filehash(a.output)}));return 1 if failures else 0
if __name__=='__main__':raise SystemExit(main())
