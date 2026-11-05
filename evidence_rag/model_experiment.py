"""Explicit offline 2x2 model experiments; loads embedding and reranker sequentially."""
import argparse, gc, hashlib, importlib.metadata, json, math, pathlib, platform, time
from .catalog import Catalog
from .retrieval import CrossEncoderReranker, rank_candidates, document_embedding_text

EMBED={'bge':('BAAI/bge-m3','5617a9f61b028005a4858fdac845db406aefb181'), 'qwen':('Qwen/Qwen3-Embedding-4B','5cf2132abc99cad020ac570b19d031efec650f2b')}
INSTRUCTION='Instruct: Retrieve relevant legal passages for this question\nQuery: '

def main():
    p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--revision',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--embedding',choices=list(EMBED),required=True);p.add_argument('--reranker',choices=['bge','qwen'],required=True);p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--dtype',choices=['float16','float32'],default='float32');p.add_argument('--batch-size',type=int,default=2);p.add_argument('--limit',type=int,default=300)
    a=p.parse_args()
    if a.batch_size<1 or a.limit<1: raise ValueError('batch-size and limit must be positive')
    import torch
    from sentence_transformers import SentenceTransformer
    if a.device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('requested MPS is unavailable; no CPU fallback')
    dtype=getattr(torch,a.dtype);model_id,revision=EMBED[a.embedding]
    cat=Catalog(a.db);rows=cat.db.execute('SELECT unit_id FROM units WHERE revision=? ORDER BY unit_id',(a.revision,)).fetchall();docs=[cat.get(a.revision,r[0]) for r in rows]
    if not docs: raise ValueError('empty corpus')
    data=json.loads(pathlib.Path(a.dataset).read_text());queries=data['queries'][:a.limit]
    profile=json.loads(cat.revision(a.revision)['manifest']).get('profile','legal')
    model=SentenceTransformer(model_id,revision=revision,device=a.device,model_kwargs={'dtype':dtype},local_files_only=True)
    actual_device=str(model.device); actual_dtypes=sorted({str(x.dtype) for x in model.parameters() if getattr(x,'is_floating_point',lambda:True)()})
    if not actual_device.startswith(a.device) or 'torch.'+a.dtype not in actual_dtypes: raise RuntimeError('embedding device or dtype differs from requested configuration')
    documents=[document_embedding_text(d) for d in docs]
    query_texts=[(INSTRUCTION+item['query']) if a.embedding=='qwen' else item['query'] for item in queries]
    limit=model.max_seq_length
    if not limit: raise ValueError('embedding tokenizer limit unknown')
    for text in documents+query_texts:
        if len(model.tokenizer(text,add_special_tokens=True,truncation=False)['input_ids'])>limit: raise ValueError('embedding input exceeds model limit; no silent truncation')
    index_start=time.perf_counter()
    vectors=model.encode(documents,normalize_embeddings=True,convert_to_numpy=True,batch_size=a.batch_size,prompt='')
    corpus_embedding_ms=round((time.perf_counter()-index_start)*1000,3)
    query_vectors=[]; query_embedding_ms=[]
    for text in query_texts:
        started=time.perf_counter()
        query_vectors.append(model.encode([text],normalize_embeddings=True,convert_to_numpy=True,batch_size=a.batch_size,prompt='')[0])
        query_embedding_ms.append(round((time.perf_counter()-started)*1000,3))
    embed_meta={'device':actual_device,'parameter_dtypes':actual_dtypes,'max_seq_length':limit,'dimension':int(vectors.shape[1]),'normalized':True,'corpus_embedding_ms':corpus_embedding_ms,'query_instruction':INSTRUCTION if a.embedding=='qwen' else None}
    del model;gc.collect()
    if a.device=='mps':torch.mps.empty_cache()
    reranker=CrossEncoderReranker(a.reranker,device=a.device,dtype=a.dtype,batch_size=a.batch_size)
    rerank_dtypes=sorted({str(x.dtype) for x in reranker.model.parameters() if getattr(x,'is_floating_point',lambda:True)()});rerank_device=str(reranker.model.device)
    if not rerank_device.startswith(a.device) or 'torch.'+a.dtype not in rerank_dtypes: raise RuntimeError('reranker device or dtype differs from requested configuration')
    byid={d['unit_id']:d for d in docs};records=[]
    for item,q,embed_ms in zip(queries,query_vectors,query_embedding_ms):
        query=item['query'];start=time.perf_counter();indices=(vectors@q).argsort()[::-1][:40]
        dense_pairs=[(docs[int(i)]['unit_id'],float(vectors[int(i)]@q)) for i in indices];exact_ids=cat.exact_documents(a.revision,query) if profile=='general' else cat.exact(a.revision,query);lexical=cat.lexical_scored(a.revision,query)
        expected=item.get('expected_ids',item.get('expected_labels',[]))
        try:
            result=rank_candidates(cat,a.revision,query,lexical,dense_pairs,exact_ids,reranker);sources=result['sources'];ranked=[(d,result['reranker_scores'][d['unit_id']]) for d in result['candidates']];labels=[d['document']['doc_id'] if profile=='general' else d['label'] for d,score in ranked]
            hits=len(set(labels[:20])&set(expected));dcg=sum(1/math.log2(i+2) for i,x in enumerate(labels[:10]) if x in expected);ideal=sum(1/math.log2(i+2) for i in range(min(len(expected),10)))
            records.append({'query':query,'expected':expected,'ranking':[{'label':d['label'] if profile=='legal' else None,'document_id':d['document']['doc_id'] if profile=='general' else None,'score':s,'retrieval_sources':sources[d['unit_id']]} for d,s in ranked],'recall_at_20':hits/len(expected) if expected else None,'ndcg_at_10':dcg/ideal if ideal else None,'query_embedding_ms':embed_ms,'ranking_ms':round((time.perf_counter()-start)*1000,3),'latency_ms':round(embed_ms+(time.perf_counter()-start)*1000,3),'failure':None if not expected or hits==len(expected) else 'miss'})
        except Exception as exc:
            records.append({'query':query,'expected':expected,'ranking':[],'recall_at_20':None,'ndcg_at_10':None,'query_embedding_ms':embed_ms,'ranking_ms':round((time.perf_counter()-start)*1000,3),'latency_ms':round(embed_ms+(time.perf_counter()-start)*1000,3),'failure':type(exc).__name__+': '+str(exc)})
    def mean(key):
        values=[x[key] for x in records if x.get(key) is not None]
        return sum(values)/len(values) if values else None
    out={'kind':data.get('kind'),'embedding':model_id,'embedding_revision':revision,'embedding_runtime':embed_meta,'reranker':a.reranker,'reranker_revision':reranker.revision,'reranker_runtime':{'device':rerank_device,'parameter_dtypes':rerank_dtypes,'max_length':reranker.model.max_length},'requested_device':a.device,'requested_dtype':a.dtype,'batch_size':a.batch_size,'models_loaded_sequentially':True,'corpus_revision':a.revision,'corpus_manifest':json.loads(cat.revision(a.revision)['manifest']),'dataset_sha256':hashlib.sha256(pathlib.Path(a.dataset).read_bytes()).hexdigest(),'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path('evidence_rag').glob('*.py'))},'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'platform':platform.platform(),'python':platform.python_version(),'sentence_transformers_version':importlib.metadata.version('sentence-transformers'),'torch_version':importlib.metadata.version('torch'),'transformers_version':importlib.metadata.version('transformers'),'mean_recall_at_20':mean('recall_at_20'),'mean_ndcg_at_10':mean('ndcg_at_10'),'scored_queries':sum(x['recall_at_20'] is not None for x in records),'failed_queries':sum(bool(x['failure']) and x['failure']!='miss' for x in records),'latency_note':'query embedding time plus later ranking time, excludes model loading and corpus indexing; not contiguous end-to-end timing','results':records}
    pathlib.Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__': main()
