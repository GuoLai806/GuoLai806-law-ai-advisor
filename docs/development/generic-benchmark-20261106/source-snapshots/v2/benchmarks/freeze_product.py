"""Freeze product-path evaluation configuration before any model scoring."""
import argparse,json,platform,importlib.metadata
from pathlib import Path
from .prepare import filehash
from .run import EMBED,RERANK,INSTRUCTION
from .run_product import source_hashes

def main():
 p=argparse.ArgumentParser();p.add_argument('--scifact-manifest',required=True);p.add_argument('--nfcorpus-manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 target=Path(a.output)
 if target.exists():raise FileExistsError('Frozen configuration already exists; do not overwrite')
 manifests={}
 for name,path in [('scifact',a.scifact_manifest),('nfcorpus',a.nfcorpus_manifest)]:
  m=json.loads(Path(path).read_text())
  if m.get('dataset')!=name or m.get('validation',{}).get('count')!=32:raise ValueError('Unexpected validation manifest')
  manifests[name]=filehash(path)
 import jieba
 with jieba.get_dict_file() as f: dictionary_sha256=filehash(f.name)
 data={'version':1,'jieba_dictionary_sha256':dictionary_sha256,'platform':{'system':platform.system(),'machine':platform.machine()},'library_versions':{x:importlib.metadata.version(x) for x in ('torch','transformers','sentence-transformers','numpy','jieba')},'purpose':'frozen before model runs; selection only from validation','datasets':manifests,'source_sha256':source_hashes(),'embedding_models':EMBED,'rerank_models':RERANK,'device':'mps','dtype':'float16','batch_size':2,'candidate_top_per_source':40,'candidate_cap':60,'rrf_k':60,'embedding_input':'product document_embedding_text','query_instruction_qwen':INSTRUCTION,'dense_backend':'exact cosine reference; not OpenViking','ranking_stage':'product shared rank_candidates before evidence budget'}
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'freeze_sha256':filehash(target)}))
if __name__=='__main__':main()
