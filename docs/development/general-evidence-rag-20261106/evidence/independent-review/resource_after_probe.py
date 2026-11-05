import json,os,tempfile
from pathlib import Path
from types import SimpleNamespace
from fastapi.testclient import TestClient
import evidence_rag.api as api
from evidence_rag.ingest import load_documents
os.environ.update(EVIDENCE_RAG_MODEL_CONTEXT_WINDOW='4096',EVIDENCE_RAG_FRAMING_RESERVE='256',EVIDENCE_RAG_OUTPUT_RESERVE='1024')
b={'budget':{'state':'measured','used_tokens':5000,'omitted':[]},'status':'ready','query_plan':{'profile':'general','mode':'source_lookup','applicability_review':False},'evidence':[{}],'degraded':False}
a=SimpleNamespace(state=SimpleNamespace(generator=SimpleNamespace(max_output_tokens=512)))
over=api.generation_allowed(b,None,a);b['budget']['used_tokens']=2000;within=api.generation_allowed(b,None,a);a.state.generator.max_output_tokens=None;missing=api.generation_allowed(b,None,a)
out={'gate':{'window':4096,'framing_reserve':256,'output_reserve':1024,'generator_max':512,'prompt5000_allowed':over,'prompt2000_allowed':within,'unbounded_output_allowed':missing}}
with tempfile.TemporaryDirectory(prefix='rag-size-retest-') as tmp:
 root=Path(tmp); target=root/'huge.txt'
 with target.open('wb') as f:f.truncate(110_000_000)
 os.environ.update(EVIDENCE_RAG_DB=str(root/'db'),EVIDENCE_RAG_AUTH=json.dumps({'token':'admin'}),EVIDENCE_RAG_ADMINS='admin',EVIDENCE_RAG_INGEST_DIR=str(root))
 original_open=os.open; attempted=[]
 def guard(path,*a,**kw):
  if Path(path).resolve()==target.resolve():attempted.append(str(path));raise RuntimeError('source opened')
  return original_open(path,*a,**kw)
 os.open=guard
 try:
  with TestClient(api.app) as client:
   response=client.post('/api/v2/ingest-jobs',headers={'Authorization':'Bearer token'},json={'filename':'huge.txt','manifest':{},'idempotency_key':'probe-key'})
   out['http']={'status':response.status_code,'source_open_attempts':len(attempted)}
  try:load_documents(target);out['cli']={'rejected':False}
  except Exception as exc:out['cli']={'rejected':True,'type':type(exc).__name__,'error':str(exc),'source_open_attempts':len(attempted)}
 finally:os.open=original_open
print(json.dumps(out,indent=2))
