import json,os,tempfile
from pathlib import Path
from types import SimpleNamespace
from fastapi.testclient import TestClient
import evidence_rag.api as api

def main():
    # This example models a configured generator whose context size is smaller
    # than a measured prompt: generation_allowed has no context-window input.
    bundle={'budget':{'state':'measured','used_tokens':5000,'omitted':[]},'status':'ready',
            'query_plan':{'profile':'general','mode':'source_lookup','applicability_review':False},
            'evidence':[{'reranker_score':1.0}],'degraded':False}
    allowed=api.generation_allowed(bundle,SimpleNamespace(budget_mode='tokenizer'),SimpleNamespace(state=SimpleNamespace(generator=SimpleNamespace(max_output_tokens=None))))
    out={'measured_gate':{'used_prompt_tokens':5000,'hypothetical_model_window':4096,'max_output_tokens':None,'allowed':allowed,'limitation':'gate has no configured model window; illustrative fake inputs, no provider called'}}
    with tempfile.TemporaryDirectory(prefix='rag-ingest-probe-') as temp:
        root=Path(temp);target=root/'huge.txt'
        with target.open('wb') as stream: stream.truncate(60_000_000)
        os.environ.update(EVIDENCE_RAG_DB=str(root/'db'),EVIDENCE_RAG_AUTH=json.dumps({'token':'admin'}),EVIDENCE_RAG_ADMINS='admin',EVIDENCE_RAG_INGEST_DIR=str(root))
        original=Path.read_bytes
        def guard(path):
            if path.resolve()==target.resolve(): raise RuntimeError('READ_ATTEMPTED_BEFORE_SIZE_CHECK')
            return original(path)
        Path.read_bytes=guard
        try:
            with TestClient(api.app) as client:
                try:
                    r=client.post('/api/v2/ingest-jobs',headers={'Authorization':'Bearer token'},json={'filename':'huge.txt','manifest':{},'idempotency_key':'probe-key'})
                    out['ingest']={'response_status':r.status_code,'text':r.text}
                except RuntimeError as exc:out['ingest']={'exception':str(exc),'logical_file_bytes':target.stat().st_size,'real_file_read':False}
        finally:Path.read_bytes=original
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
