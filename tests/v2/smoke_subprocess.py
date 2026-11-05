"""Starts a real loopback-only service with disposable synthetic data and identities."""
import json, os, socket, subprocess, sys, tempfile, time
from pathlib import Path
import httpx
from legal_rag.catalog import Catalog

def main():
    results=[]; start=time.time()
    with tempfile.TemporaryDirectory(prefix='law-smoke-') as temp:
        path=str(Path(temp)/'db.sqlite3'); c=Catalog(path)
        c.stage(['第一条 测试资料，不是法律原文。\n第二条 测试例外。'],{'doc_id':'smoke','version_id':'v1','title':'合成测试','verification_state':'approved','source_kind':'synthetic'},'fixture'); c.activate('fixture',None); c.db.close()
        env={**os.environ,'LEGAL_RAG_DB':path,'LEGAL_RAG_AUTH':json.dumps({'local-test-a':'a','local-test-b':'b'})}
        with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        process=subprocess.Popen([sys.executable,'-m','uvicorn','legal_rag.api:app','--host','127.0.0.1','--port',str(port)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            base=f'http://127.0.0.1:{port}'
            for _ in range(100):
                try:
                    if httpx.get(base+'/health/live',timeout=.2,trust_env=False).status_code==200: break
                except httpx.HTTPError: time.sleep(.05)
            else: raise RuntimeError('startup failed')
            body={'query':'第一条','session_id':'session1','event_date':'2025-01-01'}
            checks=[('live',httpx.get(base+'/health/live',trust_env=False).status_code,200),('readiness',httpx.get(base+'/health/ready',trust_env=False).status_code,503),('unauthenticated',httpx.post(base+'/api/v2/retrieve',json=body,trust_env=False).status_code,401),('retrieve',httpx.post(base+'/api/v2/retrieve',json=body,headers={'Authorization':'Bearer local-test-a'},trust_env=False).status_code,200),('cross_user',httpx.post(base+'/api/v2/retrieve',json=body,headers={'Authorization':'Bearer local-test-b'},trust_env=False).status_code,403)]
            stream=httpx.post(base+'/api/v2/chat/stream',json=body,headers={'Authorization':'Bearer local-test-a'},trust_env=False); checks.append(('stream_events', 'event: evidence_ready' in stream.text and 'event: completed' in stream.text,True))
            results=[{'check':name,'actual':got,'expected':expected,'pass':got==expected} for name,got,expected in checks]
            if not all(x['pass'] for x in results): raise AssertionError(results)
        finally:
            process.terminate()
            try: logs=process.communicate(timeout=5)[0]
            except subprocess.TimeoutExpired: process.kill(); logs=process.communicate()[0]
    output={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(start)),'corpus_kind':'synthetic','checks':results,'process_exit':process.returncode,'server_log':logs}
    Path('docs/development/legal-rag-20261105/evidence/subprocess-smoke.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':len(results),'pass':all(x['pass'] for x in results),'exit':process.returncode}))
if __name__=='__main__': main()
