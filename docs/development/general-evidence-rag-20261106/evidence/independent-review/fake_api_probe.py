import asyncio,json,os,tempfile
from pathlib import Path
import httpx
from fastapi.testclient import TestClient
from evidence_rag.answer import OpenAIStream,validate_citations
import evidence_rag.api as api

async def check_finish(reason):
    body='data: '+json.dumps({'choices':[{'delta':{'content':'partial [E1]'},'finish_reason':None}]})+'\n\n'
    body+='data: '+json.dumps({'choices':[{'delta':{},'finish_reason':reason}]})+'\n\ndata: [DONE]\n\n'
    transport=httpx.MockTransport(lambda req:httpx.Response(200,text=body))
    generator=OpenAIStream('http://fake/v1','fake','',transport=transport)
    evidence=[{'citation_id':'E1','title':'Example','text':'No model calls were made.'}]
    try:
        parts=[event async for event in generator.stream_events('question',evidence)]
        return {'reason':reason,'accepted_without_exception':True,'events':parts,'validation':validate_citations(''.join(x.get('delta','') for x in parts),evidence)}
    except Exception as exc:
        return {'reason':reason,'accepted_without_exception':False,'error':str(exc)}

class Generator:
    async def stream(self,query,evidence):
        yield 'Unverified factual claim [E999]'

def main():
    out={'finish_reason':[asyncio.run(check_finish(reason)) for reason in ('length','content_filter','tool_calls')]}
    with tempfile.TemporaryDirectory(prefix='rag-readonly-probe-') as temp:
        os.environ['EVIDENCE_RAG_DB']=str(Path(temp)/'test.sqlite')
        os.environ['EVIDENCE_RAG_AUTH']=json.dumps({'token':'test-user'})
        os.environ['EVIDENCE_RAG_STORE_CONVERSATIONS']='1'
        original_get=api.get_bundle; original_check=api.check_access; original_allow=api.generation_allowed
        async def fake_bundle(body,request,user,pinned_revision=None):
            api.bind(request.app.state.catalog.db,body.session_id,user)
            return {'trace_id':'fake','status':'ready','query_plan':{'profile':'general','mode':'source_lookup','applicability_review':False},'evidence':[{'citation_id':'E1','title':'Source','text':'valid evidence'}],'degraded':False,'budget':{'state':'measured','omitted':[]},'corpus_revision':'r1'}
        def check(body,request,user): api.bind(request.app.state.catalog.db,body.session_id,user); return 'r1'
        api.get_bundle=fake_bundle;api.check_access=check;api.generation_allowed=lambda *args:True
        try:
            with TestClient(api.app) as client:
                api.app.state.generator=Generator(); body={'query':'question','session_id':'test-session'}; headers={'Authorization':'Bearer token'}
                normal=client.post('/api/v2/chat',json=body,headers=headers)
                streamed=client.post('/api/v2/chat/stream',json=body,headers=headers)
                turns=client.get('/api/v2/sessions/test-session/turns',headers=headers)
                out['invalid_citation']={'normal':normal.json(),'sse':streamed.text,'turns':turns.json()}
        finally:api.get_bundle=original_get;api.check_access=original_check;api.generation_allowed=original_allow
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
