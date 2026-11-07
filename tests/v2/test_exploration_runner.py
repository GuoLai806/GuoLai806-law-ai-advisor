import hashlib
import httpx
import pytest
from benchmarks.run_user_flow import ClockTracker,explore_before_answer
from evidence_rag.exploration import explore_bundle

@pytest.mark.asyncio
@pytest.mark.parametrize('failure',[None,'degraded','read404','corrupt'])
async def test_preflight_preserves_observed_responses_and_full_hash_checks(failure):
    text='Complete long source. '*30
    sha=hashlib.sha256(text.encode()).hexdigest()
    evidence={'unit_id':'a'*32,'title':'source','text':text,'content_sha256':sha,'source_locator':{}}
    preview=explore_bundle({'corpus_revision':'r1','evidence':[evidence],'degraded':failure=='degraded',
                            'degraded_reasons':[],'retrieval_ms':1})
    calls=[]
    async def handle(request):
        calls.append(str(request.url))
        if request.url.path=='/api/v2/explore':return httpx.Response(200,json=preview)
        return httpx.Response(404 if failure=='read404' else 200,json={'text':'changed' if failure=='corrupt' else text,'revision':'r1'})
    record={}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle),trust_env=False) as client:
        if failure:
            with pytest.raises((ValueError,httpx.HTTPStatusError)):
                await explore_before_answer(client,'http://fixture',{},ClockTracker(),record)
        else:
            assert await explore_before_answer(client,'http://fixture',{},ClockTracker(),record) is record
            assert record['original_verified'] and len(calls)==2
    assert record['preview_response_text'] and record['preview_received']['wall_ns']
    if failure!='degraded':assert record['original_response_text']
    if failure:assert not record.get('original_verified')
