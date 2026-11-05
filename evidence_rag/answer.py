"""Streaming model protocol; citations are checked structurally, never treated as claim verification."""
import asyncio, json, re
import httpx

SYSTEM = 'Only answer from the provided evidence; cite evidence IDs, preserve conditions and note uncertainty. Do not invent sources or unsupported claims.'
LEGAL_SYSTEM = '仅根据给定证据解释原文；说明条件与例外，不得编造法条。每项法律主张标注证据编号。法律版本、事实及过渡规则未经核验时，不得断言具体案件的法律适用或给出确定结论。'
def messages_for(query, evidence):
    facts='\n\n'.join(f"[{e['citation_id']}] {e['title']} {e.get('article_label','')}\n{e['text']}" for e in evidence)
    return [{'role':'system','content':LEGAL_SYSTEM if evidence and evidence[0].get('profile')=='legal' else SYSTEM},{'role':'user','content':f'问题：{query}\n证据：\n{facts}'}]

class OpenAIStream:
    def __init__(self, base_url, model, api_key, timeout=30, transport=None, include_usage=False, max_output_tokens=None):
        self.base_url, self.model, self.api_key, self.timeout, self.transport = base_url.rstrip('/'), model, api_key, timeout, transport
        self.include_usage=include_usage
        if max_output_tokens is not None and (not isinstance(max_output_tokens,int) or isinstance(max_output_tokens,bool) or max_output_tokens<1): raise ValueError('max_output_tokens must be a positive integer')
        self.max_output_tokens=max_output_tokens
    async def stream_events(self, query, evidence):
        if not evidence: raise ValueError('evidence required')
        messages=messages_for(query,evidence)
        completed=False; finish_reason=None
        async with asyncio.timeout(self.timeout):
            async with httpx.AsyncClient(timeout=self.timeout,transport=self.transport,trust_env=False) as client:
                request_body={'model':self.model,'messages':messages,'stream':True}
                if self.include_usage: request_body['stream_options']={'include_usage':True}
                if self.max_output_tokens is not None: request_body['max_tokens']=self.max_output_tokens
                async with client.stream('POST', self.base_url+'/chat/completions',headers={'Authorization':'Bearer '+self.api_key},json=request_body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'): continue
                        data=line[5:].strip()
                        if data=='[DONE]':
                            completed=True; break
                        payload=json.loads(data); choices=payload.get('choices',[])
                        if choices:
                            choice=choices[0]; reason=choice.get('finish_reason')
                            if reason is not None:
                                if reason!='stop': raise RuntimeError('model stream incomplete: '+str(reason))
                                finish_reason=reason
                            changes=choice.get('delta',{})
                            if changes.get('tool_calls') or changes.get('function_call'): raise RuntimeError('unexpected model tool call')
                            delta=changes.get('content')
                            if delta: yield {'type':'delta','delta':delta}
                        usage=payload.get('usage')
                        if isinstance(usage,dict):
                            safe={k:v for k,v in usage.items() if k in ('prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens') and isinstance(v,int) and not isinstance(v,bool) and v>=0}
                            if safe: yield {'type':'usage','usage':safe}
        if not completed or finish_reason!='stop': raise RuntimeError('model stream ended without successful completion marker')

    async def stream(self, query, evidence):
        """Compatibility interface for consumers that only accept text deltas."""
        async for event in self.stream_events(query,evidence):
            if event['type']=='delta': yield event['delta']

async def provider_events(provider,query,evidence):
    if callable(getattr(provider,'stream_events',None)):
        async for event in provider.stream_events(query,evidence): yield event
    else:
        async for delta in provider.stream(query,evidence): yield {'type':'delta','delta':delta}

def validate_citations(answer, evidence):
    known={x['citation_id'] for x in evidence}; cited=set(re.findall(r'\[(E\d+)\]',answer)); unknown=sorted(cited-known)
    return {'status':'structurally_valid' if cited and not unknown else 'invalid','unknown_citations':unknown,'cited':sorted(cited),'claim_support':'not_verified'}
