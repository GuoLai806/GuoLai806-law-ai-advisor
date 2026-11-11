"""Streaming model protocol; citations are checked structurally, never treated as claim verification."""
import asyncio, json, re, os
import httpx
from evidence_rag.grounding import GroundingRejected, MAX_BUFFER_BYTES, parse_grounded_answer

SYSTEM = 'Only answer from the provided evidence; cite evidence IDs, preserve conditions and note uncertainty. Do not invent sources or unsupported claims.'
LEGAL_SYSTEM = '仅根据给定证据解释原文；说明条件与例外，不得编造法条。每项法律主张标注证据编号。法律版本、事实及过渡规则未经核验时，不得断言具体案件的法律适用或给出确定结论。'
GROUNDED_OUTPUT = '''
Output override: The JSON format below overrides any previous first-line,
prose-format, or inline-citation instructions. All factual restrictions remain.
Return only a JSON object, without markdown fences, with exactly these keys:
{"decision":"SUPPORT","answer":"brief explanation","evidence":[{"id":"E1","quote":"exact source passage"}]}
The decision must be exactly SUPPORT, CONTRADICT, NOT_ENOUGH_INFO or ANSWER.
For a scientific claim, use SUPPORT only when the evidence establishes every
material part; CONTRADICT only when evidence establishes the opposite; otherwise
NOT_ENOUGH_INFO. Use ANSWER for an ordinary information question supported by the
given evidence. Check the exact intervention, population, outcome, direction,
negation, causal relation, conditions and material restrictions. Merely related
evidence or a missing material relation means NOT_ENOUGH_INFO. Do not use external
knowledge to fill gaps. Preserve what the original study says, including limits.
Select only the minimum sufficient source set, at most four sources, each ID
once. Copy an exact 12-1200 character passage from that source's title or text;
do not invent, paraphrase, change case, punctuation or negation in the quote.
SUPPORT, CONTRADICT and ANSWER require at least one such quote.
NOT_ENOUGH_INFO may use an empty evidence list. Quotes must be relevant to the
reasoning; occurrence of a passage alone does not establish semantic support.
Keep the answer brief and naturally readable. Its inline citations, if any,
must use only selected evidence IDs; an empty evidence list means no inline
citations. Do not place the decision token at the start of the answer text.
'''

class ModelStreamIncomplete(RuntimeError):
    def __init__(self, reason):
        self.reason = reason if reason in ('length','content_filter','tool_calls','missing_stop','missing_done') else 'invalid_completion'
        super().__init__('model stream ended without successful completion marker: '+self.reason)

def _answer_format(answer_format):
    if answer_format not in ('text','grounded_json'):
        raise ValueError('answer_format must be text or grounded_json')
    return answer_format

def messages_for(query, evidence, answer_format=None):
    if answer_format is None:
        answer_format=os.environ.get('EVIDENCE_RAG_ANSWER_FORMAT',os.environ.get('LEGAL_RAG_ANSWER_FORMAT','text'))
    _answer_format(answer_format)
    facts='\n\n'.join(f"[{e['citation_id']}] {e['title']} {' / '.join(e.get('structure') or [])} {e.get('article_label','')}\n{e['text']}" for e in evidence)
    legal=bool(evidence and evidence[0].get('profile')=='legal')
    system=LEGAL_SYSTEM if legal else os.environ.get('EVIDENCE_RAG_GENERAL_SYSTEM_PROMPT',SYSTEM)
    if not legal and answer_format=='grounded_json': system+='\n'+GROUNDED_OUTPUT
    return [{'role':'system','content':system},{'role':'user','content':f'问题：{query}\n证据：\n{facts}'}]

class OpenAIStream:
    def __init__(self, base_url, model, api_key, timeout=30, transport=None, include_usage=False, max_output_tokens=None, temperature=None, thinking=None, reasoning_effort=None, empty_retry_thinking=None, retry_timeout=None, retry_max_output_tokens=None, answer_format='text'):
        self.base_url, self.model, self.api_key, self.timeout, self.transport = base_url.rstrip('/'), model, api_key, timeout, transport
        self.answer_format=_answer_format(answer_format)
        self.include_usage=include_usage
        if max_output_tokens is not None and (not isinstance(max_output_tokens,int) or isinstance(max_output_tokens,bool) or max_output_tokens<1): raise ValueError('max_output_tokens must be a positive integer')
        self.max_output_tokens=max_output_tokens
        if temperature is not None and (not isinstance(temperature,(int,float)) or isinstance(temperature,bool) or not 0<=temperature<=2): raise ValueError('temperature must be in [0,2]')
        self.temperature=temperature
        if thinking not in (None,'enabled','disabled'): raise ValueError('thinking must be enabled or disabled')
        if reasoning_effort not in (None,'low','high','max'): raise ValueError('invalid reasoning effort')
        self.thinking,self.reasoning_effort=thinking,reasoning_effort
        if empty_retry_thinking not in (None,'disabled'): raise ValueError('empty retry must explicitly disable thinking')
        if empty_retry_thinking is not None:
            if thinking!='enabled': raise ValueError('empty retry requires enabled primary thinking')
            if max_output_tokens is None: raise ValueError('empty retry requires bounded primary output')
            if not isinstance(retry_timeout,(int,float)) or isinstance(retry_timeout,bool) or not 0<retry_timeout<=120: raise ValueError('retry timeout must be in (0,120]')
            if not isinstance(retry_max_output_tokens,int) or isinstance(retry_max_output_tokens,bool) or not 0<retry_max_output_tokens<=max_output_tokens: raise ValueError('retry output must fit primary output reserve')
        self.empty_retry_thinking=empty_retry_thinking
        self.retry_timeout,self.retry_max_output_tokens=retry_timeout,retry_max_output_tokens
    async def stream_events(self, query, evidence):
        if self.answer_format=='text' or (evidence and evidence[0].get('profile')=='legal'):
            async for event in self._raw_stream_events(query,evidence): yield event
            return
        chunks=[]; buffered_bytes=0; rejection=None; attempt=None
        async for event in self._raw_stream_events(query,evidence):
            if event['type']!='delta':
                yield event
                continue
            attempt=event.get('attempt')
            if rejection is not None: continue
            delta=event.get('delta')
            if not isinstance(delta,str):
                rejection='invalid_schema'
                continue
            try:
                encoded=delta.encode('utf-8')
            except UnicodeError:
                rejection='invalid_encoding'
                continue
            buffered_bytes+=len(encoded)
            if buffered_bytes>MAX_BUFFER_BYTES:
                rejection='buffer_limit'
                chunks.clear()
                continue
            chunks.append(delta)
        # Keep consuming after a rejection so terminal usage is preserved. Never
        # retry or reinterpret a nonempty malformed answer as insufficient evidence.
        if rejection is not None: raise GroundingRejected(rejection)
        rendered,metadata=parse_grounded_answer(''.join(chunks),evidence)
        yield {'type':'grounding_validation',**metadata}
        event={'type':'delta','delta':rendered}
        if attempt is not None: event['attempt']=attempt
        yield event

    async def _raw_stream_events(self, query, evidence):
        if self.empty_retry_thinking is None:
            async for event in self._stream_once(query,evidence): yield event
            return
        yield {'type':'generation_attempt','attempt':1,'thinking':self.thinking,'timeout_seconds':self.timeout,'max_output_tokens':self.max_output_tokens}
        seen_content=False
        retry_reason='empty_answer'
        try:
            async for event in self._stream_once(query,evidence):
                if event['type']=='delta': seen_content=True
                yield {**event,'attempt':1}
            if seen_content:return
        except (TimeoutError,httpx.TimeoutException,ModelStreamIncomplete) as exc:
            retry_reason='timeout' if isinstance(exc,(TimeoutError,httpx.TimeoutException)) else exc.reason
            if seen_content or retry_reason not in ('timeout','length','missing_done','missing_stop'):raise
        yield {'type':'generation_retry','reason':retry_reason,'discard_pending_answer':True,'attempt':1,'next_attempt':2}
        fallback=OpenAIStream(self.base_url,self.model,self.api_key,timeout=self.retry_timeout,
                              transport=self.transport,include_usage=self.include_usage,
                              max_output_tokens=self.retry_max_output_tokens,temperature=self.temperature,
                              thinking=self.empty_retry_thinking,answer_format=self.answer_format)
        yield {'type':'generation_attempt','attempt':2,'thinking':fallback.thinking,'timeout_seconds':fallback.timeout,'max_output_tokens':fallback.max_output_tokens}
        async for event in fallback._stream_once(query,evidence): yield {**event,'attempt':2}

    async def _stream_once(self, query, evidence):
        if not evidence: raise ValueError('evidence required')
        messages=messages_for(query,evidence,answer_format=self.answer_format)
        completed=False; finish_reason=None;observed_model=None;incomplete_reason=None
        async with asyncio.timeout(self.timeout):
            async with httpx.AsyncClient(timeout=self.timeout,transport=self.transport,trust_env=False) as client:
                request_body={'model':self.model,'messages':messages,'stream':True}
                if self.include_usage: request_body['stream_options']={'include_usage':True}
                if self.max_output_tokens is not None: request_body['max_tokens']=self.max_output_tokens
                if self.temperature is not None: request_body['temperature']=self.temperature
                if self.thinking is not None: request_body['thinking']={'type':self.thinking}
                if self.reasoning_effort is not None: request_body['reasoning_effort']=self.reasoning_effort
                async with client.stream('POST', self.base_url+'/chat/completions',headers={'Authorization':'Bearer '+self.api_key},json=request_body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'): continue
                        data=line[5:].strip()
                        if data=='[DONE]':
                            completed=True; break
                        payload=json.loads(data); choices=payload.get('choices',[])
                        if isinstance(payload.get('model'),str) and payload['model']!=observed_model:
                            observed_model=payload['model']
                            yield {'type':'model_identity','model':observed_model,'system_fingerprint':payload.get('system_fingerprint')}
                        if choices:
                            choice=choices[0]; reason=choice.get('finish_reason')
                            if reason is not None:
                                if reason!='stop': incomplete_reason=reason
                                else: finish_reason=reason
                            changes=choice.get('delta',{})
                            if changes.get('tool_calls') or changes.get('function_call'): raise RuntimeError('unexpected model tool call')
                            delta=changes.get('content')
                            if delta: yield {'type':'delta','delta':delta}
                        usage=payload.get('usage')
                        if isinstance(usage,dict):
                            safe={k:v for k,v in usage.items() if k in ('prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens') and isinstance(v,int) and not isinstance(v,bool) and v>=0}
                            details=usage.get('completion_tokens_details')
                            if isinstance(details,dict):
                                reasoning=details.get('reasoning_tokens')
                                if isinstance(reasoning,int) and not isinstance(reasoning,bool) and reasoning>=0: safe['reasoning_tokens']=reasoning
                            if safe: yield {'type':'usage','usage':safe}
        if incomplete_reason is not None: raise ModelStreamIncomplete(incomplete_reason)
        if not completed or finish_reason!='stop': raise ModelStreamIncomplete('missing_done' if not completed else 'missing_stop')

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
