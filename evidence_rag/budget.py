"""Evidence budgeting using the configured generation tokenizer, never character counts."""
import json
class HuggingFaceTokenizer:
    def __init__(self, model, revision):
        from transformers import AutoTokenizer
        self.model,self.revision=model,revision
        self.tokenizer=AutoTokenizer.from_pretrained(model,revision=revision,local_files_only=True)
    def count(self, text): return len(self.tokenizer.encode(text,add_special_tokens=False))
    def count_chat(self, messages): return len(self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))

class ConservativeByteBudget:
    """Opt-in byte envelope; conservative heuristic, not a tokenizer guarantee."""
    def __init__(self, context_window, framing_reserve, output_reserve, paragraph_loader=None):
        if any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in (context_window,framing_reserve,output_reserve)):
            raise ValueError('budget values must be nonnegative integers')
        if context_window<=0 or framing_reserve<=0 or output_reserve<=0 or framing_reserve+output_reserve>=context_window:
            raise ValueError('context window and explicit reserves are required')
        self.context_window=context_window;self.framing_reserve=framing_reserve;self.output_reserve=output_reserve;self.paragraph_loader=paragraph_loader

    def measure_content(self,messages):
        return sum(len(str(m['content']).encode('utf-8')) for m in messages)

def assemble(evidence, tokenizer, max_tokens, framing=''):
    if tokenizer is None: return [], {'state':'tokenizer_unavailable','used_tokens':None,'omitted':[e['unit_id'] for e in evidence]}
    from .answer import messages_for
    if isinstance(tokenizer,ConservativeByteBudget):
        selected=[];omitted=[];units=[]
        for original in evidence:
            source=tokenizer.paragraph_loader(original['unit_id']) if tokenizer.paragraph_loader and original.get('profile')=='general' and 'part_no' not in original else None
            parts=source.get('parts',[]) if source else []
            if len(parts)>1:
                for part in parts:
                    units.append({**original,'text':part['text'],'content_sha256':part['sha256'],'source_locator':part['locator'],'document_sha256':original['content_sha256'],'part_no':part['part_no'],'has_more':True})
            else: units.append(dict(original))
        def size(items): return tokenizer.measure_content(messages_for(framing,items))
        capacity=min(max_tokens,tokenizer.context_window-tokenizer.framing_reserve-tokenizer.output_reserve)
        used=size([])
        for unit in units:
            candidate={**unit,'citation_id':f'E{len(selected)+1}'}
            proposed=size(selected+[candidate])
            if proposed>capacity or len(selected)>=60: omitted.append(unit['unit_id']+(('#part-'+str(unit['part_no'])) if 'part_no' in unit else ''))
            else:selected.append(candidate);used=proposed
        return selected,{'state':'conservative_bytes','mode':'conservative_bytes','used_tokens':None,'content_bytes':used,'max_content_bytes':capacity,'framing_reserve':tokenizer.framing_reserve,'output_reserve':tokenizer.output_reserve,'model_context_window':tokenizer.context_window,'omitted':omitted,'fits':used<=capacity,'exact_tokens':False,'guarantee':'heuristic; unknown tokenizer and provider chat template are not formally bounded'}
    def measure(items):
        messages=messages_for(framing,items)
        if hasattr(tokenizer,'count_chat'): return tokenizer.count_chat(messages)
        return tokenizer.count(json.dumps(messages,ensure_ascii=False))
    used=measure([]); selected=[]; omitted=[]
    for e in evidence:
        size=measure(selected+[e])
        if size > max_tokens: omitted.append(e['unit_id']); continue
        used=size; selected.append(e)
    return selected, {'state':'measured' if hasattr(tokenizer,'count_chat') else 'approximate_test_only','used_tokens':used,'max_tokens':max_tokens,'omitted':omitted}
