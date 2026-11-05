"""Offline comparison of a few synthetic strings with one known tokenizer.
This does not validate any other model or provider chat template.
"""
import hashlib,json,pathlib,time
from transformers import AutoTokenizer
from evidence_rag.answer import messages_for
from evidence_rag.budget import ConservativeByteBudget,assemble
REV='5617a9f61b028005a4858fdac845db406aefb181'
SAMPLES=['English: the ocean carries heat across the planet.','中文：证据应保留完整段落和来源。','Mixed English 中文, 2026-11-06, αβγ.','Code: x = {"key": [1, 2, 3]}\nreturn x','Emoji: 🌊🔬✅ and punctuation?!']
tok=AutoTokenizer.from_pretrained('BAAI/bge-m3',revision=REV,local_files_only=True)
rows=[{'text':s,'utf8_bytes':len(s.encode('utf-8')),'bge_tokens_without_special':len(tok.encode(s,add_special_tokens=False))} for s in SAMPLES]
evidence=[{'citation_id':'E1','unit_id':'synthetic-1','document_id':'d1','title':'Fixture','text':'The ocean transports heat.','profile':'general'}]
budget=ConservativeByteBudget(16000,1024,2048);selected,info=assemble(evidence,budget,6000,'What does the ocean do?')
messages=messages_for('What does the ocean do?',selected)
content_token_count=sum(len(tok.encode(m['content'],add_special_tokens=False)) for m in messages)
result={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'tokenizer':'BAAI/bge-m3','revision':REV,'rows':rows,'all_sample_content_tokens_lte_bytes':all(x['bge_tokens_without_special']<=x['utf8_bytes'] for x in rows),'message_content_tokens_without_special':content_token_count,'budget':info,'message_content_tokens_lte_bytes':content_token_count<=info['content_bytes'],'limitation':'These fixtures test one local tokenizer without chat-template tokens; no general bound for any provider or model.'}
path=pathlib.Path('docs/development/general-evidence-rag-20261106/evidence/byte-budget-tokenizer-fixtures.json');path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'sample_count':len(rows),'all_match':result['all_sample_content_tokens_lte_bytes'],'message_match':result['message_content_tokens_lte_bytes'],'bytes':info['content_bytes'],'tokens':content_token_count}))
